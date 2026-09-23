"""Feint breaks protection only on a target it reaches (IKA-153).

Showdown runs a move's hit steps in this order (vendor/pokemon-showdown/sim/battle-actions.ts
:553-578, gen 9, no swaps):

    0. hitStepInvulnerabilityEvent   3. hitStepTryImmunity
    1. hitStepTryHitEvent            4. hitStepAccuracy
    2. hitStepTypeImmunity           5. hitStepBreakProtect

and after every step keeps only the targets that step let through (``:605``,
``targets = targets.filter((val, i) => hitResults[i] || hitResults[i] === 0)``). So a Feint
into a Ghost (Normal, step 2) or a Feint that misses (step 4) never reaches step 5: the
Protect stays up and the partner's move into the same Pokemon is still blocked. The
resolver broke protection for every target before any of this (IKA-61 section 7), and the
port copied it.

Every case is played by Showdown first and the Python turn is held to its HP; one control
case (a Feint that lands) shows the break does fire, so an answer where nothing is ever
broken would fail it.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.resolve import Budget, resolve_turn

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None
) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
        item=item,
    )


# Lucario Feints, Garchomp follows into the same target. Distinct Speeds, so no tie.
TEAM_A = [
    _mon("Lucario", "Inner Focus", ["feint", "protect", "aurasphere", "closecombat"], 20),
    _mon("Garchomp", "Rough Skin", ["dragonclaw", "protect", "earthquake", "rockslide"], 10),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "protect"], 4),
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"], 5),
]


def _team_b(kingambit_item: str | None = None) -> list[TeamSet]:
    return [
        # Ghost/Grass: Feint (Normal) cannot touch it.
        _mon("Sinistcha", "Hospitality", ["protect", "wideguard", "strengthsap", "matchagotcha"], 2),
        _mon(
            "Kingambit", "Defiant", ["protect", "kowtowcleave", "suckerpunch", "swordsdance"], 3,
            item=kingambit_item,
        ),
        _mon("Sylveon", "Pixilate", ["protect", "calmmind", "hypervoice", "wish"], 8),
        _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], 24),
    ]


#: name -> (Kingambit's item, choices, Showdown's accuracy policy, the Showdown log line
#: that says what happened to the partner's move).
CASES = {
    # Feint into the Protecting Ghost; Dragon Claw into it after.
    "ghost-protect": (
        None, ["move 1 1, move 1 1", "move 1, move 4"], "hit",
        "|-activate|p2a: Sinistcha|move: Protect",
    ),
    # Feint into the Ghost whose side has Wide Guard up; Rock Slide after.
    "ghost-wide-guard": (
        None, ["move 1 1, move 4", "move 2, move 4"], "hit",
        "|-activate|p2a: Sinistcha|move: Wide Guard",
    ),
    # Feint misses a Bright Powder Kingambit behind Protect; Dragon Claw after.
    "missed": (
        "Bright Powder", ["move 1 2, move 1 2", "move 1, move 1"], "miss",
        "|-activate|p2b: Kingambit|move: Protect",
    ),
    # Control: the same Feint lands and breaks the Protect, and Dragon Claw hits.
    "control-lands": (
        None, ["move 1 2, move 1 2", "move 1, move 1"], "hit",
        "|-activate|p2b: Kingambit|move: Feint",
    ),
}

#: No crit, no secondary, the maximum roll -- what the oracle's policy pins. Accuracy is
#: enumerated so the missed Feint has a branch of its own.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _hp(pos: Position) -> dict[str, int]:
    return {
        f"p{index + 1} {mon.species}": mon.hp
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list, dict[str, int], list[str]]:
    item, choices, accuracy, _line = CASES[name]
    handle = oracle.create(
        FORMAT_ID, TEAM_A, _team_b(item), policy=RandomnessPolicy(accuracy=accuracy)
    )
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _hp(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    return before, choices, theirs, log


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side])
        for side in (0, 1)
    ]


def _feint_missed(events: list[str]) -> bool:
    return any(e.endswith("Feint missed") for e in events)


@pytest.mark.parametrize("name", sorted(CASES))
def test_feint_breaks_only_what_it_reaches(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    before, choices, theirs, log = _play(oracle, name)
    assert CASES[name][3] in log, f"Showdown did not do what the case says: {log}"

    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    missed = name == "missed"
    # Showdown's policy pins one accuracy outcome; the Python branches that match it are
    # the ones where Feint did the same. The partner's own accuracy roll is pinned to
    # "hit" in Showdown too, so only a branch where the partner hit is comparable -- except
    # when the partner was blocked, where no roll was made and every branch must agree.
    ours = [
        b for b in result.branches
        if _feint_missed(b.events) == missed
        and not any(e.endswith(("Dragon Claw missed", "Rock Slide missed")) for e in b.events)
    ]
    assert ours, "no Python branch plays the same accuracy outcomes"
    for branch in ours:
        assert _hp(branch.position) == theirs, (
            f"{name}: showdown {theirs} != python {_hp(branch.position)}; "
            + " / ".join(branch.events)
        )


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_breaks_only_what_it_reaches(
    reg, oracle: Oracle, bridged: None, name: str  # noqa: ANN001
) -> None:
    """The port resolves the same turns branch for branch, positions included."""
    before, choices, _theirs, _log = _play(oracle, name)
    # Showdown's positions carry its final stats, which the port refuses as it would a
    # transformed Pokemon's. Nobody here is transformed, so they follow from the spreads.
    for side in before.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    actions = _actions(reg, before, choices)
    budget = replace(Budget.matrix(), enumerate_accuracy=True)
    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    here = resolve_turn(reg, before, actions, budget=budget)
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, budget)
    assert there is not None, "the port refused the turn"
    mine = [b.probability for b in here.branches]
    assert len(mine) == len(there.branches)
    assert all(abs(x - y) < 1e-12 for x, y in zip(mine, there.branches, strict=True))
    for index, branch in enumerate(here.branches):
        chosen = node.resolve(before, actions, budget, select=index)
        assert chosen is not None and chosen.position is not None
        assert chosen.position.to_json() == branch.position.to_json(), (name, index)
