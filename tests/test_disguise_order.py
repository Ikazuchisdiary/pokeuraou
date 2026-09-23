"""Disguise takes only a hit that lands (IKA-155).

Showdown's Disguise is an `onDamage` handler (vendor/pokemon-showdown/data/abilities.ts,
`disguise`), so it acts inside step 7 of `trySpreadMoveHit` (sim/battle-actions.ts:553-578),
the hit loop. Steps 2 (type immunity) and 4 (accuracy) run before it, and after every step
only the targets it let through go on (`:605`). So a Normal or Dragon move into Mimikyu
(Ghost/Fairy) is immune and a move that misses it misses: neither busts the disguise nor
costs the eighth. The resolver busted it before looking at either.

Feint is the same shape twice over: a Feint into a Protecting Mimikyu is immune at step 2,
so it neither busts the disguise nor breaks the Protect (step 5), and the partner's Rock
Slide is still blocked. IKA-153 moved the break behind the immunity and the accuracy for
every other target but left it on the disguise path, which saw neither.

Every case is played by Showdown first and the Python turn is held to its HP and formes.
The control cases (a Rock Slide and a Knock Off that land) show the bust still fires, so an
answer where Disguise never breaks would fail them. Ice Face is not tested: Eiscue's intact
forme is not in Reg M-B or M-C (only `eiscuenoice` is), so nothing can hold it there.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.resolve import Budget, resolve_turn

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
    )


TEAM_A = [
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "protect"], 12),
    _mon("Garchomp", "Rough Skin", ["dragonclaw", "protect", "earthquake", "rockslide"], 10),
    _mon("Lucario", "Inner Focus", ["feint", "protect", "aurasphere", "closecombat"], 20),
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "toxic"], 5),
]
# Mimikyu leads beside a Protecting Kingambit, so a Rock Slide reaches Mimikyu alone.
TEAM_B = [
    _mon("Mimikyu", "Disguise", ["swordsdance", "protect", "playrough", "shadowsneak"], 2),
    _mon("Kingambit", "Defiant", ["protect", "kowtowcleave", "suckerpunch", "swordsdance"], 3),
    _mon("Sylveon", "Pixilate", ["protect", "calmmind", "hypervoice", "wish"], 8),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], 24),
]

#: name -> (side 0's team order, choices, Showdown's accuracy policy, the Showdown log line
#: that says what happened to the move into Mimikyu).
CASES = {
    # Fake Out (Normal) into the Ghost.
    "immune-normal": (
        "team 1234", ["move 1 1, move 2", "move 1, move 1"], "hit", "|-immune|p2a: Mimikyu",
    ),
    # Dragon Claw into the Fairy.
    "immune-dragon": (
        "team 1234", ["move 4, move 1 1", "move 1, move 1"], "hit", "|-immune|p2a: Mimikyu",
    ),
    # Rock Slide misses.
    "missed": (
        "team 1234", ["move 4, move 4", "move 1, move 1"], "miss",
        "|-miss|p1b: Garchomp|p2a: Mimikyu",
    ),
    # Feint into the Protecting Mimikyu, Rock Slide after: Protect holds for both.
    "feint-immune": (
        "team 3214", ["move 1 1, move 4", "move 2, move 1"], "hit",
        "|-activate|p2a: Mimikyu|move: Protect",
    ),
    # Controls: a Rock Slide and a Knock Off that land do bust it.
    "control-rock-slide": (
        "team 1234", ["move 4, move 4", "move 1, move 1"], "hit",
        "|-activate|p2a: Mimikyu|ability: Disguise",
    ),
    "control-knock-off": (
        "team 1234", ["move 3 1, move 2", "move 1, move 1"], "hit",
        "|-activate|p2a: Mimikyu|ability: Disguise",
    ),
}

#: No crit, no secondary, the maximum roll -- what the oracle's policy pins. Accuracy is
#: enumerated so the missed Rock Slide has a branch of its own.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple[str, int]]:
    return {
        f"p{index + 1}.{slot}": (mon.species, mon.hp)
        for index, side in enumerate(pos.sides)
        for slot, mon in enumerate(side.pokemon)
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict, list[str]]:
    order, choices, accuracy, _line = CASES[name]
    handle = oracle.create(
        FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy(accuracy=accuracy)
    )
    handle.step([order, "team 1234"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    return before, choices, theirs, log


@pytest.mark.parametrize("name", sorted(CASES))
def test_disguise_takes_only_a_hit_that_lands(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    before, choices, theirs, log = _play(oracle, name)
    assert CASES[name][3] in log, f"Showdown did not do what the case says: {log}"

    actions = [
        next(a for a in side_actions(reg, before, side) if a.to_choice() == choices[side])
        for side in (0, 1)
    ]
    result = resolve_turn(reg, before, actions, budget=BUDGET)
    # Showdown's policy pins one accuracy outcome; the Python branches that match it are
    # the ones where every move did the same.
    missed = name == "missed"
    ours = [
        b for b in result.branches
        if any(e.endswith("missed") for e in b.events) == missed
    ]
    assert ours, "no Python branch plays the same accuracy outcomes"
    for branch in ours:
        assert _state(branch.position) == theirs, (
            f"{name}: showdown {theirs} != python {_state(branch.position)}; "
            + " / ".join(branch.events)
        )
