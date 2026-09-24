"""Psychic Terrain stops a priority move per target, not per move (IKA-156).

Showdown's terrain is an ``onTryHit`` (vendor/pokemon-showdown/data/moves.ts:14116):

    if (effect && (effect.priority <= 0.1 || effect.target === 'self')) return;
    if (target.isSemiInvulnerable() || target.isAlly(source)) return;
    if (!target.isGrounded()) { ...; return; }
    this.add('-activate', target, 'move: Psychic Terrain');
    return null;

It runs in ``hitStepTryHitEvent`` (step 1 of ``trySpreadMoveHit``), once for each target,
after the move has started -- PP is spent, the user's move counter is bumped -- and only
the targets it returns ``null`` for drop out. The resolver and the port stopped the whole
move before it started whenever *any* live foe was grounded: a Fake Out into a Corviknight
beside a Garchomp was cancelled, a Prankster Cotton Spore into the same pair touched
neither, and a priority move into the user's own ally was stopped by the foes' footing.

Every case is played by Showdown first and the Python turn is held to its HP, boosts and
PP. The controls are a Fake Out into the grounded Pokemon (stopped in both) and one into a
Flying type beside another floater (never stopped), so an answer that always or never
blocks fails one of them. The old code agreed with the first on HP and still failed it on
PP: stopping the move before it started spent none, where Showdown spends one. The Mold
Breaker case is the positive control for the per-target footing -- a check that ignored
the attacker would let that Bullet Punch into the Levitate Rotom land.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, SideAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int, item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
        item=item,
    )


# Indeedee's Psychic Surge lays the terrain as the leads come in. Distinct Speeds.
TEAM_A = [
    _mon("Indeedee", "Psychic Surge", ["protect", "calmmind", "psychic", "followme"], 2),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "protect"], 4),
    _mon("Whimsicott", "Prankster", ["cottonspore", "protect", "moonblast", "tailwind"], 30),
    _mon("Pangoro", "Mold Breaker", ["bulletpunch", "protect", "knockoff", "closecombat"], 6),
    _mon("Lucario", "Inner Focus", ["bulletpunch", "protect", "aurasphere", "closecombat"], 7),
]

TEAM_B = [
    _mon("Corviknight", "Unnerve", ["bulkup", "protect", "bravebird", "roost"], 8),
    _mon("Garchomp", "Rough Skin", ["swordsdance", "protect", "earthquake", "dragonclaw"], 10),
    _mon("Rotom-Wash", "Levitate", ["nastyplot", "protect", "hydropump", "thunderbolt"], 12),
    _mon("Dragonite", "Inner Focus", ["dragondance", "protect", "extremespeed", "roost"], 14),
]

#: name -> (team orders, choices, the Showdown log line that says what happened).
CASES = {
    # Fake Out into the Flying Corviknight while the Garchomp beside it is grounded.
    "flying-beside-grounded": (
        ["team 1234", "team 1234"],
        ["move 1, move 1 1", "move 1, move 1"],
        "|move|p1b: Incineroar|Fake Out|p2a: Corviknight",
    ),
    # Control: the same Fake Out into the grounded Garchomp is stopped.
    "control-grounded": (
        ["team 1234", "team 1234"],
        ["move 1, move 1 2", "move 1, move 1"],
        "|-activate|p2b: Garchomp|move: Psychic Terrain",
    ),
    # Control: into a Flying type beside a floater -- no foe is grounded, never stopped.
    "control-no-foe-grounded": (
        ["team 1234", "team 1324"],
        ["move 1, move 1 1", "move 1, move 1"],
        "|move|p1b: Incineroar|Fake Out|p2a: Corviknight",
    ),
    # Bullet Punch into a Levitate Rotom beside the Garchomp.
    "levitate-beside-grounded": (
        ["team 1523", "team 3214"],
        ["move 1, move 1 1", "move 1, move 1"],
        "|move|p1b: Lucario|Bullet Punch|p2a: Rotom",
    ),
    # Mold Breaker ignores Levitate in `isGrounded` (`suppressingAbility`), so the same
    # Bullet Punch from Pangoro is stopped.
    "mold-breaker-levitate": (
        ["team 1423", "team 3214"],
        ["move 1, move 1 1", "move 1, move 1"],
        "|-activate|p2a: Rotom|move: Psychic Terrain",
    ),
    # Prankster Cotton Spore into both: Corviknight's Speed falls, Garchomp is spared.
    "spread-one-grounded": (
        ["team 1324", "team 1234"],
        ["move 1, move 1", "move 1, move 1"],
        "|-activate|p2b: Garchomp|move: Psychic Terrain",
    ),
    # Fake Out into the user's own ally: `target.isAlly(source)` exempts it.
    "into-own-ally": (
        ["team 1234", "team 1234"],
        ["move 2, move 1 -1", "move 1, move 1"],
        "|move|p1b: Incineroar|Fake Out|p1a: Indeedee",
    ),
}

BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1} {mon.species}": (
            mon.hp,
            tuple(sorted((k, v) for k, v in mon.boosts.items() if v)),
            tuple((m.id, m.pp) for m in mon.moves),
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict[str, tuple], list[str]]:
    orders, choices, _line = CASES[name]
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(orders)
    before = Position.from_json(handle.position)
    assert before.field.terrain == "psychicterrain"
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    return before, choices, theirs, log


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        found = [a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]]
        if not found and choices[side] == ALLY_FAKE_OUT.to_choice():
            # The action generator never aims a `normal` move at an ally, so the search
            # cannot choose this; the resolver still has to play it as Showdown does.
            found = [ALLY_FAKE_OUT]
        out.append(found[0])
    return out


#: Indeedee Calm Minds, Incineroar Fake Outs its own partner.
ALLY_FAKE_OUT = SideAction(slots=(MoveAction(0, 2, "calmmind", None), MoveAction(1, 1, "fakeout", -1)))


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_stops_each_grounded_target(
    reg,
    oracle: Oracle,
    bridged: None,
    name: str,  # noqa: ANN001
) -> None:
    """The port plays Showdown's turn, and every branch it has is Showdown's position."""
    before, choices, theirs, log = _play(oracle, name)
    assert CASES[name][2] in log, f"Showdown did not do what the case says: {log}"
    # Showdown's positions carry its final stats, which the port refuses as it would a
    # transformed Pokemon's. Nobody here is transformed, so they follow from the spreads.
    for side in before.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    actions = _actions(reg, before, choices)
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    assert there.branches
    for index in range(len(there.branches)):
        chosen = node.resolve(before, actions, BUDGET, select=index)
        assert chosen is not None and chosen.position is not None
        assert _state(chosen.position) == theirs, (name, index)
