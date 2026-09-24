"""Disguise against a Mold Breaker, in the port against Showdown (IKA-208).

Disguise has `flags: { breakable: 1 }` (data/abilities.ts), so a Mold Breaker's move is
not stopped by it: the hit lands whole and the forme stays. IKA-207's diff_turn found it the
day the port started answering Disguise (seed 2, battle 205, turn 6: a Mega Gyarados' Crunch
into a Mimikyu). Python busts it all the same and is not asserted here; the port's damage
layer zeroed the hit for any attacker, and `hit_target` guarded it for any attacker.

The positive control is the exe before this fix (it busts); the control is the same hit
without Mold Breaker, which busts in both.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID
from .test_self_destruct_oracle import bridged  # noqa: F401

pytestmark = pytest.mark.oracle

FILL = ["protect", "helpinghand", "irondefense", "substitute"]


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    spread = {"hp": 32, "atk": 20, "def": 20, "spa": 20, "spd": 20, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=spread)


def _teams(ability: str) -> tuple[list[TeamSet], list[TeamSet]]:
    return (
        [_mon("Excadrill", ability, ["ironhead", "protect", "helpinghand", "earthquake"], 20),
         _mon("Alakazam", "Inner Focus", FILL, 5)],
        [_mon("Mimikyu", "Disguise", FILL, 0), _mon("Kangaskhan", "Scrappy", FILL, 1)],
    )


CASES = {
    "mold-breaker-hits-through-disguise": ("Mold Breaker", "|-damage|p2a: Mimikyu|"),
    "control-sand-rush-is-disguised": ("Sand Rush", "|-activate|p2a: Mimikyu|ability: Disguise"),
}
CHOICES = ["move 1 1, move 2 -1", "move 2 -2, move 2 -1"]
BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {f"p{i + 1}.{m.species}": m.hp for i, side in enumerate(pos.sides) for m in side.pokemon}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001, F811
    ability, shown = CASES[name]
    handle = oracle.create(FORMAT_ID, *_teams(ability), policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.step(CHOICES)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    assert any(line.startswith(shown) for line in log), log
    node = rustnode.node_for(reg)
    assert node is not None
    actions = [next(a for a in side_actions(reg, before, s) if a.to_choice() == CHOICES[s]) for s in (0, 1)]
    picked = node.resolve(before, actions, BUDGET, select=0)
    assert picked is not None and picked.position is not None, "the port refused the turn"
    assert _state(picked.position) == theirs
