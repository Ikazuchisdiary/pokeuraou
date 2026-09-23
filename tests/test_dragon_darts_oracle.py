"""Dragon Darts, and Cursed Body against a multi-hit move, in the port against Showdown (IKA-208).

The port refused Dragon Darts ("move field smartTarget") and any chance secondary on a
multi-hit move ("secondary on a multi-hit move") -- Cursed Body's Disable is one, pushed on
every hit. Python fills Dragon Darts as an ordinary two-hit move at one foe, which is not
Showdown's, and is not asserted here.

`smartTarget` (sim/pokemon.ts `getSmartTargets`): the target and its adjacent ally, if that
ally stands and is not the user. The hit steps of `trySpreadMoveHit` run on both, and any
failure -- Protect, the type immunity, a miss -- sets `move.smartTarget = false`, leaving
the two hits to whoever is left; with both still there, `hitStepMoveHitLoop` sends hit 1 to
the first and hit 2 to the second, each a single-target hit.

Cursed Body (data/abilities.ts): `onDamagingHit` returns at once `if (source.volatiles
['disable'])`, and otherwise disables on `randomChance(3, 10)`: over two hits that is
1 - 0.7^2 = 0.51, whichever hit it was.

The positive control is the exe before this change, which refuses these turns.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.resolve import Budget

from .conftest import FORMAT_ID
from .test_self_destruct_oracle import bridged  # noqa: F401

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    spread = {"hp": 32, "atk": 20, "def": 20, "spa": 20, "spd": 20, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=spread)


#: Dragapult: 1 Dragon Darts, 2 Protect, 3 Helping Hand, 4 Dual Wingbeat.
USER = _mon("Dragapult", "Clear Body", ["dragondarts", "protect", "helpinghand", "dualwingbeat"], 20)
FILL = ["protect", "helpinghand", "irondefense", "substitute"]
PARTNER = _mon("Alakazam", "Inner Focus", FILL, 5)


def _foes(first: str = "Garchomp", second: str = "Kangaskhan") -> list[TeamSet]:
    abilities = {
        "Garchomp": "Rough Skin", "Kangaskhan": "Scrappy", "Sylveon": "Pixilate", "Gengar": "Cursed Body",
    }
    return [_mon(first, abilities[first], FILL, 0), _mon(second, abilities[second], FILL, 1)]


MINE = "move 2 -1"
QUIET = "move 2 -2, move 2 -1"
CHANCE = RandomnessPolicy(secondary=True)

#: name -> (their team, the compared turn, the policy, a Showdown log line that shows it).
CASES: dict[str, tuple] = {
    "darts-split-between-two-foes": (
        _foes(), [f"move 1 1, {MINE}", QUIET], None, "|-anim|p1a: Dragapult|Dragon Darts|p2b: Kangaskhan",
    ),
    "darts-both-into-one-when-the-other-protects": (
        # Protect's `onTryHit` stays silent for a smart target: it only turns it off.
        _foes(), [f"move 1 1, {MINE}", "move 1, move 2 -1"], None, "|move|p2a: Garchomp|Protect",
    ),
    "darts-both-into-one-when-the-other-is-a-fairy": (
        _foes(second="Sylveon"), [f"move 1 1, {MINE}", QUIET], None, "|-anim|p1a: Dragapult|Dragon Darts|p2a",
    ),
    "darts-at-a-fairy-go-to-its-partner": (
        _foes(first="Sylveon"), [f"move 1 1, {MINE}", QUIET], None, "|-anim|p1a: Dragapult|Dragon Darts|p2b",
    ),
    "cursed-body-on-a-two-hit-move": (
        _foes(first="Gengar"), [f"move 4 1, {MINE}", QUIET], CHANCE, "|-start|p1a: Dragapult|Disable",
    ),
}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)
#: Chance branched, for Cursed Body.
DRAWN = replace(BUDGET, enumerate_secondary=True)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (mon.hp, mon.has_volatile("disable"))
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict]:
    theirs, choices, policy, shown = CASES[name]
    handle = oracle.create(FORMAT_ID, [USER, PARTNER], theirs, policy=policy or RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    assert any(line.startswith(shown) for line in log), f"Showdown did not do what {name} says: {log}"
    return before, choices, after


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [next(a for a in side_actions(reg, pos, s) if a.to_choice() == choices[s]) for s in (0, 1)]


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001, F811
    before, choices, theirs = _play(oracle, name)
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, before, choices)
    budget = DRAWN if CASES[name][2] is CHANCE else BUDGET
    there = node.resolve(before, actions, budget)
    assert there is not None, "the port refused the turn"
    states = []
    for index in range(len(there.branches)):
        picked = node.resolve(before, actions, budget, select=index)
        assert picked is not None and picked.position is not None
        states.append(_state(picked.position))
    if CASES[name][2] is CHANCE:
        # Showdown's pin answers the chance with yes: the Disable branch, of 1 - 0.7^2.
        disabled = [w for w, s in zip(there.branches, states, strict=True) if s["p1.dragapult"][1]]
        assert sum(disabled) == pytest.approx(0.51), there.branches
        assert theirs in states
        return
    assert len(states) == 1, there.branches
    assert states[0] == theirs
