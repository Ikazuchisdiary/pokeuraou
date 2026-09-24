"""The volatiles a Showdown position carries in `unmodelledVolatiles`, in the port (IKA-208).

The bridge files every volatile not on its `MODELLED_VOLATILES` list under
`unmodelledVolatiles` (packages/sim-bridge/src/position.ts), and the port refused any
position carrying one; Python ignores them. IKA-207's diff_turn met them as the port's most
common refusal: `throatchop`, the charging move's own volatile (`electroshot`, `solarbeam`),
`stockpile`, `flashfire`, `metronome`. `resolve::showdown_volatiles` now:

* drops the charging move's own volatile, which `twoturnmove` already carries
  (test_charge_target's "showdown position" case);
* moves `flashfire` into `volatiles`, where the port's own `absorb` puts it. Flash Fire's
  `onModifyAtk`/`onModifySpA` 1.5x for the holder's Fire moves (data/abilities.ts) is in
  neither engine's damage -- both add the volatile and never read it -- so that HP is a
  strict expected failure here, not this issue's to fix;
* drops the volatile of an item the port ignores (Metronome adds `metronome` in `onStart`);
* keeps anything else and names it (`volatile not modelled: throatchop`).

Each case is played by Showdown and the port resolves the compared turn from Showdown's
position, as-is. The positive control is the exe before this change, which refuses them.
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
from .test_self_destruct_oracle import _state, bridged  # noqa: F401

pytestmark = pytest.mark.oracle

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)
FILL = ["protect", "helpinghand", "irondefense", "substitute"]


def _mon(species: str, ability: str, moves: list[str], spe: int, item: str | None = None) -> TeamSet:
    spread = {"hp": 32, "atk": 20, "def": 20, "spa": 20, "spd": 20, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


FIRE = _mon("Arcanine", "Flash Fire", ["flamethrower", "protect", "helpinghand", "irondefense"], 10)
FIRE_FOES = [
    _mon("Charizard", "Blaze", ["flamethrower", "protect", "helpinghand", "irondefense"], 20),
    _mon("Garchomp", "Rough Skin", FILL, 2),
]
CHOPPER = [
    _mon("Kingambit", "Defiant", ["throatchop", "protect", "helpinghand", "ironhead"], 20),
    _mon("Garchomp", "Rough Skin", FILL, 2),
]
METRONOME = _mon(
    "Kangaskhan", "Scrappy", ["bodyslam", "protect", "helpinghand", "irondefense"], 10, "Metronome"
)
PARTNER = _mon("Alakazam", "Inner Focus", FILL, 5)

#: name -> (teams, setup turns, the compared turn, the volatile Showdown's position carries,
#: the note the port gives, or None).
CASES: dict[str, tuple] = {
    # Charizard's Flamethrower lights Arcanine's Flash Fire; then Arcanine's is 1.5x.
    "flash-fire-boosts-the-next-fire-move": (
        ([FIRE, PARTNER], FIRE_FOES),
        [["move 4, move 2 -1", "move 1 1, move 2 -1"]],
        ["move 1 2, move 2 -1", "move 3 -2, move 2 -1"],
        "flashfire", None,
    ),
    "metronomes-own-volatile-is-dropped": (
        ([METRONOME, PARTNER], [FIRE_FOES[1], FIRE_FOES[0]]),
        [], ["move 1 1, move 2 -1", "move 2 -2, move 3 -1"],
        "metronome", None,
    ),
    "throat-chop-is-named": (
        ([METRONOME, PARTNER], CHOPPER),
        [["move 3 -2, move 2 -1", "move 1 1, move 2 -1"]],
        ["move 1 1, move 2 -1", "move 3 -2, move 2 -1"],
        "throatchop", "volatile not modelled: throatchop",
    ),
}


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict]:
    (mine, theirs), setup, choices, volatile, _note = CASES[name]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = _state(Position.from_json(handle.position))
    handle.close()
    carried = {v for side in before.sides for m in side.pokemon for v in m.unmodelled_volatiles}
    assert volatile in carried, f"{name}: Showdown's position carries {carried}"
    return before, choices, after


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [next(a for a in side_actions(reg, pos, s) if a.to_choice() == choices[s]) for s in (0, 1)]


def _resolve(reg, pos: Position, choices: list[str]):  # noqa: ANN001, ANN202
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, pos, choices)
    there = node.resolve(pos, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    assert len(there.branches) == 1, there.branches
    picked = node.resolve(pos, actions, BUDGET, select=0)
    assert picked is not None and picked.position is not None
    return picked


def _unboosted(name: str):  # noqa: ANN202
    if name != "flash-fire-boosts-the-next-fire-move":
        return name
    return pytest.param(
        name,
        marks=pytest.mark.xfail(strict=True, reason="Flash Fire's 1.5x is in neither engine's damage"),
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_answers_and_names(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001, F811
    """Answered, with the note for what it does not model and none for the rest; Flash Fire
    lands in `volatiles`, where the port's own Flash Fire puts it."""
    before, choices, _theirs = _play(oracle, name)
    picked = _resolve(reg, before, choices)
    note = CASES[name][4]
    named = {n for n in picked.unmodelled if n.startswith("volatile not modelled")}
    assert named == ({note} if note else set()), picked.unmodelled
    volatile = CASES[name][3]
    for side in picked.position.sides:
        for mon in side.pokemon:
            if volatile == "flashfire" and mon.species == "arcanine":
                assert mon.has_volatile("flashfire") and "flashfire" not in mon.unmodelled_volatiles


@pytest.mark.parametrize("name", [_unboosted(n) for n in sorted(CASES)])
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001, F811
    before, choices, theirs = _play(oracle, name)
    picked = _resolve(reg, before, choices)
    assert _state(picked.position) == theirs
