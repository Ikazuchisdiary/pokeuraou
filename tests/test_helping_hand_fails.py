"""Helping Hand fails on a partner that has already moved, played against Showdown (IKA-184).

`data/moves.ts`, helpinghand (the champions mod does not override it):

    priority: 5,
    volatileStatus: 'helpinghand',
    onTryHit(target) {
        if (!target.newlySwitched && !this.queue.willMove(target)) return false;
    },
    condition: { duration: 1, onStart ... multiplier = 1.5, onRestart ... multiplier *= 1.5, ... }

So the move fails -- `-fail`, and a `false` that Stomping Tantrum counts -- when the partner
has already used its move this turn and did not come in this turn. At +5 that happens when
the two partners help each other (the slower one's fails) or when a Prankster partner's
Protect (+4 +1) goes first in the same bracket. A partner that switched in this turn is
`newlySwitched` and has no move queued, and the move goes through.

The resolver and the port put the volatile on whoever the target was and never failed the
move (IKA-172 found it reading the code; its Perish Song test left the flag out for this).

Each case plays one turn in Showdown; the Python turn and the port's are held to
Showdown's HP, faint and failed-move flag for every Pokemon after every turn.
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

#: One move list for every set, so one choice string means the same thing on every
#: Pokemon. Not legal sets: the bridge does not check learnsets.
MOVES = ["helpinghand", "protect", "hypervoice", "memento"]


def _mon(species: str, ability: str, spe: int) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=list(MOVES),
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
    )


POLITOED = _mon("Politoed", "Damp", 10)
WHIMSICOTT = _mon("Whimsicott", "Chlorophyll", 21)
PRANKSTER = _mon("Whimsicott", "Prankster", 21)
SLOW_PARTNER = _mon("Hippowdon", "Sand Force", 1)
ROTOM = _mon("Rotom-Wash", "Levitate", 4)

INCINEROAR = _mon("Incineroar", "Blaze", 2)
GARCHOMP = _mon("Garchomp", "Rough Skin", 5)
HIPPOWDON = _mon("Hippowdon", "Sand Force", 3)
TEAM_B = [INCINEROAR, GARCHOMP, HIPPOWDON, ROTOM]

MUTUAL = "move 1 -2, move 1 -1"
PROTECT_BOTH = "move 2, move 2"

#: name -> (side 0's team, the turns' choices per side). Side 1 is TEAM_B, slower; its
#: MUTUAL fails Incineroar's help, and its PROTECT_BOTH fails Incineroar's Protect when
#: nobody is left to move after it.
CASES: dict[str, tuple[list[TeamSet], list[tuple[str, str]]]] = {
    # Both partners help each other: Whimsicott is faster, its Helping Hand lands, and
    # Politoed's finds a partner who has already moved. Side 1 does the same, slower.
    "mutual": ([POLITOED, WHIMSICOTT], [(MUTUAL, MUTUAL)]),
    # The same with a slower partner: now the partner's is the one that fails.
    "mutual-swapped": ([POLITOED, SLOW_PARTNER], [(MUTUAL, PROTECT_BOTH)]),
    # A Prankster partner's Protect is +5 as well and goes first: the help fails.
    "prankster-protect": ([POLITOED, PRANKSTER], [("move 1 -2, move 2", PROTECT_BOTH)]),
    # Control: the partner attacks at +0, after the help, and Hyper Voice is boosted.
    "partner-attacks": ([POLITOED, WHIMSICOTT], [("move 1 -2, move 3", MUTUAL)]),
    # The same Hyper Voice with nobody helping: what the boost is measured against.
    "unhelped-attack": ([POLITOED, WHIMSICOTT], [("move 2, move 3", MUTUAL)]),
    # Control: the faster one helps the slower one, who has not moved yet.
    "fast-helper": ([POLITOED, WHIMSICOTT], [("move 3, move 1 -1", PROTECT_BOTH)]),
    # Control: the partner switches out; the one coming in is `newlySwitched`, so the help
    # goes through although nobody on that slot will move.
    "partner-switched": ([POLITOED, WHIMSICOTT, ROTOM], [("move 1 -2, switch 3", PROTECT_BOTH)]),
}

BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    """Per Pokemon: HP, fainted, and the failed-move flag while it stands."""
    out = {}
    for index, side in enumerate(pos.sides):
        for mon in side.pokemon:
            key = f"p{index + 1} {mon.species}"
            out[key] = (mon.hp, mon.fainted, None if mon.fainted else mon.move_last_turn_failed)
    return out


def _play(oracle: Oracle, name: str) -> list[Position]:
    team_a, turns = CASES[name]
    handle = oracle.create(FORMAT_ID, team_a, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team " + "".join(str(i + 1) for i in range(len(team_a))), "team 1234"])
    positions = [Position.from_json(handle.position)]
    for choice_a, choice_b in turns:
        handle.step([choice_a, choice_b])
        assert handle.choice_errors == [], (name, handle.choice_errors)
        positions.append(Position.from_json(handle.position))
    handle.close()
    return positions


def _check_showdown(name: str, positions: list[Position]) -> None:
    """Whose move Showdown failed: the positive control that each case is the one named."""
    flags = {k: v[2] for k, v in _state(positions[-1]).items() if v[2] is not None}
    failed = sorted(k for k, v in flags.items() if v)
    expected = {
        "mutual": ["p1 politoed", "p2 incineroar"],
        "mutual-swapped": ["p1 hippowdon", "p2 incineroar"],
        "prankster-protect": ["p1 politoed", "p2 incineroar"],
        "partner-attacks": ["p2 incineroar"],
        "unhelped-attack": ["p2 incineroar"],
        "fast-helper": [],
        # Incineroar's is its Protect: the last to move, it has nothing left to guard.
        "partner-switched": ["p2 incineroar"],
    }[name]
    assert failed == expected, (name, flags)


def _actions(reg, pos: Position, choices: tuple[str, str]) -> list:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


def _only(result) -> Position:  # noqa: ANN001
    assert len(result.branches) == 1, [b.events for b in result.branches]
    return result.branches[0].position


def test_the_boost_is_in_the_control(oracle: Oracle) -> None:
    """Positive control for the damage: the helped Hyper Voice does more than an unhelped one."""
    helped = _play(oracle, "partner-attacks")
    alone = _play(oracle, "unhelped-attack")
    for key in ("p2 incineroar", "p2 garchomp"):
        lost_helped = _state(helped[0])[key][0] - _state(helped[1])[key][0]
        lost_alone = _state(alone[0])[key][0] - _state(alone[1])[key][0]
        assert lost_helped > lost_alone > 0, (key, lost_helped, lost_alone)


@pytest.mark.parametrize("name", sorted(CASES))
def test_helping_hand_against_showdown(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    positions = _play(oracle, name)
    _check_showdown(name, positions)
    mine = positions[0]
    for turn, choices in enumerate(CASES[name][1], start=1):
        mine = _only(resolve_turn(reg, mine, _actions(reg, mine, choices), budget=BUDGET))
        assert _state(mine) == _state(positions[turn]), (
            name, turn, _state(mine), _state(positions[turn]),
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


def _clear_stats(pos: Position) -> Position:
    # Showdown's positions carry its final stats, which the port refuses as it would a
    # transformed Pokemon's. Nobody here is transformed, so they follow from the spreads.
    for side in pos.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    return pos


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_fails_it_too(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    """The port plays Showdown's turns, and matches Python's position turn by turn."""
    positions = _play(oracle, name)
    node = rustnode.node_for(reg)
    assert node is not None
    mine = _clear_stats(positions[0])
    for turn, choices in enumerate(CASES[name][1], start=1):
        actions = _actions(reg, mine, choices)
        there = node.resolve(mine, actions, BUDGET, select=0)
        assert there is not None and there.position is not None, f"the port refused turn {turn}"

        os.environ[rustnode.ENV_ENABLE] = "0"
        rustnode.reset()
        here = _only(resolve_turn(reg, mine, actions, budget=BUDGET))
        os.environ[rustnode.ENV_ENABLE] = "1"
        rustnode.reset()
        node = rustnode.node_for(reg)
        assert node is not None

        assert _state(there.position) == _state(positions[turn]), (
            name, turn, _state(there.position), _state(positions[turn]),
        )
        assert here.to_json() == there.position.to_json(), (name, turn)
        mine = there.position
