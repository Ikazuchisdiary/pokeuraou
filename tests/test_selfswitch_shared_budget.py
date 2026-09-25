"""The pauses of one resumed turn share its branch budget (IKA-284).

The self-switch node flattens every option's rest of the turn (`alternativesEncoded`), and
a rest of the turn can pause again: a Parting Shot, then a Heat Wave that takes an
Emergency Exit Golisopod below half. Under `Budget.exact()` -- the budget a game advances
on, and so the one every self-switch node's pause carries -- that second pause happened in
hundreds of ways (every roll, crit and burn of the Heat Wave that crossed half), and each
was resumed at the whole 512-branch budget. A board game (IKA-282, seed 28201, game 159)
flattened 363,000 leaves and passed 4 GB in one node. The pauses now share the budget, as
`run_queue`'s live branches do.

The fact first: Showdown asks the same side twice in one turn, so the nested pause is the
game's and the fix must keep it -- a choice inside the option, coarser, not dropped. Then
the port: the node is small, it still folds through the second choice, and Python's
`turn_leaves` (the positions road) walks the same leaves. The control is the same turn with
no Emergency Exit: nothing pauses twice and the budget is not touched.

Positive control (records/IKA-284.md): the pre-fix exe (`POKEURAOU_RUST_NODE_BIN`) fails
`test_the_node_stays_small_and_keeps_the_second_choice` with tens of thousands of leaves.
"""

from __future__ import annotations

import pytest

from pokeuraou import port as door
from pokeuraou.actions import side_actions
from pokeuraou.budget import Budget
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], sp: dict[str, int]) -> TeamSet:
    base = {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
    base.update(sp)
    return TeamSet(species=species, ability=ability, nature="Modest", moves=moves, sp=base)


TEAM_A = [
    _mon("Delphox", "Blaze", ["heatwave", "protect"], {"spe": 32}),
    _mon("Politoed", "Water Absorb", ["hypervoice", "protect"], {}),
    _mon("Garchomp", "Rough Skin", ["protect", "earthquake"], {}),
    _mon("Kingambit", "Defiant", ["protect", "ironhead"], {}),
]


def _team_b(ability: str) -> list[TeamSet]:
    return [
        _mon("Grimmsnarl", "Prankster", ["partingshot", "protect"], {"hp": 32}),
        _mon("Golisopod", ability, ["protect", "swordsdance"], {}),
        _mon("Archaludon", "Stamina", ["protect", "flashcannon"], {}),
        _mon("Garchomp", "Rough Skin", ["protect", "earthquake"], {}),
    ]


#: Turn 1 puts Golisopod above half (Heat Wave; Grimmsnarl and Politoed protect). Turn 2 is
#: the one asked about: Parting Shot first (Prankster), then Heat Wave and Hyper Voice.
SETUP = ["move 1, move 2", "move 2, move 2"]
TURN = ["move 1, move 1", "move 1 1, move 2"]

#: Leaves in the whole node, both options. The pre-fix exe made tens of thousands here;
#: sharing the budget keeps a pause's replacement options at about the budget each.
MOST_LEAVES = 8 * 512


def _force(request: dict | None) -> list[bool]:
    return list((request or {}).get("forceSwitch") or [])


def _play(oracle: Oracle, ability: str) -> tuple[Position, list[list[bool]], list[str]]:
    """The position before TURN, each switch request Showdown made for p2 during it, the log."""
    handle = oracle.create(FORMAT_ID, TEAM_A, _team_b(ability), policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    handle.step(SETUP)
    assert handle.choice_errors == [], handle.choice_errors
    assert not any(_force(r) for r in handle.requests)
    before = Position.from_json(handle.position)
    handle.step(TURN)
    assert handle.choice_errors == [], handle.choice_errors
    asked: list[list[bool]] = []
    log = list(handle.log)
    while _force(handle.requests[1]):
        flags = _force(handle.requests[1])
        asked.append(flags)
        bench = 3
        handle.step([None, ", ".join(f"switch {bench}" if f else "pass" for f in flags)])
        assert handle.choice_errors == [], handle.choice_errors
        log += list(handle.log)
    handle.close()
    return before, asked, log


def _node(reg, before: Position):  # noqa: ANN001, ANN202
    actions = [
        next(a for a in side_actions(reg, before, side) if a.to_choice() == TURN[side])
        for side in (0, 1)
    ]
    turn = door.turn(reg, before, actions, Budget.exact(), full=True)
    assert not turn.outcomes and len(turn.pauses) == 1, "Parting Shot pauses first, once"
    return turn.pauses[0]


def _chooses(fold: dict) -> bool:
    """Whether a plan's fold still has a replacement choice inside it."""
    if "best" in fold:
        return True
    return any(_chooses(part) for _weight, part in fold.get("avg", []))


def test_showdown_asks_the_same_side_twice_in_one_turn(oracle: Oracle) -> None:
    _before, asked, log = _play(oracle, "Emergency Exit")
    assert asked == [[True, False], [False, True]], asked
    assert "|-activate|p2b: Golisopod|ability: Emergency Exit" in log, log


def test_the_control_asks_once(oracle: Oracle) -> None:
    _before, asked, log = _play(oracle, "Shell Armor")
    assert asked == [[True, False]], asked
    assert not any("Emergency Exit" in line for line in log)


def test_the_node_stays_small_and_keeps_the_second_choice(reg, oracle: Oracle) -> None:  # noqa: ANN001
    before, _asked, _log = _play(oracle, "Emergency Exit")
    pause = _node(reg, before)
    answer = door.alternatives_encoded(reg, pause, objectives=["hp-share"], encode=False)
    assert len(answer.options) == 2
    counts = [plan.count for plan in answer.plans]
    assert sum(counts) <= MOST_LEAVES, counts
    # Coarser, not gone: every option still folds through Golisopod's replacement.
    assert all(plan.suspended and _chooses(plan.fold) for plan in answer.plans)
    # The positions road asks for the same shared budget and walks the same leaves.
    _chooser, alternatives = door.resume_alternatives(reg, pause)
    walked = [len(door.turn_leaves(reg, resumed).positions) for _o, resumed in alternatives]
    assert walked == counts


def test_the_control_is_untouched(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Nothing pauses twice: every leaf is a branch of the resumed turn, as before."""
    before, _asked, _log = _play(oracle, "Shell Armor")
    pause = _node(reg, before)
    answer = door.alternatives_encoded(reg, pause, objectives=["hp-share"], encode=False)
    _chooser, alternatives = door.resume_alternatives(reg, pause)
    assert [plan.count for plan in answer.plans] == [
        len(resumed.outcomes) for _o, resumed in alternatives
    ]
    assert not any(plan.suspended or _chooses(plan.fold) for plan in answer.plans)
