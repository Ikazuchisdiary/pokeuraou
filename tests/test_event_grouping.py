"""Which action each event belongs to, and why the resolver has to be the one to say.

`Branch.events` is a flat trace, and "what happened this turn" is a question about
actions: a reader wants Flare Blitz's damage and its recoil under Flare Blitz, not
interleaved with the partner's move and the end-of-turn poison. `Branch.acts` carries the
cut points, and this file tests the two things a reader depends on.

**Coverage.** Every event belongs to exactly one action, the offsets run forwards, and the
last group is the residual phase. A grouping that silently drops the tail of a turn is
worse than no grouping at all, because it reads as a turn where nothing else happened.

**Attribution that cannot be guessed.** The trace names the move that caused each line but
never who used it -- `p1b -78 (flareblitz)` -- so in a mirror, where both sides use the
same moves, the flat trace is genuinely ambiguous. A reader reconstructing the grouping by
matching reasons against the chosen moves would be right most of the time, which is the
worst way for a log to be wrong. The mirror test below is the one that fails if attribution
ever goes back to being inferred.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.position import Position
from pokeuraou.resolve import RESIDUAL_PHASE, Budget, resolve_turn
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from ._harness import load_tool


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _mirror_turns(roster, turns: int = 5, seed: int = 3):  # noqa: ANN001, ANN202
    """Positions from a mirror game, one per turn. Same shape as test_symmetry's."""
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    rng = np.random.default_rng(seed)
    out: list[Position] = []
    for _ in range(turns):
        if pos.ended:
            break
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            break
        out.append(pos.copy())
        result = resolve_turn(
            reg,
            pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        weights = np.array([b.probability for b in result.branches], dtype=np.float64)
        pos = result.branches[int(rng.choice(len(weights), p=weights / weights.sum()))].position
    return out


def test_acts_cover_every_event_exactly_once(roster) -> None:  # noqa: ANN001
    """The offsets partition the trace, in order, ending with the residual phase."""
    reg = roster.reg
    positions = _mirror_turns(roster)
    assert len(positions) >= 3, "the game ended too early to test anything"
    checked = 0
    for pos in positions:
        row = narrow(reg, pos, 0, limit=4).actions
        col = narrow(reg, pos, 1, limit=4).actions
        for ours in row[:2]:
            for theirs in col[:2]:
                result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.matrix())
                for branch in result.branches:
                    acts = branch.acts
                    assert acts, "a resolved turn always ran at least the residual phase"
                    offsets = [start for start, _label in acts]
                    assert offsets == sorted(offsets), f"out of order: {acts}"
                    assert offsets[0] == 0, (
                        f"{offsets[0]} event(s) before the first action: "
                        f"{branch.events[: offsets[0]]}"
                    )
                    assert offsets[-1] <= len(branch.events), (
                        f"offset past the end of the trace: {acts} for {branch.events}"
                    )
                    assert acts[-1][1] == RESIDUAL_PHASE, (
                        f"the last group must be the end of the turn: {acts[-1]}"
                    )
                    for _start, label in acts[:-1]:
                        assert label[:2] in ("p1", "p2"), (
                            f"an action's label names the slot that acted: {label!r}"
                        )
                    checked += 1
    assert checked >= 40, f"only {checked} branches checked"


def test_a_mirror_attributes_the_same_move_to_both_sides(roster) -> None:  # noqa: ANN001
    """Both sides use one move; the two consequences land under different actions.

    This is the case the flat trace cannot express. Each side's Pokemon uses the same move
    at the other, so both damage lines read `(flareblitz)` and differ only in who took the
    damage -- and for a move that hits its user (recoil, Life Orb) not even in that.
    """
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    moves = [m.id for m in pos.sides[0].pokemon[pos.sides[0].active[0]].moves]
    shared = next((m for m in moves if m not in ("protect", "fakeout")), moves[0])
    actions = []
    for side in (0, 1):
        wanted = next(
            (
                a
                for a in narrow(reg, pos, side, limit=40).actions
                if a.to_choice().startswith(f"move {moves.index(shared) + 1} ")
            ),
            None,
        )
        assert wanted is not None, f"{shared} should be offered to side {side}"
        actions.append(wanted)

    result = resolve_turn(reg, pos, actions, budget=Budget.exact())
    branch = max(result.branches, key=lambda b: b.probability)
    name = reg.moves[shared].name
    # A slot can appear more than once in a turn -- a Mega Evolution is its own queued
    # action, ahead of the move -- so the users of *this move* are what matters.
    users = [
        label.split()[0] for _start, label in branch.acts if label.endswith(f" {name}")
    ]
    assert len(users) == 2, f"both sides used {name} once each: {branch.acts}"
    assert users[0][:2] != users[1][:2], (
        f"the two uses of {name} must be attributed to opposite sides: {branch.acts}"
    )


def test_the_log_groups_without_losing_a_line(roster) -> None:  # noqa: ANN001
    """`show_game.group_events` reflows the trace and keeps every line.

    The renderer drops a move name that the header already gave, so the *text* changes --
    but the number of lines may not, because a log that quietly swallows an event sends a
    reader looking for a bug in the game rather than in the log.
    """
    show_game = load_tool("show_game")
    reg = roster.reg
    loc = show_game.Localiser(reg, show_game.load_names("ja"))
    positions = _mirror_turns(roster)
    assert positions, "no positions to render"
    total = labelled = 0
    for pos in positions:
        row = narrow(reg, pos, 0, limit=3).actions
        col = narrow(reg, pos, 1, limit=3).actions
        for ours in row[:2]:
            for theirs in col[:2]:
                result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.exact())
                for branch in result.branches:
                    groups = show_game.group_events(
                        loc, list(branch.events), list(branch.acts), pos
                    )
                    kept = sum(len(lines) for _header, lines in groups)
                    assert kept == len(branch.events), (
                        f"{len(branch.events) - kept} line(s) lost in grouping: "
                        f"{branch.events} -> {groups}"
                    )
                    headers = [h for h, _l in groups if h]
                    assert headers, f"every group needs a header: {groups}"
                    labelled += any("ターン終了" in h for h in headers)
                    total += 1
    assert total >= 8, f"only {total} branches rendered"
    # Not every branch has one: a turn with no weather, status, item or trap in play has
    # an empty residual phase, and an empty group is not printed. But a mirror that runs
    # several turns accumulates them, so zero across the lot would mean the end-of-turn
    # group is never labelled at all.
    assert labelled > 0, "no branch labelled its end-of-turn group"


# ---------------------------------------------------------------------------
# The port's trace (IKA-215): the same lines and the same cuts, when asked for
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def node(roster):  # noqa: ANN001, ANN201
    from pokeuraou import rustnode

    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    opened = rustnode.RustNode(roster.reg)
    yield opened
    opened.close()


def test_the_port_cuts_the_trace_as_python_does(roster, node) -> None:  # noqa: ANN001
    """`turn` with `events` gives every branch Python's lines and Python's `acts`."""
    reg = roster.reg
    positions = _mirror_turns(roster)
    assert len(positions) >= 3, "the game ended too early to test anything"
    lines = 0
    for pos in positions:
        row = narrow(reg, pos, 0, limit=4).actions
        col = narrow(reg, pos, 1, limit=4).actions
        for ours in row[:2]:
            for theirs in col[:2]:
                for budget in (Budget.matrix(), Budget.exact()):
                    result = resolve_turn(reg, pos, [ours, theirs], budget=budget)
                    port = node.turn(pos, [ours, theirs], budget, full=True, events=True)
                    assert port is not None, "the port refused the turn"
                    assert len(port.outcomes) == len(result.branches)
                    for mine, answered in zip(result.branches, port.outcomes, strict=True):
                        assert answered.events == mine.events
                        assert answered.acts == [tuple(a) for a in mine.acts]
                        lines += len(mine.events)
    assert lines >= 200, f"only {lines} lines compared"


def test_the_port_keeps_nothing_unless_asked(roster, node) -> None:  # noqa: ANN001
    """Off by default: no trace comes back, so the generation road carries none."""
    reg = roster.reg
    pos = _mirror_turns(roster, turns=1)[0]
    ours = narrow(reg, pos, 0, limit=1).actions[0]
    theirs = narrow(reg, pos, 1, limit=1).actions[0]
    port = node.turn(pos, [ours, theirs], Budget.matrix(), full=True)
    assert port is not None and port.outcomes
    assert all(not b.events and not b.acts for b in port.outcomes)


def test_the_port_attributes_a_mirror_to_both_sides(roster, node) -> None:  # noqa: ANN001
    """The mirror above, answered by the port: the two uses land on opposite sides."""
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    moves = [m.id for m in pos.sides[0].pokemon[pos.sides[0].active[0]].moves]
    shared = next((m for m in moves if m not in ("protect", "fakeout")), moves[0])
    actions = [
        next(
            a
            for a in narrow(reg, pos, side, limit=40).actions
            if a.to_choice().startswith(f"move {moves.index(shared) + 1} ")
        )
        for side in (0, 1)
    ]
    port = node.turn(pos, actions, Budget.exact(), full=True, events=True)
    assert port is not None and port.outcomes
    branch = max(port.outcomes, key=lambda b: b.probability)
    name = reg.moves[shared].name
    users = [label.split()[0] for _start, label in branch.acts if label.endswith(f" {name}")]
    assert len(users) == 2 and users[0][:2] != users[1][:2], branch.acts
    assert branch.acts[-1][1] == RESIDUAL_PHASE
