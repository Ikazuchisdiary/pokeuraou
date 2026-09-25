"""A depth-2 pass asks the port by the stage, and reads a clean turn off another completion's (IKA-295).

Two changes to how `search._refine_cells` reaches the port, neither of which may move an
answer:

* **Batched crossings.** A pass used to ask the port per refined cell (its turn) and per
  sub-game (two damage scores and a fill). Now it asks every cell's turn in one line
  (`port.turns`, the port's `many`), every branch's two menus in one (`narrow_many`), and
  the sub-games' nodes a few at a time (`port.pending_payoffs`, the port's `fills`). Each
  answer must be the one its own crossing gave -- the same turn, the same candidates, the
  same arrays to the byte.
* **Sub-game cells read off.** The same branch of such a cell is a sub-game in every
  completion, the same position but for the bench; its cells that bring no hidden
  Pokemon in resolve to the same turns. The port fills the first and reads the others'
  cells off it within one `fills` crossing (`Like` in `encoded_node.rs`, which checks the
  two positions differ only in the hidden slots and each read turn left them untouched).
  Every node must be the node filled alone, to the byte.
* **A shared turn.** The cells of a hidden depth-2 pass are the same two actions in every
  completion of the hidden bench. Where the actions cannot bring a hidden Pokemon in, the
  turn is the same turn in every completion and only the bench differs, so it is resolved
  once and read off (`beliefnode.turn_in_completion`). The claim is checked against the
  port resolving each completion itself, cell by cell, and the check that guards it --
  every branch must leave the hidden slots as the reference had them -- is shown to fire
  on cells that do bring one in, where reading the turn off would be wrong.

The end-to-end answer (strategies, values, notes, counts of a hidden depth-2 decision) is
`tests/test_subgame_batch.py`'s comparison with the per-sub-game pass, which resolves every
cell's turn itself; that comparison now covers both changes.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from pokeuraou import port, timing
from pokeuraou import search as search_mod
from pokeuraou.beliefnode import reaches_bench, turn_in_completion
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder, EncodingRules
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow, narrow_many
from pokeuraou.position import Position
from pokeuraou.rustnode import PortBranch
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from ._port import resolve_turn
from .test_subgame_batch import _hidden, _hidden_answers, _SizedLeaf


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _played(roster, first: int, seed: int, turns: int = 3) -> list[Position]:  # noqa: ANN001
    """Turn 1 and a few turns into one game of two different fours."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    fours = [sheet[:4], sheet[2:6], sheet[1:5], [sheet[0], sheet[2], sheet[4], sheet[5]]]
    pos = position_from_sets(reg, fours[first], fours[(first + 1) % 4])
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
            reg, pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        pos = max(result.branches, key=lambda b: b.probability).position
    return out


@pytest.fixture(scope="module")
def positions(roster):  # noqa: ANN001, ANN201
    out: list[Position] = []
    for first, seed in ((0, 11), (1, 5), (2, 7), (3, 2)):
        out.extend(_played(roster, first, seed))
    return out


def _turn_bytes(turn: port.PortTurn) -> tuple:
    return (
        [float(p).hex() for p in turn.branches],
        [float(p).hex() for p in turn.suspended],
        turn.exact,
        sorted(turn.unmodelled),
        [(float(b.probability).hex(), b.position.to_json()) for b in turn.outcomes or []],
        len(turn.pauses or []),
    )


def _unchecked(reference: Position, turn: port.PortTurn, position: Position, side: int, slots):  # noqa: ANN001, ANN202
    """`turn_in_completion` without its checks: the fault the checks are there to stop."""
    outcomes = []
    for branch in turn.outcomes or []:
        made = branch.position.copy()
        for slot in slots:
            made.sides[side].pokemon[slot] = position.sides[side].pokemon[slot].copy()
        made.sides[side].mega_capable_slots = list(position.sides[side].mega_capable_slots)
        outcomes.append(PortBranch(branch.probability, made))
    return port.PortTurn(
        branches=list(turn.branches), suspended=[], exact=turn.exact,
        unmodelled=tuple(turn.unmodelled), outcomes=outcomes, pauses=[],
    )


def test_a_shared_turn_is_the_completions_own_turn(roster, positions) -> None:  # noqa: ANN001
    """Every cell of 6x6 menus, every hidden side, every completion: where
    `turn_in_completion` reads the turn off the first completion's, it is the turn the
    port resolves in that completion to the byte (branch weights, notes, every branch's
    position). And the guard has teeth: on cells that bring a hidden Pokemon in, it
    declines, and the unchecked read would have been a different turn."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    budget = Budget.matrix()
    seen = Counter()
    for pos in positions:
        ours, theirs = (narrow(reg, pos, s, limit=6).actions for s in (0, 1))
        for side in (0, 1):
            items = completions(reg, pos, side, sheet)
            if len(items) < 2:
                continue
            slots = tuple(items[0].slots)
            dirty = reaches_bench(reg, ours, theirs, {side: slots}, items[0].position)
            for i, a in enumerate(ours):
                for j, b in enumerate(theirs):
                    reference = port.turn(reg, items[0].position, [a, b], budget, full=True)
                    for item in items[1:]:
                        own = port.turn(reg, item.position, [a, b], budget, full=True)
                        read = turn_in_completion(
                            items[0].position, reference, item.position, side, slots
                        )
                        kind = "dirty" if dirty[i, j] else "clean"
                        if read is None:
                            seen[f"{kind} declined"] += 1
                            if dirty[i, j] and reference.outcomes and not reference.pauses:
                                wrong = _unchecked(
                                    items[0].position, reference, item.position, side, slots
                                )
                                seen["declined, unchecked would differ"] += int(
                                    _turn_bytes(wrong) != _turn_bytes(own)
                                )
                            continue
                        assert _turn_bytes(read) == _turn_bytes(own), (pos.turn, side, i, j)
                        seen[f"{kind} read"] += 1
                        before = item.position.sides[side].mega_capable_slots
                        seen["read with other mega slots"] += int(
                            before != items[0].position.sides[side].mega_capable_slots
                        )
    # The road is taken, on many cells, and it is the port's own answer each time.
    assert seen["clean read"] > 500, seen
    # A clean cell is only declined for a paused turn (none in these positions).
    assert seen["clean declined"] <= seen["clean read"] // 20, seen
    # The positive control: cells that bring a hidden Pokemon in are declined by the check,
    # and reading them off anyway would have been wrong.
    assert seen["dirty declined"] > 20, seen
    assert seen["declined, unchecked would differ"] > 20, seen
    # A completion whose bench holds a different set of mega stones was among those read.
    assert seen["read with other mega slots"] > 0, seen


def _with_dirty_cells(roster, positions) -> Position:  # noqa: ANN001
    """The first position whose hidden depth-2 decision refines a cell that can bring a
    hidden Pokemon in (counted by `_cell_turns`; `timing.count` must be patched)."""
    for pos in positions:
        seen = Counter()
        saved = timing.count
        timing.count = lambda name, n=1, seen=seen: seen.update({name: n})
        try:
            _hidden(roster, _SizedLeaf(Encoder(roster.reg)), pos)
        finally:
            timing.count = saved
        if seen["depth2.turns.dirty"] > 0:
            return pos
    raise AssertionError("no position refines a cell that reaches a hidden bench")


def test_the_check_alone_catches_a_cell_marked_clean_wrongly(roster, positions, monkeypatch) -> None:  # noqa: ANN001
    """With `reaches_bench` saying every cell is clean, the hidden depth-2 answer is still the
    per-cell one: every dirty cell's read is declined by the branch check and resolved
    again (counted as touched). The check is what stands between a wrong mask and a wrong
    turn, not only the mask."""
    seen = Counter()
    monkeypatch.setattr(timing, "ON", True)
    monkeypatch.setattr(timing, "count", lambda name, n=1: seen.update({name: n}))
    pos = _with_dirty_cells(roster, positions)
    seen.clear()
    reference = _hidden_answers(_hidden(roster, _SizedLeaf(Encoder(roster.reg)), pos))
    assert seen["depth2.turns.dirty"] > 0
    assert seen["depth2.turns.touched"] == 0
    seen.clear()
    # Only the depth-2 pass's mask: the depth-1 node's own sharing keeps the true one.
    monkeypatch.setattr(search_mod, "_cell_reaches_bench", lambda *_a, **_k: False)
    forced = _hidden_answers(_hidden(roster, _SizedLeaf(Encoder(roster.reg)), pos))
    assert seen["depth2.turns.dirty"] == 0
    assert seen["depth2.turns.touched"] > 0, seen
    assert forced == reference


def test_without_the_check_a_wrong_mask_moves_the_answer(roster, positions, monkeypatch) -> None:  # noqa: ANN001
    """The control for the one above: every cell marked clean and the branch check taken
    out, and the hidden depth-2 answer is no longer the per-cell one."""
    import pokeuraou.beliefnode as beliefnode

    monkeypatch.setattr(timing, "ON", True)
    monkeypatch.setattr(timing, "count", lambda _name, _n=1: None)
    pos = _with_dirty_cells(roster, positions)
    reference = _hidden_answers(_hidden(roster, _SizedLeaf(Encoder(roster.reg)), pos))
    # Only the depth-2 pass's mask: the depth-1 node's own sharing keeps the true one.
    monkeypatch.setattr(search_mod, "_cell_reaches_bench", lambda *_a, **_k: False)
    monkeypatch.setattr(beliefnode, "turn_in_completion", _unchecked)
    broken = _hidden_answers(_hidden(roster, _SizedLeaf(Encoder(roster.reg)), pos))
    assert broken != reference


def test_batched_turns_are_the_turns(roster, positions, monkeypatch) -> None:  # noqa: ANN001
    """`port.turns` in one line is `port.turn` per cell, byte for byte."""
    reg = roster.reg
    budget = Budget.matrix()
    asks = []
    for pos in positions[:4]:
        ours, theirs = (narrow(reg, pos, s, limit=4).actions for s in (0, 1))
        asks.extend((pos, [a, b]) for a in ours for b in theirs)
    seen = Counter()
    monkeypatch.setattr(timing, "count", lambda name, n=1: seen.update({name: n}))
    batched = port.turns(reg, asks, budget, full=True)
    assert seen["port.many.lines"] == 1
    assert seen["port.many.requests"] == len(asks)
    alone = [port.turn(reg, pos, actions, budget, full=True) for pos, actions in asks]
    assert [_turn_bytes(t) for t in batched] == [_turn_bytes(t) for t in alone]


def _candidates(narrowed) -> list:  # noqa: ANN001
    return [(c.action.to_choice(), float(c.score).hex(), c.detail) for c in narrowed.kept]


def test_narrow_many_is_narrow(roster, positions, monkeypatch) -> None:  # noqa: ANN001
    """Both sides' menus at many positions in one line are `narrow`'s, candidate for
    candidate (actions, scores and the detail lines)."""
    reg = roster.reg
    asks = [(pos, side) for pos in positions for side in (0, 1)]
    seen = Counter()
    monkeypatch.setattr(timing, "count", lambda name, n=1: seen.update({name: n}))
    batched = narrow_many(reg, asks, limit=8)
    assert seen["port.many.lines"] == 1
    alone = [narrow(reg, pos, side, limit=8) for pos, side in asks]
    assert [_candidates(n) for n in batched] == [_candidates(n) for n in alone]
    assert [n.considered for n in batched] == [n.considered for n in alone]


def _node_bytes(pending: port.PendingPayoff) -> tuple:
    node = pending.filled
    arrays = tuple(
        getattr(node.encoded, name).tobytes()
        for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    )
    return (
        arrays, node.encoded.decided.tobytes(), node.spans, node.folded, node.exact,
        node.refused, sorted(node.unmodelled), pending.shape, pending.rows,
    )


def test_batched_fills_are_the_fills(roster, positions, monkeypatch) -> None:  # noqa: ANN001
    """The sub-games' nodes in one `fills` crossing are `pending_payoff` per node: the same
    arrays to the byte, the same spans, folds and notes -- cut out of one body."""
    reg = roster.reg
    leaf = _SizedLeaf(Encoder(reg))
    budget = Budget.matrix()
    asks = []
    for pos in positions[:6]:
        ours, theirs = (narrow(reg, pos, s, limit=8).actions for s in (0, 1))
        asks.append((pos, ours, theirs))
    seen = Counter()
    monkeypatch.setattr(timing, "count", lambda name, n=1: seen.update({name: n}))
    batched = port.pending_payoffs(reg, asks, leaf.__call__, budget=budget)
    assert seen["port.fills.lines"] == 1
    assert seen["port.fills.requests"] == len(asks)
    alone = [
        port.pending_payoff(reg, pos, ours, theirs, leaf.__call__, budget=budget)
        for pos, ours, theirs in asks
    ]
    assert [_node_bytes(p) for p in batched] == [_node_bytes(p) for p in alone]


def _sub_games(roster, pos, side):  # noqa: ANN001, ANN202
    """The same branch of clean refined cells in every completion of `side`'s bench: one
    sub-game position per completion, with that position's own menus."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    items = completions(reg, pos, side, sheet)
    ours, theirs = (narrow(reg, pos, s, limit=4).actions for s in (0, 1))
    dirty = reaches_bench(reg, ours, theirs, {side: tuple(items[0].slots)}, pos)
    groups = []
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            if dirty[i, j]:
                continue
            group = []
            for item in items:
                turn = port.turn(reg, item.position, [a, b], Budget.matrix(), full=True)
                if turn.pauses or not turn.outcomes:
                    break
                sub = turn.outcomes[0].position
                if sub.ended:
                    break
                menus = [narrow(reg, sub, s, limit=8).actions for s in (0, 1)]
                group.append((sub, menus[0], menus[1]))
            else:
                groups.append(group)
    return items, groups


def test_a_fill_read_off_another_completions_is_its_own_fill(roster, positions, monkeypatch) -> None:  # noqa: ANN001
    """Sub-games of the same branch in several completions, in one `fills` crossing with
    the first kept and the rest reading their cells' turns off it: every node is the node
    filled on its own, to the byte, and the port did read cells off (most of them -- the
    rest bring a hidden Pokemon in, or are not in the first node's menus). A link between
    two positions that are not completions of one is refused, and changes nothing.

    Twice: under the current encoding, and under IKA-141's revision-1 rule, which reads a
    side's `mega_capable_slots` -- the one side field a completion changes -- into the
    arrays, so a read that kept the first completion's list would show there."""
    reg = roster.reg
    leaf = _SizedLeaf(Encoder(reg))
    budget = Budget.matrix()
    seen = Counter()
    monkeypatch.setattr(timing, "ON", True)
    monkeypatch.setattr(timing, "count", lambda name, n=1: seen.update({name: n}))
    checked = 0
    old_rule = _SizedLeaf(Encoder(reg, rules=EncodingRules(mega_from_slots=True)))
    for pos in positions[:6]:
        for side in (0, 1):
            items, groups = _sub_games(roster, pos, side)
            if len(items) < 2:
                continue
            slots = list(items[0].slots)
            for group in groups[:3]:
                links = [{"keep": True}] + [
                    {"like": {"node": 0, "side": side, "slots": slots}} for _ in group[1:]
                ]
                for scorer in (leaf, old_rule):
                    linked = port.pending_payoffs(
                        reg, group, scorer.__call__, budget=budget, links=links
                    )
                    alone = port.pending_payoffs(reg, group, scorer.__call__, budget=budget)
                    assert [_node_bytes(p) for p in linked] == [_node_bytes(p) for p in alone]
                    checked += len(group) - 1
    assert checked > 40
    assert seen["fill.like"] == checked
    assert seen["fill.like.refused"] == 0
    assert seen["fill.cells.read_off"] > 10 * checked, seen

    # Two different positions are not two completions of one: refused, the same nodes.
    seen.clear()
    one = positions[0]
    other = positions[1]
    menus = [
        (p, narrow(reg, p, 0, limit=6).actions, narrow(reg, p, 1, limit=6).actions)
        for p in (one, other)
    ]
    links = [{"keep": True}, {"like": {"node": 0, "side": 1, "slots": [2, 3]}}]
    linked = port.pending_payoffs(reg, menus, leaf.__call__, budget=budget, links=links)
    alone = port.pending_payoffs(reg, menus, leaf.__call__, budget=budget)
    assert [_node_bytes(p) for p in linked] == [_node_bytes(p) for p in alone]
    assert seen["fill.like.refused"] == 1
    assert seen["fill.cells.read_off"] == 0


def test_a_pass_asks_the_port_by_the_stage(roster, positions, monkeypatch) -> None:  # noqa: ANN001
    """The count, with its positive control: the per-sub-game pass (before IKA-291) crosses
    to the port at least three times a sub-game (two menus and a fill) and once a cell; the
    staged pass crosses a few times a pass."""
    pos = positions[0]
    seen = Counter()
    monkeypatch.setattr(timing, "count", lambda name, n=1: seen.update({name: n}))
    fills = []
    real = search_mod.port.pending_payoffs

    def counting(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        fills.append(len(args[1]))
        return real(*args, **kwargs)

    monkeypatch.setattr(search_mod.port, "pending_payoffs", counting)
    got = _hidden(roster, _SizedLeaf(Encoder(roster.reg)), pos)
    subgames = sum(a.subgames for a in got.values())
    assert subgames >= 40
    crossings = seen["port.many.lines"] + seen["port.fills.lines"]
    assert sum(fills) >= subgames
    assert max(fills) <= search_mod.FILL_BATCH
    # Two sides, two passes: a turns line, a menus line and ceil(sub-games / FILL_BATCH)
    # fills lines a pass (and a second turns line where a read was declined).
    assert crossings <= 4 * 3 + -(-subgames // search_mod.FILL_BATCH) + 4, seen
    assert crossings * 3 < subgames
