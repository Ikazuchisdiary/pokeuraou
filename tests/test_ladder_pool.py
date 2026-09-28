"""One position on every core: a ladder's cells read by worker processes (IKA-364).

What these hold:

- **the pool is the serial read** on the count clock: with the serial chunks and with the
  chunks the pool cuts for its workers, every rung (stage, value, strategy, counted work),
  the stop and the stage left unfinished are the serial read's to the bit -- open roots at
  depth 2 and 3, a Bayesian root of several completions, and a budget that ends inside a
  stage. The positive control: the workers read cells (`POOL_COUNTS`), the reader none;
- **the wall clock** reads with the pool, and the caller's stop ends a read inside a stage
  with the stages completed before it (the chunks out are thrown away, and the next read
  is the serial one again).

Played on `rizabanadohido` under hp-share with damage-ordered children (``n``): no Q.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou import humanplay, ladder
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.teams import load_roster

from .test_ladder import LEAF, _played

WORKERS = 2


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture(scope="module")
def pool(roster):  # noqa: ANN001, ANN201
    got = humanplay.use_ladder_pool(WORKERS + 1, roster.reg, ("hp-share",))
    assert got == WORKERS
    yield got
    ladder.stop_pool()


def _node(reg, pos, width, spreads=None, side=0):  # noqa: ANN001, ANN202
    ours = narrow(reg, pos, 0, limit=width).actions
    theirs = narrow(reg, pos, 1, limit=width).actions
    if spreads is None:
        got = humanplay.search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
        eq = got.equilibrium
        m = np.asarray(got.payoff, dtype=np.float64)
        start = SimpleNamespace(
            row_strategy=eq.row_strategy if side == 0 else eq.col_strategy,
            col_strategies=[eq.col_strategy if side == 0 else eq.row_strategy])
        return ours, theirs, [ladder.Item(pos)], [m if side == 0 else -m.T], [1.0], start
    answers = humanplay.belief_solve(reg, pos, ours, theirs, spreads,
                                     {side: LEAF, 1 - side: humanplay._not_asked},
                                     budget=Budget.matrix(), sides=(side,))
    got = answers[side]
    _row, _col, built, weights = got.node_payoff
    start = SimpleNamespace(row_strategy=got.strategy, col_strategies=list(got.replies))
    matrices = [m if side == 0 else -np.asarray(m).T for m in built]
    return ours, theirs, list(spreads[1 - side]), matrices, weights, start


def _read(reg, node, stages, budget_ms=None, *, side=0, workers, **how):  # noqa: ANN001, ANN202
    ours, theirs, items, matrices, weights, start = node
    saved = ladder._POOL
    if not workers:
        ladder._POOL = None
    try:
        return ladder.read(reg, side, ours, theirs, items, matrices, weights, start, LEAF,
                           budget=Budget.matrix(), stages=ladder.parse_ladder(stages),
                           budget_ms=budget_ms, **how)
    finally:
        ladder._POOL = saved


def _same(a, b) -> None:  # noqa: ANN001
    assert [r.stage for r in a.rungs] == [r.stage for r in b.rungs]
    for x, y in zip(a.rungs, b.rungs, strict=True):
        assert x.value == y.value and x.fresh == y.fresh and x.work == y.work
        assert (x.rows, x.cols) == (y.rows, y.cols)
        np.testing.assert_array_equal(x.strategy, y.strategy)
        assert x.spent_ms == y.spent_ms
    assert (a.stopped, a.unfinished, a.abandoned) == (b.stopped, b.unfinished, b.abandoned)
    np.testing.assert_array_equal(a.strategy, b.strategy)
    assert a.value == b.value and a.work == b.work
    assert a.unmodelled == b.unmodelled


@pytest.mark.parametrize("chunk", [ladder.CHUNK, None])
def test_the_pool_is_the_serial_read_on_the_count_clock(roster, pool, monkeypatch, chunk) -> None:  # noqa: ANN001
    reg = roster.reg
    monkeypatch.setattr(ladder, "POOL_CHUNK", chunk)
    stages = "d2r2b3n4+d2r4ban4x+d3r2ban4/r2ban4"
    compared = cut = 0
    for pos in _played(roster)[:3]:
        node = _node(reg, pos, 6)
        before = dict(ladder.POOL_COUNTS)
        pooled = _read(reg, node, stages, workers=True)
        assert pooled.workers == WORKERS
        # The positive control: the workers read every fresh cell of every stage.
        assert ladder.POOL_COUNTS["cells"] - before["cells"] == sum(r.fresh for r in pooled.rungs)
        assert ladder.POOL_COUNTS["chunks"] > before["chunks"]
        serial = _read(reg, node, stages, workers=False)
        assert serial.workers == 0
        _same(pooled, serial)
        compared += 1
        if len(serial.rungs) >= 2 and serial.rungs[0].spent_ms < serial.rungs[1].spent_ms:
            # A budget inside the second stage: the same place to stop, the same answer.
            budget_ms = (serial.rungs[0].spent_ms + serial.rungs[1].spent_ms) / 2
            a = _read(reg, node, stages, budget_ms, workers=True)
            b = _read(reg, node, stages, budget_ms, workers=False)
            _same(a, b)
            assert a.stopped == "budget" and len(a.rungs) == 1
            cut += 1
    assert compared == 3 and cut >= 1


def test_the_pool_reads_a_bayesian_root_as_the_serial_read(roster, pool) -> None:  # noqa: ANN001
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    checked = 0
    for pos in _played(roster)[:3]:
        spreads = {s: completions(reg, pos, s, sheet, seen=frozenset({0, 1})) for s in (0, 1)}
        if len(spreads[1]) < 2:
            continue
        for side in (0, 1):
            node = _node(reg, pos, 5, spreads, side)
            pooled = _read(reg, node, "d2r2b3n4+d2r3ban4", side=side, workers=True)
            serial = _read(reg, node, "d2r2b3n4+d2r3ban4", side=side, workers=False)
            assert pooled.rungs, "no stage completed"
            _same(pooled, serial)
            checked += 1
    assert checked >= 2


def test_the_wall_clock_and_the_stop(roster, pool, monkeypatch) -> None:  # noqa: ANN001
    reg = roster.reg
    pos = _played(roster)[1]
    node = _node(reg, pos, 6)
    stages = "d2r2b3n4+d3r4ban4/r3ban4"
    walled = _read(reg, node, stages, 60_000.0, workers=True, clock="wall")
    assert walled.stopped == "done" and len(walled.rungs) == 2 and walled.workers == WORKERS
    # The stop, set when the first cell of the depth-3 stage comes back: the read ends
    # inside that stage with the first stage's answer, and the chunks still out are thrown
    # away.
    stop = threading.Event()
    cells = ladder._Pool.cells

    def stopping(self, asks, stage, *rest, **how):  # noqa: ANN001, ANN003, ANN202
        inner = cells(self, asks, stage, *rest, **how)
        try:
            for index, got in inner:
                if stage.depth == 3 and got is not ladder._TICK:
                    stop.set()
                yield index, got
        finally:
            inner.close()

    monkeypatch.setattr(ladder._Pool, "cells", stopping)
    before = dict(ladder.POOL_COUNTS)
    got = _read(reg, node, stages, workers=True, clock="wall", stop=stop)
    monkeypatch.undo()
    assert [r.stage for r in got.rungs] == ["d2r2b3n4"] and got.stopped == "stop"
    assert got.unfinished == "d3r4ban4/r3ban4" and got.abandoned
    np.testing.assert_array_equal(got.strategy, walled.rungs[0].strategy)
    assert ladder.POOL_COUNTS["dropped"] > before["dropped"]
    # The pool is clean for the next read.
    again = _read(reg, node, "d2r2b3n4", workers=True)
    assert again.rungs[0].value == walled.rungs[0].value


def test_the_tail_reads_the_next_stage_ahead_on_the_wall_clock(roster, pool, monkeypatch) -> None:  # noqa: ANN001
    """IKA-370, IKA-374: on the wall clock a stage's tail reads the likely cells of its
    oracle pass and of the next stage, in chunks, and a stage takes those it asks for --
    back, or still out (waited for), a chunk's cells it did not ask kept for a later stage.
    No value moves (hp-share answers a cell alike in any chunk) and only the cells taken
    are counted, cell by cell: every stage, value, strategy and counted work is the serial
    read's. With `TAIL` the last chunks are cut small. The positive controls: stages took
    cells read ahead, some from a chunk they took only a part of."""
    reg = roster.reg
    monkeypatch.setattr(ladder, "AHEAD", True)
    stages = "d2r2b3n4+d2r3b3n4+d2r4ban4x+d2r5ban4x+d3r2ban4/r2ban4"
    used = partial = 0
    for tail, pos in [(tail, pos) for tail in (False, True) for pos in _played(roster)[:3]]:
        monkeypatch.setattr(ladder, "TAIL", tail)
        node = _node(reg, pos, 6)
        walled = _read(reg, node, stages, workers=True, clock="wall")
        serial = _read(reg, node, stages, workers=False)
        assert walled.stopped == serial.stopped == "done"
        assert [r.stage for r in walled.rungs] == [r.stage for r in serial.rungs]
        for x, y in zip(walled.rungs, serial.rungs, strict=True):
            assert x.value == y.value and x.fresh == y.fresh and x.work == y.work
            assert (x.rows, x.cols) == (y.rows, y.cols)
            np.testing.assert_array_equal(x.strategy, y.strategy)
        assert walled.work == serial.work and walled.unmodelled == serial.unmodelled
        used += walled.pool["specUsed"]
        partial += walled.pool["specPartial"]
        assert walled.pool["specCells"] >= walled.pool["specUsed"]
    assert used > 0, "no stage took a cell read ahead"
    assert partial > 0, "no stage took a part of a chunk read ahead"


def test_a_worker_reads_a_stage_chunk_before_one_read_ahead() -> None:
    """IKA-374: the worker's order (`ladder._next_message`): the first sent, but a stage's
    chunk before a chunk read ahead sent before it; nothing passes a read's context or the
    end."""
    cells, ahead, read = ("cells",), ("ahead",), ("read",)
    assert ladder._next_message([cells, ahead]) == 0
    assert ladder._next_message([ahead, ahead, cells]) == 2
    assert ladder._next_message([ahead, read, cells]) == 0
    assert ladder._next_message([read, cells]) == 0
    assert ladder._next_message([ahead, None]) == 0
    assert ladder._next_message([None, cells]) == 0


def test_the_tail_is_cut_small() -> None:
    """IKA-374: `TAIL`'s chunks -- never more than a worker's share of the cells left, so the
    last are single; without it the stage's even chunks, as before."""
    pool = ladder._Pool([], [object()] * 4, None)
    keys = list(range(50))
    even = pool.split(keys)
    size = pool.chunk(50)
    assert even == [keys[at:at + size] for at in range(0, 50, size)]
    cut = pool.split(keys, shrink=True)
    assert [k for c in cut for k in c] == keys
    sizes = [len(c) for c in cut]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] == size and sizes[-3:] == [1] * 3
    assert all(len(c) <= -(-(50 - sum(sizes[:n])) // 4) for n, c in enumerate(cut))


def test_a_filled_budget_ends_at_the_budget(roster, pool, monkeypatch) -> None:  # noqa: ANN001
    """IKA-370: filling the wall clock's budget (`FILL_WALL`) begins every stage while time
    is left, so a read ends at its budget inside a stage it began (never before one it did
    not), and ends there without waiting for the chunks still out (they are taken, and
    thrown away, when the next read begins). The answer is its last completed stage."""
    reg = roster.reg
    monkeypatch.setattr(ladder, "FILL_WALL", True)
    stages = "d2r2b3n4+d3r4ban4/r3ban4+d4r3ban4/r3ban4/r3ban4"
    cut = 0
    for pos in _played(roster)[:3]:
        node = _node(reg, pos, 6)
        got = _read(reg, node, stages, 300.0, workers=True, clock="wall")
        assert got.wall_ms < 300.0 + 1000.0, "the read waited past its budget"
        if got.stopped == "budget":
            assert got.abandoned and got.unfinished is not None
            cut += 1
            if got.rungs:
                np.testing.assert_array_equal(got.strategy, got.rungs[-1].strategy)
        again = _read(reg, node, "d2r2b3n4", workers=True)
        serial = _read(reg, node, "d2r2b3n4", workers=False)
        assert again.rungs[0].value == serial.rungs[0].value
    assert cut >= 1, "no read met its budget"
