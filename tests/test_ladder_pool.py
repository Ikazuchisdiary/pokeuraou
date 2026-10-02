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

import collections
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou import humanplay, ladder, portlp
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


def test_a_sub_game_filled_once_reads_as_every_one_filled(roster, pool, monkeypatch) -> None:  # noqa: ANN001
    """IKA-378 (`SHARE`): a read that fills each child sub-game once -- here in a table of
    its own, on the workers in the shared one -- is the read that fills every one where it
    is met: every rung, the counted work and the notes, on open roots (depth 2 and 3, the
    knock-out fork) and on Bayesian roots (a branch read off another completion's turn).
    hp-share scores a row the same in any batch, so the values are the same to the bit. The
    positive control: sub-games taken from the table, here and across the workers, and from
    an earlier sub-game of the same call. (A deep cell's children are shared on the learned
    leaf's road, `tests/test_ladder_leaf.py`: hp-share reads them one at a time.)"""
    reg = roster.reg
    # Children met again with other menus (n3, n4) and in the other fork (x).
    stages = "d2r2b3n3+d2r3b3n4+d2r4ban4x+d3r2ban4/r2ban4"
    table = {"serial": 0, "pool": 0, "same": 0}
    nodes = [(_node(reg, pos, 6), stages, 0) for pos in _played(roster)[:3]]
    sheet = list(roster.sets)[:6]
    for pos in _played(roster)[:3]:
        spreads = {s: completions(reg, pos, s, sheet, seen=frozenset({0, 1})) for s in (0, 1)}
        if len(spreads[1]) >= 2:
            nodes += [(_node(reg, pos, 5, spreads, side), "d2r2b3n4+d2r3ban4x", side)
                      for side in (0, 1)]
    assert len(nodes) >= 5
    for node, how, side in nodes:
        monkeypatch.setattr(ladder, "SHARE", False)
        plain = _read(reg, node, how, side=side, workers=False)
        assert plain.here["sharedSubgames"] == 0 and plain.rungs
        monkeypatch.setattr(ladder, "SHARE", True)
        serial = _read(reg, node, how, side=side, workers=False)
        pooled = _read(reg, node, how, side=side, workers=True)
        _same(serial, plain)
        _same(pooled, plain)
        table["serial"] += serial.here["sharedSubgames"] - serial.here["sameSubgames"]
        table["same"] += serial.here["sameSubgames"]
        table["pool"] += pooled.pool["sharedSubgames"] - pooled.pool["sameSubgames"]
    assert table["serial"] > 0 and table["pool"] > 0 and table["same"] > 0, table
    # A read takes nothing an earlier read kept: the same node read again fills its own.
    node, how, side = nodes[0]
    again = _read(reg, node, how, side=side, workers=True)
    assert again.pool["keptSubgames"] > 0


def test_the_ports_lps_read_as_pythons(roster, pool) -> None:  # noqa: ANN001
    """IKA-381 (`portlp`): a read whose sub-games are solved in the port (HiGHS in Rust, one
    crossing a call) is the read that solves them with scipy, to the bit -- every rung, the
    counted work and the notes -- in this process and on the workers, on open roots (depth 2
    and 3, the knock-out fork) and on Bayesian roots. hp-share has no forward pass to share,
    so its sub-games reach the port whole (`folds` with the matrix). The positive control:
    LPs the ports solved, here and on the workers."""
    reg = roster.reg
    stages = "d2r2b3n3+d2r3b3n4+d2r4ban4x+d3r2ban4/r2ban4"
    nodes = [(_node(reg, pos, 6), stages, 0) for pos in _played(roster)[:3]]
    sheet = list(roster.sets)[:6]
    for pos in _played(roster)[:3]:
        spreads = {s: completions(reg, pos, s, sheet, seen=frozenset({0, 1})) for s in (0, 1)}
        if len(spreads[1]) >= 2:
            nodes += [(_node(reg, pos, 5, spreads, side), "d2r2b3n4+d2r3ban4x", side)
                      for side in (0, 1)]
    assert len(nodes) >= 5
    solved = {"here": 0, "pool": 0, "bayes": 0, "readerLps": 0}
    try:
        for node, how, side in nodes:
            portlp.set_on(False)
            plain = _read(reg, node, how, side=side, workers=False)
            assert plain.rungs and plain.here["portLps"] == 0
            portlp.set_on(True)
            bayes = portlp.COUNTS["bayes"]
            serial = _read(reg, node, how, side=side, workers=False)
            # IKA-387: the stages' rectangles and the deep children's, in this process.
            solved["bayes"] += portlp.COUNTS["bayes"] - bayes
            pooled = _read(reg, node, how, side=side, workers=True)
            _same(serial, plain)
            _same(pooled, plain)
            solved["here"] += serial.here["portLps"]
            solved["pool"] += pooled.pool["portLps"]
            # IKA-387: the reader's own port solves its stages' rectangles.
            solved["readerLps"] += pooled.here["portLps"]
    finally:
        portlp.set_on(False)
    assert all(v > 0 for v in solved.values()), solved


def test_a_sub_game_another_worker_fills_is_taken_or_filled_again(roster, monkeypatch) -> None:  # noqa: ANN001
    """IKA-380: a sub-game another process is filling (`subshare.Table.claim` answers `BUSY`)
    is looked at again once the call's own are scored: taken where it is kept by then,
    filled here where it is not -- either way every cell's value, notes and the counted work
    are those of the call that fills every sub-game. The positive controls: sub-games
    deferred, then taken (``waited``) or filled again."""
    from pokeuraou import rustnode, search, subshare

    reg = roster.reg
    pos = _played(roster)[1]
    ours = narrow(reg, pos, 0, limit=6).actions
    theirs = narrow(reg, pos, 1, limit=6).actions
    cells = [(pos, a, b) for a in ours[:3] for b in theirs[:3]]

    class Busy:
        """Every key claimed is being filled elsewhere; a get finds ``rows``' rows."""

        def __init__(self, rows: dict) -> None:
            self.rows, self.claimed, self.puts = rows, 0, 0

        def claim(self, key):  # noqa: ANN001, ANN202
            self.claimed += 1
            return subshare.BUSY

        def get(self, key, *, kid=False):  # noqa: ANN001, ANN202
            return self.rows.get(key)

        def put(self, *args, **named) -> None:  # noqa: ANN002, ANN003
            self.puts += 1

    def refine(table):  # noqa: ANN001, ANN202
        work = ladder._zero()
        monkeypatch.setattr(search, "SHARE", table)
        monkeypatch.setattr(search, "WORK", work)
        got = search._refine_cells(reg, cells, LEAF, budget=Budget.matrix(), sub_limit=4,
                                   sub_branches=3)
        return got, work

    saved = rustnode.DIGESTS[0]
    rustnode.DIGESTS[0] = True
    try:
        plain = refine(None)
        kept = subshare.Local()
        assert refine(kept) == plain
        taken, again = Busy(dict(kept._rows)), Busy({})
        before = dict(subshare.COUNTS)
        assert refine(taken) == plain
        waited = subshare.COUNTS["waited"] - before["waited"]
        assert refine(again) == plain
    finally:
        rustnode.DIGESTS[0] = saved
    assert taken.claimed > 0 and waited == taken.claimed and taken.puts == 0
    # Not kept elsewhere by the second look: filled here, and kept.
    assert again.claimed == taken.claimed and again.puts == again.claimed


@pytest.mark.parametrize("split", [False, "split", "passes"])
def test_the_wall_clock_and_the_stop(roster, pool, monkeypatch, split) -> None:  # noqa: ANN001
    reg = roster.reg
    pos = _played(roster)[1]
    node = _node(reg, pos, 6)
    stages = "d2r2b3n4+d3r4ban4/r3ban4"
    # IKA-375: a deep stage's cells go through `deep_cells` with `SPLIT`, `cells` without;
    # IKA-380: `deep_passes` with `PASSES` too.
    monkeypatch.setattr(ladder, "SPLIT", bool(split))
    monkeypatch.setattr(ladder, "PASSES", split == "passes")
    walled = _read(reg, node, stages, 60_000.0, workers=True, clock="wall")
    assert walled.stopped == "done" and len(walled.rungs) == 2 and walled.workers == WORKERS
    # The stop, set when the first cell of the depth-3 stage comes back: the read ends
    # inside that stage with the first stage's answer, and the chunks still out are thrown
    # away.
    stop = threading.Event()
    method = {False: "cells", "split": "deep_cells", "passes": "deep_passes"}[split]
    cells = getattr(ladder._Pool, method)

    def stopping(self, asks, stage, *rest, **how):  # noqa: ANN001, ANN003, ANN202
        inner = cells(self, asks, stage, *rest, **how)
        try:
            for index, got in inner:
                if stage.depth == 3 and got is not ladder._TICK:
                    stop.set()
                yield index, got
        finally:
            inner.close()

    monkeypatch.setattr(ladder._Pool, method, stopping)
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


@pytest.mark.parametrize("workers", [True, False])
def test_a_read_stopped_and_begun_again_with_its_cells(roster, pool, monkeypatch, workers) -> None:  # noqa: ANN001
    """IKA-369: the long reference's read is cut into calls. Stopped inside its depth-3 stage
    and begun again with the cells it kept (``memo``), it is the read that ran to the end --
    every stage's rectangle, value and answer -- and the second call reads only what the
    first did not keep."""
    reg = roster.reg
    pos = _played(roster)[1]
    node = _node(reg, pos, 6)
    stages = "d2r2b3n4+d2r4ban4x+d3r4ban4/r3ban4"
    whole = _read(reg, node, stages, workers=workers)
    assert whole.stopped == "done" and len(whole.rungs) == 3
    total = sum(r.fresh for r in whole.rungs)
    stop = threading.Event()
    memo: dict = {}
    if workers:
        cells = ladder._Pool.cells

        def stopping(self, asks, stage, *rest, **named):  # noqa: ANN001, ANN003, ANN202
            inner = cells(self, asks, stage, *rest, **named)
            try:
                for index, got in inner:
                    if stage.depth == 3 and got is not ladder._TICK and index >= 1:
                        stop.set()
                    yield index, got
            finally:
                inner.close()

        monkeypatch.setattr(ladder._Pool, "cells", stopping)
    else:
        # Read here, a depth-3 cell is `_deep_cell`: the stop once two of them are read.
        deep_cell = ladder._deep_cell
        seen = [0]

        def counting(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            got = deep_cell(*args, **kwargs)
            seen[0] += 1
            if seen[0] >= 2:
                stop.set()
            return got

        monkeypatch.setattr(ladder, "_deep_cell", counting)
    first = _read(reg, node, stages, workers=workers, stop=stop, memo=memo)
    monkeypatch.undo()
    assert first.stopped == "stop" and first.unfinished == "d3r4ban4/r3ban4"
    kept = len(memo)
    # The cells of the stages done, and some of the one stopped inside.
    assert sum(r.fresh for r in first.rungs) < kept < total
    second = _read(reg, node, stages, workers=workers, memo=memo)
    assert second.stopped == "done"
    assert [r.stage for r in second.rungs] == [r.stage for r in whole.rungs]
    for x, y in zip(second.rungs, whole.rungs, strict=True):
        assert x.value == y.value and (x.rows, x.cols) == (y.rows, y.cols)
        np.testing.assert_array_equal(x.strategy, y.strategy)
    for x, y in zip(second.prices, whole.prices, strict=True):
        np.testing.assert_array_equal(x, y)
    # Only the cells the first call did not keep were read again.
    assert sum(r.fresh for r in second.rungs) == total - kept
    assert len(memo) == total


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


def test_a_deep_cell_is_read_child_by_child_on_the_wall_clock(roster, pool, monkeypatch) -> None:  # noqa: ANN001
    """IKA-375: on the wall clock a deep cell's turn and children's menus are read on one
    worker and each child on whichever worker is free (`SPLIT`), the cell's value folded by
    the reader. No value moves (hp-share answers a child alike in any batch): every stage,
    value, strategy, counted work and note is the serial read's. The positive controls:
    cells were split and their children sent (depth 3 and 4); the workers' ports held
    positions by number (`HOLD`)."""
    reg = roster.reg
    monkeypatch.setattr(ladder, "SPLIT", True)
    monkeypatch.setattr(ladder, "PASSES", False)
    monkeypatch.setattr(ladder, "HOLD", True)
    stages = "d2r2b3n4+d3r3ban4/r2ban4+d4r2b3n4/r2b3n4/r2b3n4"
    split = children = held = 0
    for pos in _played(roster)[:3]:
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
        split += walled.pool["splitCells"]
        children += walled.pool["childTasks"]
        held += walled.pool["heldPositions"]
    assert split > 0, "no deep cell was read child by child"
    assert children > split
    assert held > 0, "the workers' ports held no position by number"


@pytest.mark.parametrize("stages", ["d2r2b3n4+d3r3ban4/r2ban4+d4r2b3n4/r2b3n4/r2b3n4",
                                    "d2r2b3n4+d5r2b2n3/r2b2n3/r2b2n3/r2b2n3"])
def test_a_deep_cell_is_read_pass_by_pass_on_the_wall_clock(roster, pool, monkeypatch,  # noqa: ANN001
                                                            stages) -> None:  # noqa: ANN001
    """IKA-380 (`PASSES`): on the wall clock the reader walks a deep cell's tree -- the cell
    opened on a worker, each child read by its stage pass by pass, a deeper child's cells
    opened in turn -- and every stage, value, strategy, counted work and note is the serial
    read's (hp-share answers a cell alike in any batch). hp-share has no encoded road, so a
    depth-3 cell's children are read whole where it is opened (`_deep_open`'s own road); a
    depth-4 cell's children (read by a depth-3 stage, `_deep_children`'s road) are read here.
    The positive controls: cells opened, children read pass by pass. (The depth-2 passes
    in chunks are `tests/test_ladder_leaf.py`'s, on a learned leaf.) IKA-376: and depth 5,
    which the rule's ladder (L6) reaches, nests one more deep child."""
    reg = roster.reg
    monkeypatch.setattr(ladder, "SPLIT", True)
    monkeypatch.setattr(ladder, "PASSES", True)
    opened = kids = 0
    for pos in _played(roster)[:3]:
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
        opened += walled.pool["openCells"]
        kids += walled.pool["kidReads"]
    assert opened > 0, "no deep cell was opened"
    assert kids > 0, "no child was read pass by pass"


@pytest.mark.parametrize("how", ["count", "split", "passes"])
def test_the_workers_read_the_children_as_side_one_reads_them(roster, pool, monkeypatch,  # noqa: ANN001
                                                              how) -> None:  # noqa: ANN001
    """IKA-422 (`ladder.CHILD` ``seat``, the default): a read for side 1 -- its depth-3 and
    depth-4 cells' children read as side 1 -- by the workers is the serial read: on the count
    clock whole cells on a worker, on the wall clock a cell's children on whichever worker is
    free (`SPLIT`) and pass by pass by the reader (`PASSES`). The positive controls: side 1's
    serial read differs from the reference's (``guarantee``) at depth 3 or 4 for some
    position, and the wall clock's reader opened cells and read children pass by pass."""
    reg = roster.reg
    stages = "d2r2b3n4+d3r3ban4/r2ban4+d4r2b3n4/r2b3n4/r2b3n4"
    monkeypatch.setattr(ladder, "SPLIT", how != "count")
    monkeypatch.setattr(ladder, "PASSES", how == "passes")
    clock = {"clock": "wall"} if how != "count" else {}
    differed = opened = kids = 0
    for pos in _played(roster)[:3]:
        node = _node(reg, pos, 6, side=1)
        monkeypatch.setattr(ladder, "CHILD", "guarantee")
        reference = _read(reg, node, stages, side=1, workers=False)
        monkeypatch.setattr(ladder, "CHILD", "seat")
        serial = _read(reg, node, stages, side=1, workers=False)
        pooled = _read(reg, node, stages, side=1, workers=True, **clock)
        assert serial.stopped == pooled.stopped == "done"
        (_same if how == "count" else _same_but_time)(pooled, serial)
        differed += any(abs(x.value - y.value) > 1e-6 for x, y in
                        zip(serial.rungs, reference.rungs, strict=True) if x.stage[1] in "34")
        if how != "count":
            opened += pooled.pool["openCells"] if how == "passes" else pooled.pool["splitCells"]
            kids += pooled.pool["kidReads"] if how == "passes" else pooled.pool["childTasks"]
    assert differed >= 1, "side 1 read the children as the reference does"
    if how != "count":
        assert opened > 0 and kids > 0, "the reader opened no cell or read no child"


def test_the_readers_kid_lps_in_the_port_pass_by_pass(roster, pool, monkeypatch) -> None:  # noqa: ANN001
    """IKA-387 (`portlp` on): the reader walking deep cells pass by pass (`deep_passes`) solves
    the children whose pass is back in one crossing a round of answers, and every stage,
    value, strategy, counted work and note is the serial read's with scipy (hp-share answers
    a cell alike in any batch and order). The positive controls: children's rectangles
    solved in the reader's port, in no more crossings than rectangles (hp-share's two workers
    answer about one child a round; on 16 a round holds more)."""
    reg = roster.reg
    monkeypatch.setattr(ladder, "SPLIT", True)
    monkeypatch.setattr(ladder, "PASSES", True)
    stages = "d2r2b3n4+d3r3ban4/r2ban4+d4r2b3n4/r2b3n4/r2b3n4"
    solves = trips = 0
    try:
        for pos in _played(roster)[:3]:
            node = _node(reg, pos, 6)
            portlp.set_on(False)
            serial = _read(reg, node, stages, workers=False)
            portlp.set_on(True)
            walled = _read(reg, node, stages, workers=True, clock="wall")
            assert walled.stopped == serial.stopped == "done"
            _same_but_time(walled, serial)
            solves += walled.pool["kidSolves"]
            trips += walled.pool["kidSolveTrips"]
    finally:
        portlp.set_on(False)
    assert solves > 0, "no child's rectangle was solved in the reader's port"
    assert 0 < trips <= solves, (trips, solves)


def _same_but_time(a, b) -> None:  # noqa: ANN001
    assert [r.stage for r in a.rungs] == [r.stage for r in b.rungs]
    for x, y in zip(a.rungs, b.rungs, strict=True):
        assert x.value == y.value and x.fresh == y.fresh and x.work == y.work
        assert (x.rows, x.cols) == (y.rows, y.cols)
        np.testing.assert_array_equal(x.strategy, y.strategy)
    np.testing.assert_array_equal(a.strategy, b.strategy)
    assert a.work == b.work and a.unmodelled == b.unmodelled


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


@pytest.mark.parametrize("via", ["env", "ladder"])
def test_a_filled_budget_ends_at_the_budget(roster, pool, monkeypatch, via) -> None:  # noqa: ANN001
    """IKA-370: filling the wall clock's budget (`FILL_WALL`) begins every stage while time
    is left, so a read ends at its budget inside a stage it began (never before one it did
    not), and ends there without waiting for the chunks still out (they are taken, and
    thrown away, when the next read begins). The answer is its last completed stage.
    IKA-376: a ladder named in `FILLS` does the same with `FILL_WALL` off."""
    reg = roster.reg
    stages = "d2r2b3n4+d3r4ban4/r3ban4+d4r3ban4/r3ban4/r3ban4"
    if via == "env":
        monkeypatch.setattr(ladder, "FILL_WALL", True)
    else:
        monkeypatch.setattr(ladder, "FILL_WALL", False)
        monkeypatch.setitem(ladder.LADDERS, "T", tuple(stages.split("+")))
        monkeypatch.setattr(ladder, "FILLS", frozenset({"T"}))
        stages = "T"
        assert ladder.parse_ladder("T").fills
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


def _orphaned_pool(orphans: dict[int, list[int]], free: tuple[int, ...] = ()):  # noqa: ANN202
    """A `_Pool` on real pipes with no worker behind them (the test reads the workers' ends):
    a worker in ``orphans`` is inside chunks (its task ids) an earlier read gave up, as
    `_Pool._settle` leaves it; a worker in ``free`` has nothing out."""
    import multiprocessing

    ends = [multiprocessing.Pipe() for _ in range(len(orphans) + len(free))]
    mine = [a for a, _b in ends]
    theirs = [b for _a, b in ends]
    pool_ = ladder._Pool([], mine, multiprocessing.get_context().Event())  # noqa: SLF001
    pool_.stats = collections.defaultdict(float)
    pool_._last_end = time.perf_counter()  # noqa: SLF001
    for i, tasks in orphans.items():
        pool_._out[i] = list(tasks)  # noqa: SLF001
    pool_._orphans = set(orphans)  # noqa: SLF001
    pool_._orphan_tasks = {t for tasks in orphans.values() for t in tasks}  # noqa: SLF001
    return pool_, theirs


def test_no_chunk_goes_to_a_worker_still_in_an_earlier_reads_chunk() -> None:
    """A worker inside a chunk an earlier read gave up is sent the new read's context only when
    that chunk is back (`_Pool.begin`); a chunk sent to it before is read under the earlier
    read's rows, menus and sub-game table generation, and is answered as this read's.
    CI (818a902): `test_a_filled_budget_ends_at_the_budget` read 0.4216 for 0.4302 -- with
    every worker still in a deep chunk of the read before and each with one task left out, the
    send loop chose the least loaded worker by its count plus the orphan penalty but stopped on
    the bare count, and sent this read's chunk to a worker that had not its context.
    The positive control: a worker with nothing out is sent the chunk."""
    stage = SimpleNamespace(label="d2")
    # Both workers are in an orphan chunk, one task each left out (below POOL_DEPTH).
    pool_, theirs = _orphaned_pool({0: [5], 1: [6]})
    walk = pool_.cells([[("cell",)]], stage, None, ladder._zero(), set())  # noqa: SLF001
    assert next(walk) == (None, ladder._TICK)  # noqa: SLF001
    assert not any(end.poll(0) for end in theirs), "a chunk went to a worker not told this read"
    assert pool_._out == {0: [5], 1: [6]}  # noqa: SLF001
    walk.close()
    # The positive control: one worker is free, the other is still in its orphan chunk.
    pool_, theirs = _orphaned_pool({0: [5]}, free=(1,))
    walk = pool_.cells([[("cell",)]], stage, None, ladder._zero(), set())  # noqa: SLF001
    assert next(walk) == (None, ladder._TICK)  # noqa: SLF001
    assert [end.poll(0) for end in theirs] == [False, True], "the free worker got no chunk"
    walk.close()
