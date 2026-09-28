"""The ladder with a learned leaf (IKA-367): its depth-3 children read at once.

A depth-3 stage reads each cell's child games -- their matrices and their own depth-2
stage -- together (`ladder.BATCH_CHILDREN`), and scores what it gathers in one forward pass
(`ladder.STACK`, `port.score_stacked`). What these hold, on an untrained net (seed 7):

- **deterministic**: the same position read twice on the node clock is the same mixture to
  the bit, and so is a read with the port on another number of threads;
- **gathered, not changed**: with the passes kept one a child game (`STACK` off), reading
  the children at once is the one-at-a-time road to the bit -- mixture, value, stages and
  counted work -- and the new road was taken (the positive control);
- **stacked, nearly the same**: one pass for everything moves the values only in the last
  places (the leaf's answer for a row moves with its batch), and the road was taken.

hp-share has no encoded road, so `tests/test_ladder.py` never reaches these; torch is
needed, and the suite job skips this module for it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pokeuraou import humanplay, ladder, rustnode  # noqa: E402
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.selfplay import position_from_sets  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402
from pokeuraou.value import BatchedValue, ValueConfig, build  # noqa: E402

STAGES = "d2r2b2n4x+d3r2b2n4x/r2b2n3"


@pytest.fixture(scope="module")
def kit():  # noqa: ANN201
    roster = load_roster("rizabanadohido")
    register_mega_stones(roster.reg)
    encoder = Encoder(roster.reg)
    torch.set_num_threads(1)  # many small passes: threads only contend (and xdist runs several)
    torch.manual_seed(7)
    leaf = BatchedValue(build(encoder, ValueConfig()).eval(), encoder, device=torch.device("cpu"))
    pos = position_from_sets(roster.reg, list(roster.sets[:4]), list(roster.sets[2:6]))
    return roster.reg, pos, leaf


def _read(kit, monkeypatch, *, batch: bool, stack: bool):  # noqa: ANN001, ANN202
    reg, pos, leaf = kit
    monkeypatch.setattr(ladder, "BATCH_CHILDREN", batch)
    monkeypatch.setattr(ladder, "STACK", stack)
    ours = narrow(reg, pos, 0, limit=5).actions
    theirs = narrow(reg, pos, 1, limit=5).actions
    return humanplay.solve_move(
        reg, pos, 0, ours, theirs, None, leaf, budget=Budget.matrix(), exact=True,
        ladder={"stages": ladder.parse_ladder(STAGES), "budget_ms": None})


def _same(a, b) -> bool:  # noqa: ANN001
    return (np.asarray(a.strategy).tobytes() == np.asarray(b.strategy).tobytes()
            and a.value == b.value
            and [(r.stage, r.spent_ms) for r in a.ladder.rungs]
            == [(r.stage, r.spent_ms) for r in b.ladder.rungs]
            and a.ladder.work == b.ladder.work)


def _counting(monkeypatch) -> dict:  # noqa: ANN001
    seen = {"batched": 0, "alone": 0}
    inner = ladder._children_at_once

    def counted(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        got = inner(*args, **kwargs)
        seen["alone" if got is ladder._ALONE else "batched"] += 1
        return got

    monkeypatch.setattr(ladder, "_children_at_once", counted)
    return seen


def test_a_stacked_read_is_deterministic(kit, monkeypatch) -> None:  # noqa: ANN001
    seen = _counting(monkeypatch)
    first = _read(kit, monkeypatch, batch=True, stack=True)
    again = _read(kit, monkeypatch, batch=True, stack=True)
    assert [r.stage for r in first.ladder.rungs] == STAGES.split("+")
    assert seen["batched"] > 0
    assert _same(first, again)
    before = rustnode.port_threads() or 1
    try:
        rustnode.set_port_threads(4 if before == 1 else 1)
        other = _read(kit, monkeypatch, batch=True, stack=True)
    finally:
        rustnode.set_port_threads(before)
    assert _same(first, other)


def test_children_at_once_are_the_one_at_a_time_road(kit, monkeypatch) -> None:  # noqa: ANN001
    seen = _counting(monkeypatch)
    alone = _read(kit, monkeypatch, batch=False, stack=False)
    assert seen["batched"] == 0
    gathered = _read(kit, monkeypatch, batch=True, stack=False)
    assert seen["batched"] > 0, "no cell took the new road"
    assert _same(alone, gathered)


def test_a_child_read_once_is_every_one_read(kit, monkeypatch) -> None:  # noqa: ANN001
    """IKA-378 (`ladder.SHARE`): with a pass a child game (`STACK` off, so a block's values do
    not depend on what else is scored), a read that takes a deep cell's child and a sub-game
    it has read before is the read that reads every one: mixture, value, stages and counted
    work to the bit. The positive control: children taken (`sharedKids`) and sub-games."""
    reg, pos, leaf = kit
    ours = narrow(reg, pos, 0, limit=5).actions
    theirs = narrow(reg, pos, 1, limit=5).actions
    stages = "d2r2b2n4x+d3r4ban4x/r2b2n3"
    got = {}
    for share in (False, True):
        monkeypatch.setattr(ladder, "SHARE", share)
        monkeypatch.setattr(ladder, "STACK", False)
        got[share] = humanplay.solve_move(
            reg, pos, 0, ours, theirs, None, leaf, budget=Budget.matrix(), exact=True,
            ladder={"stages": ladder.parse_ladder(stages), "budget_ms": None})
    assert _same(got[False], got[True])
    assert got[False].ladder.unmodelled == got[True].ladder.unmodelled
    assert got[False].ladder.here["sharedKids"] == 0
    assert got[True].ladder.here["sharedKids"] > 0
    assert got[True].ladder.here["sharedSubgames"] > 0


def stackless_leaf(reg, path):  # noqa: ANN001, ANN201
    """A ladder worker's leaf for `test_a_child_read_pass_by_pass_is_the_child_read_whole`:
    the saved net on the CPU at one thread, a forward pass a child game (`ladder.STACK` off
    in the worker too), so a value never depends on what it is scored with."""
    from pokeuraou import humanplay

    ladder.STACK = False
    torch.set_num_threads(1)
    return humanplay.process_leaf(reg, [path], "cpu", False)


def test_a_child_read_pass_by_pass_is_the_child_read_whole(kit, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """IKA-380 (`ladder.PASSES`): on the wall clock with worker processes, a depth-3 cell's
    children read pass by pass -- each pass's depth-2 cells in chunks on the workers, its
    rectangle solved by the reader, a child two cells reach read once -- is the read that
    reads each cell whole on one worker (`PASSES` off, IKA-375's split) and the read here
    without workers: mixture, value, stages, counted work and notes to the bit (a pass a
    child game, `STACK` off, here and in the workers). The positive controls: chunks of
    passes were sent and children read by the reader, and children joined."""
    from pokeuraou.value import ValueConfig as _Config, save_model

    reg, pos, leaf = kit
    encoder = Encoder(reg)
    path = tmp_path / "seed7.pt"
    net = leaf.nets[0] if hasattr(leaf, "nets") else leaf.net
    save_model(path, net, net.state_dict(), encoder.vocab, _Config(), meta={})
    ours = narrow(reg, pos, 0, limit=5).actions
    theirs = narrow(reg, pos, 1, limit=5).actions
    stages = "d2r2b2n4x+d3r4ban4x/r2b2n3"
    monkeypatch.setattr(ladder, "STACK", False)
    monkeypatch.setattr(ladder, "SPLIT", True)

    def solved(passes: bool, workers: bool):  # noqa: ANN202
        monkeypatch.setattr(ladder, "PASSES", passes)
        saved = ladder._POOL
        if not workers:
            ladder._POOL = None
        try:
            return humanplay.solve_move(
                reg, pos, 0, ours, theirs, None, leaf, budget=Budget.matrix(), exact=True,
                ladder={"stages": ladder.parse_ladder(stages), "budget_ms": 600_000.0,
                        "clock": "wall"})
        finally:
            ladder._POOL = saved

    assert ladder.start_pool(reg, 2, stackless_leaf, (str(path),)) == 2
    try:
        passes = solved(True, True)
        split = solved(False, True)
    finally:
        ladder.stop_pool()
    alone = solved(True, False)
    assert [r.stage for r in passes.ladder.rungs] == stages.split("+")
    for other in (split, alone):
        # The wall clock's milliseconds aside.
        assert np.asarray(passes.strategy).tobytes() == np.asarray(other.strategy).tobytes()
        assert passes.value == other.value and passes.ladder.work == other.ladder.work
        assert ([(r.stage, r.value, r.work, r.fresh) for r in passes.ladder.rungs]
                == [(r.stage, r.value, r.work, r.fresh) for r in other.ladder.rungs])
        assert passes.ladder.unmodelled == other.ladder.unmodelled
    assert passes.ladder.pool["passChunks"] > 0 and passes.ladder.pool["kidReads"] > 0
    assert passes.ladder.pool["kidsJoined"] + passes.ladder.pool["sharedKids"] > 0
    assert split.ladder.pool["passChunks"] == 0


def test_one_pass_moves_the_values_only_in_the_last_places(kit, monkeypatch) -> None:  # noqa: ANN001
    per_game = _read(kit, monkeypatch, batch=True, stack=False)
    seen = _counting(monkeypatch)
    stacked = _read(kit, monkeypatch, batch=True, stack=True)
    assert seen["batched"] > 0
    assert [r.stage for r in stacked.ladder.rungs] == [r.stage for r in per_game.ladder.rungs]
    assert abs(stacked.value - per_game.value) < 1e-5
    for a, b in zip(stacked.ladder.rungs, per_game.ladder.rungs, strict=True):
        assert abs(a.value - b.value) < 1e-5
