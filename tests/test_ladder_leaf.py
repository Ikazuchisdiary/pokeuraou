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


def test_one_pass_moves_the_values_only_in_the_last_places(kit, monkeypatch) -> None:  # noqa: ANN001
    per_game = _read(kit, monkeypatch, batch=True, stack=False)
    seen = _counting(monkeypatch)
    stacked = _read(kit, monkeypatch, batch=True, stack=True)
    assert seen["batched"] > 0
    assert [r.stage for r in stacked.ladder.rungs] == [r.stage for r in per_game.ladder.rungs]
    assert abs(stacked.value - per_game.value) < 1e-5
    for a, b in zip(stacked.ladder.rungs, per_game.ladder.rungs, strict=True):
        assert abs(a.value - b.value) < 1e-5
