"""IKA-307: the deepening's depth guard (``g<L>``) and its children's menus by a Q (``c<k>``).

What these tests hold:

- **the labels**: ``g<L>`` and ``c<k>`` read after ``h``; every earlier label has neither;
  breadth only (``b``) refines nothing, so it takes neither;
- **the guard**: no cell is refined at or past it, raising it is the same deepening until
  it is met, and it is `MAX_LEVELS` when not given -- the same deepening, only the record
  says nothing of the stops then;
- **why the lines stopped**: every branch the tree ends in is counted once, as a finished
  game, a node with nothing worth a step, a node at the guard, or a node the budget
  stopped first; the steps the guard changed are counted; the deepening's own stop is the
  budget or nothing left;
- **the children's menus by a Q**: each side's menu in every child is the k best of its
  whole pool by the Q solved over both pools, without the cover, and a refined cell's
  children ask the Q in one pass (``batched``); served, that is one round trip (op
  ``q_batch``); with torch, the batched pass is each request's own pass to rounding;
- **in a game**: ``g8`` plays the game without it, with the stops in the record; ``c<k>``
  asks the Q and says so; either repeats.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import qhead, qrank, selfplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow
from pokeuraou.port import batched_payoff
from pokeuraou.regulation import load_regulation
from pokeuraou.search import DEFAULT_SUB_BRANCHES, DEFAULT_SUB_LIMIT
from pokeuraou.teams import load_roster

from ._port import Budget
from .test_q_rank import _ServedStub, _stub_matrix, _StubQ, _untrained_q
from .test_search import LEAF, _played


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


class _BatchedStubQ(_StubQ):
    """`_StubQ` with `batched`: one pass (counted) for several positions."""

    def __init__(self, encoder: Encoder) -> None:
        super().__init__(encoder)
        self.passes = 0
        self.ranked = 0

    def batched(self, reg, asks):  # noqa: ANN001, ANN201
        self.passes += 1
        self.ranked += len(asks)
        return [
            _stub_matrix(qrank._pool_arrays(reg, self.encoder, pos, pools, True))  # noqa: SLF001
            for pos, pools in asks
        ]


@pytest.fixture()
def stub_q(roster):  # noqa: ANN001, ANN201
    model = _BatchedStubQ(Encoder(roster.reg))
    qrank.install(model)
    yield model
    qrank._INSTALLED.clear()  # noqa: SLF001


def _menus(reg, pos, limit):  # noqa: ANN001, ANN202
    return narrow(reg, pos, 0, limit=limit).actions, narrow(reg, pos, 1, limit=limit).actions


def _deepen(reg, pos, ours, theirs, cells, **kw):  # noqa: ANN001, ANN202, ANN003
    payoff, notes = batched_payoff(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
    trace: list = []
    got = deepen_mod.deepen_root(
        reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), payoff=payoff,
        equilibrium=solve(payoff), cells=cells, sub_limit=DEFAULT_SUB_LIMIT,
        sub_branches=DEFAULT_SUB_BRANCHES, unmodelled=set(notes), trace=trace, **kw,
    )
    return got, trace


def _line_ends(root) -> list:  # noqa: ANN001
    """Every branch the tree ends in: a finished game's value, or a node with no refined cell."""
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        for cell in sorted(node.children):
            for _w, child in node.children[cell]:
                if not isinstance(child, deepen_mod._Node) or not child.children:
                    out.append(child)
                else:
                    queue.append(child)
    return out


def test_the_labels_read_the_guard_and_the_childrens_q() -> None:
    spec = deepen_mod.deepen_spec("m1200sallhc6g16")
    assert (spec.cells, spec.oracle, spec.swap, spec.hidden) == (1200, deepen_mod.ALL_ACTIONS, True, True)
    assert (spec.child_q, spec.levels) == (6, 16)
    assert deepen_mod.deepen_spec("m800c4").child_q == 4
    assert deepen_mod.deepen_spec("m800c4").levels is None
    assert deepen_mod.deepen_spec("r40g3").levels == 3
    for old in ("none", "m400", "r25", "m400o24", "m400sallh", "m1200sq3h", "b200s24"):
        spec = deepen_mod.deepen_spec(old)
        assert spec.child_q is None and spec.levels is None
    for bad in ("m100g0", "m100c0", "m100g8c6", "m100hg", "b50sallc6", "b50sallhg12", "noneg8"):
        with pytest.raises(ValueError, match="deepen"):
            deepen_mod.deepen_spec(bad)


def test_the_guard_bounds_the_tree_and_says_why_each_line_stopped(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    at_guard = guarded = 0
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos, 3)
        for guard in (1, 2):
            got, trace = _deepen(reg, pos, ours, theirs, 1500, levels=guard)
            report = got.report
            levels = [node.level for node, _c, took in trace[1:] if took]
            assert all(level < guard for level in levels)
            assert report.depth <= guard + 1
            ends = _line_ends(trace[0])
            assert sum(report.lines) == len(ends)
            assert report.lines[0] == sum(not isinstance(e, deepen_mod._Node) for e in ends)
            # A line past the guard is at it, and is counted there only if a cell is still
            # worth a step.
            for end in ends:
                if isinstance(end, deepen_mod._Node):
                    assert end.level <= guard
            assert report.stop in ("budget", "exhausted")
            if report.stop == "exhausted":
                assert report.lines[3] == 0 and report.cells < 1500
            at_guard += report.lines[2]
            guarded += report.guarded
            js = report.to_json()
            assert js["levels"] == guard and js["stop"] == report.stop
            assert list(js["lines"]) == list(deepen_mod.LINE_STOPS)
            assert js["guarded"] == report.guarded
    # Positive control: the guard was met with cells still worth a step, and changed steps.
    assert at_guard > 0 and guarded > 0, (at_guard, guarded)


def test_without_g_the_guard_is_max_levels_and_the_record_is_unchanged(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    for pos in _played(roster)[:2]:
        ours, theirs = _menus(reg, pos, 4)
        plain, _ = _deepen(reg, pos, ours, theirs, 400)
        told, _ = _deepen(reg, pos, ours, theirs, 400, levels=deepen_mod.MAX_LEVELS)
        high, _ = _deepen(reg, pos, ours, theirs, 400, levels=64)
        for other in (told, high):
            assert np.array_equal(plain.payoff, other.payoff)
            assert np.array_equal(plain.equilibrium.row_strategy, other.equilibrium.row_strategy)
        # The report without g says nothing new; with it, the same plus the stops.
        js, js_told = plain.report.to_json(), told.report.to_json()
        assert not {"levels", "stop", "lines", "guarded"} & set(js)
        assert {k: v for k, v in js_told.items() if k not in ("levels", "stop", "lines", "guarded")} == js
        # Counted either way (for the timers), written only with g.
        assert plain.report.lines == told.report.lines and plain.report.stop == told.report.stop


def test_a_small_budget_stops_on_the_budget(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    pos = _played(roster)[0]
    ours, theirs = _menus(reg, pos, 6)
    got, _ = _deepen(reg, pos, ours, theirs, 30, levels=8)
    assert got.report.stop == "budget" and got.report.cells >= 30
    zero, _ = _deepen(reg, pos, ours, theirs, 0, levels=8)
    assert zero.report.stop == "" and zero.report.lines == (0, 0, 0, 0)


def _q_menu(reg, pos, side, width):  # noqa: ANN001, ANN202
    """What a child's menu should be: the k best by the Q's solve over both pools, uncovered."""
    pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
    m = _stub_matrix(qrank._pool_arrays(reg, Encoder(reg), pos, pools, True))  # noqa: SLF001
    e = solve(m)
    score = m @ e.col_strategy if side == 0 else -(e.row_strategy @ m)
    return narrow(reg, pos, side, limit=width, cover=False,
                  rank=lambda _p, _s=None: score).actions


def test_the_childrens_menus_are_the_qs_best_asked_in_one_pass_a_step(roster, stub_q) -> None:  # noqa: ANN001
    reg = roster.reg
    checked = differs = 0
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos, 4)
        passes, ranked = stub_q.passes, stub_q.ranked
        got, trace = _deepen(reg, pos, ours, theirs, 300, child_q=3)
        report = got.report
        steps = [(node, cell) for node, cell, took in trace[1:] if cell is not None]
        children = [
            child
            for node, cell in steps if cell in node.children
            for _w, child in node.children[cell] if isinstance(child, deepen_mod._Node)
        ]
        assert children, "nothing was refined"
        for child in children:
            for side, menu in ((0, child.rows), (1, child.cols)):
                want = _q_menu(reg, child.pos, side, 3)
                assert [a.to_choice() for a in menu] == [a.to_choice() for a in want]
                damage = narrow(reg, child.pos, side, limit=3).actions
                differs += [a.to_choice() for a in damage] != [a.to_choice() for a in want]
                checked += 1
        # One pass a refined cell with a child to rank, over all its children at once.
        assert stub_q.passes - passes == report.child_passes <= len(steps)
        assert stub_q.ranked - ranked == report.child_ranked >= len(children)
        assert report.child_q and {"childQ", "childRanked"} <= set(report.to_json())
    # Positive control: the Q's menus are not the damage ranking's.
    assert checked and differs, (checked, differs)


def test_the_childrens_q_needs_a_q(roster) -> None:  # noqa: ANN001
    qrank._INSTALLED.clear()  # noqa: SLF001
    reg = roster.reg
    pos = _played(roster)[0]
    ours, theirs = _menus(reg, pos, 4)
    with pytest.raises(RuntimeError, match="needs a Q"):
        _deepen(reg, pos, ours, theirs, 300, child_q=3)


def test_a_served_batch_is_one_round_trip(roster) -> None:  # noqa: ANN001
    from pokeuraou.inference import serve

    reg = roster.reg
    encoder = Encoder(reg)
    served = _ServedStub(encoder.vocab.fingerprint())
    batches: list[int] = []

    def batch(requests):  # noqa: ANN001, ANN202
        batches.append(len(requests))
        return [_stub_matrix(r) for r in requests]

    served.batch = batch
    server, address = serve({}, q_models={"q": served})
    try:
        remote = qrank.RemoteQ(address, "q", encoder)
        try:
            positions = _played(roster)[:3]
            asks = [(p, (qhead.legal_pool(reg, p, 0), qhead.legal_pool(reg, p, 1))) for p in positions]
            got = remote.batched(reg, asks)
            want = [_stub_matrix(qrank._pool_arrays(reg, encoder, p, pools, True)) for p, pools in asks]  # noqa: SLF001
            assert all(np.array_equal(g, w) for g, w in zip(got, want, strict=True))
            assert batches == [3] and served.calls == 0 and remote.trips == 1
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_the_batched_pass_is_each_requests_own_pass(roster, tmp_path) -> None:  # noqa: ANN001
    pytest.importorskip("torch")
    reg = roster.reg
    encoder = Encoder(reg)
    local = qrank.LocalQ(_untrained_q(reg, encoder, tmp_path / "q.pt"), encoder)
    positions = _played(roster)[:4]
    asks = [(p, (qhead.legal_pool(reg, p, 0), qhead.legal_pool(reg, p, 1))) for p in positions]
    sizes = {(len(a), len(b)) for _p, (a, b) in asks}
    assert len(sizes) > 1, "the pools should differ in size, so the batch pads"
    one = [local.matrix(reg, p, pools) for p, pools in asks]
    batch = local.batched(reg, asks)
    again = local.batched(reg, asks)
    for a, b, c in zip(one, batch, again, strict=True):
        assert a.shape == b.shape
        np.testing.assert_allclose(b, a, atol=1e-6)
        assert np.array_equal(b, c)


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    return reg, sheet


def _hidden_game(setup, label):  # noqa: ANN001, ANN202
    reg, sheet = setup
    return selfplay.play_game(
        reg, np.random.default_rng(33), sheet[:4], sheet[2:6], "test",
        search_limit=3, max_turns=12, sheets=(sheet, sheet), deepen=label,
    )


def _payload(record) -> dict:  # noqa: ANN001
    out = record.to_json(objective="hp-share", search_limit=3)
    out.pop("searchSeconds", None)
    out.pop("engine", None)
    return out


def test_in_a_game_g8_is_the_game_and_c_asks_the_q(setup) -> None:  # noqa: ANN001
    reg, _sheet = setup
    plain = _payload(_hidden_game(setup, "m60h"))
    told = _payload(_hidden_game(setup, "m60hg8"))
    stops = 0
    for decision in told["decisions"]:
        for side in decision.get("deepened") or []:
            if side is not None:
                assert side["levels"] == 8 and side["stop"] in ("budget", "exhausted")
                stops += 1
                for key in ("levels", "stop", "lines", "guarded"):
                    side.pop(key)
    assert stops > 0
    assert told["deepen"] == ["m60hg8", "m60hg8"]
    told["deepen"] = plain["deepen"]
    assert told == plain

    model = _BatchedStubQ(Encoder(reg))
    qrank.install(model)
    try:
        first = _payload(_hidden_game(setup, "m60hc3"))
        second = _payload(_hidden_game(setup, "m60hc3"))
    finally:
        qrank._INSTALLED.clear()  # noqa: SLF001
    assert first == second
    asked = sum(
        side["childQ"]
        for decision in first["decisions"]
        for side in decision.get("deepened") or []
        if side is not None
    )
    # Two games' passes; a node with no bench hidden is one tree for both sides, so its
    # report is written on each side and counted twice here.
    assert asked > 0 and asked <= model.passes <= 2 * asked
    assert first["decisions"] != plain["decisions"]
