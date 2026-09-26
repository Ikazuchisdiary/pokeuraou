"""IKA-32 stage 2: a move spread over more cores is the same move.

Two things spread a person's-clock move over more cores, and neither may change it:

- **the deepening's cells expanded ahead** (`deepen.set_ahead`): a helper thread expands
  the cells the loop is likely to take next, in batches (one crossing each for the turns,
  the branches' menus and the child matrices, one call into the leaf for their blocks),
  and the loop takes its cell's expansion from it. The loop picks its cells exactly as
  before, so the tree, the answer, the notes and the report are the serial deepening's to
  the bit -- with a leaf whose answer moves with the size of its call, so a block scored
  in a call of another size would show; when the helper guessed wrong (its guesses
  reversed here: every cell the loop takes is one it was not expanding); and when a batch
  fails, so each cell is expanded the serial way;
- **a big game's two LPs at once** (`equilibrium.set_lp_pair`): the row player's and the
  column player's LPs are independent, and each is the same model handed to the same
  HiGHS on a second thread, so the same vertex.

Each comparison carries its positive control: the helper's counts (cells taken from it,
cells it expanded that the loop never took, cells expanded the serial way) and the LP
pairs solved at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import equilibrium as eq_mod
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.search import belief_solve, search
from pokeuraou.teams import load_roster

from .test_hidden_depth2 import LEAF, SEEN_ALL, _played
from .test_subgame_batch import _SizedLeaf


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture(autouse=True)
def _serial_after():  # noqa: ANN202
    yield
    deepen_mod.set_ahead(0)
    eq_mod.set_lp_pair(0)


def _leaf(roster, kind: str):  # noqa: ANN001, ANN202
    if kind == "hp-share":
        return LEAF
    return _SizedLeaf(Encoder(roster.reg)).__call__


def _hidden(roster, pos):  # noqa: ANN001, ANN202
    sheet = list(roster.sets)[:6]
    return {side: completions(roster.reg, pos, side, sheet) for side in (0, 1)}


def _menus(reg, pos, limit=5):  # noqa: ANN001, ANN202
    return narrow(reg, pos, 0, limit=limit).actions, narrow(reg, pos, 1, limit=limit).actions


def _belief(roster, pos, side, leaf, cells, ahead):  # noqa: ANN001, ANN202
    reg = roster.reg
    ours, theirs = _menus(reg, pos)
    deepen_mod.set_ahead(ahead)
    trace: list = []
    got = belief_solve(
        reg, pos, ours, theirs, _hidden(roster, pos), {0: leaf, 1: leaf},
        budget=Budget.matrix(), sides=(side,),
        deepen={side: {"cells": cells, "trace": trace}},
    )[side]
    return got, trace


def _steps(trace: list) -> list[tuple]:
    """The loop's steps as (level, cell, took): the cells it chose, in order."""
    return [(node.level, tuple(cell), took) for node, cell, took in trace[1:]]


def _same_belief(a, b) -> None:  # noqa: ANN001
    assert np.array_equal(a.strategy, b.strategy)
    assert a.value == b.value
    assert len(a.replies) == len(b.replies)
    for x, y in zip(a.replies, b.replies, strict=True):
        assert np.array_equal(x, y)
    assert [x.to_choice() for x in a.ours] == [x.to_choice() for x in b.ours]
    assert [x.to_choice() for x in a.theirs] == [x.to_choice() for x in b.theirs]
    assert a.unmodelled == b.unmodelled
    assert a.deepened == b.deepened


def _same_tree(serial: list, fast: list) -> None:
    assert _steps(serial) == _steps(fast)
    a, b = serial[0], fast[0]
    for x, y in zip(a.prices, b.prices, strict=True):
        assert np.array_equal(x, y)
    # Every child node the steps opened holds the same matrix and the same answer.
    for (na, cell, took), (nb, _cell, _took) in zip(serial[1:], fast[1:], strict=True):
        if not took:
            continue
        for (wa, ca), (wb, cb) in zip(na.children[cell], nb.children[cell], strict=True):
            assert wa == wb
            if isinstance(ca, deepen_mod._Node):
                assert np.array_equal(ca.payoff, cb.payoff)
                assert ca.value == cb.value
                assert [r.to_choice() for r in ca.rows] == [r.to_choice() for r in cb.rows]
            else:
                assert ca == cb


def _counts() -> dict[str, int]:
    return deepen_mod.ahead_counts()


def _delta(before: dict[str, int]) -> dict[str, int]:
    now = _counts()
    return {k: now[k] - before[k] for k in now}


@pytest.mark.parametrize("kind", ["hp-share", "sized"])
def test_cells_expanded_ahead_are_the_serial_deepening_to_the_bit(roster, kind) -> None:  # noqa: ANN001
    reg = roster.reg
    leaf = _leaf(roster, kind)
    before = _counts()
    steps = 0
    deeper = 0
    for pos in _played(roster)[:2]:
        for side in (0, 1):
            serial, trace_s = _belief(roster, pos, side, leaf, 400, 0)
            fast, trace_f = _belief(roster, pos, side, leaf, 400, 4)
            _same_belief(serial, fast)
            _same_tree(trace_s, trace_f)
            steps += len(trace_s) - 1
            deeper += sum(1 for level, _c, took in _steps(trace_s) if took and level > 0)
    moved = _delta(before)
    # Positive control: every step's cell came from the helper, and it expanded cells
    # the loop never took (its guesses were not all right, and it did not matter).
    assert steps > 20 and deeper > 0, "too little deepening to compare"
    assert moved["hits"] == steps and moved["misses"] == 0
    assert moved["expanded"] > moved["hits"]
    assert moved["batches"] > 0
    del reg


def test_a_wrong_guess_takes_the_same_cells(roster, monkeypatch) -> None:  # noqa: ANN001
    """The helper's guesses reversed -- the worst cells first -- so the cell the loop
    takes is almost never one it expanded ahead: the loop asks for it and waits, and the
    answer is the same."""
    leaf = _leaf(roster, "sized")
    real = deepen_mod._ranked

    def worst(root, count):  # noqa: ANN001, ANN202
        return list(reversed(real(root, 10 * count)))[:count]

    pos = _played(roster)[1]
    serial, trace_s = _belief(roster, pos, 0, leaf, 1000, 0)
    monkeypatch.setattr(deepen_mod, "_ranked", worst)
    before = _counts()
    fast, trace_f = _belief(roster, pos, 0, leaf, 1000, 3)
    moved = _delta(before)
    _same_belief(serial, fast)
    _same_tree(trace_s, trace_f)
    steps = len(trace_s) - 1
    assert steps > 10
    assert moved["hits"] == steps
    # Most of what it expanded was never taken.
    assert moved["expanded"] - moved["hits"] > steps // 2


def test_a_failed_batch_is_expanded_the_serial_way(roster, monkeypatch) -> None:  # noqa: ANN001
    leaf = _leaf(roster, "sized")
    pos = _played(roster)[0]
    serial, trace_s = _belief(roster, pos, 1, leaf, 200, 0)

    def broken(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("a batch that fails")

    monkeypatch.setattr(deepen_mod, "_expand_many", broken)
    before = _counts()
    fast, trace_f = _belief(roster, pos, 1, leaf, 200, 4)
    moved = _delta(before)
    _same_belief(serial, fast)
    _same_tree(trace_s, trace_f)
    assert moved["misses"] == len(trace_s) - 1 > 0 and moved["hits"] == 0


def test_a_child_whose_lp_fails_stops_the_step_where_the_serial_one_does(roster, monkeypatch) -> None:  # noqa: ANN001
    """A child's LP that fails (here: every child matrix whose first cell falls in a third
    of the unit interval, the same ones for both loops) refuses the cell after that
    child's fill, as `_expand` does; the helper solved it ahead and says so."""
    leaf = _leaf(roster, "sized")
    real = deepen_mod.solve
    failed = [0]

    def picky(payoff):  # noqa: ANN001, ANN202
        a = np.asarray(payoff)
        if float(a.flat[0]) * 1e6 % 3.0 < 1.0:
            failed[0] += 1
            raise deepen_mod.EquilibriumError("a child LP that fails")
        return real(payoff)

    monkeypatch.setattr(deepen_mod, "solve", picky)
    pos = _played(roster)[1]
    serial, trace_s = _belief(roster, pos, 0, leaf, 600, 0)
    before = _counts()
    fast, trace_f = _belief(roster, pos, 0, leaf, 600, 4)
    moved = _delta(before)
    _same_belief(serial, fast)
    _same_tree(trace_s, trace_f)
    refused = sum(1 for _level, _cell, took in _steps(trace_s) if not took)
    assert failed[0] > 0 and refused > 0
    assert moved["hits"] == len(trace_s) - 1


def test_an_open_root_expanded_ahead_is_the_same_search(roster) -> None:  # noqa: ANN001
    """Both benches seen: `search`'s deepening (`deepen_root`), the same with the helper."""
    reg = roster.reg
    leaf = _leaf(roster, "sized")
    before = _counts()
    checked = 0
    for pos in _played(roster)[:2]:
        ours, theirs = _menus(reg, pos)
        answers = []
        for ahead in (0, 4):
            deepen_mod.set_ahead(ahead)
            answers.append(search(reg, pos, ours, theirs, leaf, budget=Budget.matrix(), deepen=300))
        a, b = answers
        assert np.array_equal(a.payoff, b.payoff)
        assert np.array_equal(a.equilibrium.row_strategy, b.equilibrium.row_strategy)
        assert np.array_equal(a.equilibrium.col_strategy, b.equilibrium.col_strategy)
        assert a.equilibrium.value == b.equilibrium.value
        assert a.unmodelled == b.unmodelled and a.deepened == b.deepened
        checked += a.deepened.expanded
    assert checked > 10
    assert _delta(before)["hits"] > 0


def test_the_two_lps_at_once_are_the_same_lps() -> None:
    rng = np.random.default_rng(32)
    games = [rng.random((int(rng.integers(3, 30)), int(rng.integers(3, 30)))) for _ in range(30)]
    bayes = []
    for _ in range(12):
        k = int(rng.integers(2, 5))
        m = int(rng.integers(4, 20))
        bayes.append(([rng.random((m, int(rng.integers(4, 20)))) for _ in range(k)], rng.random(k) + 0.1))
    eq_mod.set_lp_pair(0)
    serial = [eq_mod.solve(g) for g in games]
    serial_b = [eq_mod.solve_bayesian(mats, w) for mats, w in bayes]
    eq_mod.set_lp_pair(1)
    pairs = eq_mod.LP_PAIRS[0]
    fast = [eq_mod.solve(g) for g in games]
    fast_b = [eq_mod.solve_bayesian(mats, w) for mats, w in bayes]
    assert eq_mod.LP_PAIRS[0] - pairs == len(games) + len(bayes)
    for a, b in zip(serial, fast, strict=True):
        assert a.value == b.value and a.duality_gap == b.duality_gap
        assert np.array_equal(a.row_strategy, b.row_strategy)
        assert np.array_equal(a.col_strategy, b.col_strategy)
    for a, b in zip(serial_b, fast_b, strict=True):
        assert a.value == b.value and a.duality_gap == b.duality_gap
        assert np.array_equal(a.row_strategy, b.row_strategy)
        for x, y in zip(a.col_strategies, b.col_strategies, strict=True):
            assert np.array_equal(x, y)
    # Under the threshold a game stays on one thread.
    eq_mod.set_lp_pair(10_000)
    pairs = eq_mod.LP_PAIRS[0]
    eq_mod.solve(games[0])
    assert eq_mod.LP_PAIRS[0] == pairs


def test_the_error_of_two_lps_at_once_is_the_serial_orders() -> None:
    eq_mod.set_lp_pair(1)
    ran = []

    def fails(name):  # noqa: ANN001, ANN202
        def go():  # noqa: ANN202
            ran.append(name)
            raise eq_mod.EquilibriumError(name)
        return go

    with pytest.raises(eq_mod.EquilibriumError, match="first"):
        eq_mod._both(10, fails("first"), fails("second"))
    assert sorted(ran) == ["first", "second"]  # the second was waited for
    with pytest.raises(eq_mod.EquilibriumError, match="second"):
        eq_mod._both(10, lambda: 1, fails("second"))
    assert eq_mod._both(10, lambda: 1, lambda: 2) == (1, 2)


def test_seen_benches_are_not_hidden_here(roster) -> None:  # noqa: ANN001
    """The hidden tests above did deepen a Bayesian root (several completions)."""
    pos = _played(roster)[0]
    spreads = _hidden(roster, pos)
    assert len(spreads[0]) > 1 and len(spreads[1]) > 1
    seen = {side: completions(roster.reg, pos, side, list(roster.sets)[:6], seen=SEEN_ALL) for side in (0, 1)}
    assert len(seen[0]) == 1
