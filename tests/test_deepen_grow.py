"""IKA-354: widening the root mid-read, the tree kept (`deepen.Grow`).

What these tests hold:

- **a grow before the first step is the wide menu from the start**: the narrow root
  grown to the wide menu at once, then deepened, is the wide root deepened -- the same
  prices, strategies and tree to the bit, on an open root and on a Bayesian one, in
  either side's orientation (the grow's fill charged, so the budget is the wide read's
  plus it);
- **a grow midway keeps the tree**: every refined cell keeps its subtree and its value,
  the new cells hold their depth-1 values, the root's answer is its matrix's, and the
  steps after it refine new cells too; the report counts what was added and filled;
- **no grow, no change**: a grow that never asks (or asks for actions already on the
  menu) is the deepening without one, to the bit;
- **with the oracle** the added actions become its candidates;
- **the restricted reading** refuses it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.port import batched_payoff
from pokeuraou.search import DEFAULT_SUB_BRANCHES, DEFAULT_SUB_LIMIT, belief_solve
from pokeuraou.teams import load_roster

from ._port import Budget
from .test_search import LEAF, _played


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _menus(reg, pos, limit):  # noqa: ANN001, ANN202
    return narrow(reg, pos, 0, limit=limit).actions, narrow(reg, pos, 1, limit=limit).actions


def _union(narrow_menu, wide_menu):  # noqa: ANN001, ANN202
    """The narrow menu, then the wide menu's actions not on it: the grown root's order."""
    on = {a.to_choice() for a in narrow_menu}
    return [*narrow_menu, *[a for a in wide_menu if a.to_choice() not in on]]


def _once(at: float, wanted):  # noqa: ANN001, ANN202
    """A grow that asks for `wanted` once, at the first step whose reading is `at` or more."""
    asked = []

    def grow(spent: float):  # noqa: ANN202
        asked.append(spent)
        if spent >= at and not getattr(grow, "done", False):
            grow.done = True  # type: ignore[attr-defined]
            return wanted
        return None

    grow.asked = asked  # type: ignore[attr-defined]
    return grow


def _root(reg, pos, ours, theirs, cells, **kw):  # noqa: ANN001, ANN202
    payoff, notes = batched_payoff(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
    trace: list = []
    got = deepen_mod.deepen_root(
        reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), payoff=payoff,
        equilibrium=solve(payoff), cells=cells, sub_limit=DEFAULT_SUB_LIMIT,
        sub_branches=DEFAULT_SUB_BRANCHES, unmodelled=set(notes), trace=trace, **kw,
    )
    return got, trace


def _tree(root) -> list:  # noqa: ANN001
    """Every node's refined cells, their menus and values, in a fixed order."""
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        rows = node.rows if isinstance(node, deepen_mod._Node) else node.own
        cols = node.cols if isinstance(node, deepen_mod._Node) else node.other
        out.append((
            [a.to_choice() for a in rows], [a.to_choice() for a in cols], sorted(node.children),
        ))
        for cell in sorted(node.children):
            for _w, child in node.children[cell]:
                if isinstance(child, deepen_mod._Node):
                    queue.append(child)
                    out.append(child.payoff.tobytes())
                else:
                    out.append(child)
    return out


def test_a_grow_before_the_first_step_is_the_wide_menu_from_the_start(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    for pos in _played(roster)[:3]:
        small = _menus(reg, pos, 3)
        big = _menus(reg, pos, 6)
        rows, cols = _union(small[0], big[0]), _union(small[1], big[1])
        added_cells = len(rows) * len(cols) - len(small[0]) * len(small[1])
        assert added_cells > 0
        wide, wide_trace = _root(reg, pos, rows, cols, 300)
        grown, grown_trace = _root(
            reg, pos, *small, 300 + added_cells, grow=_once(0.0, big),
        )
        assert [a.to_choice() for a in grown.rows] == [a.to_choice() for a in rows]
        assert [a.to_choice() for a in grown.cols] == [a.to_choice() for a in cols]
        assert np.array_equal(grown.payoff, wide.payoff)
        assert np.array_equal(grown.equilibrium.row_strategy, wide.equilibrium.row_strategy)
        assert np.array_equal(grown.equilibrium.col_strategy, wide.equilibrium.col_strategy)
        assert grown.equilibrium.value == wide.equilibrium.value
        assert _tree(grown_trace[0]) == _tree(wide_trace[0])
        assert grown.report.expanded == wide.report.expanded
        assert grown.report.grown == len(rows) + len(cols) - len(small[0]) - len(small[1])
        assert grown.report.grown_cells == added_cells
        assert grown.report.cells == wide.report.cells + added_cells
        assert grown_trace[1] == (grown_trace[0], None, "grow")
        assert grown.report.to_json()["grown"] == grown.report.grown


def test_a_grow_midway_keeps_the_tree(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    new_refined = 0
    for pos in _played(roster)[:4]:
        small = _menus(reg, pos, 3)
        big = _menus(reg, pos, 6)
        seen: dict = {}

        def watch(step, seen=seen):  # noqa: ANN001, ANN202
            # The tree just before the grow: the refined root cells, their branches.
            if step.kind != "grow":
                seen["children"] = {c: list(b) for c, b in step.root.children.items()}
                seen["payoff"] = step.root.payoff.copy()

        grow = _once(150.0, big)
        got, trace = _root(reg, pos, *small, 600, grow=grow, progress=watch)
        at = next(n for n, s in enumerate(trace) if n and s[2] == "grow")
        root = trace[0]
        before = trace[1:at]
        assert before, "nothing was refined before the grow"
        r0, c0 = len(small[0]), len(small[1])
        # Every cell refined before the grow keeps its branches (the same objects).
        kept = {node_cell[1] for node_cell in before if node_cell[0] is root and node_cell[2]}
        for cell in kept:
            assert cell[0] < r0 and cell[1] < c0
            assert cell in root.children
        # The new cells: depth-1 values of the grown menus, unless refined since.
        fresh, _ = batched_payoff(reg, pos, got.rows, got.cols, LEAF, budget=Budget.matrix())
        for i in range(len(got.rows)):
            for j in range(len(got.cols)):
                if (i, j) in root.children:
                    assert got.payoff[i, j] == deepen_mod._cell_value(root.children[(i, j)])
                    if i >= r0 or j >= c0:
                        new_refined += 1
                else:
                    assert got.payoff[i, j] == fresh[i, j]
        assert got.equilibrium.value == solve(got.payoff).value
        assert got.report.grown == len(got.rows) + len(got.cols) - r0 - c0 > 0
        assert got.report.grown_cells == len(got.rows) * len(got.cols) - r0 * c0
        # Asked at the top of every step, with the budget's reading, never past it.
        assert grow.asked[0] == 0.0 and all(a < 600 for a in grow.asked)
        assert grow.asked == sorted(grow.asked)
    assert new_refined > 0, "no cell the grow added was ever refined"


def test_no_grow_no_change(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    pos = _played(roster)[1]
    small = _menus(reg, pos, 4)
    plain, plain_trace = _root(reg, pos, *small, 300)
    for grow in (lambda spent: None, _once(0.0, small), _once(100.0, ([], []))):
        got, trace = _root(reg, pos, *small, 300, grow=grow)
        assert got.report == plain.report
        assert np.array_equal(got.payoff, plain.payoff)
        assert _tree(trace[0]) == _tree(plain_trace[0])
        assert "grown" not in got.report.to_json()


def test_the_oracle_takes_the_added_actions_as_candidates(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    pos = _played(roster)[0]
    small = _menus(reg, pos, 3)
    big = _menus(reg, pos, 6)
    got, trace = _root(
        reg, pos, *small, 400, outside=small, swap=True, grow=_once(50.0, big)
    )
    names = [a.to_choice() for a in got.rows]
    assert len(set(names)) == len(names)
    assert got.report.grown > 0
    with pytest.raises(ValueError, match="whole-matrix"):
        _root(reg, pos, *small, 10, reading="restricted", grow=_once(0.0, big))


def _hidden(roster, pos):  # noqa: ANN001, ANN202
    sheet = list(roster.sets)[:6]
    return {side: completions(roster.reg, pos, side, sheet) for side in (0, 1)}


@pytest.mark.parametrize("side", [0, 1])
def test_a_bayesian_root_grows_as_the_wide_menu_from_the_start(roster, side) -> None:  # noqa: ANN001
    reg = roster.reg
    for pos in _played(roster)[:2]:
        spreads = _hidden(roster, pos)
        count = len(spreads[1 - side])
        small = _menus(reg, pos, 3)
        big = _menus(reg, pos, 5)
        rows, cols = _union(small[0], big[0]), _union(small[1], big[1])
        added = (len(rows) * len(cols) - len(small[0]) * len(small[1])) * count
        wide_trace: list = []
        wide = belief_solve(
            reg, pos, rows, cols, spreads, {0: LEAF, 1: LEAF}, budget=Budget.matrix(),
            sides=(side,), deepen={side: {"cells": 200, "trace": wide_trace}},
        )[side]
        grown_trace: list = []
        grown = belief_solve(
            reg, pos, *small, spreads, {0: LEAF, 1: LEAF}, budget=Budget.matrix(),
            sides=(side,),
            deepen={side: {"cells": 200 + added, "trace": grown_trace,
                           "grow": _once(0.0, big)}},
        )[side]
        assert [a.to_choice() for a in grown.ours] == [a.to_choice() for a in rows]
        assert [a.to_choice() for a in grown.theirs] == [a.to_choice() for a in cols]
        assert np.array_equal(grown.strategy, wide.strategy)
        assert grown.value == wide.value
        for a, b in zip(grown_trace[0].prices, wide_trace[0].prices, strict=True):
            assert np.array_equal(a, b)
        assert _tree(grown_trace[0]) == _tree(wide_trace[0])
        assert grown.deepened.grown_cells == added
        assert grown.deepened.expanded == wide.deepened.expanded


@pytest.mark.parametrize("side", [0, 1])
def test_a_bayesian_root_grown_midway_keeps_its_tree(roster, side) -> None:  # noqa: ANN001
    reg = roster.reg
    pos = _played(roster)[1]
    spreads = _hidden(roster, pos)
    small = _menus(reg, pos, 3)
    big = _menus(reg, pos, 5)
    trace: list = []
    got = belief_solve(
        reg, pos, *small, spreads, {0: LEAF, 1: LEAF}, budget=Budget.matrix(), sides=(side,),
        deepen={side: {"cells": 500, "trace": trace, "grow": _once(120.0, big)}},
    )[side]
    root = trace[0]
    at = next(n for n, s in enumerate(trace) if n and s[2] == "grow")
    own0 = len(small[0] if side == 0 else small[1])
    other0 = len(small[1] if side == 0 else small[0])
    refined_before = {s[1] for s in trace[1:at] if s[2] is True}
    assert refined_before
    for cell in refined_before:
        k, i, j = cell
        assert i < own0 and j < other0 and cell in root.children
    for (k, i, j), branches in root.children.items():
        won = deepen_mod._cell_value(branches)
        assert root.prices[k][i, j] == (won if side == 0 else -won)
    assert got.deepened.grown > 0
    assert len(got.strategy) == len(got.ours if side == 0 else got.theirs)
