"""IKA-342: the deepening's depth discount (``d<P>``).

What these tests hold:

- **the label**: ``d<P>`` reads after ``g<L>`` as P / 100 a ply (1 to 99); every earlier
  label has none; breadth only (``b``) refines nothing, so it takes none;
- **the discount moves the search up the tree**: with the same budget the refined cells
  sit nearer the root than without it (the positive control), and a strong discount
  keeps them nearer still;
- **the step's choice is the discounted priority**: `_best` takes the cell whose
  inherited priority times ``discount**level`` is highest, and the ahead helpers' guess
  (`_ranked`) puts that cell first;
- **in a game**: without ``d`` the game is the game (the label changes nothing else), with
  it the moves change, and it repeats.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import selfplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow
from pokeuraou.port import batched_payoff
from pokeuraou.regulation import load_regulation
from pokeuraou.search import DEFAULT_SUB_BRANCHES, DEFAULT_SUB_LIMIT
from pokeuraou.teams import load_roster

from ._port import Budget
from .test_search import LEAF, _played


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _deepen(reg, pos, ours, theirs, cells, **kw):  # noqa: ANN001, ANN202, ANN003
    payoff, notes = batched_payoff(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
    trace: list = []
    got = deepen_mod.deepen_root(
        reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), payoff=payoff,
        equilibrium=solve(payoff), cells=cells, sub_limit=DEFAULT_SUB_LIMIT,
        sub_branches=DEFAULT_SUB_BRANCHES, unmodelled=set(notes), trace=trace, **kw,
    )
    return got, trace


def _levels(trace) -> list[int]:  # noqa: ANN001
    return [node.level for node, _cell, took in trace[1:] if took and _cell is not None]


def test_the_label_reads_the_discount() -> None:
    spec = deepen_mod.deepen_spec("m1200sallhc6g32d80")
    assert (spec.child_q, spec.levels, spec.discount) == (6, 32, 0.8)
    assert deepen_mod.deepen_spec("m800d5").discount == 0.05
    assert deepen_mod.deepen_spec("m800hd90").levels is None
    for old in ("none", "m400", "r25", "m400o24", "m400sallh", "m1200sq3h", "b200s24",
                "m1200hc6g16", "m100hg32"):
        assert deepen_mod.deepen_spec(old).discount is None
    for bad in ("m100d0", "m100d100", "m100d", "m100d80g8", "b50salld80", "noned80", "m100d080"):
        with pytest.raises(ValueError, match="deepen"):
            deepen_mod.deepen_spec(bad)


def test_the_discount_moves_the_steps_up_the_tree(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    means = {None: [], 0.8: [], 0.3: []}
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos, 3)
        for discount in means:
            got, trace = _deepen(reg, pos, ours, theirs, 1200, levels=32, discount=discount)
            levels = _levels(trace)
            assert levels, "the budget refines cells"
            assert all(level < 32 for level in levels)
            means[discount].append(float(np.mean(levels)))
    plain, mild, strong = (float(np.mean(means[d])) for d in (None, 0.8, 0.3))
    # The positive control: the discount is what moves the steps up.
    assert mild < plain
    assert strong < mild


def _menus(reg, pos, limit):  # noqa: ANN001, ANN202
    return narrow(reg, pos, 0, limit=limit).actions, narrow(reg, pos, 1, limit=limit).actions


def test_the_step_takes_the_best_discounted_cell(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    discount = 0.6
    checked = 0
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos, 3)
        got, trace = _deepen(reg, pos, ours, theirs, 600, levels=16, discount=discount)
        root = trace[0]
        # Every unrefined cell's discounted priority, walking the tree from the root.
        best = None
        stack = [(root, None)]
        while stack:
            node, inherited = stack.pop()
            scores = deepen_mod._scores(node, inherited)  # noqa: SLF001
            for i in range(scores.shape[0]):
                for j in range(scores.shape[1]):
                    cell = (i, j)
                    if cell in node.children or cell in node.refused or node.level >= 16:
                        continue
                    if scores[cell] > 0 and (best is None or scores[cell] > best[0]):
                        best = (float(scores[cell]), node, cell)
            for cell, branches in node.children.items():
                for weight, child in branches:
                    if isinstance(child, deepen_mod._Node):
                        stack.append((child, float(scores[cell]) * weight * discount))
        taken = deepen_mod._best(root, 16, None, discount)  # noqa: SLF001
        if best is None:
            assert taken is None
            continue
        node, cell = taken
        score = deepen_mod._scores(node, _inherited(node, discount))[cell]  # noqa: SLF001
        assert score == pytest.approx(best[0], rel=1e-12)
        ranked = deepen_mod._ranked(root, 4, 16, discount)  # noqa: SLF001
        assert ranked[0] == (node, cell)
        checked += 1
    assert checked > 0


def _inherited(node, discount):  # noqa: ANN001, ANN202
    """The priority `node` inherits: its parent cell's, times its branch's weight, times the
    discount (None at the root)."""
    if node.parent is None:
        return None
    parent, cell = node.parent
    up = deepen_mod._scores(parent, _inherited(parent, discount))[cell]  # noqa: SLF001
    for weight, child in parent.children[cell]:
        if child is node:
            return float(up) * weight * discount
    raise AssertionError("a child not among its parent's branches")


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


def test_in_a_game_the_discount_changes_the_moves_and_repeats(setup) -> None:  # noqa: ANN001
    plain = _payload(_hidden_game(setup, "m200hg16"))
    first = _payload(_hidden_game(setup, "m200hg16d30"))
    second = _payload(_hidden_game(setup, "m200hg16d30"))
    assert first == second
    assert first["deepen"] == ["m200hg16d30", "m200hg16d30"]
    assert first["decisions"] != plain["decisions"]


def test_the_meter_prices_a_refined_cell_by_its_level() -> None:
    plain = deepen_mod.Cost(fill=5.0, refine=1.5, cell=0.1)
    leveled = deepen_mod.Cost(fill=5.0, refine=1.5, cell=0.1, level=0.4)
    a, b = deepen_mod._Meter(plain), deepen_mod._Meter(leveled)  # noqa: SLF001
    for meter in (a, b):
        meter.refined(10, 1, 0)
        meter.refined(4, 1, 7)
        meter.refined(2, 1, 12)
    assert a.levels == b.levels == 19
    # Without the price a level costs nothing: the old reading, to the bit.
    assert a.spent == plain.ms(3, 3, 16) / 0.1
    assert b.spent == pytest.approx((plain.ms(3, 3, 16) + 0.4 * 19) / 0.1)
    # A clock that stands in for a Cost (no `level`) reads as before.

    class Clock:
        cell = 1.0

        def ms(self, fills, refines, cells, probed=0, qs=0):  # noqa: ANN001, ANN202, ARG002
            return 42.0

    c = deepen_mod._Meter(Clock())  # noqa: SLF001
    c.refined(3, 1, 9)
    assert c.spent == 42.0
