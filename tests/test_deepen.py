"""IKA-33: deepening a solved node best first, one cell at a time, to a budget of cells.

What these tests hold:

- **zero is the old search**: ``deepen=0`` is `search` at depth 1 to the bit, and a game
  played without the setting writes nothing new;
- **one step is `_refined_value`**: with a budget of one cell exactly one root cell is
  refined, it is the depth-1 matrix's highest ``bern/gap`` cell, and its value is the
  shipped depth-2 value of that cell to the bit;
- **the tree is what it says**: every node's equilibrium is its matrix's, every refined
  cell's value is its branches' values weighted, unrefined cells keep their leaf values,
  and a larger budget reaches past depth 2;
- **the budget stops it**, overshooting by at most one refined cell's children, and the
  report says what was spent -- the same numbers on every run (no clock in it);
- **the restricted reading** plays only its rectangle, reports what the strategy
  guarantees, refines no root cell outside the rectangle, and grows it only once every
  cell in it is settled -- by a row or a column that beats the rectangle's value;
- **it fires where nothing is hidden**: under a hidden bench, a deepening agent deepens
  exactly the move decisions where both sides have shown all four, and plays the default
  game until the first of them; in the open game it deepens every move decision;
- **it follows its arm** into either seat of a match, and names another agent.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import selfplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.port import batched_payoff
from pokeuraou.position import Position
from pokeuraou.provenance import LEGACY_DEEPEN, agent_name, provenance
from pokeuraou.regulation import load_regulation
from pokeuraou.search import (
    DEFAULT_SUB_BRANCHES,
    DEFAULT_SUB_LIMIT,
    _refined_value,
    search,
)
from pokeuraou.teams import load_roster

from ._port import Budget
from .test_poolplay import _stub, _variants, _write_pool
from .test_search import LEAF, _played


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _menus(reg, pos, limit=6):  # noqa: ANN001, ANN202
    return narrow(reg, pos, 0, limit=limit).actions, narrow(reg, pos, 1, limit=limit).actions


def _grow(reg, pos, ours, theirs, cells):  # noqa: ANN001, ANN202
    """The deepened root's answer and the trace of its tree."""
    payoff, notes = batched_payoff(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
    trace: list = []
    eq, prices, report = deepen_mod.best_first(
        reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), payoff=payoff,
        equilibrium=solve(payoff), cells=cells, sub_limit=DEFAULT_SUB_LIMIT,
        sub_branches=DEFAULT_SUB_BRANCHES, unmodelled=set(notes), trace=trace,
    )
    return payoff, eq, prices, report, trace


def _nodes(root) -> list:  # noqa: ANN001
    out, queue = [], [root]
    while queue:
        node = queue.pop(0)
        out.append(node)
        for cell in sorted(node.children):
            queue.extend(c for _, c in node.children[cell] if isinstance(c, deepen_mod._Node))
    return out


def test_zero_is_the_depth_one_search(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos)
        plain = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
        zero = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), deepen=0)
        assert np.array_equal(plain.payoff, zero.payoff)
        assert np.array_equal(plain.equilibrium.row_strategy, zero.equilibrium.row_strategy)
        assert np.array_equal(plain.equilibrium.col_strategy, zero.equilibrium.col_strategy)
        assert plain.equilibrium.value == zero.equilibrium.value
        assert zero.deepened is None and plain.unmodelled == zero.unmodelled
    with pytest.raises(ValueError, match="deepen"):
        search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=2, deepen=5)


def test_one_step_is_the_shipped_depth_two_value_of_the_best_cell(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    checked = 0
    for pos in _played(roster):
        ours, theirs = _menus(reg, pos)
        if not ours or not theirs:
            continue
        payoff, eq, prices, report, trace = _grow(reg, pos, ours, theirs, cells=1)
        steps = trace[1:]
        assert len(steps) >= 1
        node, cell, took = steps[0]
        # The first cell is the depth-1 matrix's highest bern/gap -- IKA-281's rule.
        d1 = solve(payoff)
        gap = d1.row_ev_loss[:, None] + d1.col_ev_loss[None, :]
        signal = payoff * (1 - payoff) / (gap + deepen_mod.GAP_FLOOR)
        assert cell == tuple(int(k) for k in np.unravel_index(int(np.argmax(signal)), signal.shape))
        if not took:
            continue
        # One refined cell spends its turn and its children's cells, which passes a budget
        # of one: exactly one step.
        assert len(steps) == 1 and report.expanded == 1 and report.depth == 2
        want, _notes, _solved = _refined_value(
            reg, pos, ours[cell[0]], theirs[cell[1]], LEAF, budget=Budget.matrix(),
            sub_limit=DEFAULT_SUB_LIMIT, sub_branches=DEFAULT_SUB_BRANCHES,
        )
        assert prices[cell] == want, "the refined cell is not `_refined_value`'s number"
        moved = np.argwhere(prices != payoff)
        assert all(tuple(int(k) for k in m) == cell for m in moved)
        assert eq.value == solve(prices).value
        checked += 1
    assert checked >= 2, "no cell was refined, so nothing was checked"


def test_the_tree_holds_its_values_and_reaches_past_two(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    deepest = 0
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos, limit=3)
        payoff, eq, prices, report, trace = _grow(reg, pos, ours, theirs, cells=1500)
        root = trace[0]
        assert root.payoff is not payoff
        assert np.array_equal(prices, root.payoff)
        for node in _nodes(root):
            assert node.equilibrium.value == solve(node.payoff).value
            for cell, branches in node.children.items():
                assert node.payoff[cell] == deepen_mod._cell_value(branches)
                assert sum(w for w, _ in branches) == pytest.approx(1.0, abs=1e-12)
            if node is root:
                untouched = np.ones(payoff.shape, dtype=bool)
                for cell in node.children:
                    untouched[cell] = False
                assert np.array_equal(node.payoff[untouched], payoff[untouched])
            else:
                fresh, _ = batched_payoff(
                    reg, node.pos, node.rows, node.cols, LEAF, budget=Budget.matrix()
                )
                untouched = np.ones(fresh.shape, dtype=bool)
                for cell in node.children:
                    untouched[cell] = False
                assert np.array_equal(node.payoff[untouched], fresh[untouched])
        levels = [node.level for node, _cell, took in trace[1:] if took]
        assert report.expanded == len(levels)
        assert report.depth == 1 + (max(levels) + 1 if levels else 0)
        deepest = max(deepest, report.depth)
    assert deepest >= 3, f"a 1,500-cell budget on a 3x3 menu never passed depth 2 ({deepest})"


def test_the_budget_stops_it_and_the_answer_repeats(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    overshoot = 1 + DEFAULT_SUB_BRANCHES * DEFAULT_SUB_LIMIT**2
    for pos in _played(roster)[:2]:
        ours, theirs = _menus(reg, pos)
        spent = []
        for cells in (50, 200, 400):
            got = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), deepen=cells)
            again = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), deepen=cells)
            report = got.deepened
            assert report == again.deepened
            assert np.array_equal(got.payoff, again.payoff)
            assert np.array_equal(got.equilibrium.row_strategy, again.equilibrium.row_strategy)
            assert report.budget == cells and report.cells < cells + overshoot
            assert got.refined == report.expanded
            spent.append(report.cells)
        assert spent == sorted(spent)


def test_the_restricted_reading_keeps_to_its_rectangle(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    grew = 0
    for pos in _played(roster)[:4]:
        ours, theirs = _menus(reg, pos, limit=8)
        payoff, notes = batched_payoff(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
        d1 = solve(payoff)
        trace: list = []
        eq, prices, report = deepen_mod.best_first(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), payoff=payoff,
            equilibrium=d1, cells=600, sub_limit=DEFAULT_SUB_LIMIT,
            sub_branches=DEFAULT_SUB_BRANCHES, unmodelled=set(notes), trace=trace,
            reading="restricted", refine=2,
        )
        root = trace[0]
        rows, cols = root.rect
        start = (
            [int(i) for i in deepen_mod._top(d1.row_strategy, 2)],
            [int(j) for j in deepen_mod._top(d1.col_strategy, 2)],
        )
        assert rows[: len(start[0])] == start[0] and cols[: len(start[1])] == start[1]
        # The strategy is the rectangle's, and the value what it guarantees.
        assert set(np.flatnonzero(eq.row_strategy > 1e-12)) <= set(rows)
        assert set(np.flatnonzero(eq.col_strategy > 1e-12)) <= set(cols)
        assert eq.value == pytest.approx(float((eq.row_strategy @ prices).min()), abs=1e-12)
        # Every root cell refined was inside the rectangle as it stood when it was refined:
        # replaying the steps, a row or column joins only once the rectangle is settled.
        used = [len(start[0]), len(start[1])]
        settled: set = set()
        for node, cell, _took in trace[1:]:
            if node is not root:
                continue
            p, q = rows.index(cell[0]), cols.index(cell[1])
            if p >= used[0] or q >= used[1]:
                # A row or column that joined since: the rectangle before it was settled.
                assert all(
                    (i, j) in settled for i in rows[: used[0]] for j in cols[: used[1]]
                ), cell
                used = [max(used[0], p + 1), max(used[1], q + 1)]
                grew += 1
            settled.add(cell)
        assert report.cells >= 0
    assert grew >= 1, "no rectangle ever grew, so the oracle was never exercised"


def test_a_decided_rectangle_still_asks_the_oracle() -> None:
    """A rectangle whose only cell is decided (exactly 1: no ``bern``, never refined) is
    settled, so the column outside that takes its strategy apart joins it.

    Found on IKA-254's answer set: a rectangle waiting on a cell worth 1.0 never grew, and
    its strategy lost 0.63 to a column it never asked about.
    """
    payoff = np.array([[1.0, 0.1], [0.9, 0.8]])
    node = deepen_mod._Node(
        pos=None, rows=[], cols=[], payoff=payoff, equilibrium=solve(payoff), level=0,
        rect=([0], [0]),
    )
    deepen_mod._reread(node)
    assert node.rect == ([0], [0, 1])
    # And the rectangle's strategy is read against the whole matrix: it guarantees 0.1.
    assert node.equilibrium.value == pytest.approx(0.1)


def test_the_label_parses_and_a_bad_one_stops() -> None:
    assert deepen_mod.DEFAULT_DEEPEN == LEGACY_DEEPEN == "none"
    assert deepen_mod.parse_deepen("none") == (None, 0)
    assert deepen_mod.parse_deepen("m400") == ("mixed", 400)
    assert deepen_mod.parse_deepen("r25") == ("restricted", 25)
    for bad in ("", "0", "400", "m0", "r", "x400", "R400", "m-1", "none1", "r04"):
        with pytest.raises(ValueError, match="deepen"):
            deepen_mod.parse_deepen(bad)


def test_seconds_turn_into_cells_outside_the_search() -> None:
    assert deepen_mod.cells_for_seconds(0.1) == deepen_mod.CELLS_PER_SECOND // 10
    assert deepen_mod.cells_for_seconds(45, cores=2) == 90 * deepen_mod.CELLS_PER_SECOND
    assert deepen_mod.cells_for_seconds(-1) == 0


# --------------------------------------------------------------------------------------
# play_game


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
        search_limit=3, max_turns=25, sheets=(sheet, sheet), deepen=label,
    )


def _both_shown(decision) -> bool:  # noqa: ANN001
    """Neither bench hidden: every one of each side's four is in `shown`."""
    pos = Position.from_json(decision.position)
    return all(len(decision.shown[i]) == len(pos.sides[i].pokemon) for i in (0, 1))


def _payload(record) -> dict:  # noqa: ANN001
    out = record.to_json(objective="hp-share", search_limit=3)
    out.pop("searchSeconds", None)
    out.pop("engine", None)
    return out


@pytest.mark.parametrize("label", ["r60", "m60"])
def test_under_a_hidden_bench_it_deepens_exactly_where_nothing_is_hidden(setup, label) -> None:  # noqa: ANN001
    base = _hidden_game(setup, "none")
    again = _hidden_game(setup, "none")
    assert _payload(base) == _payload(again)
    payload = base.to_json(objective="hp-share", search_limit=3)
    assert "deepen" not in payload
    assert all("deepened" not in d for d in payload["decisions"])
    moves = [d for d in base.decisions if d.kind == "move"]
    open_nodes = [d for d in moves if _both_shown(d)]
    # The test is only worth something if the game reaches such nodes and hidden ones.
    assert open_nodes, "the game never showed both fours; nothing could fire"
    assert len(open_nodes) < len(moves)

    deep = _hidden_game(setup, label)
    assert deep.deepen == [label, label]
    fired = 0
    first = None
    for k, d in enumerate(deep.decisions):
        if d.kind != "move":
            assert d.deepened is None
            continue
        if _both_shown(d):
            assert d.deepened is not None and d.deepened[0] is not None, k
            assert d.deepened[1] == d.deepened[0]
            fired += 1
            first = k if first is None else first
        else:
            assert d.deepened is None, f"decision {k} deepened with a bench hidden"
    assert fired > 0
    # Until the first deepened node the game is the default one, decision for decision.
    base_json = _payload(base)["decisions"]
    deep_json = _payload(deep)["decisions"]
    assert deep_json[:first] == base_json[:first]
    assert _payload(deep)["deepen"] == [label, label]
    # Some node actually refined something (the report is not only a label).
    assert any(d.deepened[0]["expanded"] > 0 for d in deep.decisions if d.deepened)


def test_in_the_open_game_every_move_decision_deepens(setup) -> None:  # noqa: ANN001
    reg, sheet = setup
    record = selfplay.play_game(
        reg, np.random.default_rng(34), sheet[:4], sheet[2:6], "test",
        search_limit=3, max_turns=3, open_information=True, deepen="r40",
    )
    moves = [d for d in record.decisions if d.kind == "move"]
    assert moves and all(d.deepened is not None for d in moves)
    assert all(d.deepened[0]["budget"] == 40 for d in moves)
    with pytest.raises(ValueError, match="deepen"):
        selfplay.play_game(
            reg, np.random.default_rng(34), sheet[:4], sheet[2:6], "test",
            search_limit=3, max_turns=1, open_information=True, deepen="m40", depth=2,
        )
    with pytest.raises(ValueError, match="deepen"):
        selfplay.play_game(
            reg, np.random.default_rng(34), sheet[:4], sheet[2:6], "test",
            search_limit=3, max_turns=1, open_information=True, deepen="40",
        )


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def test_the_budget_follows_its_arm_into_either_seat(pool) -> None:  # noqa: ANN001
    solver = SolvedSelections(pool.reg, pool.teams, _stub)

    def arm(label: str) -> PoolArm:
        return PoolArm(name="arm", evaluate=_stub, solver=solver, limit=2,
                       rank_by_leaf=False, deepen=label)

    tested, other = arm("r30"), arm("none")
    for which in (0, 1):
        record, sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=33, game_index=0, which=which,
            hide_bench=False, max_turns=2,
        )
        assert record.deepen[which] == "r30" and record.deepen[1 - which] == "none"
        assert sides["deepens"][which] == "r30"
        moves = [d for d in record.decisions if d.kind == "move"]
        assert moves
        for d in moves:
            assert d.deepened[which] is not None and d.deepened[1 - which] is None
        payload = record.to_json(objective="value:arm", search_limit=(2, 2))
        assert payload["deepen"] == record.deepen
    same = arm("none")
    record, _ = pool_match_game(
        pool.reg, pool, (same, same), seed=33, game_index=0, which=0,
        hide_bench=False, max_turns=2,
    )
    assert "deepen" not in record.to_json(objective="value:arm", search_limit=(2, 2))


def test_a_deepening_agent_is_another_agent() -> None:
    base = dict(seat="s", leaves=("m", "m"), limits=(12, 12), rankings=("leaf", "leaf"),
                information=("hidden-bench", "hidden-bench"))
    old = provenance("pool-match", **base)
    assert "deepens" not in old
    assert provenance("pool-match", **base, deepens=(LEGACY_DEEPEN,) * 2) == old
    new = provenance("pool-match", **base, deepens=("r400", LEGACY_DEEPEN))
    assert new["deepens"] == ["r400", "none"]
    assert agent_name(new, 0) == agent_name(old, 0) + "/deepen:r400"
    assert agent_name(new, 1) == agent_name(old, 1)
