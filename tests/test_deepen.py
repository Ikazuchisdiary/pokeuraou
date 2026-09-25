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
- **it follows its arm** into either seat of a match, and names another agent;
- **the root's double oracle** (IKA-293, ``m<N>o<W>``): over known cells it reaches the
  whole game's value, adding the largest gain first; with nothing outside the menu it
  is ``m<N>`` to the bit; on real positions it adds only outside actions, keeps the
  grown root's unrefined cells at their depth-1 values, and repeats; the wider menus
  are the agent's own; in a game it widens exactly where it deepens and the recorded
  menus are the grown ones;
- **a cost** charges fills and refinements as well as cells (`cells_for_seconds`).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import selfplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow, slot_options
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
    assert deepen_mod.parse_deepen("m400o24") == ("mixed", 400)
    assert deepen_mod.deepen_spec("m400o24") == deepen_mod.DeepenSpec("mixed", 400, 24)
    assert deepen_mod.deepen_spec("m50oall").oracle == deepen_mod.ALL_ACTIONS
    assert deepen_mod.deepen_spec("m50").oracle is None
    assert deepen_mod.deepen_spec("none") == deepen_mod.DeepenSpec(None, 0)
    assert deepen_mod.deepen_spec("m400s24") == deepen_mod.DeepenSpec("mixed", 400, 24, True)
    assert deepen_mod.deepen_spec("b200sall") == deepen_mod.DeepenSpec(
        "breadth", 200, deepen_mod.ALL_ACTIONS, True
    )
    assert deepen_mod.deepen_spec("b200o24").swap is False
    for bad in ("", "0", "400", "m0", "r", "x400", "R400", "m-1", "none1", "r04",
                "m40o0", "m40o", "m40oAll", "m40o04", "nonoe24", "o24", "b200", "m40x24",
                "m40s", "s24"):
        with pytest.raises(ValueError, match="deepen"):
            deepen_mod.parse_deepen(bad)
    # The oracle goes with the whole-matrix reading only.
    for bad in ("r40o24", "r40s24"):
        with pytest.raises(ValueError, match="mixed|whole-matrix"):
            deepen_mod.parse_deepen(bad)


def test_seconds_turn_into_a_budget_at_measured_prices() -> None:
    local = deepen_mod.COSTS["local", 1]
    assert deepen_mod.cells_for_seconds(0.1) == int(100.0 / local.cell)
    served = deepen_mod.COSTS["served", 1]
    assert deepen_mod.cells_for_seconds(2, form="served") == int(2000.0 / served.cell)
    assert deepen_mod.cells_for_seconds(-1) == 0
    # A price is measured at its core count, never scaled from one core (IKA-32).
    with pytest.raises(ValueError, match="core"):
        deepen_mod.cells_for_seconds(45, cores=2)
    assert local.ms(2, 3, 100) == pytest.approx(2 * local.fill + 3 * local.refine + 100 * local.cell)


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


# --------------------------------------------------------------------------------------
# IKA-293: the root's double oracle


class _Act:
    """An action as the oracle sees it: only its choice."""

    def __init__(self, name: str) -> None:
        self.name = name

    def to_choice(self) -> str:
        return self.name


def _grown_by_oracle(  # noqa: ANN202
    full: np.ndarray, rows: list[int], cols: list[int], swap: bool = False
):
    """The oracle run to the end over a matrix whose every cell it already knows."""
    names_r = [_Act(f"r{i}") for i in range(full.shape[0])]
    names_c = [_Act(f"c{j}") for j in range(full.shape[1])]
    payoff = full[np.ix_(rows, cols)]
    root = deepen_mod._Node(
        pos=None, rows=[names_r[i] for i in rows], cols=[names_c[j] for j in cols],
        payoff=payoff.copy(), equilibrium=solve(payoff), level=0,
    )
    oracle = deepen_mod._Oracle(root, (names_r, names_c), swap=swap)
    for i, a in enumerate(names_r):
        for j, b in enumerate(names_c):
            oracle.known[(a.to_choice(), b.to_choice())] = float(full[i, j])
    meter = deepen_mod._Meter(None)
    added = []
    for _ in range(1_000):
        before = (len(root.rows), len(root.cols))
        oracle.probe(None, None, None, meter, set())
        if not oracle.widen(None, None, None, meter, set()):
            break
        added.append("row" if len(root.rows) > before[0] else "col")
    else:
        raise AssertionError("the oracle never stopped")
    return root, oracle, meter, added


def test_the_double_oracle_reaches_the_whole_games_value() -> None:
    """Known cells, no port: the oracle adds only actions that gain, and when none is left
    the root's value is the whole matrix's -- what makes it a double oracle."""
    rng = np.random.default_rng(293)
    grew = 0
    for _ in range(40):
        full = rng.random((7, 6))
        root, oracle, meter, added = _grown_by_oracle(full, [0, 1], [0])
        want = solve(full)
        assert root.equilibrium.value == pytest.approx(want.value, abs=1e-7)
        # The strategy it plays is an equilibrium of the whole game: no action outside
        # the grown menus beats it.
        x = np.zeros(full.shape[0])
        y = np.zeros(full.shape[1])
        for k, a in enumerate(root.rows):
            x[int(a.name[1:])] = root.equilibrium.row_strategy[k]
        for k, b in enumerate(root.cols):
            y[int(b.name[1:])] = root.equilibrium.col_strategy[k]
        assert (full @ y).max() <= want.value + 1e-6
        assert (x @ full).min() >= want.value - 1e-6
        # Nothing was filled (every cell known), and the count says what joined.
        assert meter.fills == 0 and meter.cells == 0 and oracle.probed == 0
        assert oracle.widened == len(added) == len(root.rows) + len(root.cols) - 3
        grew += oracle.widened
    assert grew > 40, "the oracle hardly ever added anything"


def test_the_oracle_adds_the_largest_gain_rows_first_on_a_tie() -> None:
    # Root [[0.5]]: row 1 gains 0.125, row 2 0.25, column 1 gains 0.5 - 0.25 = 0.25 (all
    # exact in binary) -> row 2: the largest, not the first that gains, and the tie
    # with the column going to the rows.
    full = np.array([[0.5, 0.25], [0.625, 0.125], [0.75, 0.875]])
    root, _oracle, _meter, added = _grown_by_oracle(full, [0], [0])
    assert added[0] == "row" and root.rows[1].name == "r2"
    # A matrix it cannot improve is left alone: the root's row is dominant.
    flat = np.array([[0.9, 0.9], [0.1, 0.2]])
    root, oracle, _meter, added = _grown_by_oracle(flat, [0], [0])
    assert added == [] and oracle.widened == 0


def test_swapping_still_reaches_the_whole_games_value() -> None:
    """With swaps the menus stay small, each action leaves at most once, and the oracle
    still ends where nothing outside gains: at the whole game's value."""
    rng = np.random.default_rng(310)
    finished = swapped = 0
    for _ in range(60):
        full = rng.random((9, 8))
        root, oracle, _meter, added = _grown_by_oracle(full, [0, 1, 2], [0, 1, 2], swap=True)
        assert len(root.rows) + len(root.cols) == 6 + oracle.widened - oracle.swapped
        swapped += oracle.swapped
        assert not oracle.stalled and oracle.swapped == len(oracle.left)
        finished += 1
        assert root.equilibrium.value == pytest.approx(solve(full).value, abs=1e-7)
    assert swapped > 60 and finished == 60, (swapped, finished)


def test_a_swap_pushes_out_only_a_weightless_action_and_keeps_the_tree() -> None:
    a, b, c = _Act("a"), _Act("b"), _Act("c")
    x, y = _Act("x"), _Act("y")

    def root_of(payoff):  # noqa: ANN001, ANN202
        payoff = np.array(payoff, dtype=np.float64)
        return deepen_mod._Node(
            pos=None, rows=[a, b, c], cols=[x, y], payoff=payoff,
            equilibrium=solve(payoff), level=0,
        )

    # Matching pennies on a, b; c is dominated, so it carries no weight.
    root = root_of([[1.0, 0.0], [0.0, 1.0], [0.25, 0.25]])
    oracle = deepen_mod._Oracle(root, ([], []), swap=True)
    oracle._swap_out(0, added="a")
    assert [r.name for r in root.rows] == ["a", "b"] and oracle.swapped == 1
    assert root.equilibrium.value == pytest.approx(0.5)
    # Nothing weightless (other than the one that joined): nothing leaves.
    oracle._swap_out(0, added="a")
    assert [r.name for r in root.rows] == ["a", "b"] and oracle.swapped == 1
    # A weightless row with a refined cell stays; the refined cells of the rows after a
    # removed one move up with their child nodes' parent links.
    root = root_of([[0.25, 0.25], [1.0, 0.0], [0.0, 1.0]])
    child = deepen_mod._Node(pos=None, rows=[], cols=[], payoff=np.zeros((1, 1)),
                             equilibrium=solve(np.array([[0.5]])), level=1)
    child.parent = (root, (2, 1))
    root.children = {(2, 1): [(1.0, child)]}
    root.refused = {(1, 0)}
    oracle = deepen_mod._Oracle(root, ([], []), swap=True)
    oracle._swap_out(0, added="b")
    assert [r.name for r in root.rows] == ["b", "c"]
    assert root.children == {(1, 1): [(1.0, child)]} and child.parent == (root, (1, 1))
    assert root.refused == {(0, 0)}
    assert root.payoff.shape == (2, 2) and root.payoff[1, 1] == 1.0
    blocked = root_of([[0.25, 0.25], [1.0, 0.0], [0.0, 1.0]])
    blocked.children = {(0, 0): []}
    oracle = deepen_mod._Oracle(blocked, ([], []), swap=True)
    oracle._swap_out(0, added="b")
    assert len(blocked.rows) == 3 and oracle.swapped == 0


def _oracle_root(reg, pos, ours, theirs, wide, cells, swap=False, reading="mixed"):  # noqa: ANN001, ANN202
    payoff, notes = batched_payoff(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
    trace: list = []
    got = deepen_mod.deepen_root(
        reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), payoff=payoff,
        equilibrium=solve(payoff), cells=cells, sub_limit=DEFAULT_SUB_LIMIT,
        sub_branches=DEFAULT_SUB_BRANCHES, unmodelled=set(notes), trace=trace,
        outside=wide, swap=swap, reading=reading,
    )
    return payoff, got, trace


def test_with_nothing_outside_the_oracle_is_the_deepening_to_the_bit(roster) -> None:  # noqa: ANN001
    """The null control: candidates that are all on the menu leave `m<N>` unchanged."""
    reg = roster.reg
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos, limit=4)
        payoff, got, trace = _oracle_root(reg, pos, ours, theirs, (ours, theirs), 300)
        _p, eq, prices, report, _t = _grow(reg, pos, ours, theirs, cells=300)
        assert np.array_equal(got.payoff, prices)
        assert np.array_equal(got.equilibrium.row_strategy, eq.row_strategy)
        assert np.array_equal(got.equilibrium.col_strategy, eq.col_strategy)
        assert got.equilibrium.value == eq.value
        assert [a.to_choice() for a in got.rows] == [a.to_choice() for a in ours]
        assert got.report.oracle and got.report.probed == 0 and got.report.widened == 0
        plain_json = report.to_json()
        oracle_json = got.report.to_json()
        assert {k: v for k, v in oracle_json.items() if k not in ("probed", "widened", "fills")} == plain_json
        assert "probed" not in plain_json


@pytest.mark.parametrize("swap", [False, True])
def test_the_oracle_widens_the_root_with_outside_actions(roster, monkeypatch, swap) -> None:  # noqa: ANN001
    reg = roster.reg
    widened = swapped = 0
    # Every cell deepening fills, counted where it is filled: the report must charge them all.
    filled = [0]
    many = deepen_mod.port.batched_payoffs

    def counted_many(reg_, pos_, rows, cols, evaluators, *, budget, cells=None):  # noqa: ANN001, ANN202
        filled[0] += len(cells) if cells is not None else len(rows) * len(cols)
        return many(reg_, pos_, rows, cols, evaluators, budget=budget, cells=cells)

    for pos in _played(roster)[:4]:
        ours, theirs = _menus(reg, pos, limit=3)
        wide = _menus(reg, pos, limit=8)
        with monkeypatch.context() as patched:
            # Every fill goes through `port.batched_payoffs` (`batched_payoff` calls it).
            patched.setattr(deepen_mod.port, "batched_payoffs", counted_many)
            filled[0] = 0
            payoff, got, trace = _oracle_root(reg, pos, ours, theirs, wide, 400, swap)
            refines = got.report.expanded + got.report.refused
            spent = filled[0] - len(ours) * len(theirs)  # less the depth-1 root's own fill
            assert got.report.cells == refines + spent, "a filled cell went uncharged"
        again = _oracle_root(reg, pos, ours, theirs, wide, 400, swap)[1]
        # The same answer twice (no clock, no order from a set).
        assert got.report == again.report and np.array_equal(got.payoff, again.payoff)
        assert [a.to_choice() for a in got.rows] == [a.to_choice() for a in again.rows]
        root = trace[0]
        report = got.report
        # Each action once, every one from the menu or the candidates.
        for grown, given, pool in ((got.rows, ours, wide[0]), (got.cols, theirs, wide[1])):
            names = [a.to_choice() for a in grown]
            assert len(set(names)) == len(names)
            assert set(names) <= {a.to_choice() for a in (*pool, *given)}
        assert report.widened - report.swapped == (
            len(got.rows) + len(got.cols) - len(ours) - len(theirs)
        )
        if not swap:
            # Adding only: the menus it was given come first, then what joined.
            assert got.rows[: len(ours)] == list(ours)
            assert got.cols[: len(theirs)] == list(theirs)
            assert report.swapped == 0 and report.uncovered == ((), ())
        else:
            # What a swap took out of the cover is said, in `narrow`'s words.
            for side, (given, now) in enumerate(((ours, got.rows), (theirs, got.cols))):
                before, after = slot_options(reg, given), slot_options(reg, now)
                assert report.uncovered[side] == tuple(
                    sorted(before[k] for k in set(before) - set(after))
                )
        swapped += report.swapped
        assert report.widened == sum(1 for step in trace[1:] if step[1] is None)
        # Unrefined root cells are depth-1 values of the grown menus; refined ones are
        # their branches' values; the equilibrium is the matrix's.
        fresh, _ = batched_payoff(reg, pos, got.rows, got.cols, LEAF, budget=Budget.matrix())
        untouched = np.ones(fresh.shape, dtype=bool)
        for cell, branches in root.children.items():
            untouched[cell] = False
            assert got.payoff[cell] == deepen_mod._cell_value(branches)
        assert np.array_equal(got.payoff[untouched], fresh[untouched])
        assert got.equilibrium.value == solve(got.payoff).value
        assert report.fills >= (1 if report.probed else 0)
        widened += report.widened
        # A budget of one cell is one step: a probe and one widening or deepening.
        single = _oracle_root(reg, pos, ours, theirs, wide, 1, swap)[1].report
        assert single.widened + single.expanded + single.refused <= 1
        # Breadth only never deepens, and stops once nothing gains.
        alone = _oracle_root(reg, pos, ours, theirs, wide, 10_000, swap, "breadth")[1]
        assert alone.report.expanded == 0 and alone.report.depth == 1
    assert widened >= 1, "no outside action ever joined, so the oracle was not exercised"
    if swap:
        assert swapped >= 1, "nothing was ever swapped out"


def test_a_cost_counts_fills_and_refinements_too(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    pos = _played(roster)[1]
    ours, theirs = _menus(reg, pos)
    cost = deepen_mod.Cost(fill=3.0, refine=6.0, cell=0.03)
    plain = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), deepen=2_000)
    priced = search(
        reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), deepen=2_000, deepen_cost=cost
    )
    # At these prices a refinement with its fills is worth hundreds of cells: fewer steps.
    assert 0 < priced.deepened.expanded < plain.deepened.expanded
    with pytest.raises(ValueError, match="deepen"):
        search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), outside=(ours, theirs))


def test_the_wider_menus_come_from_the_same_ranking(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    for pos in _played(roster)[:2]:
        for ranked in (False, True):
            plain = selfplay._menus(reg, pos, (4, 4), LEAF, Budget.matrix(), ranked)
            wider: dict = {}
            got = selfplay._menus(
                reg, pos, (4, 4), LEAF, Budget.matrix(), ranked, wide=[8, 8], wider=wider,
            )
            assert [[a.to_choice() for a in m] for m in got] == [
                [a.to_choice() for a in m] for m in plain
            ]
            alone = selfplay._menus(reg, pos, (8, 8), LEAF, Budget.matrix(), ranked)
            assert sorted(wider) == [8]
            assert [[a.to_choice() for a in m] for m in wider[8]] == [
                [a.to_choice() for a in m] for m in alone
            ]


@pytest.mark.parametrize("label", ["m60o6", "m60oall", "m60s6", "b60sall"])
def test_under_a_hidden_bench_the_oracle_widens_where_it_deepens(setup, label) -> None:  # noqa: ANN001
    base = _hidden_game(setup, "none")
    plain = _hidden_game(setup, "m60")
    deep = _hidden_game(setup, label)
    fired = widened = 0
    first = None
    for k, d in enumerate(deep.decisions):
        if d.deepened is None:
            continue
        assert d.kind == "move" and _both_shown(d)
        report = d.deepened[0]
        assert {"probed", "widened", "fills"} <= set(report)
        assert ("swapped" in report) == ("s" in label[1:])
        if label.startswith("b"):
            assert report["expanded"] == 0 and report["depth"] == 1
        fired += 1
        first = k if first is None else first
        widened += report["widened"]
        # The recorded menus are the grown ones, and the policies index them.
        assert len(d.own_policy) == len(d.own_actions) and len(d.foe_policy) == len(d.foe_actions)
        assert d.own_chosen in d.own_actions and d.foe_chosen in d.foe_actions
    assert fired > 0 and widened > 0, (fired, widened)
    # Until the first deepened node the game is the default one.
    assert _payload(deep)["decisions"][:first] == _payload(base)["decisions"][:first]
    assert _payload(deep)["deepen"] == [label, label]
    # With nothing outside (the oracle's width is the menu's) it plays `m60` exactly.
    null = _hidden_game(setup, "m60o3")
    strip = {"probed", "widened", "fills"}
    for got in (null,):
        payload = _payload(got)
        want = _payload(plain)
        assert payload["deepen"] == ["m60o3", "m60o3"]
        payload["deepen"] = want["deepen"]
        for decision in payload["decisions"]:
            for side in decision.get("deepened") or []:
                if side is not None:
                    assert side["probed"] == 0 and side["widened"] == 0
                    for key in strip:
                        side.pop(key)
        assert payload == want


def test_the_oracle_follows_its_arm_into_either_seat(pool) -> None:  # noqa: ANN001
    solver = SolvedSelections(pool.reg, pool.teams, _stub)

    def arm(label: str) -> PoolArm:
        return PoolArm(name="arm", evaluate=_stub, solver=solver, limit=2,
                       rank_by_leaf=False, deepen=label)

    tested, other = arm("m30oall"), arm("m30")
    for which in (0, 1):
        record, sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=33, game_index=0, which=which,
            hide_bench=False, max_turns=2,
        )
        assert record.deepen[which] == "m30oall" and record.deepen[1 - which] == "m30"
        moves = [d for d in record.decisions if d.kind == "move"]
        assert moves
        for d in moves:
            assert "widened" in d.deepened[which] and "widened" not in d.deepened[1 - which]
