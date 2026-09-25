"""IKA-294: deepening and the root's double oracle where a bench is still hidden.

A label ending in ``h`` spends its budget on the Bayesian root `belief_solve` answers
from (`deepen.deepen_belief`). What these tests hold:

- **the label**: ``h`` reads, goes with ``m`` and ``b`` and not with ``r``;
- **the Bayesian oracle over known cells** reaches the whole Bayesian game's value,
  adding or swapping; with one completion of weight 1 it adds what the open oracle adds,
  in the same order; a swap pushes out only a column no completion plays;
- **one step is IKA-111's cell**: with a budget of one cell exactly one (completion, row,
  column) is refined -- the depth-1 answer's highest weighted ``bern/gap`` -- and its value
  is `_refined_value` in that completion's position, in either side's orientation;
- **the tree is what it says**: refined root cells hold their branches' values, the rest
  their depth-1 values, and the root's answer is the Bayesian game's at those prices;
- **in a game** it fires at the nodes with a bench hidden, the recorded menus are the
  grown ones, and the nodes with none hidden are deepened as without the ``h``;
- **it follows its arm** into either seat of a hidden-bench match.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import selfplay
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve, solve_bayesian
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.position import Position
from pokeuraou.search import (
    DEFAULT_SUB_BRANCHES,
    DEFAULT_SUB_LIMIT,
    _refined_value,
    belief_solve,
)
from pokeuraou.teams import load_roster

from .test_deepen import _Act, _grown_by_oracle, _hidden_game, _payload, pool, setup  # noqa: F401
from .test_hidden_depth2 import LEAF, _played
from .test_poolplay import _stub


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def test_the_hidden_label_parses_and_goes_with_m_and_b() -> None:
    spec = deepen_mod.deepen_spec("m100sallh")
    assert spec == deepen_mod.DeepenSpec(
        "mixed", 100, deepen_mod.ALL_ACTIONS, swap=True, hidden=True
    )
    assert deepen_mod.deepen_spec("m60h").hidden and deepen_mod.deepen_spec("m60h").oracle is None
    assert deepen_mod.deepen_spec("b50s24h") == deepen_mod.DeepenSpec(
        "breadth", 50, 24, swap=True, hidden=True
    )
    assert not deepen_mod.deepen_spec("m100sall").hidden
    assert deepen_mod.parse_deepen("m100sallh") == ("mixed", 100)
    for bad in ("r60h", "b60h", "m60hh", "h", "noneh", "m60oallx"):
        with pytest.raises(ValueError):
            deepen_mod.deepen_spec(bad)


# --------------------------------------------------------------------------------------
# The Bayesian oracle over known cells (no port)


def _bayes_value(mats: list[np.ndarray], w: np.ndarray) -> float:
    return float(solve_bayesian(mats, w).value)


def _grown_by_belief_oracle(  # noqa: ANN202
    full: list[np.ndarray], w: np.ndarray, rows: list[int], cols: list[int], swap: bool = False
):
    """The Bayesian oracle run to the end over matrices whose every cell it knows."""
    names_r = [_Act(f"r{i}") for i in range(full[0].shape[0])]
    names_c = [_Act(f"c{j}") for j in range(full[0].shape[1])]
    prices = [m[np.ix_(rows, cols)] for m in full]
    root = deepen_mod._BeliefRoot(
        0, [names_r[i] for i in rows], [names_c[j] for j in cols], [None] * len(full), w,
        prices, solve_bayesian(prices, w),
    )

    def fill(r, c):  # noqa: ANN001, ANN202
        raise AssertionError("every cell is known; nothing should be filled")

    oracle = deepen_mod._BeliefOracle(root, (names_r, names_c), fill, swap=swap)
    for i, a in enumerate(names_r):
        for j, b in enumerate(names_c):
            oracle.known[(a.to_choice(), b.to_choice())] = np.array([m[i, j] for m in full])
    meter = deepen_mod._Meter(None)
    added = []
    for _ in range(1_000):
        before = (len(root.own), len(root.other))
        oracle.probe(meter)
        if not oracle.widen(meter):
            break
        added.append("row" if len(root.own) > before[0] else "col")
    else:
        raise AssertionError("the oracle never stopped")
    return root, oracle, meter, added


@pytest.mark.parametrize("swap", [False, True])
def test_the_bayesian_oracle_reaches_the_whole_games_value(swap) -> None:  # noqa: ANN001
    """When nothing outside gains, the root's answer is an equilibrium of the whole
    Bayesian game: no row beats it against the completions' replies, and no completion's
    column beats what its reply earns against the row strategy."""
    rng = np.random.default_rng(294)
    grew = 0
    for _ in range(40):
        k = int(rng.integers(2, 5))
        full = [rng.random((7, 6)) for _ in range(k)]
        w = rng.random(k) + 0.1
        w = w / w.sum()
        root, oracle, meter, added = _grown_by_belief_oracle(full, w, [0, 1], [0], swap=swap)
        want = _bayes_value(full, w)
        assert root.equilibrium.value == pytest.approx(want, abs=1e-6)
        x = np.zeros(full[0].shape[0])
        for n, a in enumerate(root.own):
            x[int(a.name[1:])] = root.equilibrium.row_strategy[n]
        ys = []
        for y_k in root.equilibrium.col_strategies:
            y = np.zeros(full[0].shape[1])
            for n, b in enumerate(root.other):
                y[int(b.name[1:])] = y_k[n]
            ys.append(y)
        guarantee = sum(w[q] * (x @ full[q]).min() for q in range(k))
        best_row = max(sum(w[q] * (full[q] @ ys[q]) for q in range(k)))
        assert guarantee >= want - 1e-6 and best_row <= want + 1e-6
        assert meter.fills == 0 and oracle.probed == 0 and not oracle.stalled
        assert oracle.widened == len(added)
        if swap:
            assert len(root.own) + len(root.other) == 3 + oracle.widened - oracle.swapped
            assert oracle.swapped == len(oracle.left)
        grew += oracle.widened
    assert grew > 40, "the oracle hardly ever added anything"


def test_with_one_completion_it_adds_what_the_open_oracle_adds() -> None:
    """K = 1, weight 1: the Bayesian gains are the open ones, so the same actions join in
    the same order and the value is the same (the null control of the generalisation)."""
    rng = np.random.default_rng(2941)
    compared = 0
    for _ in range(40):
        full = rng.random((7, 6))
        open_root, _o, _m, open_added = _grown_by_oracle(full, [0, 1], [0])
        root, _b, _m2, added = _grown_by_belief_oracle([full], np.ones(1), [0, 1], [0])
        assert added == open_added
        assert [a.name for a in root.own] == [a.name for a in open_root.rows]
        assert [b.name for b in root.other] == [b.name for b in open_root.cols]
        assert root.equilibrium.value == pytest.approx(open_root.equilibrium.value, abs=1e-7)
        compared += len(added)
    assert compared > 40


def test_a_column_leaves_only_when_no_completion_plays_it() -> None:
    a, b = _Act("a"), _Act("b")
    x, y, z = _Act("x"), _Act("y"), _Act("z")
    # Completion 0 answers with x (y and z lose); completion 1 with y (x and z lose).
    # z is played by neither: the only column a swap may push out.
    first = np.array([[0.1, 0.9, 0.95], [0.2, 0.8, 0.9]])
    second = np.array([[0.9, 0.1, 0.95], [0.8, 0.2, 0.9]])
    w = np.array([0.5, 0.5])
    root = deepen_mod._BeliefRoot(0, [a, b], [x, y, z], [None, None], w, [first, second],
                                  solve_bayesian([first, second], w))
    oracle = deepen_mod._BeliefOracle(root, ([], []), lambda r, c: [], swap=True)
    oracle._swap_out(1, added="x")
    assert [c.name for c in root.other] == ["x", "y"] and oracle.swapped == 1
    assert all(p.shape == (2, 2) for p in root.prices)
    oracle._swap_out(1, added="x")
    assert [c.name for c in root.other] == ["x", "y"] and oracle.swapped == 1


def test_a_cells_priority_carries_its_completions_weight() -> None:
    """Two completions with the same matrix: every cell's priority is its weight times
    the one ``bern/gap``, so the likelier completion's cells come first."""
    a, b, x, y = _Act("a"), _Act("b"), _Act("x"), _Act("y")
    m = np.array([[0.6, 0.3], [0.2, 0.7]])
    w = np.array([0.25, 0.75])
    for side, prices in ((0, [m, m]), (1, [-m.T, -m.T])):
        root = deepen_mod._BeliefRoot(side, [a, b], [x, y], [None, None], w, prices,
                                      solve_bayesian(prices, w))
        scores = root.scores()
        assert scores.shape == (2, 2, 2)
        assert np.allclose(scores[1], 3.0 * scores[0]) and scores[0].max() > 0
        # bern/gap reads side 0's value whichever side the root belongs to.
        eq = root.equilibrium
        won = m if side == 0 else m.T
        gap = eq.row_ev_loss[:, None] + eq.col_ev_loss[0][None, :]
        assert np.allclose(scores[0], 0.25 * won * (1 - won) / (gap + deepen_mod.GAP_FLOOR))


# --------------------------------------------------------------------------------------
# Real hidden positions (the port)


def _hidden(roster, pos):  # noqa: ANN001, ANN202
    """Each side's completions, only the actives seen: both benches hidden."""
    sheet = list(roster.sets)[:6]
    return {side: completions(roster.reg, pos, side, sheet) for side in (0, 1)}


def _menus(reg, pos, limit=5):  # noqa: ANN001, ANN202
    return narrow(reg, pos, 0, limit=limit).actions, narrow(reg, pos, 1, limit=limit).actions


@pytest.mark.parametrize("side", [0, 1])
def test_one_step_is_the_determinized_cell_of_the_best_weighted_signal(roster, side) -> None:  # noqa: ANN001
    reg = roster.reg
    checked = 0
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos)
        spreads = _hidden(roster, pos)
        items = spreads[1 - side]
        assert len(items) > 1 and not items[0].exact
        plain = belief_solve(reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF},
                             budget=Budget.matrix(), sides=(side,))
        trace: list = []
        got = belief_solve(reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF},
                           budget=Budget.matrix(), sides=(side,),
                           deepen={side: {"cells": 1, "trace": trace}})
        report = got[side].deepened
        root = trace[0]
        steps = trace[1:]
        assert len(steps) == 1 and report.classes == len(items)
        _node, cell, took = steps[0]
        # The first cell is the depth-1 answer's highest bern/gap times its weight; the
        # signal reads side 0's value (1 - it, for side 1's own game).
        depth1 = [m if side == 0 else -m.T for m in _matrices(reg, pos, ours, theirs, spreads, side)]
        eq = solve_bayesian(depth1, np.array([item.weight for item in items]))
        signal = np.stack([
            root.w[k] * won * (1 - won)
            / (eq.row_ev_loss[:, None] + eq.col_ev_loss[k][None, :] + deepen_mod.GAP_FLOOR)
            for k, won in enumerate(m if side == 0 else -m for m in depth1)
        ])
        assert cell == tuple(int(v) for v in np.unravel_index(int(np.argmax(signal)), signal.shape))
        # Without a deepening the answer is the plain one, to the bit.
        assert got[side].deepened is not None and plain[side].deepened is None
        if not took:
            continue
        k, i, j = cell
        own, other = (ours, theirs) if side == 0 else (theirs, ours)
        pair = (own[i], other[j]) if side == 0 else (other[j], own[i])
        want, _notes, _solved = _refined_value(
            reg, items[k].position, pair[0], pair[1], LEAF, budget=Budget.matrix(),
            sub_limit=DEFAULT_SUB_LIMIT, sub_branches=DEFAULT_SUB_BRANCHES,
        )
        assert root.prices[k][i, j] == (want if side == 0 else -want)
        for q, (now, before) in enumerate(zip(root.prices, depth1, strict=True)):
            moved = {tuple(int(v) for v in at) for at in np.argwhere(now != before)}
            assert moved <= ({(i, j)} if q == k else set())
        assert report.expanded == 1 and report.depth == 2
        assert got[side].value == pytest.approx(
            float(solve_bayesian(root.prices, root.w).value), abs=1e-9
        )
        checked += 1
    assert checked >= 2, "no cell was refined, so nothing was checked"


def _matrices(reg, pos, ours, theirs, spreads, side):  # noqa: ANN001, ANN202
    from pokeuraou.beliefnode import belief_payoffs

    node = belief_payoffs(reg, pos, list(ours), list(theirs), LEAF, budget=Budget.matrix(),
                          spreads={1 - side: spreads[1 - side]})
    return node.matrices[side]


def test_the_bayesian_tree_holds_its_values(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    deepest = 0
    for pos in _played(roster)[:2]:
        ours, theirs = _menus(reg, pos, limit=4)
        spreads = _hidden(roster, pos)
        for side in (0, 1):
            trace: list = []
            got = belief_solve(reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF},
                               budget=Budget.matrix(), sides=(side,),
                               deepen={side: {"cells": 600, "trace": trace}})
            root = trace[0]
            depth1 = [m if side == 0 else -m.T
                      for m in _matrices(reg, pos, ours, theirs, spreads, side)]
            for (k, i, j), branches in root.children.items():
                won = deepen_mod._cell_value(branches)
                assert root.prices[k][i, j] == (won if side == 0 else -won)
                for _w, child in branches:
                    if isinstance(child, deepen_mod._Node):
                        assert child.parent == (root, (k, i, j))
            for k, (now, before) in enumerate(zip(root.prices, depth1, strict=True)):
                untouched = np.ones(now.shape, dtype=bool)
                for q, i, j in root.children:
                    if q == k:
                        untouched[i, j] = False
                assert np.array_equal(now[untouched], before[untouched])
            eq = solve_bayesian(root.prices, root.w)
            assert got[side].value == pytest.approx(float(eq.value), abs=1e-9)
            assert np.array_equal(got[side].strategy, root.equilibrium.row_strategy)
            report = got[side].deepened
            # Every refined cell, at the root and below it, is one it reports.
            assert report.expanded == len(root.children) + sum(
                len(node.children)
                for kept in root.children.values()
                for _w, child in kept if isinstance(child, deepen_mod._Node)
                for node in [child, *_below(child)]
            )
            deepest = max(deepest, report.depth)
            # Below the root every node is an open game at its own equilibrium.
            for kept in root.children.values():
                for _w, child in kept:
                    if isinstance(child, deepen_mod._Node):
                        for node in [child, *_below(child)]:
                            assert node.equilibrium.value == solve(node.payoff).value
    assert deepest >= 3, f"600 cells on a 4x4 Bayesian root never passed depth 2 ({deepest})"


def _below(node) -> list:  # noqa: ANN001
    out = []
    for kept in node.children.values():
        for _w, child in kept:
            if isinstance(child, deepen_mod._Node):
                out.extend([child, *_below(child)])
    return out


@pytest.mark.parametrize("swap", [False, True])
def test_the_bayesian_oracle_widens_with_outside_actions(roster, swap, monkeypatch) -> None:  # noqa: ANN001
    from pokeuraou import beliefnode

    reg = roster.reg
    widened = 0
    real = beliefnode.belief_payoffs
    filled: list[int] = []

    def counted(reg_, position, ours_, theirs_, evaluate, **kw):  # noqa: ANN001, ANN202
        node = real(reg_, position, ours_, theirs_, evaluate, **kw)
        filled.append(len(ours_) * len(theirs_) * sum(len(m) for m in node.matrices.values()))
        return node

    monkeypatch.setattr(beliefnode, "belief_payoffs", counted)
    for pos in _played(roster)[:3]:
        ours, theirs = _menus(reg, pos, limit=3)
        wide = _menus(reg, pos, limit=8)
        spreads = _hidden(roster, pos)
        for side in (0, 1):
            how = {"cells": 300, "outside": wide, "swap": swap}
            filled.clear()
            got = belief_solve(reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF},
                               budget=Budget.matrix(), sides=(side,), deepen={side: how})
            # Every cell the oracle filled -- probes and the rest of an added line, in
            # every completion -- is charged: the first fill is the node's own.
            assert sum(filled[1:]) == got[side].deepened.probed
            assert got[side].deepened.fills >= len(filled) - 1
            alone = belief_solve(reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF},
                                 budget=Budget.matrix(), sides=(side,),
                                 deepen={side: {**how, "reading": "breadth"}})[side].deepened
            assert alone.expanded == 0 and alone.probed > 0 and alone.cells == alone.probed
            again = belief_solve(reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF},
                                 budget=Budget.matrix(), sides=(side,), deepen={side: dict(how)})
            answer, report = got[side], got[side].deepened
            assert report == again[side].deepened
            assert np.array_equal(answer.strategy, again[side].strategy)
            # Every action on the grown menus is a candidate or was on the menu, once.
            for menu, given, candidates in ((answer.ours, ours, wide[0]),
                                            (answer.theirs, theirs, wide[1])):
                names = [a.to_choice() for a in menu]
                assert len(set(names)) == len(names)
                assert set(names) <= {a.to_choice() for a in (*given, *candidates)}
            mine = answer.ours if side == 0 else answer.theirs
            assert len(answer.strategy) == len(mine)
            assert all(len(y) == len(answer.theirs if side == 0 else answer.ours)
                       for y in answer.replies)
            if not swap:
                assert len(answer.ours) + len(answer.theirs) == len(ours) + len(theirs) + report.widened
            else:
                assert len(answer.ours) + len(answer.theirs) == (
                    len(ours) + len(theirs) + report.widened - report.swapped
                )
            assert report.oracle and report.probed > 0 and report.swap == swap
            widened += report.widened
    assert widened > 0


@pytest.mark.parametrize("label", ["m60h", "m60s6h", "b60sallh"])
def test_in_a_game_it_fires_where_a_bench_is_hidden(setup, label) -> None:  # noqa: ANN001, F811
    game = _hidden_game(setup, label)
    assert game.deepen == [label, label]
    hidden = both = 0
    for d in game.decisions:
        if d.kind != "move":
            assert d.deepened is None
            continue
        assert d.deepened is not None
        pos = Position.from_json(d.position)
        shown = all(len(d.shown[i]) == len(pos.sides[i].pokemon) for i in (0, 1))
        if shown:
            # Nothing hidden: the open game's deepening, one tree for both sides.
            assert "classes" not in d.deepened[0] and d.deepened[1] == d.deepened[0]
            both += 1
        else:
            for side in (0, 1):
                report = d.deepened[side]
                assert report["classes"] >= 1
                if label.startswith("b"):
                    assert report["expanded"] == 0 and report["depth"] == 1
                else:
                    assert report["cells"] > 0
                assert ("swapped" in report) == ("s" in label[1:])
            hidden += 1
        assert len(d.own_policy) == len(d.own_actions) and len(d.foe_policy) == len(d.foe_actions)
        assert d.own_chosen in d.own_actions and d.foe_chosen in d.foe_actions
    assert hidden > 0
    if "s" in label[1:]:
        # A swap changed some recorded menu away from the width-3 menu it started from.
        reg = setup[0]
        moved = 0
        for d in game.decisions:
            if d.kind == "move" and d.deepened and d.deepened[0].get("widened") and (
                "classes" in d.deepened[0]
            ):
                pos = Position.from_json(d.position)
                start = [a.to_choice() for a in narrow(reg, pos, 0, limit=3).actions]
                moved += d.own_actions != start
        assert moved > 0
    again = _hidden_game(setup, label)
    assert _payload(again) == _payload(game)


@pytest.mark.parametrize("label", ["m60", "m60s6", "b60sall"])
def test_where_nothing_is_hidden_h_plays_the_same_game(setup, label) -> None:  # noqa: ANN001, F811
    """From a position where each side has shown all four (every bench member one HP
    down, so the position itself says it has been out), every node is one with no bench
    hidden: the label with h plays IKA-293's game, byte for byte but for the label."""
    reg, sheet = setup
    start = selfplay.position_from_sets(
        reg, sheet[:4], sheet[2:6], rng=np.random.default_rng(294)
    )
    for side in start.sides:
        for mon in side.pokemon:
            if mon.active_index is None:
                mon.hp = mon.maxhp - 1

    def game(which):  # noqa: ANN001, ANN202
        return selfplay.play_game(
            reg, np.random.default_rng(294), sheet[:4], sheet[2:6], "test",
            search_limit=3, max_turns=25, sheets=(sheet, sheet), deepen=which, start=start,
        )

    plain, hidden = _payload(game(label)), _payload(game(label + "h"))
    assert plain["deepen"] == [label, label] and hidden["deepen"] == [label + "h"] * 2
    hidden["deepen"] = plain["deepen"]
    assert hidden == plain
    fired = [d for d in plain["decisions"] if d.get("deepened")]
    assert fired and all("classes" not in d["deepened"][0] for d in fired)


def test_without_h_a_hidden_node_stays_at_depth_one(setup) -> None:  # noqa: ANN001, F811
    game = _hidden_game(setup, "m60s6")
    for d in game.decisions:
        if d.deepened is not None:
            assert "classes" not in d.deepened[0]


def test_the_hidden_deepening_follows_its_arm_into_either_seat(pool) -> None:  # noqa: ANN001, F811
    solver = SolvedSelections(pool.reg, pool.teams, _stub)

    def arm(label: str) -> PoolArm:
        return PoolArm(name="arm", evaluate=_stub, solver=solver, limit=2,
                       rank_by_leaf=False, deepen=label)

    tested, other = arm("m30sallh"), arm("none")
    for which in (0, 1):
        record, sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=33, game_index=0, which=which,
            hide_bench=True, max_turns=3,
        )
        assert record.deepen[which] == "m30sallh" and record.deepen[1 - which] == "none"
        assert sides["deepens"][which] == "m30sallh"
        moves = [d for d in record.decisions if d.kind == "move"]
        assert moves
        for d in moves:
            assert d.deepened[which] is not None and d.deepened[1 - which] is None
            assert "classes" in d.deepened[which]
