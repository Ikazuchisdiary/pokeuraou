"""IKA-196: which point of the optimal set a side plays.

What these tests hold:

- **the label**: ``lp``, ``unif``, ``ment<D>`` and ``qre<T>`` parse, anything else stops;
  ``lp`` hands back the LP's very strategy;
- **an exact selection keeps the value**: ``unif`` and ``ment0`` guarantee what the LP
  guarantees, to 1e-9, in Bayesian games of 1 to 4 completions. ``ment<D>`` gives up at
  most ``D``;
- **``unif`` is what it says**: no strategy of the optimal set takes more from a uniform
  reply, and on degenerate games it moves off the LP's vertex and takes strictly more (the
  positive control that the second LP is not the first one again);
- **the QRE**: its first-order conditions hold (each side's soft best reply to the other),
  it gives up at most T (ln m + ln n), and that goes to zero with T;
- **the column player** is read off the negated transpose;
- **in a game**: a label follows its arm into either seat, is recorded only when it
  differs from ``lp``, and ``lp`` named is the default game to the byte. A label that
  cannot be honoured (a deepening) stops.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import eqselect, selfplay
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.eqselect import (
    guarantee,
    parse_eq_select,
    qre_row,
    reselect,
    reselect_bayesian,
    reselected,
)
from pokeuraou.equilibrium import solve, solve_bayesian
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.regulation import load_regulation
from pokeuraou.search import belief_solve
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from .test_beliefnode import _Leaf
from .test_poolplay import _stub, _variants, _write_pool


def _games(seed: int, count: int, size: int, grid: float):  # noqa: ANN202
    rng = np.random.default_rng(seed)
    for trial in range(count):
        k = 1 + trial % 4
        mats = [np.round(rng.random((size, size)) / grid) * grid for _ in range(k)]
        yield mats, rng.random(k) + 0.1


def test_the_label_parses_and_a_bad_one_stops() -> None:
    assert parse_eq_select("lp").kind == "lp"
    assert parse_eq_select("unif").kind == "unif"
    assert parse_eq_select("ment0").param == 0.0
    assert parse_eq_select("ment0.001").param == pytest.approx(0.001)
    assert parse_eq_select("qre0.005").param == pytest.approx(0.005)
    assert parse_eq_select("qre1e-3").param == pytest.approx(0.001)
    for bad in ("", "LP", "unif1", "qre", "qre0", "ment", "maxent", "qre-1", "lp0"):
        with pytest.raises(ValueError, match="equilibrium selection"):
            parse_eq_select(bad)


def test_lp_hands_back_the_lp_strategy() -> None:
    a = np.array([[0.2, 0.7], [0.6, 0.3]])
    eq = solve(a)
    assert reselect(a, eq.value, eq.row_strategy, "lp") is eq.row_strategy
    assert reselected(a, eq, ("lp", "lp")) is eq


@pytest.mark.parametrize("label", ["unif", "ment0"])
def test_an_exact_selection_keeps_the_value(label: str) -> None:
    for mats, w in _games(196, 60, 6, 0.25):
        eq = solve_bayesian(mats, w)
        x = reselect_bayesian(mats, w, eq.value, eq.row_strategy, label)
        assert x.min() >= 0.0
        assert x.sum() == pytest.approx(1.0, abs=1e-12)
        assert guarantee(mats, w, x) >= eq.value - 1e-9


def test_a_slack_gives_up_at_most_the_slack() -> None:
    for mats, w in _games(197, 30, 6, 0.25):
        eq = solve_bayesian(mats, w)
        x = reselect_bayesian(mats, w, eq.value, eq.row_strategy, "ment0.01")
        assert guarantee(mats, w, x) >= eq.value - 0.01 - 1e-6


def _uniform_take(mats, w, x) -> float:  # noqa: ANN001
    w = np.asarray(w) / np.sum(w)
    return float(sum(w[k] * (x @ m).mean() for k, m in enumerate(mats)))


def test_unif_takes_the_most_from_a_uniform_reply_and_moves() -> None:
    moved = 0
    for mats, w in _games(198, 80, 4, 0.5):
        eq = solve_bayesian(mats, w)
        x = reselect_bayesian(mats, w, eq.value, eq.row_strategy, "unif")
        lp_take = _uniform_take(mats, w, eq.row_strategy)
        take = _uniform_take(mats, w, x)
        assert take >= lp_take - 1e-9
        # Every vertex of the optimal set is a candidate, and the LP's is one: none takes
        # more than the unif answer.
        if take > lp_take + 1e-6:
            moved += 1
            assert 0.5 * np.abs(x - eq.row_strategy).sum() > 1e-6
    # The positive control: on games this tied the second LP finds a better point.
    assert moved >= 3


def _qre_residual(mats, w, tau, x, replies) -> float:  # noqa: ANN001
    w = np.asarray(w) / np.sum(w)
    u = sum(w[k] * (m @ replies[k]) for k, m in enumerate(mats))
    want = np.exp((u - u.max()) / tau)
    out = float(np.abs(want / want.sum() - x).max())
    for k, m in enumerate(mats):
        c = -(x @ m) / tau
        e = np.exp(c - c.max())
        out = max(out, float(np.abs(e / e.sum() - replies[k]).max()))
    return out


def test_the_qre_is_its_own_soft_best_reply_and_near_the_value() -> None:
    for tau in (0.2, 0.02, 0.002):
        for mats, w in _games(199, 20, 8, 0.1):
            eq = solve_bayesian(mats, w)
            x, replies = qre_row(mats, np.asarray(w) / np.sum(w), tau)
            assert _qre_residual(mats, w, tau, x, replies) < 1e-6
            m, n = mats[0].shape
            assert guarantee(mats, w, x) >= eq.value - tau * (np.log(m) + np.log(n)) - 1e-9


def test_a_large_temperature_is_near_uniform() -> None:
    """The knob's own control: hot, the QRE forgets the matrix."""
    a = np.array([[0.9, 0.1, 0.5], [0.2, 0.8, 0.4], [0.5, 0.5, 0.5]])
    eq = solve(a)
    x = reselect(a, eq.value, eq.row_strategy, "qre100")
    assert np.abs(x - 1 / 3).max() < 0.01


def test_the_column_player_is_the_negated_transpose() -> None:
    a = np.array([[0.5, 0.5, 0.9], [0.5, 0.5, 0.2], [0.1, 0.6, 0.6]])
    eq = solve(a)
    got = reselected(a, eq, ("qre0.05", "qre0.05"))
    assert np.allclose(got.row_strategy, reselect(a, eq.value, eq.row_strategy, "qre0.05"))
    assert np.allclose(got.col_strategy, reselect(-a.T, -eq.value, eq.col_strategy, "qre0.05"))
    assert got.value == eq.value
    assert np.allclose(got.row_ev, a @ got.col_strategy)


# ----------------------------------------------------------------------------- in a game


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _arm(label: str, solver) -> PoolArm:  # noqa: ANN001
    return PoolArm(name="arm", evaluate=_stub, solver=solver, limit=3, rank_by_leaf=False,
                   eq_select=label)


def test_the_label_follows_its_arm_into_either_seat(pool) -> None:  # noqa: ANN001
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    tested, other = _arm("qre0.05", solver), _arm("lp", solver)
    for which in (0, 1):
        before = dict(eqselect.STATS.get("qre0.05", {"calls": 0, "moved": 0}))
        record, sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=196, game_index=0, which=which,
            hide_bench=True, max_turns=3,
        )
        assert record.eq_select[which] == "qre0.05"
        assert record.eq_select[1 - which] == "lp"
        assert sides["eq_selects"][which] == "qre0.05"
        payload = record.to_json(objective="value:arm", search_limit=(3, 3))
        assert payload["eqSelect"] == record.eq_select
        # The positive control: the label's solves ran, and moved the strategy.
        after = eqselect.STATS["qre0.05"]
        assert after["calls"] > before["calls"]
        assert after["moved"] > before["moved"]

    same = _arm("lp", solver)
    record, _ = pool_match_game(
        pool.reg, pool, (same, same), seed=196, game_index=0, which=0,
        hide_bench=True, max_turns=3,
    )
    assert "eqSelect" not in record.to_json(objective="value:arm", search_limit=(3, 3))


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    return reg, sheet, position_from_sets(reg, sheet[:4], sheet[:4])


def _game(setup, **kwargs):  # noqa: ANN001, ANN003, ANN202
    reg, sheet, _position = setup
    return selfplay.play_game(
        reg, np.random.default_rng(196), sheet[:4], sheet[:4], "test",
        search_limit=3, max_turns=3, sheets=(sheet, sheet), **kwargs,
    )


def _payload(record) -> dict:  # noqa: ANN001
    payload = record.to_json(objective="hp-share", search_limit=3)
    payload.pop("searchSeconds", None)
    payload.pop("engine", None)
    return payload


def test_the_move_node_plays_the_selected_point(setup) -> None:  # noqa: ANN001
    """`belief_solve`'s strategy per side is the label's point of that side's own Bayesian
    game, and its value is the LP's."""
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=6).actions
    theirs = narrow(reg, position, 1, limit=6).actions
    leaf = _Leaf(Encoder(reg))
    spreads = {side: completions(reg, position, side, sheet) for side in (0, 1)}
    kwargs = {"budget": Budget.matrix()}
    plain = belief_solve(reg, position, ours, theirs, spreads, {0: leaf, 1: leaf}, **kwargs)
    got = belief_solve(reg, position, ours, theirs, spreads, {0: leaf, 1: leaf},
                       select=("lp", "qre0.05"), **kwargs)
    assert np.array_equal(got[0].strategy, plain[0].strategy)
    assert got[1].value == plain[1].value
    row, col, built, weights = got[1].node_payoff
    game = [-m.T for m in built]
    want = reselect_bayesian(game, weights, plain[1].value, plain[1].strategy, "qre0.05")
    assert np.array_equal(got[1].strategy, want)
    # The control: the label moved side 1's strategy.
    assert 0.5 * np.abs(got[1].strategy - plain[1].strategy).sum() > 1e-3
    with pytest.raises(ValueError, match="IKA-196"):
        belief_solve(reg, position, ours, theirs, spreads, {0: leaf, 1: leaf},
                     select=("unif", "lp"), depth=(2, 1), **kwargs)


def test_lp_named_is_the_default_game(setup) -> None:  # noqa: ANN001
    assert _payload(_game(setup)) == _payload(_game(setup, eq_select="lp"))
    # The comparison's own control: another label plays another game.
    assert _payload(_game(setup)) != _payload(_game(setup, eq_select="qre0.2"))


def test_a_label_that_cannot_be_honoured_stops(setup) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="IKA-196"):
        _game(setup, eq_select=("unif", "lp"), deepen=("m8", "none"))
    with pytest.raises(ValueError, match="equilibrium selection"):
        _game(setup, eq_select="nash")
