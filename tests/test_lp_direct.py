"""`equilibrium._lp` hands HiGHS the model `linprog(method="highs")` does (IKA-265).

It skips linprog's wrapping, which was two thirds of an LP's time at these sizes, and must
return the same vertex to the bit: a different vertex is a different game. The positive
control is a change that is *not* the same call -- presolve off -- which these games do
tell apart, so a test that agreed with everything would fail there.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from pokeuraou import equilibrium
from pokeuraou.equilibrium import _lp


def _bayesian_row_lp(matrices: list[np.ndarray], weights: np.ndarray):  # noqa: ANN202
    """`_bayesian_maximin`'s arrays, as it builds them."""
    m, k = matrices[0].shape[0], len(matrices)
    columns = [mat.shape[1] for mat in matrices]
    c = np.zeros(m + k)
    c[m:] = -1.0
    a_ub = np.zeros((sum(columns), m + k))
    row = 0
    for index, mat in enumerate(matrices):
        n = mat.shape[1]
        a_ub[row : row + n, :m] = -weights[index] * mat.T
        a_ub[row : row + n, m + index] = 1.0
        row += n
    a_eq = np.zeros((1, m + k))
    a_eq[0, :m] = 1.0
    return c, a_ub, np.zeros(sum(columns)), a_eq, np.array([1.0]), m


def _games(count: int, seed: int) -> list[tuple[list[np.ndarray], np.ndarray]]:
    rng = np.random.default_rng(seed)
    games = []
    for index in range(count):
        m = int(rng.integers(1, 13))
        k = int(rng.choice([1, 3, 6]))
        mats = [rng.random((m, int(rng.integers(1, 13)))) for _ in range(k)]
        if index % 3 == 0:  # ties and degenerate vertices, as win probabilities often are
            mats = [np.round(mat * 4) / 4 for mat in mats]
        weights = rng.random(k)
        games.append((mats, weights / weights.sum()))
    return games


def _linprog_x(problem, **options) -> np.ndarray:  # noqa: ANN001
    c, a_ub, b_ub, a_eq, b_eq, free_from = problem
    bounds = [(0.0, None)] * free_from + [(None, None)] * (len(c) - free_from)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                  method="highs", options=options or None)
    assert res.success
    return res.x


def test_the_direct_call_returns_linprogs_vertex_to_the_bit() -> None:
    problems = [_bayesian_row_lp(*game) for game in _games(150, 265)]
    for problem in problems:
        c, a_ub, b_ub, a_eq, b_eq, free_from = problem
        got = _lp(c, a_ub, b_ub, a_eq, b_eq, free_from=free_from)
        assert got.success
        assert got.x.tobytes() == _linprog_x(problem).tobytes()


def test_a_call_that_is_not_the_same_is_told_apart() -> None:
    """Positive control: presolve off is another path through HiGHS, and these games show it."""
    problems = [_bayesian_row_lp(*game) for game in _games(150, 265)]
    differ = sum(
        _linprog_x(problem).tobytes() != _linprog_x(problem, presolve=False).tobytes()
        for problem in problems
    )
    assert differ > 0


def test_solve_bayesian_is_linprogs_with_the_direct_call_or_without(monkeypatch) -> None:  # noqa: ANN001
    games = _games(60, 7)
    direct = [equilibrium.solve_bayesian(mats, w) for mats, w in games]
    monkeypatch.setattr(equilibrium, "_highs_core", None)
    wrapped = [equilibrium.solve_bayesian(mats, w) for mats, w in games]
    for a, b in zip(direct, wrapped, strict=True):
        assert a.row_strategy.tobytes() == b.row_strategy.tobytes()
        pairs = zip(a.col_strategies, b.col_strategies, strict=True)
        assert all(x.tobytes() == y.tobytes() for x, y in pairs)
        assert a.value == b.value
        assert a.duality_gap == b.duality_gap


def test_an_infeasible_lp_is_a_failure_not_an_answer() -> None:
    # x >= 0 with x <= -1: HiGHS says infeasible, and the caller must see success False.
    got = _lp(np.array([1.0]), np.array([[1.0]]), np.array([-1.0]), np.zeros((0, 1)),
              np.zeros(0), free_from=1)
    assert not got.success
    assert got.x is None


def test_a_failed_solve_raises_as_before(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        equilibrium, "_lp", lambda *args, **kwargs: equilibrium._Solved(False, None, "no")
    )
    with pytest.raises(equilibrium.EquilibriumError):
        equilibrium.solve_bayesian([np.eye(2)], np.array([1.0]))
