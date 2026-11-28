"""Equilibria compared as equilibria, not as bits (IKA-470).

The port's HiGHS runs without presolve and scaling (`rust/src/lp.rs`), so on a degenerate game
it can end on another optimal vertex than scipy's options do: the same game value, another
strategy. A test that holds the port to `equilibrium.solve` therefore asks two things instead:
the values agree within `VALUE_TOL`, and the port's strategies are an equilibrium of the game
-- against the other side's best reply the row strategy earns at least the value, and the
column strategy concedes at most the value, each within `GUARD_TOL`.
"""

from __future__ import annotations

import numpy as np

#: The values of the two option sets differ by at most 9.87e-8 on the 5,578 LPs of
#: records/IKA-470.md; a test passes at ten times that.
VALUE_TOL = 1e-6
#: How far a strategy may fall short of the value against the other side's best reply.
GUARD_TOL = 1e-6


def is_distribution(p: np.ndarray) -> bool:
    p = np.asarray(p, dtype=np.float64)
    return bool(np.all(p >= 0.0) and abs(p.sum() - 1.0) <= 1e-9)


def row_guarantee(matrix: np.ndarray, x: np.ndarray) -> float:
    """What the row strategy earns against the column player's best reply."""
    return float(np.min(np.asarray(x) @ matrix))


def col_concession(matrix: np.ndarray, y: np.ndarray) -> float:
    """What the column strategy concedes to the row player's best reply."""
    return float(np.max(matrix @ np.asarray(y)))


def is_equilibrium(matrix: np.ndarray, x: np.ndarray, y: np.ndarray, value: float,
                   tol: float = GUARD_TOL) -> bool:
    return (is_distribution(x) and is_distribution(y)
            and row_guarantee(matrix, x) >= value - tol
            and col_concession(matrix, y) <= value + tol)


def is_the_solve(matrix: np.ndarray, got, want, tol: float = VALUE_TOL) -> bool:  # noqa: ANN001
    """`got` (an `Equilibrium`) is an answer to the game `want = equilibrium.solve(matrix)` gave."""
    return (abs(got.value - want.value) <= tol
            and is_equilibrium(matrix, got.row_strategy, got.col_strategy, want.value)
            and np.allclose(got.row_ev, matrix @ got.col_strategy, atol=1e-9)
            and np.allclose(got.col_ev, got.row_strategy @ matrix, atol=1e-9))


def bayesian_row_guarantee(mats: list[np.ndarray], weights: np.ndarray, x: np.ndarray) -> float:
    """The row strategy against each class's best reply, weighted."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(sum(w[k] * np.min(np.asarray(x) @ mats[k]) for k in range(len(mats))))


def bayesian_col_concession(mats: list[np.ndarray], weights: np.ndarray,
                            ys: list[np.ndarray]) -> float:
    """The per-class column strategies against the row player's best reply."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(np.max(sum(w[k] * (mats[k] @ np.asarray(ys[k])) for k in range(len(mats)))))


def is_bayesian_equilibrium(mats: list[np.ndarray], weights: np.ndarray, x: np.ndarray,
                            ys: list[np.ndarray], value: float, tol: float = GUARD_TOL) -> bool:
    return (is_distribution(x) and all(is_distribution(y) for y in ys)
            and bayesian_row_guarantee(mats, weights, x) >= value - tol
            and bayesian_col_concession(mats, weights, ys) <= value + tol)
