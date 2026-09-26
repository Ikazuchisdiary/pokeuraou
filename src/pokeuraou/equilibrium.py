"""Exact equilibria of two-player zero-sum matrix games, by linear programming.

Each turn of a doubles battle is a simultaneous-move game, so there is generally no single
best action -- the equilibrium is a mixed strategy. The frequencies this module returns
are what the tool reports, so they are computed exactly (HiGHS) rather than approximated
by search visit counts, whose relationship to the equilibrium would be unjustifiable.

Two games live here. :func:`solve` is the ordinary one, where both players know the
matrix. :func:`solve_bayesian` is the one this project needs: the opponent knows their own
SP spread and we do not, so they hold private information and can condition on it. Solving
the belief-averaged matrix with :func:`solve` instead would answer a different question --
what if the opponent also had to guess their own investment -- and would overstate our
value, because information cannot hurt the player who has it.

Convention: ``payoff[i, j]`` is the ROW player's payoff when row plays ``i`` and column
plays ``j``. Throughout this project that payoff is the row player's win probability, so
the equilibrium value is directly a win probability in [0, 1]. Adding a constant to every
entry shifts the value and leaves the strategies unchanged, so a constant-sum game
(``rowPayoff + colPayoff = 1``) is handled by the same code as a strictly zero-sum one.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from . import timing


class EquilibriumError(RuntimeError):
    pass


#: Cells (rows x columns, all classes) from which a game's two LPs -- the row player's and
#: the column player's, independent of each other -- are solved at once on two threads
#: (IKA-32 stage 2). HiGHS lets go of the GIL while it pivots, so a 64 x 64 x 4 Bayesian
#: game takes about half the wall time; each LP is the same model handed to the same
#: HiGHS, so the answer is the same to the bit. 0 is off (generation and the board: one
#: core a worker already). A small game stays on one thread: its LPs are mostly Python.
LP_PAIR_ENV = "POKEURAOU_LP_PAIR_CELLS"
_LP_PAIR = [int(os.environ.get(LP_PAIR_ENV, "0") or 0)]
_LP_POOL: list[ThreadPoolExecutor | None] = [None]
_LP_POOL_LOCK = threading.Lock()
#: How many games had their two LPs solved at once (the stage's positive control).
LP_PAIRS = [0]


def set_lp_pair(cells: int) -> None:
    """Solve a game's two LPs at once from `cells` cells up; 0 turns it off."""
    if cells < 0:
        raise ValueError(f"cells must be >= 0, not {cells}")
    _LP_PAIR[0] = int(cells)


def lp_pair() -> int:
    """`set_lp_pair`'s threshold in cells, 0 when off."""
    return _LP_PAIR[0]


def _both[A, B](cells: int, first: Callable[[], A], second: Callable[[], B]) -> tuple[A, B]:
    """``(first(), second())``, the second on a helper thread when the game is big enough.

    An error is the one the serial order meets first: `first`'s, else `second`'s.
    """
    threshold = _LP_PAIR[0]
    if threshold <= 0 or cells < threshold:
        return first(), second()
    with _LP_POOL_LOCK:
        if _LP_POOL[0] is None:
            _LP_POOL[0] = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lp-pair")
        pool = _LP_POOL[0]
    later = pool.submit(second)
    try:
        a = first()
    except BaseException:
        later.exception()  # wait: the helper's LP is not left running
        raise
    LP_PAIRS[0] += 1
    return a, later.result()


@dataclass(frozen=True, slots=True)
class Equilibrium:
    """Solved matrix game.

    Attributes:
        value: the game value -- the row player's win probability under equilibrium play.
        row_strategy: (m,) probabilities over the row player's actions.
        col_strategy: (n,) probabilities over the column player's actions.
        row_ev: (m,) EV of each row action against ``col_strategy``.
        col_ev: (n,) value of each column action against ``row_strategy``, still measured
            as the ROW player's payoff.
        row_ev_loss: (m,) ``value - row_ev``, clipped at 0. Zero on the equilibrium
            support; a positive number reads directly as how bad the action is.
        col_ev_loss: (n,) ``col_ev - value``, clipped at 0 -- the same quantity from the
            column player's point of view.
        duality_gap: |row LP value - column LP value|. Both LPs solve the same game, so
            this is a residual and should be ~1e-12; it is reported rather than assumed.
    """

    value: float
    row_strategy: np.ndarray
    col_strategy: np.ndarray
    row_ev: np.ndarray
    col_ev: np.ndarray
    row_ev_loss: np.ndarray
    col_ev_loss: np.ndarray
    duality_gap: float

    def row_support(self, threshold: float = 1e-6) -> np.ndarray:
        return np.flatnonzero(self.row_strategy > threshold)

    def col_support(self, threshold: float = 1e-6) -> np.ndarray:
        return np.flatnonzero(self.col_strategy > threshold)

    def max_exploitability(self) -> float:
        """Largest incentive either player has to deviate. Zero at a true equilibrium."""
        return float(max(self.row_ev.max() - self.value, self.value - self.col_ev.min(), 0.0))


def _clean(p: np.ndarray, eps: float) -> np.ndarray:
    """Removes solver noise: clip small negatives, drop dust, renormalise."""
    p = np.asarray(p, dtype=np.float64).copy()
    p[p < eps] = 0.0
    total = p.sum()
    if total <= 0:
        # Degenerate solve; fall back to uniform rather than returning garbage.
        return np.full(p.shape, 1.0 / p.shape[0])
    return p / total


try:  # scipy's own HiGHS binding, the one `linprog(method="highs")` hands the problem to
    import scipy.optimize._highspy._core as _highs_core
    from scipy.optimize._linprog_util import _check_result
except ImportError:  # pragma: no cover - a scipy that moved them; `linprog` it is
    _highs_core = None


@dataclass(frozen=True, slots=True)
class _Solved:
    success: bool
    x: np.ndarray | None
    message: str


_HIGHS_OPTIONS = None


def _highs_options():  # noqa: ANN202 - a HighsOptions
    """What `linprog(method="highs")` sets with its defaults, and nothing else."""
    global _HIGHS_OPTIONS  # noqa: PLW0603 - built once per process, read-only after
    if _HIGHS_OPTIONS is None:
        options = _highs_core.HighsOptions()
        options.presolve = "on"
        options.highs_debug_level = int(_highs_core.HighsDebugLevel.kHighsDebugLevelNone)
        options.log_to_console = False
        options.output_flag = False
        options.simplex_strategy = int(
            _highs_core.simplex_constants.SimplexStrategy.kSimplexStrategyDual
        )
        _HIGHS_OPTIONS = options
    return _HIGHS_OPTIONS


def _lp(  # noqa: PLR0913 - linprog's arguments, less the ones every caller leaves alone
    c: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    *,
    free_from: int,
) -> _Solved:
    """``linprog(..., method="highs")`` with bounds ``x >= 0`` below `free_from` and free
    from it on, without linprog's input checking and result dressing (IKA-265).

    The same HiGHS, handed the same model -- the same column-wise matrix (built the way
    `csc_array` builds it from the dense one: nonzeros, column by column), bounds, row
    bounds and options -- so it takes the same pivots and returns the same vertex to the
    bit; the test compares the two on random games and `records/IKA-265.md` on every LP of
    1,200 games. Success is linprog's: HiGHS optimal, then linprog's own residual check.
    At these sizes (up to 12 by 72) linprog's wrapping was two thirds of the time.
    """
    if _highs_core is None:  # pragma: no cover
        bounds = [(0.0, None)] * free_from + [(None, None)] * (len(c) - free_from)
        res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds,
                      method="highs")
        return _Solved(bool(res.success), res.x, str(res.message))
    h = _highs_core
    inf = h.kHighsInf
    a = np.vstack((a_ub, a_eq))
    rows, cols = a.shape
    columns = a.T
    nonzero = columns != 0
    start = np.zeros(cols + 1, dtype=np.int32)
    np.cumsum(nonzero.sum(axis=1), out=start[1:])
    lower = np.zeros(cols)
    lower[free_from:] = -inf
    upper = np.full(cols, inf)
    row_upper = np.concatenate((b_ub, b_eq))
    model = h.HighsLp()
    model.num_col_ = cols
    model.num_row_ = rows
    model.a_matrix_.num_col_ = cols
    model.a_matrix_.num_row_ = rows
    model.a_matrix_.format_ = h.MatrixFormat.kColwise
    model.col_cost_ = c
    model.col_lower_ = lower
    model.col_upper_ = upper
    model.row_lower_ = np.concatenate((np.full(len(b_ub), -inf), b_eq))
    model.row_upper_ = row_upper
    model.a_matrix_.start_ = start
    model.a_matrix_.index_ = np.nonzero(nonzero)[1].astype(np.int32)
    model.a_matrix_.value_ = columns[nonzero]
    highs = h._Highs()
    highs.passOptions(_highs_options())
    if highs.passModel(model) == h.HighsStatus.kError:
        return _Solved(False, None, "HiGHS refused the model")
    if highs.run() == h.HighsStatus.kError:
        return _Solved(False, None, highs.modelStatusToString(highs.getModelStatus()))
    status = highs.getModelStatus()
    if status != h.HighsModelStatus.kOptimal:
        return _Solved(False, None, highs.modelStatusToString(status))
    solution = highs.getSolution()
    x = np.array(solution.col_value)
    slack = row_upper - solution.row_value
    bounds = np.empty((cols, 2))
    bounds[:, 0] = 0.0
    bounds[free_from:, 0] = -np.inf
    bounds[:, 1] = np.inf
    checked, message = _check_result(
        x, highs.getInfo().objective_function_value, 0, slack[: len(b_ub)],
        slack[len(b_ub):], bounds, 1e-9, "optimal", None,
    )
    return _Solved(checked == 0, x, message)


def _maximin(payoff: np.ndarray) -> tuple[float, np.ndarray]:
    """Row player's LP: maximise v subject to (A^T x) >= v, sum x = 1, x >= 0."""
    m, n = payoff.shape
    # variables: x_0..x_{m-1}, v
    c = np.zeros(m + 1)
    c[-1] = -1.0  # maximise v

    # For each column j:  v - sum_i A[i,j] x_i <= 0
    a_ub = np.empty((n, m + 1))
    a_ub[:, :m] = -payoff.T
    a_ub[:, m] = 1.0
    b_ub = np.zeros(n)

    a_eq = np.zeros((1, m + 1))
    a_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # x >= 0, v free
    res = _lp(c, a_ub, b_ub, a_eq, b_eq, free_from=m)
    if not res.success:
        raise EquilibriumError(f"row LP failed: {res.message}")
    return float(res.x[m]), res.x[:m]


@timing.timed("lp")
def solve(payoff: np.ndarray, eps: float = 1e-9) -> Equilibrium:
    """Solves the zero-sum game with the given row-player payoff matrix.

    Both players' LPs are solved independently and their values compared, so a numerical
    problem surfaces as a non-zero ``duality_gap`` instead of silently biasing the
    reported frequencies.
    """
    a = np.asarray(payoff, dtype=np.float64)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"payoff must be a non-empty 2-D array, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError("payoff contains non-finite entries")
    m, n = a.shape

    # The column player minimises A, which is the row player of the game -A^T.
    (value_row, x_raw), (value_col_neg, y_raw) = _both(
        m * n, lambda: _maximin(a), lambda: _maximin(-a.T)
    )
    value_col = -value_col_neg

    x = _clean(x_raw, eps)
    y = _clean(y_raw, eps)
    value = 0.5 * (value_row + value_col)

    row_ev = a @ y
    col_ev = x @ a
    return Equilibrium(
        value=value,
        row_strategy=x,
        col_strategy=y,
        row_ev=row_ev,
        col_ev=col_ev,
        row_ev_loss=np.maximum(value - row_ev, 0.0),
        col_ev_loss=np.maximum(col_ev - value, 0.0),
        duality_gap=abs(value_row - value_col),
    )


@dataclass(frozen=True, slots=True)
class BayesianEquilibrium:
    """Solved zero-sum game where the column player knows their own private type.

    Attributes:
        value: the game value, in the same units as the payoff matrices.
        row_strategy: (m,) our probabilities. One strategy, because we do not know which
            type we face.
        col_strategies: one (n_k,) distribution per class -- what the opponent does when
            their spread *is* that class. These differ, which is the whole point.
        col_marginal: (n,) the frequencies we would observe if every class used the same
            action list: ``sum_k w_k y_k``. Only defined when the classes share an action
            set, which they do while the team sheet reveals moves.
        row_ev: (m,) EV of each of our actions against the opponent's per-class play.
        row_ev_loss: (m,) ``value - row_ev``, clipped at 0.
        col_ev_loss: per class, how much EV each of the opponent's actions gives up.
        weights: (K,) the class probabilities the solve used.
        duality_gap: |row LP value - column LP value|, reported rather than assumed.
    """

    value: float
    row_strategy: np.ndarray
    col_strategies: tuple[np.ndarray, ...]
    col_marginal: np.ndarray | None
    row_ev: np.ndarray
    row_ev_loss: np.ndarray
    col_ev_loss: tuple[np.ndarray, ...]
    weights: np.ndarray
    duality_gap: float

    def row_support(self, threshold: float = 1e-6) -> np.ndarray:
        return np.flatnonzero(self.row_strategy > threshold)


def _bayesian_maximin(
    matrices: list[np.ndarray], weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Our LP: maximise sum_k v_k subject to v_k <= w_k (x' M_k)_j for every k, j."""
    m = matrices[0].shape[0]
    k = len(matrices)
    columns = [mat.shape[1] for mat in matrices]

    # variables: x_0..x_{m-1}, then v_0..v_{k-1}
    c = np.zeros(m + k)
    c[m:] = -1.0

    a_ub = np.zeros((sum(columns), m + k))
    row = 0
    for index, mat in enumerate(matrices):
        n = mat.shape[1]
        a_ub[row : row + n, :m] = -weights[index] * mat.T
        a_ub[row : row + n, m + index] = 1.0
        row += n
    b_ub = np.zeros(sum(columns))

    a_eq = np.zeros((1, m + k))
    a_eq[0, :m] = 1.0
    # x >= 0, v free
    res = _lp(c, a_ub, b_ub, a_eq, np.array([1.0]), free_from=m)
    if not res.success:
        raise EquilibriumError(f"Bayesian row LP failed: {res.message}")
    return float(res.x[m:].sum()), res.x[:m]


def _bayesian_minimax(
    matrices: list[np.ndarray], weights: np.ndarray
) -> tuple[float, list[np.ndarray]]:
    """The opponent's LP: choose y_k per class to minimise our best response."""
    m = matrices[0].shape[0]
    columns = [mat.shape[1] for mat in matrices]
    total = sum(columns)

    # variables: y (concatenated per class), then u
    c = np.zeros(total + 1)
    c[-1] = 1.0

    # For each of our actions i:  sum_k w_k (M_k y_k)_i - u <= 0
    a_ub = np.zeros((m, total + 1))
    offset = 0
    for index, mat in enumerate(matrices):
        n = mat.shape[1]
        a_ub[:, offset : offset + n] = weights[index] * mat
        offset += n
    a_ub[:, -1] = -1.0
    b_ub = np.zeros(m)

    a_eq = np.zeros((len(matrices), total + 1))
    offset = 0
    for index, n in enumerate(columns):
        a_eq[index, offset : offset + n] = 1.0
        offset += n
    # y >= 0, u free
    res = _lp(c, a_ub, b_ub, a_eq, np.ones(len(matrices)), free_from=total)
    if not res.success:
        raise EquilibriumError(f"Bayesian column LP failed: {res.message}")
    strategies = []
    offset = 0
    for n in columns:
        strategies.append(res.x[offset : offset + n])
        offset += n
    return float(res.x[-1]), strategies


@timing.timed("lp")
def solve_bayesian(
    matrices: list[np.ndarray], weights: np.ndarray, eps: float = 1e-9
) -> BayesianEquilibrium:
    """Equilibrium when the column player observes their own type and we do not.

    ``matrices[k]`` is our payoff against a column player of class ``k``, and
    ``weights[k]`` is that class's probability. Averaging the matrices and calling
    :func:`solve` instead would solve a different game -- one where the opponent must move
    before learning their own spread -- and would overstate our value.
    """
    if not matrices:
        raise ValueError("no matrices to solve")
    mats = [np.asarray(mat, dtype=np.float64) for mat in matrices]
    rows = {mat.shape[0] for mat in mats}
    if len(rows) != 1:
        raise ValueError(f"every class must offer us the same actions, got {rows}")
    for mat in mats:
        if mat.size == 0 or not np.isfinite(mat).all():
            raise ValueError("a payoff matrix is empty or non-finite")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != len(mats):
        raise ValueError(f"{w.shape[0]} weights for {len(mats)} matrices")
    if w.min() < 0:
        raise ValueError("class weights must be non-negative")
    w = w / w.sum()

    (value_row, x_raw), (value_col, y_raw) = _both(
        sum(mat.size for mat in mats),
        lambda: _bayesian_maximin(mats, w),
        lambda: _bayesian_minimax(mats, w),
    )

    x = _clean(x_raw, eps)
    ys = tuple(_clean(y, eps) for y in y_raw)
    value = 0.5 * (value_row + value_col)

    row_ev = sum(w[k] * (mats[k] @ ys[k]) for k in range(len(mats)))
    col_ev_loss = []
    for mat in mats:
        per_action = x @ mat
        col_ev_loss.append(np.maximum(per_action - per_action.min(), 0.0))

    shared = {mat.shape[1] for mat in mats}
    marginal = None
    if len(shared) == 1:
        marginal = sum(w[k] * ys[k] for k in range(len(mats)))

    return BayesianEquilibrium(
        value=value,
        row_strategy=x,
        col_strategies=ys,
        col_marginal=marginal,
        row_ev=np.asarray(row_ev, dtype=np.float64),
        row_ev_loss=np.maximum(value - np.asarray(row_ev, dtype=np.float64), 0.0),
        col_ev_loss=tuple(col_ev_loss),
        weights=w,
        duality_gap=abs(value_row - value_col),
    )


def best_response(payoff: np.ndarray, col_strategy: np.ndarray) -> tuple[int, float]:
    """Row player's best pure response to a fixed column strategy."""
    ev = np.asarray(payoff, dtype=np.float64) @ np.asarray(col_strategy, dtype=np.float64)
    i = int(np.argmax(ev))
    return i, float(ev[i])


def value_of(payoff: np.ndarray, row_strategy: np.ndarray, col_strategy: np.ndarray) -> float:
    """Row player's expected payoff under a given strategy pair."""
    x = np.asarray(row_strategy, dtype=np.float64)
    y = np.asarray(col_strategy, dtype=np.float64)
    return float(x @ np.asarray(payoff, dtype=np.float64) @ y)
