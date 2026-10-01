"""Which equilibrium a side plays when its game has many (IKA-196).

`equilibrium.solve_bayesian` returns one vertex of the set of optimal strategies, the one
HiGHS's pivots land on. The game value is the same at every point of that set, so every
value the search reads is unchanged by which point is played; what changes is how much
each point takes from an opponent who does not play the equilibrium -- and every opponent
this agent meets is one of those. A label here picks the point, for the move actually
played, after the LP has given the value:

* ``lp`` -- the LP's vertex, unchanged (the default, what every game before played).
* ``unif`` -- among the strategies that still guarantee the value, the one that takes the
  most from an opponent who picks his reply uniformly (per completion, at its weight). One
  more LP of the same size.
* ``ment<D>`` -- the strategy of largest entropy among those guaranteeing the value less
  ``D`` (win probability). ``ment0`` stays exactly in the optimal set. A small convex
  program (SLSQP).
* ``qre<T>`` -- the logit quantal response equilibrium at temperature ``T``: both sides'
  entropy-regularised game, solved by Newton's method. No longer an equilibrium of the
  game itself; it gives up at most ``T`` times the log of the action counts.

The opponent's side of the answer (the replies per completion) is left as the LP gave it:
each side samples its own move from its own solve, so the row strategy is all that a label
changes. `STATS` counts the calls, the ones whose strategy moved, the fallbacks and the
seconds, per label: the positive control that a label reached the move.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace

import numpy as np

from .equilibrium import Equilibrium, EquilibriumError, _clean, _lp, bayesian_inputs

#: The label every game played before IKA-196: the LP's own vertex.
DEFAULT_EQ_SELECT = "lp"

#: Slack on the value a refining program keeps, below the least of the LP's value and what
#: the LP's own strategy guarantees.
VALUE_TOLERANCE = 1e-10

#: A strategy "moved" when its total variation from the LP's is above this.
MOVED = 1e-6

_LABEL = re.compile(r"^(lp|unif|ment|qre)((?:\d+(?:\.\d*)?|\.\d+)(?:e-?\d+)?)?$")

#: label -> {"calls", "moved", "fallbacks", "seconds"}.
STATS: dict[str, dict[str, float]] = {}


@dataclass(frozen=True, slots=True)
class EqSelect:
    kind: str
    param: float


def parse_eq_select(label: str) -> EqSelect:
    """``lp``, ``unif``, ``ment<D>`` (``D`` >= 0) or ``qre<T>`` (``T`` > 0)."""
    got = _LABEL.match(label)
    if got is None:
        raise ValueError(f"equilibrium selection {label!r}: lp, unif, ment<D> or qre<T>")
    kind, number = got.group(1), got.group(2)
    if kind in ("lp", "unif"):
        if number is not None:
            raise ValueError(f"equilibrium selection {label!r}: {kind} takes no number")
        return EqSelect(kind, 0.0)
    if number is None:
        raise ValueError(f"equilibrium selection {label!r}: {kind} needs a number")
    param = float(number)
    if kind == "qre" and param <= 0.0:
        raise ValueError(f"equilibrium selection {label!r}: the temperature must be > 0")
    return EqSelect(kind, param)


def guarantee(matrices: list[np.ndarray], weights: np.ndarray, x: np.ndarray) -> float:
    """What row strategy ``x`` guarantees: sum_k w_k min_j (x M_k)_j."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return float(sum(w[k] * float((x @ m).min()) for k, m in enumerate(matrices)))


def _uniform_reply(mats: list[np.ndarray], w: np.ndarray, value: float) -> np.ndarray | None:
    """`unif`: max sum_k w_k mean_j (x M_k)_j subject to guaranteeing ``value``."""
    m = mats[0].shape[0]
    k = len(mats)
    columns = [mat.shape[1] for mat in mats]
    gain = sum(w[i] * mats[i].mean(axis=1) for i in range(k))
    c = np.zeros(m + k)
    c[:m] = -gain
    a_ub = np.zeros((sum(columns) + 1, m + k))
    row = 0
    for index, mat in enumerate(mats):
        n = mat.shape[1]
        a_ub[row : row + n, :m] = -w[index] * mat.T
        a_ub[row : row + n, m + index] = 1.0
        row += n
    a_ub[row, m:] = -1.0
    b_ub = np.zeros(sum(columns) + 1)
    b_ub[row] = -(value - VALUE_TOLERANCE)
    a_eq = np.zeros((1, m + k))
    a_eq[0, :m] = 1.0
    res = _lp(c, a_ub, b_ub, a_eq, np.array([1.0]), free_from=m)
    if not res.success:
        return None
    return res.x[:m]


def _max_entropy(
    mats: list[np.ndarray], w: np.ndarray, value: float, slack: float, start: np.ndarray
) -> np.ndarray | None:
    """`ment<D>`: max H(x) subject to guaranteeing ``value - slack``."""
    from scipy.optimize import minimize

    m = mats[0].shape[0]
    k = len(mats)
    floor = value - slack - VALUE_TOLERANCE
    stacked = np.vstack([w[i] * mats[i].T for i in range(k)])  # (sum n, m)
    owner = np.concatenate([np.full(mat.shape[1], i) for i, mat in enumerate(mats)])
    pick = np.zeros((len(owner), k))
    pick[np.arange(len(owner)), owner] = 1.0
    a_ineq = np.hstack([stacked, -pick])  # w_k (x M_k)_j - v_k >= 0
    a_sum = np.concatenate([np.zeros(m), np.ones(k)])
    v0 = np.array([w[i] * float((start @ mats[i]).min()) for i in range(k)])
    z0 = np.concatenate([start, v0])

    def negentropy(z: np.ndarray) -> float:
        x = np.clip(z[:m], 1e-300, None)
        return float(np.sum(x * np.log(x)))

    def gradient(z: np.ndarray) -> np.ndarray:
        g = np.zeros(m + k)
        g[:m] = np.log(np.clip(z[:m], 1e-300, None)) + 1.0
        return g

    constraints = [
        {"type": "ineq", "fun": lambda z: a_ineq @ z, "jac": lambda _z: a_ineq},
        {"type": "ineq", "fun": lambda z: np.array([a_sum @ z - floor]),
         "jac": lambda _z: a_sum[None, :]},
        {"type": "eq", "fun": lambda z: np.array([z[:m].sum() - 1.0]),
         "jac": lambda _z: np.concatenate([np.ones(m), np.zeros(k)])[None, :]},
    ]
    bounds = [(0.0, 1.0)] * m + [(None, None)] * k
    res = minimize(negentropy, z0, jac=gradient, bounds=bounds, constraints=constraints,
                   method="SLSQP", options={"maxiter": 200, "ftol": 1e-12})
    x = np.clip(res.x[:m], 0.0, None)
    if not np.isfinite(x).all() or x.sum() <= 0:
        return None
    x = x / x.sum()
    if guarantee(mats, w, x) < floor - 1e-6:
        return None
    return x


def qre_row(
    mats: list[np.ndarray], w: np.ndarray, tau: float, *, iterations: int = 200
) -> tuple[np.ndarray, list[np.ndarray]]:
    """The logit QRE of the Bayesian game at temperature ``tau``: the row strategy and the
    replies per completion.

    Solved in the row side's logits ``z`` (x = softmax z), where the QRE is the root of
    F(z) = z - u(z) / tau up to a constant, u the row actions' payoffs against the column
    side's soft best replies (closed form). Its Jacobian I + Q P / tau^2 (Q, P positive
    semidefinite) is never singular. Newton steps with a backtracking line search on |F|,
    and the temperature brought down from 1 by halves, each solve starting from the last.
    """
    m = mats[0].shape[0]
    z = np.zeros(m)
    replies: list[np.ndarray] = []

    def softmax(v: np.ndarray) -> np.ndarray:
        e = np.exp(v - v.max())
        return e / e.sum()

    def residual(z: np.ndarray, t: float, jacobian: bool):  # noqa: ANN202
        x = softmax(z)
        u = np.zeros(m)
        q = np.zeros((m, m)) if jacobian else None
        ys = []
        for k, mat in enumerate(mats):
            y = softmax(-(x @ mat) / t)
            ys.append(y)
            my = mat @ y
            u += w[k] * my
            if jacobian:
                q += w[k] * ((mat * y) @ mat.T - np.outer(my, my))
        f = z - u / t
        f -= f.mean()
        if not jacobian:
            return f, x, ys, None
        p = np.diag(x) - np.outer(x, x)
        return f, x, ys, np.eye(m) + (q @ p) / (t * t)

    temperatures = []
    t = 1.0
    while t > tau:
        temperatures.append(t)
        t *= 0.5
    temperatures.append(tau)
    for t in temperatures:
        f, x, replies, jac = residual(z, t, True)
        for _ in range(iterations):
            size = float(np.abs(f).max())
            if size < 1e-11:
                break
            try:
                step = np.linalg.solve(jac, -f)
            except np.linalg.LinAlgError:
                break
            step -= step.mean()
            scale = 1.0
            while True:
                trial = z + scale * step
                got = residual(trial, t, False)
                if float(np.abs(got[0]).max()) < size or scale < 1e-10:
                    break
                scale *= 0.5
            z = trial
            f, x, replies, jac = residual(z, t, True)
    return softmax(z), replies


def reselect_bayesian(
    matrices: list[np.ndarray],
    weights: np.ndarray,
    value: float,
    strategy: np.ndarray,
    label: str,
) -> np.ndarray:
    """The row strategy ``label`` plays in the Bayesian game whose LP gave ``value`` and
    ``strategy``. ``lp`` hands ``strategy`` back unchanged; a program that fails does too
    (counted under ``fallbacks``)."""
    spec = parse_eq_select(label)
    if spec.kind == "lp":
        return strategy
    started = time.perf_counter()
    stats = STATS.setdefault(label, {"calls": 0, "moved": 0, "fallbacks": 0, "seconds": 0.0})
    stats["calls"] += 1
    mats, w = bayesian_inputs(matrices, weights)
    strategy = np.asarray(strategy, dtype=np.float64)
    # The LP's own strategy is feasible at what it guarantees, which may sit a rounding
    # below the reported value.
    value = min(float(value), guarantee(mats, w, strategy))
    got: np.ndarray | None
    try:
        if spec.kind == "unif":
            got = _uniform_reply(mats, w, value)
        elif spec.kind == "ment":
            got = _max_entropy(mats, w, value, spec.param, strategy)
        else:
            got, _ = qre_row(mats, w, spec.param)
    except (EquilibriumError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        got = None
    if got is None or not np.isfinite(got).all():
        stats["fallbacks"] += 1
        stats["seconds"] += time.perf_counter() - started
        return strategy
    out = _clean(got, 1e-9)
    if 0.5 * float(np.abs(out - strategy).sum()) > MOVED:
        stats["moved"] += 1
    stats["seconds"] += time.perf_counter() - started
    return out


def reselect(payoff: np.ndarray, value: float, strategy: np.ndarray, label: str) -> np.ndarray:
    """`reselect_bayesian` of the open game: one matrix, the row player's."""
    return reselect_bayesian([np.asarray(payoff, dtype=np.float64)], np.ones(1), value,
                             strategy, label)


def reselected(payoff: np.ndarray, eq: Equilibrium, select: tuple[str, str]) -> Equilibrium:
    """``eq`` with the row player's strategy chosen by ``select[0]`` and the column
    player's by ``select[1]`` (the column player's game is the negated transpose). The
    value and the duality gap are the LP's."""
    a = np.asarray(payoff, dtype=np.float64)
    x = reselect(a, eq.value, np.asarray(eq.row_strategy), select[0])
    y = reselect(-a.T, -eq.value, np.asarray(eq.col_strategy), select[1])
    if x is eq.row_strategy and y is eq.col_strategy:
        return eq
    row_ev = a @ y
    col_ev = x @ a
    return replace(
        eq, row_strategy=x, col_strategy=y, row_ev=row_ev, col_ev=col_ev,
        row_ev_loss=np.maximum(eq.value - row_ev, 0.0),
        col_ev_loss=np.maximum(col_ev - eq.value, 0.0),
    )
