"""A reading that gets more exact the longer it runs: stages, each a uniform layer (IKA-367).

A person's move is 30 to 45 s, and the analysis view may read one position for hours. The
best-first deepening (`deepen`) spends any budget, but its tree mixes depths, and the
optimizer's curse of a mixed-depth root was about +0.03 (IKA-362 §3): it did not beat the
width alone at equal time. A fixed depth-2 read of the support rectangle (`depth2_auto`)
has almost no curse and beat three branches with every branch kept (IKA-366), but it is
one fixed read: it fills in about a second and has no use for the rest of the budget.

This is iterative deepening in the depth-2 read's terms. A *stage* reads every cell of a
rectangle of the root the same way -- depth, branches kept, children's menus, the
knock-out fork -- and solves the rectangle as the game it is (`search._restricted_search`'s
reading: the rectangle's equilibrium, its value what the strategy guarantees against every
column at the prices in hand, the rest of the matrix an oracle that may add a row or a
column). Stages run from coarse to fine; each starts from the one before it:

* its rectangle is the previous stage's answer -- its support first, then the actions that
  do best against the other side's answer (a wider rectangle than the support alone,
  which IKA-362 found fills no further than four actions);
* a cell a previous stage already read the same way is not read again;
* it stops at the budget, and the answer is the last stage that *completed*: every cell of
  its rectangle is of one kind, however long the reading was.

The budget is read on a count clock (the work `search._refine_cells` reports in
`search.WORK`, priced by `LadderCost`), so the same budget is the same answer on a busy
machine and an idle one, or on the wall clock. A stage is not begun when its predicted
cost does not fit what is left, and is abandoned once what it has spent says it will not.

One reader for both roots: the open game is the Bayesian game of one completion of weight 1,
as `search._restricted_belief` is `_restricted_search` there. Cells are (row, column,
completion), each completion's cell read in that completion's position (the determinization
`_restricted_belief` uses).

Stages are written ``d<depth>r<rect>b<branches>k<child>[x]`` -- the depth (2 and up), the
rectangle's side, the branches kept per refined cell (``a``: all), each child's menus by the
Q's k best (``n<k>``: `narrow`'s damage order instead), ``x``: knock-outs forked
(`Budget.enumerate_knockouts`, at every level). A depth-d stage adds d - 2 segments
``/r<rect>b<branches>k<child>``, one per level below: a depth-3 cell's children are each
read as this module reads a root, one stage of depth 2 (``/r..b..k..``), a depth-4 cell's
as one stage of depth 3, and so on -- so every cell of a stage is the same tree.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import port, search
from .budget import Budget
from .equilibrium import EquilibriumError, solve, solve_bayesian
from .position import Position

#: All branches of a refined cell, in practice.
ALL_BRANCHES = 1000

_HEAD = re.compile(r"d([2-9])r([1-9][0-9]*)b([1-9][0-9]*|a)([kn])([1-9][0-9]*)(x)?")
_LEVEL = re.compile(r"r([1-9][0-9]*)b([1-9][0-9]*|a)([kn])([1-9][0-9]*)")


@dataclass(frozen=True, slots=True)
class Stage:
    """One uniform layer of the reading: how every cell of its rectangle is read."""

    depth: int
    rect: int
    branches: int
    child: int
    knockouts: bool = False
    #: Children's menus by the Q (``k``) or `narrow`'s damage order (``n``).
    q: bool = True
    #: Oracle passes after the rectangle: each may add a row and a column.
    passes: int = 1
    #: A depth-3-or-more stage's child read: one stage of depth - 1 on each child.
    sub: Stage | None = None

    @property
    def kind(self) -> tuple:
        """What a cell's value depends on (all but the top rectangle): two stages of one
        kind share their cells."""
        return (self.depth, self.branches, self.child, self.q, self.knockouts,
                None if self.sub is None else self.sub.label)

    @property
    def label(self) -> str:
        head = (f"d{self.depth}r{self.rect}b{_b(self.branches)}{'k' if self.q else 'n'}"
                f"{self.child}{'x' if self.knockouts else ''}")
        return head + "".join(f"/{level}" for level in self._levels())

    def _levels(self) -> list[str]:
        out = []
        sub = self.sub
        while sub is not None:
            out.append(f"r{sub.rect}b{_b(sub.branches)}{'k' if sub.q else 'n'}{sub.child}")
            sub = sub.sub
        return out


def _b(n: int) -> str:
    return "a" if n >= ALL_BRANCHES else str(n)


def parse_stage(label: str) -> Stage:
    parts = label.split("/")
    head = _HEAD.fullmatch(parts[0])
    levels = [_LEVEL.fullmatch(p) for p in parts[1:]]
    if head is None or any(m is None for m in levels):
        raise ValueError(
            f"a stage is d<depth>r<rect>b<branches|a>k<child>[x] (n<child>: damage order), "
            f"then one /r<rect>b<branches|a>k<child> per level below depth 2; not {label!r}"
        )
    depth = int(head.group(1))
    if len(levels) != depth - 2:
        raise ValueError(f"a depth-{depth} stage names {depth - 2} level(s) below: {label!r}")
    knockouts = head.group(6) is not None

    def b(text: str) -> int:
        return ALL_BRANCHES if text == "a" else int(text)

    sub = None
    for d, m in zip(range(2, depth), reversed(levels), strict=True):
        sub = Stage(depth=d, rect=int(m.group(1)), branches=b(m.group(2)),
                    child=int(m.group(4)), knockouts=knockouts, q=m.group(3) == "k", sub=sub)
    return Stage(depth=depth, rect=int(head.group(2)), branches=b(head.group(3)),
                 child=int(head.group(5)), knockouts=knockouts, q=head.group(4) == "k", sub=sub)


#: Named ladders. ``L1`` is the issue's order (IKA-367): depth 2 with three branches and
#: narrow children, every branch and the knock-outs with children at the Q's 24 (IKA-366's
#: +30), the rectangle 4 -> 8, then the support's depth 3 with three branches, then every
#: branch there, then wider, then depth 4 (the analysis view's hours).
LADDERS: dict[str, tuple[str, ...]] = {
    "L1": ("d2r4b3k8", "d2r4bak24x", "d2r8bak24x", "d3r4b3k24x/r3b3k16",
           "d3r4bak24x/r4bak24", "d3r8bak24x/r4bak24", "d3r8bak24x/r6bak24",
           "d4r4b3k24x/r3b3k24/r3b3k16", "d4r4bak24x/r4bak24/r4bak24"),
    # L1 widened at three branches before every branch at depth 3 (IKA-367: every branch at
    # depth 3 on the 4 x 4 moved the answer away from the long reference).
    "L2": ("d2r4b3k8", "d2r4bak24x", "d2r8bak24x", "d3r4b3k24x/r3b3k16",
           "d3r8b3k24x/r3b3k16", "d3r8bak24x/r4bak24", "d3r12bak24x/r4bak24",
           "d4r4b3k24x/r3b3k24/r3b3k16"),
    # Depth 2 only, ever wider: what depth 3 adds over the same time spent on width.
    "L0": ("d2r4b3k8", "d2r4bak24x", "d2r8bak24x", "d2r12bak24x", "d2r16bak24x",
           "d2r24bak24x"),
}


def parse_ladder(spec: str) -> tuple[Stage, ...]:
    """A named ladder (`LADDERS`) or stages joined by ``+``."""
    labels = LADDERS.get(spec) or tuple(spec.split("+"))
    return tuple(parse_stage(label) for label in labels)


@dataclass(frozen=True, slots=True)
class LadderCost:
    """Milliseconds of the counted work (IKA-367): a refined cell's turn, a child game's
    node (its crossing, fold and LP), a resolved cell of a child's matrix, a forward pass
    of the Q for children's menus, and a child game read by a stage of its own (depth 3
    and up: its orderings, rectangles and LPs). Measured on the local form, one core, the
    GPU leaf (`tools/position_set.py fit`)."""

    turn: float
    subgame: float
    cell: float
    q: float
    read: float = 0.0

    def ms(self, work: dict[str, int]) -> float:
        return (work["turns"] * self.turn + work["subgames"] * self.subgame
                + work["cells"] * self.cell + work["qs"] * self.q
                + work.get("reads", 0) * self.read)


#: Measured (IKA-367, `tools/position_set.py fit`): non-negative least squares of each
#: stage's wall milliseconds on the work it added -- ladder L1 read at 64 s on 24 recorded
#: M-C positions (width 36), eight processes sharing the card, depth-2 and depth-3 stages
#: (151 stages, R^2 0.934, wall over the priced clock 1.05). The depth-2 stages alone, in
#: one process, fitted subgame 6.2 ms and cell 0.058 ms (71 stages, R^2 0.979): the prices
#: trade against each other, the totals agree. The Q's pass and a child's own read came out
#: at 0 (their cost is in the others).
LADDER_COSTS: dict[tuple[str, int], LadderCost] = {
    ("local", 1): LadderCost(turn=7.0, subgame=11.5, cell=0.027, q=0.0, read=0.0),
}


def _zero() -> dict[str, int]:
    return {"turns": 0, "subgames": 0, "cells": 0, "qs": 0, "reads": 0}


@dataclass
class Item:
    """A completion as the ladder reads it (`hidden.Completion`'s fields it needs)."""

    position: Position
    weight: float = 1.0
    exact: bool = True
    slots: tuple[int, ...] = ()


@dataclass
class Rung:
    """One completed stage."""

    stage: str
    value: float
    strategy: np.ndarray
    replies: tuple[np.ndarray, ...]
    rows: int
    cols: tuple[int, ...]
    #: Cells read at this stage (not given by an earlier one of the same kind).
    fresh: int
    #: The clock's reading at its end, in milliseconds (count or wall), and the wall's.
    spent_ms: float
    wall_ms: float
    optimism: float = 0.0
    #: The counted work at its end (`LadderCost`'s kinds): what `fit` prices.
    work: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"stage": self.stage, "value": round(self.value, 6), "rows": self.rows,
                "cols": list(self.cols), "fresh": self.fresh,
                "spentMs": round(self.spent_ms, 1), "wallMs": round(self.wall_ms, 1),
                "optimism": round(self.optimism, 6), "work": dict(self.work)}


@dataclass
class LadderResult:
    strategy: np.ndarray
    value: float
    replies: tuple[np.ndarray, ...]
    #: Each completion's matrix in the solving side's orientation, at the prices in hand.
    prices: list[np.ndarray]
    rungs: list[Rung] = field(default_factory=list)
    #: Why it stopped: ``done`` (every stage), ``budget``, ``stop`` (the caller's event).
    stopped: str = "done"
    #: The stage that did not complete (begun and abandoned, or not begun), or None.
    unfinished: str | None = None
    abandoned: bool = False
    spent_ms: float = 0.0
    wall_ms: float = 0.0
    work: dict[str, int] = field(default_factory=_zero)
    unmodelled: set[str] = field(default_factory=set)
    #: The depth-1 answer it started from (its strategy and guaranteed value).
    start: tuple[np.ndarray, float] | None = None

    @property
    def depth_reached(self) -> str:
        return self.rungs[-1].stage if self.rungs else "d1"

    def to_json(self) -> dict[str, Any]:
        return {"rungs": [r.to_json() for r in self.rungs], "stopped": self.stopped,
                "unfinished": self.unfinished, "abandoned": self.abandoned,
                "spentMs": round(self.spent_ms, 1), "wallMs": round(self.wall_ms, 1),
                "work": dict(self.work)}


#: A priori branches kept per refined cell, by (all kept, knock-outs forked): a stage's
#: prediction before it has seen its own cells. IKA-366: the knock-out fork adds branches.
_BRANCHES_GUESS = {(False, False): 2.3, (False, True): 2.6, (True, False): 3.0,
                   (True, True): 5.0}
#: A child's legal actions average 17 to 18 (IKA-362 §4): a menu wider than that is the
#: same menu.
_CHILD_LEGAL = 17.5


def _guess_cell_ms(stage: Stage, cost: LadderCost) -> float:
    """A priori milliseconds of one cell of ``stage`` (scaled in `read` by how far the
    guesses of the stages already done were off)."""
    b = min(float(stage.branches), _BRANCHES_GUESS[stage.branches >= ALL_BRANCHES,
                                                  stage.knockouts])
    k = min(float(stage.child), _CHILD_LEGAL)
    if stage.sub is None:
        return cost.turn + b * (cost.subgame + k * k * cost.cell) + cost.q / 16.0
    # Each child: its node, then its own stage's rectangle (support grown by the oracle).
    rect = min(float(stage.sub.rect), k) + stage.sub.passes
    child = cost.subgame + k * k * cost.cell + rect * rect * _guess_cell_ms(stage.sub, cost)
    return cost.turn + cost.q + b * child


def _order(strategy: np.ndarray, ev: np.ndarray, count: int, *, larger: bool) -> list[int]:
    """The rectangle's side: the actions carrying weight, heaviest first, then the rest by
    how they do against the other side's answer (``larger``: higher is better)."""
    strategy = np.asarray(strategy, dtype=np.float64)
    live = np.flatnonzero(strategy > 1e-9)
    live = live[np.argsort(-strategy[live], kind="stable")]
    rest = np.flatnonzero(strategy <= 1e-9)
    key = -np.asarray(ev, dtype=np.float64)[rest] if larger else np.asarray(ev)[rest]
    rest = rest[np.argsort(key, kind="stable")]
    return [int(i) for i in [*live, *rest][:count]]


class _Clock:
    def __init__(self, kind: str, cost: LadderCost, start_ms: float, budget_ms: float | None,
                 stop: Any) -> None:  # noqa: ANN401 - threading.Event
        self.kind = kind
        self.cost = cost
        self.start_ms = start_ms
        self.budget_ms = budget_ms
        self.stop = stop
        self.began = time.perf_counter()

    def wall_ms(self) -> float:
        return (time.perf_counter() - self.began) * 1000.0

    def spent_ms(self, work: dict[str, int]) -> float:
        if self.kind == "wall":
            return self.start_ms + self.wall_ms()
        return self.start_ms + self.cost.ms(work)

    def left_ms(self, work: dict[str, int]) -> float:
        if self.budget_ms is None:
            return float("inf")
        return self.budget_ms - self.spent_ms(work)

    def stopped(self) -> bool:
        return self.stop is not None and self.stop.is_set()


#: Cells a crossing of a depth-2 stage holds before the clock is read again.
CHUNK = 8


class Stopped(Exception):  # noqa: N818 - a signal, not an error
    """The caller's stop event, met inside a cell's own read (depth 3 and up)."""


def read(  # noqa: PLR0913, PLR0912, PLR0915, C901 - the root, the stages, the clock
    reg: Any,  # noqa: ANN401
    side: int,
    row: Sequence[Any],
    col: Sequence[Any],
    items: Sequence[Any],
    matrices: Sequence[np.ndarray],
    weights: Sequence[float],
    start: Any,  # noqa: ANN401 - the depth-1 answer: row_strategy, col_strategies
    leaf: Any,  # noqa: ANN401
    *,
    budget: Budget,
    stages: Sequence[Stage],
    budget_ms: float | None,
    cost: LadderCost | None = None,
    clock: str = "count",
    start_ms: float = 0.0,
    stop: Any = None,  # noqa: ANN401
    on_rung: Callable[[Rung], None] | None = None,
) -> LadderResult:
    """Side ``side``'s answer, stage by stage (the module's docstring).

    ``row`` / ``col`` are side 0's and side 1's menus; ``matrices[k]`` completion k's
    depth-1 matrix in ``side``'s orientation (its own actions the rows; side 1's is the
    negated transpose of side 0's), ``items[k]`` its position; ``start`` the depth-1
    Bayesian answer. ``budget_ms`` is what the stages may spend (None: every stage, the
    caller's ``stop`` event the only brake), read on ``clock`` (``count``: `LadderCost`,
    ``wall``); ``start_ms`` is what the move has spent before (the node). ``on_rung`` is
    called with each completed stage. ``budget`` is the root's (the stages fork the
    knock-outs on it where they say so).
    """
    cost = cost or LADDER_COSTS["local", 1]
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    kinds = len(items)
    prices = [np.array(m, dtype=np.float64, copy=True) for m in matrices]
    x = np.asarray(start.row_strategy, dtype=np.float64)
    ys = [np.asarray(y, dtype=np.float64) for y in start.col_strategies]
    value = float(sum(w[k] * float((x @ prices[k]).min()) for k in range(kinds)))
    result = LadderResult(strategy=x, value=value, replies=tuple(ys), prices=prices,
                          start=(x, value))
    #: (completion, own row, own column, kind) -> the cell's value in side 0's units, or None.
    memo: dict[tuple, float | None] = {}
    #: The turns shared across completions, per knock-out setting (`search.TurnShare`).
    turns: dict[bool, dict] = {False: {}, True: {}}
    clean: dict[tuple[int, int], bool] = {}
    hidden_side = 1 - side
    work = _zero()
    outer = search.WORK
    search.WORK = work
    run = _Clock(clock, cost, start_ms, budget_ms, stop)
    #: Measured milliseconds per fresh cell, by kind: the next stage's prediction.
    measured: dict[tuple, tuple[float, int]] = {}
    #: By depth, the completed stages' measured and a-priori (`_guess_cell_ms`) totals: a
    #: new kind's guess is scaled by how far the guesses have been off in this read.
    calib: dict[int, tuple[float, float]] = {}
    try:
        for stage in stages:
            if run.stopped():
                result.stopped, result.unfinished = "stop", stage.label
                break
            stage_budget = replace(budget, enumerate_knockouts=True) if stage.knockouts else budget
            rows = _order(x, sum(w[k] * (prices[k] @ ys[k]) for k in range(kinds)), stage.rect,
                          larger=True)
            cols = [_order(ys[k], x @ prices[k], stage.rect, larger=False) for k in range(kinds)]
            trial = [p.copy() for p in prices]
            guess = _guess_cell_ms(stage, cost)
            if stage.kind in measured:
                per_cell = measured[stage.kind][0] / max(measured[stage.kind][1], 1)
            else:
                have = calib.get(stage.depth) or (
                    (sum(v[0] for v in calib.values()), sum(v[1] for v in calib.values()))
                    if calib else None)
                ratio = 1.0 if not have or have[1] <= 0 else have[0] / have[1]
                per_cell = guess * min(max(ratio, 0.05), 3.0)
            fresh_keys = {_key(side, i, j, k, stage) for k in range(kinds) for i in rows
                          for j in cols[k]} - set(memo)
            if per_cell * len(fresh_keys) > run.left_ms(work):
                result.stopped, result.unfinished = "budget", stage.label
                break
            stage_began = run.spent_ms(work)
            fresh = 0
            abandoned = False
            answer = None
            step = CHUNK if stage.sub is None else 1
            for attempt in range(stage.passes + 1):
                todo = [(i, j, k) for k in range(kinds) for i in rows for j in cols[k]]
                asked: dict[tuple, tuple] = {}
                for i, j, k in todo:
                    key = _key(side, i, j, k, stage)
                    if key not in memo and key not in asked:
                        ours, theirs = (i, j) if side == 0 else (j, i)
                        asked[key] = (k, ours, theirs)
                keys = list(asked)
                for at in range(0, len(keys), step):
                    chunk = keys[at:at + step]
                    try:
                        got = _read_cells(reg, [asked[key] for key in chunk], row, col, items,
                                          leaf, stage, budget, stage_budget, turns, clean,
                                          hidden_side, result.unmodelled, cost, stop)
                    except Stopped:
                        abandoned = True
                        break
                    for key, v in zip(chunk, got, strict=True):
                        memo[key] = v
                    fresh += len(chunk)
                    left = len(keys) - (at + len(chunk))
                    per = (run.spent_ms(work) - stage_began) / max(fresh, 1)
                    if run.stopped() or run.left_ms(work) < 0 or per * left > run.left_ms(work):
                        abandoned = True
                        break
                if abandoned:
                    break
                for i, j, k in todo:
                    v = memo[_key(side, i, j, k, stage)]
                    if v is not None:
                        trial[k][i, j] = v if side == 0 else -v
                try:
                    restricted = solve_bayesian(
                        [trial[k][np.ix_(rows, cols[k])] for k in range(kinds)], w)
                except EquilibriumError:
                    break
                sx = np.zeros(len(trial[0]), dtype=np.float64)
                sx[rows] = restricted.row_strategy
                sys_ = []
                for k in range(kinds):
                    y = np.zeros(trial[k].shape[1], dtype=np.float64)
                    y[cols[k]] = restricted.col_strategies[k]
                    sys_.append(y)
                col_ev = [sx @ trial[k] for k in range(kinds)]
                row_ev = sum(w[k] * (trial[k] @ sys_[k]) for k in range(kinds))
                guarantee = float(sum(w[k] * float(col_ev[k].min()) for k in range(kinds)))
                answer = (sx, sys_, guarantee, float(restricted.value) - guarantee)
                if attempt == stage.passes:
                    break
                grew = False
                best_row = int(np.argmax(row_ev))
                if best_row not in rows and float(row_ev[best_row]) > float(
                        restricted.value) + search.ORACLE_TOLERANCE:
                    rows.append(best_row)
                    grew = True
                for k in range(kinds):
                    earned = float(col_ev[k] @ sys_[k])
                    best_col = int(np.argmin(col_ev[k]))
                    if best_col not in cols[k] and float(
                            col_ev[k][best_col]) < earned - search.ORACLE_TOLERANCE:
                        cols[k].append(best_col)
                        grew = True
                if not grew:
                    break
            if abandoned or answer is None:
                result.stopped = "stop" if run.stopped() else "budget"
                result.unfinished, result.abandoned = stage.label, abandoned
                break
            spent = run.spent_ms(work) - stage_began
            total, count = measured.get(stage.kind, (0.0, 0))
            measured[stage.kind] = (total + spent, count + fresh)
            got_ms, guessed = calib.get(stage.depth, (0.0, 0.0))
            calib[stage.depth] = (got_ms + spent, guessed + guess * fresh)
            prices = trial
            x, ys = answer[0], answer[1]
            rung = Rung(stage=stage.label, value=answer[2], strategy=x, replies=tuple(ys),
                        rows=len(rows), cols=tuple(len(c) for c in cols), fresh=fresh,
                        spent_ms=run.spent_ms(work), wall_ms=run.wall_ms(), optimism=answer[3],
                        work=dict(work))
            result.rungs.append(rung)
            result.strategy, result.value, result.replies = x, answer[2], tuple(ys)
            result.prices = prices
            if on_rung is not None:
                on_rung(rung)
    finally:
        search.WORK = outer
        if outer is not None:
            # A cell's own read (depth 3 and up): its work is the cell's.
            for kind, n in work.items():
                outer[kind] += n
    result.spent_ms = run.spent_ms(work)
    result.wall_ms = run.wall_ms()
    result.work = work
    if result.rungs:
        result.unmodelled.add(f"ladder read to {result.depth_reached} ({len(result.rungs)} "
                              f"stage(s)); stopped: {result.stopped}")
    return result


def _key(side: int, i: int, j: int, k: int, stage: Stage) -> tuple:
    ours, theirs = (i, j) if side == 0 else (j, i)
    return (k, ours, theirs, stage.kind)


def _read_cells(  # noqa: PLR0913 - the cells and how they are read
    reg: Any,  # noqa: ANN401
    cells: list[tuple[int, int, int]],
    row: Sequence[Any],
    col: Sequence[Any],
    items: Sequence[Any],
    leaf: Any,  # noqa: ANN401
    stage: Stage,
    root_budget: Budget,
    budget: Budget,
    turns: dict[bool, dict],
    clean: dict[tuple[int, int], bool],
    hidden_side: int,
    unmodelled: set[str],
    cost: LadderCost,
    stop: Any,  # noqa: ANN401
) -> list[float | None]:
    """Each cell's value in side 0's units (None: not refinable, it keeps its price)."""
    if stage.sub is not None:
        return [_deep_cell(reg, items[k].position, row[i], col[j], leaf, stage, root_budget,
                           budget, unmodelled, cost, stop) for k, i, j in cells]
    shares = []
    for k, i, j in cells:
        item = items[k]
        if item.exact or not item.slots:
            shares.append(None)
            continue
        if (i, j) not in clean:
            clean[(i, j)] = not search._cell_reaches_bench(
                reg, row[i], col[j], hidden_side, tuple(item.slots), item.position)
        shares.append(search.TurnShare(cache=turns[stage.knockouts], key=(i, j),
                                       side=hidden_side, slots=tuple(item.slots),
                                       clean=clean[(i, j)]))
    asked = [(items[k].position, row[i], col[j]) for k, i, j in cells]
    try:
        found = search._refine_cells(reg, asked, leaf, budget=budget, sub_limit=stage.child,
                                     sub_branches=stage.branches, shares=shares,
                                     child_q=stage.child if stage.q else None)
    except port.PortRefused:
        if len(cells) == 1:
            unmodelled.add("ladder: the port refused a refined cell; it keeps its price")
            return [None]
        return [v for cell in cells for v in _read_cells(
            reg, [cell], row, col, items, leaf, stage, root_budget, budget, turns, clean,
            hidden_side, unmodelled, cost, stop)]
    out = []
    for value, notes, _solved in found:
        unmodelled.update(notes)
        out.append(value)
    return out


def _deep_cell(  # noqa: PLR0913 - one cell and its stage
    reg: Any,  # noqa: ANN401
    pos: Position,
    ours: Any,  # noqa: ANN401
    theirs: Any,  # noqa: ANN401
    leaf: Any,  # noqa: ANN401
    stage: Stage,
    root_budget: Budget,
    budget: Budget,
    unmodelled: set[str],
    cost: LadderCost,
    stop: Any,  # noqa: ANN401
) -> float | None:
    """One cell read ``stage.depth`` plies: its turn's kept branches, each child read as a
    root by the one stage ``stage.sub`` (its menus by the Q's or damage's ``stage.child``)."""
    from .deepen import _q_menus
    from .narrow import narrow

    work = search.WORK
    try:
        result = port.turn(reg, pos, [ours, theirs], budget, full=True)
    except port.PortRefused:
        unmodelled.add("ladder: the port refused a refined cell; it keeps its price")
        return None
    if work is not None:
        work["turns"] += 1
    kept = search._kept_from(result, stage.branches, unmodelled)
    if kept is None:
        return None
    branches, weights = kept
    live = [b.position for b in branches if not b.position.ended]
    if stage.q:
        menus = iter(_q_menus(reg, live, stage.child) if live else [])
        if work is not None and live:
            work["qs"] += 1
    else:
        menus = iter([(narrow(reg, p, 0, limit=stage.child).actions,
                       narrow(reg, p, 1, limit=stage.child).actions) for p in live])
    values = []
    for branch in branches:
        child = branch.position
        if child.ended:
            values.append(float(np.asarray(leaf([child]))[0]))
            continue
        crow, ccol = next(menus)
        if not crow or not ccol:
            return None
        if stop is not None and stop.is_set():
            raise Stopped
        try:
            # The child's own matrix as the stage resolves its turns (the knock-out fork).
            m, notes = search.batched_payoff(reg, child, crow, ccol, leaf, budget=budget)
            m = np.asarray(m, dtype=np.float64)
            eq = solve(m)
        except (EquilibriumError, port.PortRefused):
            return None
        unmodelled.update(notes)
        if work is not None:
            work["subgames"] += 1
            work["cells"] += m.size
        start = _Start(eq.row_strategy, [eq.col_strategy])
        got = read(reg, 0, crow, ccol, [Item(child)], [m], [1.0], start, leaf,
                   budget=root_budget, stages=[stage.sub], budget_ms=None, cost=cost, stop=stop)
        if work is not None:
            work["reads"] += 1
        if not got.rungs:
            if got.stopped == "stop":
                raise Stopped
            return None
        values.append(float(got.value))
    return float(np.asarray(values) @ weights)


@dataclass
class _Start:
    row_strategy: np.ndarray
    col_strategies: list[np.ndarray]


__all__ = ["LADDERS", "LADDER_COSTS", "Item", "LadderCost", "LadderResult", "Rung", "Stage",
           "Stopped", "parse_ladder", "parse_stage", "read"]
