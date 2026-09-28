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

import contextlib
import os
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
    # L2 with the 8 x 8 rectangle first (IKA-367: every branch on the 4 x 4 blundered where
    # the 4 x 4 left the opponent's answer out -- one position lost 0.96 -- and the 8 x 8
    # took most of the gain), then every branch on it, then wider.
    "L3": ("d2r4b3k8", "d2r8b3k8", "d2r8bak24x", "d2r12bak24x", "d3r8b3k24x/r3b3k16",
           "d3r8bak24x/r4bak24", "d3r12bak24x/r4bak24", "d4r4b3k24x/r3b3k24/r3b3k16"),
    # Depth 2 only, ever wider: what depth 3 adds over the same time spent on width.
    "L0": ("d2r4b3k8", "d2r4bak24x", "d2r8bak24x", "d2r12bak24x", "d2r16bak24x",
           "d2r24bak24x"),
    # Width first at three branches and narrow children, then the children wider, then every
    # branch (IKA-367: behind a hidden bench the 4 -> 8 rectangle at three branches took the
    # gain, and every branch with the knock-outs cost 11 s of the clock for nothing seen).
    "L5": ("d2r4b3k8", "d2r8b3k8", "d2r12b3k8", "d2r12b3k16", "d2r16b3k16", "d2r16bak24x",
           "d2r24bak24x", "d3r8b3k24x/r3b3k16", "d3r8bak24x/r4bak24", "d3r12bak24x/r6bak24",
           "d4r4b3k24x/r3b3k24/r3b3k16", "d4r8b3k24x/r4b3k24/r3b3k16"),
    # L3's start (the 8 x 8 before every branch), L0's widening (it wins on the recorded
    # positions, IKA-367), then the support's depth 3 and 4 on the wide depth 2: the
    # analysis view's minutes and hours.
    "L4": ("d2r4b3k8", "d2r8b3k8", "d2r8bak24x", "d2r12bak24x", "d2r16bak24x",
           "d2r24bak24x", "d3r8b3k24x/r3b3k16", "d3r8bak24x/r4bak24", "d3r12bak24x/r6bak24",
           "d4r4b3k24x/r3b3k24/r3b3k16", "d4r8b3k24x/r4b3k24/r3b3k16"),
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
#: stage's wall milliseconds on the work it added -- ladder L1 read at 64 s, one process with
#: the machine to itself (a person's game), the stacked pass (`STACK`), the GPU leaf without
#: CUDA graphs: 10 recorded open M-C positions and 6 hidden ones (3-6 completions), width 36;
#: 83 stages, R^2 0.951, wall over the priced clock 1.01. The Q's pass and a child's own read
#: came out at 0 (their cost is in the others). Before the stacked pass, and with eight
#: processes sharing the card, the same fit gave about 1.7 times these.
LADDER_COSTS: dict[tuple[str, int], LadderCost] = {
    ("local", 1): LadderCost(turn=0.66, subgame=3.6, cell=0.048, q=0.0, read=0.0),
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
    #: Worker processes that read its cells (`start_pool`, IKA-364); 0: read here.
    workers: int = 0
    #: With workers: where the reader's time went (`_Pool.stats`: its waits on them, its
    #: sends and receives, the workers' own reading time, its CPU and wall).
    pool: dict[str, float] = field(default_factory=dict)

    @property
    def depth_reached(self) -> str:
        return self.rungs[-1].stage if self.rungs else "d1"

    def to_json(self) -> dict[str, Any]:
        return {"rungs": [r.to_json() for r in self.rungs], "stopped": self.stopped,
                "unfinished": self.unfinished, "abandoned": self.abandoned,
                "spentMs": round(self.spent_ms, 1), "wallMs": round(self.wall_ms, 1),
                "work": dict(self.work),
                **({"workers": self.workers,
                    "pool": {k: round(v, 1) for k, v in self.pool.items()}}
                   if self.workers else {})}


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
                 stop: Any, began: float | None = None) -> None:  # noqa: ANN401 - threading.Event
        self.kind = kind
        self.cost = cost
        self.start_ms = start_ms
        self.budget_ms = budget_ms
        self.stop = stop
        self.began = time.perf_counter() if began is None else began

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
    began: float | None = None,
) -> LadderResult:
    """Side ``side``'s answer, stage by stage (the module's docstring).

    ``row`` / ``col`` are side 0's and side 1's menus; ``matrices[k]`` completion k's
    depth-1 matrix in ``side``'s orientation (its own actions the rows; side 1's is the
    negated transpose of side 0's), ``items[k]`` its position; ``start`` the depth-1
    Bayesian answer. ``budget_ms`` is what the stages may spend (None: every stage, the
    caller's ``stop`` event the only brake), read on ``clock`` (``count``: `LadderCost`,
    ``wall``); ``start_ms`` is what the move has spent before (the node). ``on_rung`` is
    called with each completed stage. ``budget`` is the root's (the stages fork the
    knock-outs on it where they say so). ``began`` is when the wall clock started
    (`time.perf_counter`; None: now): a move's reading counts the depth-1 node it built.

    With a pool of worker processes (`start_pool`, IKA-364) a top-level read sends each
    stage's cells to the workers and takes their values back in the order it asked them,
    so the budget is read at the same points as without; on the count clock, with the
    serial chunks (`POOL_CHUNK` = `CHUNK`), it is the serial read to the bit.
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
    run = _Clock(clock, cost, start_ms, budget_ms, stop, began)
    # The workers read a top-level read's cells; a cell's own read (depth 3 and up) is
    # always read where it is.
    pool = _POOL if outer is None and _POOL is not None and _POOL.alive() else None
    if pool is not None:
        pool.begin(row, col, items, budget, hidden_side, cost)
        result.workers = len(pool.conns)
        cpu_began = time.process_time()
    #: The completed stages' counted and wall milliseconds (the wall clock's scale).
    stages_count = stages_wall = 0.0
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
                # IKA-364: the depth-3 guesses run 100-200 times the counted cost (1,794 ms a
                # cell guessed for d3r8b3k24x, 12-24 counted), so a floor of 0.05 predicted
                # the next depth-3 stage at 4-8 times its cost and a 32 s read stopped at
                # 5-14 s.
                per_cell = guess * min(max(ratio, 0.002), 3.0)
            fresh_keys = {_key(side, i, j, k, stage) for k in range(kinds) for i in rows
                          for j in cols[k]} - set(memo)
            # The prediction is in counted milliseconds; on the wall clock it is scaled by
            # this read's wall a counted millisecond so far (the workers read several at once:
            # IKA-364), 1 before a stage has completed.
            speed = (stages_wall / stages_count
                     if run.kind == "wall" and stages_count > 0 else 1.0)
            if per_cell * len(fresh_keys) * speed > run.left_ms(work):
                result.stopped, result.unfinished = "budget", stage.label
                break
            stage_began = run.spent_ms(work)
            count_began, wall_began = cost.ms(work), run.wall_ms()
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
                size = step if pool is None or stage.sub is not None else pool.chunk(len(keys))
                chunks = [keys[at:at + size] for at in range(0, len(keys), size)]
                asks = [[asked[key] for key in chunk] for chunk in chunks]
                if pool is None:
                    reads = _serial(asks, lambda cells, stage=stage, stage_budget=stage_budget:
                                    _read_cells(reg, cells, row, col, items, leaf, stage, budget,
                                                stage_budget, turns, clean, hidden_side,
                                                result.unmodelled, cost, stop))
                else:
                    reads = pool.cells(asks, stage, stage_budget, work, result.unmodelled)
                done = 0
                try:
                    for index, got in reads:
                        if got is _TICK:
                            # Waiting on the workers: the caller's stop, and the wall clock.
                            if run.stopped() or (run.kind == "wall" and run.left_ms(work) < 0):
                                abandoned = True
                                break
                            continue
                        if got is _STOPPED:
                            abandoned = True
                            break
                        chunk = chunks[index]
                        for key, v in zip(chunk, got, strict=True):
                            memo[key] = v
                        fresh += len(chunk)
                        done += len(chunk)
                        left = len(keys) - done
                        per = (run.spent_ms(work) - stage_began) / max(fresh, 1)
                        # What the rest will take: at the rate so far, or -- the workers on
                        # the wall clock -- at the workers' own rate, spread over them.
                        need = (pool.eta_ms(left) if pool is not None and run.kind == "wall"
                                else per * left)
                        if (run.stopped() or run.left_ms(work) < 0
                                or need > run.left_ms(work)):
                            abandoned = True
                            break
                finally:
                    reads.close()
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
            # What the stage cost, counted (on the count clock the clock's own reading).
            spent = cost.ms(work) - count_began
            stages_count += spent
            stages_wall += run.wall_ms() - wall_began
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
    if pool is not None:
        pool.end()
        result.pool = dict(pool.stats, parentCpuMs=(time.process_time() - cpu_began) * 1000.0,
                           wallMs=run.wall_ms())
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
                                     child_q=stage.child if stage.q else None, stack=STACK)
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
    if BATCH_CHILDREN and stage.sub.sub is None and work is not None:
        listed = list(menus)
        mark = dict(work)
        got = _children_at_once(reg, branches, weights, listed, leaf, stage, root_budget, budget,
                                unmodelled, stop, work)
        if got is not _ALONE:
            return got
        # Something the one-at-a-time road meets on its own terms: take it, as it was.
        work.clear()
        work.update(mark)
        menus = iter(listed)
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


#: Read a depth-3 cell's children together (IKA-367): their matrices in one crossing and one
#: call to the leaf, and each pass of their depth-2 stages as one `search._refine_cells`
#: (the Q asked in the groups the one-at-a-time road asks it in). The same values, the same
#: counted work; off reads every child as a root of its own (`read`), the reference.
BATCH_CHILDREN = True

#: Score the leaves a stage gathers in one forward pass (`port.score_stacked`) rather than
#: a pass per child game (IKA-367; the user's call on 9/28: the ladder is new and off by
#: default, so it has no games to keep to the last bit, and a reading is not slowed for a
#: determinism it does not need). Deterministic still: the same position, clock and budget
#: give the same answer, whatever the threads. Off: `score_segments`, each game's pass its
#: own, the answer `BATCH_CHILDREN` must match to the bit.
STACK = True

#: `_children_at_once`'s answer when a child needs the one-at-a-time road: a side with no
#: action, a refused node or cell, an unsolvable matrix.
_ALONE = object()


@dataclass
class _Child:
    """One live child of a depth-3 cell, read by its stage (`read` with one stage, K = 1)."""

    position: Position
    row: list
    col: list
    prices: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rows: list[int] = field(default_factory=list)
    cols: list[int] = field(default_factory=list)
    trial: np.ndarray | None = None
    memo: dict = field(default_factory=dict)
    answer: tuple | None = None
    active: bool = True


def _children_at_once(  # noqa: PLR0913, PLR0912, C901 - a cell's children through one stage
    reg: Any,  # noqa: ANN401
    branches: list,
    weights: np.ndarray,
    menus: list,
    leaf: Any,  # noqa: ANN401
    stage: Stage,
    root_budget: Budget,
    budget: Budget,
    unmodelled: set[str],
    stop: Any,  # noqa: ANN401
    work: dict[str, int],
) -> float | None | object:
    """`_deep_cell`'s loop over the children with ``stage.sub`` of depth 2, every child at once
    (`BATCH_CHILDREN`): each step is `read`'s for one child, the arithmetic written the same
    way, so each child's value is the one its own `read` gives. `_ALONE` where that road
    would stop or fall back on its own."""
    sub = stage.sub
    kids: list[_Child] = []
    listed = iter(menus)
    for branch in branches:
        if branch.position.ended:
            continue
        crow, ccol = next(listed)
        if not crow or not ccol:
            return _ALONE
        kids.append(_Child(branch.position, list(crow), list(ccol), None, None, None))
    if stop is not None and stop.is_set():
        raise Stopped
    if kids:
        pending = port.pending_payoffs(reg, [(k.position, k.row, k.col) for k in kids], leaf,
                                       budget=budget)
        if pending is None or any(isinstance(p, port.PortRefused) for p in pending):
            return _ALONE
        scores = (port.score_stacked if STACK else port.score_segments)(
            leaf, [p.encoded for p in pending])
        for p, got in zip(pending, scores, strict=True):
            p.scored(got)
        for kid, p in zip(kids, pending, strict=True):
            m = np.asarray(p.finish(), dtype=np.float64)
            try:
                eq = solve(m)
            except EquilibriumError:
                return _ALONE
            unmodelled.update(p.unmodelled)
            work["subgames"] += 1
            work["cells"] += m.size
            kid.prices, kid.x, kid.y = m, np.asarray(eq.row_strategy), np.asarray(eq.col_strategy)
    w = np.asarray([1.0], dtype=np.float64)
    w = w / w.sum()
    sub_budget = replace(root_budget, enumerate_knockouts=True) if sub.knockouts else root_budget
    for kid in kids:
        prices = [np.array(kid.prices, dtype=np.float64, copy=True)]
        x = np.asarray(kid.x, dtype=np.float64)
        ys = [np.asarray(kid.y, dtype=np.float64)]
        kid.rows = _order(x, sum(w[k] * (prices[k] @ ys[k]) for k in range(1)), sub.rect,
                          larger=True)
        kid.cols = _order(ys[0], x @ prices[0], sub.rect, larger=False)
        kid.trial = prices[0].copy()
    for attempt in range(sub.passes + 1):
        cells: list = []
        groups: list[int] = []
        owners: list = []
        for kid in kids:
            if not kid.active:
                continue
            asked: list[tuple[int, int]] = []
            for i in kid.rows:
                for j in kid.cols:
                    if (i, j) not in kid.memo and (i, j) not in asked:
                        asked.append((i, j))
            for at in range(0, len(asked), CHUNK):
                chunk = asked[at:at + CHUNK]
                groups.append(len(chunk))
                cells += [(kid.position, kid.row[i], kid.col[j]) for i, j in chunk]
                owners += [(kid, cell) for cell in chunk]
        if stop is not None and stop.is_set():
            raise Stopped
        if cells:
            try:
                found = search._refine_cells(reg, cells, leaf, budget=sub_budget,
                                             sub_limit=sub.child, sub_branches=sub.branches,
                                             child_q=sub.child if sub.q else None,
                                             q_groups=groups, stack=STACK)
            except port.PortRefused:
                return _ALONE
            for (kid, cell), (value, _notes, _solved) in zip(owners, found, strict=True):
                kid.memo[cell] = value
        for kid in kids:
            if not kid.active:
                continue
            trial = [kid.trial]
            for i in kid.rows:
                for j in kid.cols:
                    v = kid.memo[(i, j)]
                    if v is not None:
                        trial[0][i, j] = v
            try:
                restricted = solve_bayesian([trial[0][np.ix_(kid.rows, kid.cols)]], w)
            except EquilibriumError:
                kid.active = False
                continue
            sx = np.zeros(len(trial[0]), dtype=np.float64)
            sx[kid.rows] = restricted.row_strategy
            y = np.zeros(trial[0].shape[1], dtype=np.float64)
            y[kid.cols] = restricted.col_strategies[0]
            sys_ = [y]
            col_ev = [sx @ trial[k] for k in range(1)]
            row_ev = sum(w[k] * (trial[k] @ sys_[k]) for k in range(1))
            guarantee = float(sum(w[k] * float(col_ev[k].min()) for k in range(1)))
            kid.answer = (sx, sys_, guarantee)
            if attempt == sub.passes:
                kid.active = False
                continue
            grew = False
            best_row = int(np.argmax(row_ev))
            if best_row not in kid.rows and float(row_ev[best_row]) > float(
                    restricted.value) + search.ORACLE_TOLERANCE:
                kid.rows.append(best_row)
                grew = True
            earned = float(col_ev[0] @ sys_[0])
            best_col = int(np.argmin(col_ev[0]))
            if best_col not in kid.cols and float(
                    col_ev[0][best_col]) < earned - search.ORACLE_TOLERANCE:
                kid.cols.append(best_col)
                grew = True
            if not grew:
                kid.active = False
        if not any(kid.active for kid in kids):
            break
    values = []
    live = iter(kids)
    for branch in branches:
        if branch.position.ended:
            values.append(float(np.asarray(leaf([branch.position]))[0]))
            continue
        kid = next(live)
        work["reads"] += 1
        if kid.answer is None:
            return None
        values.append(float(kid.answer[2]))
    return float(np.asarray(values) @ weights)


# ------------------------------------------------------------ one position on every core
#
# IKA-364: a stage's cells do not depend on each other -- each is its own turn, its
# children's games and their leaves -- and the only synchronisation is the stage's LP. So a
# position is read on every core by handing a stage's cells to worker processes (each its
# own port, GIL and, through the machine's inference server, the leaf and the Q: IKA-363)
# while the reading process keeps the stages, the memo, the clock and the LPs.

#: `_serial`'s and `_Pool.cells`' answer while the workers have not returned the next
#: chunk: the reader looks at its stop and its wall clock, and waits on.
_TICK = object()
#: A chunk whose cell met the caller's stop (`Stopped`).
_STOPPED = object()

#: Seconds the reader waits on the workers before it looks at its clock again.
TICK_SECONDS = 0.02

#: Cells a worker is sent at once from a depth-2 stage. None: the stage spread over the
#: workers, one chunk each (at most `CHUNK`); `CHUNK`: the serial read's chunks, so the
#: same forward passes (the Q's and the stacked leaf's move with their batch), and on the
#: count clock the serial read to the bit.
POOL_CHUNK: int | None = None

#: With `POOL_CHUNK` None: chunks a worker gets from a stage, about. More chunks wait less
#: on a stage's slowest chunk and send more round trips (``POKEURAOU_LADDER_SPREAD``).
#: IKA-364, 16 threads, 7 positions at 8 s: 1 -> 4 took the workers' wait on a stage's
#: last chunks from 45% to 19% of their time and the server's share from 16% to 28%.
POOL_SPREAD = int(os.environ.get("POKEURAOU_LADDER_SPREAD", "4"))

#: Chunks a worker holds at once: one it reads, one waiting, so it never waits on a trip.
POOL_DEPTH = 2

#: What the workers did (the positive control that the pool read the cells): chunks and
#: cells read, reads begun, and chunks read past the stop and thrown away.
POOL_COUNTS: dict[str, int] = {"chunks": 0, "cells": 0, "reads": 0, "dropped": 0}


def _serial(asks: list[list], read_one: Callable[[list], list]) -> Any:  # noqa: ANN401 - a generator
    """The chunks read here, one after the other: ``(index, values)``, `_STOPPED` where a
    cell met the caller's stop."""
    for index, cells in enumerate(asks):
        try:
            got = read_one(cells)
        except Stopped:
            yield index, _STOPPED
            return
        yield index, got


def _pool_main(conn: Any, format_id: str, factory: Any, args: tuple[Any, ...],  # noqa: ANN401
               cancel: Any) -> None:  # noqa: ANN401 - a multiprocessing Event
    """A worker: its own regulation, leaf (``factory(reg, *args)``, which also installs the
    Q its children's menus need) and a port of one cell thread; `_read_cells` on each chunk
    of the read it was last told of. ``cancel`` is its stop: a depth-3 cell met by it is
    dropped (`Stopped`)."""
    import traceback

    from . import rustnode
    from .damage import register_mega_stones
    from .regulation import load_regulation

    reg = load_regulation(format_id)
    register_mega_stones(reg)
    rustnode.set_port_threads(1)
    try:
        leaf = factory(reg, *args)
    except Exception as error:  # noqa: BLE001 - said to the parent, which reads without it
        conn.send(("failed", f"{type(error).__name__}: {error}"))
        return
    conn.send(("ready", None))
    context: tuple | None = None
    turns: dict[bool, dict] = {False: {}, True: {}}
    clean: dict[tuple[int, int], bool] = {}
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        if message[0] == "read":
            # A new read: its menus, completions and budget. The shared turns and the cells'
            # bench reach are keyed by the read's own indices, so they start again.
            context = message[1:]
            turns = {False: {}, True: {}}
            clean = {}
            continue
        _kind, task, stage, stage_budget, cells = message
        row, col, items, root_budget, hidden_side, cost = context
        work = _zero()
        notes: set[str] = set()
        if cancel.is_set():
            # Sent before the reader stopped: not read.
            conn.send(("stopped", task, None, work, notes, (0.0, 0.0, 0)))
            continue
        search.WORK = work
        clock = time.perf_counter()
        served = _served(leaf)
        try:
            got = _read_cells(reg, cells, row, col, items, leaf, stage, root_budget,
                              stage_budget, turns, clean, hidden_side, notes, cost, cancel)
            status = "done"
        except Stopped:
            got, status = None, "stopped"
        except Exception as error:  # noqa: BLE001 - raised again in the reader
            got, status = (f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
                           "error")
        finally:
            search.WORK = None
        after = _served(leaf)
        took = (time.perf_counter() - clock, after[0] - served[0], after[1] - served[1])
        conn.send((status, task, got, work, notes, took))


def _served(leaf: Any) -> tuple[float, int]:  # noqa: ANN401
    """Seconds this process has waited on the inference server, and its round trips: the
    leaf's and the Q's (0 for a leaf and a Q held here)."""
    from . import qrank

    q = qrank._INSTALLED.get("")
    waited = float(getattr(leaf, "waited", 0.0)) + float(getattr(q, "waited", 0.0))
    trips = int(getattr(leaf, "calls", 0)) + int(getattr(q, "trips", 0))
    return waited, trips


class _Pool:
    """Worker processes reading stages' cells (`start_pool`)."""

    def __init__(self, processes: list, conns: list, cancel: Any) -> None:  # noqa: ANN401
        self.processes = processes
        self.conns = conns
        self.cancel = cancel
        self.stats: dict[str, float] = {}
        self._got_ms, self._got_cells = 0.0, 0

    def alive(self) -> bool:
        return bool(self.conns) and all(p.is_alive() for p in self.processes)

    def eta_ms(self, cells: int) -> float:
        """Wall milliseconds ``cells`` more of the chunks in hand will take: the workers'
        own milliseconds a cell so far, spread over them."""
        per = self._got_ms / max(self._got_cells, 1)
        return per * cells / len(self.conns)

    def chunk(self, cells: int) -> int:
        if POOL_CHUNK is not None:
            return POOL_CHUNK
        return max(1, min(CHUNK, -(-cells // (POOL_SPREAD * len(self.conns)))))

    def begin(self, row: Sequence[Any], col: Sequence[Any], items: Sequence[Any],
              budget: Budget, hidden_side: int, cost: LadderCost) -> None:
        plain = [Item(position=it.position, weight=float(getattr(it, "weight", 1.0)),
                      exact=bool(getattr(it, "exact", True)),
                      slots=tuple(getattr(it, "slots", ()) or ())) for it in items]
        for conn in self.conns:
            conn.send(("read", list(row), list(col), plain, budget, hidden_side, cost))
        POOL_COUNTS["reads"] += 1
        #: This read's: chunks taken, the reader's milliseconds blocked on the workers, in
        #: sends and in receives (unpickling included), and the workers' own reading.
        #: The workers' idle time, by what they waited for: ``supplyMs`` the reader's
        #: sends (chunks left to send), ``tailMs`` the stage's last chunks on the other
        #: workers (all sent, theirs back), ``betweenMs`` the reader between two stages' cells
        #: (its LP, the next rectangle), and ``serverMs`` / ``serverTrips`` the inference
        #: server inside their reading.
        self.stats = {"chunks": 0, "waitMs": 0.0, "sendMs": 0.0, "recvMs": 0.0,
                      "workerMs": 0.0, "serverMs": 0.0, "serverTrips": 0, "supplyMs": 0.0,
                      "tailMs": 0.0, "betweenMs": 0.0}
        self._last_end = time.perf_counter()

    def end(self) -> None:
        """The read is over: the workers idle since the last stage's cells were back."""
        self.stats["betweenMs"] += (time.perf_counter() - self._last_end) * 1000.0 * len(
            self.conns)

    def cells(self, asks: list[list], stage: Stage, stage_budget: Budget,
              work: dict[str, int], notes: set[str]) -> Any:  # noqa: ANN401 - a generator
        """`_serial`'s answers from the workers: each chunk's values in the order asked,
        its counted work and notes added as it is taken; `_TICK` while waiting. Closed
        early (the reader stopped), the chunks still out are cancelled and thrown away."""
        from multiprocessing.connection import wait

        out: dict[int, list[int]] = {i: [] for i in range(len(self.conns))}
        which = {id(conn): i for i, conn in enumerate(self.conns)}
        results: dict[int, tuple] = {}
        sent = taken = 0
        #: The workers' milliseconds and cells of the chunks back so far (`eta_ms`).
        self._got_ms, self._got_cells = 0.0, 0
        now = time.perf_counter()
        self.stats["betweenMs"] += (now - self._last_end) * 1000.0 * len(self.conns)
        idle_since = [now] * len(self.conns)
        try:
            while taken < len(asks):
                while sent < len(asks):
                    i = min(out, key=lambda k: len(out[k]))
                    if len(out[i]) >= POOL_DEPTH:
                        break
                    clock = time.perf_counter()
                    if not out[i]:
                        self.stats["supplyMs"] += (clock - idle_since[i]) * 1000.0
                    self.conns[i].send(("cells", sent, stage, stage_budget, asks[sent]))
                    self.stats["sendMs"] += (time.perf_counter() - clock) * 1000.0
                    out[i].append(sent)
                    sent += 1
                if taken in results:
                    status, values, done_work, done_notes = results.pop(taken)
                    self.stats["chunks"] += 1
                    for kind, n in done_work.items():
                        work[kind] += n
                    notes.update(done_notes)
                    POOL_COUNTS["chunks"] += 1
                    POOL_COUNTS["cells"] += len(asks[taken])
                    index = taken
                    taken += 1
                    yield index, (_STOPPED if status == "stopped" else values)
                    continue
                clock = time.perf_counter()
                ready = wait(self.conns, timeout=TICK_SECONDS)
                self.stats["waitMs"] += (time.perf_counter() - clock) * 1000.0
                if not ready:
                    yield None, _TICK
                    continue
                for conn in ready:
                    clock = time.perf_counter()
                    status, task, values, done_work, done_notes, times = conn.recv()
                    back = time.perf_counter()
                    self.stats["recvMs"] += (back - clock) * 1000.0
                    took, server_s, trips = times
                    self.stats["workerMs"] += took * 1000.0
                    self.stats["serverMs"] += server_s * 1000.0
                    self.stats["serverTrips"] += trips
                    self._got_ms += took * 1000.0
                    self._got_cells += len(asks[task])
                    i = which[id(conn)]
                    out[i].remove(task)
                    if not out[i]:
                        idle_since[i] = back
                    if status == "error":
                        raise RuntimeError(f"a ladder worker failed on a chunk: {values}")
                    results[task] = (status, values, done_work, done_notes)
        finally:
            self._drain(out)
            self._last_end = time.perf_counter()
            if sent == len(asks):
                for since in idle_since:
                    self.stats["tailMs"] += max(0.0, self._last_end - since) * 1000.0

    def _drain(self, out: dict[int, list[int]]) -> None:
        if not any(out.values()):
            return
        self.cancel.set()
        try:
            for i, tasks in out.items():
                for _ in tasks:
                    self.conns[i].recv()
                    POOL_COUNTS["dropped"] += 1
                tasks.clear()
        finally:
            self.cancel.clear()

    def close(self) -> None:
        for process, conn in zip(self.processes, self.conns, strict=True):
            with contextlib.suppress(OSError, EOFError):
                conn.send(None)
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
            conn.close()


#: The worker processes a top-level read hands its cells to, or None: read here.
_POOL: _Pool | None = None


def start_pool(reg: Any, count: int, factory: Any, args: tuple[Any, ...] = ()) -> int:  # noqa: ANN401
    """Start ``count`` worker processes that read the ladder's cells (IKA-364), each with
    the leaf ``factory(reg, *args)`` -- a module-level function, so a spawned process can
    import it, that also installs the Q the children's menus are ranked by -- and a port of
    one thread. Kept for the process's life (`stop_pool`). The leaf must answer as the
    reader's own does (the same model; the machine's inference server keeps one copy of it
    for them all). Returns how many are ready; 0 reads every cell here."""
    import multiprocessing

    global _POOL  # noqa: PLW0603 - the process's one pool
    stop_pool()
    if count <= 0:
        return 0
    context = multiprocessing.get_context("spawn")
    cancel = context.Event()
    started = []
    for _ in range(count):
        mine, theirs = context.Pipe()
        process = context.Process(target=_pool_main,
                                  args=(theirs, reg.meta.format_id, factory, tuple(args), cancel),
                                  daemon=True)
        process.start()
        theirs.close()
        started.append((process, mine))
    ready = []
    for process, conn in started:
        try:
            state, why = conn.recv()
        except EOFError:
            state, why = "failed", "the worker exited"
        if state == "ready":
            ready.append((process, conn))
        else:
            print(f"[ladder] a worker did not start: {why}", flush=True)
            process.join(timeout=5)
    if ready:
        _POOL = _Pool([p for p, _c in ready], [c for _p, c in ready], cancel)
    return len(ready)


def stop_pool() -> None:
    """Stop the worker processes (`start_pool`)."""
    global _POOL  # noqa: PLW0603
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def pool_workers() -> int:
    """How many worker processes read the ladder's cells (0: none)."""
    return 0 if _POOL is None else len(_POOL.conns)


__all__ = ["LADDERS", "LADDER_COSTS", "Item", "LadderCost", "LadderResult", "Rung", "Stage",
           "Stopped", "parse_ladder", "parse_stage", "pool_workers", "read", "start_pool",
           "stop_pool"]
