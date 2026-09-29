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
cost does not fit what is left, and is abandoned once what it has spent says it will not --
except on the wall clock for a ladder that fills its budget (`FILLS`, IKA-376: L6, whose
stages a rule writes past L5's, `unending`), which begins every stage while time is left.

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

from . import port, portlp, portmenus, portserved, rustnode, search, subshare
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


def _level(rect: int, branches: str, child: int) -> str:
    return f"r{rect}b{branches}k{child}"


def unending(depth: int = 4, *, last: int = 9) -> tuple[str, ...]:
    """The stages a rule writes past a hand-written ladder, depth ``depth`` to ``last``
    (IKA-376): at each depth D the rounds L5 wrote for depth 4 and IKA-369's extensions
    asked for, each a few times the one before it --

    1. narrow: the root's 4 x 4 at three branches, each level below 3 x 3 at three
       branches, the children at the Q's 24 and the lowest level's at 16;
    2. the root's 8 x 8 (the level below it 4 x 4) -- L5's two depth-4 stages;
    3. every branch at the root's cells;
    4. every branch at every level, the lowest level's children at 24;
    5. the children wider (the level below the root 6 x 6, the next 4 x 4) -- IKA-369: the
       root wider moved no answer (12 x 12 at depth 4: 0.0000), the children wider and depth
       5 did (0.0025, 0.0006), so the root stays at 8;

    then depth D + 1 from step 1. The stage syntax stops at depth 9, which no machine
    reaches (a depth-5 cell reads a depth-4 read per child), so the ladder never runs out
    in a move or an analysis."""
    out: list[str] = []
    for d in range(depth, last + 1):
        below = d - 2  # the levels under the root's cells

        def stage(top: int, top_b: str, rects: list[int], b: str, low_k: int, d: int = d,
                  below: int = below) -> str:
            levels = [_level(rects[i], b, 24 if i < below - 1 else low_k) for i in range(below)]
            return f"d{d}r{top}b{top_b}k24x/" + "/".join(levels)

        narrow = [3] * below
        eight = [4] + [3] * (below - 1)
        wide = ([6, 4] + [3] * (below - 2))[:below]
        out += [stage(4, "3", narrow, "3", 16), stage(8, "3", eight, "3", 16),
                stage(8, "a", eight, "3", 16), stage(8, "a", eight, "a", 24),
                stage(8, "a", wide, "a", 24)]
    return tuple(out)


# L5, then the rule (`unending`) from where L5 leaves depth 4: its stages never run out, and
# on the wall clock it fills the budget (`FILLS`).
LADDERS["L6"] = LADDERS["L5"] + unending(4)[2:]

#: Ladders that fill a wall-clock budget (`FILL_WALL` for one ladder): a stage is begun while
#: time is left and given up at the budget, never left unbegun on a prediction (IKA-376:
#: L5's 12 stages were read in 46% of a 64 s budget, IKA-369).
FILLS = frozenset({"L6"})


class Ladder(tuple):
    """A ladder's stages; ``fills``: it fills a wall-clock budget (`FILLS`)."""

    fills: bool = False


def parse_ladder(spec: str) -> tuple[Stage, ...]:
    """A named ladder (`LADDERS`) or stages joined by ``+``."""
    labels = LADDERS.get(spec) or tuple(spec.split("+"))
    out = Ladder(parse_stage(label) for label in labels)
    out.fills = spec in FILLS
    return out


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
    #: IKA-370: what the stage was predicted to take before it began (the clock's units).
    predicted_ms: float = 0.0
    #: The counted work at its end (`LadderCost`'s kinds): what `fit` prices.
    work: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"stage": self.stage, "value": round(self.value, 6), "rows": self.rows,
                "cols": list(self.cols), "fresh": self.fresh,
                "spentMs": round(self.spent_ms, 1), "wallMs": round(self.wall_ms, 1),
                "optimism": round(self.optimism, 6), "work": dict(self.work),
                "predictedMs": round(self.predicted_ms, 1)}


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
    #: IKA-370: the unfinished stage's prediction (the clock's units) and what was left.
    next_predicted_ms: float = 0.0
    next_left_ms: float = 0.0
    #: IKA-370: the abandoned stage's cells read before it was given up, and its cells.
    cut: tuple[int, int] = (0, 0)
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
    #: IKA-370: this process's waits on its own port in the read (ms) and the reads waited
    #: on, and its CPU (ms): the serial read's side of the workers' breakdown.
    here: dict[str, float] = field(default_factory=dict)

    @property
    def depth_reached(self) -> str:
        return self.rungs[-1].stage if self.rungs else "d1"

    def to_json(self) -> dict[str, Any]:
        return {"rungs": [r.to_json() for r in self.rungs], "stopped": self.stopped,
                "unfinished": self.unfinished, "abandoned": self.abandoned,
                "nextPredictedMs": round(self.next_predicted_ms, 1),
                "nextLeftMs": round(self.next_left_ms, 1), "cut": list(self.cut),
                "spentMs": round(self.spent_ms, 1), "wallMs": round(self.wall_ms, 1),
                "work": dict(self.work),
                **({"workers": self.workers,
                    "pool": {k: round(v, 1) for k, v in self.pool.items()}}
                   if self.workers else {}),
                **({"here": {k: round(v, 1) for k, v in self.here.items()}} if self.here else {})}


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

#: On the wall clock a read fills its budget (IKA-370): every stage is begun while time is
#: left, and one is given up only when the budget is spent (or the caller stops). The
#: answer is the last completed stage either way, so a stage predicted not to fit costs
#: nothing but the time the read would otherwise have handed back unused -- and the
#: predictions were far off: the first depth-3 stage 5-9 times its wall (the a-priori
#: guess 80-400 times the counted cost), so 8 s reads stopped at 3.5-6.8 s with it
#: unbegun. The count clock keeps its predictions (its reads are the references).
#: With the worker processes only (they are stopped within 20 ms; a cell read here is not).
#: Off by default (``POKEURAOU_LADDER_FILL=1`` turns it on): at 8 s on 16 threads it
#: completed a further stage in 12 of 24 open and 6 of 17 hidden positions and moved no
#: reference's loss past its error (open deep -0.0014 se 0.0024, R24 +0.0031 se 0.0021;
#: hidden deep -0.0012 se 0.0012) -- the stages it reaches are depth 3 and 4, which these
#: references cannot score (IKA-364, IKA-367 §6).
FILL_WALL = os.environ.get("POKEURAOU_LADDER_FILL", "0") != "0"


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
    memo: dict[tuple, float | None] | None = None,
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

    ``memo`` (IKA-369): the caller's table of cells read, filled in place and read again --
    a read stopped and begun again with the same table reads no cell twice (the long
    reference, `tools/position_set.py ladder-ref`). With it, a worker's cells are kept as
    they arrive, also those a stop throws away before their turn; the answer is the same
    (a cell's value does not depend on when it was read). None: a table of this read's own.
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
    kept = memo is not None
    memo = {} if memo is None else memo
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
    # IKA-378 (`SHARE`): a read's child sub-games, each filled once -- here in a table of the
    # read's own, on the workers in theirs (`_Pool.begin`).
    shared_before = (search.SHARE, rustnode.DIGESTS[0])
    if outer is None and pool is None and SHARE:
        search.SHARE = subshare.Local()
        rustnode.DIGESTS[0] = True
    here_began = (tuple(rustnode.PORT_WAITED), time.process_time(), _served(leaf),
                  dict(subshare.COUNTS), portlp.COUNTS["lps"], portserved.COUNTS["requests"],
                  portmenus.COUNTS["asked"])
    if pool is not None:
        pool.reg = reg
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

    def per_cell_ms(stage: Stage, guess: float) -> float:
        """A fresh cell of ``stage``, predicted in counted milliseconds."""
        if stage.kind in measured:
            return measured[stage.kind][0] / max(measured[stage.kind][1], 1)
        have = calib.get(stage.depth) or (
            (sum(v[0] for v in calib.values()), sum(v[1] for v in calib.values()))
            if calib else None)
        ratio = 1.0 if not have or have[1] <= 0 else have[0] / have[1]
        # IKA-364: the depth-3 guesses run 100-200 times the counted cost (1,794 ms a cell
        # guessed for d3r8b3k24x, 12-24 counted), so a floor of 0.05 predicted the next
        # depth-3 stage at 4-8 times its cost and a 32 s read stopped at 5-14 s.
        return guess * min(max(ratio, 0.002), 3.0)

    def speed() -> float:
        # The prediction is in counted milliseconds; on the wall clock it is scaled by this
        # read's wall a counted millisecond so far (the workers read several at once:
        # IKA-364), 1 before a stage has completed.
        return stages_wall / stages_count if run.kind == "wall" and stages_count > 0 else 1.0

    # IKA-376: a ladder named in `FILLS` fills the budget as `FILL_WALL` does.
    fills = (run.kind == "wall" and (FILL_WALL or getattr(stages, "fills", False))
             and pool is not None)

    def begins(stage: Stage, cells: int) -> bool:
        """IKA-374 (`AHEAD`): whether a stage of ``cells`` fresh cells would be begun now."""
        if fills:
            return run.left_ms(work) > 0
        return (per_cell_ms(stage, _guess_cell_ms(stage, cost)) * cells * speed()
                <= run.left_ms(work))

    try:
        for at_stage, stage in enumerate(stages):
            if run.stopped():
                result.stopped, result.unfinished = "stop", stage.label
                break
            stage_budget = replace(budget, enumerate_knockouts=True) if stage.knockouts else budget
            row_ev0 = sum(w[k] * (prices[k] @ ys[k]) for k in range(kinds))
            rows = _order(x, row_ev0, stage.rect, larger=True)
            cols = [_order(ys[k], x @ prices[k], stage.rect, larger=False) for k in range(kinds)]
            trial = [p.copy() for p in prices]
            guess = _guess_cell_ms(stage, cost)
            per_cell = per_cell_ms(stage, guess)
            fresh_keys = {_key(side, i, j, k, stage) for k in range(kinds) for i in rows
                          for j in cols[k]} - set(memo)
            # IKA-374: less the cells read ahead and back (none on the count clock).
            held = pool.ready(fresh_keys) if pool is not None else 0
            predicted = per_cell * (len(fresh_keys) - held) * speed()
            result.next_predicted_ms, result.next_left_ms = predicted, run.left_ms(work)
            looks_ahead = run.kind == "wall" and AHEAD and pool is not None
            tail = run.kind == "wall" and TAIL and pool is not None
            split = run.kind == "wall" and SPLIT and pool is not None and stage.sub is not None
            if predicted > run.left_ms(work) and not (fills and run.left_ms(work) > 0):
                result.stopped, result.unfinished = "budget", stage.label
                break
            if pool is not None and pool._trace is not None:
                ahead = stages[at_stage + 1] if at_stage + 1 < len(stages) else None
                pool.note({"stage": stage.label, "at": at_stage, "rows": list(rows),
                           "cols": [list(c) for c in cols], "fresh": len(fresh_keys),
                           "predicted": round(predicted, 1),
                           **({"next": ahead.label, "sameKind": ahead.kind == stage.kind,
                               "staleRows": _order(x, row_ev0, ahead.rect, larger=True),
                               "staleCols": [_order(ys[k], x @ prices[k], ahead.rect, larger=False)
                                             for k in range(kinds)]} if ahead else {})})
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
                done = 0
                adopt: dict[int, tuple[int, list[tuple]]] = {}
                head: list[list[tuple]] = []
                spec_ready: dict = {}
                if pool is not None:
                    # IKA-370, IKA-374: the cells an earlier tail read ahead for this one --
                    # back (taken now) and still out (waited for, as chunks of their own).
                    spec_ready, spec_flying = pool.claim(keys)
                    for key, (value, done_work, done_notes) in spec_ready.items():
                        memo[key] = value
                        for kind, n in done_work.items():
                            work[kind] += n
                        result.unmodelled.update(done_notes)
                    fresh += len(spec_ready)
                    done += len(spec_ready)
                    flying = {k for wanted in spec_flying.values() for k in wanted}
                    keys = [k for k in keys if k not in spec_ready and k not in flying]
                    for task, wanted in spec_flying.items():
                        adopt[len(head)] = (task, wanted)
                        head.append(wanted)
                if pool is None or stage.sub is not None:
                    chunks = head + [keys[at:at + step] for at in range(0, len(keys), step)]
                else:
                    chunks = head + pool.split(keys, shrink=tail, of=len(asked))
                keys = [k for chunk in chunks for k in chunk]
                asks = [[asked[key] for key in chunk] for chunk in chunks]
                if pool is None:
                    reads = _serial(asks, lambda cells, stage=stage, stage_budget=stage_budget:
                                    _read_cells(reg, cells, row, col, items, leaf, stage, budget,
                                                stage_budget, turns, clean, hidden_side,
                                                result.unmodelled, cost, stop))
                elif split and not adopt:
                    reads = (pool.deep_passes if PASSES else pool.deep_cells)(
                        asks, stage, stage_budget, work, result.unmodelled,
                        early=(lambda index, values, chunks=chunks: memo.update(
                            zip(chunks[index], values, strict=True))) if kept else None)
                else:
                    ahead = (_Ahead(side, stage, stages[at_stage + 1:at_stage + 2], budget,
                                    stage_budget, rows, cols, attempt, w, trial, memo, todo,
                                    set(asked), pool, (x, ys), begins, reg)
                             if looks_ahead else None)
                    reads = pool.cells(
                        asks, stage, stage_budget, work, result.unmodelled, adopt=adopt,
                        ahead=ahead, tail=tail,
                        early=(lambda index, values, chunks=chunks: memo.update(
                            zip(chunks[index], values, strict=True))) if kept else None)
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
                        left = len(keys) + len(spec_ready) - done
                        per = (run.spent_ms(work) - stage_began) / max(fresh, 1)
                        # What the rest will take: at the rate so far, or -- the workers on
                        # the wall clock -- at the workers' own rate, spread over them.
                        need = (pool.eta_ms(left) if pool is not None and run.kind == "wall"
                                else per * left)
                        if (run.stopped() or run.left_ms(work) < 0
                                or (need > run.left_ms(work) and not fills)):
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
                    # IKA-387: in the port with `portlp` on (the same answer to the bit).
                    restricted = portlp.solve_bayesian_one(
                        reg, [trial[k][np.ix_(rows, cols[k])] for k in range(kinds)], w)
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
                result.cut = (fresh, len(fresh_keys))
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
            if pool is not None:
                pool.note({"done": stage.label, "rows": list(rows), "cols": [list(c) for c in cols],
                           "support": [int(i) for i in np.flatnonzero(x > 1e-9)],
                           "replies": [[int(j) for j in np.flatnonzero(y > 1e-9)] for y in ys]})
            rung = Rung(stage=stage.label, value=answer[2], strategy=x, replies=tuple(ys),
                        rows=len(rows), cols=tuple(len(c) for c in cols), fresh=fresh,
                        spent_ms=run.spent_ms(work), wall_ms=run.wall_ms(), optimism=answer[3],
                        predicted_ms=predicted,
                        work=dict(work))
            result.rungs.append(rung)
            result.strategy, result.value, result.replies = x, answer[2], tuple(ys)
            result.prices = prices
            if on_rung is not None:
                on_rung(rung)
    finally:
        search.WORK = outer
        if outer is None:
            search.SHARE, rustnode.DIGESTS[0] = shared_before
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
    if outer is None:
        result.here = {"portMs": (rustnode.PORT_WAITED[0] - here_began[0][0]) * 1000.0,
                       "portReads": rustnode.PORT_WAITED[1] - here_began[0][1],
                       "cpuMs": (time.process_time() - here_began[1]) * 1000.0,
                       "serverMs": (_served(leaf)[0] - here_began[2][0]) * 1000.0,
                       "serverTrips": _served(leaf)[1] - here_began[2][1],
                       # IKA-378: the sub-games this process took rather than filled.
                       "sharedSubgames": sum(subshare.COUNTS[k] - here_began[3][k]
                                             for k in ("hits", "same")),
                       "sameSubgames": subshare.COUNTS["same"] - here_began[3]["same"],
                       "keptSubgames": subshare.COUNTS["puts"] - here_began[3]["puts"],
                       "sharedKids": subshare.COUNTS["kidHits"] - here_began[3]["kidHits"],
                       # IKA-381: the LPs this process's port solved (`portlp`).
                       "portLps": portlp.COUNTS["lps"] - here_began[4],
                       # IKA-386: the inference server's requests its port sent.
                       "portServed": portserved.COUNTS["requests"] - here_began[5],
                       # IKA-389: the children whose Q menus this process's port built.
                       "portMenus": portmenus.COUNTS["asked"] - here_began[6]}
    result.work = work
    if result.rungs:
        result.unmodelled.add(f"ladder read to {result.depth_reached} ({len(result.rungs)} "
                              f"stage(s)); stopped: {result.stopped}")
    return result


#: On the wall clock with the worker processes, a worker with nothing of the stage left to
#: read reads ahead (IKA-374, after IKA-370's one cell a trip): the depth-2 cells the
#: stage's oracle pass and the next stage will likely ask, most likely first, in chunks the
#: size the stage cuts, kept by the pool until a stage asks for them (`_Ahead`) -- a stage
#: takes a part of a chunk cell by cell (`search.WORK_CELLS`). The likely cells are read off
#: the answer the rectangle has on the values back so far (an LP on the partial rectangle):
#: 89-94% of the cells read ahead were taken (IKA-370's, off the answer before the stage:
#: 69-81%). A worker reads a stage's chunk before a chunk read ahead that came before it
#: (`_next_message`), and is given one only when it has nothing else. Not for a stage the
#: read will not begin, and not for depth 3 and up: a deeper cell runs for seconds and
#: looks at the stop only between its children, so the cells read ahead past the last
#: stage held the next read's workers (one read began 4.4 s late) and it read 3.58 against
#: 5.39 without them. A cell taken is counted when taken; a cell no stage takes is never
#: counted (the node time is the read's). The answer is still the last completed stage's,
#: and the count clock never reads ahead (its reads are the serial read to the bit). With
#: `TAIL`, against master (ABBA, 7 positions at 8 s): the workers' idle 27-31% -> 18-19%
#: (open, 16 threads) and 14-16% -> 5-6% (hidden), the node time a wall second 5.17 ->
#: 5.46 (open, 16) and 4.52 -> 4.73 (hidden, 8), the same at open 8 and hidden 16: the
#: work filling the idle makes every port read and server trip slower (8 physical cores).
#: ``POKEURAOU_LADDER_AHEAD=0`` turns it off.
AHEAD = os.environ.get("POKEURAOU_LADDER_AHEAD", "1") != "0"

#: On the wall clock with the worker processes, a stage's last chunks are cut smaller as the
#: cells left to send run out (never more than a worker's share of them), and go only to a
#: worker with nothing out, never behind a chunk in hand (IKA-374). ``POKEURAOU_LADDER_TAIL=0``
#: turns it off.
TAIL = os.environ.get("POKEURAOU_LADDER_TAIL", "1") != "0"


class _Ahead:
    """The chunks a stage's tail reads ahead (`AHEAD`): called by `_Pool.cells` for the next
    one, a list of ``(stage, stage_budget, key, cell)`` of one stage, or None when there is
    nothing left to read ahead.

    Made at the pass's start with its rectangle (``rows`` / ``cols``), the prices before it
    (``trial``), the memo it fills and its cells (``todo``, ``asked``); planned at the first
    call, with the values back by then: the rectangle's answer on them (the prices where a
    cell is not back), then -- when the stage has an oracle pass left -- the row and the
    columns that pass would add, and the next stage's rectangle as that answer orders it, in
    growing corners from the support."""

    def __init__(self, side: int, stage: Stage, after: Sequence[Stage], budget: Budget,  # noqa: PLR0913
                 stage_budget: Budget, rows: list[int], cols: list[list[int]], attempt: int,
                 w: np.ndarray, trial: list[np.ndarray], memo: dict, todo: list,
                 asked: set, pool: _Pool, before: tuple,
                 begins: Callable[[Stage, int], bool] | None = None,
                 reg: Any = None) -> None:  # noqa: ANN401 - a Regulation
        self.side, self.stage, self.budget, self.stage_budget = side, stage, budget, stage_budget
        #: IKA-387: the port that solves the answer's LP (`portlp`).
        self.reg = reg
        self.begins = begins
        self.after = after[0] if after else None
        self.rows, self.cols = list(rows), [list(c) for c in cols]
        self.attempt, self.w, self.trial, self.memo = attempt, w, trial, memo
        self.todo, self.asked, self.pool, self.before = todo, asked, pool, before
        self._chunks: Any = None

    def __call__(self) -> list | None:
        if self._chunks is None:
            self._chunks = iter(self._plan())
        for chunk in self._chunks:
            live = [c for c in chunk if c[2] not in self.memo and not self.pool.holds(c[2])]
            if live:
                return live
        return None

    def _answer(self) -> tuple[np.ndarray, list[np.ndarray], float, list[np.ndarray]]:
        side, kinds = self.side, len(self.trial)
        prices = [t.copy() for t in self.trial]
        for i, j, k in self.todo:
            v = self.memo.get(_key(side, i, j, k, self.stage))
            if v is not None:
                prices[k][i, j] = v if side == 0 else -v
        try:
            got = portlp.solve_bayesian_one(self.reg, [prices[k][np.ix_(self.rows, self.cols[k])]
                                                       for k in range(kinds)], self.w)
        except EquilibriumError:
            x, ys = self.before
            return x, list(ys), float("nan"), prices
        x = np.zeros(len(prices[0]), dtype=np.float64)
        x[self.rows] = got.row_strategy
        ys = []
        for k in range(kinds):
            y = np.zeros(prices[k].shape[1], dtype=np.float64)
            y[self.cols[k]] = got.col_strategies[k]
            ys.append(y)
        return x, ys, float(got.value), prices

    def _plan(self) -> list[list[tuple]]:
        side, kinds, w = self.side, len(self.trial), self.w
        x, ys, value, prices = self._answer()
        col_ev = [x @ prices[k] for k in range(kinds)]
        row_ev = sum(w[k] * (prices[k] @ ys[k]) for k in range(kinds))
        plan: list[list[tuple]] = []
        seen: set[tuple] = set()

        def cells(stage: Stage, budget: Budget, pairs: Any) -> list[tuple]:  # noqa: ANN401
            got = []
            for i, j, k in pairs:
                key = _key(side, i, j, k, stage)
                if key in seen or key in self.memo or key in self.asked or self.pool.holds(key):
                    continue
                seen.add(key)
                ours, theirs = (i, j) if side == 0 else (j, i)
                got.append((stage, budget, key, (k, ours, theirs)))
            return got

        def cut(stage: Stage, got: list[tuple]) -> None:
            size = 1 if stage.sub is not None else self.pool.chunk(len(got))
            plan.extend(got[at:at + size] for at in range(0, len(got), size))

        if self.attempt < self.stage.passes and value == value and self.stage.sub is None:
            # The oracle's pass: the row and the columns it would add on this answer.
            rows, cols = list(self.rows), [list(c) for c in self.cols]
            best_row = int(np.argmax(row_ev))
            if best_row not in rows and float(row_ev[best_row]) > value + search.ORACLE_TOLERANCE:
                rows.append(best_row)
            for k in range(kinds):
                earned = float(col_ev[k] @ ys[k])
                best_col = int(np.argmin(col_ev[k]))
                if best_col not in cols[k] and float(
                        col_ev[k][best_col]) < earned - search.ORACLE_TOLERANCE:
                    cols[k].append(best_col)
            cut(self.stage, cells(self.stage, self.stage_budget,
                                  [(i, j, k) for k in range(kinds) for i in rows for j in cols[k]]))
        if self.after is not None and self.after.sub is None:
            after = self.after
            budget = replace(self.budget, enumerate_knockouts=True) if after.knockouts else self.budget
            rows = _order(x, row_ev, after.rect, larger=True)
            cols = [_order(ys[k], col_ev[k], after.rect, larger=False) for k in range(kinds)]
            corners = []
            for r in range(max(len(rows), *(len(c) for c in cols))):
                for k in range(kinds):
                    corners += [(rows[r], j, k) for j in cols[k][:r + 1]] if r < len(rows) else []
                    corners += [(i, cols[k][r], k) for i in rows[:r]] if r < len(cols[k]) else []
            got = cells(after, budget, corners)
            # Not for a stage the read will not begin: its cells would be read for nothing,
            # and still be out when the next read begins.
            if self.begins is None or self.begins(after, len(got)):
                cut(after, got)
        return plan


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
                                       clean=clean[(i, j)], completion=k))
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
    got = _deep_expand(reg, pos, ours, theirs, stage, budget, unmodelled)
    if got is None:
        return None
    branches, weights, listed = got
    return _deep_children(reg, branches, weights, listed, leaf, stage, root_budget, budget,
                          unmodelled, cost, stop)


def _deep_expand(  # noqa: PLR0913 - one cell and its stage
    reg: Any,  # noqa: ANN401
    pos: Position,
    ours: Any,  # noqa: ANN401
    theirs: Any,  # noqa: ANN401
    stage: Stage,
    budget: Budget,
    unmodelled: set[str],
) -> tuple[list, np.ndarray, list] | None:
    """`_deep_cell`'s first step: the cell's turn, its kept branches and their weights, and
    each live child's menus (in the live children's order). None: the port refused the turn
    or kept nothing."""
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
        listed = _q_menus(reg, live, stage.child) if live else []
        if work is not None and live:
            work["qs"] += 1
    else:
        listed = [(narrow(reg, p, 0, limit=stage.child).actions,
                   narrow(reg, p, 1, limit=stage.child).actions) for p in live]
    return branches, weights, list(listed)


def _deep_children(  # noqa: PLR0913 - a cell's children and their stage
    reg: Any,  # noqa: ANN401
    branches: list,
    weights: np.ndarray,
    listed: list,
    leaf: Any,  # noqa: ANN401
    stage: Stage,
    root_budget: Budget,
    budget: Budget,
    unmodelled: set[str],
    cost: LadderCost,
    stop: Any,  # noqa: ANN401
) -> float | None:
    """`_deep_cell`'s second step: each child read by ``stage.sub`` (``listed`` its menus),
    the branches' values weighted."""
    work = search.WORK
    menus = iter(listed)
    if BATCH_CHILDREN and stage.sub.sub is None and work is not None:
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


def _deep_split(  # noqa: PLR0913 - one cell and its stage
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
) -> tuple:
    """IKA-375 (`SPLIT`): a deep cell's first step on a worker. ``("value", v)`` when the cell
    is read here to the end -- fewer than two children to read, or a child with an empty
    menu (the one-at-a-time road's own end) -- else ``("split", weights, entries, kids)``:
    per kept branch ``("leaf", value)`` (a battle over) or ``("kid", n)``, and each child's
    (position, row menu, column menu, digest), for `_deep_child` on any worker."""
    got = _deep_expand(reg, pos, ours, theirs, stage, budget, unmodelled)
    if got is None:
        return ("value", None)
    branches, weights, listed = got
    if len(listed) < 2 or any(not r or not c for r, c in listed):
        return ("value", _deep_children(reg, branches, weights, listed, leaf, stage, root_budget,
                                        budget, unmodelled, cost, stop))
    entries: list[tuple[str, Any]] = []
    kids: list[tuple[Position, list, list, Any]] = []
    menus = iter(listed)
    for branch in branches:
        if branch.position.ended:
            entries.append(("leaf", float(np.asarray(leaf([branch.position]))[0])))
            continue
        crow, ccol = next(menus)
        entries.append(("kid", len(kids)))
        kids.append((branch.position, list(crow), list(ccol), getattr(branch, "digest", None)))
    return ("split", np.asarray(weights, dtype=np.float64), entries, kids)


@dataclass
class _Kid:
    """A child as `_deep_children` reads a branch: its position (and its digest, IKA-378)."""

    position: Position
    digest: Any = None


def _deep_child(  # noqa: PLR0913 - one child and its stage
    reg: Any,  # noqa: ANN401
    child: tuple[Position, list, list, Any],
    leaf: Any,  # noqa: ANN401
    stage: Stage,
    root_budget: Budget,
    budget: Budget,
    unmodelled: set[str],
    cost: LadderCost,
    stop: Any,  # noqa: ANN401
) -> float | None:
    """IKA-375 (`SPLIT`): one child of a deep cell (`_deep_split`), read by ``stage.sub`` as
    `_deep_children` reads it among its siblings; None where that road gives the cell None."""
    position, crow, ccol, digest = child
    return _deep_children(reg, [_Kid(position, digest)], np.ones(1, dtype=np.float64), [(crow, ccol)],
                          leaf, stage, root_budget, budget, unmodelled, cost, stop)


def _deep_value(weights: np.ndarray, entries: list, kids: list) -> float | None:
    """A split cell's value (`_deep_split`): its branches' values weighted, as `_deep_children`
    folds them; None when a child's is."""
    values = []
    for kind, got in entries:
        if kind == "leaf":
            values.append(got)
            continue
        if kids[got] is None:
            return None
        values.append(float(kids[got]))
    return float(np.asarray(values) @ weights)


def _deep_open(  # noqa: PLR0913, PLR0912, C901 - one cell and its stage
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
) -> tuple:
    """IKA-380 (`PASSES`): a deep cell's first step on a worker -- its turn, kept branches and
    children's menus (`_deep_expand`), then each child made ready for the reader to read pass
    by pass (`_Pool.deep_passes`). ``("value", v)`` where the cell is read here to the end, as
    its own road reads it (a child with an empty menu, a matrix refused or with no solution,
    no forward pass to share); else ``("open", road, weights, entries)``, per kept branch
    ``("leaf", value)`` (a battle over), ``("taken", value)`` (a child this read has read,
    `SHARE`; its work is this task's) or ``("kid", position, row menu, column menu, key,
    matrix, x, y, notes, work)``: a child to read, its own matrix filled and solved, ``work``
    what that counted (not this task's: the reader charges a child to every cell that reads
    it, as the table charges a child taken).

    ``road`` is ``"at-once"`` for a child read by a depth-2 stage -- as `_children_at_once`
    reads it: the matrices in one crossing, the child taken from the read's table where it
    is kept -- and ``"alone"`` for a deeper one, as `_deep_children` reads it one at a time
    (each matrix its own `search.batched_payoff`)."""
    got = _deep_expand(reg, pos, ours, theirs, stage, budget, unmodelled)
    if got is None:
        return ("value", None)
    branches, weights, listed = got
    work = search.WORK
    sub = stage.sub

    def whole() -> tuple:
        return ("value", _deep_children(reg, branches, weights, listed, leaf, stage, root_budget,
                                        budget, unmodelled, cost, stop))

    if (work is None or not listed or any(not r or not c for r, c in listed)
            or (sub.sub is None and not BATCH_CHILDREN)):
        return whole()
    mark = dict(work)
    entries: list[tuple] = []
    if sub.sub is None:
        road = "at-once"
        everyone = _kids_open(branches, listed, budget, sub, unmodelled, work)
        kids = [kid for kid in everyone if not kid.taken]
        if stop is not None and stop.is_set():
            raise Stopped
        if kids and _kids_fill(reg, kids, leaf, budget, unmodelled) is _ALONE:
            work.clear()
            work.update(mark)
            return whole()
        live = iter(everyone)
        for branch in branches:
            if branch.position.ended:
                entries.append(("leaf", float(np.asarray(leaf([branch.position]))[0])))
                continue
            kid = next(live)
            if kid.taken:
                entries.append(("taken", None if kid.answer is None else kid.answer[2]))
            else:
                entries.append(("kid", kid.position, kid.row, kid.col, kid.key, kid.prices,
                                kid.x, kid.y, kid.notes, dict(kid.work, reads=0)))
    else:
        road = "alone"
        menus = iter(listed)
        for branch in branches:
            child = branch.position
            if child.ended:
                entries.append(("leaf", float(np.asarray(leaf([child]))[0])))
                continue
            crow, ccol = next(menus)
            if stop is not None and stop.is_set():
                raise Stopped
            try:
                m, notes = search.batched_payoff(reg, child, crow, ccol, leaf, budget=budget)
                m = np.asarray(m, dtype=np.float64)
                eq = solve(m)
            except (EquilibriumError, port.PortRefused):
                work.clear()
                work.update(mark)
                return whole()
            unmodelled.update(notes)
            digest = getattr(branch, "digest", None)
            key = (None if digest is None or search.SHARE is None
                   else subshare.kid_key(digest, crow, ccol, budget.enumerate_knockouts,
                                         sub.label))
            entries.append(("kid", child, list(crow), list(ccol), key, m,
                            np.asarray(eq.row_strategy), np.asarray(eq.col_strategy), set(notes),
                            {"turns": 0, "subgames": 1, "cells": m.size, "qs": 0, "reads": 0}))
    return ("open", road, np.asarray(weights, dtype=np.float64), entries)


def _pass_cells(reg: Any, cells: list, leaf: Any, stage: Stage,  # noqa: ANN401
                budget: Budget) -> list | None:
    """IKA-380 (`PASSES`): a chunk of one pass of a child's depth-2 read (`_Pool.deep_passes`),
    each cell's value as `_children_at_once` reads it (the chunk its Q's group); None where
    the port refused one (the cells that read the child are then read whole, as that road
    falls back)."""
    try:
        found = search._refine_cells(reg, cells, leaf, budget=budget, sub_limit=stage.child,
                                     sub_branches=stage.branches,
                                     child_q=stage.child if stage.q else None,
                                     q_groups=[len(cells)], stack=STACK)
    except port.PortRefused:
        return None
    return [value for value, _notes, _solved in found]


def _open_fold(road: str, weights: np.ndarray, entries: list, values: list,
               charges: list) -> tuple[float | None, list[dict[str, int]]]:
    """IKA-380: an opened cell's value (`_deep_open`) from its children's, and the counted
    work its children add, as its own road counts it: `_children_at_once` counts every child
    it read and a read for each child up to the first without a value; `_deep_children`
    stops at that child (it and those before it counted)."""
    works: list[dict[str, int]] = []
    if road == "at-once":
        works += [c for c in charges if c is not None]
    reads = 0
    out: list[float] = []
    value: float | None = 0.0
    for n, entry in enumerate(entries):
        if entry[0] == "leaf":
            out.append(entry[1])
            continue
        reads += 1
        if road == "alone":
            works.append(charges[n])
        if values[n] is None:
            value = None
            break
        out.append(float(values[n]))
    works.append({"reads": reads})
    if value is None:
        return None, works
    return float(np.asarray(out) @ weights), works


@dataclass
class _Open:
    """IKA-380: a deep cell the reader reads (`_Pool.deep_passes`): what it is a cell of (a
    stage's ask, or a child read by a deep stage), its tasks' counted work and notes, its
    `_deep_open` answer and its children's values and charges."""

    stage: Stage
    budget: Budget
    payload: tuple
    #: ``("top", index)`` or ``("kid", drive, (i, j))``.
    owner: tuple
    works: list = field(default_factory=list)
    notes: set = field(default_factory=set)
    road: str | None = None
    weights: np.ndarray | None = None
    entries: list | None = None
    values: list = field(default_factory=list)
    charges: list = field(default_factory=list)
    left: int = 0
    #: Its value where it was read to the end in one task (``value``, ``whole``).
    value: float | None = None
    done: bool = False
    whole: bool = False
    #: The stage's cell it is part of (None: it is one), and (a stage's cell) the workers'
    #: milliseconds of its tasks.
    top: Any = None
    ms: float = 0.0


@dataclass
class _Drive:
    """IKA-380: a child the reader reads by ``stage`` pass by pass (`_Pool.deep_passes`), and
    the cells waiting on it (each an `_Open` and its entry)."""

    kid: _Child
    stage: Stage
    budget: Budget
    road: str
    attempt: int = 0
    out: int = 0
    waiters: list = field(default_factory=list)
    done: bool = False
    #: A pass the port refused: its cells are read whole.
    refused: bool = False
    value: float | None = None
    #: The stage's cell whose task began it (its milliseconds go there).
    top: Any = None


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
    #: IKA-378 (`SHARE`): its key in the read's table, whether it was taken from there, and
    #: the notes and counted work (less the read) of its read here, to keep.
    key: tuple | None = None
    taken: bool = False
    notes: set = field(default_factory=set)
    work: dict = field(default_factory=dict)


# IKA-380: `_children_at_once`'s steps, one child at a time, so the reader can take a child
# through them pass by pass on the worker processes (`_Pool.deep_passes`) with the same
# arithmetic.

def _kids_open(  # noqa: PLR0913 - a cell's children and their stage
    branches: list,
    menus: list,
    budget: Budget,
    sub: Stage,
    unmodelled: set[str],
    work: dict[str, int],
) -> list[_Child] | object:
    """A `_Child` per live branch, in order; a child this read has read already (`SHARE`,
    IKA-378) taken, with the work its read counted. `_ALONE` where a side has no action."""
    everyone: list[_Child] = []
    listed = iter(menus)
    table = search.SHARE
    for branch in branches:
        if branch.position.ended:
            continue
        crow, ccol = next(listed)
        if not crow or not ccol:
            return _ALONE
        kid = _Child(branch.position, list(crow), list(ccol), None, None, None)
        digest = getattr(branch, "digest", None)
        if table is not None and digest is not None:
            # IKA-378: a child this read has read already -- the same menus, the same fork
            # for its matrix, the same stage for its read -- is taken, with the work its read
            # counted.
            kid.key = subshare.kid_key(digest, kid.row, kid.col, budget.enumerate_knockouts,
                                       sub.label)
            got = table.get(kid.key, kid=True)
            if got is not None:
                value, cells, notes, packed = got
                kid.taken, kid.active = True, False
                kid.answer = None if value is None else (None, None, value)
                unmodelled.update(notes)
                for name, n in subshare.unpack_work(packed).items():
                    work[name] += n
                work["cells"] += cells
        everyone.append(kid)
    return everyone


def _kids_fill(reg: Any, kids: list[_Child], leaf: Any, budget: Budget,  # noqa: ANN401
               unmodelled: set[str]) -> object | None:
    """Each child's own matrix -- in one crossing, scored in one call -- and its solution;
    each child's notes and work (its matrix: a sub-game) kept on it, counted by the caller.
    `_ALONE` where a node is refused, no forward pass can be shared or a matrix has none."""
    pending = port.pending_payoffs(reg, [(k.position, k.row, k.col) for k in kids], leaf,
                                   budget=budget)
    if pending is None or any(isinstance(p, port.PortRefused) for p in pending):
        return _ALONE
    scores = (port.score_stacked if STACK else port.score_segments)(
        leaf, [p.encoded for p in pending])
    for p, got in zip(pending, scores, strict=True):
        p.scored(got)
    mats = [np.asarray(p.finish(), dtype=np.float64) for p in pending]
    # IKA-387: with `portlp` on, every child's matrix solved in the port in one crossing.
    solved = portlp.solve_many(reg, mats) if portlp.ON[0] else None
    for n, (kid, p) in enumerate(zip(kids, pending, strict=True)):
        m = mats[n]
        try:
            if solved is None:
                eq = solve(m)
            elif isinstance(solved[n], Exception):
                raise solved[n]
            else:
                eq = solved[n]
        except EquilibriumError:
            return _ALONE
        unmodelled.update(p.unmodelled)
        kid.prices, kid.x, kid.y = m, np.asarray(eq.row_strategy), np.asarray(eq.col_strategy)
        kid.notes = set(p.unmodelled)
        kid.work = {"turns": 0, "subgames": 1, "cells": m.size, "qs": 0}
    return None


def _one() -> np.ndarray:
    """The weight of a child's one completion, as `read` makes it (K = 1)."""
    w = np.asarray([1.0], dtype=np.float64)
    return w / w.sum()


def _kid_begin(kid: _Child, sub: Stage) -> None:
    """A child's rectangle and prices for its stage ``sub``, as `read` starts a stage."""
    w = _one()
    prices = [np.array(kid.prices, dtype=np.float64, copy=True)]
    x = np.asarray(kid.x, dtype=np.float64)
    ys = [np.asarray(kid.y, dtype=np.float64)]
    kid.rows = _order(x, sum(w[k] * (prices[k] @ ys[k]) for k in range(1)), sub.rect,
                      larger=True)
    kid.cols = _order(ys[0], x @ prices[0], sub.rect, larger=False)
    kid.trial = prices[0].copy()


def _kid_asked(kid: _Child) -> list[tuple[int, int]]:
    """The cells of a child's pass not read yet, in `read`'s order."""
    asked: list[tuple[int, int]] = []
    for i in kid.rows:
        for j in kid.cols:
            if (i, j) not in kid.memo and (i, j) not in asked:
                asked.append((i, j))
    return asked


def _kid_solve(kid: _Child, attempt: int, passes: int) -> None:
    """A child's pass read (its cells in ``kid.memo``): the rectangle solved (``kid.answer``)
    and the oracle's row and column added, as `read` does after a pass; ``kid.active`` off
    when it is done (the last pass, nothing added, or no solution)."""
    game = _kid_game(kid)
    try:
        restricted = solve_bayesian([game], _one())
    except EquilibriumError:
        kid.active = False
        return
    _kid_after(kid, restricted, attempt, passes)


def _kids_solve(reg: Any, jobs: list[tuple[_Child, int, int]]) -> None:  # noqa: ANN401
    """`_kid_solve(kid, attempt, passes)` of each job, in order -- with `portlp` on, their
    rectangles solved in the port in one crossing (IKA-387: the same answers to the bit)."""
    if not portlp.ON[0] or not jobs:
        for kid, attempt, passes in jobs:
            _kid_solve(kid, attempt, passes)
        return
    games = [_kid_game(kid) for kid, _attempt, _passes in jobs]
    solved = portlp.solve_bayesian_many(reg, [([g], _one()) for g in games])
    for (kid, attempt, passes), got in zip(jobs, solved, strict=True):
        if isinstance(got, EquilibriumError):
            kid.active = False
            continue
        if isinstance(got, Exception):
            raise got
        _kid_after(kid, got, attempt, passes)


def _kid_game(kid: _Child) -> np.ndarray:
    """A child's pass's values written into its prices, and the rectangle `_kid_solve` solves."""
    trial = [kid.trial]
    for i in kid.rows:
        for j in kid.cols:
            v = kid.memo[(i, j)]
            if v is not None:
                trial[0][i, j] = v
    return trial[0][np.ix_(kid.rows, kid.cols)]


def _kid_after(kid: _Child, restricted: Any, attempt: int, passes: int) -> None:  # noqa: ANN401
    """`_kid_solve` once its rectangle is solved: the answer, and the oracle's row and column."""
    w = _one()
    trial = [kid.trial]
    sx = np.zeros(len(trial[0]), dtype=np.float64)
    sx[kid.rows] = restricted.row_strategy
    y = np.zeros(trial[0].shape[1], dtype=np.float64)
    y[kid.cols] = restricted.col_strategies[0]
    sys_ = [y]
    col_ev = [sx @ trial[k] for k in range(1)]
    row_ev = sum(w[k] * (trial[k] @ sys_[k]) for k in range(1))
    guarantee = float(sum(w[k] * float(col_ev[k].min()) for k in range(1)))
    kid.answer = (sx, sys_, guarantee)
    if attempt == passes:
        kid.active = False
        return
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


def _kid_keep(table: Any, kid: _Child) -> None:  # noqa: ANN401
    """A child read, kept in the read's table (`SHARE`) with the work its read counted."""
    packed = subshare.pack_work(kid.work)
    if kid.key is not None and packed:
        table.put(kid.key, None if kid.answer is None else kid.answer[2],
                  kid.work["cells"], kid.notes, packed, kid=True)


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
    table = search.SHARE
    everyone = _kids_open(branches, menus, budget, sub, unmodelled, work)
    if everyone is _ALONE:
        return _ALONE
    kids = [kid for kid in everyone if not kid.taken]
    if stop is not None and stop.is_set():
        raise Stopped
    if kids:
        if _kids_fill(reg, kids, leaf, budget, unmodelled) is _ALONE:
            return _ALONE
        for kid in kids:
            work["subgames"] += 1
            work["cells"] += kid.work["cells"]
    sub_budget = replace(root_budget, enumerate_knockouts=True) if sub.knockouts else root_budget
    for kid in kids:
        _kid_begin(kid, sub)
    for attempt in range(sub.passes + 1):
        cells: list = []
        groups: list[int] = []
        group_kids: list[_Child] = []
        owners: list = []
        for kid in kids:
            if not kid.active:
                continue
            asked = _kid_asked(kid)
            for at in range(0, len(asked), CHUNK):
                chunk = asked[at:at + CHUNK]
                groups.append(len(chunk))
                group_kids.append(kid)
                cells += [(kid.position, kid.row[i], kid.col[j]) for i, j in chunk]
                owners += [(kid, cell) for cell in chunk]
        if stop is not None and stop.is_set():
            raise Stopped
        if cells:
            # IKA-378: the pass's work cell by cell, so each child's is known (and kept).
            outer_cells, search.WORK_CELLS = search.WORK_CELLS, []
            try:
                found = search._refine_cells(reg, cells, leaf, budget=sub_budget,
                                             sub_limit=sub.child, sub_branches=sub.branches,
                                             child_q=sub.child if sub.q else None,
                                             q_groups=groups, stack=STACK)
            except port.PortRefused:
                return _ALONE
            finally:
                by_cell, search.WORK_CELLS = search.WORK_CELLS, outer_cells
            if outer_cells is not None:
                outer_cells.extend(by_cell)
            for (kid, cell), (value, _notes, _solved), done in zip(owners, found, by_cell,
                                                                  strict=True):
                kid.memo[cell] = value
                for name in ("turns", "subgames", "cells"):
                    kid.work[name] += done[name]
            if sub.q:
                # The Q is asked once a group (`search._refine_cells`' ``qs``).
                for kid, size in zip(group_kids, groups, strict=True):
                    kid.work["qs"] += 1 if size else 0
        _kids_solve(reg, [(kid, attempt, sub.passes) for kid in kids if kid.active])
        if not any(kid.active for kid in kids):
            break
    if table is not None:
        for kid in kids:
            _kid_keep(table, kid)
    values = []
    live = iter(everyone)
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

#: IKA-374: a directory the reader writes each pooled read's timeline to (one JSON line a
#: read, ``<pid>.jsonl``): every chunk's worker, cells, send / back times and its worker's
#: own milliseconds, each `cells` call's stage and bounds, and each stage's rectangle. For
#: measuring where the workers wait; None (the default): nothing is kept.
TRACE_DIR = os.environ.get("POKEURAOU_LADDER_TRACE") or None

#: IKA-375: a worker names each position to its port by number for the length of a chunk
#: (`rustnode.hold_positions`, IKA-302's road, which generation takes): a position crosses
#: once a chunk, and a turn's branches go back to the port as the numbers it gave them
#: (`refs`) rather than as the JSON it wrote a moment before -- to the damage scores, the
#: Q's features and the sub-games' fills. The same requests, the same answers. Before, the
#: workers' ports read 4.6 MB of requests a node second (open, 16 threads) and built 540
#: positions from JSON; after, 0.36 MB and 23. ``POKEURAOU_LADDER_HOLD=0`` sends every
#: position whole.
HOLD = os.environ.get("POKEURAOU_LADDER_HOLD", "1") != "0"

#: IKA-375: on the wall clock, a deep stage's cell (depth 3 and up) is read in two steps on
#: the workers -- its turn and its children's menus on one (`_deep_split`), then each child's
#: read on whichever worker is free (`_deep_child`) -- and its value folded here
#: (`_deep_value`). A deep stage waited on its slowest cells: at 24 s (open, 16 threads) the
#: longest cell of a stage was most of the stage (one 10.5 s where the mean was 0.46 s) and
#: the workers were idle 38% of the deep stages. The same arithmetic; the children's leaves
#: go to the server in other batches, so the last places move as they do between any two
#: wall-clock reads. The count clock reads a cell on one worker, as before.
#: ``POKEURAOU_LADDER_SPLIT=0`` turns it off.
SPLIT = os.environ.get("POKEURAOU_LADDER_SPLIT", "1") != "0"

#: IKA-380: with `SPLIT`, a deep cell's children are read pass by pass on the workers
#: (`_Pool.deep_passes`): a child's depth-2 pass in chunks, its rectangle solved by the
#: reader, a deeper child's cells opened in turn; a child two cells of the stage reach is
#: read once. At 24-45 s (open, 16 threads, L5, master) the workers were idle 21-29% of a
#: read, all but 1-2 points in the deep stages' tails, where a stage waited on its slowest
#: child read whole on one worker (the longest task of a deep stage: median 0.5-1.9 s, up to
#: 4.8 s). ``POKEURAOU_LADDER_PASSES=0``: IKA-375's split (a child read whole on a worker).
PASSES = os.environ.get("POKEURAOU_LADDER_PASSES", "1") != "0"

#: IKA-380: with `PASSES`, a pass's chunk sent while fewer tasks wait than workers are idle
#: (a deep stage's tail) is cut into parts, one a worker. The chunks of `CHUNK` cells were
#: 70% of what ran in the deep stages' tails (open, 24 s, 16 threads). A part is the Q's
#: group of its cells (the Q's answer for a row moves with its batch, in the last places),
#: so the cut may move a child's menu where two actions' Q values are that close.
#: ``POKEURAOU_LADDER_PASS_SPLIT=0``: whole chunks.
PASS_SPLIT = os.environ.get("POKEURAOU_LADDER_PASS_SPLIT", "1") != "0"

#: IKA-378: a read fills each child sub-game once (`subshare`, `search.SHARE`) -- one table
#: for the read, in shared memory for its worker processes. In a 1-process read at 16 s
#: (L5), 36% (open) and 43% (hidden) of the sub-games were a child with the same menus filled
#: before in the read; on every core 55-60% (IKA-375), most on another worker in the same
#: stage. Charged to the count clock as if filled, so the stages are the same; a value taken
#: is the value where it was filled, the same but for the last places the leaf's and the Q's
#: batches move. ``POKEURAOU_LADDER_SHARE=0`` fills every sub-game where it is met.
SHARE = os.environ.get("POKEURAOU_LADDER_SHARE", "1") != "0"


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


def _pool_main(conn: Any, format_id: str, factory: Any, args: tuple[Any, ...],  # noqa: ANN401, PLR0913
               cancel: Any, share: str | None = None) -> None:  # noqa: ANN401 - a multiprocessing Event
    """A worker: its own regulation, leaf (``factory(reg, *args)``, which also installs the
    Q its children's menus need) and a port of one cell thread; `_read_cells` on each chunk
    of the read it was last told of. ``cancel`` is its stop: a depth-3 cell met by it is
    dropped (`Stopped`). ``share`` names the reader's table of sub-games (`SHARE`)."""
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
    table = None
    if share is not None:
        try:
            table = subshare.Table(share)
        except (OSError, ValueError) as error:
            conn.send(("failed", f"the sub-game table: {type(error).__name__}: {error}"))
            return
        search.SHARE = table
        rustnode.DIGESTS[0] = True
    conn.send(("ready", None))
    context: tuple | None = None
    turns: dict[bool, dict] = {False: {}, True: {}}
    clean: dict[tuple[int, int], bool] = {}
    #: IKA-374: what has come and is not read yet.
    pending: list = []
    # IKA-375: the messages are taken off the pipe as they come, by a thread of their own:
    # on Windows a message past the pipe's buffer (8 KB) holds its sender until it is taken,
    # and a deep cell's child (`SPLIT`) carries its position -- the reader would wait out
    # this worker's whole chunk to hand it one.
    import queue
    import threading

    inbox: queue.Queue = queue.Queue()

    def receive() -> None:
        while True:
            try:
                got = conn.recv()
            except (EOFError, OSError):
                inbox.put(None)
                return
            inbox.put(got)
            if got is None:
                return

    threading.Thread(target=receive, daemon=True).start()
    while True:
        if not pending:
            pending.append(inbox.get())
        while True:
            try:
                pending.append(inbox.get_nowait())
            except queue.Empty:
                break
        message = pending.pop(_next_message(pending))
        if message is None:
            return
        if message[0] == "read":
            # A new read: its menus, completions and budget. The shared turns and the cells'
            # bench reach are keyed by the read's own indices, so they start again.
            context = message[1:7]
            turns = {False: {}, True: {}}
            clean = {}
            if table is not None:
                # IKA-378: the read's own rows of the table (`_Pool.begin`).
                table.begin(message[7])
            # IKA-381: the reader's `portlp` setting, for this read's chunks.
            portlp.set_on(bool(message[8]))
            # IKA-386: and whether their ports ask the inference server (`portserved`).
            portserved.ON[0] = bool(message[9])
            # IKA-389: and whether they build the children's Q menus (`portmenus`).
            portmenus.ON[0] = bool(message[10])
            continue
        kind, task, stage, stage_budget, cells = message
        row, col, items, root_budget, hidden_side, cost = context
        work = _zero()
        notes: set[str] = set()
        if cancel.is_set():
            # Sent before the reader stopped: not read.
            conn.send(("stopped", task, None, [work], notes,
                       (0.0, 0.0, 0, 0.0, 0.0, 0, 0, (0, 0, 0, 0, 0), 0, 0, 0, 0)))
            continue
        shared_began = dict(subshare.COUNTS)
        lps_began = portlp.COUNTS["lps"]
        served_began = portserved.COUNTS["requests"]
        menus_began = portmenus.COUNTS["asked"]
        search.WORK = work
        # IKA-374: the work is said cell by cell: a stage may take a part of a chunk read
        # ahead (`AHEAD`), and counts only the cells it takes.
        by_cell = [] if stage.sub is None else None
        search.WORK_CELLS = by_cell
        clock = time.perf_counter()
        cpu = time.process_time()
        served = _served(leaf)
        ported = tuple(rustnode.PORT_WAITED)
        if HOLD:
            # A chunk is the span its positions are not changed in place over: what the port
            # held for the last one is forgotten ahead of this one's first request.
            rustnode.hold_positions()
        try:
            if kind == "expand":
                # IKA-375 (`SPLIT`): a deep cell's turn and children's menus, its children
                # read by whichever workers are free.
                k, i, j = cells[0]
                got = _deep_split(reg, items[k].position, row[i], col[j], leaf, stage,
                                  root_budget, stage_budget, notes, cost, cancel)
            elif kind == "child":
                got = _deep_child(reg, cells, leaf, stage, root_budget, stage_budget, notes,
                                  cost, cancel)
            elif kind == "open":
                # IKA-380 (`PASSES`): a deep cell's turn, its children's menus and matrices;
                # the children read pass by pass by the reader.
                position, ours, theirs = cells
                got = _deep_open(reg, position, ours, theirs, leaf, stage, root_budget,
                                 stage_budget, notes, cost, cancel)
            elif kind == "whole":
                position, ours, theirs = cells
                got = _deep_cell(reg, position, ours, theirs, leaf, stage, root_budget,
                                 stage_budget, notes, cost, cancel)
            elif kind == "pass":
                got = _pass_cells(reg, cells, leaf, stage, stage_budget)
            else:
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
            search.WORK_CELLS = None
        # One cell of a deeper stage (`_deep_cell` reads nothing through `WORK_CELLS`), or a
        # chunk that stopped, or one the port refused -- read a cell at a time, its refused
        # cells counting nothing: the chunk's work goes on its first cell (the total kept).
        if kind in ("expand", "child", "open", "whole"):
            by_cell = [work]
        elif status != "done" or by_cell is None or len(by_cell) != len(cells):
            by_cell = [work] + [_zero() for _ in cells[1:]]
        work = by_cell
        after = _served(leaf)
        # IKA-370: the chunk's wall, the server's share and trips, this process's CPU, and
        # its port's share and reads -- what is left of the wall is the time it was ready
        # to run and did not (the machine's other processes). IKA-375: and the positions its
        # port held by number at the chunk's end (`HOLD`'s positive control).
        took = (time.perf_counter() - clock, after[0] - served[0], after[1] - served[1],
                time.process_time() - cpu, rustnode.PORT_WAITED[0] - ported[0],
                rustnode.PORT_WAITED[1] - ported[1], len(rustnode._HELD) if HOLD else 0,  # noqa: SLF001
                # IKA-378: sub-games taken from the table or the same call, and kept in it.
                # IKA-380: and those another process was filling when claimed, and of those
                # the ones taken when looked at again.
                (subshare.COUNTS["hits"] - shared_began["hits"],
                 subshare.COUNTS["same"] - shared_began["same"],
                 subshare.COUNTS["kidHits"] - shared_began["kidHits"],
                 subshare.COUNTS["busy"] - shared_began["busy"],
                 subshare.COUNTS["waited"] - shared_began["waited"]),
                subshare.COUNTS["puts"] - shared_began["puts"],
                # IKA-381: the LPs its port solved (`portlp`).
                portlp.COUNTS["lps"] - lps_began,
                # IKA-386: the inference server's requests its port sent (`portserved`).
                portserved.COUNTS["requests"] - served_began,
                # IKA-389: the children whose Q menus its port built (`portmenus`).
                portmenus.COUNTS["asked"] - menus_began)
        conn.send((status, task, got, work, notes, took))


def _next_message(pending: list) -> int:
    """IKA-374: which of the messages come to a worker it reads next -- the first sent, but a
    stage's chunk before a chunk read ahead: a read ahead never holds up a stage's chunk
    that is waiting behind it. A read's context (and the end) keeps its place: nothing sent
    after it is read before it."""
    bound = next((n for n, m in enumerate(pending) if m is None or m[0] == "read"), len(pending))
    return next((n for n in range(bound) if pending[n][0] != "ahead"), 0)


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

    def __init__(self, processes: list, conns: list, cancel: Any,  # noqa: ANN401
                 table: subshare.Table | None = None) -> None:
        self.processes = processes
        self.conns = conns
        self.cancel = cancel
        #: IKA-378: the workers' table of sub-games (`SHARE`), and the reads begun on it.
        self.table = table
        #: IKA-387: the read's regulation, for the reader's port (`portlp`, `deep_passes`).
        self.reg: Any = None
        self.generation = 0
        self.stats: dict[str, float] = {}
        self._got_ms, self._got_cells = 0.0, 0
        #: Tasks out, per worker (a stage's chunks and the speculation), across `cells` calls.
        self._out: dict[int, list[int]] = {i: [] for i in range(len(conns))}
        #: Tasks a read gave up that were still being read when the next read began, and
        #: the workers reading them (`_settle`).
        self._orphan_tasks: set[int] = set()
        self._orphans: set[int] = set()
        #: IKA-374: the read's context message (`begin`), and the workers it is still due to.
        self._context: tuple = ()
        self._due: set[int] = set()
        self._tasks = 0
        #: The cells read ahead (IKA-370 `SPECULATE`, IKA-374 `AHEAD`): chunks out (task ->
        #: their keys), cells back (key -> value, work, notes), and cells out (key -> task).
        self._spec_out: dict[int, list[tuple]] = {}
        self._spec_ready: dict[tuple, tuple] = {}
        self._spec_flying: dict[tuple, int] = {}
        #: IKA-374: this read's timeline (`TRACE_DIR`), or None.
        self._trace: dict[str, Any] | None = None
        self._trace_sent: dict[int, tuple] = {}

    def note(self, what: dict[str, Any]) -> None:
        """A stage's record in the timeline (`TRACE_DIR`)."""
        if self._trace is not None:
            self._trace["stages"].append(
                dict(what, t=round((time.perf_counter() - self._trace["t0"]) * 1000.0, 2)))

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

    def split(self, keys: list, *, shrink: bool = False, of: int | None = None) -> list[list]:
        """A depth-2 pass's cells cut into chunks of `chunk` cells -- sized by the pass's
        ``of`` cells (default: ``keys``), those read ahead (`AHEAD`) included; ``shrink``
        (`TAIL`): a chunk never more than a worker's share of the cells left, so the last
        are single."""
        size = self.chunk(len(keys) if of is None else of)
        if not shrink:
            return [keys[at:at + size] for at in range(0, len(keys), size)]
        chunks, at = [], 0
        while at < len(keys):
            step = max(1, min(size, -(-(len(keys) - at) // len(self.conns))))
            chunks.append(keys[at:at + step])
            at += step
        return chunks

    def begin(self, row: Sequence[Any], col: Sequence[Any], items: Sequence[Any],
              budget: Budget, hidden_side: int, cost: LadderCost) -> None:
        plain = [Item(position=it.position, weight=float(getattr(it, "weight", 1.0)),
                      exact=bool(getattr(it, "exact", True)),
                      slots=tuple(getattr(it, "slots", ()) or ())) for it in items]
        self._settle()
        # IKA-374: a worker still inside a chunk an earlier read gave up gets this read's
        # context when that chunk is back (`cells`), not now: a message over the pipe's
        # buffer (8 KB on Windows) waits until the worker takes it, and an 8 s read began
        # 4.4 s late behind depth-3 and 4 cells read ahead for the read before it.
        # IKA-378: each read its own rows of the sub-game table (a chunk of an earlier read
        # still out keeps to that read's: its worker is sent this after it is back).
        self.generation += 1
        if self.table is not None:
            # IKA-380: the reader keeps the children it reads pass by pass (`deep_passes`).
            self.table.begin(self.generation)
        # IKA-381: and whether the workers solve their LPs in their ports (`portlp`), as here.
        self._context = ("read", list(row), list(col), plain, budget, hidden_side, cost,
                         self.generation, portlp.ON[0], portserved.ON[0], portmenus.ON[0])
        self._due = set(self._orphans)
        for i, conn in enumerate(self.conns):
            if i not in self._due:
                conn.send(self._context)
        POOL_COUNTS["reads"] += 1
        #: This read's: chunks taken, the reader's milliseconds blocked on the workers, in
        #: sends and in receives (unpickling included), and the workers' own reading.
        #: The workers' idle time, by what they waited for: ``supplyMs`` the reader's
        #: sends (chunks left to send), ``tailMs`` the stage's last chunks on the other
        #: workers (all sent, theirs back), ``betweenMs`` the reader between two stages' cells
        #: (its LP, the next rectangle), and ``serverMs`` / ``serverTrips`` the inference
        #: server inside their reading. IKA-370: ``cpuMs`` the workers' own CPU in their
        #: reading, ``portMs`` / ``portReads`` their waits on their ports; the speculation's
        #: cells sent, their workers' milliseconds, the cells a stage took from it and the
        #: cells it read for nothing (``specCells``, ``specMs``, ``specUsed``, ``specWasted``).
        #: IKA-374 (`AHEAD`): ``specChunks`` the chunks read ahead, ``specPartial`` the
        #: chunks a stage took a part of (the rest kept). IKA-375 (`SPLIT`): ``splitCells``
        #: the deep cells read child by child, ``childTasks`` the children sent; (`HOLD`)
        #: ``heldPositions`` the positions the workers' ports held by number, summed over
        #: the chunks' ends. IKA-378 (`SHARE`): ``sharedSubgames`` the sub-games the workers
        #: took from the table or an earlier one of the same call (``sameSubgames`` the latter),
        #: ``keptSubgames`` put in it; ``sharedKids`` a deep cell's children taken from it.
        #: IKA-380: ``busySubgames`` the sub-games another worker was filling when claimed
        #: (`subshare.Table.claim`), ``waitedSubgames`` of those, taken when looked at again;
        #: (`PASSES`) ``openCells`` the deep cells opened, ``kidReads`` the children read here
        #: pass by pass, ``kidsJoined`` a cell's child joined to one already read or being
        #: read, ``passChunks`` the chunks of their depth-2 passes, ``wholeCells`` the cells
        #: read whole after a pass was refused, ``passSplits`` chunks cut in the tail
        #: (`PASS_SPLIT`). IKA-387 (`portlp` on): ``kidSolves`` the children's pass
        #: rectangles solved here in the port, in ``kidSolveTrips`` crossings.
        self.stats = {"chunks": 0, "waitMs": 0.0, "sendMs": 0.0, "recvMs": 0.0,
                      "workerMs": 0.0, "serverMs": 0.0, "serverTrips": 0, "supplyMs": 0.0,
                      "tailMs": 0.0, "betweenMs": 0.0, "cpuMs": 0.0, "portMs": 0.0,
                      "portReads": 0, "specCells": 0, "specMs": 0.0, "specUsed": 0,
                      "specWasted": 0, "specChunks": 0, "specPartial": 0, "splitCells": 0,
                      "childTasks": 0, "heldPositions": 0, "sharedSubgames": 0,
                      "sameSubgames": 0, "keptSubgames": 0, "sharedKids": 0,
                      "busySubgames": 0, "waitedSubgames": 0, "openCells": 0, "kidReads": 0,
                      "kidsJoined": 0, "passChunks": 0, "wholeCells": 0, "passSplits": 0,
                      "portLps": 0, "kidSolves": 0, "kidSolveTrips": 0, "portServed": 0,
                      "portMenus": 0}
        self._last_end = time.perf_counter()
        self._trace = ({"t0": self._last_end, "workers": len(self.conns), "calls": [],
                        "chunks": [], "stages": []} if TRACE_DIR else None)
        #: The timeline's chunks out: task -> (worker, sent, cells, call, speculation).
        self._trace_sent: dict[int, tuple] = {}

    def _t(self, clock: float) -> float:
        return round((clock - self._trace["t0"]) * 1000.0, 2)

    def end(self) -> None:
        """The read is over: the workers idle since the last stage's cells were back, and
        the speculation not taken thrown away."""
        self.stats["betweenMs"] += (time.perf_counter() - self._last_end) * 1000.0 * len(
            self.conns)
        self.stats["specWasted"] += len(self._spec_ready) + sum(
            len(keys) for keys in self._spec_out.values())
        self._drain()
        self.stats["lingering"] = sum(len(t) for t in self._out.values())
        if self._trace is not None:
            import json

            trace, self._trace = self._trace, None
            trace["end"] = self._t_of(trace, time.perf_counter())
            trace.pop("t0")
            os.makedirs(TRACE_DIR, exist_ok=True)
            with open(os.path.join(TRACE_DIR, f"{os.getpid()}.jsonl"), "ab") as fh:
                fh.write((json.dumps(trace) + "\n").encode("utf-8"))

    @staticmethod
    def _t_of(trace: dict[str, Any], clock: float) -> float:
        return round((clock - trace["t0"]) * 1000.0, 2)

    def claim(self, keys: Sequence[tuple]) -> tuple[dict, dict]:
        """The cells read ahead for ``keys``: back (key -> value, work, notes) and still out
        (task -> the keys of ``keys`` it reads: `cells`' ``adopt``). Both are the stage's from
        here."""
        ready = {}
        origins: dict[int, list] = {}
        for k in keys:
            if k in self._spec_ready:
                value, done_work, notes, (task, size) = self._spec_ready.pop(k)
                ready[k] = (value, done_work, notes)
                origins.setdefault(task, [size, 0])[1] += 1
        by_task: dict[int, list[tuple]] = {}
        for k in keys:
            task = self._spec_flying.pop(k, None)
            if task is not None:
                by_task.setdefault(task, []).append(k)
        for task, wanted in by_task.items():
            origins.setdefault(-task, [len(self._spec_out[task]), 0])[1] += len(wanted)
        self.stats["specUsed"] += len(ready) + sum(len(w) for w in by_task.values())
        # The positive control that a chunk's cells are counted cell by cell.
        self.stats["specPartial"] += sum(1 for size, got in origins.values() if got < size)
        return ready, by_task

    def holds(self, key: tuple) -> bool:
        return key in self._spec_ready or key in self._spec_flying

    def ready(self, keys: Any) -> int:  # noqa: ANN401
        """How many of ``keys`` are read ahead and back."""
        return sum(1 for k in keys if k in self._spec_ready)

    def _took(self, times: tuple) -> tuple:
        """A task's timings (`_pool_main`) added to the read's stats; the first six back."""
        took, server_s, trips, cpu_s, port_s, port_reads, held, shared, kept, lps, asked, menus = times
        self.stats["portLps"] += lps
        self.stats["portServed"] += asked
        self.stats["portMenus"] += menus
        self.stats["heldPositions"] += held
        self.stats["sharedSubgames"] += shared[0] + shared[1]
        self.stats["sameSubgames"] += shared[1]
        self.stats["sharedKids"] += shared[2]
        self.stats["busySubgames"] += shared[3]
        self.stats["waitedSubgames"] += shared[4]
        self.stats["keptSubgames"] += kept
        self.stats["workerMs"] += took * 1000.0
        self.stats["serverMs"] += server_s * 1000.0
        self.stats["serverTrips"] += trips
        self.stats["cpuMs"] += cpu_s * 1000.0
        self.stats["portMs"] += port_s * 1000.0
        self.stats["portReads"] += port_reads
        return took, server_s, trips, cpu_s, port_s, port_reads

    def _ahead_back(self, task: int, status: str, values: Any, cell_work: list,  # noqa: ANN401
                    notes: set[str]) -> None:
        """A chunk read ahead is back: its cells kept for `claim` (a cell by cell's work)."""
        keys = self._spec_out.pop(task, [])
        for n, key in enumerate(keys):
            self._spec_flying.pop(key, None)
            if status == "done":
                self._spec_ready[key] = (values[n], cell_work[n], notes, (task, len(keys)))

    def cells(self, asks: list[list], stage: Stage, stage_budget: Budget,  # noqa: C901, PLR0912, PLR0915
              work: dict[str, int], notes: set[str], *,
              adopt: dict[int, tuple[int, list[tuple]]] | None = None, ahead: Any = None,
              tail: bool = False,
              early: Callable[[int, list], None] | None = None) -> Any:  # noqa: ANN401 - a generator
        """`_serial`'s answers from the workers: each chunk's values in the order asked,
        its counted work and notes added as it is taken; `_TICK` while waiting. Closed
        early (the reader stopped), the chunks still out are cancelled and thrown away.

        ``adopt`` names chunks (by index) that are already out, read ahead: index -> (task,
        the chunk's keys among the task's), taken here when back. ``ahead()``
        (IKA-370, IKA-374) gives the next chunk to read ahead -- a list of ``(stage,
        stage_budget, key, cell)`` of one stage, or None -- sent to a worker with nothing
        out once this call's chunks are all sent, one at a time, and kept for `claim`.
        ``tail`` (IKA-374): the last chunks go only to a worker with nothing out, never
        behind a chunk in hand.

        IKA-369: ``early(index, values)``: each of this stage's own chunks read to the end,
        as it arrives (before its turn, and also one a stop then throws away)."""
        from multiprocessing.connection import wait

        out = self._out
        which = {id(conn): i for i, conn in enumerate(self.conns)}
        results: dict[int, tuple] = {}
        mine: dict[int, int] = {}
        adopted: dict[int, tuple[int, list[tuple]]] = {}
        for index, (task, wanted) in (adopt or {}).items():
            adopted[task] = (index, wanted)
        to_send = [n for n in range(len(asks)) if n not in (adopt or {})]
        at = taken = 0
        #: The workers' milliseconds and cells of the chunks back so far (`eta_ms`).
        self._got_ms, self._got_cells = 0.0, 0
        now = time.perf_counter()
        self.stats["betweenMs"] += (now - self._last_end) * 1000.0 * len(self.conns)
        idle_since = [now] * len(self.conns)
        trace = self._trace
        sent = self._trace_sent
        if trace is not None:
            call = {"stage": stage.label, "t": self._t(now), "chunks": len(asks),
                    "cells": sum(len(a) for a in asks)}
            trace["calls"].append(call)
        try:
            while taken < len(asks):
                while True:
                    # A worker still inside a chunk a read gave up is full (`_settle`).
                    i = min(out, key=lambda k: len(out[k]) + (POOL_DEPTH if k in self._orphans
                                                               else 0))
                    if len(out[i]) >= POOL_DEPTH:
                        break
                    clock = time.perf_counter()
                    if at < len(to_send):
                        if tail and out[i] and len(to_send) - at <= len(self.conns):
                            break
                        index = to_send[at]
                        at += 1
                        self._tasks += 1
                        task = self._tasks
                        mine[task] = index
                        if not out[i]:
                            self.stats["supplyMs"] += (clock - idle_since[i]) * 1000.0
                        self.conns[i].send(("cells", task, stage, stage_budget, asks[index]))
                        size = len(asks[index])
                    elif ahead is not None and not out[i]:
                        got = ahead()
                        if got is None:
                            ahead = None
                            break
                        self._tasks += 1
                        task = self._tasks
                        keys = [key for _s, _b, key, _c in got]
                        self._spec_out[task] = keys
                        for key in keys:
                            self._spec_flying[key] = task
                        self.stats["specCells"] += len(keys)
                        self.stats["specChunks"] += 1
                        self.conns[i].send(("ahead", task, got[0][0], got[0][1],
                                            [cell for _s, _b, _k, cell in got]))
                        size = len(keys)
                    else:
                        break
                    self.stats["sendMs"] += (time.perf_counter() - clock) * 1000.0
                    out[i].append(task)
                    if trace is not None:
                        sent[task] = (i, self._t(clock), size, len(trace["calls"]) - 1,
                                      task not in mine)
                if taken in results:
                    status, values, done_work, done_notes = results.pop(taken)
                    self.stats["chunks"] += 1
                    for cell_work in done_work:
                        for kind, n in cell_work.items():
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
                    took, server_s, trips, cpu_s, port_s, port_reads = self._took(times)
                    if trace is not None and task in sent:
                        w_i, t_sent, n_cells, at_call, spec = sent.pop(task)
                        trace["chunks"].append(
                            [at_call, w_i, n_cells, t_sent, self._t(back), round(took * 1000.0, 2),
                             round(server_s * 1000.0, 2), trips, round(cpu_s * 1000.0, 2),
                             round(port_s * 1000.0, 2), port_reads, int(spec), status])
                    i = which[id(conn)]
                    out[i].remove(task)
                    self._orphan_tasks.discard(task)
                    if not any(t in self._orphan_tasks for t in out[i]):
                        self._orphans.discard(i)
                        if i in self._due:
                            self._due.discard(i)
                            self.conns[i].send(self._context)
                    if status == "error":
                        raise RuntimeError(f"a ladder worker failed on a chunk: {values}")
                    if task in mine:
                        index = mine.pop(task)
                        if early is not None and status == "done":
                            early(index, values)
                        self._got_ms += took * 1000.0
                        self._got_cells += len(asks[index])
                        results[index] = (status, values, done_work, done_notes)
                    elif task in adopted:
                        # A chunk read ahead that this stage asked for: its cells taken as
                        # `claim` takes them, the rest kept for a later stage.
                        index, wanted = adopted.pop(task)
                        self.stats["specMs"] += took * 1000.0
                        self._ahead_back(task, status, values, done_work, done_notes)
                        if status == "done":
                            got = [self._spec_ready.pop(key) for key in wanted]
                            results[index] = (status, [g[0] for g in got], [g[1] for g in got],
                                              done_notes)
                        else:
                            results[index] = (status, None, [], done_notes)
                    elif task in self._spec_out:
                        self.stats["specMs"] += took * 1000.0
                        self._ahead_back(task, status, values, done_work, done_notes)
                    if not any(t in mine or t in adopted for t in out[i]):
                        idle_since[i] = back
        finally:
            if taken < len(asks):
                self._drain()
            self._last_end = time.perf_counter()
            if trace is not None:
                call["end"] = self._t(self._last_end)
                call["taken"] = taken
            if at == len(to_send):
                for since in idle_since:
                    self.stats["tailMs"] += max(0.0, self._last_end - since) * 1000.0

    def deep_cells(self, asks: list[list], stage: Stage, stage_budget: Budget,  # noqa: C901, PLR0912, PLR0915
                   work: dict[str, int], notes: set[str], *,
                   early: Callable[[int, list], None] | None = None) -> Any:  # noqa: ANN401 - a generator
        """IKA-375 (`SPLIT`): `cells` for a deep stage (a cell a chunk), each cell read in two
        steps: ``expand`` on one worker (`_deep_split`: its turn and its children's menus,
        or its value when it is read there to the end), then a ``child`` task per child on
        whichever workers are free (`_deep_child`), a child sent before a cell not yet
        expanded. The cell's value is its branches' values weighted (`_deep_value`), its
        counted work the sum of its tasks'. Yields as `cells` does: each cell's value in the
        order asked, its work and notes added as it is taken; `_TICK` while waiting."""
        from collections import deque
        from multiprocessing.connection import wait

        out = self._out
        which = {id(conn): i for i, conn in enumerate(self.conns)}
        expands = deque(range(len(asks)))
        children: deque = deque()
        #: task -> (index, kid number or None for the expand)
        mine: dict[int, tuple[int, int | None]] = {}
        #: index -> [works, notes, split answer or None, kids' values, kids left, stopped]
        state: dict[int, list] = {n: [[], set(), None, None, 0, False] for n in range(len(asks))}
        results: dict[int, tuple] = {}
        taken = 0
        self._got_ms, self._got_cells = 0.0, 0
        now = time.perf_counter()
        self.stats["betweenMs"] += (now - self._last_end) * 1000.0 * len(self.conns)
        idle_since = [now] * len(self.conns)
        trace = self._trace
        sent = self._trace_sent
        if trace is not None:
            call = {"stage": stage.label, "t": self._t(now), "chunks": len(asks),
                    "cells": sum(len(a) for a in asks)}
            trace["calls"].append(call)

        def finish(index: int) -> None:
            cell = state.pop(index)
            works, cell_notes, split, kid_values, _left, stopped = cell
            if stopped:
                results[index] = ("stopped", None, works, cell_notes)
                return
            value = split[1] if split[0] == "value" else _deep_value(split[1], split[2],
                                                                     kid_values)
            results[index] = ("done", [value], works, cell_notes)
            self._got_cells += 1
            if early is not None:
                early(index, [value])

        try:
            while taken < len(asks):
                while True:
                    i = min(out, key=lambda k: len(out[k]) + (POOL_DEPTH if k in self._orphans
                                                               else 0))
                    if len(out[i]) >= POOL_DEPTH or not (children or expands):
                        break
                    clock = time.perf_counter()
                    self._tasks += 1
                    task = self._tasks
                    if not out[i]:
                        self.stats["supplyMs"] += (clock - idle_since[i]) * 1000.0
                    if children:
                        index, kid, payload = children.popleft()
                        mine[task] = (index, kid)
                        self.conns[i].send(("child", task, stage, stage_budget, payload))
                        self.stats["childTasks"] += 1
                    else:
                        index = expands.popleft()
                        mine[task] = (index, None)
                        self.conns[i].send(("expand", task, stage, stage_budget, asks[index]))
                    self.stats["sendMs"] += (time.perf_counter() - clock) * 1000.0
                    out[i].append(task)
                    if trace is not None:
                        sent[task] = (i, self._t(clock), 1, len(trace["calls"]) - 1, False)
                if taken in results:
                    status, values, done_work, done_notes = results.pop(taken)
                    self.stats["chunks"] += 1
                    for cell_work in done_work:
                        for kind, n in cell_work.items():
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
                    took, server_s, trips, cpu_s, port_s, port_reads = self._took(times)
                    if trace is not None and task in sent:
                        w_i, t_sent, n_cells, at_call, spec = sent.pop(task)
                        trace["chunks"].append(
                            [at_call, w_i, n_cells, t_sent, self._t(back), round(took * 1000.0, 2),
                             round(server_s * 1000.0, 2), trips, round(cpu_s * 1000.0, 2),
                             round(port_s * 1000.0, 2), port_reads, int(spec), status])
                    i = which[id(conn)]
                    out[i].remove(task)
                    self._orphan_tasks.discard(task)
                    if not any(t in self._orphan_tasks for t in out[i]):
                        self._orphans.discard(i)
                        if i in self._due:
                            self._due.discard(i)
                            self.conns[i].send(self._context)
                    if status == "error":
                        raise RuntimeError(f"a ladder worker failed on a chunk: {values}")
                    if task in mine:
                        index, kid = mine.pop(task)
                        self._got_ms += took * 1000.0
                        cell = state.get(index)
                        if cell is not None:
                            cell[0].extend(done_work)
                            cell[1].update(done_notes)
                            if status == "stopped":
                                cell[5] = True
                            elif kid is None:
                                cell[2] = values
                                if values[0] == "split":
                                    kids = values[3]
                                    cell[3] = [None] * len(kids)
                                    cell[4] = len(kids)
                                    # Ahead of any cell not yet expanded, in their order.
                                    children.extendleft(reversed(
                                        [(index, n, one) for n, one in enumerate(kids)]))
                                    self.stats["splitCells"] += 1
                            else:
                                cell[3][kid] = values
                                cell[4] -= 1
                            if cell[5] or cell[2][0] == "value" or cell[4] == 0:
                                finish(index)
                    if not any(t in mine for t in out[i]):
                        idle_since[i] = back
        finally:
            if taken < len(asks):
                self._drain()
            self._last_end = time.perf_counter()
            if trace is not None:
                call["end"] = self._t(self._last_end)
                call["taken"] = taken
            if not expands and not children:
                for since in idle_since:
                    self.stats["tailMs"] += max(0.0, self._last_end - since) * 1000.0

    def deep_passes(self, asks: list[list], stage: Stage, stage_budget: Budget,  # noqa: C901, PLR0912, PLR0915
                    work: dict[str, int], notes: set[str], *,
                    early: Callable[[int, list], None] | None = None) -> Any:  # noqa: ANN401 - a generator
        """IKA-380 (`PASSES`): `cells` for a deep stage (a cell a chunk), each cell's tree read
        by the workers in small tasks the reader hands out and joins:

        * ``open`` (`_deep_open`): a cell's turn, its children's menus and matrices;
        * a child read by a depth-2 stage: each of its passes in ``pass`` chunks of `CHUNK`
          cells (`_pass_cells`, the Q's groups of `_children_at_once`), the pass's rectangle
          solved here (`_kid_solve`), then the next pass; the child kept in the read's table;
        * a child read by a deeper stage: each cell of its passes opened as above, and so on.

        IKA-375's split read a child whole on one worker: a deep stage waited on its longest
        child (up to 4.8 s at 24-45 s, open, 16 threads), and the workers were idle 21-29% of
        a read, nearly all of it in the deep stages' tails. A child met by two cells of the
        stage is read once (by its key), charged to both as the table charges a child taken.
        The same arithmetic and counted work as the cell read on one worker (`_open_fold`);
        the leaves and the Q go to the server in other batches, so the last places move as
        between any two wall-clock reads. A task a worker stops ends the stage (`_STOPPED`).
        Yields as `cells` does."""
        from collections import deque
        from multiprocessing.connection import wait

        out = self._out
        which = {id(conn): i for i, conn in enumerate(self.conns)}
        row, col, items, root_budget = self._context[1:5]

        def budget_of(st: Stage) -> Budget:
            return replace(root_budget, enumerate_knockouts=True) if st.knockouts else root_budget

        #: What to send: the tasks of children and cells begun (first), then the stage's cells.
        inner: deque = deque()
        tops: deque = deque()
        #: task -> (kind, the `_Open` or (the `_Drive`, its cells))
        mine: dict[int, tuple] = {}
        drives: dict[Any, _Drive] = {}
        results: dict[int, tuple] = {}
        halted = False
        taken = 0
        for index, ask in enumerate(asks):
            (k, i, j), = ask
            op = _Open(stage, stage_budget, (items[k].position, row[i], col[j]), ("top", index))
            tops.append(("open", stage, stage_budget, op.payload, op))
        self._got_ms, self._got_cells = 0.0, 0
        now = time.perf_counter()
        self.stats["betweenMs"] += (now - self._last_end) * 1000.0 * len(self.conns)
        idle_since = [now] * len(self.conns)
        ticked = now
        trace = self._trace
        sent = self._trace_sent
        if trace is not None:
            call = {"stage": stage.label, "t": self._t(now), "chunks": len(asks),
                    "cells": sum(len(a) for a in asks)}
            trace["calls"].append(call)

        def start_pass(drive: _Drive) -> None:
            kid = drive.kid
            asked = _kid_asked(kid)
            drive.out = len(asked)
            if not asked:
                step(drive)
                return
            if drive.stage.sub is None:
                for at in range(0, len(asked), CHUNK):
                    chunk = asked[at:at + CHUNK]
                    inner.append(("pass", drive.stage, drive.budget,
                                  [(kid.position, kid.row[a], kid.col[b]) for a, b in chunk],
                                  (drive, chunk)))
            else:
                for a, b in asked:
                    op = _Open(drive.stage, drive.budget, (kid.position, kid.row[a], kid.col[b]),
                               ("kid", drive, (a, b)), top=drive.top)
                    inner.append(("open", drive.stage, drive.budget, op.payload, op))

        #: IKA-387: with `portlp` on, the children whose pass is back wait here, and are
        #: solved together in one crossing once the answers back are taken (`flush`).
        together = portlp.ON[0]
        waiting: list[_Drive] = []

        def step(drive: _Drive) -> None:
            if together:
                waiting.append(drive)
                return
            _kid_solve(drive.kid, drive.attempt, drive.stage.passes)
            stepped(drive)

        def flush() -> None:
            while waiting:
                now_waiting = list(waiting)
                waiting.clear()
                _kids_solve(self.reg, [(d.kid, d.attempt, d.stage.passes) for d in now_waiting])
                self.stats["kidSolves"] += len(now_waiting)
                self.stats["kidSolveTrips"] += 1
                for drive in now_waiting:
                    stepped(drive)

        def stepped(drive: _Drive) -> None:
            if drive.kid.active:
                drive.attempt += 1
                start_pass(drive)
            else:
                finish_drive(drive)

        def finish_drive(drive: _Drive) -> None:
            kid = drive.kid
            drive.done = True
            drive.value = None if kid.answer is None else kid.answer[2]
            if drive.road == "at-once" and self.table is not None:
                _kid_keep(self.table, kid)
            self.stats["kidReads"] += 1
            waiters, drive.waiters = drive.waiters, []
            for op, n in waiters:
                settle(op, n, drive)

        def settle(op: _Open, n: int, drive: _Drive) -> None:
            if op.done or op.whole:
                return
            op.values[n] = drive.value
            op.charges[n] = dict(drive.kid.work)
            op.notes |= drive.kid.notes
            op.left -= 1
            if op.left == 0:
                finish(op)

        def refused(drive: _Drive) -> None:
            """A pass the port refused: every cell reading the child is read whole."""
            drive.done, drive.refused = True, True
            waiters, drive.waiters = drive.waiters, []
            for op, _n in waiters:
                whole(op)

        def whole(op: _Open) -> None:
            if op.done or op.whole:
                return
            op.whole = True
            op.works = []
            self.stats["wholeCells"] += 1
            inner.appendleft(("whole", op.stage, op.budget, op.payload, op))

        def finish(op: _Open) -> None:
            op.done = True
            if op.road is None or op.whole:
                value, extra = op.value, []
            else:
                value, extra = _open_fold(op.road, op.weights, op.entries, op.values, op.charges)
            works = op.works + extra
            if op.owner[0] == "top":
                index = op.owner[1]
                results[index] = ("done", [value], works, op.notes)
                self._got_ms += op.ms
                self._got_cells += 1
                if early is not None:
                    early(index, [value])
                return
            _kind, drive, cell = op.owner
            drive.kid.memo[cell] = value
            for done in works:
                for name, n in done.items():
                    drive.kid.work[name] = drive.kid.work.get(name, 0) + n
            drive.out -= 1
            if drive.out == 0:
                step(drive)

        def opened(op: _Open, answer: tuple) -> None:
            if answer[0] == "value":
                op.value = answer[1]
                finish(op)
                return
            _open, op.road, op.weights, op.entries = answer
            op.values = [None] * len(op.entries)
            op.charges = [None] * len(op.entries)
            self.stats["openCells"] += 1
            sub = op.stage.sub
            joined: list[tuple[_Drive, int]] = []
            begun: list[_Drive] = []
            for n, entry in enumerate(op.entries):
                if entry[0] in ("leaf", "taken"):
                    op.values[n] = entry[1]
                    continue
                _k, position, crow, ccol, key, m, x, y, kid_notes, kid_work = entry
                drive = drives.get(key) if key is not None else None
                if drive is None:
                    kid = _Child(position, list(crow), list(ccol), m, x, y)
                    kid.key, kid.notes, kid.work = key, set(kid_notes), dict(kid_work)
                    drive = _Drive(kid, sub, budget_of(sub), op.road, top=op.top or op)
                    if key is not None:
                        drives[key] = drive
                    begun.append(drive)
                else:
                    self.stats["kidsJoined"] += 1
                op.left += 1
                joined.append((drive, n))
            if any(drive.refused for drive, _n in joined):
                # A child whose pass the port refused: this cell is read whole (the children
                # it began are still read: another cell may join them).
                whole(op)
            else:
                for drive, n in joined:
                    if drive.done:
                        settle(op, n, drive)
                    else:
                        drive.waiters.append((op, n))
            for drive in begun:
                _kid_begin(drive.kid, drive.stage)
                start_pass(drive)
            if op.left == 0 and not op.done and not op.whole:
                finish(op)

        try:
            while taken < len(asks):
                while True:
                    i = min(out, key=lambda k: len(out[k]) + (POOL_DEPTH if k in self._orphans
                                                               else 0))
                    if len(out[i]) >= POOL_DEPTH or not (inner or tops):
                        break
                    clock = time.perf_counter()
                    self._tasks += 1
                    task = self._tasks
                    if not out[i]:
                        self.stats["supplyMs"] += (clock - idle_since[i]) * 1000.0
                    kind, st, bud, payload, ref = inner.popleft() if inner else tops.popleft()
                    if kind == "pass" and PASS_SPLIT and len(payload) > 1:
                        # The tail: fewer tasks than idle workers -- the chunk in parts.
                        idle = sum(1 for k in out if not out[k] and k not in self._orphans)
                        want = min(len(payload), idle - len(inner) - len(tops))
                        if want > 1:
                            drive, cells = ref
                            size = -(-len(payload) // want)
                            parts = [(payload[a:a + size], cells[a:a + size])
                                     for a in range(0, len(payload), size)]
                            for part, part_cells in reversed(parts[1:]):
                                inner.appendleft(("pass", st, bud, part, (drive, part_cells)))
                            payload, ref = parts[0][0], (drive, parts[0][1])
                            self.stats["passSplits"] += 1
                    mine[task] = (kind, ref)
                    self.conns[i].send((kind, task, st, bud, payload))
                    if kind == "pass":
                        self.stats["passChunks"] += 1
                    self.stats["sendMs"] += (time.perf_counter() - clock) * 1000.0
                    out[i].append(task)
                    if trace is not None:
                        sent[task] = (i, self._t(clock), len(payload) if kind == "pass" else 1,
                                      len(trace["calls"]) - 1, False)
                if halted:
                    yield taken, _STOPPED
                    return
                if time.perf_counter() - ticked >= TICK_SECONDS:
                    # IKA-376: the clock is looked at while the workers keep answering, too --
                    # small tasks back every few milliseconds never left `wait` empty, and a
                    # depth-5 stage's top cell is seconds: a 30 s read ended at 43 s.
                    ticked = time.perf_counter()
                    yield None, _TICK
                if taken in results:
                    status, values, done_work, done_notes = results.pop(taken)
                    self.stats["chunks"] += 1
                    for cell_work in done_work:
                        for kind, n in cell_work.items():
                            work[kind] += n
                    notes.update(done_notes)
                    POOL_COUNTS["chunks"] += 1
                    POOL_COUNTS["cells"] += len(asks[taken])
                    index = taken
                    taken += 1
                    yield index, values
                    continue
                clock = time.perf_counter()
                ready = wait(self.conns, timeout=TICK_SECONDS)
                self.stats["waitMs"] += (time.perf_counter() - clock) * 1000.0
                if not ready:
                    ticked = time.perf_counter()
                    yield None, _TICK
                    continue
                for conn in ready:
                    clock = time.perf_counter()
                    status, task, values, done_work, done_notes, times = conn.recv()
                    back = time.perf_counter()
                    self.stats["recvMs"] += (back - clock) * 1000.0
                    took, server_s, trips, cpu_s, port_s, port_reads = self._took(times)
                    if trace is not None and task in sent:
                        w_i, t_sent, n_cells, at_call, spec = sent.pop(task)
                        trace["chunks"].append(
                            [at_call, w_i, n_cells, t_sent, self._t(back), round(took * 1000.0, 2),
                             round(server_s * 1000.0, 2), trips, round(cpu_s * 1000.0, 2),
                             round(port_s * 1000.0, 2), port_reads, int(spec), status])
                    i = which[id(conn)]
                    out[i].remove(task)
                    self._orphan_tasks.discard(task)
                    if not any(t in self._orphan_tasks for t in out[i]):
                        self._orphans.discard(i)
                        if i in self._due:
                            self._due.discard(i)
                            self.conns[i].send(self._context)
                    if status == "error":
                        raise RuntimeError(f"a ladder worker failed on a chunk: {values}")
                    if task in mine:
                        kind, ref = mine.pop(task)
                        # The workers' milliseconds go to the stage's cell the task is part of,
                        # counted (`eta_ms`) once that cell is done: many cells are part read at
                        # once, and a first cell done with every task's milliseconds so far
                        # predicted a stage at 15 times its length (abandoned after a cell).
                        (ref[0].top if kind == "pass" else ref.top or ref).ms += took * 1000.0
                        if status == "stopped":
                            halted = True
                        elif kind == "pass":
                            drive, chunk = ref
                            if drive.done:
                                pass
                            elif values is None:
                                refused(drive)
                            else:
                                kid = drive.kid
                                for cell, value, done in zip(chunk, values, done_work, strict=True):
                                    kid.memo[cell] = value
                                    for name in ("turns", "subgames", "cells", "qs"):
                                        kid.work[name] += done[name]
                                drive.out -= len(chunk)
                                if drive.out == 0:
                                    step(drive)
                        elif kind == "whole":
                            ref.works = list(done_work)
                            ref.notes.update(done_notes)
                            ref.value = values
                            finish(ref)
                        elif not (ref.done or ref.whole):
                            ref.works.extend(done_work)
                            ref.notes.update(done_notes)
                            opened(ref, values)
                    if not any(t in mine for t in out[i]):
                        idle_since[i] = back
                flush()
        finally:
            if taken < len(asks):
                self._drain()
            self._last_end = time.perf_counter()
            if trace is not None:
                call["end"] = self._t(self._last_end)
                call["taken"] = taken
            if not inner and not tops:
                for since in idle_since:
                    self.stats["tailMs"] += max(0.0, self._last_end - since) * 1000.0

    def _drain(self) -> None:
        """Everything out cancelled and thrown away, the speculation with it -- without
        waiting (IKA-370): a read ends at its budget, not when the chunks still out end (a
        depth-2 chunk runs to its end, a deeper cell to its next look at the stop: an 8 s
        read that waited for them ended at up to 17 s). The workers answer the cancel in
        the background; `_settle` takes those answers before anything is sent again."""
        if any(self._out.values()) and not self.cancel.is_set():
            self.cancel.set()
            POOL_COUNTS["dropped"] += sum(len(t) for t in self._out.values())
        self._spec_out.clear()
        self._spec_ready.clear()
        self._spec_flying.clear()

    def _settle(self) -> None:
        """The answers of the chunks `_drain` gave up, taken and thrown away where they are
        back, then the workers' stop lifted. A worker still inside one (a deeper cell looks
        at the stop only between its children's passes, and one pass can be seconds: an 8 s
        read that waited here began its first stage at 8 s) is not waited for: it is full
        until its answer comes back (`cells` passes it over and throws the answer away)."""
        try:
            for i, tasks in self._out.items():
                while tasks and self.conns[i].poll():
                    _status, task, *_rest = self.conns[i].recv()
                    tasks.remove(task)
                    self._orphan_tasks.discard(task)
                if tasks:
                    self._orphan_tasks.update(tasks)
                    self._orphans.add(i)
                else:
                    self._orphans.discard(i)
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
        if self.table is not None:
            self.table.close()
            self.table = None


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
    # IKA-378: the table the workers share their sub-games in (`SHARE`).
    table = subshare.Table() if SHARE else None
    started = []
    for _ in range(count):
        mine, theirs = context.Pipe()
        process = context.Process(target=_pool_main,
                                  args=(theirs, reg.meta.format_id, factory, tuple(args), cancel,
                                        None if table is None else table.name),
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
        _POOL = _Pool([p for p, _c in ready], [c for _p, c in ready], cancel, table)
    elif table is not None:
        table.close()
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
