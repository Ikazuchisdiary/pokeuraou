"""The best-first deepening's shared pieces: its budget, its report and its progress steps.

The deepening itself is the port's (`portdeepen`, `rust/src/deepen.rs`, IKA-464): a move's
root is the depth-1 node, and a budget of further cells is spent one cell at a time, best
first over the whole tree (IKA-33), the root re-solved after every step, so wherever the
budget stops the last equilibrium is an answer. The tree, the priorities (``bern/gap``,
IKA-281), the depth guard (IKA-307), the depth discount (IKA-342), the root's swap oracle
(IKA-293/310), the Bayesian root where a bench is hidden (IKA-294), the children's menus by
the Q (IKA-307 after IKA-322 section 18) and the widening mid-read (IKA-354) are there.
Python's own deepening (this module's tree before IKA-464) was removed in IKA-464 stage A3.

What stays here is what the callers and the port's road share:

* **the budget** (`Cost`, `COSTS`, `cells_for_seconds`, `_Meter`): a budget is cells, or a
  clock's units when a `Cost` prices each kind of counted work (IKA-293, IKA-342). A game on
  counted cells is the same game on any machine.
* **the report** (`Deepened`): what a deepened move did, written into its decision's record.
* **the progress** (`Step`, `Root`, `announce_depth1`): what a page reads while the answer
  forms (IKA-332).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .actions import SideAction
from .narrow import slot_options
from .regulation import Regulation

#: Levels below the root a refined cell may open. A guard, not a tuning knob: at the
#: default widths the budget runs out long before it, and a node with nothing left but
#: decided cells stops on its own (priority 0). A person's game raises it (IKA-307, IKA-342).
MAX_LEVELS = 8


@dataclass(frozen=True, slots=True)
class Cost:
    """What each kind of counted work costs, in milliseconds, on one machine form at one
    number of cores (IKA-293).

    A deepening spends three kinds of work, and their prices do not move together: a
    `fill` is one crossing to the port and one call into the leaf, whatever its size; a
    `refine` is one refined cell's own turn with every branch, the menus of each child
    and the re-solves up to the root; a `cell` is one resolved turn of a matrix and its
    leaves. One number of cells a second (IKA-33's 1,500) was 4x off on the middle game
    with a local GPU, because the count of fills and refinements per cell moves with the
    size of the child matrices. On more cores each kind shrinks differently (IKA-32), so a
    price belongs to a core count and is measured there, never scaled.
    """

    fill: float
    refine: float
    cell: float
    #: A probed cell of the root's double oracle (IKA-294's fourth kind, IKA-322), or
    #: None: priced as any `cell`. Probed cells are among `cells`; this reprices them.
    probe: float | None = None
    #: One inference of a Q that narrows the oracle's probe (``q<k>``, IKA-322).
    q: float = 0.0
    #: Each level below the root of a refined cell's node (IKA-342): a step re-solves every
    #: node on the path back to the root and walks a deeper tree, so a deep refinement
    #: costs more than a shallow one. `_Meter` adds it and `root`; `ms` does not read them
    #: (the clocks that stand in for a `Cost` -- `humanplay.WallCost`, `analysis.StopCost`
    #: -- have neither).
    level: float = 0.0
    #: Each cell of the root's game, a refinement (IKA-342): the root's LP is solved again
    #: after every step, and a wide Bayesian root (menus x completions) is most of a step.
    root: float = 0.0

    def ms(
        self, fills: int, refines: int, cells: int, probed: int = 0, qs: int = 0
    ) -> float:
        """Milliseconds of the counted work; `probed` of the `cells` were probes."""
        rest = cells - probed if self.probe is not None else cells
        return (
            fills * self.fill + refines * self.refine + rest * self.cell
            + (probed * self.probe if self.probe is not None else 0.0) + qs * self.q
        )


#: Measured prices, by (machine form, logical cores) -- IKA-293, 2026-09-26.
#:
#: ``local``: one process, the shipped ensemble leaf (value-mc0 + value-mc0-s1) on the
#: local GPU -- a person's opponent. Wall milliseconds, least squares over 624 deepenings
#: (78 recorded M-C positions x 8 labels, m25 to m1000o24): R^2 0.979 against 0.581 for
#: one number of cells a second (4,478/s at best; IKA-33's 1,500/s was 3.06x slow). Held
#: out by halves: R^2 0.976 / 0.980, totals within 2%.
#:
#: ``served``: the board's shape (24 workers over 2 inference servers, one server arm, the
#: 16 cores busy), CPU milliseconds of every process (workers, port, servers) above the
#: width-12 depth-1 arm, least squares over 11 same-arm board runs of 300 games (m100 to
#: m3000o24, swaps and breadth alone). A fill there costs what it costs locally; a cell
#: 5.6x (the port, the server's forward passes and the Python are all on the CPU).
#:
#: One core only (the counts above were run one process a worker); IKA-32 measures the
#: others, per kind, rather than scaling these.
#:
#: ``local``'s `level` and `root` (IKA-342, 2026-09-27): the long reads of the analysis
#: mode (width 64, a Bayesian root, the guard at 8 and 32) ran 1.5x (guard 8) to 2.2x
#: (guard 32) past the three prices above -- a step's time did not fall with its cells, and
#: rose with the refined cell's level and the root's size. Least squares over the steps of
#: 24 reads (8 recorded M-C positions x guards 8 / 32 / 32 with a discount, 30 s each, the
#: GPU leaf, one thread), the three prices held: 0.36 ms a level, 0.52 us a root cell (halves:
#: 0.41 / 0.40 ms and 0.54 / 0.12 us). The reads' wall over the priced clock went from 1.49
#: / 2.18 (guard 8 / 32) to 0.87 / 0.95.
#:
#: These prices were measured on Python's deepening (before IKA-464). The port's deepening
#: reads its budget in the same units (`portdeepen`), so a count-clock move stops where the
#: same work would have; how many cells the port reads in those milliseconds of wall time
#: was not measured again.
COSTS: dict[tuple[str, int], Cost] = {
    ("local", 1): Cost(fill=4.925, refine=1.494, cell=0.0286, level=0.36, root=0.00052),
    ("served", 1): Cost(fill=4.990, refine=1.354, cell=0.1602),
}

#: A record's deepening label where a side did not deepen (generation and the board never
#: deepen; `selfplay.GameRecord.deepen`).
DEFAULT_DEEPEN = "none"

#: The swap oracle's candidates as a width: every legal action (``sall``).
ALL_ACTIONS = 1 << 30


def cells_for_seconds(seconds: float, cores: int = 1, *, form: str = "local") -> int:
    """The budget that fits in `seconds` on `cores` logical cores (the human's clock), in
    units of one cell at `COSTS[form, cores]`'s prices.

    Spend it with ``cost=COSTS[form, cores]``: each fill and each refinement is then charged
    its own price in cells, so the budget stops where the clock would. It is still a count
    -- the same answer on a busy machine and an idle one.
    """
    try:
        cost = COSTS[form, cores]
    except KeyError:
        measured = sorted(COSTS)
        raise ValueError(
            f"no measured prices for {form!r} on {cores} core(s); measured: {measured}. "
            "A price is measured at its core count, not scaled (IKA-32)"
        ) from None
    return max(0, int(seconds * 1000.0 / cost.cell))


class _Meter:
    """What a deepening has spent, by kind, and the budget's reading of it.

    Without a `Cost` the budget counts cells as IKA-33 did (a refined cell's own turn is
    one); with one, each fill and refinement is charged its price in cells as well.
    `portdeepen` reads the port's counts into one before every step.
    """

    def __init__(self, cost: Any) -> None:  # noqa: ANN401 - a Cost or a clock in its place
        self.cost = cost
        self.fills = 0
        self.refines = 0
        self.cells = 0
        #: Of `cells`, the oracle's probes (IKA-322's fourth kind).
        self.probed = 0
        #: Inferences of the Q that narrows the probe. Counted, not charged to cells.
        self.qs = 0
        #: The refined cells' nodes' levels, summed (`Cost.level`'s count, IKA-342).
        self.levels = 0
        #: The root game's cells at each refinement, summed (`Cost.root`'s count).
        self.roots = 0

    def refined(self, cells: int, fills: int, level: int = 0, root: int = 0) -> None:
        self.refines += 1
        self.cells += cells
        self.fills += fills
        self.levels += level
        self.roots += root

    @property
    def spent(self) -> float:
        if self.cost is None:
            return self.refines + self.cells
        ms = self.cost.ms(self.fills, self.refines, self.cells, self.probed, self.qs)
        per_level = getattr(self.cost, "level", 0.0)
        if per_level and self.levels:
            ms += per_level * self.levels
        per_root = getattr(self.cost, "root", 0.0)
        if per_root and self.roots:
            ms += per_root * self.roots
        units = ms / self.cost.cell
        # A cost may hold the reading back (IKA-344's ponder: `humanplay.PonderCost` keeps
        # a move deepening past its budget until the person has chosen). ``work`` grows at
        # every step, so a stop point written as it replays the same stop.
        gate = getattr(self.cost, "gate", None)
        return units if gate is None else gate(units, self.work)

    @property
    def work(self) -> int:
        """Counted work so far: fills, refinements and cells. Every step adds to it."""
        return self.fills + self.refines + self.cells


#: Why a deepened line stopped (`Deepened.lines`, in this order).
LINE_STOPS = ("ended", "decided", "guard", "budget")


@dataclass(frozen=True, slots=True)
class Deepened:
    """What one deepened move did, for the record of the decision it answered."""

    #: The budget asked for, in cells.
    budget: int
    #: Cells resolved beyond the depth-1 matrix: each refined cell's turn and every cell
    #: of the child matrices it opened. Can pass `budget` by the last step.
    cells: int
    #: Cells given a deeper value, at any level.
    expanded: int
    #: Plies the deepest refined cell looks: 1 when nothing was refined, 2 when only root
    #: cells were, 3 when a child's cell was, and so on.
    depth: int
    #: Cells found unrefinable (a turn that paused for a mid-turn switch, a child with no
    #: action on one side, a child the LP could not solve). They keep their value.
    refused: int = 0
    #: Whether the root's double oracle was asked. The three counts below are written only
    #: then.
    oracle: bool = False
    #: Cells the oracle's probes filled (outside actions against the other side's support).
    probed: int = 0
    #: Actions the oracle added to the root, both sides.
    widened: int = 0
    #: Fills this deepening made (children, probes, added rows and columns): `Cost.fill`'s count.
    fills: int = 0
    #: Whether the oracle swapped: each action it added pushed a weightless one out.
    swap: bool = False
    #: Actions it pushed out.
    swapped: int = 0
    #: Each side's single slot options the root's menu covered before and no longer does:
    #: what the swaps took out of `narrow`'s cover, said rather than dropped.
    uncovered: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    #: The completions of the other side's bench the root was solved over (IKA-294's
    #: Bayesian root), or 0: a node with no bench hidden, read as one game.
    classes: int = 0
    #: Whether a Q narrowed the oracle's probe (``q<k>``, IKA-322; the port's oracle does
    #: not). The two counts below are written only then.
    q_probe: bool = False
    #: Inferences of the Q: one for an open root, one per completion for a Bayesian one.
    q: int = 0
    #: Steps whose short list gained nothing, so every outside action was probed.
    qfull: int = 0
    #: The depth guard the caller set, or None. The four below are always counted and
    #: written only when it is set.
    levels: int | None = None
    #: Why the deepening ended: "budget" (spent), "exhausted" (nothing left worth a
    #: step, the guard included), or "" (it never stepped: a zero budget).
    stop: str = ""
    #: The deepened lines -- the branches the tree ends in -- by why each stopped:
    #: (a finished game, a node with nothing left worth a step, a node at the guard with
    #: cells still worth one, a node the budget stopped first).
    lines: tuple[int, int, int, int] = (0, 0, 0, 0)
    #: Steps where a cell past the guard was worth more than the one taken.
    guarded: int = 0
    #: Whether the children's menus came from a Q. The two below then.
    child_q: bool = False
    #: The Q's forward passes for the children (one a refined cell) and the child
    #: positions they ranked.
    child_passes: int = 0
    child_ranked: int = 0
    #: Actions the root was widened by mid-read (`grow`, IKA-354), both sides, and the
    #: cells that filled. Written only when an action was added.
    grown: int = 0
    grown_cells: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "cells": self.cells,
            "expanded": self.expanded,
            "depth": self.depth,
            **({"classes": self.classes} if self.classes else {}),
            **({"refused": self.refused} if self.refused else {}),
            **(
                {"probed": self.probed, "widened": self.widened, "fills": self.fills}
                if self.oracle
                else {}
            ),
            **({"swapped": self.swapped} if self.swap else {}),
            **({"q": self.q, "qfull": self.qfull} if self.q_probe else {}),
            **(
                {"childQ": self.child_passes, "childRanked": self.child_ranked}
                if self.child_q
                else {}
            ),
            **(
                {
                    "levels": self.levels,
                    "stop": self.stop,
                    "lines": dict(zip(LINE_STOPS, self.lines, strict=True)),
                    "guarded": self.guarded,
                }
                if self.levels is not None
                else {}
            ),
            **(
                {"uncovered": [list(side) for side in self.uncovered]}
                if any(self.uncovered)
                else {}
            ),
            **(
                {"grown": self.grown, "grownCells": self.grown_cells}
                if self.grown
                else {}
            ),
        }


@dataclass
class Root:
    """A side's root as a page reads it (`progress.Reader`): the Bayesian game of
    `search.belief_solve` (an open node is one exact completion of weight 1).

    One row set (this side's actions, ``own``), a matrix per completion ``k`` of the other
    side's bench in this side's orientation (``prices``; side 1's is side 0's negated
    transpose), their weights ``w`` (summing to 1), and the Bayesian equilibrium over them.
    ``deep`` and ``refused`` are the root's cells ``(k, i, j)`` the deepening read deeper
    than the leaf, and the ones it could not. ``port`` is the port's snapshot it was read
    from (None at a depth-1 answer)."""

    side: int
    own: list[SideAction]
    other: list[SideAction]
    items: list[Any]
    w: np.ndarray
    prices: list[np.ndarray]
    equilibrium: Any  # BayesianEquilibrium
    deep: set[tuple[int, int, int]] = field(default_factory=set)
    refused: set[tuple[int, int, int]] = field(default_factory=set)
    port: dict[str, Any] | None = None

    @property
    def value(self) -> float:
        return float(self.equilibrium.value)


@dataclass(frozen=True, slots=True)
class Step:
    """One report of a deepening's progress (IKA-332), for a `progress` callback.

    The counts are the ones `Deepened` ends with, as they stand after this step. `root`
    is a `Root`: read it inside the callback, and never write to it.
    """

    #: 0 at the depth-1 answer, then 1, 2, ... per slice the port ran; ``done`` repeats
    #: the last.
    index: int
    #: ``start``, ``refine`` (a slice of the read) or ``done``.
    kind: str
    root: Any
    budget: int
    #: The budget's reading of the work so far (`_Meter.spent`): cells, or the cost's units.
    spent: float
    cells: int
    fills: int
    refines: int
    expanded: int
    depth: int
    refused: int
    probed: int = 0
    widened: int = 0
    swapped: int = 0
    #: Actions the root was widened by mid-read so far (`grow`, IKA-354).
    grown: int = 0


#: A progress callback: called with each `Step`, returns nothing, changes nothing.
Progress = Callable[[Step], None]

#: A widening of the root mid-read (IKA-354): called before each slice of the read with the
#: budget's reading so far (`_Meter.spent`), it returns each side's actions to add -- (side
#: 0's, side 1's) -- or None.
Grow = Callable[[float], "tuple[Sequence[SideAction], Sequence[SideAction]] | None"]


def announce_depth1(
    progress: Progress,
    side: int,
    own: Sequence[SideAction],
    other: Sequence[SideAction],
    items: Sequence[Any],
    weights: np.ndarray,
    prices: Sequence[np.ndarray],
    equilibrium: Any,  # noqa: ANN401 - BayesianEquilibrium
) -> None:
    """A side's node that is not deepened (`belief_solve`), reported as one: ``start`` and
    ``done`` at depth 1."""
    w = np.asarray(weights, dtype=np.float64)
    root = Root(side=side, own=list(own), other=list(other), items=list(items), w=w / w.sum(),
                prices=[np.asarray(p, dtype=np.float64) for p in prices], equilibrium=equilibrium)
    for kind in ("start", "done"):
        progress(Step(index=0, kind=kind, root=root, budget=0, spent=0.0, cells=0, fills=0,
                      refines=0, expanded=0, depth=1, refused=0))


def _lost_options(
    reg: Regulation, given: Sequence[SideAction], now: Sequence[SideAction]
) -> tuple[str, ...]:
    """Labels of the single slot options `given` covered and `now` does not."""
    before = slot_options(reg, given)
    after = slot_options(reg, now)
    return tuple(sorted(before[key] for key in set(before) - set(after)))


__all__ = [
    "ALL_ACTIONS",
    "COSTS",
    "DEFAULT_DEEPEN",
    "LINE_STOPS",
    "MAX_LEVELS",
    "Cost",
    "Deepened",
    "Grow",
    "Progress",
    "Root",
    "Step",
    "announce_depth1",
    "cells_for_seconds",
]
