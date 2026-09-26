"""Deepening a solved node one cell at a time, best first, until a cell budget runs out.

`search` at depth 1 always has an answer: every cell of the menu has a leaf value and the
matrix is solved. This spends a budget of further cells on top of that answer, one cell at
a time, and after every step the matrix is solved again -- so wherever the budget stops,
the last equilibrium is an answer (IKA-33, "anytime").

**One step.** The cell with the highest priority anywhere in the tree is given a value a
ply further, exactly as `search._refined_value` does it: the turn is resolved with every
branch, the `sub_branches` likeliest are kept and renormalised, and each branch position
gets its own depth-1 matrix over `narrow`'s `sub_limit` damage-ranked candidates. The first
step at the root is that function's number to the bit (tested). What differs is that the
child matrices are kept: their cells are candidates too, so a cell can go three, four plies
deep when that is where the budget is best spent. Every node on the path back to the root
is re-solved and its value handed up.

**The priority** is IKA-281's best cheap signal, ``bern/gap``: d(1 - d) / (regret_i +
regret_j + 0.001), with d the cell's current value and the regrets its two actions' losses
against the node's current equilibrium. How far a cell moves when deepened is predicted by
how close its value is to 0.5 (AUC 0.83 over 44,722 cells), and whether moving it can
change the answer by how close its actions are to a best response. At the root that
number is the priority. Below it, a cell inherits its parent cell's priority times the
branch's weight, scaled by its own signal over the best one in its node -- so the best cell
of a child is worth exactly what the parent cell was, discounted by how likely the child
is, and a child is only deepened once nothing at the level above is worth more.

**The budget is cells, not seconds** -- one per resolved turn: the refined cell's own
full turn and every cell of each child matrix. Generation and the board stop on it, so a
game with the same seed plays the same moves on a busy machine and an idle one. Seconds
are for a human opponent only, and `cells_for_seconds` turns one into the other outside
the search. The last step may overshoot by one refined cell's children (at most
``1 + sub_branches * sub_limit**2``, 193 at the defaults); it is counted, not hidden.

**Two readings of the root** (the label's letter, `parse_deepen`). ``m`` reads the whole
matrix: deepened cells hold their deeper value, the rest their leaf value. IKA-12 lost with
that reading when the refined cells were the support's (x_i * y_j); IKA-281 measured that
choice as no better than random against the all-cells depth-2 answer, and ``bern/gap`` as
the best cheap rule under the same reading. ``r`` reads the root as IKA-68's restricted
game: the rectangle of the depth-1 support (`refine` actions a side) is the game, its cells
the only root candidates, and once all of them are refined the full matrix is asked, as a
double oracle, whether a row or a column outside beats the rectangle's value -- if so it
joins and its cells become candidates. The strategy is the rectangle's, and the value what
it guarantees against every column. A matrix of mixed depths is what IKA-12 lost to; the
rectangle keeps the cells a strategy is read from at one depth or deeper. Child nodes are
read whole under either (their menus are `sub_limit` wide).

Each decision reports what it did (`Deepened`): cells spent, cells refined, and the depth
the deepest refined cell reached, so a recorded game says how far each answer looked.

**Breadth at the root, as a double oracle** (IKA-293, the label's ``o<W>`` / ``oall``,
mixed reading only). Before every step the actions outside the root's menu -- the rest of
a width-W menu built with the same ranking, or every legal action -- are asked, in one
fill, what they get against the other side's current support: a row's gain is its payoff
against the column strategy less the root's value, a column's the root's value less what
the row strategy gets against it. That is the best-response gain, the bound on how far
adding the action can move the root's value; IKA-293 measured it picking the action that
moves the root most first in 78% of positions. If any action gains more than
`ORACLE_TOLERANCE`, the one that gains most joins the root (the rest of its row or column
filled at depth 1) and the root is re-solved; otherwise the step is a deepening step as
above. Cells already asked are kept, so a probe fills only what the support's changes
made new. Every probed and added cell is charged to the budget; a probe and what
follows it (a widening or a deepening step) are one step, so the budget is checked
before each and the last may pass it. Without candidates outside the menu this is
`m<N>` to the bit.

Two variants (the user's decision of 9/26, IKA-310's measurement). ``s<W>`` / ``sall``
**swaps** instead of only adding: once the action joins and the root is re-solved,
an action of the same side that carries no weight in that equilibrium leaves -- the
one that loses most against it, never one with a refined cell -- so the menu keeps
its size. Removing a weightless action leaves the equilibrium standing. That may
break `narrow`'s cover (every single slot option somewhere on the menu); the options
the root no longer covers are reported (`Deepened.uncovered`, written into the
decision's record) rather than dropped quietly, and a removed action is outside the
menu again, so the next probe asks it like any other and it can come back. The
``b<N>`` reading is **breadth only**: the oracle alone, no deepening -- it stops when
nothing outside gains, or at the budget.

**Where a bench is hidden** (IKA-294, a label ending in ``h``). Each side's game there is
Bayesian (`search.belief_solve`): one row set, a matrix per completion of the other
side's bench, the other side's reply per completion. `deepen_belief` deepens that root
the same way, over cells ``(completion, row, column)``: refining one resolves the turn in
that completion's position and solves the next turn as if that bench were known (IKA-111's
determinization), and the root is re-solved as the Bayesian game after every step. A
cell's priority is ``bern/gap`` read in its completion, times the completion's weight
(`BELIEF_WEIGHTED`). The double oracle asks rows against each completion's reply and
columns against the one row strategy, completion by completion (`_BeliefOracle`); it
swaps under the same three rules. Without the ``h`` only the nodes with no bench hidden
deepen, as before.

**A probe narrowed by a Q** (IKA-322, the oracle's ``q<k>`` in place of a width:
``m<N>sq3h``). The candidates are every legal action, as ``all``'s, but a step's probe
fills only the cells of each side's `k` outside actions a learned Q (`qrank`, IKA-274)
ranks best against the root's current equilibrium: a row by its Q payoff against the
column strategy (per completion's reply, weighted, on a Bayesian root), a column by
what it saves against the row strategy (per completion, weighted, the ``max 0`` of
`_BeliefOracle`). The real cells decide, exactly as the unnarrowed oracle's: the one
of the short list that gains most joins. When none of them gains, the step probes
every outside action once (the proof that nothing outside gains -- IKA-322's
``Bqk3f``, which kept the whole game's equilibrium in reach with 36% fewer probed
cells). Once per menu: when that full probe finds nothing too, the deepening steps
after it -- which move the prices a little -- ask the short list alone, until an action
joins and the menu changes (probing everything before every deepening step would be
``sall`` again). The Q is asked once per node: over both sides' whole candidate lists, in the
root's position, or in each completion's on a Bayesian root; it never becomes a
value, it only picks which cells are probed first. Its calls are counted
(`Deepened.q`), not charged to a budget of cells; `Cost` prices them.

**What a budget costs.** Labels count cells, so a game is the same game on any machine.
Time is counted separately (`Cost`): fills, refinements and cells each have a price in
milliseconds, measured per machine form and per number of cores, and `cells_for_seconds`
turns a clock into a budget in those prices' units (IKA-293; IKA-32 keeps the prices per
kind of work because each shrinks differently on more cores). Two more kinds are counted
apart (IKA-322): the oracle's probed cells (IKA-294 found a hidden root's probe cheaper a
cell than a refinement's, since a cell no hidden slot reaches is resolved once for every
completion) and the Q's inferences.

**Watching it think** (IKA-332). `deepen_root` and `deepen_belief` take an optional
`progress` callback: it is called with a `Step` once at the depth-1 answer (``start``),
once after every step (``refine``, ``refused``, ``widen``) and once at the end (``done``,
the answer returned). A `Step` holds the counts and a live reference to the root; the
callback reads what is already there and must not write. Without one nothing is called
and nothing is computed that was not before; with one the search does the same work in
the same order, so a game on counted cells is the same game either way (a wall-clock game
is not replayed by seed in any case). `announce_depth1` / `announce_belief_depth1` give a
node that is not deepened the same two calls.

**Expanding ahead on more cores** (IKA-32 stage 2, `set_ahead`; off by default, and
generation and the board never turn it on). The loop is serial by definition -- each
step's cell is the best under the equilibrium the last step left -- but what a step
*does* to its cell is not: a refined cell's turn, its branches' menus, their matrices and
leaves depend on the cell's position and its two actions alone. With `set_ahead(n)` a
helper thread expands the cells the loop is likely to take next (the `n` best under the
tree as it stands), in batches: every turn in one crossing to the port, every branch's
two menus in one (`narrow_many`), every child matrix in one (`pending_payoffs`), their
leaves in one call (`score_segments`, each block scored as it would be alone). The loop
still picks its cell as before and takes the expansion from the helper when it is there;
a cell the helper did not guess is asked of it first. The LP of each child is solved
when its cell is taken, and a cell whose ahead-expansion met an error is expanded again
the serial way, so it raises what the serial loop would raise. What the step writes --
children, notes, counts -- is the serial step's, so a game on counted cells is the same
game with or without it; the wasted expansions of cells never taken cost only time.
While the loop re-solves the root (HiGHS lets go of the GIL) the helper talks to the
port and the leaf.
"""

from __future__ import annotations

import atexit
import contextlib
import heapq
import os
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import port, timing
from .actions import SideAction
from .budget import Budget
from .equilibrium import Equilibrium, EquilibriumError, solve
from .narrow import narrow, slot_options
from .port import batched_payoff
from .position import Position
from .regulation import Regulation

LeafEvaluator = Callable[[list[Position]], np.ndarray]

#: Added to a regret before dividing by it (IKA-281's `GAP_FLOOR`), so an action on the
#: support ranks by its uncertainty rather than by an infinity.
GAP_FLOOR = 1e-3

#: A strategy's weight at or under this is none: the actions a swap may push out.
WEIGHTLESS = 1e-12

#: How far a best response outside the restricted rectangle has to beat its value to join
#: it (`search.ORACLE_TOLERANCE`, repeated so this module does not import the search).
ORACLE_TOLERANCE = 1e-6

#: Levels below the root a refined cell may open. A guard, not a tuning knob: at the
#: default widths the budget runs out long before it, and a node with nothing left but
#: decided cells stops on its own (priority 0).
MAX_LEVELS = 8

@dataclass(frozen=True, slots=True)
class Cost:
    """What each kind of counted work costs, in milliseconds, on one machine form at one
    number of cores (IKA-293).

    A deepening spends three kinds of work, and their prices do not move together: a
    `fill` is one crossing to the port and one call into the leaf, whatever its size; a
    `refine` is one refined cell's own turn with every branch, the `narrow` of each child
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
COSTS: dict[tuple[str, int], Cost] = {
    ("local", 1): Cost(fill=4.925, refine=1.494, cell=0.0286),
    ("served", 1): Cost(fill=4.990, refine=1.354, cell=0.1602),
}

#: What ships: no deepening. A label is ``none``, ``m<N>`` (the whole matrix read, N cells)
#: or ``r<N>`` (the restricted reading, N cells), and ``m<N>`` may end in ``o<W>`` or
#: ``oall``: the root's double oracle over the rest of a width-W menu, or every legal action
#: -- or in ``s<W>`` / ``sall``, the oracle that swaps a weightless action out for each one
#: it adds. ``b<N>`` with either suffix is the oracle alone, without deepening.
DEFAULT_DEEPEN = "none"

READINGS = {"m": "mixed", "r": "restricted", "b": "breadth"}

#: The oracle's candidates as a width: ``oall`` is every legal action.
ALL_ACTIONS = 1 << 30

_DEEPEN = re.compile(r"none|([mrb])([1-9][0-9]*)(?:([os])([1-9][0-9]*|all|q[1-9][0-9]*))?(h)?")


@dataclass(frozen=True, slots=True)
class DeepenSpec:
    """A deepen label read: how the root is read, the cells, and the oracle's width."""

    reading: str | None
    cells: int
    #: Width of the menu whose rest the root's double oracle asks (`ALL_ACTIONS` for every
    #: legal action), or None: no breadth at the root.
    oracle: int | None = None
    #: Whether the oracle swaps a weightless action out for each one it adds (``s``).
    swap: bool = False
    #: Whether it deepens (and widens) where a bench is still hidden too (``h``, IKA-294):
    #: the Bayesian root of `deepen_belief`. Without it only the nodes with no bench
    #: hidden deepen, as IKA-33 and IKA-293 did.
    hidden: bool = False
    #: The oracle's probe narrowed to each side's k best outside actions by a Q
    #: (``q<k>``, IKA-322; the candidates are every legal action), or None: it probes
    #: every candidate.
    q_probe: int | None = None


def deepen_spec(label: str) -> DeepenSpec:
    """`DeepenSpec` of a label: ``none``, ``m400``, ``r25``, ``m400o24``, ``m400sall``,
    ``b200s24``, ``m400sallh``, ``m1200sq3h``."""
    got = _DEEPEN.fullmatch(label)
    if got is None:
        raise ValueError(
            f"deepen {label!r} is not none, m<N>, r<N>, m<N>o<W>, m<N>s<W>, b<N>o<W> or "
            "b<N>s<W> (N cells, N >= 1; W the oracle's width, all, or q<k>: every action, "
            "the probe narrowed to a Q's k best a side), each but r<N> optionally ending "
            "in h (hidden nodes too)"
        )
    if got.group(1) is None:
        return DeepenSpec(None, 0)
    letter, kind, oracle = got.group(1), got.group(3), got.group(4)
    hidden = got.group(5) is not None
    if oracle is not None and letter == "r":
        raise ValueError(
            f"deepen {label!r}: the root's double oracle goes with the whole-matrix reading "
            "(m); the restricted one (r) already grows its rectangle by its own oracle"
        )
    if hidden and letter == "r":
        raise ValueError(
            f"deepen {label!r}: a hidden bench's restricted reading is depth 2 "
            "(`--depth 2 --solve-restricted`, IKA-111); h goes with m and b"
        )
    if oracle is None and letter == "b":
        raise ValueError(
            f"deepen {label!r}: breadth only (b) is the root's double oracle alone; it "
            "needs o<W> / s<W> / oall / sall"
        )
    q_probe = None
    if oracle is None:
        width = None
    elif oracle == "all":
        width = ALL_ACTIONS
    elif oracle.startswith("q"):
        width, q_probe = ALL_ACTIONS, int(oracle[1:])
    else:
        width = int(oracle)
    return DeepenSpec(
        READINGS[letter], int(got.group(2)), width, kind == "s", hidden, q_probe
    )


def parse_deepen(label: str) -> tuple[str | None, int]:
    """(reading, cells) from a deepen label: (None, 0), ("mixed", 400), ("restricted", 400)."""
    spec = deepen_spec(label)
    return spec.reading, spec.cells


def cells_for_seconds(seconds: float, cores: int = 1, *, form: str = "local") -> int:
    """The budget that fits in `seconds` on `cores` logical cores (the human's clock), in
    units of one cell at `COSTS[form, cores]`'s prices.

    Spend it with ``cost=COSTS[form, cores]`` (`search(deepen_cost=)`): each fill and each
    refinement is then charged its own price in cells, so the budget stops where the
    clock would. It is still a count -- the same answer on a busy machine and an idle one.
    Generation and the board never call this: they count cells alone.
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


@dataclass(frozen=True, slots=True)
class Deepened:
    """What one deepened search did, for the record of the decision it answered."""

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
    #: Whether the root's double oracle was asked (a label with ``o``). The three counts
    #: below are written only then.
    oracle: bool = False
    #: Cells the oracle's probes filled (outside actions against the other side's support).
    probed: int = 0
    #: Actions the oracle added to the root, both sides.
    widened: int = 0
    #: Fills this deepening made (children, probes, added rows and columns): `Cost.fill`'s count.
    fills: int = 0
    #: Whether the oracle swapped (``s``): each action it added pushed a weightless one out.
    swap: bool = False
    #: Actions it pushed out (``s``).
    swapped: int = 0
    #: Each side's single slot options the root's menu covered before and no longer does
    #: (``s``): what the swaps took out of `narrow`'s cover, said rather than dropped.
    uncovered: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    #: The completions of the other side's bench the root was solved over (IKA-294's
    #: Bayesian root), or 0: a node with no bench hidden, read as one game.
    classes: int = 0
    #: Whether a Q narrowed the oracle's probe (``q<k>``, IKA-322). The two counts below
    #: are written only then.
    q_probe: bool = False
    #: Inferences of the Q: one for an open root, one per completion for a Bayesian one.
    q: int = 0
    #: Steps whose short list gained nothing, so every outside action was probed.
    qfull: int = 0

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
                {"uncovered": [list(side) for side in self.uncovered]}
                if any(self.uncovered)
                else {}
            ),
        }


@dataclass(eq=False)
class _Node:
    """One matrix game in the tree: the root, or a branch position a refined cell opened."""

    pos: Position | None
    rows: list[SideAction]
    cols: list[SideAction]
    payoff: np.ndarray
    equilibrium: Equilibrium
    #: Plies from the root's own turn: 0 at the root.
    level: int
    #: (node, cell) this one is a branch of, and its weight there. None at the root.
    parent: tuple[_Node, tuple[int, int]] | None = None
    weight: float = 1.0
    #: Refined cells: their branches, each a child node or a finished position's value.
    children: dict[tuple[int, int], list[tuple[float, _Node | float]]] = field(
        default_factory=dict
    )
    refused: set[tuple[int, int]] = field(default_factory=set)
    #: `_signal`'s matrix while the equilibrium it was read from stands.
    signal: np.ndarray | None = None
    #: The restricted reading's rectangle (rows, columns), at the root under ``r``; None
    #: where the node is read whole.
    rect: tuple[list[int], list[int]] | None = None

    @property
    def value(self) -> float:
        return float(self.equilibrium.value)


@dataclass(frozen=True, slots=True)
class Deepening:
    """`deepen_root`'s answer: the root's game as it stands when the budget ran out."""

    equilibrium: Equilibrium
    payoff: np.ndarray
    #: The root's menus -- the ones it was given, plus whatever the oracle added, in the
    #: order it added them. The strategies index these.
    rows: list[SideAction]
    cols: list[SideAction]
    report: Deepened


@dataclass(frozen=True, slots=True)
class Step:
    """One report of a deepening's progress (IKA-332), for a `progress` callback.

    The counts are the ones `Deepened` ends with, as they stand after this step. `root`
    is the live root (`_Node`, or `_BeliefRoot` where a bench is hidden): read it inside
    the callback, since the next step moves it, and never write to it.
    """

    #: 0 at the depth-1 answer, then 1, 2, ... per step; ``done`` repeats the last.
    index: int
    #: ``start``, ``refine`` (a cell took a deeper value), ``refused``, ``widen`` (the
    #: oracle added an action) or ``done``.
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
    #: The cell this step refined or refused, and its node's level (0 at the root).
    cell: tuple[int, ...] | None = None
    level: int | None = None


#: A progress callback: called with each `Step`, returns nothing, changes nothing.
Progress = Callable[[Step], None]


def _stepper(
    progress: Progress, root: Any, meter: _Meter, cells: int, oracle: Any  # noqa: ANN401
) -> Callable[..., None]:
    """The callback, bound to one deepening's root, meter and oracle."""
    count = [0]

    def announce(
        kind: str, expanded: int, deepest: int, refused: int,
        cell: tuple[int, ...] | None = None, level: int | None = None,
    ) -> None:
        if kind not in ("start", "done"):
            count[0] += 1
        progress(Step(
            index=count[0], kind=kind, root=root, budget=cells, spent=float(meter.spent),
            cells=meter.refines + meter.cells, fills=meter.fills, refines=meter.refines,
            expanded=expanded, depth=1 + deepest, refused=refused,
            probed=0 if oracle is None else oracle.probed,
            widened=0 if oracle is None else oracle.widened,
            swapped=0 if oracle is None else oracle.swapped,
            cell=None if cell is None else tuple(int(c) for c in cell), level=level,
        ))

    return announce


def announce_depth1(
    progress: Progress,
    pos: Position,
    rows: Sequence[SideAction],
    cols: Sequence[SideAction],
    payoff: np.ndarray,
    equilibrium: Equilibrium,
) -> None:
    """A node that is not deepened, reported as one: ``start`` and ``done`` at depth 1.
    The root handed over holds references to the answer, nothing copied or solved."""
    root = _Node(pos=pos, rows=list(rows), cols=list(cols), payoff=payoff,
                 equilibrium=equilibrium, level=0)
    announce = _stepper(progress, root, _Meter(None), 0, None)
    announce("start", 0, 0, 0)
    announce("done", 0, 0, 0)


def announce_belief_depth1(
    progress: Progress,
    side: int,
    own: Sequence[SideAction],
    other: Sequence[SideAction],
    items: Sequence[Any],
    weights: np.ndarray,
    prices: Sequence[np.ndarray],
    equilibrium: Any,  # noqa: ANN401 - BayesianEquilibrium
) -> None:
    """`announce_depth1` for a side's Bayesian node (`belief_solve` without deepening)."""
    root = _BeliefRoot(side, own, other, items, weights, prices, equilibrium)
    announce = _stepper(progress, root, _Meter(None), 0, None)
    announce("start", 0, 0, 0)
    announce("done", 0, 0, 0)


def best_first(
    reg: Regulation,
    pos: Position,
    rows: Sequence[SideAction],
    cols: Sequence[SideAction],
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    payoff: np.ndarray,
    equilibrium: Equilibrium,
    cells: int,
    sub_limit: int,
    sub_branches: int,
    unmodelled: set[str],
    reading: str = "mixed",
    refine: int = 4,
    trace: list | None = None,
) -> tuple[Equilibrium, np.ndarray, Deepened]:
    """The root's equilibrium and prices after spending `cells` on deepening, best first.

    `payoff` and `equilibrium` are the depth-1 answer, which is returned unchanged when
    the budget is zero or nothing is worth deepening. `unmodelled` collects the port's
    notes from every turn resolved here. `trace`, when given, receives the root node and
    then one ``(node, cell, took)`` per step -- for the tests, which check the tree.
    """
    got = deepen_root(
        reg, pos, rows, cols, evaluate, budget=budget, payoff=payoff,
        equilibrium=equilibrium, cells=cells, sub_limit=sub_limit,
        sub_branches=sub_branches, unmodelled=unmodelled, reading=reading, refine=refine,
        trace=trace,
    )
    return got.equilibrium, got.payoff, got.report


def deepen_root(
    reg: Regulation,
    pos: Position,
    rows: Sequence[SideAction],
    cols: Sequence[SideAction],
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    payoff: np.ndarray,
    equilibrium: Equilibrium,
    cells: int,
    sub_limit: int,
    sub_branches: int,
    unmodelled: set[str],
    reading: str = "mixed",
    refine: int = 4,
    trace: list | None = None,
    outside: tuple[Sequence[SideAction], Sequence[SideAction]] | None = None,
    cost: Cost | None = None,
    swap: bool = False,
    q_probe: int | None = None,
    progress: Progress | None = None,
) -> Deepening:
    """`best_first`, and the root's double oracle when `outside` is given (IKA-293).

    `outside` is each side's candidates for the root: actions already on its menu are
    skipped, the rest are asked every step whether they gain against the other side's
    support, and the best that gains joins the root (see the module's docstring). Given,
    even with nothing outside the menu, the report says what the oracle did.

    `swap` pushes a weightless action of the same side out for each one that joins, and
    the ``breadth`` reading is the oracle alone, without deepening (both need `outside`).
    `q_probe` narrows each probe to the process's Q's best `q_probe` outside actions a
    side, all of them once when those gain nothing (``q<k>``, IKA-322).

    `cost` changes only how the budget is counted: None counts cells (a refined cell's
    turn is one), a `Cost` charges each fill, refinement and cell its price in units of
    one cell (`cells_for_seconds`'s budget).

    `progress`, when given, is called with a `Step` at the start, after every step and at
    the end (the module's docstring); it changes nothing the deepening computes.
    """
    if outside is not None and reading not in ("mixed", "breadth"):
        raise ValueError("the root's double oracle goes with the mixed reading")
    if outside is None and (swap or reading == "breadth" or q_probe is not None):
        raise ValueError(
            "swapping, breadth only and a Q's probe are the root's double oracle's; no outside"
        )
    deepens = reading != "breadth"
    root = _Node(
        pos=pos, rows=list(rows), cols=list(cols),
        payoff=np.array(payoff, dtype=np.float64, copy=True),
        equilibrium=equilibrium, level=0,
    )
    if reading == "restricted":
        root.rect = (
            [int(i) for i in _top(equilibrium.row_strategy, refine)],
            [int(j) for j in _top(equilibrium.col_strategy, refine)],
        )
        _reread(root)
    elif reading not in ("mixed", "breadth"):
        raise ValueError(f"reading {reading!r} is not one of {sorted(READINGS.values())}")
    if trace is not None:
        trace.append(root)
    meter = _Meter(cost)
    oracle = (
        None if outside is None else _Oracle(root, outside, swap=swap, q_probe=q_probe)
    )
    expanded = 0
    deepest = 0
    refused = 0
    announce = None if progress is None else _stepper(progress, root, meter, cells, oracle)
    if announce is not None:
        announce("start", expanded, deepest, refused)
    helper = (
        _start_ahead(reg, evaluate, budget=budget, sub_limit=sub_limit, sub_branches=sub_branches)
        if deepens and cells > 0
        else None
    )
    try:
        while meter.spent < cells:
            if oracle is not None:
                # One step: the probe and what it leads to -- a widening, or else a deepening.
                with _held(helper):
                    joined = oracle.step(reg, evaluate, budget, meter, unmodelled)
                if joined:
                    if trace is not None:
                        trace.append((root, None, True))
                    if announce is not None:
                        announce("widen", expanded, deepest, refused)
                    continue
                if not deepens:
                    break
            target = _best(root)
            if target is None:
                break
            node, cell = target
            if helper is None:
                spent, ok, fills = _expand(
                    reg, node, cell, evaluate, budget=budget, sub_limit=sub_limit,
                    sub_branches=sub_branches, unmodelled=unmodelled,
                )
            else:
                helper.post(root)
                spent, ok, fills = helper.expand(node, cell, unmodelled)
            meter.refined(spent - 1, fills)
            if trace is not None:
                trace.append((node, cell, ok))
            if not ok:
                node.refused.add(cell)
                refused += 1
                if node.rect is not None:
                    _reread(node)
                if announce is not None:
                    announce("refused", expanded, deepest, refused, cell, node.level)
                continue
            expanded += 1
            deepest = max(deepest, node.level + 1)
            _propagate(node, cell)
            if announce is not None:
                announce("refine", expanded, deepest, refused, cell, node.level)
    finally:
        if helper is not None:
            helper.close()
    if timing.ON:
        timing.count("deepen.cells", meter.refines + meter.cells)
        timing.count("deepen.expanded", expanded)
        timing.count("deepen.fills", meter.fills)
        timing.count("deepen.refines", meter.refines)
        if oracle is not None:
            timing.count("deepen.oracle.calls", 1)
            timing.count("deepen.oracle.probes", oracle.probes)
            timing.count("deepen.oracle.probed", oracle.probed)
            timing.count("deepen.oracle.widened", oracle.widened)
            timing.count("deepen.oracle.swapped", oracle.swapped)
            timing.count("deepen.oracle.stalled", int(oracle.stalled))
            if q_probe is not None:
                timing.count("deepen.oracle.q", meter.qs)
                timing.count("deepen.oracle.qfull", oracle.fallbacks)
    uncovered: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    if oracle is not None and oracle.swapped:
        uncovered = (
            _lost_options(reg, rows, root.rows),
            _lost_options(reg, cols, root.cols),
        )
        if timing.ON:
            timing.count("deepen.oracle.uncovered", len(uncovered[0]) + len(uncovered[1]))
    report = Deepened(
        budget=cells, cells=meter.refines + meter.cells, expanded=expanded,
        depth=1 + deepest, refused=refused, oracle=oracle is not None,
        probed=0 if oracle is None else oracle.probed,
        widened=0 if oracle is None else oracle.widened,
        fills=meter.fills, swap=swap,
        swapped=0 if oracle is None else oracle.swapped, uncovered=uncovered,
        q_probe=q_probe is not None, q=meter.qs,
        qfull=0 if oracle is None else oracle.fallbacks,
    )
    if announce is not None:
        announce("done", expanded, deepest, refused)
    return Deepening(
        equilibrium=root.equilibrium, payoff=root.payoff, rows=root.rows, cols=root.cols,
        report=report,
    )


def _lost_options(
    reg: Regulation, given: Sequence[SideAction], now: Sequence[SideAction]
) -> tuple[str, ...]:
    """Labels of the single slot options `given` covered and `now` does not."""
    before = slot_options(reg, given)
    after = slot_options(reg, now)
    return tuple(sorted(before[key] for key in set(before) - set(after)))


class _Meter:
    """What a deepening has spent, by kind, and the budget's reading of it.

    Without a `Cost` the budget counts cells as IKA-33 did (a refined cell's own turn is
    one); with one, each fill and refinement is charged its price in cells as well.
    """

    def __init__(self, cost: Cost | None) -> None:
        self.cost = cost
        self.fills = 0
        self.refines = 0
        self.cells = 0
        #: Of `cells`, the oracle's probes (IKA-322's fourth kind).
        self.probed = 0
        #: Inferences of the Q that narrows the probe. Counted, not charged to cells.
        self.qs = 0

    def refined(self, cells: int, fills: int) -> None:
        self.refines += 1
        self.cells += cells
        self.fills += fills

    def filled(self, cells: int, *, probe: bool = False) -> None:
        self.fills += 1
        self.cells += cells
        if probe:
            self.probed += cells

    def asked_q(self, count: int) -> None:
        self.qs += count

    @property
    def spent(self) -> float:
        if self.cost is None:
            return self.refines + self.cells
        return (
            self.cost.ms(self.fills, self.refines, self.cells, self.probed, self.qs)
            / self.cost.cell
        )


class _Oracle:
    """The root's double oracle over actions outside its menu (IKA-293).

    Keeps every cell it has asked, by the two actions' choices, so a probe fills only the
    pairs the support's last change made new, and an action that joins brings the cells
    already known into its row or column.
    """

    def __init__(
        self,
        root: _Node,
        outside: tuple[Sequence[SideAction], Sequence[SideAction]],
        *,
        swap: bool = False,
        q_probe: int | None = None,
    ) -> None:
        self.root = root
        self.swap = swap
        self.q_probe = q_probe
        #: The Q over both sides' candidates (side 0's value), asked at the first step.
        self.q: np.ndarray | None = None
        self.q_at: tuple[dict[str, int], dict[str, int]] = ({}, {})
        #: Steps whose Q short list gained nothing, so every outside action was probed.
        self.fallbacks = 0
        #: Whether a full probe found nothing and no action has joined since: the
        #: short list alone is asked until one does (IKA-322, once per menu).
        self.proved = False
        # The menu's own actions are candidates too, after the given ones: an action a
        # swap pushed out is outside again and can come back.
        self.candidates: tuple[list[SideAction], list[SideAction]] = ([], [])
        for side, menu in ((0, root.rows), (1, root.cols)):
            seen: set[str] = set()
            for action in (*outside[side], *menu):
                if action.to_choice() not in seen:
                    seen.add(action.to_choice())
                    self.candidates[side].append(action)
        self.known: dict[tuple[str, str], float] = {}
        #: Actions that could not join (the LP failed with them); never asked again.
        self.dropped: tuple[set[str], set[str]] = (set(), set())
        self.probes = 0
        self.probed = 0
        self.widened = 0
        self.swapped = 0
        # Actions a swap has pushed out. Each leaves at most once: swapping every
        # weightless action cycled (a row joins, pushes one out, a column's answer to it
        # brings that one back, ...) in 57 of 60 random 9x8 games, and a cell asked once
        # costs nothing the second time, so the budget would not stop it. With each
        # leaving once, every action joins at most twice and the oracle ends -- where
        # nothing outside gains, which is the whole game's equilibrium.
        self.left: set[str] = set()
        # A guard, not a rule: widenings past this mean the argument above is wrong.
        self.cap = 4 * (len(self.candidates[0]) + len(self.candidates[1])) + 4
        self.stalled = False

    def _outside(self, side: int) -> list[SideAction]:
        menu = self.root.rows if side == 0 else self.root.cols
        on = {action.to_choice() for action in menu}
        return [
            action for action in self.candidates[side]
            if action.to_choice() not in on and action.to_choice() not in self.dropped[side]
        ]

    def _support(self, side: int) -> list[int]:
        eq = self.root.equilibrium
        strategy = eq.row_strategy if side == 0 else eq.col_strategy
        return [int(k) for k in np.flatnonzero(strategy > 1e-9)]

    def step(
        self,
        reg: Regulation,
        evaluate: LeafEvaluator,
        budget: Budget,
        meter: _Meter,
        unmodelled: set[str],
    ) -> bool:
        """One oracle step: probe, then widen. Returns whether an action joined.

        With a Q (``q<k>``) the probe is the Q's short list first, and every outside
        action only when none of the short list gains (IKA-322's ``Bqk3f``) -- once
        per menu: after a full probe has found nothing, the steps that follow (the
        deepening's, which move the prices) ask the short list alone until an
        action joins.
        """
        if self.q_probe is None:
            self.probe(reg, evaluate, budget, meter, unmodelled)
            return self.widen(reg, evaluate, budget, meter, unmodelled)
        short = self._shortlist(reg, meter)
        self.probe(reg, evaluate, budget, meter, unmodelled, only=short)
        if self.widen(reg, evaluate, budget, meter, unmodelled, only=short):
            self.proved = False
            return True
        if self.proved or all(
            len(short[side]) == len(self._outside(side)) for side in (0, 1)
        ):
            return False
        self.fallbacks += 1
        self.probe(reg, evaluate, budget, meter, unmodelled)
        joined = self.widen(reg, evaluate, budget, meter, unmodelled)
        self.proved = not joined
        return joined

    def _shortlist(
        self, reg: Regulation, meter: _Meter
    ) -> tuple[list[SideAction], list[SideAction]]:
        """Each side's `q_probe` outside actions the Q ranks best against the root's
        current equilibrium: rows by their Q payoff against the column strategy, columns
        by minus the row strategy's Q payoff against them (ties: the candidates' order)."""
        from . import qrank

        root = self.root
        if self.q is None:
            self.q = np.asarray(
                qrank.installed().matrix(reg, root.pos, self.candidates), dtype=np.float64
            )
            meter.asked_q(1)
            for side in (0, 1):
                self.q_at[side].update(
                    (action.to_choice(), n) for n, action in enumerate(self.candidates[side])
                )
        eq = root.equilibrium
        rows = [self.q_at[0][a.to_choice()] for a in root.rows]
        cols = [self.q_at[1][b.to_choice()] for b in root.cols]
        out: tuple[list[SideAction], list[SideAction]] = ([], [])
        for side in (0, 1):
            outside = self._outside(side)
            if side == 0:
                at = [self.q_at[0][a.to_choice()] for a in outside]
                score = self.q[np.ix_(at, cols)] @ np.asarray(eq.col_strategy)
            else:
                at = [self.q_at[1][b.to_choice()] for b in outside]
                score = -(np.asarray(eq.row_strategy) @ self.q[np.ix_(rows, at)])
            order = sorted(range(len(outside)), key=lambda n: (-float(score[n]), n))
            keep = set(order[: self.q_probe])
            out[side].extend(action for n, action in enumerate(outside) if n in keep)
        return out

    def probe(
        self,
        reg: Regulation,
        evaluate: LeafEvaluator,
        budget: Budget,
        meter: _Meter,
        unmodelled: set[str],
        only: tuple[list[SideAction], list[SideAction]] | None = None,
    ) -> None:
        """Fill, in one call, every outside action (or those of `only`) against the other
        side's support that is not known yet."""
        root = self.root
        out_rows, out_cols = (
            (self._outside(0), self._outside(1)) if only is None else only
        )
        sup_rows, sup_cols = self._support(0), self._support(1)
        rows: list[SideAction] = []
        cols: list[SideAction] = []
        at_row: dict[str, int] = {}
        at_col: dict[str, int] = {}
        wanted: list[tuple[int, int]] = []

        def row_index(action: SideAction) -> int:
            key = action.to_choice()
            if key not in at_row:
                at_row[key] = len(rows)
                rows.append(action)
            return at_row[key]

        def col_index(action: SideAction) -> int:
            key = action.to_choice()
            if key not in at_col:
                at_col[key] = len(cols)
                cols.append(action)
            return at_col[key]

        for action in out_rows:
            for j in sup_cols:
                if (action.to_choice(), root.cols[j].to_choice()) not in self.known:
                    wanted.append((row_index(action), col_index(root.cols[j])))
        for action in out_cols:
            for i in sup_rows:
                if (root.rows[i].to_choice(), action.to_choice()) not in self.known:
                    wanted.append((row_index(root.rows[i]), col_index(action)))
        if not wanted:
            return
        matrices, notes, _exact = port.batched_payoffs(
            reg, root.pos, rows, cols, [evaluate], budget=budget, cells=wanted
        )
        unmodelled.update(notes)
        for i, j in wanted:
            self.known[(rows[i].to_choice(), cols[j].to_choice())] = float(matrices[0][i, j])
        meter.filled(len(wanted), probe=True)
        self.probes += 1
        self.probed += len(wanted)

    def widen(
        self,
        reg: Regulation,
        evaluate: LeafEvaluator,
        budget: Budget,
        meter: _Meter,
        unmodelled: set[str],
        only: tuple[list[SideAction], list[SideAction]] | None = None,
    ) -> bool:
        """Add the outside action (of `only`, when given) with the largest best-response
        gain, if one gains.

        Rows before columns, first in the candidates' order on a tie. The rest of its row
        or column is filled at depth 1 (charged) and the root re-solved. Returns whether
        an action joined.
        """
        if self.stalled:
            return False
        root = self.root
        eq = root.equilibrium
        value = float(eq.value)
        best: tuple[float, int, SideAction] | None = None
        asked = None if only is None else tuple(
            {action.to_choice() for action in only[side]} for side in (0, 1)
        )
        for side in (0, 1):
            support = self._support(1 - side)
            strategy = eq.col_strategy if side == 0 else eq.row_strategy
            for action in self._outside(side):
                key = action.to_choice()
                if asked is not None and key not in asked[side]:
                    continue
                if side == 0:
                    got = sum(
                        float(strategy[j]) * self.known[(key, root.cols[j].to_choice())]
                        for j in support
                    )
                    gain = got - value
                else:
                    got = sum(
                        float(strategy[i]) * self.known[(root.rows[i].to_choice(), key)]
                        for i in support
                    )
                    gain = value - got
                if gain > ORACLE_TOLERANCE and (best is None or gain > best[0]):
                    best = (gain, side, action)
        if best is None:
            return False
        _gain, side, action = best
        key = action.to_choice()
        others = root.cols if side == 0 else root.rows
        pairs = [
            (key, other.to_choice()) if side == 0 else (other.to_choice(), key)
            for other in others
        ]
        missing = [k for k, pair in enumerate(pairs) if pair not in self.known]
        if missing:
            ours, theirs = (
                ([action], [others[k] for k in missing])
                if side == 0
                else ([others[k] for k in missing], [action])
            )
            filled, notes = port.batched_payoff(
                reg, root.pos, ours, theirs, evaluate, budget=budget
            )
            unmodelled.update(notes)
            flat = filled.reshape(-1)
            for n, k in enumerate(missing):
                self.known[pairs[k]] = float(flat[n])
            meter.filled(len(missing))
        line = np.array([self.known[pair] for pair in pairs], dtype=np.float64)
        grown = (
            np.vstack([root.payoff, line[None, :]])
            if side == 0
            else np.hstack([root.payoff, line[:, None]])
        )
        try:
            solved = solve(grown)
        except EquilibriumError:
            self.dropped[side].add(key)
            return False
        root.payoff = grown
        (root.rows if side == 0 else root.cols).append(action)
        root.equilibrium = solved
        root.signal = None
        self.widened += 1
        if self.swap:
            self._swap_out(side, key)
        if self.widened >= self.cap:
            self.stalled = True
        return True

    def _swap_out(self, side: int, added: str) -> None:
        """Push out the action of `side` that carries no weight and loses most (the later
        on a tie), other than the one that just joined, any with a refined or refused
        cell, and any that has left once already. Nothing leaves when there is none.

        A weightless action's removal leaves the equilibrium an equilibrium of the smaller
        game -- the other side's replies to it were never read -- so the root is re-solved
        only to keep its arrays in step (the value stays).
        """
        root = self.root
        eq = root.equilibrium
        menu = root.rows if side == 0 else root.cols
        weight = eq.row_strategy if side == 0 else eq.col_strategy
        loss = eq.row_ev_loss if side == 0 else eq.col_ev_loss
        busy = {cell[side] for cell in (*root.children, *root.refused)}
        free = [
            k for k in range(len(menu))
            if weight[k] <= WEIGHTLESS
            and k not in busy
            and menu[k].to_choice() != added
            and menu[k].to_choice() not in self.left
        ]
        if not free:
            return
        out = max(free, key=lambda k: (float(loss[k]), k))
        smaller = np.delete(root.payoff, out, axis=side)
        try:
            solved = solve(smaller)
        except EquilibriumError:
            return
        self.left.add(menu.pop(out).to_choice())
        root.payoff = smaller
        root.equilibrium = solved
        root.signal = None

        def moved(cell: tuple[int, int]) -> tuple[int, int]:
            if cell[side] < out:
                return cell
            return (cell[0] - 1, cell[1]) if side == 0 else (cell[0], cell[1] - 1)

        root.children = {moved(cell): kept for cell, kept in root.children.items()}
        root.refused = {moved(cell) for cell in root.refused}
        for cell, kept in root.children.items():
            for _weight, child in kept:
                if isinstance(child, _Node):
                    child.parent = (root, cell)
        self.swapped += 1


#: Whether a Bayesian root's cell priority is ``bern/gap`` times its completion's weight
#: (IKA-294). Without the weight a cell of a completion the belief barely holds would be
#: deepened as early as one of the likeliest, though it moves the answer w_k as much.
BELIEF_WEIGHTED = True


class _BeliefRoot:
    """The root of a side that cannot see the other side's bench (IKA-294).

    `belief_solve`'s game for `side`: one row set (this side's actions -- it does not
    know which completion it faces), a matrix per completion `k` of the other bench,
    each in this side's own orientation (side 1's is side 0's negated transpose), and
    the Bayesian equilibrium over them. Its cells are ``(k, i, j)``, and refining one is
    `_expand` in completion `k`'s position: the turn resolved there and the next turn's
    game solved as if that bench were known -- IKA-111's determinization. The children
    are ordinary open nodes, in side 0's orientation as every node below the root is.
    """

    level = 0
    parent = None
    rect = None

    def __init__(
        self,
        side: int,
        own: Sequence[SideAction],
        other: Sequence[SideAction],
        items: Sequence[Any],
        weights: np.ndarray,
        prices: Sequence[np.ndarray],
        equilibrium: Any,  # noqa: ANN401 - BayesianEquilibrium
    ) -> None:
        self.side = side
        self.own = list(own)
        self.other = list(other)
        self.items = list(items)
        w = np.asarray(weights, dtype=np.float64)
        self.w = w / w.sum()
        self.prices = [np.array(p, dtype=np.float64, copy=True) for p in prices]
        self.equilibrium = equilibrium
        self.children: dict[tuple[int, int, int], list[tuple[float, _Node | float]]] = {}
        self.refused: set[tuple[int, int, int]] = set()
        self.signal: np.ndarray | None = None

    @property
    def value(self) -> float:
        return float(self.equilibrium.value)

    def pair(self, i: int, j: int) -> list[SideAction]:
        """Row `i` and column `j` as the turn's two actions, side 0's first."""
        return [self.own[i], self.other[j]] if self.side == 0 else [self.other[j], self.own[i]]

    def turn_of(self, cell: tuple[int, int, int]) -> tuple[Position, list[SideAction]]:
        k, i, j = cell
        return self.items[k].position, self.pair(i, j)

    def scores(self) -> np.ndarray:
        """``bern/gap`` of every cell (k, i, j) -- its side-0 value's d(1 - d) over the
        row's regret plus the column's in completion k -- times w_k (`BELIEF_WEIGHTED`)."""
        if self.signal is None:
            eq = self.equilibrium
            out = np.empty((len(self.prices), len(self.own), len(self.other)))
            for k, prices in enumerate(self.prices):
                won = prices if self.side == 0 else -prices
                gap = eq.row_ev_loss[:, None] + eq.col_ev_loss[k][None, :]
                out[k] = won * (1.0 - won) / (gap + GAP_FLOOR)
                if BELIEF_WEIGHTED:
                    out[k] *= self.w[k]
            self.signal = out
        return self.signal

    def solve(self) -> bool:
        """Re-solve the Bayesian game at today's prices; keep the last answer if it fails."""
        from .equilibrium import solve_bayesian

        try:
            self.equilibrium = solve_bayesian(self.prices, self.w)
        except EquilibriumError:
            return False
        finally:
            self.signal = None
        return True

    def write(self, cell: tuple[int, int, int], won: float) -> None:
        """A refined cell's side-0 value into its completion's matrix, then re-solve."""
        k, i, j = cell
        self.prices[k][i, j] = won if self.side == 0 else -won
        self.solve()


@dataclass(frozen=True, slots=True)
class BeliefDeepening:
    """`deepen_belief`'s answer: the side's Bayesian root when the budget ran out."""

    equilibrium: Any  # BayesianEquilibrium over `prices`
    prices: list[np.ndarray]
    #: This side's actions (the strategy indexes them) and the other side's, grown and
    #: swapped by the oracle.
    own: list[SideAction]
    other: list[SideAction]
    report: Deepened


class _BeliefOracle:
    """The root's double oracle on a Bayesian root (IKA-294): `_Oracle`, per completion.

    A row (this side's action) gains what it gets against each completion's reply,
    weighted, less the root's value: sum_k w_k (M_k y_k)_a - v. A column (the other
    side's) gains what the completions that would take it save by it: sum_k w_k
    max(0, x M_k y_k - (x M_k)_b) -- the other side knows its bench, so each completion
    answers on its own, and a column only one of them wants still joins. With one
    completion both are `_Oracle`'s. The cells are asked in `belief_payoffs`, so a cell
    that reaches no hidden slot is resolved once for every completion.
    """

    def __init__(
        self,
        root: _BeliefRoot,
        outside: tuple[Sequence[SideAction], Sequence[SideAction]],
        fill: Callable[[list[SideAction], list[SideAction]], list[np.ndarray]],
        *,
        swap: bool = False,
        q_probe: int | None = None,
        ask_q: Callable[[list[SideAction], list[SideAction]], list[np.ndarray]] | None = None,
    ) -> None:
        self.root = root
        self.fill = fill
        self.swap = swap
        self.q_probe = q_probe
        #: (own candidates, other candidates) -> the Q per completion, in this side's
        #: orientation (IKA-322). Asked once, at the first step.
        self.ask_q = ask_q
        self.q: list[np.ndarray] | None = None
        self.q_at: tuple[dict[str, int], dict[str, int]] = ({}, {})
        self.fallbacks = 0
        #: Whether a full probe found nothing and no action has joined since: the
        #: short list alone is asked until one does (IKA-322, once per menu).
        self.proved = False
        self.candidates: tuple[list[SideAction], list[SideAction]] = ([], [])
        for role, menu in ((0, root.own), (1, root.other)):
            seen: set[str] = set()
            for action in (*outside[role], *menu):
                if action.to_choice() not in seen:
                    seen.add(action.to_choice())
                    self.candidates[role].append(action)
        #: (own choice, other choice) -> the cell's price in every completion.
        self.known: dict[tuple[str, str], np.ndarray] = {}
        self.dropped: tuple[set[str], set[str]] = (set(), set())
        self.probes = 0
        self.probed = 0
        self.widened = 0
        self.swapped = 0
        self.left: set[str] = set()
        self.cap = 4 * (len(self.candidates[0]) + len(self.candidates[1])) + 4
        self.stalled = False

    def _menu(self, role: int) -> list[SideAction]:
        return self.root.own if role == 0 else self.root.other

    def _outside(self, role: int) -> list[SideAction]:
        on = {action.to_choice() for action in self._menu(role)}
        return [
            action for action in self.candidates[role]
            if action.to_choice() not in on and action.to_choice() not in self.dropped[role]
        ]

    def _rows(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self.root.equilibrium.row_strategy > 1e-9)]

    def _cols(self) -> list[list[int]]:
        return [
            [int(j) for j in np.flatnonzero(y > 1e-9)]
            for y in self.root.equilibrium.col_strategies
        ]

    def _ask(
        self, rows: list[SideAction], cols: list[SideAction], meter: _Meter,
        *, probe: bool = False,
    ) -> None:
        """Fill the rectangle `rows` x `cols` in every completion; remember each cell."""
        if not rows or not cols:
            return
        prices = self.fill(rows, cols)
        for r, a in enumerate(rows):
            for c, b in enumerate(cols):
                self.known[(a.to_choice(), b.to_choice())] = np.array(
                    [float(p[r, c]) for p in prices], dtype=np.float64
                )
        meter.filled(len(rows) * len(cols) * len(prices), probe=probe)
        self.probed += len(rows) * len(cols) * len(prices)

    def step(self, meter: _Meter) -> bool:
        """`_Oracle.step` on the Bayesian root: the Q's short list, every outside action
        only when none of it gains, once per menu (`_Oracle.step`)."""
        if self.q_probe is None:
            self.probe(meter)
            return self.widen(meter)
        short = self._shortlist(meter)
        self.probe(meter, only=short)
        if self.widen(meter, only=short):
            self.proved = False
            return True
        if self.proved or all(
            len(short[role]) == len(self._outside(role)) for role in (0, 1)
        ):
            return False
        self.fallbacks += 1
        self.probe(meter)
        joined = self.widen(meter)
        self.proved = not joined
        return joined

    def _shortlist(self, meter: _Meter) -> tuple[list[SideAction], list[SideAction]]:
        """Each role's `q_probe` outside actions the Q ranks best against the Bayesian
        equilibrium, read as `widen` reads the real cells: a row by its weighted Q
        payoff against each completion's reply, a column by the weighted ``max 0`` of
        what it saves each completion against the row strategy (ties: candidates' order)."""
        root = self.root
        if self.q is None:
            if self.ask_q is None:
                raise ValueError("a Q's probe (q<k>) on a Bayesian root needs ask_q")
            self.q = [np.asarray(q, dtype=np.float64) for q in self.ask_q(*self.candidates)]
            meter.asked_q(len(self.q))
            for role in (0, 1):
                self.q_at[role].update(
                    (action.to_choice(), n) for n, action in enumerate(self.candidates[role])
                )
        eq = root.equilibrium
        x = np.asarray(eq.row_strategy)
        rows = [self.q_at[0][a.to_choice()] for a in root.own]
        cols = [self.q_at[1][b.to_choice()] for b in root.other]
        out: tuple[list[SideAction], list[SideAction]] = ([], [])
        for role in (0, 1):
            outside = self._outside(role)
            at = [self.q_at[role][a.to_choice()] for a in outside]
            score = np.zeros(len(outside))
            for k, q in enumerate(self.q):
                y = np.asarray(eq.col_strategies[k])
                if role == 0:
                    score += float(root.w[k]) * (q[np.ix_(at, cols)] @ y)
                else:
                    earned = float(x @ q[np.ix_(rows, cols)] @ y)
                    score += float(root.w[k]) * np.maximum(
                        0.0, earned - x @ q[np.ix_(rows, at)]
                    )
            order = sorted(range(len(outside)), key=lambda n: (-float(score[n]), n))
            keep = set(order[: self.q_probe])
            out[role].extend(action for n, action in enumerate(outside) if n in keep)
        return out

    def probe(
        self, meter: _Meter, only: tuple[list[SideAction], list[SideAction]] | None = None
    ) -> None:
        """Ask every outside action (or those of `only`) against the other side's
        support, where not known.

        Two rectangles: the outside rows against the union of the completions' column
        supports, and the row support against the outside columns. A rectangle is filled
        whole, so a row is asked against a column of the union a completion does not use.
        """
        root = self.root
        union = sorted({j for cols in self._cols() for j in cols})
        support = self._rows()
        out_rows, out_cols = (
            (self._outside(0), self._outside(1)) if only is None else only
        )
        rows = [
            a for a in out_rows
            if any((a.to_choice(), root.other[j].to_choice()) not in self.known for j in union)
        ]
        cols_needed = [
            root.other[j] for j in union
            if any((a.to_choice(), root.other[j].to_choice()) not in self.known for a in rows)
        ]
        outside_cols = [
            b for b in out_cols
            if any((root.own[i].to_choice(), b.to_choice()) not in self.known for i in support)
        ]
        rows_needed = [
            root.own[i] for i in support
            if any((root.own[i].to_choice(), b.to_choice()) not in self.known for b in outside_cols)
        ]
        if not (rows and cols_needed) and not (outside_cols and rows_needed):
            return
        self._ask(rows, cols_needed, meter, probe=True)
        self._ask(rows_needed, outside_cols, meter, probe=True)
        self.probes += 1

    def widen(
        self, meter: _Meter, only: tuple[list[SideAction], list[SideAction]] | None = None
    ) -> bool:
        """Add the outside action (of `only`, when given) with the largest gain, if one
        gains (rows first on a tie, then the candidates' order); its whole row or column
        is filled, the root re-solved. Returns whether an action joined."""
        if self.stalled:
            return False
        root = self.root
        eq = root.equilibrium
        value = float(eq.value)
        w = root.w
        x = eq.row_strategy
        ys = eq.col_strategies
        support = self._rows()
        cols = self._cols()
        earned = [float(x @ root.prices[k] @ ys[k]) for k in range(len(root.prices))]
        best: tuple[float, int, SideAction] | None = None
        asked = None if only is None else tuple(
            {action.to_choice() for action in only[role]} for role in (0, 1)
        )
        for role in (0, 1):
            for action in self._outside(role):
                key = action.to_choice()
                if asked is not None and key not in asked[role]:
                    continue
                if role == 0:
                    got = sum(
                        float(w[k]) * float(ys[k][j])
                        * float(self.known[(key, root.other[j].to_choice())][k])
                        for k in range(len(root.prices))
                        for j in cols[k]
                    )
                    gain = got - value
                else:
                    gain = 0.0
                    for k in range(len(root.prices)):
                        cost = sum(
                            float(x[i]) * float(self.known[(root.own[i].to_choice(), key)][k])
                            for i in support
                        )
                        gain += float(w[k]) * max(0.0, earned[k] - cost)
                if gain > ORACLE_TOLERANCE and (best is None or gain > best[0]):
                    best = (gain, role, action)
        if best is None:
            return False
        _gain, role, action = best
        key = action.to_choice()
        others = root.other if role == 0 else root.own
        pairs = [
            (key, other.to_choice()) if role == 0 else (other.to_choice(), key)
            for other in others
        ]
        missing = [others[n] for n, pair in enumerate(pairs) if pair not in self.known]
        if role == 0:
            self._ask([action], missing, meter)
        else:
            self._ask(missing, [action], meter)
        lines = np.array([self.known[pair] for pair in pairs], dtype=np.float64)  # (n, K)
        grown = [
            np.vstack([p, lines[:, k][None, :]]) if role == 0 else np.hstack([p, lines[:, k][:, None]])
            for k, p in enumerate(root.prices)
        ]
        before = (root.prices, root.equilibrium)
        root.prices = grown
        if not root.solve():
            root.prices, root.equilibrium = before
            self.dropped[role].add(key)
            return False
        self._menu(role).append(action)
        self.widened += 1
        if self.swap:
            self._swap_out(role, key)
        if self.widened >= self.cap:
            self.stalled = True
        return True

    def _swap_out(self, role: int, added: str) -> None:
        """`_Oracle._swap_out` on the Bayesian root: a column is weightless when no
        completion plays it, and its loss is the completions' weighted losses."""
        root = self.root
        eq = root.equilibrium
        menu = self._menu(role)
        if role == 0:
            weightless = eq.row_strategy <= WEIGHTLESS
            loss = np.asarray(eq.row_ev_loss, dtype=np.float64)
        else:
            weightless = np.all(
                np.stack([y <= WEIGHTLESS for y in eq.col_strategies]), axis=0
            )
            loss = sum(float(root.w[k]) * eq.col_ev_loss[k] for k in range(len(root.prices)))
        axis = 1 + role  # a cell is (k, i, j)
        busy = {cell[axis] for cell in (*root.children, *root.refused)}
        free = [
            n for n in range(len(menu))
            if weightless[n]
            and n not in busy
            and menu[n].to_choice() != added
            and menu[n].to_choice() not in self.left
        ]
        if not free:
            return
        out = max(free, key=lambda n: (float(loss[n]), n))
        before = (root.prices, root.equilibrium)
        root.prices = [np.delete(p, out, axis=role) for p in root.prices]
        if not root.solve():
            root.prices, root.equilibrium = before
            return
        self.left.add(menu.pop(out).to_choice())

        def moved(cell: tuple[int, int, int]) -> tuple[int, int, int]:
            if cell[axis] < out:
                return cell
            return (cell[0], cell[1] - 1, cell[2]) if role == 0 else (cell[0], cell[1], cell[2] - 1)

        root.children = {moved(cell): kept for cell, kept in root.children.items()}
        root.refused = {moved(cell) for cell in root.refused}
        for cell, kept in root.children.items():
            for _weight, child in kept:
                if isinstance(child, _Node):
                    child.parent = (root, cell)
        self.swapped += 1


def deepen_belief(
    reg: Regulation,
    side: int,
    position: Position,
    own: Sequence[SideAction],
    other: Sequence[SideAction],
    items: Sequence[Any],
    weights: np.ndarray,
    prices: Sequence[np.ndarray],
    equilibrium: Any,  # noqa: ANN401 - BayesianEquilibrium, the depth-1 answer
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    cells: int,
    sub_limit: int,
    sub_branches: int,
    unmodelled: set[str],
    reading: str = "mixed",
    outside: tuple[Sequence[SideAction], Sequence[SideAction]] | None = None,
    swap: bool = False,
    cost: Cost | None = None,
    trace: list | None = None,
    q_probe: int | None = None,
    progress: Progress | None = None,
) -> BeliefDeepening:
    """`deepen_root` for a side whose opponent's bench is hidden (IKA-294, label ``h``).

    `prices[k]` is this side's own game (rows `own`) when the other bench is completion
    `items[k]`, and `equilibrium` its Bayesian answer over `weights` -- `belief_solve`'s
    depth-1 answer. The budget is spent best first over the cells ``(k, i, j)`` and the
    trees under them (`_BeliefRoot`), the root re-solved as the Bayesian game after
    every step; with `outside` (this side's candidates, the other side's) the root's
    double oracle asks them first every step (`_BeliefOracle`), and `swap` / the
    ``breadth`` reading are `deepen_root`'s. A cell spent is one resolved turn in one
    completion: a probed cell counts once per completion. `q_probe` narrows the probe by
    the process's Q, asked once per completion in that completion's position (IKA-322).
    """
    if reading not in ("mixed", "breadth"):
        raise ValueError(f"a Bayesian root is read whole or breadth only, not {reading!r}")
    if outside is None and (swap or reading == "breadth" or q_probe is not None):
        raise ValueError(
            "swapping, breadth only and a Q's probe are the root's double oracle's; no outside"
        )
    from .beliefnode import belief_payoffs

    deepens = reading != "breadth"
    root = _BeliefRoot(side, own, other, items, weights, prices, equilibrium)
    if trace is not None:
        trace.append(root)
    spreads = {1 - side: list(items)}

    def fill(rows: list[SideAction], cols: list[SideAction]) -> list[np.ndarray]:
        """This side's prices of `rows` x `cols` in every completion, in its orientation."""
        ours, theirs = (rows, cols) if side == 0 else (cols, rows)
        node = belief_payoffs(
            reg, position, ours, theirs, evaluate, budget=budget, spreads=spreads
        )
        unmodelled.update(node.unmodelled)
        return [m if side == 0 else -m.T for m in node.matrices[side]]

    def ask_q(own_c: list[SideAction], other_c: list[SideAction]) -> list[np.ndarray]:
        """The Q of `own_c` x `other_c` in every completion, in this side's orientation."""
        from . import qrank

        model = qrank.installed()
        pools = (own_c, other_c) if side == 0 else (other_c, own_c)
        out = []
        for item in items:
            q = np.asarray(model.matrix(reg, item.position, pools), dtype=np.float64)
            out.append(q if side == 0 else -q.T)
        return out

    meter = _Meter(cost)
    oracle = (
        None if outside is None
        else _BeliefOracle(root, outside, fill, swap=swap, q_probe=q_probe, ask_q=ask_q)
    )
    expanded = 0
    deepest = 0
    refused = 0
    announce = None if progress is None else _stepper(progress, root, meter, cells, oracle)
    if announce is not None:
        announce("start", expanded, deepest, refused)
    helper = (
        _start_ahead(reg, evaluate, budget=budget, sub_limit=sub_limit, sub_branches=sub_branches)
        if deepens and cells > 0
        else None
    )
    try:
        while meter.spent < cells:
            if oracle is not None:
                with _held(helper):
                    joined = oracle.step(meter)
                if joined:
                    if trace is not None:
                        trace.append((root, None, True))
                    if announce is not None:
                        announce("widen", expanded, deepest, refused)
                    continue
                if not deepens:
                    break
            target = _best(root)  # type: ignore[arg-type]
            if target is None:
                break
            node, cell = target
            if helper is None:
                spent, ok, fills = _expand(
                    reg, node, cell, evaluate, budget=budget, sub_limit=sub_limit,
                    sub_branches=sub_branches, unmodelled=unmodelled,
                )
            else:
                helper.post(root)
                spent, ok, fills = helper.expand(node, cell, unmodelled)
            meter.refined(spent - 1, fills)
            if trace is not None:
                trace.append((node, cell, ok))
            if not ok:
                node.refused.add(cell)
                if isinstance(node, _BeliefRoot):
                    node.signal = None
                refused += 1
                if announce is not None:
                    announce("refused", expanded, deepest, refused, cell, node.level)
                continue
            expanded += 1
            deepest = max(deepest, node.level + 1)
            _propagate(node, cell)
            if announce is not None:
                announce("refine", expanded, deepest, refused, cell, node.level)
    finally:
        if helper is not None:
            helper.close()
    if timing.ON:
        timing.count("deepen.hidden.calls", 1)
        timing.count("deepen.hidden.classes", len(root.prices))
        timing.count("deepen.cells", meter.refines + meter.cells)
        timing.count("deepen.expanded", expanded)
        timing.count("deepen.fills", meter.fills)
        timing.count("deepen.refines", meter.refines)
        if oracle is not None:
            timing.count("deepen.oracle.calls", 1)
            timing.count("deepen.oracle.probes", oracle.probes)
            timing.count("deepen.oracle.probed", oracle.probed)
            timing.count("deepen.oracle.widened", oracle.widened)
            timing.count("deepen.oracle.swapped", oracle.swapped)
            timing.count("deepen.oracle.stalled", int(oracle.stalled))
            if q_probe is not None:
                timing.count("deepen.oracle.q", meter.qs)
                timing.count("deepen.oracle.qfull", oracle.fallbacks)
    uncovered: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    if oracle is not None and oracle.swapped:
        mine = _lost_options(reg, own, root.own)
        theirs = _lost_options(reg, other, root.other)
        uncovered = (mine, theirs) if side == 0 else (theirs, mine)
        if timing.ON:
            timing.count("deepen.oracle.uncovered", len(mine) + len(theirs))
    report = Deepened(
        budget=cells, cells=meter.refines + meter.cells, expanded=expanded,
        depth=1 + deepest, refused=refused, oracle=oracle is not None,
        probed=0 if oracle is None else oracle.probed,
        widened=0 if oracle is None else oracle.widened,
        fills=meter.fills, swap=swap,
        swapped=0 if oracle is None else oracle.swapped, uncovered=uncovered,
        classes=len(root.prices), q_probe=q_probe is not None, q=meter.qs,
        qfull=0 if oracle is None else oracle.fallbacks,
    )
    if announce is not None:
        announce("done", expanded, deepest, refused)
    return BeliefDeepening(
        equilibrium=root.equilibrium, prices=root.prices, own=root.own, other=root.other,
        report=report,
    )


def _signal(node: _Node) -> np.ndarray:
    """``bern/gap`` for every cell of `node`, refined or not, at its current value."""
    if node.signal is None:
        eq = node.equilibrium
        values = node.payoff
        gap = eq.row_ev_loss[:, None] + eq.col_ev_loss[None, :]
        node.signal = values * (1.0 - values) / (gap + GAP_FLOOR)
    return node.signal


def _scores(node: _Node, inherited: float | None) -> np.ndarray:
    """Every cell's priority in `node`: its own signal at the root, inherited below it.

    Below the root the signals are scaled so the node's largest -- over every cell that
    is not refused, refined or not -- is worth `inherited`, which is the parent cell's
    priority times this branch's weight.
    """
    signal = _signal(node)
    if inherited is None:
        return signal
    live = np.ones(signal.shape, dtype=bool)
    for cell in node.refused:
        live[cell] = False
    top = float(signal[live].max()) if live.any() else 0.0
    if top <= 0.0:
        return np.zeros_like(signal)
    return inherited * signal / top


def _best(root: _Node) -> tuple[_Node, tuple[int, int]] | None:
    """The unrefined cell with the highest positive priority under `root`, or None.

    Every priority under a node is at most what that node inherited, so the tree is
    searched best first from the root and a subtree is not opened once the best cell in
    hand is worth at least its bound. Nodes are opened in order of that bound, ties by
    the order they were met (a node, then its refined cells in row-major order and each
    one's branches in the order kept), and within a node a tie goes to the first cell --
    so the choice is a function of the tree alone.
    """
    best: tuple[float, _Node, tuple[int, int]] | None = None
    met = 0
    heap: list[tuple[float, int, _Node, float | None]] = [(-np.inf, met, root, None)]
    while heap:
        bound, _order, node, inherited = heapq.heappop(heap)
        if best is not None and -bound <= best[0]:
            break
        if isinstance(node, _BeliefRoot):
            # The Bayesian root (IKA-294): its cells are (completion, row, column), the
            # first in that order on a tie. It is only ever the root, so nothing is
            # in hand yet and nothing is inherited.
            scores = node.scores()
            candidates = scores.copy()
            for cell in (*node.children, *node.refused):
                candidates[cell] = -np.inf
            if candidates.size:
                flat = int(np.argmax(candidates))
                score = float(candidates.flat[flat])
                if score > 0.0:
                    best = (
                        score, node,
                        tuple(int(v) for v in np.unravel_index(flat, candidates.shape)),
                    )
            for cell in sorted(node.children):
                for weight, child in node.children[cell]:
                    if isinstance(child, _Node):
                        passed = float(scores[cell]) * weight
                        if passed > 0.0:
                            met += 1
                            heapq.heappush(heap, (-passed, met, child, passed))
            continue
        scores = _scores(node, inherited)
        candidates = scores.copy()
        if node.rect is not None:
            inside = np.zeros(candidates.shape, dtype=bool)
            inside[np.ix_(*node.rect)] = True
            candidates[~inside] = -np.inf
        for cell in node.children:
            candidates[cell] = -np.inf
        for cell in node.refused:
            candidates[cell] = -np.inf
        if node.level < MAX_LEVELS and candidates.size:
            flat = int(np.argmax(candidates))
            score = float(candidates.flat[flat])
            if score > 0.0 and (best is None or score > best[0]):
                best = (score, node, divmod(flat, candidates.shape[1]))
        for cell in sorted(node.children):
            for weight, child in node.children[cell]:
                if isinstance(child, _Node):
                    passed = float(scores[cell]) * weight
                    if passed > 0.0:
                        met += 1
                        heapq.heappush(heap, (-passed, met, child, passed))
    if best is None:
        return None
    return best[1], best[2]


def _expand(
    reg: Regulation,
    node: _Node,
    cell: tuple[int, int],
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    sub_limit: int,
    sub_branches: int,
    unmodelled: set[str],
) -> tuple[int, bool, int]:
    """Refine one cell of `node` as `search._refined_value` does, keeping the children.

    Returns the cells spent, whether the cell took a deeper value, and the fills made.
    """
    if isinstance(node, _BeliefRoot):
        # The cell's completion, as if its bench were known (IKA-111's determinization).
        at, pair = node.turn_of(cell)
    else:
        i, j = cell
        at, pair = node.pos, [node.rows[i], node.cols[j]]
    result = port.turn(reg, at, pair, budget, full=True)
    unmodelled.update(result.unmodelled)
    spent = 1
    fills = 0
    if result.suspended or not result.outcomes:
        return spent, False, fills
    branches = sorted(result.outcomes, key=lambda b: -b.probability)[:sub_branches]
    if len(branches) < len(result.outcomes):
        unmodelled.add(
            f"depth-2 kept the {sub_branches} likeliest branches of a refined cell"
        )
    weights = np.array([b.probability for b in branches], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return spent, False, fills
    weights /= total

    # Finished branches are scored by the leaf, as `_subgame_value` scores them, in one
    # call for the cell -- a row-wise leaf gives the same numbers either way.
    ended = [branch.position for branch in branches if branch.position.ended]
    finished = iter(np.asarray(evaluate(ended), dtype=np.float64) if ended else ())
    kept: list[tuple[float, _Node | float]] = []
    for weight, branch in zip(weights, branches, strict=True):
        child_pos = branch.position
        if child_pos.ended:
            kept.append((float(weight), float(next(finished))))
            continue
        row = narrow(reg, child_pos, 0, limit=sub_limit).actions
        col = narrow(reg, child_pos, 1, limit=sub_limit).actions
        if not row or not col:
            return spent, False, fills
        payoff, notes = batched_payoff(reg, child_pos, row, col, evaluate, budget=budget)
        fills += 1
        spent += len(row) * len(col)
        unmodelled.update(notes)
        try:
            equilibrium = solve(payoff)
        except EquilibriumError:
            return spent, False, fills
        kept.append((
            float(weight),
            _Node(
                pos=child_pos, rows=list(row), cols=list(col),
                payoff=np.asarray(payoff, dtype=np.float64), equilibrium=equilibrium,
                level=node.level + 1, parent=(node, cell), weight=float(weight),
            ),
        ))
    node.children[cell] = kept
    return spent, True, fills


#: Cells expanded ahead of the loop (`set_ahead`): 0 is off, the serial `_expand` as before.
AHEAD_ENV = "POKEURAOU_DEEPEN_AHEAD"
_AHEAD = [int(os.environ.get(AHEAD_ENV, "0") or 0)]
#: Helper threads, and the cell threads of each one's port (None: the module's count).
_HELPERS: list[Any] = [1, None]


def set_ahead(cells: int, helpers: int = 1, port_threads: int | None = None) -> None:
    """Expand up to `cells` of the best cells ahead of the loop, on `helpers` threads with
    a port process each of `port_threads` cell threads (the module's docstring). 0 cells
    turns it off. Changes how long a step takes, never what it does; generation and the
    board leave it at 0."""
    if cells < 0 or helpers < 1:
        raise ValueError(f"cells ahead must be >= 0 and helpers >= 1, not {cells}, {helpers}")
    _AHEAD[0] = int(cells)
    _HELPERS[0] = int(helpers)
    _HELPERS[1] = port_threads


def ahead() -> int:
    """The cells expanded ahead of the loop (`set_ahead`), 0 when off."""
    return _AHEAD[0]


@dataclass(slots=True)
class _Expansion:
    """`_expand`'s work on one cell, done ahead and not yet written into the tree."""

    #: The turn's notes (and the note on the branches kept).
    notes: set[str]
    #: The turn paused, or had no branch of any weight: `_expand` returns (1, False, 0).
    early: bool = False
    #: The kept branches in order, each (weight, kind, data): ``ended`` with the leaf's
    #: value, ``empty`` (a side had no action: the serial step stops there), ``error``
    #: (the serial step expands the cell itself) or ``node`` with (position, rows,
    #: columns, payoff, notes of the fill, the child's equilibrium or None: its LP failed).
    branches: list[tuple[float, str, Any]] = field(default_factory=list)
    #: Something the batch met that the serial step has to meet in its own place.
    error: bool = False


def _expand_many(
    reg: Regulation,
    asks: Sequence[tuple[Position, list[SideAction]]],
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    sub_limit: int,
    sub_branches: int,
    leaf_lock: Any = None,  # noqa: ANN401 - a lock, or None
) -> list[_Expansion]:
    """`_expand` of many cells at once, without the tree and without the child LPs: one
    crossing for the turns, one for every branch's menus, one for the child matrices and
    one call into the leaf for their blocks. Each cell's answer is the one `_expand` would
    compute alone (the turns and menus are the port's answers to the same requests; each
    block is scored in a call of its own size, `score_segments`). `leaf_lock` is held
    around every call into the leaf, which is not safe on two threads at once (an
    ensemble's stacked call swaps its parameters into one shared module)."""
    from .narrow import narrow_many

    leaf = _NoLock() if leaf_lock is None else leaf_lock

    turns = port.turns(reg, asks, budget, full=True)
    out: list[_Expansion] = []
    menus: list[tuple[Position, int]] = []
    # (expansion, branch index) of each non-ended branch, in the order of `menus` / 2.
    open_: list[tuple[_Expansion, int]] = []
    for result in turns:
        if isinstance(result, Exception):
            out.append(_Expansion(set(), error=True))
            continue
        exp = _Expansion(set(result.unmodelled))
        out.append(exp)
        if result.suspended or not result.outcomes:
            exp.early = True
            continue
        branches = sorted(result.outcomes, key=lambda b: -b.probability)[:sub_branches]
        if len(branches) < len(result.outcomes):
            exp.notes.add(
                f"depth-2 kept the {sub_branches} likeliest branches of a refined cell"
            )
        weights = np.array([b.probability for b in branches], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            exp.early = True
            continue
        weights /= total
        ended = [branch.position for branch in branches if branch.position.ended]
        try:
            with leaf:
                finished = iter(np.asarray(evaluate(ended), dtype=np.float64) if ended else ())
        except Exception:  # noqa: BLE001 - met again by the serial step, in its place
            exp.error = True
            continue
        for weight, branch in zip(weights, branches, strict=True):
            if branch.position.ended:
                exp.branches.append((float(weight), "ended", float(next(finished))))
                continue
            exp.branches.append((float(weight), "open", branch.position))
            open_.append((exp, len(exp.branches) - 1))
            menus.append((branch.position, 0))
            menus.append((branch.position, 1))
    narrowed = narrow_many(reg, menus, limit=sub_limit) if menus else []
    fills: list[tuple[_Expansion, int, Position, list[SideAction], list[SideAction]]] = []
    for n, (exp, b) in enumerate(open_):
        weight, _kind, child_pos = exp.branches[b]
        got_row, got_col = narrowed[2 * n], narrowed[2 * n + 1]
        if isinstance(got_row, Exception) or isinstance(got_col, Exception):
            exp.branches[b] = (weight, "error", None)
            continue
        row, col = got_row.actions, got_col.actions
        if not row or not col:
            exp.branches[b] = (weight, "empty", None)
            continue
        fills.append((exp, b, child_pos, list(row), list(col)))
    if not fills:
        return out
    pending = port.pending_payoffs(
        reg, [(pos, row, col) for _e, _b, pos, row, col in fills], evaluate, budget=budget
    )
    if pending is None:
        # A ported (or hand-written) objective: no forward pass to share.
        for exp, b, pos, row, col in fills:
            weight = exp.branches[b][0]
            try:
                with leaf:
                    payoff, notes = batched_payoff(reg, pos, row, col, evaluate, budget=budget)
            except Exception:  # noqa: BLE001 - met again by the serial step
                exp.branches[b] = (weight, "error", None)
                continue
            exp.branches[b] = (weight, "node", (pos, row, col, payoff, set(notes)))
        _solve_children(out)
        return out
    scored = [p for p in pending if not isinstance(p, Exception)]
    try:
        with leaf:
            values = port.score_segments(evaluate, [p.encoded for p in scored]) if scored else []
    except Exception:  # noqa: BLE001 - met again by the serial step
        for exp, b, *_rest in fills:
            exp.branches[b] = (exp.branches[b][0], "error", None)
        return out
    for p, v in zip(scored, values, strict=True):
        p.scored(v)
    for (exp, b, pos, row, col), p in zip(fills, pending, strict=True):
        weight = exp.branches[b][0]
        if isinstance(p, Exception):
            exp.branches[b] = (weight, "error", None)
            continue
        exp.branches[b] = (
            weight, "node", (pos, row, col, np.asarray(p.finish(), dtype=np.float64), set(p.unmodelled))
        )
    _solve_children(out)
    return out


def _solve_children(expansions: list[_Expansion]) -> None:
    """Each child matrix's equilibrium, as `_expand` solves it: kept with the matrix, or
    None where the LP failed (the serial step stops there). Any other error is the serial
    step's to meet."""
    for exp in expansions:
        for b, (weight, kind, data) in enumerate(exp.branches):
            if kind != "node":
                continue
            try:
                solved: Equilibrium | None = solve(data[3])
            except EquilibriumError:
                solved = None
            except Exception:  # noqa: BLE001 - met again by the serial step
                exp.branches[b] = (weight, "error", None)
                continue
            exp.branches[b] = (weight, "node", (*data, solved))


def _taken(
    exp: _Expansion, node: _Node, cell: tuple[int, ...], unmodelled: set[str]
) -> tuple[int, bool, int] | None:
    """Write an expansion done ahead into the tree, as `_expand` would have, or None when
    the serial step has to expand the cell itself (the batch met an error before the
    point where `_expand` stops)."""
    if exp.error:
        return None
    if not exp.early:
        for _weight, kind, _data in exp.branches:
            if kind == "error":
                return None
            if kind == "empty":
                break
    unmodelled.update(exp.notes)
    if exp.early:
        return 1, False, 0
    spent = 1
    fills = 0
    kept: list[tuple[float, _Node | float]] = []
    for weight, kind, data in exp.branches:
        if kind == "ended":
            kept.append((weight, float(data)))
            continue
        if kind == "empty":
            return spent, False, fills
        child_pos, row, col, payoff, notes, equilibrium = data
        fills += 1
        spent += len(row) * len(col)
        unmodelled.update(notes)
        if equilibrium is None:  # the LP failed, as `_expand`'s `solve` would have
            return spent, False, fills
        kept.append((
            weight,
            _Node(
                pos=child_pos, rows=list(row), cols=list(col), payoff=payoff,
                equilibrium=equilibrium, level=node.level + 1, parent=(node, cell),
                weight=weight,
            ),
        ))
    node.children[cell] = kept
    return spent, True, fills


def _key(node: Any, cell: tuple[int, ...]) -> tuple[Any, ...]:  # noqa: ANN401
    """A cell by its node and its two actions (an index moves when the oracle swaps)."""
    if isinstance(node, _BeliefRoot):
        k, i, j = cell
        return (node, k, node.own[i].to_choice(), node.other[j].to_choice())
    i, j = cell
    return (node, -1, node.rows[i].to_choice(), node.cols[j].to_choice())


def _turn_of(node: Any, cell: tuple[int, ...]) -> tuple[Position, list[SideAction]]:  # noqa: ANN401
    if isinstance(node, _BeliefRoot):
        return node.turn_of(cell)
    i, j = cell
    return node.pos, [node.rows[i], node.cols[j]]


def _ranked(root: Any, count: int) -> list[tuple[Any, tuple[int, ...]]]:  # noqa: ANN401
    """The `count` unrefined cells of highest positive priority under `root`, best first --
    `_best`'s search, keeping `count` instead of one. Only a guess at the cells the loop
    takes next: the priorities move with every step."""
    found: list[tuple[float, int, Any, tuple[int, ...]]] = []  # min-heap of the best
    order = 0
    met = 0
    heap: list[tuple[float, int, Any, float | None]] = [(-np.inf, met, root, None)]

    def keep(score: float, node: Any, cell: tuple[int, ...]) -> None:  # noqa: ANN401
        nonlocal order
        order += 1
        item = (score, -order, node, cell)
        if len(found) < count:
            heapq.heappush(found, item)
        elif score > found[0][0]:
            heapq.heapreplace(found, item)

    while heap:
        bound, _order, node, inherited = heapq.heappop(heap)
        if len(found) >= count and -bound <= found[0][0]:
            break
        scores = node.scores() if isinstance(node, _BeliefRoot) else _scores(node, inherited)
        candidates = scores.copy()
        if node.rect is not None:
            inside = np.zeros(candidates.shape, dtype=bool)
            inside[np.ix_(*node.rect)] = True
            candidates[~inside] = -np.inf
        for cell in (*node.children, *node.refused):
            candidates[cell] = -np.inf
        if node.level < MAX_LEVELS and candidates.size:
            flat = candidates.reshape(-1)
            take = min(count, flat.size)
            top = np.argpartition(-flat, take - 1)[:take] if take < flat.size else np.arange(flat.size)
            for index in top:
                score = float(flat[index])
                if score > 0.0:
                    keep(score, node, tuple(int(v) for v in np.unravel_index(int(index), candidates.shape)))
        for cell in sorted(node.children):
            for weight, child in node.children[cell]:
                if isinstance(child, _Node):
                    passed = float(scores[cell]) * weight
                    if passed > 0.0:
                        met += 1
                        heapq.heappush(heap, (-passed, met, child, passed))
    return [(node, cell) for _score, _o, node, cell in sorted(found, key=lambda t: (-t[0], -t[1]))]


class _Helpers:
    """The threads that expand cells ahead of the deepening loop (`set_ahead`), started once
    and kept for the process's life.

    Kept, not started per deepening, because a thread that has run HiGHS can hang the
    process when it exits: the second of three helper threads' tests (IKA-32 stage 2) sat
    in the next `Thread.start` for eighteen minutes, the exiting thread holding the GIL.
    The helpers serve one deepening at a time (`_Ahead`, the session): they wait on the
    session's wanted cells, take a batch, expand it and hand the expansions back.

    Each helper expands on a port process of its own (`rustnode.own_node`), or -- where
    positions are held by number (`rustnode.hold_positions`, generation's) -- on the loop's
    port under the session's lock, or by sending the batch to a worker process of its own
    (`start_workers`; these helpers never run HiGHS and are stopped with the workers).
    """

    def __init__(
        self, reg: Regulation, count: int, *, port_threads: int | None, conns: list[Any] | None
    ) -> None:
        from . import rustnode

        self.reg = reg
        self.port_threads = port_threads
        self.conns = conns
        self.shared_port = conns is None and bool(rustnode._HOLD[0])
        self.cv = threading.Condition()
        self.session: _Ahead | None = None
        self.stop = False
        self.threads = [
            threading.Thread(target=self._run, args=(n,), name=f"deepen-ahead-{n}", daemon=True)
            for n in range(max(1, count if conns is None else len(conns)))
        ]
        for thread in self.threads:
            thread.start()

    def _run(self, index: int) -> None:
        from . import rustnode

        if self.conns is not None:
            self._serve(conn=self.conns[index])
            return
        if self.shared_port:
            self._serve()
            return
        try:
            with rustnode.own_node(self.reg, threads=self.port_threads):
                self._serve(own=True)
        except Exception:  # noqa: BLE001 - no port of its own: share the loop's
            self._serve()

    def _serve(self, *, own: bool = False, conn: Any = None) -> None:  # noqa: ANN401
        while True:
            with self.cv:
                while not self.stop and (self.session is None or not self.session.want):
                    self.cv.wait()
                if self.stop:
                    return
                session = self.session
                batch = session.want[: session.batch]
                del session.want[: session.batch]
                for key, _pos, _pair in batch:
                    session.busy.add(key)
                session.active += 1
            asks = [(pos, pair) for _key, pos, pair in batch]
            remote = False
            try:
                if conn is not None:
                    # A worker process: the batch goes over, the expansions come back.
                    conn.send((asks, session.budget, session.sub_limit, session.sub_branches))
                    got = conn.recv()
                    remote = True
                elif own:
                    got = _expand_many(
                        session.reg, asks, session.evaluate, budget=session.budget,
                        sub_limit=session.sub_limit, sub_branches=session.sub_branches,
                        leaf_lock=session.lock,
                    )
                else:
                    with session.lock:
                        got = _expand_many(
                            session.reg, asks, session.evaluate, budget=session.budget,
                            sub_limit=session.sub_limit, sub_branches=session.sub_branches,
                        )
            except Exception:  # noqa: BLE001 - each cell is met again the serial way
                got = [_Expansion(set(), error=True) for _ in batch]
            with self.cv:
                for (key, _pos, _pair), exp in zip(batch, got, strict=True):
                    session.busy.discard(key)
                    session.done[key] = exp
                session.batches += 1
                session.remote_batches += int(remote)
                session.expanded += len(batch)
                session.active -= 1
                self.cv.notify_all()

    def close(self) -> None:
        """Stop the helpers (the worker processes' ones: they never ran HiGHS)."""
        with self.cv:
            self.stop = True
            self.cv.notify_all()
        for thread in self.threads:
            thread.join()


#: The kept helpers of this process, by (format, helpers, port threads).
_LOCAL_HELPERS: dict[tuple[Any, ...], _Helpers] = {}


class _Ahead:
    """One deepening's cells expanded ahead of its loop by the helpers (`_Helpers`).

    The loop posts the cells it is likely to take next (`post`) and takes each step's
    expansion (`expand`). The leaf is one object and is called under `lock`, by the
    helpers and by the loop's own uses of it (a cell expanded the serial way, the oracle).
    `close` waits for the batches in hand and lets the helpers go back to waiting.
    """

    def __init__(
        self,
        reg: Regulation,
        evaluate: LeafEvaluator,
        *,
        budget: Budget,
        sub_limit: int,
        sub_branches: int,
        count: int,
        helpers: int = 1,
        port_threads: int | None = None,
    ) -> None:
        self.reg = reg
        self.evaluate = evaluate
        self.budget = budget
        self.sub_limit = sub_limit
        self.sub_branches = sub_branches
        self.count = count
        #: The leaf (and, where the helpers share it, the loop's port).
        self.lock = threading.Lock()
        found = _WORKERS.get(reg.meta.format_id)
        if found and found[2] is not None:
            self.pool = found[2]
        else:
            key = (reg.meta.format_id, max(1, helpers), port_threads)
            if key not in _LOCAL_HELPERS:
                _LOCAL_HELPERS[key] = _Helpers(reg, max(1, helpers), port_threads=port_threads, conns=None)
            self.pool = _LOCAL_HELPERS[key]
        self.batch = max(1, count // len(self.pool.threads))
        self.cv = self.pool.cv
        self.want: list[tuple[tuple[Any, ...], Position, list[SideAction]]] = []
        self.busy: set[tuple[Any, ...]] = set()
        self.done: dict[tuple[Any, ...], _Expansion] = {}
        #: Batches in a helper's hands.
        self.active = 0
        #: Counts: batches, cells expanded ahead, taken from them, expanded the serial way,
        #: batches a worker process expanded.
        self.batches = 0
        self.expanded = 0
        self.hits = 0
        self.misses = 0
        self.remote_batches = 0
        with self.cv:
            if self.pool.session is not None:
                raise RuntimeError("the deepening's helpers already serve another deepening")
            self.pool.session = self

    def post(self, root: Any) -> None:  # noqa: ANN401
        """The cells worth expanding now: the best `count` under `root`."""
        wanted = []
        for node, cell in _ranked(root, self.count):
            key = _key(node, cell)
            if key in self.done or key in self.busy:
                continue
            pos, pair = _turn_of(node, cell)
            wanted.append((key, pos, pair))
        with self.cv:
            self.want = wanted
            if wanted:
                self.cv.notify_all()

    def expand(
        self, node: Any, cell: tuple[int, ...], unmodelled: set[str]  # noqa: ANN401
    ) -> tuple[int, bool, int]:
        """`_expand`'s answer for the cell, from the helpers when they can give it."""
        key = _key(node, cell)
        with self.cv:
            while key not in self.done:
                if key not in self.busy and not any(w[0] == key for w in self.want):
                    pos, pair = _turn_of(node, cell)
                    self.want.insert(0, (key, pos, pair))
                    self.cv.notify_all()
                self.cv.wait()
            exp = self.done.pop(key)
        got = _taken(exp, node, cell, unmodelled)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1
        with self.lock:
            return _expand(
                self.reg, node, cell, self.evaluate, budget=self.budget,
                sub_limit=self.sub_limit, sub_branches=self.sub_branches,
                unmodelled=unmodelled,
            )

    def close(self) -> None:
        with self.cv:
            self.want = []
            while self.active:
                self.cv.wait()
            self.pool.session = None
        if timing.ON:
            timing.count("deepen.ahead.batches", self.batches)
            timing.count("deepen.ahead.expanded", self.expanded)
            timing.count("deepen.ahead.hits", self.hits)
            timing.count("deepen.ahead.misses", self.misses)
        _AHEAD_COUNTS["batches"] += self.batches
        _AHEAD_COUNTS["expanded"] += self.expanded
        _AHEAD_COUNTS["hits"] += self.hits
        _AHEAD_COUNTS["misses"] += self.misses
        _AHEAD_COUNTS["remote"] += self.remote_batches


#: Worker processes that expand cells ahead (`start_workers`), by regulation: a list of
#: (process, connection), and the leaf spec they were started with.
_WORKERS: dict[str, tuple[list[tuple[Any, Any]], tuple[Any, ...], _Helpers | None]] = {}


def _worker_main(conn: Any, format_id: str, factory: Any, args: tuple[Any, ...],  # noqa: ANN401
                 port_threads: int | None) -> None:
    """A worker process: its own regulation, leaf and port; `_expand_many` on each batch."""
    from . import rustnode
    from .damage import register_mega_stones
    from .regulation import load_regulation

    reg = load_regulation(format_id)
    register_mega_stones(reg)
    if port_threads is not None:
        rustnode.set_port_threads(port_threads)
    try:
        leaf = factory(reg, *args)
    except Exception as error:  # noqa: BLE001 - said to the parent, which stops using it
        conn.send(("failed", f"{type(error).__name__}: {error}"))
        return
    conn.send(("ready", None))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        asks, budget, sub_limit, sub_branches = message
        try:
            got = _expand_many(
                reg, asks, leaf, budget=budget, sub_limit=sub_limit, sub_branches=sub_branches
            )
        except Exception:  # noqa: BLE001 - each cell is met again the serial way
            got = [_Expansion(set(), error=True) for _ in asks]
        conn.send(got)


def start_workers(
    reg: Regulation, count: int, factory: Any, args: tuple[Any, ...] = (),  # noqa: ANN401
    *, port_threads: int | None = 1,
) -> int:
    """Start `count` worker processes that expand the deepening's cells ahead (IKA-32
    stage 2), each with the leaf ``factory(reg, *args)`` (a module-level function, so a
    spawned process can import it), its own port of `port_threads` cell threads and its own
    GIL. Kept for the process's life (`stop_workers`); the helpers of every later deepening
    on this regulation send their batches to them. Returns how many are ready. The leaf
    must answer as the loop's own does -- the same model on the same device -- since the
    workers' blocks stand for the loop's."""
    import multiprocessing

    stop_workers(reg)
    if count <= 0:
        return 0
    context = multiprocessing.get_context("spawn")
    started = []
    for _ in range(count):
        mine, theirs = context.Pipe()
        process = context.Process(
            target=_worker_main,
            args=(theirs, reg.meta.format_id, factory, tuple(args), port_threads),
            daemon=True,
        )
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
            print(f"[deepen] a worker did not start: {why}", flush=True)
            process.join(timeout=5)
    helpers = (
        _Helpers(reg, len(ready), port_threads=None, conns=[conn for _p, conn in ready])
        if ready else None
    )
    _WORKERS[reg.meta.format_id] = (ready, (factory, tuple(args)), helpers)
    return len(ready)


def stop_workers(reg: Regulation | None = None) -> None:
    """Stop the worker processes of `reg` (all of them when None)."""
    keys = list(_WORKERS) if reg is None else [reg.meta.format_id]
    for key in keys:
        found = _WORKERS.pop(key, None)
        if found is None:
            continue
        if found[2] is not None:
            found[2].close()
        for process, conn in found[0]:
            with contextlib.suppress(OSError, EOFError):
                conn.send(None)
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
            conn.close()


def workers(reg: Regulation) -> int:
    """How many worker processes expand ahead for `reg`."""
    found = _WORKERS.get(reg.meta.format_id)
    return 0 if found is None else len(found[0])


atexit.register(stop_workers)


#: Totals over the process of every `_Ahead` (the positive control of IKA-32 stage 2).
_AHEAD_COUNTS = {"batches": 0, "expanded": 0, "hits": 0, "misses": 0, "remote": 0}


def ahead_counts() -> dict[str, int]:
    """Batches, cells expanded ahead, taken from the helper, and expanded the serial way
    (a batch error), over the process so far."""
    return dict(_AHEAD_COUNTS)


def _start_ahead(
    reg: Regulation, evaluate: LeafEvaluator, *, budget: Budget, sub_limit: int,
    sub_branches: int,
) -> _Ahead | None:
    count = _AHEAD[0]
    if count <= 0:
        return None
    return _Ahead(
        reg, evaluate, budget=budget, sub_limit=sub_limit, sub_branches=sub_branches,
        count=count, helpers=_HELPERS[0], port_threads=_HELPERS[1],
    )


class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_exc: object) -> bool:
        return False


def _held(helper: _Ahead | None) -> Any:  # noqa: ANN401
    """The port and the leaf for the loop's own use: the helper's lock, or nothing."""
    return _NoLock() if helper is None else helper.lock


def _cell_value(branches: list[tuple[float, _Node | float]]) -> float:
    """A refined cell's value: its branches' values, weighted -- `_refined_value`'s sum."""
    weights = np.array([weight for weight, _ in branches], dtype=np.float64)
    values = [child.value if isinstance(child, _Node) else child for _, child in branches]
    return float(np.array(values) @ weights)


def _propagate(node: _Node, cell: tuple[int, int]) -> None:
    """Write the refined cell's value and re-solve every node up to the root.

    A node whose re-solve fails keeps its last equilibrium (and so hands up its last
    value): the answer in hand is still an answer to a well-posed game.
    """
    here: _Node | None = node
    at = cell
    while here is not None:
        if isinstance(here, _BeliefRoot):
            here.write(at, _cell_value(here.children[at]))
            return
        here.payoff[at] = _cell_value(here.children[at])
        here.signal = None
        if here.rect is not None:
            _reread(here)
            return
        try:
            here.equilibrium = solve(here.payoff)
        except EquilibriumError:
            return
        if here.parent is None:
            return
        here, at = here.parent


def _top(strategy: np.ndarray, count: int) -> np.ndarray:
    """`search._top`: the heaviest `count` actions carrying any probability."""
    live = np.flatnonzero(strategy > 1e-9)
    if live.size <= count:
        return live
    order = np.argsort(-strategy[live], kind="stable")
    return live[order[:count]]


def _reread(node: _Node) -> None:
    """The restricted reading of `node` (IKA-68's `_restricted_search`, one pass of it).

    The rectangle is solved as the game; the strategy is its, extended by zeros; the value
    is what it guarantees against every column at today's prices. Once every cell of the
    rectangle has been refined or refused, the full matrix is the oracle: the best row and
    the best column outside join if they beat the rectangle's own value. A rectangle the
    LP cannot solve keeps the last answer.
    """
    assert node.rect is not None
    rows, cols = node.rect
    prices = node.payoff
    try:
        restricted = solve(prices[np.ix_(rows, cols)])
    except EquilibriumError:
        return
    strategy = np.zeros(prices.shape[0], dtype=np.float64)
    strategy[rows] = restricted.row_strategy
    reply = np.zeros(prices.shape[1], dtype=np.float64)
    reply[cols] = restricted.col_strategy
    row_ev = prices @ reply
    col_ev = strategy @ prices
    guarantee = float(col_ev.min())
    node.equilibrium = Equilibrium(
        value=guarantee,
        row_strategy=strategy,
        col_strategy=reply,
        row_ev=row_ev,
        col_ev=col_ev,
        row_ev_loss=np.clip(guarantee - row_ev, 0.0, None),
        col_ev_loss=np.clip(col_ev - guarantee, 0.0, None),
        duality_gap=float(restricted.duality_gap),
    )
    node.signal = None
    # Settled: nothing left in the rectangle the rule would refine -- each cell refined,
    # refused, or worth nothing (a value of exactly 0 or 1 has no `bern`). Waiting on a
    # decided cell would keep the oracle from ever being asked, and a rectangle that
    # cannot grow reads a strategy whose guarantee a column outside it takes apart.
    signal = _signal(node)
    settled = all(
        (i, j) in node.children or (i, j) in node.refused or signal[i, j] <= 0.0
        for i in rows
        for j in cols
    )
    if not settled:
        return
    best_row = int(np.argmax(row_ev))
    best_col = int(np.argmin(col_ev))
    if best_row not in rows and float(row_ev[best_row]) > float(restricted.value) + ORACLE_TOLERANCE:
        rows.append(best_row)
    if best_col not in cols and float(col_ev[best_col]) < float(restricted.value) - ORACLE_TOLERANCE:
        cols.append(best_col)


__all__ = [
    "ALL_ACTIONS",
    "BELIEF_WEIGHTED",
    "COSTS",
    "DEFAULT_DEEPEN",
    "READINGS",
    "GAP_FLOOR",
    "MAX_LEVELS",
    "Cost",
    "DeepenSpec",
    "Deepened",
    "Deepening",
    "BeliefDeepening",
    "Progress",
    "Step",
    "announce_belief_depth1",
    "announce_depth1",
    "best_first",
    "cells_for_seconds",
    "deepen_belief",
    "deepen_root",
    "deepen_spec",
    "parse_deepen",
]
