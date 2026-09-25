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
made new. Every probed and added cell is charged to the budget. Without candidates
outside the menu this is `m<N>` to the bit.

**What a budget costs.** Labels count cells, so a game is the same game on any machine.
Time is counted separately (`Cost`): fills, refinements and cells each have a price in
milliseconds, measured per machine form and per number of cores, and `cells_for_seconds`
turns a clock into a budget in those prices' units (IKA-293; IKA-32 keeps the prices per
kind of work because each shrinks differently on more cores).
"""

from __future__ import annotations

import heapq
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import port, timing
from .actions import SideAction
from .budget import Budget
from .equilibrium import Equilibrium, EquilibriumError, solve
from .narrow import narrow
from .port import batched_payoff
from .position import Position
from .regulation import Regulation

LeafEvaluator = Callable[[list[Position]], np.ndarray]

#: Added to a regret before dividing by it (IKA-281's `GAP_FLOOR`), so an action on the
#: support ranks by its uncertainty rather than by an infinity.
GAP_FLOOR = 1e-3

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

    def ms(self, fills: int, refines: int, cells: int) -> float:
        return fills * self.fill + refines * self.refine + cells * self.cell


#: Measured prices, by (machine form, logical cores). ``local``: one process, the shipped
#: ensemble leaf on the local GPU -- a person's opponent. ``served``: the board's shape
#: (24 workers over 2 inference servers, 16 cores busy), CPU milliseconds a worker spends.
#: Only one core is measured (IKA-293 section 10); IKA-32 adds the others.
COSTS: dict[tuple[str, int], Cost] = {
    ("local", 1): Cost(fill=3.79, refine=2.64, cell=0.030),
    ("served", 1): Cost(fill=3.79, refine=2.64, cell=0.030),
}

#: What ships: no deepening. A label is ``none``, ``m<N>`` (the whole matrix read, N cells)
#: or ``r<N>`` (the restricted reading, N cells), and ``m<N>`` may end in ``o<W>`` or
#: ``oall``: the root's double oracle over the rest of a width-W menu, or every legal action.
DEFAULT_DEEPEN = "none"

READINGS = {"m": "mixed", "r": "restricted"}

#: The oracle's candidates as a width: ``oall`` is every legal action.
ALL_ACTIONS = 1 << 30

_DEEPEN = re.compile(r"none|([mr])([1-9][0-9]*)(?:o([1-9][0-9]*|all))?")


@dataclass(frozen=True, slots=True)
class DeepenSpec:
    """A deepen label read: how the root is read, the cells, and the oracle's width."""

    reading: str | None
    cells: int
    #: Width of the menu whose rest the root's double oracle asks (`ALL_ACTIONS` for every
    #: legal action), or None: no breadth at the root.
    oracle: int | None = None


def deepen_spec(label: str) -> DeepenSpec:
    """`DeepenSpec` of a label: ``none``, ``m400``, ``r25``, ``m400o24``, ``m400oall``."""
    got = _DEEPEN.fullmatch(label)
    if got is None:
        raise ValueError(
            f"deepen {label!r} is not none, m<N>, r<N>, m<N>o<W> or m<N>oall "
            "(N cells, N >= 1; W the oracle's width)"
        )
    if got.group(1) is None:
        return DeepenSpec(None, 0)
    oracle = got.group(3)
    if oracle is not None and got.group(1) != "m":
        raise ValueError(
            f"deepen {label!r}: the root's double oracle goes with the whole-matrix reading "
            "(m); the restricted one (r) already grows its rectangle by its own oracle"
        )
    width = None if oracle is None else ALL_ACTIONS if oracle == "all" else int(oracle)
    return DeepenSpec(READINGS[got.group(1)], int(got.group(2)), width)


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

    def to_json(self) -> dict[str, int]:
        return {
            "budget": self.budget,
            "cells": self.cells,
            "expanded": self.expanded,
            "depth": self.depth,
            **({"refused": self.refused} if self.refused else {}),
            **(
                {"probed": self.probed, "widened": self.widened, "fills": self.fills}
                if self.oracle
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
) -> Deepening:
    """`best_first`, and the root's double oracle when `outside` is given (IKA-293).

    `outside` is each side's candidates for the root: actions already on its menu are
    skipped, the rest are asked every step whether they gain against the other side's
    support, and the best that gains joins the root (see the module's docstring). Given,
    even with nothing outside the menu, the report says what the oracle did.

    `cost` changes only how the budget is counted: None counts cells (a refined cell's
    turn is one), a `Cost` charges each fill, refinement and cell its price in units of
    one cell (`cells_for_seconds`'s budget).
    """
    if outside is not None and reading != "mixed":
        raise ValueError("the root's double oracle goes with the mixed reading")
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
    elif reading != "mixed":
        raise ValueError(f"reading {reading!r} is not one of {sorted(READINGS.values())}")
    if trace is not None:
        trace.append(root)
    meter = _Meter(cost)
    oracle = None if outside is None else _Oracle(root, outside)
    expanded = 0
    deepest = 0
    refused = 0
    while meter.spent < cells:
        if oracle is not None:
            oracle.probe(reg, evaluate, budget, meter, unmodelled)
            if meter.spent >= cells:
                break
            if oracle.widen(reg, evaluate, budget, meter, unmodelled):
                if trace is not None:
                    trace.append((root, None, True))
                continue
        target = _best(root)
        if target is None:
            break
        node, cell = target
        spent, ok, fills = _expand(
            reg, node, cell, evaluate, budget=budget, sub_limit=sub_limit,
            sub_branches=sub_branches, unmodelled=unmodelled,
        )
        meter.refined(spent - 1, fills)
        if trace is not None:
            trace.append((node, cell, ok))
        if not ok:
            node.refused.add(cell)
            refused += 1
            if node.rect is not None:
                _reread(node)
            continue
        expanded += 1
        deepest = max(deepest, node.level + 1)
        _propagate(node, cell)
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
    report = Deepened(
        budget=cells, cells=meter.refines + meter.cells, expanded=expanded,
        depth=1 + deepest, refused=refused, oracle=oracle is not None,
        probed=0 if oracle is None else oracle.probed,
        widened=0 if oracle is None else oracle.widened,
        fills=meter.fills,
    )
    return Deepening(
        equilibrium=root.equilibrium, payoff=root.payoff, rows=root.rows, cols=root.cols,
        report=report,
    )


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

    def refined(self, cells: int, fills: int) -> None:
        self.refines += 1
        self.cells += cells
        self.fills += fills

    def filled(self, cells: int) -> None:
        self.fills += 1
        self.cells += cells

    @property
    def spent(self) -> float:
        if self.cost is None:
            return self.refines + self.cells
        return self.cost.ms(self.fills, self.refines, self.cells) / self.cost.cell


class _Oracle:
    """The root's double oracle over actions outside its menu (IKA-293).

    Keeps every cell it has asked, by the two actions' choices, so a probe fills only the
    pairs the support's last change made new, and an action that joins brings the cells
    already known into its row or column.
    """

    def __init__(
        self, root: _Node, outside: tuple[Sequence[SideAction], Sequence[SideAction]]
    ) -> None:
        self.root = root
        self.candidates = (list(outside[0]), list(outside[1]))
        self.known: dict[tuple[str, str], float] = {}
        #: Actions that could not join (the LP failed with them); never asked again.
        self.dropped: tuple[set[str], set[str]] = (set(), set())
        self.probes = 0
        self.probed = 0
        self.widened = 0

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

    def probe(
        self,
        reg: Regulation,
        evaluate: LeafEvaluator,
        budget: Budget,
        meter: _Meter,
        unmodelled: set[str],
    ) -> None:
        """Fill, in one call, every outside action against the other side's support that
        is not known yet."""
        root = self.root
        out_rows, out_cols = self._outside(0), self._outside(1)
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
        meter.filled(len(wanted))
        self.probes += 1
        self.probed += len(wanted)

    def widen(
        self,
        reg: Regulation,
        evaluate: LeafEvaluator,
        budget: Budget,
        meter: _Meter,
        unmodelled: set[str],
    ) -> bool:
        """Add the outside action with the largest best-response gain, if one gains.

        Rows before columns, first in the candidates' order on a tie. The rest of its row
        or column is filled at depth 1 (charged) and the root re-solved. Returns whether
        an action joined.
        """
        root = self.root
        eq = root.equilibrium
        value = float(eq.value)
        best: tuple[float, int, SideAction] | None = None
        for side in (0, 1):
            support = self._support(1 - side)
            strategy = eq.col_strategy if side == 0 else eq.row_strategy
            for action in self._outside(side):
                key = action.to_choice()
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
        return True


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
    i, j = cell
    result = port.turn(reg, node.pos, [node.rows[i], node.cols[j]], budget, full=True)
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
    "COSTS",
    "DEFAULT_DEEPEN",
    "READINGS",
    "GAP_FLOOR",
    "MAX_LEVELS",
    "Cost",
    "DeepenSpec",
    "Deepened",
    "Deepening",
    "best_first",
    "cells_for_seconds",
    "deepen_root",
    "deepen_spec",
    "parse_deepen",
]
