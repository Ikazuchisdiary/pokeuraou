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

#: Cells of deepening a second of one logical core buys, for the human opponent's clock and
#: nothing else. IKA-279's generation NPS (about 26,000 leaves a second a logical core, 2.6
#: leaves a cell) would say 10,000; a deepened cell costs more than a matrix cell, because
#: each refined cell pays its own turn, a narrow and a forward pass per child for a handful
#: of cells (IKA-33: 0.43-0.64 CPU ms a deepened cell in M-C generation, 0.74 ms wall at
#: the median on IKA-254's endgames with a local GPU leaf, 3.7 ms at p90).
CELLS_PER_SECOND = 1_500

#: What ships: no deepening. A label is ``none``, ``m<N>`` (the whole matrix read, N cells)
#: or ``r<N>`` (the restricted reading, N cells).
DEFAULT_DEEPEN = "none"

READINGS = {"m": "mixed", "r": "restricted"}

_DEEPEN = re.compile(r"none|([mr])([1-9][0-9]*)")


def parse_deepen(label: str) -> tuple[str | None, int]:
    """(reading, cells) from a deepen label: (None, 0), ("mixed", 400), ("restricted", 400)."""
    got = _DEEPEN.fullmatch(label)
    if got is None:
        raise ValueError(f"deepen {label!r} is not none, m<N> or r<N> (N cells, N >= 1)")
    if got.group(1) is None:
        return None, 0
    return READINGS[got.group(1)], int(got.group(2))


def cells_for_seconds(seconds: float, cores: int = 1) -> int:
    """The cell budget that fits in `seconds` of `cores` logical cores (the human's clock).

    Generation and the board never call this: a budget in seconds would read the load of
    the machine into the game.
    """
    return max(0, int(seconds * cores * CELLS_PER_SECOND))


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

    def to_json(self) -> dict[str, int]:
        return {
            "budget": self.budget,
            "cells": self.cells,
            "expanded": self.expanded,
            "depth": self.depth,
            **({"refused": self.refused} if self.refused else {}),
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
    spent = 0
    expanded = 0
    deepest = 0
    refused = 0
    while spent < cells:
        target = _best(root)
        if target is None:
            break
        node, cell = target
        cost, ok = _expand(
            reg, node, cell, evaluate, budget=budget, sub_limit=sub_limit,
            sub_branches=sub_branches, unmodelled=unmodelled,
        )
        spent += cost
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
        timing.count("deepen.cells", spent)
        timing.count("deepen.expanded", expanded)
    report = Deepened(
        budget=cells, cells=spent, expanded=expanded, depth=1 + deepest, refused=refused
    )
    return root.equilibrium, root.payoff, report


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
) -> tuple[int, bool]:
    """Refine one cell of `node` as `search._refined_value` does, keeping the children.

    Returns the cells spent and whether the cell took a deeper value.
    """
    i, j = cell
    result = port.turn(reg, node.pos, [node.rows[i], node.cols[j]], budget, full=True)
    unmodelled.update(result.unmodelled)
    spent = 1
    if result.suspended or not result.outcomes:
        return spent, False
    branches = sorted(result.outcomes, key=lambda b: -b.probability)[:sub_branches]
    if len(branches) < len(result.outcomes):
        unmodelled.add(
            f"depth-2 kept the {sub_branches} likeliest branches of a refined cell"
        )
    weights = np.array([b.probability for b in branches], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return spent, False
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
            return spent, False
        payoff, notes = batched_payoff(reg, child_pos, row, col, evaluate, budget=budget)
        spent += len(row) * len(col)
        unmodelled.update(notes)
        try:
            equilibrium = solve(payoff)
        except EquilibriumError:
            return spent, False
        kept.append((
            float(weight),
            _Node(
                pos=child_pos, rows=list(row), cols=list(col),
                payoff=np.asarray(payoff, dtype=np.float64), equilibrium=equilibrium,
                level=node.level + 1, parent=(node, cell), weight=float(weight),
            ),
        ))
    node.children[cell] = kept
    return spent, True


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
    "CELLS_PER_SECOND",
    "DEFAULT_DEEPEN",
    "READINGS",
    "GAP_FLOOR",
    "MAX_LEVELS",
    "Deepened",
    "best_first",
    "cells_for_seconds",
    "parse_deepen",
]
