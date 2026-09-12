"""Looking further than one turn, where looking further is worth paying for.

The search this project has had until now is one ply: narrow both sides to a few dozen
candidates, resolve every cell of the matrix, score the resulting positions with the
learned value function, solve the zero-sum LP. Everything past that turn is whatever the
value function believes.

Two measurements say that is where the remaining strength is.

**Breadth is saturated.** 24 candidates beat 16 by +8.8 +-2.9 points; 48 beat 24 by
nothing at eight times the cost. There is no gain left in looking at more actions.

**More data is saturating too.** Adding 5,600 self-play games to 8,013 was worth +3.6
[+0.2, +6.9]; adding 7,000 more to 13,613 was worth +1.7 [-1.7, +5.1], which cannot be
told from zero. Another generation of the same shape is not the lever.

Depth is the axis nothing has tried.

## Why not simply search two ply

The honest depth-2 value of a cell is the equilibrium value of the position that cell
leads to -- a whole matrix game per cell, per chance branch. At 24x24 that is 576 cells
times several branches times another 24x24 each: about thirteen times a whole game's
current leaf budget, for one decision. It is not a tuning problem, it is three orders of
magnitude.

## What this does instead

Refine the cells that decide the answer, and only those.

A matrix game's value depends on the cells where both players put probability. The
equilibria here are concentrated -- 359 of 394 solved selections were pure -- so the set
that matters is small. So:

1. Solve at depth 1, which gives a value for *every* cell.
2. Take the actions carrying equilibrium probability, at most `refine` per side, and give
   their cells a depth-2 value: resolve the turn, solve the resulting position's own
   matrix, average over the chance branches.
3. Re-solve. The support may have moved, because refinement changed the numbers.
4. Repeat until the support stops moving, or `passes` is spent.

At convergence every cell carrying equilibrium probability has a depth-2 value. Cells
outside the support keep their depth-1 values, and that is the approximation: an action
can only be wrongly excluded if its depth-1 value understates it, and the iteration is
what catches that -- an action that gains support gets refined on the next pass, and
leaves again if refinement does not hold up. This is the double-oracle idea with the
depth-1 matrix standing in for the best-response oracle, which is exact for the actions
it ranks and free because that matrix was computed anyway.

## What it does not claim

It does not claim to be the depth-2 equilibrium. The unrefined cells are depth-1 values
sitting in the same matrix, and a mixed equilibrium over a matrix of mixed depths is not
the equilibrium of either. What is reported is what was refined and whether the support
converged, so a caller can tell a converged answer from a truncated one.

Nor is it assumed to be *better*. Depth costs wall clock, and the comparison that settles
it is depth-2 against depth-1 at equal wall clock, played out. Breadth is saturated at
24, so that comparison is not rigged in depth's favour: there is nothing else to spend
the time on.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .actions import SideAction
from .equilibrium import Equilibrium, EquilibriumError, solve
from .narrow import narrow
from .node_solver import solve_node
from .position import Position
from .regulation import Regulation
from .resolve import Budget, batched_payoff, resolve_turn

#: A leaf evaluator: many positions in, one probability each out.
LeafEvaluator = Callable[[list[Position]], np.ndarray]

#: How many actions a side may have refined per pass. Every pair of refined actions is a
#: cell, so this is quadratic: 4 is 16 cells, 6 is 36. Chosen to keep a refined decision
#: within a small multiple of an unrefined one rather than from any property of the game.
DEFAULT_REFINE = 4

#: How many times the support may move before the search stops anyway. Two is enough for
#: the support to shift once and settle; a third pass is worth measuring, not assuming.
DEFAULT_PASSES = 2

#: Candidate width inside a refined cell. Narrower than the root on purpose -- the point
#: of the sub-solve is a better value for one cell, not a good strategy for a position
#: nobody will play.
DEFAULT_SUB_LIMIT = 8

#: Chance branches kept per refined cell, heaviest first and renormalised. A turn can
#: produce dozens once secondaries and accuracy are enumerated, and the tail is worth
#: less than the sub-solves it would cost.
DEFAULT_SUB_BRANCHES = 3


#: Opponent actions a leaf ranking resolves each candidate against. One is the cheapest
#: thing that is not arbitrary; more than two buys little, because the ranking only has to
#: order candidates and not value them.
DEFAULT_REFERENCES = 2


def leaf_ranking(
    reg: Regulation,
    pos: Position,
    side: int,
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    references: int = DEFAULT_REFERENCES,
    reference_limit: int = 8,
) -> Callable[[list[SideAction]], np.ndarray]:
    """A ranking for `narrow` that scores candidates with the leaf instead of with damage.

    `narrow` ranks by expected damage fraction, and the matrix is filled by the leaf. The
    two disagree, and the disagreement is not cosmetic: over 59 decisions where the menu's
    best reply and the best legal reply differed by more than 0.05 of win probability,
    playing the leaf's choice was worth +14.3 points [+7.1, +21.5] in real games. The
    damage score cannot see a weak move that leads somewhere good, so such a move only
    ever reaches the menu paired with whatever partner the coverage rule happened to give
    it.

    Each candidate is resolved against a small set of opponent replies and scored by the
    leaf on the resulting positions. The replies come from the *damage* score, which is
    circular only in the harmless direction: a reference that is merely plausible is
    enough to order candidates, and using the leaf to pick the references too would cost
    another pass over the pool for no measured gain.

    Cost is `len(pool) * references` resolves against the matrix's own `limit**2`, so a
    24-wide matrix ranked over a hundred legal pairs at two references pays about a third
    again. Width 48 costs four times. Whether the cheaper one buys the same thing is a
    question for a head-to-head, which is why this is an option and not a default.
    """
    replies = narrow(reg, pos, 1 - side, limit=reference_limit).actions[:references]

    def rank(pool: list[SideAction]) -> np.ndarray:
        if not replies:
            return np.zeros(len(pool))
        # One matrix, pool x references, through the same batching the search uses: the
        # leaves of every candidate against every reference go out in a single call.
        ours, theirs = (pool, replies) if side == 0 else (replies, pool)
        payoff, _notes = batched_payoff(reg, pos, ours, theirs, evaluate, budget=budget)
        # `payoff` is always side 0's win probability, so the column player wants it low.
        return payoff.mean(axis=1) if side == 0 else -payoff.mean(axis=0)

    return rank


@dataclass(slots=True)
class SearchResult:
    """What the search decided, and what it cost to decide it."""

    equilibrium: Equilibrium
    payoff: np.ndarray
    ours: list[SideAction]
    theirs: list[SideAction]
    unmodelled: set[str] = field(default_factory=set)
    #: Cells given a depth-2 value. Zero at depth 1.
    refined: int = 0
    #: Passes actually run before the support stopped moving.
    passes: int = 0
    #: Whether the support stopped moving on its own rather than running out of passes.
    converged: bool = True
    #: Sub-games solved, which is the cost that separates this from a depth-1 search.
    subgames: int = 0


def search(
    reg: Regulation,
    pos: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    depth: int = 1,
    refine: int = DEFAULT_REFINE,
    passes: int = DEFAULT_PASSES,
    sub_limit: int = DEFAULT_SUB_LIMIT,
    sub_branches: int = DEFAULT_SUB_BRANCHES,
    solve_sparsely: bool = False,
) -> SearchResult:
    """Solve this turn's matrix game, optionally refining the cells that decide it.

    ``depth=1`` is the one-ply search unchanged, down to the arithmetic: the same
    `batched_payoff` call and the same `solve`. That is deliberate and tested, because a
    depth parameter whose lowest setting is not the old behaviour makes every comparison
    against it meaningless.
    """
    row = list(ours)
    col = list(theirs)
    if solve_sparsely and depth <= 1:
        # An equilibrium needs about a fifth of a wide matrix and can prove it; the rest
        # of the cells are work nobody reads. It reaches *an* equilibrium of the same
        # value rather than the one a full LP returns, so it is asked for, not assumed.
        solved = solve_node(reg, pos, row, col, evaluate, budget=budget)
        return SearchResult(
            equilibrium=solved.equilibrium,
            payoff=solved.payoff,
            ours=row,
            theirs=col,
            unmodelled=solved.unmodelled,
        )
    payoff, unmodelled = batched_payoff(reg, pos, row, col, evaluate, budget=budget)
    equilibrium = solve(payoff)
    if depth <= 1:
        return SearchResult(
            equilibrium=equilibrium,
            payoff=payoff,
            ours=row,
            theirs=col,
            unmodelled=unmodelled,
        )

    #: (i, j) -> the depth-2 value, so a cell is never refined twice across passes.
    refined: dict[tuple[int, int], float] = {}
    subgames = 0
    converged = False
    used = 0
    for attempt in range(1, passes + 1):
        used = attempt
        wanted = _cells_to_refine(equilibrium, refine)
        fresh = [cell for cell in wanted if cell not in refined]
        if not fresh:
            converged = True
            break
        for i, j in fresh:
            value, notes, solved = _refined_value(
                reg,
                pos,
                row[i],
                col[j],
                evaluate,
                budget=budget,
                sub_limit=sub_limit,
                sub_branches=sub_branches,
            )
            subgames += solved
            unmodelled.update(notes)
            if value is not None:
                refined[(i, j)] = value
        updated = payoff.copy()
        for (i, j), value in refined.items():
            updated[i, j] = value
        payoff = updated
        try:
            equilibrium = solve(payoff)
        except EquilibriumError:
            # The refined matrix failed to solve; the depth-1 answer is still a valid
            # answer to a well-posed game, so hand that back rather than nothing.
            break

    if refined:
        unmodelled.add(
            f"depth-2 value on {len(refined)} of {payoff.size} cells; the rest are depth 1"
        )
    if not converged:
        unmodelled.add("depth-2 support had not settled when the pass budget ran out")
    return SearchResult(
        equilibrium=equilibrium,
        payoff=payoff,
        ours=row,
        theirs=col,
        unmodelled=unmodelled,
        refined=len(refined),
        passes=used,
        converged=converged,
        subgames=subgames,
    )


def _cells_to_refine(equilibrium: Equilibrium, refine: int) -> list[tuple[int, int]]:
    """The cells the current equilibrium actually weights, heaviest first.

    A cell changes the value of the game in proportion to the product of the two
    probabilities on it, so that product is the ranking. Taking the top `refine` actions
    per side rather than the top `refine**2` cells keeps the refined region a rectangle,
    which is what the re-solve needs: a lone refined cell in an otherwise depth-1 row
    makes that row look better or worse for a reason that is about depth, not about play.
    """
    rows = _top(equilibrium.row_strategy, refine)
    cols = _top(equilibrium.col_strategy, refine)
    return [(int(i), int(j)) for i in rows for j in cols]


def _top(strategy: np.ndarray, count: int) -> np.ndarray:
    """Indices of the heaviest `count` actions that carry any probability at all."""
    live = np.flatnonzero(strategy > 1e-9)
    if live.size <= count:
        return live
    order = np.argsort(-strategy[live], kind="stable")
    return live[order[:count]]


def _refined_value(
    reg: Regulation,
    pos: Position,
    ours: SideAction,
    theirs: SideAction,
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    sub_limit: int,
    sub_branches: int,
) -> tuple[float | None, set[str], int]:
    """One cell's value, taken from the next turn's equilibrium instead of the leaf.

    Returns ``None`` when the cell cannot be refined -- a turn that stopped for a
    mid-turn replacement, a position with no legal actions on one side, a subgame the LP
    could not solve -- and the caller keeps the depth-1 value. Refusing a cell is always
    safe; inventing one is not.
    """
    unmodelled: set[str] = set()
    result = resolve_turn(reg, pos, [ours, theirs], budget=budget)
    unmodelled.update(result.unmodelled)
    if result.suspended or not result.branches:
        # A self-switching move pauses the turn for a replacement choice, which is a
        # decision node and not a chance node. `batched_payoff` already folds it
        # correctly at depth 1; refining it would need the fold and the subgame at once.
        return None, unmodelled, 0

    branches = sorted(result.branches, key=lambda b: -b.probability)[:sub_branches]
    if len(branches) < len(result.branches):
        unmodelled.add(
            f"depth-2 kept the {sub_branches} likeliest branches of a refined cell"
        )
    weights = np.array([b.probability for b in branches], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        return None, unmodelled, 0
    weights /= total

    values: list[float] = []
    solved = 0
    for branch in branches:
        value, notes, did = _subgame_value(
            reg, branch.position, evaluate, budget=budget, sub_limit=sub_limit
        )
        unmodelled.update(notes)
        solved += did
        if value is None:
            return None, unmodelled, solved
        values.append(value)
    return float(np.array(values) @ weights), unmodelled, solved


def _subgame_value(
    reg: Regulation,
    pos: Position,
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    sub_limit: int,
) -> tuple[float | None, set[str], int]:
    """The equilibrium value of one position's own matrix game.

    A finished position has no game to solve and its leaf value is already the answer --
    the objectives all return exactly 1 or 0 once the battle is decided, so this is not a
    shortcut but the same number by a cheaper route.
    """
    if pos.ended:
        return float(evaluate([pos])[0]), set(), 0
    row = narrow(reg, pos, 0, limit=sub_limit).actions
    col = narrow(reg, pos, 1, limit=sub_limit).actions
    if not row or not col:
        return None, set(), 0
    payoff, unmodelled = batched_payoff(reg, pos, row, col, evaluate, budget=budget)
    try:
        return float(solve(payoff).value), unmodelled, 1
    except EquilibriumError:
        return None, unmodelled, 1


__all__ = [
    "DEFAULT_PASSES",
    "DEFAULT_REFERENCES",
    "DEFAULT_REFINE",
    "DEFAULT_SUB_BRANCHES",
    "DEFAULT_SUB_LIMIT",
    "SearchResult",
    "leaf_ranking",
    "search",
]
