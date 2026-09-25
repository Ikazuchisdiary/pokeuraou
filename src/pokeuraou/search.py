"""Looking further than one turn, where looking further is worth paying for.

The search this project has had until now is one ply: narrow both sides to a few dozen
candidates, resolve every cell of the matrix, score the resulting positions with the
learned value function, solve the zero-sum LP. Everything past that turn is whatever the
value function believes.

Two measurements say that is where the remaining strength is.

**Breadth is saturated.** 24 candidates beat 16 by +8.8 +-2.9 points; 48 beat 24 by
nothing at eight times the cost. There is no gain left in looking at more actions.

    2026-09-20, IKA-12: the second half of that is false, and it was the reason this
    module exists. "48 beat 24 by nothing" was a failure to reject on 1,696 games
    (+1.3 [-1.1, +3.7]); on 12,000 it is +3.4 +-0.7, and the three older runs all had
    the same sign. Width 48 costs 2.00x width 24 per move decision, measured alternated
    on an idle machine. There is gain left in looking at more actions, and it is the
    largest lever measured this year.

    2026-09-22, IKA-66: that gain is a *playing* gain, and it does not follow the games
    into training. Two pools built for the same wall clock -- 19,800 games at width 24
    against 8,400 at width 48, a rate ratio of 2.400 -- trained alone and played at
    width 24 where only the leaf differs: 46.55% +-0.82 for the width-48 pool over
    12,000 games. Cut to the SAME game count, so the quantity term is gone, it is
    49.41% +-0.82. A width-48 game teaches no better than a width-24 one.

    2026-09-23, IKA-73: turned the other way, it pays. Width 12 generates 1.889x the
    games per hour, and its pool wins 51.00% +-0.83 at the SAME game count and 52.83%
    +-0.83 at the same wall clock. On the board width 12 is about -6 against 24, so the
    teaching sign is the opposite of the playing sign, and generation now runs at 12
    (`tools/generate_queue.py`). The agent is untouched: 24 here, 48 at 45 seconds.
    **The pool is not the agent.**

**More data is saturating too.** Adding 5,600 self-play games to 8,013 was worth +3.6
[+0.2, +6.9]; adding 7,000 more to 13,613 was worth +1.7 [-1.7, +5.1], which cannot be
told from zero. Another generation of the same shape is not the lever.

    2026-09-22, IKA-66: "cannot be told from zero" was again an interval, not a finding.
    One paired match measures the slope directly instead of differencing two ratings:
    19,800 games against 8,250 of the same width, same generator, trained the same way,
    is 53.16% +-0.82 for the larger pool -- 1.263 doublings, so +2.50 points a doubling,
    inside the +1.7 to +3.6 band the two-point estimate had drawn. Data is not saturated
    at this scale; the old reading came from a measurement that could not see the effect.

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

    2026-09-20, IKA-12: played out, and it lost.

        depth 2 at width 24 against depth 1 at width 24   48.28% +-0.71   16,000 games
        width 48 against width 24, both depth 1           53.42% +-0.73   12,000 games
        depth 2 at width 24 against depth 1 at width 48   46.62% +-1.17    6,000 games

    Costs, per move decision, from inside the same runs: the refine is 5.21x a whole
    depth-1 decision at width 24; width 48 is 2.00x. So the sentence above is wrong in
    both halves -- there was something else to spend the time on, it is cheaper, and it
    wins where this loses.

    The numbers this produces are not the problem. On the same turn-1 positions the
    refine moves side 0's own equilibrium value by +0.0072 +-0.0010, the direction of
    that move predicts the outcome (r = +0.147 +-0.022 over 8,000 games), and its
    forecast scores better on Brier than depth 1's (-0.0051 +-0.0034, confounded by each
    arm facing the other). What loses is the *strategy*, which is read off a matrix of
    16 refined cells and 560 unrefined ones -- the object "What it does not claim" says
    is the equilibrium of neither game. That paragraph was right and this is what being
    right about it costs.

    2026-09-23, IKA-68: `solve_restricted` is the other reading. The refined rectangle is
    a game whose every cell is at depth 2, so its equilibrium is an equilibrium of
    something; the full matrix goes back to being what double-oracle uses it for, an
    oracle that says whether any action outside the rectangle is worth adding. Measured
    before it was built on 120 recorded mid-game positions -- ones where depth 1 mixes on
    both sides, since only there can the two readings differ -- they put different
    probabilities on the board in 88.3% of them, so the difference is one a match can see.
    On the board the two readings played the same game in 3 pairs of 783.

    The same measurement removed the other suspect. The refinement was thought to shift
    the cells it touches upward by a constant (+0.0072), which would make a refined cell
    attractive for a reason that is about depth and not about play -- but per cell, on
    those positions, `depth2 - depth1` is -0.00076 +-0.00656 with 48.2% of cells below
    zero. There is no constant to subtract. The +0.00715 that IKA-12 recorded is a
    different quantity: the equilibrium VALUE at turn 1, not a per-cell offset.

    Played out, the reading is what was losing:

        restricted depth 2 at width 24 against depth 1 at width 24   54.45% +-0.94
        restricted against the mixed reading, both depth 2           SPRT(0, +10) H1

    the first over 8,808 games, the second decided at 745 pairs. It costs 4.07x a depth-1
    decision, where the mixed reading cost 5.21x (0.82x of it, head to head). The numbers
    were never the problem: on the same turn-1 positions the restricted value sits lower
    than depth 1's (-0.0245, because it is a guarantee and not a game value) and its
    direction predicts the result exactly as well (r = +0.150 against +0.147). Only the
    strategy changed. And the mixed arm is still the arm IKA-12 played: on today's tree it
    moves the turn-1 value by +0.00775 +-0.00307, where IKA-12 recorded +0.00715.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import deepen as _deepen
from . import port, rank_scores, timing
from .actions import SideAction
from .budget import Budget
from .equilibrium import Equilibrium, EquilibriumError, solve
from .narrow import narrow
from .node_solver import solve_node
from .port import batched_payoff
from .position import Position
from .regulation import Regulation

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


#: How far a best response has to beat the restricted game's value before it is worth a
#: row or a column and the cells that come with it. The LP's own residual is around
#: 1e-12, so this is a floor on "actually better", not on arithmetic.
ORACLE_TOLERANCE = 1e-6


#: Opponent actions a leaf ranking resolves each candidate against. One is the cheapest
#: thing that is not arbitrary; more than two buys little, because the ranking only has to
#: order candidates and not value them.
DEFAULT_REFERENCES = 2

#: How a leaf ranking fills its cells, as one label per agent (IKA-268): ``refs<N>`` resolves
#: each candidate against the first N damage replies at the node's own matrix budget, and a
#: ``-fast`` suffix fills those cells at `Budget.fast` instead.
#:
#:     2026-09-25, IKA-268: neither is cheaper for nothing. In M-C generation (width 12,
#:     hidden, served) ``refs1`` halves the ranking's fill and runs 1.08-1.13x the games a
#:     minute, but keeps 67% of the played menu's actions and plays 51.30% +-1.34 for
#:     ``refs2`` over 4,000 board games (Elo +9.0 [-0.3, +18.3] for two replies). ``-fast``
#:     is DEARER: two damage rolls against the matrix budget's one give 2.5x the leaves.
#:     So the default stays; the label exists so an arm can play the other.
DEFAULT_RANK_FILL = f"refs{DEFAULT_REFERENCES}"

_RANK_FILL = re.compile(r"refs([1-9][0-9]*)(-fast)?")


def parse_rank_fill(label: str) -> tuple[int, bool]:
    """(references, fast) from a rank-fill label; a label that is not one stops."""
    got = _RANK_FILL.fullmatch(label)
    if got is None:
        raise ValueError(f"rank fill {label!r} is not refs<N> or refs<N>-fast")
    return int(got.group(1)), got.group(2) is not None


def leaf_ranking(
    reg: Regulation,
    pos: Position,
    side: int,
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    references: int = DEFAULT_REFERENCES,
    reference_limit: int = 8,
) -> Callable[..., np.ndarray]:
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

    def rank(pool: list[SideAction], _scored: object = None) -> np.ndarray:
        if not replies:
            return np.zeros(len(pool))
        # One matrix, pool x references, through the same batching the search uses: the
        # leaves of every candidate against every reference go out in a single call.
        ours, theirs = (pool, replies) if side == 0 else (replies, pool)
        # Its own purpose, because the fill and the forward pass it pays for are the same
        # stages a node's matrix pays for, and IKA-108 asks how the two compare.
        with timing.purpose("rank"):
            payoff, _notes = batched_payoff(reg, pos, ours, theirs, evaluate, budget=budget)
        rank_scores.saw_fill(side, pool, replies, payoff)  # IKA-278; nothing unless recording
        # `payoff` is always side 0's win probability, so the column player wants it low.
        return payoff.mean(axis=1) if side == 0 else -payoff.mean(axis=0)

    return rank


def believed_ranking(
    parts: Sequence[tuple[Callable[..., np.ndarray], float]],
) -> Callable[..., np.ndarray]:
    """One ranking averaged over the completions it was built for.

    `narrow` orders candidates by a score, and both scores that are not the cheap damage
    one -- the leaf's and the policy's -- read the whole position, the opponent's unplayed
    bench included. Solving the matrix over six possible benches and then ordering the menu
    with the real one leaves the answer conditioned on something nobody knows, in the part
    of the search that decides which actions get a number at all.

    Averaging the scores rather than the orderings, because an average of rankings is not a
    ranking of anything: two completions that disagree about the best action would produce
    a third order that neither of them argued for. The scores are comparable across
    completions -- the leaf's are win probabilities of the same cell under different
    benches, the policy's are its logits for the same action list -- so their mean is the
    score under the belief.

    The damage score needs none of this: `score_action` reads only the active Pokemon, so
    it never saw the bench to begin with.
    """
    if not parts:
        raise ValueError("no rankings to average")
    if len(parts) == 1:
        return parts[0][0]
    total = sum(weight for _rank, weight in parts) or 1.0

    def bound(pool: list[SideAction], scored: object = None) -> np.ndarray:
        out = None
        for rank, weight in parts:
            got = np.asarray(rank(pool, scored), dtype=np.float64) * (weight / total)
            out = got if out is None else out + got
        return out if out is not None else np.zeros(len(pool))

    return bound


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
    #: Rows and columns of the restricted game the strategy was read from. Zero when the
    #: strategy came from the full matrix, which is every depth-1 search and the shipped
    #: depth-2 one.
    restricted: tuple[int, int] = (0, 0)
    #: The restricted game's own value minus what its strategy guarantees against every
    #: column at the prices in hand. Zero at convergence, because then no column outside
    #: the rectangle beats it; positive when the pass budget ran out first.
    optimism: float = 0.0
    #: What a best-first deepening spent and how deep it reached (`deepen.Deepened`,
    #: IKA-33). None when the search was not asked to deepen.
    deepened: _deepen.Deepened | None = None


@timing.labelled("matrix")
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
    solve_restricted: bool = False,
    solve_sparsely: bool = False,
    deepen: int = 0,
) -> SearchResult:
    """Solve this turn's matrix game, optionally refining the cells that decide it.

    ``depth=1`` is the one-ply search unchanged, down to the arithmetic: the same
    `batched_payoff` call and the same `solve`. That is deliberate and tested, because a
    depth parameter whose lowest setting is not the old behaviour makes every comparison
    against it meaningless.

    ``solve_restricted`` changes only how the refined cells are read into a strategy, and
    only at depth 2 -- see `_restricted_search`. On the board it beat both depth 1 and the
    mixed reading (IKA-68). It is still an option and not the default, because depth 2
    itself does not run where the agent ships: a hidden bench goes through `belief_solve`,
    which had no depth at all until IKA-111 gave it this reading and no other.

    ``deepen`` is a budget of cells to spend after the depth-1 solve, best first over the
    whole tree (`deepen.best_first`, IKA-33). Zero is the depth-1 search unchanged; it
    goes with ``depth=1`` only, and there ``solve_restricted`` picks the root's reading:
    the restricted rectangle grown by its oracle, or the whole matrix.
    """
    if deepen and (depth > 1 or solve_sparsely):
        raise ValueError("deepen is a budget on top of the depth-1 full-matrix search")
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
    if deepen > 0:
        equilibrium, payoff, deepened = _deepen.best_first(
            reg, pos, row, col, evaluate, budget=budget, payoff=payoff,
            equilibrium=equilibrium, cells=deepen, sub_limit=sub_limit,
            sub_branches=sub_branches, unmodelled=unmodelled,
            reading="restricted" if solve_restricted else "mixed", refine=refine,
        )
        return SearchResult(
            equilibrium=equilibrium,
            payoff=payoff,
            ours=row,
            theirs=col,
            unmodelled=unmodelled,
            refined=deepened.expanded,
            deepened=deepened,
        )
    if depth <= 1:
        return SearchResult(
            equilibrium=equilibrium,
            payoff=payoff,
            ours=row,
            theirs=col,
            unmodelled=unmodelled,
        )

    if solve_restricted:
        return _restricted_search(
            reg,
            pos,
            row,
            col,
            evaluate,
            budget=budget,
            payoff=payoff,
            equilibrium=equilibrium,
            unmodelled=unmodelled,
            refine=refine,
            passes=passes,
            sub_limit=sub_limit,
            sub_branches=sub_branches,
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


def _restricted_search(
    reg: Regulation,
    pos: Position,
    row: list[SideAction],
    col: list[SideAction],
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
    payoff: np.ndarray,
    equilibrium: Equilibrium,
    unmodelled: set[str],
    refine: int,
    passes: int,
    sub_limit: int,
    sub_branches: int,
) -> SearchResult:
    """The double-oracle's own answer: the refined rectangle solved as the game it is.

    The loop in `search` refines the cells the depth-1 equilibrium weights and then
    re-solves the *whole* matrix -- 16 cells at depth 2 sitting in 560 at depth 1. IKA-12
    played that out and it lost 48.28% +-0.71 over 16,000 games while every number it
    produced improved, and this module had named the suspect before the match ran.

    Here the rectangle is the game. Every one of its cells has been resolved a ply
    further, so its equilibrium is an equilibrium of something, and the full matrix is
    demoted to what double-oracle uses it for: an oracle asked whether any action outside
    the rectangle beats what the rectangle guarantees. Ranking is the one thing depth-1
    values are still good at when they are wrong about the level, and ranking is all that
    is asked of them here.

    A pass adds at most one row and one column, and only when the oracle says that action
    is better than the restricted game's value. A pass that adds neither is convergence --
    no action outside the rectangle is worth playing at the prices in hand, which is the
    statement the shipped reading cannot make at all. It is also cheaper than the shipped
    second pass: that one can refine a whole second rectangle (up to 16 fresh cells),
    while a row and a column of an enlarged rectangle is 9.

    The value returned is what the strategy guarantees against *every* column at those
    prices, not the restricted game's own value -- the restricted game restricts the
    opponent too, so its value is optimistic by construction. The two coincide at
    convergence and `optimism` reports the gap when they do not.
    """
    rows = [int(i) for i in _top(equilibrium.row_strategy, refine)]
    cols = [int(j) for j in _top(equilibrium.col_strategy, refine)]
    #: The best value in hand for every cell: depth 2 inside the rectangle, depth 1
    #: outside it. Nothing is read *from* this matrix except a ranking.
    prices = np.array(payoff, dtype=np.float64, copy=True)
    refined: dict[tuple[int, int], float] = {}
    unrefinable: set[tuple[int, int]] = set()
    answer: Equilibrium | None = None
    shape = (0, 0)
    optimism = 0.0
    subgames = 0
    converged = False
    used = 0
    for attempt in range(1, passes + 1):
        used = attempt
        for i in rows:
            for j in cols:
                if (i, j) in refined or (i, j) in unrefinable:
                    continue
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
                if value is None:
                    # The cell keeps its depth-1 price, so the rectangle is that much
                    # less consistent -- one cell of sixteen rather than 560 of 576, and
                    # counted out loud rather than left to be inferred.
                    unrefinable.add((i, j))
                    continue
                refined[(i, j)] = value
                prices[i, j] = value
        try:
            restricted = solve(prices[np.ix_(rows, cols)])
        except EquilibriumError:
            break
        strategy = np.zeros(len(row), dtype=np.float64)
        strategy[rows] = restricted.row_strategy
        reply = np.zeros(len(col), dtype=np.float64)
        reply[cols] = restricted.col_strategy
        row_ev = prices @ reply
        col_ev = strategy @ prices
        guarantee = float(col_ev.min())
        optimism = float(restricted.value) - guarantee
        shape = (len(rows), len(cols))
        answer = Equilibrium(
            value=guarantee,
            row_strategy=strategy,
            col_strategy=reply,
            row_ev=row_ev,
            col_ev=col_ev,
            row_ev_loss=np.clip(guarantee - row_ev, 0.0, None),
            col_ev_loss=np.clip(col_ev - guarantee, 0.0, None),
            duality_gap=float(restricted.duality_gap),
        )
        best_row = int(np.argmax(row_ev))
        best_col = int(np.argmin(col_ev))
        grew = False
        if (
            best_row not in rows
            and float(row_ev[best_row]) > float(restricted.value) + ORACLE_TOLERANCE
        ):
            rows.append(best_row)
            grew = True
        if (
            best_col not in cols
            and float(col_ev[best_col]) < float(restricted.value) - ORACLE_TOLERANCE
        ):
            cols.append(best_col)
            grew = True
        if not grew:
            converged = True
            break

    if answer is None:
        # Nothing was solved. The depth-1 answer is still an answer to a well-posed game,
        # which is the same fallback the shipped reading takes.
        return SearchResult(
            equilibrium=equilibrium,
            payoff=payoff,
            ours=row,
            theirs=col,
            unmodelled=unmodelled,
        )
    if unrefinable:
        unmodelled.add(
            f"depth-2 left {len(unrefinable)} cell(s) of the restricted game at depth 1"
        )
    unmodelled.add(
        f"strategy read from the restricted {shape[0]}x{shape[1]} game; "
        f"{len(refined)} cells of it at depth 2"
    )
    if not converged:
        unmodelled.add("depth-2 best responses had not run out when the passes did")
    return SearchResult(
        equilibrium=answer,
        payoff=prices,
        ours=row,
        theirs=col,
        unmodelled=unmodelled,
        refined=len(refined),
        passes=used,
        converged=converged,
        subgames=subgames,
        restricted=shape,
        optimism=optimism,
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
    # Every branch, from the port (IKA-209; it was Python's `resolve_turn`).
    result = port.turn(reg, pos, [ours, theirs], budget, full=True)
    unmodelled.update(result.unmodelled)
    if result.suspended or not result.outcomes:
        # A self-switching move pauses the turn for a replacement choice, which is a
        # decision node and not a chance node. `batched_payoff` already folds it
        # correctly at depth 1; refining it would need the fold and the subgame at once.
        return None, unmodelled, 0

    branches = sorted(result.outcomes, key=lambda b: -b.probability)[:sub_branches]
    if len(branches) < len(result.outcomes):
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
    shortcut but the same number by a cheaper route. A learned leaf does too since IKA-253
    (`encode.settle`); before it, `value-all` put 0.54 on a certain loss here.
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


@dataclass
class BeliefResult:
    """One side's answer when it cannot see the other side's whole four.

    `strategy` belongs to `side`, not to the row player. The matrices are always built
    with side 0 as the row player because that is what `batched_payoff` produces and what
    every recorded position means by "own"; solving for side 1 transposes and negates
    instead of rebuilding them, so the two sides cannot drift apart in how a cell is read.
    """

    strategy: np.ndarray
    #: What `side` guarantees against an opponent who does know their own four. Lower than
    #: the omniscient value would claim, and that is the point.
    value: float
    #: One distribution per completion -- what the opponent does when their bench *is*
    #: that one. They differ, which is why averaging the matrices instead would be a
    #: different game.
    replies: tuple[np.ndarray, ...]
    ours: list[SideAction]
    theirs: list[SideAction]
    unmodelled: set[str] = field(default_factory=set)
    #: Completions averaged over. One means nothing was hidden and this is the old answer.
    classes: int = 1
    #: Cells given a depth-2 value, counted per completion: (row, column, completion).
    #: Zero at depth 1 (IKA-111).
    refined: int = 0
    #: Sub-games solved for those cells.
    subgames: int = 0
    #: Whether the depth-2 oracle ran out of actions to add before the passes did.
    converged: bool = True
    #: The restricted game's own value minus what its strategy guarantees against every
    #: column of every completion, as `SearchResult.optimism`.
    optimism: float = 0.0


def belief_solve(
    reg: Regulation,
    position: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    completions_by_side: dict[int, list],
    evaluators: dict[int, LeafEvaluator],
    *,
    budget: Budget,
    sides: Sequence[int] = (0, 1),
    depth: int | tuple[int, int] = 1,
    refine: int = DEFAULT_REFINE,
    passes: int = DEFAULT_PASSES,
    sub_limit: int = DEFAULT_SUB_LIMIT,
    sub_branches: int = DEFAULT_SUB_BRANCHES,
) -> dict[int, BeliefResult]:
    """Both sides' answers, resolving each turn as few times as it has to be resolved.

    `completions_by_side[s]` completes side `s`'s own unseen slots, so side `1 - s` is the
    one solved over it. `evaluators[s]` is the leaf side `s` searches with.

    When both sides search with the same leaf the whole node is resolved once and shared,
    because their hidden slots are disjoint and a shared cell's resolution depends on
    neither. With different leaves the scoring differs, so each side gets its own call and
    the sharing is only across that side's completions -- 3.0x instead of 2.2x, against
    6.0x for a matrix per completion.

    `sides` names the answers wanted; the result holds only those. With different leaves
    each side's node and LP are its own, so an answer nobody reads is not built at all --
    a match whose arms build different menus solves this twice a turn and reads one side
    of each (IKA-282). With one leaf the shared node is built as before, whichever side is
    asked for, so self-play is unchanged; only the unread LP is skipped.

    `depth` is per side, as `play_game`'s (IKA-111). At 1 the answer is the one above,
    untouched. At 2 that answer is only the start of the restricted reading
    (`_restricted_belief`): the double-oracle of `_restricted_search`, with a column set
    per completion and each refined cell resolved and sub-solved in that completion's
    position. The knobs are `search`'s and mean the same.
    """
    from .beliefnode import belief_payoffs
    from .equilibrium import solve_bayesian

    depths = (depth, depth) if isinstance(depth, int) else tuple(depth)
    wanted = tuple(side for side in (0, 1) if side in sides)
    if not wanted:
        raise ValueError(f"belief_solve asked for no side: {sides!r}")
    row, col = list(ours), list(theirs)
    same = evaluators[0] is evaluators[1]
    if same:
        node = belief_payoffs(
            reg, position, row, col, evaluators[0], budget=budget,
            spreads=completions_by_side,
        )
        nodes = {0: node, 1: node}
    else:
        nodes = {
            side: belief_payoffs(
                reg, position, row, col, evaluators[side], budget=budget,
                spreads={1 - side: completions_by_side[1 - side]},
            )
            for side in wanted
        }

    out: dict[int, BeliefResult] = {}
    #: Depth-2 cell values in side 0's orientation, shared by both sides' answers: with
    #: one leaf, a position both sides' lists hold (an exact bench) is one position.
    memo: dict[tuple, tuple[float | None, set[str], int]] = {}
    for side in wanted:
        built = nodes[side].matrices[side]
        items = completions_by_side[1 - side]
        weights = np.asarray([item.weight for item in items], dtype=np.float64)
        # Side 1 minimises the matrix side 0 maximises, so its own game is the transpose
        # of the negation. Solving that rather than reading the column strategy off side
        # 0's solve is what makes the uncertainty sit on the side that has it.
        matrices = [m if side == 0 else -m.T for m in built]
        solved = solve_bayesian(matrices, weights)
        out[side] = BeliefResult(
            strategy=np.asarray(solved.row_strategy, dtype=np.float64),
            value=float(solved.value),
            replies=tuple(np.asarray(y, dtype=np.float64) for y in solved.col_strategies),
            ours=row,
            theirs=col,
            unmodelled=nodes[side].unmodelled,
            classes=len(matrices),
        )
        if depths[side] >= 2:
            out[side] = _restricted_belief(
                reg, side, row, col, items, matrices, weights, solved, evaluators[side],
                out[side], memo, budget=budget, refine=refine, passes=passes,
                sub_limit=sub_limit, sub_branches=sub_branches,
            )
    return out


def _restricted_belief(  # noqa: PLR0913 - one side's Bayesian node and the depth-2 knobs
    reg: Regulation,
    side: int,
    row: list[SideAction],
    col: list[SideAction],
    items: list,
    matrices: list[np.ndarray],
    weights: np.ndarray,
    solved,  # noqa: ANN001 - BayesianEquilibrium, the depth-1 answer
    evaluate: LeafEvaluator,
    start: BeliefResult,
    memo: dict,
    *,
    budget: Budget,
    refine: int,
    passes: int,
    sub_limit: int,
    sub_branches: int,
) -> BeliefResult:
    """`_restricted_search` for a side that cannot see the other side's bench (IKA-111).

    `matrices[k]` is this side's own game (rows its actions) when the other bench is
    completion `k`. The rectangle has one row set, because this side does not know `k`,
    and a column set per completion, because the opponent does: the cells refined are
    (i, j, k). Each is `_refined_value` on completion `k`'s position -- the turn resolved
    there and the next turn's game solved as if that bench were known. That is the
    determinization the issue named as the cheap choice: the belief is not carried into the
    sub-game.

    The oracles are the Bayesian ones. Our best row maximises sum_k w_k (M_k y_k); the
    opponent's best column is per completion, the argmin of (x M_k). A row is added when
    it beats the restricted game's value, a completion's column when it beats what that
    completion's reply earns inside the rectangle. With one completion of weight 1 each
    of these is `_restricted_search`'s, and so is the answer (tested).

    The value returned is what the strategy guarantees against every column of every
    completion at the prices in hand, sum_k w_k min_j (x M_k)_j -- not the restricted
    game's own value, which restricts the opponent too.
    """
    from .equilibrium import solve_bayesian

    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    rows = [int(i) for i in _top(solved.row_strategy, refine)]
    cols = [[int(j) for j in _top(y, refine)] for y in solved.col_strategies]
    prices = [np.array(m, dtype=np.float64, copy=True) for m in matrices]
    refined: set[tuple[int, int, int]] = set()
    unrefinable: set[tuple[int, int, int]] = set()
    unmodelled = set(start.unmodelled)
    answer: BeliefResult | None = None
    subgames = 0
    converged = False
    for _attempt in range(passes):
        for k, item in enumerate(items):
            for i in rows:
                for j in cols[k]:
                    if (i, j, k) in refined or (i, j, k) in unrefinable:
                        continue
                    # This side's (i, j) is side 0's (i, j), or (j, i) for side 1.
                    ours, theirs = (i, j) if side == 0 else (j, i)
                    key = (id(item.position), id(evaluate), ours, theirs)
                    found = memo.get(key)
                    if found is None:
                        with timing.purpose("matrix"):
                            found = _refined_value(
                                reg, item.position, row[ours], col[theirs], evaluate,
                                budget=budget, sub_limit=sub_limit,
                                sub_branches=sub_branches,
                            )
                        memo[key] = found
                        subgames += found[2]
                    value, notes, _solved = found
                    unmodelled.update(notes)
                    if value is None:
                        unrefinable.add((i, j, k))
                        continue
                    refined.add((i, j, k))
                    prices[k][i, j] = value if side == 0 else -value
        try:
            restricted = solve_bayesian(
                [prices[k][np.ix_(rows, cols[k])] for k in range(len(items))], w
            )
        except EquilibriumError:
            break
        strategy = np.zeros(len(prices[0]), dtype=np.float64)
        strategy[rows] = restricted.row_strategy
        replies = []
        for k in range(len(items)):
            reply = np.zeros(prices[k].shape[1], dtype=np.float64)
            reply[cols[k]] = restricted.col_strategies[k]
            replies.append(reply)
        col_ev = [strategy @ prices[k] for k in range(len(items))]
        row_ev = sum(w[k] * (prices[k] @ replies[k]) for k in range(len(items)))
        guarantee = float(sum(w[k] * float(col_ev[k].min()) for k in range(len(items))))
        answer = BeliefResult(
            strategy=strategy,
            value=guarantee,
            replies=tuple(replies),
            ours=start.ours,
            theirs=start.theirs,
            unmodelled=unmodelled,
            classes=start.classes,
            optimism=float(restricted.value) - guarantee,
        )
        grew = False
        best_row = int(np.argmax(row_ev))
        if (
            best_row not in rows
            and float(row_ev[best_row]) > float(restricted.value) + ORACLE_TOLERANCE
        ):
            rows.append(best_row)
            grew = True
        for k in range(len(items)):
            earned = float(col_ev[k] @ replies[k])
            best_col = int(np.argmin(col_ev[k]))
            if (
                best_col not in cols[k]
                and float(col_ev[k][best_col]) < earned - ORACLE_TOLERANCE
            ):
                cols[k].append(best_col)
                grew = True
        if not grew:
            converged = True
            break

    if timing.ON:
        timing.count("depth2.cells", len(refined))
        timing.count("depth2.subgames", subgames)
    if answer is None:
        # Nothing was solved; the depth-1 answer is still an answer, as in the open game.
        return start
    if unrefinable:
        unmodelled.add(
            f"hidden depth-2 left {len(unrefinable)} cell(s) of the restricted game at depth 1"
        )
    unmodelled.add(
        f"hidden-bench strategy read from the restricted {len(rows)}x"
        f"{'/'.join(str(len(c)) for c in cols)} game over {len(items)} completion(s); "
        f"{len(refined)} cells of it at depth 2"
    )
    if not converged:
        unmodelled.add("hidden depth-2 best responses had not run out when the passes did")
    answer.refined = len(refined)
    answer.subgames = subgames
    answer.converged = converged
    return answer


__all__ = [
    "DEFAULT_PASSES",
    "DEFAULT_RANK_FILL",
    "DEFAULT_REFERENCES",
    "DEFAULT_REFINE",
    "DEFAULT_SUB_BRANCHES",
    "DEFAULT_SUB_LIMIT",
    "ORACLE_TOLERANCE",
    "BeliefResult",
    "SearchResult",
    "belief_solve",
    "believed_ranking",
    "leaf_ranking",
    "parse_rank_fill",
    "search",
]
