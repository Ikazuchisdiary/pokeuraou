"""The search's depth-1 node and the menus' leaf ranking.

`search` fills one turn's matrix over two menus (`port.batched_payoff`, the leaf at each
resolved cell) and solves the zero-sum LP (`equilibrium.solve`): the open depth-1 node, for
research tools and tests (`tools/q_teach.py`'s check, `tools/human_baseline.py`,
`tools/why_action.py`). The roads of a game solve their nodes in the port (`beliefnode`,
`search.belief_solve`, `portdeepen`). `leaf_ranking` and `believed_ranking` rank a menu's
candidates by the leaf (the ``refs<N>`` fills, IKA-268), and `resolve_rank_fill` names the
fill a menu reads (IKA-338, IKA-341).

The depth-2 reads that lived here are gone: the fixed depth-2 search (IKA-12, IKA-68, its
restricted reading and oracle) in IKA-463, Python's best-first deepening (IKA-33) and the
depth-2 refine of a list of cells (`_refine_cells`, IKA-291 to IKA-389, the ladder's) in
IKA-464. Their measurements are in the records (IKA-12, IKA-66, IKA-68, IKA-73, IKA-291).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import deepen as _deepen
from . import portlp, rank_scores, timing
from .actions import SideAction
from .budget import Budget
from .equilibrium import Equilibrium, solve
from .narrow import narrow
from .port import batched_payoff
from .position import Position
from .regulation import Regulation

#: A leaf evaluator: many positions in, one probability each out.
LeafEvaluator = Callable[[list[Position]], np.ndarray]

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
#:
#: A ``-nocover`` suffix (IKA-323) builds the leaf-ranked root menu from the ranking alone:
#: the top ``limit`` candidates, with no greedy cover of every slot option first. The
#: options that leaves off the menu are reported in `Narrowed.uncovered`, as the budget's
#: are (narrow.py's principles). Only the root's leaf-ranked menu reads it; the damage
#: menus -- a sub-game's, a deepened child's, the ranking's own references -- keep the cover.
#:
#: The refs fills (and `leaf_ranking` behind them) are kept, played only when named
#: (IKA-341, the user's decision of 9/28): M-B's roster path plays them, a person's game
#: falls back to them with no Q, the hp-share leaf and a test without torch have no Q to
#: rank by, and the records played at ``refs2`` (M-C gen-0, all of M-B) are rebuilt by them
#: (`tools/q_teach.py menus`, `tools/rank_fill_menus.py`).
#:
#: Since IKA-341 this is no leaf-ranked menu's default anywhere: an unnamed fill of a
#: leaf-ranked menu is `SHIPPED_RANK_FILL`, in the library as in generation and the board
#: (`resolve_rank_fill`). ``refs2`` is played only where it is named -- a person's game with
#: no Q (`tools/play_human.py`, a note), and research that names the refs fills (IKA-268,
#: IKA-310, IKA-323). What stays is its other job: the label of an arm whose menu reads no
#: fill (the damage or the policy ranking). Its records carry no ``rankFill``
#: (`provenance.LEGACY_RANK_FILL`), as before IKA-268, so the label is kept to keep those bytes.
DEFAULT_RANK_FILL = f"refs{DEFAULT_REFERENCES}"

#: The fill of a leaf-ranked menu that names none, in M-C generation and on the board
#: (IKA-338): rank by the default Q (`qrank.DEFAULT_Q`) without the cover.
#:
#:     2026-09-27, IKA-331: from pools made in the same wall clock (IKA-73's form),
#:     q-nocover's pool made 1.18x the games (1,243 against 1,056 a minute) and the leaf
#:     learned from it played refs2's leaf at Elo +3.8 [-6.3, +14.0] over 2,000 pairs,
#:     non-inferior (SPRT(-10, 0) H1). As a player it was +26.4 [+19.5, +33.3] over refs2
#:     at the same CPU (IKA-274 stage 2), which is why the board and human play took it
#:     first (9/26).
#:
#: The tools resolve an unnamed ``--rank-fill`` with `resolve_rank_fill`; ``refs2`` is still
#: played when named. A q fill with no Q stops -- it never falls back to ``refs2`` in
#: generation or on the board (`qrank.default_q`). Since IKA-341 the library resolves an
#: unnamed fill the same way (`play_game`, `_menus`, `PoolArm`): it loads
#: no Q itself, so a leaf-ranked call with none installed stops at its first menu
#: (`qrank.installed`).
SHIPPED_RANK_FILL = "q-nocover"


def resolve_rank_fill(given: str | None, rank_leaf: bool) -> str:
    """The fill an arm plays: the one named, else `SHIPPED_RANK_FILL` for a leaf-ranked menu.

    A damage-ranked arm reads no fill, so it keeps `DEFAULT_RANK_FILL`: its records then
    carry no ``rankFill``, as before IKA-338, and it needs no Q. ``rank_leaf`` is whether
    the arm's menu is ranked by the leaf -- False under the damage ranking and under a
    policy, which supersedes the leaf.
    """
    if given is not None:
        return given
    return SHIPPED_RANK_FILL if rank_leaf else DEFAULT_RANK_FILL


_RANK_FILL = re.compile(r"refs([1-9][0-9]*)(-fast)?(-nocover)?")


def parse_rank_fill(label: str) -> tuple[int, bool]:
    """(references, fast) from a rank-fill label; a label that is not one stops.

    ``q`` / ``q-nocover`` (IKA-274, `qrank`, and either with ``.NAME``) rank by a learned
    Q and fill no cell with the leaf: (0, False).
    """
    from .qrank import is_q

    if is_q(label):
        return 0, False
    got = _RANK_FILL.fullmatch(label)
    if got is None:
        raise ValueError(
            f"rank fill {label!r} is not refs<N>, refs<N>-fast, or either with -nocover, "
            "or q / q-nocover (either with .NAME)"
        )
    return int(got.group(1)), got.group(2) is not None


def rank_fill_covers(label: str) -> bool:
    """Whether a leaf-ranked menu of this rank-fill label is built on the cover (IKA-323)."""
    from .qrank import is_q, q_covers

    parse_rank_fill(label)
    if is_q(label):
        return q_covers(label)
    return not label.endswith("-nocover")


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

@timing.labelled("matrix")
def search(
    reg: Regulation,
    pos: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    evaluate: LeafEvaluator,
    *,
    budget: Budget,
) -> SearchResult:
    """Solve this turn's matrix game: the depth-1 node, `batched_payoff` and `solve`.

    The fixed depth-2 read (``depth=2``) was removed in IKA-463, and the best-first
    deepening (``deepen``, IKA-33) in IKA-464: a move is deepened in the port
    (`portdeepen`, through `humanplay.solve_move`).
    """
    row = list(ours)
    col = list(theirs)
    with timing.region("open.node"):
        payoff, unmodelled = batched_payoff(reg, pos, row, col, evaluate, budget=budget)
    with timing.region("lp.open"):
        equilibrium = solve(payoff)
    return SearchResult(
        equilibrium=equilibrium,
        payoff=payoff,
        ours=row,
        theirs=col,
        unmodelled=unmodelled,
    )


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
    #: The depth-1 node this answer was solved over, as (side 0's menu, side 1's menu, one
    #: matrix per completion in side 0's orientation, their weights) -- references, nothing
    #: computed.
    node_payoff: tuple | None = None


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
    progress: _deepen.Progress | None = None,
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

    A side's deepened answer is the port's (`portdeepen`, IKA-464), from the same node.

    `progress` is called with the depth-1 answer of every wanted side (`deepen.Step`'s
    ``start`` and ``done``, IKA-332; the step's root says which side). It changes nothing
    that is computed.
    """
    from .beliefnode import belief_payoffs

    wanted = tuple(side for side in (0, 1) if side in sides)
    if not wanted:
        raise ValueError(f"belief_solve asked for no side: {sides!r}")
    row, col = list(ours), list(theirs)
    same = evaluators[0] is evaluators[1]
    if same:
        node = belief_payoffs(
            reg, position, row, col, evaluators[0], budget=budget,
            spreads=completions_by_side, solve=wanted,
        )
        nodes = {0: node, 1: node}
    else:
        nodes = {
            side: belief_payoffs(
                reg, position, row, col, evaluators[side], budget=budget,
                spreads={1 - side: completions_by_side[1 - side]}, solve=(side,),
            )
            for side in wanted
        }

    out: dict[int, BeliefResult] = {}
    games: dict[int, tuple[list, np.ndarray]] = {}
    for side in wanted:
        built = nodes[side].matrices[side]
        items = completions_by_side[1 - side]
        weights = np.asarray([item.weight for item in items], dtype=np.float64)
        # Side 1 minimises the matrix side 0 maximises, so its own game is the transpose
        # of the negation. Solving that rather than reading the column strategy off side
        # 0's solve is what makes the uncertainty sit on the side that has it.
        games[side] = ([m if side == 0 else -m.T for m in built], weights)
    ported: dict[int, Any] = {}
    # IKA-455 stage 1.2: the games the port solved with the node's fold.
    ported.update({side: nodes[side].solved[side] for side in wanted if side in nodes[side].solved})
    # IKA-455: the other sides' games in one crossing to the port, each `solve_bayesian`'s
    # answer (`portlp`); a fault of the port is `portlp.PortFault` (IKA-460).
    rest = [side for side in wanted if side not in ported]
    if rest:
        with timing.region("lp.side"), timing.stage("lp"):
            got = portlp.solve_bayesian_many(reg, [games[side] for side in rest])
        ported.update(zip(rest, got, strict=True))
    # The first error is the one the loop below meets first.
    for side in wanted:
        if isinstance(ported.get(side), Exception):
            raise ported[side]
    for side in wanted:
        built = nodes[side].matrices[side]
        items = completions_by_side[1 - side]
        matrices, weights = games[side]
        solved = ported[side]
        strategy = np.asarray(solved.row_strategy, dtype=np.float64)
        out[side] = BeliefResult(
            strategy=strategy,
            value=float(solved.value),
            replies=tuple(np.asarray(y, dtype=np.float64) for y in solved.col_strategies),
            ours=row,
            theirs=col,
            unmodelled=nodes[side].unmodelled,
            classes=len(matrices),
        )
        if progress is not None:
            own, other = (row, col) if side == 0 else (col, row)
            _deepen.announce_depth1(
                progress, side, own, other, items, weights, matrices, solved
            )
        out[side].node_payoff = (row, col, built, weights)
    return out


__all__ = [
    "DEFAULT_RANK_FILL",
    "DEFAULT_REFERENCES",
    "SHIPPED_RANK_FILL",
    "BeliefResult",
    "SearchResult",
    "belief_solve",
    "believed_ranking",
    "leaf_ranking",
    "parse_rank_fill",
    "rank_fill_covers",
    "resolve_rank_fill",
    "search",
]
