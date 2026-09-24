"""Solving a node without resolving all of it.

A width-48 node is 2,304 cells and the resolver is most of a generation run, but an
equilibrium does not need every cell. To show that an unplayed action is not a better
reply you need its payoff against the *opponent's mixed strategy*, and that strategy sits
on one to five columns. Measured on the leaf generation actually runs
(`tools/cells_needed.py`), a fifth of the matrix is enough.

Double oracle is the standard way to use that, and what makes it usable here is that it
ends in a proof rather than an approximation: keep a restricted set of rows and columns,
solve the small game, then look for a better reply for each side over *all* of its
actions. When neither side can find one, the restricted solution is an equilibrium of the
whole game -- including every cell that was never resolved. `tools/diff_solve_node.py`
checks that claim against the full matrix rather than trusting the argument.

What it does not promise is *which* equilibrium. Where a game has several of equal value
the vertex reached can differ from the one a full LP would return, and this project samples
its moves from that vertex, so a generated game can differ. The current choice is not more
correct -- it is whichever vertex the LP solver happened to return -- but it is different,
and that is why this is asked for rather than assumed.

(The name is the game-solving sense of "oracle", a best-reply routine. `oracle.py` next
door is about team sets and is a different word.)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .actions import SideAction
from .budget import Budget
from .equilibrium import Equilibrium, solve
from .port import batched_payoffs
from .position import Position
from .regulation import Regulation

#: Weight below which a strategy is not on an action. The LP returns exact zeros for
#: actions outside the support, so this only guards against a solver's dust.
SUPPORT_FLOOR = 1e-12

#: Rounds before giving up and filling the rest. Convergence is three to ten on real
#: nodes; this is a guard against a cycle, not a budget.
MAX_ROUNDS = 40


@dataclass(slots=True)
class SolvedNode:
    """The equilibrium, and what it cost to be sure of it."""

    equilibrium: Equilibrium
    payoff: np.ndarray
    unmodelled: set[str]
    exact: np.ndarray
    #: Cells resolved, against the whole matrix. The number this exists to reduce.
    resolved: int
    cells: int
    rounds: int
    #: True when the rounds ran out and the matrix was filled the ordinary way.
    fell_back: bool


def solve_node(
    reg: Regulation,
    pos: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    evaluate: Callable[[list[Position]], np.ndarray],
    *,
    budget: Budget,
    seeded: int = 1,
) -> SolvedNode:
    """The equilibrium of a node, resolving the cells that prove it and no others.

    One evaluator, because the iteration is driven by one matrix. A caller that wants a
    second opinion on the same node -- the analyser's cross-check -- wants every cell of
    both anyway, and should fill it.
    """
    rows, cols = len(ours), len(theirs)
    payoff = np.zeros((rows, cols), dtype=np.float64)
    exact = np.zeros((rows, cols), dtype=bool)
    unmodelled: set[str] = set()
    have = np.zeros((rows, cols), dtype=bool)

    def resolve(wanted: list[tuple[int, int]]) -> None:
        """Fills the cells not already filled. Everything expensive happens here."""
        missing = [(i, j) for i, j in wanted if not have[i, j]]
        if not missing:
            return
        filled, notes, cell_exact = batched_payoffs(
            reg, pos, ours, theirs, [evaluate], budget=budget, cells=missing
        )
        for i, j in missing:
            payoff[i, j] = filled[0][i, j]
            exact[i, j] = cell_exact[i, j]
            have[i, j] = True
        unmodelled.update(notes)

    # `narrow` hands its candidates over best-first, so the restricted game starts from
    # the top `seeded` of each. More of them costs cells in the first round and saves
    # rounds after it, and a round is a crossing and a forward pass.
    restricted_rows: list[int] = list(range(min(seeded, rows)))
    restricted_cols: list[int] = list(range(min(seeded, cols)))

    for round_index in range(1, MAX_ROUNDS + 1):
        resolve([(i, j) for i in restricted_rows for j in restricted_cols])
        sub = solve(payoff[np.ix_(restricted_rows, restricted_cols)])
        x = np.asarray(sub.row_strategy, dtype=np.float64)
        y = np.asarray(sub.col_strategy, dtype=np.float64)

        # Against the columns the opponent actually plays, what is every row worth? That
        # is the whole saving: one cell per row per *supported* column, not per column.
        # Both sides' questions go out together -- one crossing a round, not two, and one
        # forward pass over their leaves instead of two small ones.
        live_cols = [restricted_cols[k] for k in np.flatnonzero(y > SUPPORT_FLOOR)]
        col_weights = y[y > SUPPORT_FLOOR]
        live_rows = [restricted_rows[k] for k in np.flatnonzero(x > SUPPORT_FLOOR)]
        row_weights = x[x > SUPPORT_FLOOR]
        resolve(
            [(i, j) for i in range(rows) for j in live_cols]
            + [(i, j) for i in live_rows for j in range(cols)]
        )
        against_y = payoff[:, live_cols] @ col_weights
        best_row = int(np.argmax(against_y))
        against_x = row_weights @ payoff[live_rows, :]
        best_col = int(np.argmin(against_x))

        grew = False
        if against_y[best_row] > sub.value + SUPPORT_FLOOR and best_row not in restricted_rows:
            restricted_rows.append(best_row)
            grew = True
        if against_x[best_col] < sub.value - SUPPORT_FLOOR and best_col not in restricted_cols:
            restricted_cols.append(best_col)
            grew = True
        if not grew:
            # Neither side can do better against the other's strategy, over every action
            # it has -- which is what makes this an equilibrium of the whole node and not
            # only of the part that was resolved.
            #
            # The EV columns fall out of the same two vectors: `against_y` is what every
            # row is worth against the opponent's play and `against_x` what every column
            # is, which is exactly what had to be computed to know the search was done. The
            # report the analyser prints is not paid for twice.
            return SolvedNode(
                equilibrium=Equilibrium(
                    value=sub.value,
                    row_strategy=_spread(x, restricted_rows, rows),
                    col_strategy=_spread(y, restricted_cols, cols),
                    row_ev=against_y,
                    col_ev=against_x,
                    row_ev_loss=np.clip(sub.value - against_y, 0.0, None),
                    col_ev_loss=np.clip(against_x - sub.value, 0.0, None),
                    duality_gap=sub.duality_gap,
                ),
                payoff=payoff,
                unmodelled=unmodelled,
                exact=exact,
                resolved=int(have.sum()),
                cells=rows * cols,
                rounds=round_index,
                fell_back=False,
            )

    # A cycle, which should not happen and is not worth being wrong about.
    resolve([(i, j) for i in range(rows) for j in range(cols)])
    return SolvedNode(
        equilibrium=solve(payoff),
        payoff=payoff,
        unmodelled=unmodelled,
        exact=exact,
        resolved=int(have.sum()),
        cells=rows * cols,
        rounds=MAX_ROUNDS,
        fell_back=True,
    )


def _spread(strategy: np.ndarray, indices: list[int], width: int) -> np.ndarray:
    """The restricted game's strategy, written back over the whole action list."""
    full = np.zeros(width, dtype=np.float64)
    for k, index in enumerate(indices):
        full[index] = strategy[k]
    return full
