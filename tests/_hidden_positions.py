"""Positions of one game of two different fours, for the deepening's hidden-bench tests.

Until IKA-463 these lived in `test_hidden_depth2.py`, which went with the fixed depth-2 read.
"""

from __future__ import annotations

import numpy as np

from pokeuraou.budget import Budget
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import Position
from pokeuraou.selfplay import position_from_sets

from ._port import resolve_turn

LEAF = HP_SHARE.batch
SEEN_ALL = frozenset({0, 1, 2, 3})


def _played(roster, turns: int = 4, seed: int = 3) -> list[Position]:  # noqa: ANN001
    """Positions from one game of two different fours (a mirror is 0.5 at every depth
    under hp-share, and a test of "the number moved" would pass on nothing)."""
    reg = roster.reg
    pos = position_from_sets(reg, list(roster.sets[:4]), list(roster.sets[2:6]))
    rng = np.random.default_rng(seed)
    out: list[Position] = []
    for _ in range(turns):
        if pos.ended:
            break
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            break
        out.append(pos.copy())
        result = resolve_turn(
            reg, pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        pos = max(result.branches, key=lambda b: b.probability).position
    return out
