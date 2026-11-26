"""Generation's LPs in the port (IKA-455, `portlp.GEN`) against Python's, off by default.

What these hold:

- **the same Bayesian answers**: `belief_solve` with a bench hidden gives the same strategies,
  values and replies to the bit whether both sides' games go to the port in one crossing or
  are solved in Python one at a time. The port's own count of Bayesian games is the positive
  control, and a game with one weight changed is the control that the comparison can fail.
- **the same Q rankings**: `qrank.prefetch` with the port solving each request's game gives
  `q_ranking` the same scores to the bit as solving it in `rank`, and the game both sides read
  is solved once (the second side's `q.solve.shared` read is still the memo's).

No torch: a made-up Q hands back random matrices, and the leaf is hp-share.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pokeuraou import portlp, qhead, qrank
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.regulation import load_regulation
from pokeuraou.search import belief_solve
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    roster = load_roster("rizabanadohido")
    sheet = list(roster.sets)[:6]
    position = position_from_sets(reg, sheet[:4], sheet[:4])
    return reg, sheet, position


@pytest.fixture
def gen_off():  # noqa: ANN201
    was = portlp.GEN[0]
    yield
    portlp.set_gen(was)


def _hidden(reg, position, sheet, *, limit=6):  # noqa: ANN001, ANN202
    ours = narrow(reg, position, 0, limit=limit).actions
    theirs = narrow(reg, position, 1, limit=limit).actions
    seen = frozenset({0, 1})
    spreads = {side: completions(reg, position, side, sheet, seen=seen) for side in (0, 1)}
    return ours, theirs, spreads


def _belief(reg, position, ours, theirs, spreads):  # noqa: ANN001, ANN202
    return belief_solve(
        reg, position, ours, theirs, spreads, {0: HP_SHARE.batch, 1: HP_SHARE.batch},
        budget=Budget.matrix(),
    )


def _same(a, b) -> bool:  # noqa: ANN001
    return (
        a.value == b.value
        and np.array_equal(a.strategy, b.strategy)
        and len(a.replies) == len(b.replies)
        and all(np.array_equal(x, y) for x, y in zip(a.replies, b.replies, strict=True))
    )


def test_the_ports_belief_games_are_solve_bayesians(setup, gen_off) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    ours, theirs, spreads = _hidden(reg, position, sheet)
    assert max(len(items) for items in spreads.values()) > 1, "nothing hidden: no Bayesian game"
    portlp.set_gen(False)
    before = portlp.counts()["bayes"]
    python = _belief(reg, position, ours, theirs, spreads)
    assert portlp.counts()["bayes"] == before  # off: nothing went to the port
    portlp.set_gen(True)
    ported = _belief(reg, position, ours, theirs, spreads)
    # The positive control: both sides' games crossed, in the one crossing.
    assert portlp.counts()["bayes"] - before == 2
    assert all(_same(python[side], ported[side]) for side in (0, 1))
    # The comparison can fail: another weight is another game.
    spreads[1][0] = replace(spreads[1][0], weight=spreads[1][0].weight * 0.5)
    other = _belief(reg, position, ours, theirs, spreads)
    assert not all(_same(python[side], other[side]) for side in (0, 1))


class _MadeUpQ:
    """A Q that answers each request with a seeded random matrix of the pools' shape."""

    def __init__(self, seed: int = 5) -> None:
        self.rng = np.random.default_rng(seed)

    def matrices(self, reg, asks):  # noqa: ANN001, ANN201
        return [self.rng.random((len(pools[0]), len(pools[1]))) for _at, pools in asks]


def _rankings(reg, position, gen: bool, seed: int = 5) -> list[np.ndarray]:  # noqa: ANN001
    portlp.set_gen(gen)
    model = _MadeUpQ(seed)
    given = qrank.prefetch(reg, position, (position, position), model)
    out = []
    for side in (0, 1):
        pool = qhead.legal_pool(reg, position, side)
        rank = qrank.q_ranking(reg, position, side, model, "q-nocover", given.get(side))
        out.append(rank(pool))
    return out


def test_the_ports_q_rankings_are_solves(setup, gen_off) -> None:  # noqa: ANN001
    reg, _sheet, position = setup
    before = portlp.counts()["games"]
    python = _rankings(reg, position, False)
    assert portlp.counts()["games"] == before
    ported = _rankings(reg, position, True)
    # One game for both sides: they read one position and one pair of pools.
    assert portlp.counts()["games"] - before == 1
    for a, b in zip(python, ported, strict=True):
        np.testing.assert_array_equal(a, b)
    # The comparison can fail: another Q's matrices are other games.
    again = _rankings(reg, position, True, seed=6)
    assert not np.array_equal(ported[0], again[0])
