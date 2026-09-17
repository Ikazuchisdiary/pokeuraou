"""Searching without being shown the opponent's bench, and what must not change.

The first test is the one that makes every other comparison mean anything: with nothing
hidden, `belief_solve` must return what `search` returns, exactly, for both sides. A
setting whose trivial case is not the old behaviour turns every measurement against the
old behaviour into a measurement of two unrelated things.

The rest are about the uncertainty sitting on the side that has it. Averaging the
completions' matrices and solving once would let the opponent move before learning their
own bench, which is a thing that never happens -- they packed it -- and it hands us a
value we cannot collect. That failure is quiet: the answer is still a distribution over
legal actions, and it is still wrong. So it is asserted against directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.regulation import load_regulation
from pokeuraou.resolve import Budget, batched_payoff
from pokeuraou.search import belief_solve, search
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

SEEN_ALL = frozenset({0, 1, 2, 3})


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    roster = load_roster("rizabanadohido")
    sheet = list(roster.sets)[:6]
    position = position_from_sets(reg, sheet[:4], sheet[:4])
    return reg, sheet, position


def _menus(reg, position, limit=8):  # noqa: ANN001, ANN202
    return (
        narrow(reg, position, 0, limit=limit).actions,
        narrow(reg, position, 1, limit=limit).actions,
    )


def _spreads(reg, position, sheet, *, seen=None):  # noqa: ANN001, ANN202
    return {
        side: completions(reg, position, side, sheet, seen=seen) for side in (0, 1)
    }


def _leaves():  # noqa: ANN202
    return {0: HP_SHARE.batch, 1: HP_SHARE.batch}


def test_nothing_hidden_is_the_old_search_exactly(setup) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    budget = Budget.matrix()
    before = search(reg, position, ours, theirs, HP_SHARE.batch, budget=budget, depth=1)
    after = belief_solve(
        reg, position, ours, theirs,
        _spreads(reg, position, sheet, seen=SEEN_ALL), _leaves(), budget=budget,
    )
    assert after[0].classes == 1
    assert after[1].classes == 1
    assert after[0].value == pytest.approx(float(before.equilibrium.value))
    assert after[1].value == pytest.approx(-float(before.equilibrium.value))
    np.testing.assert_array_equal(after[0].strategy, before.equilibrium.row_strategy)
    np.testing.assert_array_equal(after[1].strategy, before.equilibrium.col_strategy)


def test_each_side_gets_a_distribution_over_its_own_menu(setup) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    spreads = _spreads(reg, position, sheet)
    assert len(spreads[0]) == 6 and len(spreads[1]) == 6
    answers = belief_solve(
        reg, position, ours, theirs, spreads, _leaves(), budget=Budget.matrix()
    )
    assert answers[0].strategy.shape == (len(ours),)
    assert answers[1].strategy.shape == (len(theirs),)
    for side in (0, 1):
        assert answers[side].classes == 6
        assert answers[side].strategy.min() >= -1e-9
        assert answers[side].strategy.sum() == pytest.approx(1.0)
        assert len(answers[side].replies) == 6


def test_the_opponent_answers_per_bench_not_once(setup) -> None:  # noqa: ANN001
    """One reply per completion, and they are allowed to differ -- that is the whole
    difference between this and averaging the matrices."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    answers = belief_solve(
        reg, position, ours, theirs, _spreads(reg, position, sheet), _leaves(),
        budget=Budget.matrix(),
    )
    assert all(y.shape == (len(theirs),) for y in answers[0].replies)
    assert all(y.sum() == pytest.approx(1.0) for y in answers[0].replies)


def test_it_does_not_claim_what_averaging_would(setup) -> None:  # noqa: ANN001
    """Averaging the matrices gives the opponent less than they have: they know their own
    bench. So it must not return that game's value."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    budget = Budget.matrix()
    spreads = _spreads(reg, position, sheet)
    parts = [
        batched_payoff(reg, item.position, ours, theirs, HP_SHARE.batch, budget=budget)[0]
        for item in spreads[1]
    ]
    weights = np.array([item.weight for item in spreads[1]])
    averaged = solve(sum(w * m for w, m in zip(weights, parts, strict=True)))
    answers = belief_solve(
        reg, position, ours, theirs, spreads, _leaves(), budget=budget
    )
    assert answers[0].value <= float(averaged.value) + 1e-9


def test_knowing_the_bench_is_never_worth_less_than_nothing(setup) -> None:  # noqa: ANN001
    """The value of information is non-negative by construction, so a solver that comes
    out ahead by *not* knowing has a sign error somewhere."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    budget = Budget.matrix()
    spreads = _spreads(reg, position, sheet)
    informed = sum(
        item.weight
        * float(
            solve(
                batched_payoff(
                    reg, item.position, ours, theirs, HP_SHARE.batch, budget=budget
                )[0]
            ).value
        )
        for item in spreads[1]
    )
    answers = belief_solve(
        reg, position, ours, theirs, spreads, _leaves(), budget=budget
    )
    assert informed >= answers[0].value - 1e-9
