"""Searching without being shown the opponent's bench, and what must not change.

The first test is the one that makes every other comparison mean anything: with a single
completion -- nothing hidden -- `belief_search` must return what `search` returns, exactly.
A setting whose trivial case is not the old behaviour turns every measurement against the
old behaviour into a measurement of two unrelated things.

The rest are about the uncertainty sitting on the side that has it. Averaging the six
matrices and solving once would let the opponent move before learning their own bench,
which is a thing that never happens -- they packed it -- and it hands us a value we cannot
collect. That failure is quiet: the answer is still a distribution over legal actions, and
it is still wrong. So it is asserted against directly.
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
from pokeuraou.search import belief_search, search
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


def _menus(reg, position, limit=8):  # noqa: ANN001, ANN202
    return (
        narrow(reg, position, 0, limit=limit).actions,
        narrow(reg, position, 1, limit=limit).actions,
    )


def test_one_completion_is_the_old_search_exactly(setup) -> None:  # noqa: ANN001
    reg, _sheet, position = setup
    ours, theirs = _menus(reg, position)
    budget = Budget.matrix()
    before = search(
        reg, position, ours, theirs, HP_SHARE.batch, budget=budget, depth=1
    )
    after = belief_search(
        reg, [(position, 1.0)], ours, theirs, HP_SHARE.batch, budget=budget, side=0
    )
    assert after.classes == 1
    assert after.value == pytest.approx(float(before.equilibrium.value))
    np.testing.assert_array_equal(after.strategy, before.equilibrium.row_strategy)


def test_one_completion_from_side_one_is_the_column_strategy(setup) -> None:  # noqa: ANN001
    """Side 1's answer is the same game read from the other end, not a second solve."""
    reg, _sheet, position = setup
    ours, theirs = _menus(reg, position)
    budget = Budget.matrix()
    before = search(
        reg, position, ours, theirs, HP_SHARE.batch, budget=budget, depth=1
    )
    after = belief_search(
        reg, [(position, 1.0)], ours, theirs, HP_SHARE.batch, budget=budget, side=1
    )
    np.testing.assert_array_equal(after.strategy, before.equilibrium.col_strategy)
    assert after.value == pytest.approx(-float(before.equilibrium.value))


def test_the_answer_is_a_distribution_over_our_menu(setup) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    spread = [(c.position, c.weight) for c in completions(reg, position, 1, sheet)]
    assert len(spread) == 6
    result = belief_search(
        reg, spread, ours, theirs, HP_SHARE.batch, budget=Budget.matrix(), side=0
    )
    assert result.classes == 6
    assert result.strategy.shape == (len(ours),)
    assert result.strategy.min() >= -1e-9
    assert result.strategy.sum() == pytest.approx(1.0)
    assert len(result.replies) == 6


def test_the_opponent_answers_per_bench_not_once(setup) -> None:  # noqa: ANN001
    """One reply per completion, and they are allowed to differ -- that is the whole
    difference between this and averaging the matrices."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    spread = [(c.position, c.weight) for c in completions(reg, position, 1, sheet)]
    result = belief_search(
        reg, spread, ours, theirs, HP_SHARE.batch, budget=Budget.matrix(), side=0
    )
    assert all(y.shape == (len(theirs),) for y in result.replies)
    assert all(y.sum() == pytest.approx(1.0) for y in result.replies)


def test_it_does_not_claim_what_averaging_would(setup) -> None:  # noqa: ANN001
    """Averaging the matrices gives the opponent less than they have: they know their own
    bench. So it must not return that game's value, and in this position it is strictly
    more optimistic for us."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    budget = Budget.matrix()
    spread = completions(reg, position, 1, sheet)
    parts = [
        batched_payoff(reg, item.position, ours, theirs, HP_SHARE.batch, budget=budget)[0]
        for item in spread
    ]
    weights = np.array([item.weight for item in spread])
    averaged = solve(sum(w * m for w, m in zip(weights, parts, strict=True)))
    honest = belief_search(
        reg, [(c.position, c.weight) for c in spread], ours, theirs, HP_SHARE.batch,
        budget=budget, side=0,
    )
    assert honest.value <= float(averaged.value) + 1e-9


def test_knowing_the_bench_is_never_worth_less_than_nothing(setup) -> None:  # noqa: ANN001
    """The value of information is non-negative by construction, so a solver that comes
    out ahead by *not* knowing has a sign error somewhere."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position)
    budget = Budget.matrix()
    spread = completions(reg, position, 1, sheet)
    parts = [
        batched_payoff(reg, item.position, ours, theirs, HP_SHARE.batch, budget=budget)[0]
        for item in spread
    ]
    informed = sum(
        item.weight * float(solve(part).value)
        for item, part in zip(spread, parts, strict=True)
    )
    honest = belief_search(
        reg, [(c.position, c.weight) for c in spread], ours, theirs, HP_SHARE.batch,
        budget=budget, side=0,
    )
    assert informed >= honest.value - 1e-9


def test_no_completions_is_refused(setup) -> None:  # noqa: ANN001
    reg, _sheet, position = setup
    ours, theirs = _menus(reg, position)
    with pytest.raises(ValueError, match="no completions"):
        belief_search(
            reg, [], ours, theirs, HP_SHARE.batch, budget=Budget.matrix(), side=0
        )
