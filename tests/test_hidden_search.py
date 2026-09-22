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


def _fixture_sheets(reg, position, data):  # noqa: ANN001, ANN202
    """Each side's six as sets: the four on the board exactly, the two they left at home
    from the usage prior, which is all an observer could have anyway."""
    import numpy as np

    from pokeuraou.priors import (
        SampledSet,
        find_cached_chaos,
        load_chaos,
        sample_set,
    )

    cached = find_cached_chaos(reg.meta.format_id)
    if cached is None:
        pytest.skip("no cached usage stats to fill the unbrought two")
    prior = load_chaos(cached, reg)
    rng = np.random.default_rng(0)
    out = []
    for side, key in ((0, "ownSix"), (1, "foeSix")):
        board = {mon.species: mon for mon in position.sides[side].pokemon}
        six = []
        for name in data[key]:
            mon = board.get(name)
            six.append(
                SampledSet(
                    species=mon.species, ability=mon.ability, item=mon.item,
                    nature=mon.nature, sp=dict(mon.sp or {}),
                    moves=[m.id for m in mon.moves],
                )
                if mon is not None
                else sample_set(rng, reg, prior.species[name])
            )
        out.append(six)
    return tuple(out)


@pytest.fixture(scope="module")
def replacement():  # noqa: ANN201
    """A real mid-game replacement node, taken from a recorded game.

    A turn-1 position built for the occasion gives a forced or one-sided replacement, and a
    forced choice agrees with anything -- the first version of this test compared [1.0, 0.0]
    with [1.0, 0.0] and would have passed against a node that did nothing at all.
    """
    import json
    from pathlib import Path

    from pokeuraou.position import Position
    from pokeuraou.resolve import replacements_needed

    path = Path(__file__).parent / "fixtures" / "replacement-mixed.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    reg = load_regulation(data["position"]["format"])
    register_mega_stones(reg)
    position = Position.from_json(data["position"])
    return reg, position, replacements_needed(position), data


def _replace(reg, position, owed, sheets, shown):  # noqa: ANN001, ANN202
    import numpy as np

    from pokeuraou.selfplay import GameRecord, _do_replacement_node

    record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
    _do_replacement_node(
        reg, np.random.default_rng(3), position.copy(), owed, record,
        (HP_SHARE.batch, HP_SHARE.batch), sheets=sheets, shown=shown,
    )
    assert record.decisions, "the node recorded nothing"
    return record.decisions[0], record


def test_the_replacement_node_is_unchanged_when_nothing_is_hidden(replacement) -> None:  # noqa: ANN001
    """The last node that was still handed the opponent's whole four.

    Every slot seen must give what it always gave -- the same menus, the same mixtures and
    the same value, exactly.
    """
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    old, _ = _replace(reg, position, owed, None, None)
    new, _ = _replace(reg, position, owed, sheets, [SEEN_ALL, SEEN_ALL])
    assert old.own_actions == new.own_actions
    assert old.foe_actions == new.foe_actions
    np.testing.assert_array_equal(old.own_policy, new.own_policy)
    np.testing.assert_array_equal(old.foe_policy, new.foe_policy)
    assert old.search_value == new.search_value


def test_hiding_the_bench_changes_the_replacement(replacement) -> None:  # noqa: ANN001
    """And with slots unseen it must not give the same answer back.

    Without this the test above passes just as well against a branch that quietly falls
    through to the open path, which is the failure worth guarding: it would leave the leak
    in place and every test green.
    """
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    open_node, _ = _replace(reg, position, owed, None, None)
    blind, record = _replace(
        reg, position, owed, sheets, [frozenset({0, 1}), frozenset({0, 1})]
    )
    assert not record.unmodelled, record.unmodelled
    assert sum(blind.own_policy) == pytest.approx(1.0)
    assert sum(blind.foe_policy) == pytest.approx(1.0)
    assert blind.search_value != open_node.search_value, (
        "solving over six possible benches returned the open-information value; "
        "the belief path did not run"
    )


def test_the_menu_is_ranked_from_a_completion_and_not_from_the_position(setup) -> None:  # noqa: ANN001
    """`narrow` decides which actions get a number at all, and the leaf ranking reads the
    whole position to decide it -- so ranking from the true position leaves that choice
    conditioned on a bench nobody has seen. Measured over 60 recorded positions, the leaky
    order kept 85.3% of the full equilibrium's mass and the belief order 86.4%, both far
    over the damage order's 61.3%.

    Asserted against a spread whose first completion is deliberately *not* the true bench.
    The obvious test -- "the belief menu differs from the leaky one" -- passes or fails on
    which completion happens to come first: in this fixture the first one is the true bench,
    so the two menus agree and the test would have reported success for a ranker that had
    never been given the belief at all.
    """
    from pokeuraou.selfplay import _menus

    reg, sheet, position = setup
    budget = Budget.matrix()
    spreads = _spreads(reg, position, sheet)
    assert len(spreads[1]) > 1, "nothing is hidden here, so there is nothing to test"

    # Put a completion that is not the true bench first.
    on_board = {position.sides[1].pokemon[i].species for i in (2, 3)}
    elsewhere = [c for c in spreads[1] if set(c.species) != on_board]
    assert elsewhere, "every completion is the true bench, which cannot be"
    moved = {0: spreads[0], 1: [elsewhere[0], *[c for c in spreads[1] if c is not elsewhere[0]]]}

    ours, _ = _menus(reg, position, (24, 24), HP_SHARE.batch, budget, True, None, moved)
    direct, _ = _menus(
        reg, elsewhere[0].position, (24, 24), HP_SHARE.batch, budget, True, None, None
    )
    leaky, _ = _menus(reg, position, (24, 24), HP_SHARE.batch, budget, True, None, None)

    assert [a.to_choice() for a in ours] == [a.to_choice() for a in direct], (
        "the menu is not the one ranking from that completion gives; the spread did not "
        "reach the ranker"
    )
    assert [a.to_choice() for a in ours] != [a.to_choice() for a in leaky], (
        "ranking from a completion that is not the true bench gave the leaky menu anyway"
    )


def test_a_menu_with_nothing_hidden_is_the_old_menu(setup) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    budget = Budget.matrix()
    seen = _spreads(reg, position, sheet, seen=SEEN_ALL)
    before, before_foe = _menus_of(reg, position, budget, None)
    after, after_foe = _menus_of(reg, position, budget, seen)
    assert [a.to_choice() for a in before] == [a.to_choice() for a in after]
    assert [a.to_choice() for a in before_foe] == [a.to_choice() for a in after_foe]


def _menus_of(reg, position, budget, spreads):  # noqa: ANN001, ANN202
    from pokeuraou.selfplay import _menus

    return _menus(reg, position, (24, 24), HP_SHARE.batch, budget, True, None, spreads)
