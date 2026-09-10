"""LP equilibrium solver.

The reported frequencies, win probability and EV losses all come out of here, so this is
one of the three components that has to be right for any printed number to mean anything.

The decisive test is :func:`test_random_games_satisfy_the_equilibrium_conditions`: for a
few hundred random matrices it checks the defining property directly (no pure deviation
beats the value for either player), which is a stronger statement than agreeing with
hand-computed answers on a handful of textbook games.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.equilibrium import EquilibriumError, best_response, solve, solve_bayesian, value_of

TOL = 1e-9

RPS = np.array(
    [
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 1.0, 0.0],
    ]
)


def test_rock_paper_scissors_is_uniform_with_value_zero() -> None:
    eq = solve(RPS)
    assert eq.value == pytest.approx(0.0, abs=1e-9)
    assert eq.row_strategy == pytest.approx([1 / 3] * 3, abs=1e-7)
    assert eq.col_strategy == pytest.approx([1 / 3] * 3, abs=1e-7)
    # Every action is on the support, so no action loses EV.
    assert eq.row_ev_loss == pytest.approx([0.0] * 3, abs=1e-9)
    assert eq.col_ev_loss == pytest.approx([0.0] * 3, abs=1e-9)
    assert eq.duality_gap < TOL
    assert len(eq.row_support()) == 3


def test_dominant_row_gives_a_pure_strategy() -> None:
    # Row 1 beats row 0 in every column, and column 0 beats column 1 for the minimiser.
    payoff = np.array(
        [
            [0.1, 0.2],
            [0.6, 0.7],
        ]
    )
    eq = solve(payoff)
    assert eq.value == pytest.approx(0.6)
    assert eq.row_strategy == pytest.approx([0.0, 1.0], abs=1e-9)
    assert eq.col_strategy == pytest.approx([1.0, 0.0], abs=1e-9)
    # The dominated row is off the support and its EV loss says how much it costs.
    assert eq.row_ev_loss[0] == pytest.approx(0.5)
    assert eq.row_ev_loss[1] == pytest.approx(0.0, abs=1e-9)


def test_saddle_point_matrix_is_pure_for_both() -> None:
    payoff = np.array(
        [
            [4.0, 3.0, 2.0],
            [5.0, 4.0, 3.0],
            [6.0, 5.0, 1.0],
        ]
    )
    eq = solve(payoff)
    # Row maximin: rows have minima 2, 3, 1 -> row 1 (value 3).
    # Column minimax: columns have maxima 6, 5, 3 -> column 2 (value 3).
    assert eq.value == pytest.approx(3.0)
    assert eq.row_strategy == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert eq.col_strategy == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_transposing_and_negating_flips_the_sign_and_swaps_the_players() -> None:
    rng = np.random.default_rng(20260909)
    for _ in range(50):
        m, n = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        a = rng.normal(size=(m, n))
        eq = solve(a)
        flipped = solve(-a.T)
        assert flipped.value == pytest.approx(-eq.value, abs=1e-7)
        assert value_of(a, eq.row_strategy, flipped.row_strategy) == pytest.approx(
            eq.value, abs=1e-7
        )


def test_adding_a_constant_shifts_the_value_but_not_the_strategies() -> None:
    """A constant-sum game (win probabilities summing to 1) reduces to the zero-sum one."""
    eq = solve(RPS)
    shifted = solve(RPS + 0.5)
    assert shifted.value == pytest.approx(eq.value + 0.5, abs=1e-9)
    assert shifted.row_strategy == pytest.approx(eq.row_strategy, abs=1e-7)
    assert shifted.col_strategy == pytest.approx(eq.col_strategy, abs=1e-7)


def test_scaling_scales_the_value_and_keeps_the_strategies() -> None:
    eq = solve(RPS + 0.5)
    scaled = solve(2.0 * (RPS + 0.5))
    assert scaled.value == pytest.approx(2.0 * eq.value, abs=1e-9)
    assert scaled.row_strategy == pytest.approx(eq.row_strategy, abs=1e-7)


def test_matching_pennies_2x2_mixed() -> None:
    payoff = np.array([[1.0, -1.0], [-1.0, 1.0]])
    eq = solve(payoff)
    assert eq.value == pytest.approx(0.0, abs=1e-9)
    assert eq.row_strategy == pytest.approx([0.5, 0.5], abs=1e-7)
    assert eq.col_strategy == pytest.approx([0.5, 0.5], abs=1e-7)


def test_asymmetric_2x2_matches_the_closed_form() -> None:
    # For a 2x2 with no saddle point the mixed equilibrium has a closed form.
    a, b, c, d = 0.8, 0.2, 0.3, 0.9
    payoff = np.array([[a, b], [c, d]])
    denom = a - b - c + d
    x0 = (d - c) / denom
    y0 = (d - b) / denom
    value = (a * d - b * c) / denom
    eq = solve(payoff)
    assert eq.value == pytest.approx(value, abs=1e-9)
    assert eq.row_strategy == pytest.approx([x0, 1 - x0], abs=1e-7)
    assert eq.col_strategy == pytest.approx([y0, 1 - y0], abs=1e-7)


def test_degenerate_shapes() -> None:
    eq = solve(np.array([[0.42]]))
    assert eq.value == pytest.approx(0.42)
    assert eq.row_strategy == pytest.approx([1.0])
    assert eq.col_strategy == pytest.approx([1.0])

    # Single row: the column player picks the worst column for us.
    eq = solve(np.array([[0.7, 0.2, 0.5]]))
    assert eq.value == pytest.approx(0.2)
    assert eq.col_strategy == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)

    # Single column: we pick the best row.
    eq = solve(np.array([[0.7], [0.2], [0.5]]))
    assert eq.value == pytest.approx(0.7)
    assert eq.row_strategy == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_rejects_malformed_payoffs() -> None:
    with pytest.raises(ValueError, match="non-empty 2-D"):
        solve(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="non-empty 2-D"):
        solve(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="non-finite"):
        solve(np.array([[1.0, np.nan]]))


def test_random_games_satisfy_the_equilibrium_conditions() -> None:
    """The defining property, checked directly on random games.

    At an equilibrium (x, y) with value v:
      - no pure row action does better than v against y, and
      - no pure column action holds the row player below v against x.
    Both are equalities on the support, which is what makes EV loss readable.
    """
    rng = np.random.default_rng(7)
    for trial in range(300):
        m, n = int(rng.integers(1, 13)), int(rng.integers(1, 13))
        # Win-probability-shaped payoffs, plus some wide-range ones.
        a = rng.random((m, n)) if trial % 2 else rng.normal(scale=5.0, size=(m, n))
        eq = solve(a)

        assert eq.duality_gap < 1e-7, f"trial {trial}: duality gap {eq.duality_gap}"
        assert eq.row_strategy.sum() == pytest.approx(1.0)
        assert eq.col_strategy.sum() == pytest.approx(1.0)
        assert (eq.row_strategy >= 0).all()
        assert (eq.col_strategy >= 0).all()

        assert eq.row_ev.max() <= eq.value + 1e-7, f"trial {trial}: row can deviate upward"
        assert eq.col_ev.min() >= eq.value - 1e-7, f"trial {trial}: column can deviate downward"
        assert eq.max_exploitability() < 1e-7

        # Support actions are exactly indifferent.
        for i in eq.row_support():
            assert eq.row_ev[i] == pytest.approx(eq.value, abs=1e-7)
            assert eq.row_ev_loss[i] == pytest.approx(0.0, abs=1e-7)
        for j in eq.col_support():
            assert eq.col_ev[j] == pytest.approx(eq.value, abs=1e-7)
            assert eq.col_ev_loss[j] == pytest.approx(0.0, abs=1e-7)

        # value_of agrees with the LP value.
        assert value_of(a, eq.row_strategy, eq.col_strategy) == pytest.approx(eq.value, abs=1e-7)


def test_best_response_picks_the_top_ev_action() -> None:
    payoff = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
    i, ev = best_response(payoff, np.array([1.0, 0.0]))
    assert i == 2
    assert ev == pytest.approx(0.8)
    i, ev = best_response(payoff, np.array([0.0, 1.0]))
    assert i == 0
    assert ev == pytest.approx(0.9)


def test_ev_loss_is_never_negative_and_is_zero_somewhere() -> None:
    rng = np.random.default_rng(99)
    for _ in range(50):
        a = rng.random((int(rng.integers(2, 9)), int(rng.integers(2, 9))))
        eq = solve(a)
        assert (eq.row_ev_loss >= 0).all()
        assert (eq.col_ev_loss >= 0).all()
        assert eq.row_ev_loss.min() == pytest.approx(0.0, abs=1e-7)
        assert eq.col_ev_loss.min() == pytest.approx(0.0, abs=1e-7)


def test_error_type_is_exposed() -> None:
    assert issubclass(EquilibriumError, RuntimeError)


# ---------------------------------------------------------------------------
# The Bayesian game: the opponent knows their own spread and we do not
# ---------------------------------------------------------------------------


def test_private_information_is_worth_something() -> None:
    """The reason `solve_bayesian` exists rather than averaging and calling `solve`.

    Two classes with opposite best replies. Averaging the matrices makes the opponent
    commit before learning their own spread, which they never have to do, and the value
    that comes out is optimistic for us. Here it is 0.25 against a true 0.
    """
    first = np.array([[1.0, 0.0], [0.0, 0.0]])
    second = np.array([[0.0, 0.0], [0.0, 1.0]])
    weights = np.array([0.5, 0.5])

    averaged = solve(0.5 * first + 0.5 * second)
    informed = solve_bayesian([first, second], weights)

    assert averaged.value == pytest.approx(0.25)
    assert informed.value == pytest.approx(0.0)
    assert informed.duality_gap < 1e-9


def test_information_never_helps_us() -> None:
    """A structural check across random games: the Bayesian value cannot exceed the
    averaged one. If it ever did, the LP would be giving the opponent less than they have.
    """
    rng = np.random.default_rng(11)
    for _ in range(40):
        k = int(rng.integers(2, 5))
        m, n = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        mats = [rng.random((m, n)) for _ in range(k)]
        weights = rng.random(k) + 0.05
        weights /= weights.sum()
        averaged = solve(sum(w * mat for w, mat in zip(weights, mats, strict=True)))
        informed = solve_bayesian(mats, weights)
        assert informed.value <= averaged.value + 1e-9
        assert informed.duality_gap < 1e-7


def test_one_class_reduces_to_the_ordinary_game() -> None:
    """With nothing hidden the two solvers must agree, or one of them is wrong."""
    rng = np.random.default_rng(5)
    for _ in range(20):
        payoff = rng.random((4, 5))
        plain = solve(payoff)
        single = solve_bayesian([payoff], np.array([1.0]))
        assert single.value == pytest.approx(plain.value, abs=1e-9)
        assert single.row_ev_loss[single.row_support()] == pytest.approx(0.0, abs=1e-9)


def test_the_strategies_are_distributions_supported_on_best_replies() -> None:
    rng = np.random.default_rng(17)
    mats = [rng.random((3, 4)) for _ in range(3)]
    weights = np.array([0.5, 0.3, 0.2])
    eq = solve_bayesian(mats, weights)

    assert float(eq.row_strategy.sum()) == pytest.approx(1.0)
    assert eq.row_ev_loss[eq.row_support()] == pytest.approx(0.0, abs=1e-9)
    for strategy, loss in zip(eq.col_strategies, eq.col_ev_loss, strict=True):
        assert float(strategy.sum()) == pytest.approx(1.0)
        assert (strategy >= 0).all()
        assert loss[np.flatnonzero(strategy > 1e-6)] == pytest.approx(0.0, abs=1e-9)
    assert eq.col_marginal is not None
    assert float(eq.col_marginal.sum()) == pytest.approx(1.0)


def test_the_value_is_what_the_strategies_actually_earn() -> None:
    """Recomputed from the strategies rather than taken from the solver's objective."""
    rng = np.random.default_rng(23)
    mats = [rng.random((4, 3)) for _ in range(3)]
    weights = np.array([0.6, 0.25, 0.15])
    eq = solve_bayesian(mats, weights)
    earned = sum(
        w * float(eq.row_strategy @ mat @ y)
        for w, mat, y in zip(eq.weights, mats, eq.col_strategies, strict=True)
    )
    assert earned == pytest.approx(eq.value, abs=1e-7)


def test_classes_may_offer_the_opponent_different_action_sets() -> None:
    """What the no-team-sheet future needs: a hidden type can change the action set.

    If moves stop being public, each type brings its own legal moves, so the matrices
    have different widths. The LP does not care, and the observable marginal stops being
    defined -- which the result says rather than faking.
    """
    mats = [np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0.5], [0.25]])]
    eq = solve_bayesian(mats, np.array([0.7, 0.3]))
    assert eq.col_marginal is None
    assert [len(y) for y in eq.col_strategies] == [2, 1]
    assert eq.duality_gap < 1e-9


def test_mismatched_row_counts_are_refused() -> None:
    """Our own action set is the same whatever the opponent's spread is."""
    with pytest.raises(ValueError, match="same actions"):
        solve_bayesian([np.zeros((2, 2)), np.zeros((3, 2))], np.array([0.5, 0.5]))


def test_wrong_weight_count_is_refused() -> None:
    with pytest.raises(ValueError, match="weights"):
        solve_bayesian([np.zeros((2, 2))], np.array([0.5, 0.5]))
