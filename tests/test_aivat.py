"""IKA-193: the chance correction of `tools/aivat.py` has mean zero, and finds the branch played.

The whole estimator rests on two facts, and each is checked against something other than
itself. The first is arithmetic: whatever the evaluator says, the correction averaged over
the port's own weights is zero -- including when some outcomes are mid-turn pauses, which
are scored neutrally. The second is about the records: a recorded game's every turn lands
on a branch of `Budget.exact()` bit for bit (or on a pause whose resumption does), so the
weights the correction subtracts are the ones the game drew from. The control for the
second is the matrix budget, which collapses the damage rolls: the same game re-resolved
with it must stop finding the branches, or the match proves nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import HP_SHARE
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import play_game
from pokeuraou.sprt import cumulative_counts, llr_normal
from pokeuraou.teams import load_roster
from tests._harness import load_tool

aivat = load_tool("aivat")


@pytest.mark.parametrize("seed", range(20))
def test_the_correction_averages_to_zero_over_the_ports_weights(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    values = rng.random(n)
    weights = rng.random(n) * rng.integers(1, 1000)  # unnormalised, as the port gives them
    paused = rng.random(n) < (0.3 if seed % 2 else 0.0)
    p = weights / weights.sum()
    mean = sum(
        p[k] * aivat.neutral_correction(values, weights, paused, k)[0] for k in range(n)
    )
    assert abs(mean) < 1e-12


def test_a_pause_corrects_nothing_and_a_branch_corrects_against_the_rest() -> None:
    values = [0.9, 0.1, 123.0]  # the pause's value is never read
    weights = [1.0, 3.0, 4.0]
    paused = [False, False, True]
    c, expected = aivat.neutral_correction(values, weights, paused, 2)
    assert c == 0.0
    assert expected == pytest.approx((0.9 * 1 + 0.1 * 3) / 4)
    c, _ = aivat.neutral_correction(values, weights, paused, 0)
    assert c == pytest.approx(0.9 - 0.3)


def _row(index: int, seat: int, outcome: float, correction: float) -> dict:
    return {
        "gameIndex": index,
        "seat": f"arm = side {seat}",
        "seatIndex": seat,
        "outcome": outcome,
        "C": correction,
    }


def test_one_game_in_both_seats_is_exactly_a_half_after_the_correction() -> None:
    """A null match plays the same game twice: side 0's luck is the tested arm's in one seat
    and its opponent's in the other, so the corrections cancel inside the pair."""
    rows = [_row(0, 0, 1.0, 0.37), _row(0, 1, 1.0, 0.37), _row(1, 0, 0.0, -0.2), _row(1, 1, 0.0, -0.2)]
    indices, raw, corrected = aivat.pair_scores(rows)
    assert list(indices) == [0, 1]
    assert list(raw) == [0.5, 0.5]
    assert np.allclose(corrected, 0.5, atol=1e-15)


def test_the_correction_is_taken_off_the_side_the_arm_sat_on() -> None:
    # The arm won on side 0 with +0.4 of luck, and lost on side 1 when side 0 had +0.4.
    _, raw, corrected = aivat.pair_scores([_row(3, 0, 1.0, 0.4), _row(3, 1, 1.0, 0.4)])
    assert raw[0] == 0.5
    assert corrected[0] == pytest.approx(((1 - 0.4) + (0 + 0.4)) / 2)
    _, _, corrected = aivat.pair_scores([_row(3, 0, 1.0, 0.4), _row(3, 1, 0.0, -0.1)])
    assert corrected[0] == pytest.approx(((1 - 0.4) + (1 - 0.1)) / 2)


def test_the_normal_test_on_three_point_scores_is_fishtests_alt2() -> None:
    """On raw scores the path is `sprt.llr_normal` (unregularised: every outcome is seen)."""
    rng = np.random.default_rng(7)
    scores = rng.choice([0.0, 0.5, 1.0], size=300, p=[0.2, 0.55, 0.25])
    path = aivat.llr_normal_path(scores, 0.0, 10.0)
    counts = cumulative_counts(list(scores))
    seen = (counts > 0).all(axis=1)
    assert np.allclose(path[seen], llr_normal(counts[seen], 0.0, 10.0), rtol=1e-9)


@pytest.fixture(scope="module")
def played():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    games = []
    for seed in (5, 11):
        record = play_game(
            reg, np.random.default_rng(seed), sheet[:4], [sheet[i] for i in (1, 0, 4, 5)],
            "test", search_limit=2, max_turns=40, sheets=(sheet, sheet),
        )
        games.append(json.loads(json.dumps(record.to_json(objective="hp-share", search_limit=2))))
    return reg, games


def test_every_played_turn_lands_on_an_exact_branch(played) -> None:  # noqa: ANN001
    reg, games = played
    statuses = []
    for game in games:
        terms = aivat.game_terms(reg, game, HP_SHARE.batch)
        moves = sum(1 for d in game["decisions"] if d["kind"] == "move")
        first = [s for s in terms.stages if s.stage == 0]
        assert len(first) == moves
        statuses += [s.status for s in terms.stages]
        for s in terms.stages:
            if s.status == "matched":
                assert s.c == pytest.approx(s.landed_value - s.expected, abs=1e-12)
    assert statuses and set(statuses) <= {"matched", "paused"}


def test_a_record_without_its_final_position_corrects_the_last_turn_by_its_winner(played) -> None:  # noqa: ANN001
    """Records before IKA-87 have no `finalPosition`; the outcome decides the last turn's
    term exactly, because every branch the game can have ended on scores as that result."""
    reg, games = played
    decided = [game for game in games if game.get("outcome") is not None]
    assert decided
    for game in decided:
        with_final = aivat.game_terms(reg, game, HP_SHARE.batch)
        old = {k: v for k, v in game.items() if k not in ("finalPosition", "endReason")}
        without = aivat.game_terms(reg, old, HP_SHARE.batch)
        assert without.stages[-1].status == "by-outcome"
        assert with_final.stages[-1].status == "matched"
        assert without.total == pytest.approx(with_final.total, abs=1e-12)


def test_the_matrix_budget_does_not_find_the_branches(played) -> None:  # noqa: ANN001
    """The control: collapse the damage rolls and the played positions stop being branches.
    Were the matching loose (say, on who is standing), this would pass as well as the above."""
    reg, games = played
    statuses = [
        s.status
        for game in games
        for s in aivat.game_terms(reg, game, HP_SHARE.batch, budget=Budget.matrix()).stages
    ]
    assert statuses.count("mismatch") >= len(statuses) // 3
