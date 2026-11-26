"""Swapping the seats must take the seat out of the number, exactly.

``tools/selection_check.py`` and ``tools/book_check.py`` played every game with our roster
at side 0. For the paired differences they report that is harmless -- both arms carry the
same seat term and it subtracts out -- but the calibration line compares a measured win
rate against an LP value which has no seat term in it at all, so whatever side 0 is worth
was being read as the solver's error. ``--mirror`` was the sharp case: our six against our
six is a symmetric game, the arm asserts 50.0%, and the deviation printed there WAS the
seat, which the tool said out loud and could not subtract because it held one seat.

These tests are written against the arithmetic rather than against games, for the reason
``tests/test_symmetry.py`` gives for preferring an identity to a win rate: 420 games put a
seat estimate at +-2.7 points, which was not enough to tell -0.07 from zero. A fake engine
with a *known* seat term makes the answer exact, so the assertions are equalities.

Two fake engines, both shapes this project has actually had:

- **side 0 always wins.** The largest possible seat term. Every matchup must come back at
  exactly 50% across the two seats and the reported gap must be exactly 100 points.
- **a Speed tie broken by side index.** Side 0 wins ties; everything else is decided by a
  strength that has no seat in it. Enumerated over every ordered pair of selections, the
  one-seat estimator reads ``0.5 + p/2`` where *p* is the tie rate, and the swapped total
  reads exactly 0.5 with a gap of exactly *p*. A nine-point version of this bug is in the
  project's history, and it was found by swapping.

The last test is the one that would have caught the defect in the state it was found in:
both tools have to actually route through the shared arithmetic. A fix that lands in a
module nobody calls is not a fix.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

from ._harness import load_tool

seats = load_tool("seats")

TOOLS = Path(__file__).resolve().parents[1] / "tools"

#: Stand-ins for a selected four. Nothing here reads them except the fake engines, which
#: is the point: the swap is about which argument goes to which side, not about Pokemon.
STRENGTHS = {"a": 0, "b": 0, "c": 1, "d": 2}
PICKS = tuple(STRENGTHS)


def side0_always_wins(_seat: int, _side0: str, _side1: str) -> float:
    """The largest seat term there is."""
    return 1.0


def tie_goes_to_side0(_seat: int, side0: str, side1: str) -> float:
    """Decided by strength, with the tie broken by side index -- the Speed-tie shape."""
    return 1.0 if STRENGTHS[side0] >= STRENGTHS[side1] else 0.0


def antisymmetric(_seat: int, side0: str, side1: str) -> float:
    """No seat term at all: equal strengths are a draw and a draw is not a win."""
    if STRENGTHS[side0] == STRENGTHS[side1]:
        return 0.5
    return 1.0 if STRENGTHS[side0] > STRENGTHS[side1] else 0.0


def played_over_all_pairs(engine, arm: str = "arm") -> seats.SeatTally:  # noqa: ANN001
    """Every ordered pair of selections, both seats, with the given engine.

    Enumeration rather than sampling: a uniform draw against a uniform draw is exactly
    this set with equal weight, so the answer is a number and not an estimate with an
    interval around it.
    """
    tally = seats.SeatTally(arm)
    for ours, theirs in itertools.product(PICKS, repeat=2):
        seats.play_paired(ours, theirs, engine, tally)
    return tally


def test_the_two_seats_are_the_same_matchup_mirrored() -> None:
    """Seat 0 gets (ours, theirs); seat 1 gets (theirs, ours). Nothing else moves."""
    calls: list[tuple[int, str, str]] = []

    def record(seat: int, side0: str, side1: str) -> float:
        calls.append((seat, side0, side1))
        return 1.0

    seats.play_paired("ours", "theirs", record)
    assert calls == [(0, "ours", "theirs"), (1, "theirs", "ours")], (
        "the swap must reseat the same two fours and change nothing else; drawing "
        "anything per seat puts a second difference into the comparison"
    )


def test_the_flip_lives_in_the_writer() -> None:
    """``outcome`` is always side 0's, so our result reads the other way in seat 1."""
    assert seats.our_win(1.0, 0) is True
    assert seats.our_win(1.0, 1) is False
    assert seats.our_win(0.0, 0) is False
    assert seats.our_win(0.0, 1) is True
    # A draw is not a win in either seat. Counting it as one in seat 1 only would be a
    # seat term of its own, invented by the bookkeeping.
    assert seats.our_win(0.5, 0) is False
    assert seats.our_win(0.5, 1) is False


def test_a_seat_that_always_wins_cancels_exactly() -> None:
    """The largest possible seat term leaves the total at exactly 50%."""
    tally = played_over_all_pairs(side0_always_wins)
    assert tally.played == [len(PICKS) ** 2, len(PICKS) ** 2]
    assert tally.seat_rate(0) == 1.0
    assert tally.seat_rate(1) == 0.0
    assert tally.rate == 0.5, (
        "a constant seat advantage has to cancel exactly in the total, not on average "
        "and not to within an interval"
    )
    assert tally.gap == 1.0, "and the size of it has to survive into the report"


def test_a_speed_tie_broken_by_side_index_cancels_and_is_reported() -> None:
    """The historical shape, priced exactly: one seat reads 0.5 + p/2, both read 0.5."""
    pairs = list(itertools.product(PICKS, repeat=2))
    ties = sum(1 for a, b in pairs if STRENGTHS[a] == STRENGTHS[b])
    tie_rate = ties / len(pairs)
    assert 0.0 < tie_rate < 1.0, "the fixture has to contain ties for this to test anything"

    tally = played_over_all_pairs(tie_goes_to_side0)
    # What the tools printed before the swap: our four at side 0 in every game.
    assert tally.seat_rate(0) == pytest.approx(0.5 + tie_rate / 2, abs=1e-12)
    assert tally.seat_rate(1) == pytest.approx(0.5 - tie_rate / 2, abs=1e-12)
    assert tally.rate == pytest.approx(0.5, abs=1e-12), (
        "uniform against uniform is a symmetric game and the swapped total has to say so "
        "whatever the seat is worth"
    )
    assert tally.gap == pytest.approx(tie_rate, abs=1e-12), (
        "the seat bias is the measurement, not a nuisance: cancelling it without "
        "printing it throws away what the second seat was paid for"
    )


def test_an_engine_with_no_seat_term_has_no_gap() -> None:
    """The control. Without a seat term the two seats agree and the total is the matchup."""
    tally = played_over_all_pairs(antisymmetric)
    assert tally.gap == 0.0
    assert tally.seat_rate(0) == tally.seat_rate(1) == tally.rate


def test_a_lopsided_matchup_is_not_mistaken_for_a_seat() -> None:
    """One strong four against one weak one, played both ways, is still a 100% arm.

    Run with the seat-biased engine on purpose: the bias only decides ties, and this
    matchup has none, so the gap has to read zero even though the machine has a seat in
    it. Averaging our wins with the opponent's would give 50% instead.
    """
    tally = seats.SeatTally("strong vs weak")
    for _ in range(10):
        seats.play_paired("d", "a", tie_goes_to_side0, tally)
    assert tally.rate == 1.0, "the swap must not average our wins with the opponent's"
    assert tally.gap == 0.0


def test_the_gap_reads_the_seat_in_a_run_that_is_nowhere_near_fifty() -> None:
    """The gap needs no mirror, which is what makes it usable on a field run.

    A deliberately lopsided book of matchups: our four is stronger in most of them and the
    arm is nowhere near 50%. The engine breaks ties by side index and nothing else, so the
    total has to come out at the honest matchup rate (a tie is half a win over the two
    seats) and the gap has to be the tie rate exactly, with none of the matchup in it.
    """
    matchups = [("d", "a"), ("d", "b"), ("c", "a"), ("a", "b"), ("b", "a"), ("a", "c")]
    tally = seats.SeatTally("field")
    for ours, theirs in matchups:
        seats.play_paired(ours, theirs, tie_goes_to_side0, tally)

    strict = sum(1 for a, b in matchups if STRENGTHS[a] > STRENGTHS[b])
    ties = sum(1 for a, b in matchups if STRENGTHS[a] == STRENGTHS[b])
    n = len(matchups)
    assert tally.rate == pytest.approx((strict + ties / 2) / n, abs=1e-12)
    assert tally.rate != pytest.approx(0.5, abs=1e-9), "the fixture has to be lopsided"
    assert tally.gap == pytest.approx(ties / n, abs=1e-12)


def test_an_unpaired_seat_is_called_out_rather_than_averaged() -> None:
    """A game cut off in one seat leaves the two seats over different matchups."""
    tally = seats.SeatTally("arm")
    tally.add(0, 1.0)
    tally.add(1, 0.0)
    tally.add(0, 1.0)
    tally.add(1, None)
    printed = "\n".join(tally.seat_lines())
    assert "対が揃っていない" in printed, printed
    assert tally.played == [2, 1]


def test_unfinished_games_are_dropped_from_their_own_seat() -> None:
    """A cut-off game leaves its partner in the other seat counted, and says so."""
    tally = seats.SeatTally("arm")
    assert tally.add(0, None) is None
    assert tally.add(1, 0.0) is True
    assert tally.played == [0, 1]
    assert tally.unfinished == [1, 0]
    assert tally.total_unfinished == 1
    assert tally.total_wins == 1


def test_the_seat_label_is_the_spelling_match_result_parses(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reader that already exists has to be able to read what these tools write.

    ``tools/match_result.py`` looks for a trailing ``side 0`` / ``side 1`` and, when it
    cannot find one, prints no side-0 line at all rather than take the other branch
    silently -- which it used to, quoting 51.7% where the arithmetic gives 54.8%. So the
    contract is checked against that parser and not against a copy of it here.
    """
    match_result = load_tool("match_result")
    directory = tmp_path / "seat-labels"
    directory.mkdir()
    # A gap far wider than four intervals, which is what makes match_result print the
    # side-0 line at all.
    rows = [
        {"seat": seats.seat_label("arm", 0), "played": 400, "gen2_wins": 360},
        {"seat": seats.seat_label("arm", 1), "played": 400, "gen2_wins": 40},
    ]
    (directory / "worker0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    monkeypatch.setattr(sys, "argv", ["match_result.py", str(directory)])
    match_result.main()
    out = capsys.readouterr().out
    assert "Side 0 scored" in out, (
        "match_result could not parse these seat labels, so it printed nothing rather "
        f"than guess:\n{out}"
    )
    # 360 at side 0 plus (400 - 40) from the seat where our arm sat at side 1.
    assert "Side 0 scored 90.0%" in out
    assert "total  400/800 = 50.00%" in out
