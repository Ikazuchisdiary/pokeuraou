"""Both seats, and the arithmetic that takes the seat out of a win rate.

``tools/generation_match.py`` plays every game twice -- once with the tested arm at side 0
and once with it at side 1 -- and ``tools/match_result.py`` reads the pair back: one row
per seat, a total that sums the two, and a warning when the seats disagree by more than
the interval can explain. That is the shape every win rate here is supposed to have,
because side 0 is not a neutral place to sit. A Speed tie broken by side index and a
residual phase applied in party order were both real, both worth points, and both were
found by swapping; ``tests/test_symmetry.py`` keeps them found.

``tools/selection_check.py`` and ``tools/book_check.py`` never swapped. Our roster sat at
side 0 in every game either of them ever played. For the *paired* differences they report
-- the selection cache against a uniform draw, same opponent, same spread class -- that is
harmless, because both arms carry the same seat term and it subtracts out. For the
calibration line it is not: that line compares ONE SEAT's win rate against an LP value
which has no seat term in it at all, and G31's "the selection cache is about 5 points
optimistic about our own side" is that comparison.

``--mirror`` is the sharp case. Our six against our six, uniform on both sides, is a
symmetric game, so the arm asserts 50.0% and any deviation was the seat -- which
``selection_check`` printed, correctly, as something it could not separate from the agents
because it held one seat only. This module is what lets it.

Three pieces, three lines of arithmetic each, which is why they have been written inline
every time they were wanted and how a seat label came to be read as the wrong arm:

- :func:`seat_label` spells a seat the way ``tools/match_result.py`` PARSES one, so a run
  written by these tools is readable by the reader that already exists. That reader looks
  for a trailing ``side 0`` / ``side 1`` and prints nothing rather than guess.
- :func:`our_win` is the flip, and it belongs in the writer. ``record.outcome`` is always
  SIDE 0's, so our arm's own result is ``outcome > 0.5`` in one seat and ``outcome < 0.5``
  in the other. ``match_result`` says the same thing from the other end: both seats report
  the named arm's wins, the total needs no flip, and a second one is how a sign gets lost.
- :class:`SeatTally` keeps the two seats apart and adds them only at the end. It never
  reports the total alone. The gap between the seats IS the seat bias, measured for free
  by a run that had to play both seats anyway, and a tool that cancels a quantity without
  printing it has thrown away the measurement it just paid for.

**What the swap does and does not buy.** Every matchup is played once with our four at
side 0 and once at side 1, and our result is read from our own side in both, so a seat
term that is constant across the two cancels EXACTLY in the total -- not on average, and
not to within an interval. What is left in a mirror's deviation from 50% is the selection
draw, which is a sample-size question with an interval attached, and no longer a defect.
The worked case is a Speed tie that goes to side 0: with probability *p* of a tie, one
seat reads ``0.5 + p/2`` and the other ``0.5 - p/2``; the total is 0.5 and the gap is *p*.
``tests/test_seat_swap.py`` asserts both of those exactly.

**And the gap needs no mirror.** Because the two seats play the same matchups, a machine
with no seat in it wins the same games in both -- ``f(B, A) = 1 - f(A, B)`` makes our rate
identical at either side whatever the two fours are -- so a lopsided matchup moves both
seats together and drops out of the difference. That is what makes the gap readable in a
field run and not only in ``--mirror``: unlike the deviation from 50%, it never had the
teams in it to begin with.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

#: Side 0 and side 1. Index `i` of a run is game `i // 2` in seat `i % 2` wherever a queue
#: hands the work out, which is `tools/match_queue.py`'s rule and the reason a seat and a
#: game together name one job.
SEATS = (0, 1)


def seat_label(arm: str, seat: int) -> str:
    """The seat's name, in the spelling ``tools/match_result.py`` parses.

    That reader asks whether a label ends in ``side 0`` or ``side 1`` and, when it cannot
    tell, prints no side-0 line at all rather than take the other branch silently -- which
    it used to, quoting 51.7% where the arithmetic gives 54.8%. Producing the spelling in
    one place is cheaper than agreeing on it in several.
    """
    if seat not in SEATS:
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    return f"{arm} = side {seat}"


def sides(
    ours: Sequence[Any], theirs: Sequence[Any], seat: int
) -> tuple[Sequence[Any], Sequence[Any]]:
    """``(side 0, side 1)`` for this seat: the same matchup, mirrored.

    Nothing about the matchup changes between the seats -- same four, same opponent, same
    spreads, same dice. Only which side of the field each stands on. Anything else drawn
    per seat would put a second difference into a comparison whose entire purpose is to
    isolate the first.
    """
    if seat not in SEATS:
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    return (ours, theirs) if seat == 0 else (theirs, ours)


def our_win(outcome: float, seat: int) -> bool:
    """Did OUR arm win, given side 0's outcome and which seat we held.

    ``record.outcome`` is side 0's, always. In seat 1 our four is side 1, so the same
    number means the opposite thing. The flip lives here, once, in the writer.
    """
    if seat not in SEATS:
        raise ValueError(f"seat must be 0 or 1, got {seat!r}")
    return outcome > 0.5 if seat == 0 else outcome < 0.5


def play_paired(
    ours: Sequence[Any],
    theirs: Sequence[Any],
    play: Callable[[int, Sequence[Any], Sequence[Any]], float | None],
    tally: SeatTally | None = None,
) -> list[float | None]:
    """One matchup, both seats. Returns side 0's outcome from each, in seat order.

    ``play(seat, side0, side1)`` plays one game and returns SIDE 0's outcome, or ``None``
    when it did not finish. The caller keeps its own per-seat bookkeeping inside that
    callback -- writing a row, timing the game -- and this function owns only the part
    that is easy to get wrong: which sequence goes to which side, and which way the
    outcome reads afterwards.

    A game that does not finish is dropped from the seat it was played in, and its partner
    in the other seat is kept. Dropping both would be a defensible choice; dropping one
    silently and calling the total paired would not, so the seats are counted separately
    and the totals say how many of each there were.
    """
    outcomes: list[float | None] = []
    for seat in SEATS:
        side0, side1 = sides(ours, theirs, seat)
        outcome = play(seat, side0, side1)
        outcomes.append(outcome)
        if tally is not None:
            tally.add(seat, outcome)
    return outcomes


class SeatTally:
    """Per-seat wins for one arm, and the total that has no seat in it.

    Wins counted here are always the ARM's, never side 0's: :func:`our_win` has already
    flipped seat 1. So the total is the arm's rate directly and needs no second flip,
    which is the rule ``tools/match_result.py`` states for the same reason.
    """

    def __init__(self, arm: str) -> None:
        self.arm = arm
        self.wins = [0, 0]
        self.played = [0, 0]
        self.unfinished = [0, 0]

    def add(self, seat: int, outcome: float | None) -> bool | None:
        """Record one game. Returns our win indicator, or ``None`` if it never finished."""
        if seat not in SEATS:
            raise ValueError(f"seat must be 0 or 1, got {seat!r}")
        if outcome is None:
            self.unfinished[seat] += 1
            return None
        won = our_win(float(outcome), seat)
        self.played[seat] += 1
        self.wins[seat] += int(won)
        return won

    @property
    def total_wins(self) -> int:
        return sum(self.wins)

    @property
    def total_played(self) -> int:
        return sum(self.played)

    @property
    def total_unfinished(self) -> int:
        return sum(self.unfinished)

    @staticmethod
    def _half(wins: int, played: int) -> float:
        if played <= 0:
            return float("nan")
        rate = wins / played
        return 1.96 * (rate * (1 - rate) / played) ** 0.5

    def seat_rate(self, seat: int) -> float:
        played = self.played[seat]
        return self.wins[seat] / played if played else float("nan")

    def seat_half(self, seat: int) -> float:
        return self._half(self.wins[seat], self.played[seat])

    @property
    def rate(self) -> float:
        """The arm's rate across both seats. Seat-free by construction, not on average."""
        return self.total_wins / self.total_played if self.total_played else float("nan")

    @property
    def half(self) -> float:
        return self._half(self.total_wins, self.total_played)

    @property
    def gap(self) -> float:
        """Our rate at side 0 minus our rate at side 1: the seat bias, in win-rate units.

        Equivalently ``win(side 0 | seat A) + win(side 0 | seat B) - 1``, which is how
        ``tools/cycle_match.py`` prints the same quantity, because our loss at side 1 is
        the opponent's win at side 0.

        **Zero is the requirement, and it does not need a mirror.** The two seats play the
        SAME matchups, so for a machine with no seat in it, winning with our four at side
        0 and winning with it at side 1 are the same event: ``f(B, A) = 1 - f(A, B)``
        makes our rate identical in both seats whatever the two fours are. A lopsided
        matchup moves both seats together and cancels out of the difference. What is left
        is the resolver, the search and the leaf, which is the thing being measured.

        The one caveat is an incomplete pair: a game cut off in one seat leaves its
        partner counted in the other, and then the two seats are no longer over the same
        matchups. :meth:`seat_lines` says so when the counts differ.
        """
        if not self.played[0] or not self.played[1]:
            return float("nan")
        return self.seat_rate(0) - self.seat_rate(1)

    def lines(self, *, width: int = 20) -> list[str]:
        """The arm's row and, under it, :meth:`seat_lines`.

        The arm's own row uses the normal approximation and the ``+-`` spelling
        ``tools/generation_match.py`` prints. A caller with its own house format for that
        row -- ``tools/book_check.py`` reports a Wilson interval -- prints it itself and
        calls :meth:`seat_lines` for the rest, so the breakdown reads the same in both.
        """
        out = [
            f"  {self.arm:>{width}}  {self.total_played:>6}  {self.rate * 100:6.1f}%  "
            f"+-{self.half * 100:.1f}   (打ち切り {self.total_unfinished})"
        ]
        out.extend(self.seat_lines())
        return out

    def seat_lines(self) -> list[str]:
        """The two seat rows and the gap between them, in `match_result`'s spelling."""
        out: list[str] = []
        for seat in SEATS:
            label = seat_label(self.arm, seat)
            if not self.played[seat]:
                out.append(f"      {label:<38} 完了 0 局")
                continue
            out.append(
                f"      {label:<38} {self.wins[seat]:>5}/{self.played[seat]:<5}"
                f" = {self.seat_rate(seat):6.2%}"
            )
        if self.played[0] and self.played[1]:
            out.append(
                f"      座席差 {self.gap * 100:+.1f} ポイント"
                "（side 0 に座ったときの勝率 - side 1 のとき。上の合計では相殺済み）"
            )
            if self.played[0] != self.played[1]:
                # Then the two seats are not over the same matchups any more, and the gap
                # has whatever the missing games were worth mixed into it.
                out.append(
                    "      ! 片方の席だけ打ち切りが出ていて対が揃っていないので、"
                    "この差には欠けた対戦の分が混じっている"
                )
            elif abs(self.gap) > 2 * self.half:
                out.append(
                    "      ! 両席は同じ対戦を裏返しただけなので、この差は機構のもの"
                    "（解決器・探索・評価）であって腕や相手のものではない"
                )
        return out
