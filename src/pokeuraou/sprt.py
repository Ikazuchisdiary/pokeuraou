"""Stopping a paired match as soon as it has answered, instead of at a fixed count.

Every judgement this project makes is a match, and a match is 12,000 games -- a couple of
hours of the whole machine -- whatever the answer turns out to be. Most of those games
confirm an answer that was already plain: IKA-66's width 48 against 24 came back +21.7 Elo,
and a difference that size is visible long before the 12,000th game. The number of changes
that can be tried is set by the price of a match, and the price is set by the worst case.

Stockfish's fishtest had the same problem and answers it with a sequential probability
ratio test. Two hypotheses are fixed before the first game -- H0: the change is worth
`elo0`, H1: it is worth `elo1` -- and after every game pair the log-likelihood ratio of the
two is compared with Wald's bounds, ln(beta / (1 - alpha)) and ln((1 - beta) / alpha). The
run stops the first time it leaves them. A change worth `elo0` or less is passed with
probability at most `alpha`, one worth `elo1` or more is failed with probability at most
`beta`, and in between either answer is allowed.

**The unit is the pair, not the game.** A queued match plays game k twice with the arms
swapped (`tools/paired_result.py`), so what is independent is a pair's score for the tested
arm: 0 (lost both seats), 1/2 (split) or 1 (won both). That is fishtest's pentanomial model
with the draws taken out -- two games without draws have three joint outcomes, not five --
and it is the same three-point distribution fishtest's own code handles as L/D/W. Counting
the 2N games as independent would overstate the variance of a design where most pairs
split and the pairing is the reason the interval is narrow.

**The ratio is the generalised one.** Neither hypothesis says what the three outcome
probabilities are, only what their mean is, so each is represented by the most likely
distribution with that mean given the pairs seen so far. That maximum has a closed form up
to one scalar, the root of a "secular equation" (fishtest's `LLRcalc.MLE_expected`, after
Proposition 1.1 of the note it cites). With three outcomes the equation is a quadratic, so
it is solved exactly here instead of by a root finder, and a whole run is one vectorised
call. An outcome not yet seen counts as 1e-3 of a pair, as fishtest regularises, so that
the first pairs do not produce infinities.

**What it does not give: a size.** A run stopped at the first crossing stops more often on
a lucky stretch, so the estimate it ends with is biased away from zero, the more so the
earlier it stopped. A decision -- ship it or not -- is what the test answers. Anything that
needs the number itself (a conversion rate such as IKA-66's +2.50 a doubling, or a triangle
closed from three matches) keeps the fixed count.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: What a pair can score for the tested arm: lost both seats, split them, won both.
PAIR_SCORES = (0.0, 0.5, 1.0)

#: fishtest's `LLRcalc.regularize`: an outcome never seen counts as this much of a pair.
REGULARIZE = 1e-3

#: The files a match writes its games to: `match_queue` workers, and the older seeded runs
#: `tools/paired_result.py` also reads.
GAME_FILES = ("games-worker*.jsonl", "games-seed*.jsonl")


def expected_score(elo: float) -> float:
    """The logistic scale every Elo in this project is quoted on: +10 is 51.44%."""
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def elo_of(score: float) -> float:
    """The inverse of `expected_score`, infinite at 0 and 1."""
    if not 0.0 < score < 1.0:
        return math.copysign(math.inf, score - 0.5)
    return -400.0 * math.log10(1.0 / score - 1.0)


def bounds(alpha: float, beta: float) -> tuple[float, float]:
    """Wald's thresholds: at or below the first H0 is accepted, at or above the second H1."""
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def _regularised(counts: np.ndarray | Sequence[float]) -> np.ndarray:
    c = np.asarray(counts, dtype=np.float64)
    if c.shape[-1] != len(PAIR_SCORES):
        raise ValueError(f"counts must end in an axis of {len(PAIR_SCORES)}, got {c.shape}")
    return np.where(c == 0, REGULARIZE, c)


def _log_likelihood(freqs: np.ndarray, s: float) -> np.ndarray:
    """Per pair: the log-likelihood of the most likely distribution with mean `s`, less that
    of the empirical distribution `freqs` itself -- so never above zero, and zero exactly
    when the empirical mean already is `s`.

    The maximiser is `p_i = f_i / (1 + x d_i)` with `d_i = a_i - s`, where `x` is the root of
    `sum_i f_i d_i / (1 + x d_i) = 0` on the interval that keeps every `1 + x d_i` positive,
    `(-1 / d_2, -1 / d_0)`. The left side falls monotonically from +inf to -inf across that
    interval, so the root is unique; multiplied by the product of the (positive)
    denominators it is `a x^2 + b x + c = 0`, and the root is whichever of the two lies
    inside.
    """
    if not 0.0 < s < 1.0:
        raise ValueError(f"a mean score must lie strictly inside (0, 1), got {s}")
    d = np.asarray(PAIR_SCORES) - s
    f0, f1, f2 = freqs[..., 0], freqs[..., 1], freqs[..., 2]
    a = d[0] * d[1] * d[2]
    b = f0 * d[0] * (d[1] + d[2]) + f1 * d[1] * (d[0] + d[2]) + f2 * d[2] * (d[0] + d[1])
    c = f0 * d[0] + f1 * d[1] + f2 * d[2]
    if abs(a) < 1e-15:
        # s = 1/2, which is H0 whenever elo0 is 0: the middle outcome drops out and the
        # equation is linear. `b` is -(f_0 + f_2) / 4 there, never zero once regularised.
        x = -c / b
    else:
        root = np.sqrt(np.maximum(b * b - 4.0 * a * c, 0.0))
        # The two roots without the cancellation of the textbook formula: q / a and c / q.
        q = -0.5 * (b + np.where(b >= 0.0, root, -root))
        with np.errstate(divide="ignore", invalid="ignore"):
            near = np.where(q != 0.0, c / np.where(q != 0.0, q, 1.0), 0.0)
            far = q / a
        lo, hi = -1.0 / d[2], -1.0 / d[0]
        x = np.where((near > lo) & (near < hi), near, far)
    # `1 + x d_i` is positive on the interval; the clip only keeps a rounding error at its
    # very edge (a regularised cell the root nearly empties) from reaching log(0).
    shifted = np.maximum(x[..., None] * d, np.nextafter(-1.0, 0.0))
    return -(freqs * np.log1p(shifted)).sum(axis=-1)


def llr(counts: np.ndarray | Sequence[float], elo0: float, elo1: float) -> np.ndarray:
    """The generalised log-likelihood ratio of H1 (mean score at `elo1`) over H0 (at `elo0`).

    `counts` is `(..., 3)`: pairs the tested arm lost, split and won. Vectorised over the
    leading axes, so a run's whole path -- the cumulative counts after each pair -- is one
    call. The same number fishtest's `LLR_logistic(elo0, elo1, [lost, split, won])` gives.
    """
    c = _regularised(counts)
    n = c.sum(axis=-1)
    f = c / n[..., None]
    return n * (
        _log_likelihood(f, expected_score(elo1)) - _log_likelihood(f, expected_score(elo0))
    )


def llr_normal(counts: np.ndarray | Sequence[float], elo0: float, elo1: float) -> np.ndarray:
    """fishtest's `LLR_alt2`: the same ratio to second order in the effect,
    `n (s1 - s0) (2 mean - s0 - s1) / (2 var)`. Kept as a check on `llr`, which it should
    shadow closely at the sizes this project tests."""
    c = _regularised(counts)
    n = c.sum(axis=-1)
    f = c / n[..., None]
    values = np.asarray(PAIR_SCORES)
    mean = (f * values).sum(axis=-1)
    var = (f * (values - mean[..., None]) ** 2).sum(axis=-1)
    s0, s1 = expected_score(elo0), expected_score(elo1)
    return n * (s1 - s0) * (2.0 * mean - s0 - s1) / (2.0 * var)


def slot(score: float) -> int:
    """Which of the three outcomes a pair score is."""
    for index, value in enumerate(PAIR_SCORES):
        if score == value:
            return index
    raise ValueError(f"a pair scores 0, 1/2 or 1 for the tested arm, not {score!r}")


def cumulative_counts(scores: Sequence[float]) -> np.ndarray:
    """`(n, 3)`: the lost / split / won counts after each pair, in the order given."""
    slots = np.fromiter((slot(s) for s in scores), dtype=np.int64, count=len(scores))
    return np.eye(len(PAIR_SCORES), dtype=np.int64)[slots].cumsum(axis=0)


@dataclass(frozen=True)
class Stop:
    """Where a sequence of pairs left the bounds, or where it ran out without doing so."""

    #: Pairs consumed: the one that crossed included, or every pair when none did.
    pairs: int
    #: "H1" (worth `elo1`), "H0" (worth `elo0`), or None when the pairs ran out first.
    decision: str | None
    llr: float


def first_crossing(path: np.ndarray, lower: float, upper: float) -> Stop:
    """The first index at which a log-likelihood path leaves `(lower, upper)`."""
    path = np.asarray(path, dtype=np.float64)
    if path.size == 0:
        return Stop(0, None, 0.0)
    crossed = np.flatnonzero((path >= upper) | (path <= lower))
    if crossed.size == 0:
        return Stop(int(path.size), None, float(path[-1]))
    at = int(crossed[0])
    return Stop(at + 1, "H1" if path[at] >= upper else "H0", float(path[at]))


def replay(
    scores: Sequence[float],
    elo0: float,
    elo1: float,
    *,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> Stop:
    """Where a test registered before the first pair would have stopped this sequence."""
    if not scores:
        return Stop(0, None, 0.0)
    lower, upper = bounds(alpha, beta)
    return first_crossing(llr(cumulative_counts(scores), elo0, elo1), lower, upper)


def simulate(
    probabilities: Sequence[float],
    elo0: float,
    elo1: float,
    *,
    alpha: float = 0.05,
    beta: float = 0.05,
    cap: int,
    runs: int,
    rng: np.random.Generator,
    block: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """`runs` independent tests on pairs drawn with `probabilities` (lost, split, won).

    Returns the pairs each consumed -- `cap` for one still undecided there -- and its
    decision: +1 for H1, -1 for H0, 0 for none. Paths are grown a block at a time and a run
    leaves as soon as it crosses, because the ratio is the expensive part and most runs
    stop long before the cap: evaluating every path to the end did four times the work for
    the same answer.
    """
    lower, upper = bounds(alpha, beta)
    eye = np.eye(len(PAIR_SCORES), dtype=np.int64)
    stops = np.full(runs, cap, dtype=np.int64)
    decisions = np.zeros(runs, dtype=np.int64)
    counts = np.zeros((runs, len(PAIR_SCORES)), dtype=np.int64)
    alive = np.arange(runs)
    step = 0
    while alive.size and step < cap:
        width = min(block, cap - step)
        draws = rng.choice(len(PAIR_SCORES), size=(alive.size, width), p=probabilities)
        path_counts = counts[alive][:, None, :] + eye[draws].cumsum(axis=1)
        path = llr(path_counts, elo0, elo1)
        crossed = (path >= upper) | (path <= lower)
        hit = crossed.any(axis=1)
        first = crossed.argmax(axis=1)
        done = alive[hit]
        stops[done] = step + first[hit] + 1
        decisions[done] = np.where(path[hit, first[hit]] >= upper, 1, -1)
        counts[alive] = path_counts[:, -1, :]
        alive = alive[~hit]
        step += width
    return stops, decisions


@dataclass
class Sprt:
    """A test registered before its first pair and fed one pair at a time."""

    elo0: float
    elo1: float
    alpha: float = 0.05
    beta: float = 0.05
    counts: list[int] = field(default_factory=lambda: [0, 0, 0])

    def __post_init__(self) -> None:
        if not self.elo1 > self.elo0:
            raise ValueError(f"elo1 must exceed elo0, got ({self.elo0}, {self.elo1})")
        if not (0.0 < self.alpha < 0.5 and 0.0 < self.beta < 0.5):
            raise ValueError(f"alpha and beta are error rates below 1/2: {self.alpha}, {self.beta}")
        self.lower, self.upper = bounds(self.alpha, self.beta)

    @property
    def pairs(self) -> int:
        return sum(self.counts)

    def add(self, score: float) -> None:
        self.counts[slot(score)] += 1

    @property
    def llr(self) -> float:
        return float(llr(self.counts, self.elo0, self.elo1)) if self.pairs else 0.0

    @property
    def decision(self) -> str | None:
        if not self.pairs:
            return None
        value = self.llr
        if value >= self.upper:
            return "H1"
        if value <= self.lower:
            return "H0"
        return None

    def registration(self) -> dict[str, Any]:
        """What was fixed before the first game, in the shape `sprt.json` keeps it."""
        return {
            "elo0": self.elo0,
            "elo1": self.elo1,
            "alpha": self.alpha,
            "beta": self.beta,
            "bounds": [self.lower, self.upper],
            "unit": "pair: the tested arm's mean score over the two seats of one game",
        }


# -- reading the pairs out of a match's records --------------------------------------------

_OUTCOME = re.compile(rb'"outcome":\s*(null|-?[0-9][0-9.eE+-]*)')
_PROVENANCE = b'"provenance":'

#: (outcome, gameIndex, provenance.seat, seatIndex) -- all a pair needs from a record.
Fields = tuple[Any, Any, Any, Any]


def fields_of_record(game: dict[str, Any]) -> Fields:
    """The reference reading: from a record parsed whole."""
    return (
        game.get("outcome"),
        game.get("gameIndex"),
        (game.get("provenance") or {}).get("seat"),
        game.get("seatIndex"),
    )


def fields_of_line(line: bytes) -> Fields:
    """The same four fields without parsing the record.

    A match record is about 120 KB and four fields of it matter here. `json.loads` on a
    12,000-game directory is 1.4 GB of Python objects: measured 4 MB/s at below-normal
    priority on a machine another match was saturating, against 245 MB/s for finding the
    fields directly. The layout is `write_game`'s: `outcome` is near the head, before the
    decisions, and `provenance` and the index come after everything else, so the tail from
    `"provenance":` is itself the end of a JSON object. `PairTail` checks the first record
    of every file against `fields_of_record`, so a change of layout fails loudly instead of
    pairing the wrong games. A line without that tail is parsed whole.
    """
    at = line.rfind(_PROVENANCE)
    head = _OUTCOME.search(line)
    if at < 0 or head is None:
        return fields_of_record(json.loads(line))
    try:
        # A `provenance` nested anywhere but at the end leaves unbalanced braces behind
        # it, so this fails loudly rather than reading the wrong seat.
        tail = json.loads(b"{" + line[at:])
    except json.JSONDecodeError as error:
        raise ValueError(
            "the record does not end with its provenance, so its fields cannot be read by "
            f"position ({error}): the record layout has changed"
        ) from error
    raw = head.group(1)
    return (
        None if raw == b"null" else float(raw),
        tail.get("gameIndex"),
        (tail.get("provenance") or {}).get("seat"),
        tail.get("seatIndex"),
    )


def entry(fields: Fields) -> tuple[int, int, float] | None:
    """`tools/paired_result.py`'s reading of one record: the game index, the side the tested
    arm sat on, and 1.0 if it won that seat -- or None for a record with no index (a run from
    before 2026-09-19) or no outcome.

    `provenance.seat` names the side ("<arm> = side 0"). `outcome` is side 0's result, so it
    is flipped for side 1, and a tie is nobody's win, as the worker's own tally counts it.
    """
    outcome, index, seat_label, seat_index = fields
    if outcome is None or index is None:
        return None
    tested_at_side0 = str(seat_label or "").endswith("side 0")
    if seat_index is not None and (int(seat_index) == 0) != tested_at_side0:
        raise ValueError(
            f"game {index}: seatIndex {seat_index} but provenance.seat {seat_label!r}"
        )
    outcome = float(outcome)
    won = outcome > 0.5 if tested_at_side0 else outcome < 0.5
    return int(index), 0 if tested_at_side0 else 1, float(won)


class PairTail:
    """The pairs of a match directory, read incrementally.

    One reader for a finished run and a running one: a replay reads everything once, and
    `StopWhenDecided` calls `poll` while the workers are still appending. Only complete
    lines are consumed -- a worker may be half way through writing one -- and each file is
    resumed from where the last poll stopped.
    """

    def __init__(self, directory: Path, *, verify: bool = True) -> None:
        self.directory = Path(directory)
        self.verify = verify
        #: game index -> {side the tested arm sat on: 1.0 if it won that seat}
        self.seats: dict[int, dict[int, float]] = defaultdict(dict)
        self.records = 0
        self.unindexed = 0
        self.ties = 0
        self._offsets: dict[Path, int] = {}
        self._verified: set[Path] = set()

    def poll(self) -> int:
        """Reads whatever was appended since the last call. Returns how many records."""
        seen = 0
        for pattern in GAME_FILES:
            for path in sorted(self.directory.glob(pattern)):
                seen += self._read(path)
        return seen

    def _read(self, path: Path) -> int:
        offset = self._offsets.get(path, 0)
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return 0
        seen = 0
        for line in chunk[:end].split(b"\n"):
            if not line.strip():
                continue
            fields = fields_of_line(line)
            if self.verify and path not in self._verified:
                reference = fields_of_record(json.loads(line))
                if fields != reference:
                    raise ValueError(
                        f"{path}: reading the fields directly gave {fields}, parsing the "
                        f"record gave {reference}. The record layout has changed; fix "
                        "`fields_of_line` before trusting any pair read from it."
                    )
                self._verified.add(path)
            seen += 1
            self.records += 1
            if fields[0] == 0.5:
                self.ties += 1
            found = entry(fields)
            if found is None:
                self.unindexed += 1
                continue
            index, side, won = found
            self.seats[index][side] = won
        self._offsets[path] = offset + end + 1
        return seen

    def pair(self, index: int) -> float | None:
        """The pair score of game `index`, or None while either seat is missing."""
        seats = self.seats.get(index)
        if not seats or len(seats) != 2:
            return None
        return (seats[0] + seats[1]) / 2.0

    def complete(self) -> list[tuple[int, float]]:
        """Every complete pair, in game-index order -- the order a queue hands games out,
        which is also the order `StopWhenDecided` consumes them in. Completion order would
        put the long games last and make an early stop a verdict on the short ones."""
        return [
            (index, (seats[0] + seats[1]) / 2.0)
            for index, seats in sorted(self.seats.items())
            if len(seats) == 2
        ]


class StopWhenDecided:
    """A `run_workers` monitor that ends a match as soon as its registered test decides.

    It consumes pairs strictly in game-index order and only once both seats of a game have
    been *resolved by the queue* -- finished (written, or discarded at the turn cap) or
    abandoned. That is what makes a live run stop at exactly the pair a replay of its
    records would: the prefix it has read is the prefix a replay reads, whatever order the
    workers happened to finish in. A game resolved with a seat missing is skipped, as
    `tools/paired_result.py` skips it.

    The queue is asked what is resolved *before* the files are read, so every index it
    reports has already been flushed to its worker's file: a worker writes the record
    before it tells the queue the index is done.
    """

    def __init__(self, directory: Path, test: Sprt, *, record: Path | None = None) -> None:
        self.tail = PairTail(directory)
        self.test = test
        self.record = record
        self.next_game = 0
        self.skipped = 0
        self.trail: list[float] = []
        self.stopped_at: int | None = None

    def __call__(self, queue: Any) -> str | None:  # noqa: ANN401 - a workqueue.WorkQueue
        if self.stopped_at is not None:
            return None
        resolved = queue.resolved()
        self.tail.poll()
        while 2 * self.next_game in resolved and 2 * self.next_game + 1 in resolved:
            score = self.tail.pair(self.next_game)
            self.next_game += 1
            if score is None:
                self.skipped += 1
                continue
            self.test.add(score)
            self.trail.append(round(self.test.llr, 4))
            decision = self.test.decision
            if decision is not None:
                self.stopped_at = self.test.pairs
                self.save()
                return (
                    f"SPRT({self.test.elo0:+g}, {self.test.elo1:+g}) accepted {decision} "
                    f"after {self.test.pairs} pairs ({2 * self.test.pairs} games), "
                    f"LLR {self.test.llr:+.3f}"
                )
        return None

    def state(self) -> dict[str, Any]:
        lost, split, won = self.test.counts
        return {
            "registered": self.test.registration(),
            "decision": self.test.decision,
            "pairs": self.test.pairs,
            "games": 2 * self.test.pairs,
            "counts": {"lost": lost, "split": split, "won": won},
            "llr": self.test.llr,
            "skippedGames": self.skipped,
            "stoppedEarly": self.stopped_at is not None,
            "llrTrail": self.trail,
            "caution": "a sequentially stopped run's win rate is biased away from 50%; "
            "quote the decision, and play a fixed count for a size",
        }

    def save(self) -> None:
        if self.record is not None:
            self.record.write_text(json.dumps(self.state(), indent=1) + "\n", encoding="utf-8")


__all__ = [
    "GAME_FILES",
    "PAIR_SCORES",
    "REGULARIZE",
    "PairTail",
    "Sprt",
    "Stop",
    "StopWhenDecided",
    "bounds",
    "cumulative_counts",
    "elo_of",
    "entry",
    "expected_score",
    "fields_of_line",
    "fields_of_record",
    "first_crossing",
    "llr",
    "llr_normal",
    "replay",
    "simulate",
    "slot",
]
