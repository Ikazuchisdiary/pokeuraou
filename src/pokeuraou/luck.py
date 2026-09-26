"""AIVAT inside a match worker (IKA-193 stage 3): the luck of a game, from what it computed.

`tools/aivat.py` takes the luck out of a finished match after the fact: it re-reads every
turn of every record, re-resolves it with `Budget.exact()`, finds the branch that was
played by bit equality and, for the action term, resolves every pair of the two mixtures
exactly. That costs about as much CPU as the games (IKA-193 §6). A worker already holds
most of those numbers when it plays:

* the chance term. `_advance_turn` draws each turn from the exact weights and knows the
  index it drew, so nothing has to be matched. What it lacks is the value of every branch;
  `saw_turn` opens the same turn once more with `full=True` and scores its outcomes with
  the ledger's evaluator -- one cell of leaves a turn. A turn stopped for a mid-turn
  replacement is scored neutrally (`neutral_correction`) and the rest of the turn, which
  `_advance` draws from the resumed turn it already holds, is its own stage (`saw_resumed`,
  no extra port call). Unlike the after-the-fact reading, the worker knows WHICH pause it
  drew when two stand at one position (IKA-193 §1.1), so it resumes that one alone.
* the action term. Q(a, b) is read off the search's own matrices (`saw_mixtures`), side
  0's win probability, averaged over the matrices that hold the pair: side 0's search over
  its menu, side 1's over its own, and under a hidden bench each belief's matrices averaged
  with its completion weights. Any fixed Q keeps the term's mean at zero; a decision whose
  mixtures put weight on a pair no matrix holds is left uncorrected, which is decided before
  the draw and so is unbiased too. No leaf is scored for it.

Nothing is gathered outside `collecting`, and the hooks return at once there: the games a
worker plays are the same games with or without it (the hooks draw no random number and
leave the leaves' state alone -- the ledger scores through its own evaluator object).

The correction is in side 0's units, like `outcome`: a match subtracts it where the tested
arm sat on side 0 and adds it where it sat on side 1 (`tools/aivat.py arm_score`).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from . import timing

#: The burn-in of the normal-approximation GSPRT on corrected pair scores, registered
#: before stage 3 was written (records/IKA-193.md §11.0): IKA-193 §3.3 found the test
#: stopping at the second pair without one (alpha and beta 6-28%), 4.1-5.7% with 30. The 30
#: was chosen after looking at stage 1's bootstrap, which the record says.
BURN_IN = 30


def neutral_correction(
    values: Sequence[float],
    weights: Sequence[float],
    paused: Sequence[bool],
    landed: int | None,
) -> tuple[float, float]:
    """(c, E): one chance stage's correction and the expectation it subtracts.

    `values[k]` is V of outcome k (read only where `paused[k]` is False), `weights` the
    port's weights (normalised here, as `_sample_index` normalises them), `landed` the
    index the game drew. A pause is scored at `E`, the weighted mean of the non-pause
    outcomes, so the scoring is fixed before the draw and `sum_k p_k c(k) = 0` exactly:
    a pause landing corrects nothing, anything else corrects by `V - E`.
    """
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    mask = ~np.asarray(paused, dtype=bool)
    total = float(w.sum())
    if total <= 0 or not mask.any():
        return 0.0, math.nan
    p = w / total
    kept = float(p[mask].sum())
    expected = float((p[mask] * v[mask]).sum() / kept) if kept > 0 else math.nan
    if landed is None or not mask[landed] or kept <= 0:
        return 0.0, expected
    return float(v[landed]) - expected, expected


def action_correction(
    rows: Sequence[str],
    own: Sequence[float],
    cols: Sequence[str],
    foe: Sequence[float],
    drawn: tuple[str, str],
    matrices: Sequence[tuple[Sequence[str], Sequence[str], np.ndarray]],
) -> tuple[str, float, int]:
    """(status, c, pairs): the action term of one move decision.

    `own` is side 0's mixture over `rows`, `foe` side 1's over `cols`, `drawn` the pair the
    game played. `matrices` are (rows, columns, side 0's value) as the searches built them;
    Q of a pair is the mean over those that hold it. Status "pure" (one pair, nothing drawn),
    "uncovered" (a pair of the support is in no matrix: left at zero), "corrected".
    """
    support_a = [(rows[i], float(p)) for i, p in enumerate(own) if p > 0]
    support_b = [(cols[j], float(p)) for j, p in enumerate(foe) if p > 0]
    pairs = len(support_a) * len(support_b)
    if pairs <= 1:
        return "pure", 0.0, pairs
    looked = [
        ({a: i for i, a in enumerate(r)}, {b: j for j, b in enumerate(c)}, m)
        for r, c, m in matrices
    ]
    q = np.zeros((len(support_a), len(support_b)), dtype=np.float64)
    for i, (a, _pa) in enumerate(support_a):
        for j, (b, _pb) in enumerate(support_b):
            found = [
                float(m[ri[a], ci[b]]) for ri, ci, m in looked if a in ri and b in ci
            ]
            if not found:
                return "uncovered", 0.0, pairs
            q[i, j] = sum(found) / len(found)
    at_a = [i for i, (a, _p) in enumerate(support_a) if a == drawn[0]]
    at_b = [j for j, (b, _p) in enumerate(support_b) if b == drawn[1]]
    if len(at_a) != 1 or len(at_b) != 1:
        return "undrawn", 0.0, pairs
    pa = np.asarray([p for _a, p in support_a])
    pb = np.asarray([p for _b, p in support_b])
    expected = float(pa @ q @ pb) / float(pa.sum() * pb.sum())
    return "corrected", float(q[at_a[0], at_b[0]]) - expected, pairs


def matrix_of(result: Any) -> tuple[list[str], list[str], np.ndarray] | None:  # noqa: ANN401
    """A search's matrix as (rows, columns, side 0's value): a `SearchResult`'s payoff over
    its menus, or a `BeliefResult`'s completions averaged with their weights."""
    node = getattr(result, "node_payoff", None)
    if node is not None:
        row, col, built, weights = node
        w = np.asarray(weights, dtype=np.float64)
        if not len(built) or w.sum() <= 0:
            return None
        payoff = np.tensordot(w / w.sum(), np.stack([np.asarray(m) for m in built]), axes=1)
        rows, cols = row, col
    else:
        payoff = getattr(result, "payoff", None)
        rows, cols = getattr(result, "ours", None), getattr(result, "theirs", None)
        if payoff is None or rows is None or cols is None:
            return None
    payoff = np.asarray(payoff, dtype=np.float64)
    if payoff.shape != (len(rows), len(cols)):
        return None
    return [a.to_choice() for a in rows], [b.to_choice() for b in cols], payoff


class Ledger:
    """One game's luck, gathered while `play_game` runs inside `collecting`."""

    def __init__(self, evaluate: Callable[[list[Any]], np.ndarray], name: str) -> None:
        self.evaluate = evaluate
        self.name = name
        self.stages: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.decision = -1
        self.turn = 0
        self.stage = 0
        self.seconds = 0.0
        self.leaves = 0
        #: Extra port turns opened (one per move decision).
        self.turns = 0
        #: Of `seconds`: the extra port turns (and their positions' decoding), and the
        #: evaluator's calls.
        self.port_seconds = 0.0
        self.leaf_seconds = 0.0

    @property
    def chance(self) -> float:
        return float(sum(s["c"] for s in self.stages))

    @property
    def action(self) -> float:
        return float(sum(a["c"] for a in self.actions))

    def to_json(self) -> dict[str, Any]:
        return {
            "evaluator": self.name,
            "C": self.chance,
            "A": self.action,
            "seconds": round(self.seconds, 4),
            "leaves": self.leaves,
            "turns": self.turns,
            "portSeconds": round(self.port_seconds, 4),
            "leafSeconds": round(self.leaf_seconds, 4),
            "stages": self.stages,
            "actions": self.actions,
        }

    def _score(self, positions: list[Any]) -> np.ndarray:
        self.leaves += len(positions)
        if not positions:
            return np.zeros(0, dtype=np.float64)
        started = perf_counter()
        values = np.asarray(self.evaluate(positions), dtype=np.float64)
        self.leaf_seconds += perf_counter() - started
        return values

    def _stage(self, result: Any, landed: int, landed_position: Any) -> None:  # noqa: ANN401
        outcomes = list(result.outcomes or [])
        pauses = list(result.pauses or [])
        weights = [o.probability for o in outcomes] + [p.probability for p in pauses]
        paused = [False] * len(outcomes) + [True] * len(pauses)
        row: dict[str, Any] = {
            "d": self.decision, "t": self.turn, "s": "matched", "st": self.stage,
            "nb": len(outcomes), "np": len(pauses), "x": bool(result.exact), "c": 0.0,
        }
        if not weights or sum(weights) <= 0 or landed >= len(weights):
            row["s"] = "empty"
            self.stages.append(row)
            return
        if landed < len(outcomes) and landed_position is not None and (
            outcomes[landed].position.to_json() != landed_position.to_json()
        ):
            # The index the game drew is not the branch it played: say so, correct nothing.
            row["s"] = "mismatch"
            self.stages.append(row)
            return
        values = np.concatenate(
            [self._score([o.position for o in outcomes]), np.zeros(len(pauses))]
        )
        c, expected = neutral_correction(values, weights, paused, landed)
        row["c"] = c
        if landed >= len(outcomes):
            row["s"] = "paused"
        else:
            row["v"] = float(values[landed])
        if not math.isnan(expected):
            row["e"] = expected
        row["p"] = weights[landed] / sum(weights)
        self.stages.append(row)


_SINK: ContextVar[Ledger | None] = ContextVar("pokeuraou_luck", default=None)


@contextmanager
def collecting(evaluate: Callable[[list[Any]], np.ndarray], name: str) -> Iterator[Ledger]:
    """Gathers one game's luck inside the block, scored by `evaluate` (side 0's win
    probability of a true position). Nothing is gathered outside one."""
    ledger = Ledger(evaluate, name)
    token = _SINK.set(ledger)
    try:
        yield ledger
    finally:
        _SINK.reset(token)


def active() -> bool:
    return _SINK.get() is not None


def saw_mixtures(
    decision: int,
    turn: int,
    rows: Sequence[Any],
    own: Sequence[float],
    cols: Sequence[Any],
    foe: Sequence[float],
    chosen: Sequence[Any],
    solved: Sequence[Any],
    *,
    forced: bool = False,
) -> None:
    """A move decision's two mixtures, the pair drawn and the searches that priced it."""
    ledger = _SINK.get()
    if ledger is None:
        return
    started = perf_counter()
    ledger.decision, ledger.turn, ledger.stage = decision, turn, 0
    if forced:
        status, c, pairs = "forced", 0.0, 0
    else:
        matrices = [m for m in (matrix_of(r) for r in solved) if m is not None]
        status, c, pairs = action_correction(
            [a.to_choice() for a in rows], own, [b.to_choice() for b in cols], foe,
            (chosen[0].to_choice(), chosen[1].to_choice()), matrices,
        )
    ledger.actions.append({"d": decision, "s": status, "n": pairs, "c": c})
    ledger.seconds += perf_counter() - started


def saw_turn(
    reg: Any,  # noqa: ANN401
    pos: Any,  # noqa: ANN401
    chosen: Sequence[Any],
    counts: np.ndarray,
    index: int,
    landed: Any = None,  # noqa: ANN401
) -> None:
    """The turn `_advance_turn` drew outcome `index` of, from the exact weights `counts`;
    `landed` the position it played when that was a branch (a check on the index)."""
    ledger = _SINK.get()
    if ledger is None:
        # Counted under a profile only: the hook was reached and did nothing (IKA-193's
        # ABBA reads it as the new code's having run in the default games).
        timing.count("luck.off")
        return
    from . import port
    from .budget import Budget

    started = perf_counter()
    ledger.stage = 0
    ledger.turns += 1
    try:
        result = port.turn(reg, pos, list(chosen), Budget.exact(), full=True)
        ledger.port_seconds += perf_counter() - started
    except port.PortRefused:
        ledger.stages.append({"d": ledger.decision, "t": ledger.turn, "s": "refused",
                              "st": 0, "nb": 0, "np": 0, "x": True, "c": 0.0})
        ledger.seconds += perf_counter() - started
        return
    same = np.array_equal(
        np.asarray(result.branches + result.suspended, dtype=np.float64),
        np.asarray(counts, dtype=np.float64),
    )
    if not same:
        # Not the distribution the game drew from: leave it (decided before the draw).
        ledger.stages.append({"d": ledger.decision, "t": ledger.turn, "s": "weights",
                              "st": 0, "nb": len(result.branches),
                              "np": len(result.suspended), "x": bool(result.exact), "c": 0.0})
    else:
        ledger._stage(result, index, landed)
    ledger.seconds += perf_counter() - started


def saw_resumed(result: Any, index: int) -> None:  # noqa: ANN401
    """`_advance` drew outcome `index` of a turn resumed after a mid-turn replacement."""
    ledger = _SINK.get()
    if ledger is None:
        return
    started = perf_counter()
    ledger.stage += 1
    ledger._stage(result, index, None)
    ledger.seconds += perf_counter() - started


# -- reading a match's corrected pairs ------------------------------------------------------

_PROVENANCE = b'"provenance":'
_OUTCOME = re.compile(rb'"outcome":\s*(null|-?[0-9][0-9.eE+-]*)')


def seat_of_line(line: bytes) -> tuple[int, int, float, float, dict[str, Any]] | None:
    """(game index, side the tested arm sat on, its raw score, its corrected score, the
    ledger's block) of one record written with the ledger, or None for a record with no
    outcome or index. The correction is both terms, taken off in the tested arm's units."""
    at = line.rfind(_PROVENANCE)
    head = _OUTCOME.search(line)
    if at < 0 or head is None:
        game = json.loads(line)
        tail = {k: game.get(k) for k in ("provenance", "gameIndex", "aivat")}
        raw_outcome = game.get("outcome")
    else:
        tail = json.loads(b"{" + line[at:])
        raw_outcome = None if head.group(1) == b"null" else float(head.group(1))
    index = tail.get("gameIndex")
    if raw_outcome is None or index is None:
        return None
    terms = tail.get("aivat")
    if terms is None:
        raise ValueError(f"game {index}: no `aivat` block -- was the worker run with --aivat?")
    at0 = str((tail.get("provenance") or {}).get("seat", "")).endswith("side 0")
    z = float(raw_outcome)
    raw = float(z > 0.5) if at0 else float(z < 0.5)
    total = float(terms["C"]) + float(terms["A"])
    return int(index), 0 if at0 else 1, raw, raw - total if at0 else raw + total, terms


class CorrectedPairs:
    """A match directory's pairs, raw and corrected, read incrementally (`sprt.PairTail`'s
    reading, with the ledger's block)."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.seats: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
        self.seconds = 0.0
        self.game_seconds = 0.0
        self.records = 0
        self._offsets: dict[Path, int] = {}

    def poll(self) -> int:
        seen = 0
        for path in sorted(self.directory.glob("games-worker*.jsonl")):
            offset = self._offsets.get(path, 0)
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
            end = chunk.rfind(b"\n")
            if end < 0:
                continue
            for line in chunk[:end].split(b"\n"):
                if not line.strip():
                    continue
                seen += 1
                self.records += 1
                got = seat_of_line(line)
                if got is None:
                    continue
                index, side, raw, corrected, terms = got
                self.seats[index][side] = (raw, corrected)
                self.seconds += float(terms.get("seconds", 0.0))
                self.game_seconds += float(terms.get("gameSeconds", 0.0))
            self._offsets[path] = offset + end + 1
        return seen

    def pair(self, index: int) -> tuple[float, float] | None:
        seats = self.seats.get(index)
        if not seats or len(seats) != 2:
            return None
        return (
            (seats[0][0] + seats[1][0]) / 2.0,
            (seats[0][1] + seats[1][1]) / 2.0,
        )

    def complete(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(indices, raw, corrected) of every complete pair, in game-index order."""
        indices = sorted(i for i, s in self.seats.items() if len(s) == 2)
        pairs = [self.pair(i) for i in indices]
        return (
            np.asarray(indices, dtype=np.int64),
            np.asarray([p[0] for p in pairs], dtype=np.float64),
            np.asarray([p[1] for p in pairs], dtype=np.float64),
        )


class NormalTest:
    """The normal-approximation GSPRT on continuous pair scores, fed one pair at a time:
    `n (s1 - s0) (2 mean - s0 - s1) / (2 var)` with the running mean and population
    variance (fishtest's `LLR_alt2`, as `tools/aivat.py llr_normal_path`), held at zero over
    the first `burn` pairs and while the variance is zero."""

    def __init__(
        self, elo0: float, elo1: float, *, alpha: float = 0.05, beta: float = 0.05,
        burn: int = BURN_IN,
    ) -> None:
        from .sprt import bounds, expected_score

        if not elo1 > elo0:
            raise ValueError(f"elo1 must exceed elo0, got ({elo0}, {elo1})")
        self.elo0, self.elo1, self.alpha, self.beta, self.burn = elo0, elo1, alpha, beta, burn
        self.s0, self.s1 = expected_score(elo0), expected_score(elo1)
        self.lower, self.upper = bounds(alpha, beta)
        self.n = 0
        self.total = 0.0
        self.squares = 0.0

    def add(self, score: float) -> None:
        self.n += 1
        self.total += float(score)
        self.squares += float(score) * float(score)

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else math.nan

    @property
    def llr(self) -> float:
        if self.n <= self.burn or self.n == 0:
            return 0.0
        mean = self.total / self.n
        var = self.squares / self.n - mean * mean
        if var <= 1e-12:
            return 0.0
        return self.n * (self.s1 - self.s0) * (2 * mean - self.s0 - self.s1) / (2 * var)

    @property
    def decision(self) -> str | None:
        value = self.llr
        if value >= self.upper:
            return "H1"
        if value <= self.lower:
            return "H0"
        return None

    def registration(self) -> dict[str, Any]:
        return {
            "rule": "normal-approximation GSPRT on AIVAT-corrected pair scores "
            "(chance + action terms, IKA-193 stage 3)",
            "elo0": self.elo0,
            "elo1": self.elo1,
            "alpha": self.alpha,
            "beta": self.beta,
            "bounds": [self.lower, self.upper],
            "burnIn": self.burn,
            "unit": "pair: the tested arm's corrected mean score over the two seats of one game",
        }


class StopWhenCorrectedDecided:
    """`sprt.StopWhenDecided` on corrected pair scores (`--sprt-aivat`): pairs in game-index
    order once both seats are resolved by the queue, the normal test with its burn-in
    deciding, and the trinomial test on the raw pairs kept beside it for the record."""

    def __init__(self, directory: Path, test: NormalTest, trinomial: Any, *,  # noqa: ANN401
                 record: Path | None = None) -> None:
        self.pairs = CorrectedPairs(directory)
        self.test = test
        self.trinomial = trinomial
        self.record = record
        self.next_game = 0
        self.skipped = 0
        self.trail: list[float] = []
        self.stopped_at: int | None = None

    def __call__(self, queue: Any) -> str | None:  # noqa: ANN401 - a workqueue.WorkQueue
        if self.stopped_at is not None:
            return None
        resolved = queue.resolved()
        self.pairs.poll()
        while 2 * self.next_game in resolved and 2 * self.next_game + 1 in resolved:
            got = self.pairs.pair(self.next_game)
            self.next_game += 1
            if got is None:
                self.skipped += 1
                continue
            raw, corrected = got
            self.test.add(corrected)
            self.trinomial.add(raw)
            self.trail.append(round(self.test.llr, 4))
            decision = self.test.decision
            if decision is not None:
                self.stopped_at = self.test.n
                self.save()
                return (
                    f"corrected SPRT({self.test.elo0:+g}, {self.test.elo1:+g}) accepted "
                    f"{decision} after {self.test.n} pairs ({2 * self.test.n} games), "
                    f"LLR {self.test.llr:+.3f} (trinomial on the raw pairs "
                    f"{self.trinomial.llr:+.3f})"
                )
        return None

    def state(self) -> dict[str, Any]:
        lost, split, won = self.trinomial.counts
        return {
            "registered": self.test.registration(),
            "decision": self.test.decision,
            "pairs": self.test.n,
            "games": 2 * self.test.n,
            "llr": self.test.llr,
            "correctedMean": self.test.mean,
            "trinomial": {
                "registered": self.trinomial.registration(),
                "decision": self.trinomial.decision,
                "llr": self.trinomial.llr,
                "counts": {"lost": lost, "split": split, "won": won},
            },
            "skippedGames": self.skipped,
            "stoppedEarly": self.stopped_at is not None,
            "llrTrail": self.trail,
            "caution": "a sequentially stopped run's win rate is biased away from 50%; "
            "quote the decision, and play a fixed count for a size",
        }

    def save(self) -> None:
        if self.record is not None:
            self.record.write_bytes((json.dumps(self.state(), indent=1) + "\n").encode("utf-8"))


def summary(directory: Path, *, elo0: float = 0.0, elo1: float = 10.0,
            burn: int = BURN_IN) -> str:
    """What a finished match's corrected pairs say, beside the raw ones."""
    from .sprt import elo_of

    reader = CorrectedPairs(directory)
    reader.poll()
    _indices, raw, corrected = reader.complete()
    n = len(raw)
    if not n:
        return "  AIVAT: no complete pair"
    lines = [f"  AIVAT (chance + action terms), {n} complete pairs:"]
    for label, x in (("raw", raw), ("corrected", corrected)):
        mean = float(x.mean())
        half = 1.96 * float(x.std(ddof=1)) / math.sqrt(n) if n > 1 else math.nan
        lines.append(
            f"    {label:9s} mean {mean:.4f} +-{half:.4f}  Elo {elo_of(mean):+.1f} "
            f"[{elo_of(max(mean - half, 1e-6)):+.1f}, {elo_of(min(mean + half, 1 - 1e-6)):+.1f}]"
        )
    if n > 1 and raw.var() > 0:
        ratio = float(corrected.var(ddof=1) / raw.var(ddof=1))
        lines.append(f"    variance ratio {ratio:.3f} (pairs for the same interval x{ratio:.3f})")
    test = NormalTest(elo0, elo1, burn=burn)
    stop = None
    for x in corrected:
        test.add(float(x))
        if stop is None and test.decision is not None:
            stop = (test.n, test.decision, test.llr)
    lines.append(
        f"    normal GSPRT({elo0:+g},{elo1:+g}) on the corrected pairs in game-index order, "
        f"burn-in {burn}: "
        + (f"{stop[1]} at pair {stop[0]} (LLR {stop[2]:+.3f})" if stop else "undecided")
        + f"; LLR over all {n} pairs {test.llr:+.3f}"
    )
    ledgers = reader.game_seconds
    if ledgers > 0:
        lines.append(
            f"    cost: {reader.seconds:.1f} s in the ledger over {ledgers:.1f} s of games "
            f"({reader.seconds / ledgers:.1%})"
        )
    return "\n".join(lines)


__all__ = [
    "BURN_IN",
    "CorrectedPairs",
    "Ledger",
    "NormalTest",
    "StopWhenCorrectedDecided",
    "action_correction",
    "active",
    "collecting",
    "matrix_of",
    "neutral_correction",
    "saw_mixtures",
    "saw_resumed",
    "saw_turn",
    "seat_of_line",
    "summary",
]
