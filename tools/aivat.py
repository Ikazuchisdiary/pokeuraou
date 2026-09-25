"""The luck of the dice taken out of a queued match's score (IKA-193, AIVAT's chance term).

A match scores each game by who won, and a win carries every roll of the game: the crits,
the misses, the damage rolls, the secondary effects. Every one of those rolls was drawn by
`selfplay._advance_turn` from weights the port enumerated with `Budget.exact()`, so what
each roll was *expected* to be worth is computable after the fact. For a fixed evaluator V
(side 0's win probability) and the turn the two players actually chose,

    c_t = V(where the turn landed) - sum_k p_k V(branch k)

has expectation exactly zero whatever V is, because p is the distribution the game drew
from. Subtracting sum_t c_t from the result leaves the mean score where it was and takes
out the part of the variance the rolls put there (Burch et al., AIVAT, AAAI 2018: the
chance term of their estimator; the action term is not here).

**What is corrected and what is not.**

* Every move decision's turn: the branches of `port.turn(..., Budget.exact(), full=True)`
  for the recorded `ownChosen` / `foeChosen`. The landed branch is found by bit equality
  of the position (`Position.to_json`) with the next record entry -- the next decision, or
  the record's `finalPosition` after the last turn. A turn that stops for a mid-turn
  replacement (a `selfswitch` decision) is two chance stages: the stop itself, whose
  outcome is scored *neutrally* (a pause is worth the weighted mean of the non-pause
  branches, so drawing one corrects nothing -- still zero-mean, since the neutral value is
  fixed before the draw), and the rest of the turn, resumed with the recorded replacement
  (`port.resume`, as `_self_switch_encoded` resumes it) and corrected like a turn.
* Not corrected: the draws inside a replacement phase (`resolve_replacements(rng=...)`, not
  enumerated by the port), the lead abilities before turn 1, the selection and the action
  draws from each side's mixture (AIVAT's action term, IKA-193 stage 2).

A decision whose landed position is in no branch (the engine moved since the record) is
counted and left uncorrected -- a correction of zero is always unbiased, but choosing it
*because* of which branch landed is not, which is why the count is printed.

    compute:  aivat.py compute <match dir> --out <jsonl> [--value M ...] [--shard k/n]
    report:   aivat.py report <jsonl> ... [--sprt 0 10] [--bootstrap 2000]

`compute` writes one line per game: its index, seat, outcome and the per-decision terms.
`report` pairs the games as `tools/paired_result.py` does and prints the raw and corrected
pair scores side by side: mean, SD, the variance ratio, the pairs a fixed-width interval
needs, a sequential test on each, and the bias checks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: What a decision's correction ended as.
#: matched   the turn's landed branch was found, and corrected
#: paused    the turn stopped for a mid-turn replacement; the stop itself corrected neutrally
#:           (zero), the rest of the turn as its own stage
#: mismatch  no branch (or pause) equals the next record entry: left at zero
#: refused   the port declined the turn or its resumption: left at zero
#: illegal   a recorded choice is not a legal action here any more: left at zero
#: no-next   the record has nothing to match the last turn against (no finalPosition)
#: empty     the turn had no outcome with weight (the game ended "unresolved")
STATUSES = ("matched", "paused", "mismatch", "refused", "illegal", "no-next", "empty")


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


@dataclass
class Stage:
    """One chance stage of one move decision."""

    decision: int
    turn: int
    status: str
    branches: int = 0
    pauses: int = 0
    exact: bool = True
    c: float = 0.0
    expected: float = math.nan
    landed_value: float = math.nan
    #: Which stage of the decision: 0 the turn, 1.. each resumption after a pause.
    stage: int = 0
    #: The landed outcome's weight (normalised), for the check that luck is spread.
    p_landed: float = math.nan

    def to_json(self) -> dict[str, Any]:
        out = {
            "d": self.decision, "t": self.turn, "s": self.status, "st": self.stage,
            "nb": self.branches, "np": self.pauses, "x": self.exact, "c": self.c,
        }
        for key, value in (("e", self.expected), ("v", self.landed_value), ("p", self.p_landed)):
            if not math.isnan(value):
                out[key] = value
        return out


@dataclass
class GameTerms:
    index: int | None
    seat_label: str
    seat_index: int | None
    outcome: float | None
    stages: list[Stage] = field(default_factory=list)
    #: Replacement phases played (not corrected), and seconds spent here.
    replacements: int = 0
    seconds: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(s.c for s in self.stages))

    def to_json(self) -> dict[str, Any]:
        return {
            "gameIndex": self.index,
            "seat": self.seat_label,
            "seatIndex": self.seat_index,
            "outcome": self.outcome,
            "C": self.total,
            "replacements": self.replacements,
            "seconds": round(self.seconds, 3),
            "stages": [s.to_json() for s in self.stages],
        }


def _chosen(reg: Any, pos: Any, decision: dict[str, Any]) -> list[Any] | None:  # noqa: ANN401
    """The recorded choices as SideActions, or None if either is not legal here."""
    from pokeuraou.actions import side_actions

    out = []
    for side, key in ((0, "ownChosen"), (1, "foeChosen")):
        wanted = decision.get(key)
        found = {a.to_choice(): a for a in side_actions(reg, pos, side)}.get(wanted)
        if found is None:
            return None
        out.append(found)
    return out


def _same(position: Any, recorded: dict[str, Any]) -> bool:  # noqa: ANN401
    return position.to_json() == recorded


def game_terms(
    reg: Any,  # noqa: ANN401
    game: dict[str, Any],
    evaluate: Callable[[list[Any]], np.ndarray],
    *,
    budget: Any = None,  # noqa: ANN401
) -> GameTerms:
    """Every move decision's chance correction for one recorded game, in side 0's units."""
    from pokeuraou import port
    from pokeuraou.actions import PassAction, SideAction
    from pokeuraou.budget import Budget
    from pokeuraou.position import Position

    budget = budget or Budget.exact()
    started = time.perf_counter()
    terms = GameTerms(
        index=game.get("gameIndex"),
        seat_label=str((game.get("provenance") or {}).get("seat", "")),
        seat_index=game.get("seatIndex"),
        outcome=game.get("outcome"),
    )
    decisions = game["decisions"]
    final = game.get("finalPosition")
    terms.replacements = sum(1 for d in decisions if d["kind"] == "replacement")

    def target(j: int) -> dict[str, Any] | None:
        """The position record entry `j` stands at: a decision's, or the final one."""
        if j < len(decisions):
            return decisions[j]
        return None if final is None else {"kind": "final", "position": final}

    i = 0
    while i < len(decisions):
        decision = decisions[i]
        if decision["kind"] != "move":
            i += 1
            continue
        turn_no = int(decision["turn"])
        pos = Position.from_json(decision["position"])
        chosen = _chosen(reg, pos, decision)
        if chosen is None:
            terms.stages.append(Stage(i, turn_no, "illegal"))
            i += 1
            continue
        try:
            result = port.turn(reg, pos, chosen, budget, full=True)
        except port.PortRefused:
            terms.stages.append(Stage(i, turn_no, "refused"))
            i += 1
            continue
        j = i + 1
        stage_no = 0
        while True:
            outcomes = list(result.outcomes or [])
            pauses = list(result.pauses or [])
            weights = [o.probability for o in outcomes] + [p.probability for p in pauses]
            paused = [False] * len(outcomes) + [True] * len(pauses)
            stage = Stage(
                i, turn_no, "matched", branches=len(outcomes), pauses=len(pauses),
                exact=bool(result.exact), stage=stage_no,
            )
            if not weights or sum(weights) <= 0:
                stage.status = "empty"
                terms.stages.append(stage)
                break
            values = (
                np.asarray(evaluate([o.position for o in outcomes]), dtype=np.float64)
                if outcomes
                else np.zeros(0)
            )
            values = np.concatenate([values, np.zeros(len(pauses))])
            there = target(j)
            if there is None:
                stage.status = "no-next"
                _, stage.expected = neutral_correction(values, weights, paused, None)
                terms.stages.append(stage)
                break
            if there["kind"] == "selfswitch":
                landed = [
                    k for k, p in enumerate(pauses) if _same(p.position, there["position"])
                ]
                if not landed:
                    stage.status = "mismatch"
                    _, stage.expected = neutral_correction(values, weights, paused, None)
                    terms.stages.append(stage)
                    break
                at = len(outcomes) + landed[0]
                stage.status = "paused"
                stage.c, stage.expected = neutral_correction(values, weights, paused, at)
                stage.p_landed = weights[at] / sum(weights)
                terms.stages.append(stage)
                pause = pauses[landed[0]]
                chooser_is_0 = there.get("ownChosen") not in (None, "pass")
                answer = there["ownChosen"] if chooser_is_0 else there["foeChosen"]
                try:
                    chooser, alternatives = port.resume_alternatives(reg, pause)
                except port.PortRefused:
                    terms.stages.append(Stage(i, turn_no, "refused", stage=stage_no + 1))
                    break
                picked = next(
                    (a for a, _r in alternatives if a.to_choice() == answer), None
                )
                if chooser is None or picked is None or chooser != (0 if chooser_is_0 else 1):
                    terms.stages.append(Stage(i, turn_no, "illegal", stage=stage_no + 1))
                    break
                other = 1 - chooser
                passes = SideAction(
                    slots=tuple(
                        PassAction(slot=s)
                        for s in range(len(pause.position.sides[other].active))
                    )
                )
                try:
                    result = port.resume(
                        reg, pause, [picked, passes] if chooser == 0 else [passes, picked]
                    )
                except port.PortRefused:
                    terms.stages.append(Stage(i, turn_no, "refused", stage=stage_no + 1))
                    break
                j += 1
                stage_no += 1
                continue
            landed = [k for k, o in enumerate(outcomes) if _same(o.position, there["position"])]
            if not landed:
                stage.status = "mismatch"
                _, stage.expected = neutral_correction(values, weights, paused, None)
                terms.stages.append(stage)
                break
            stage.c, stage.expected = neutral_correction(values, weights, paused, landed[0])
            stage.landed_value = float(values[landed[0]])
            stage.p_landed = weights[landed[0]] / sum(weights)
            terms.stages.append(stage)
            break
        i = j
    terms.seconds = time.perf_counter() - started
    return terms


# -- compute ---------------------------------------------------------------------------------


def _evaluator(models: Sequence[Path], device: str | None) -> tuple[Any, Any]:  # noqa: ANN401
    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.regulation import load_regulation
    from pokeuraou.value import BatchedValue, load_ensemble

    reg = load_regulation("gen9championsvgc2026regmc")
    encoder = Encoder(reg)
    torch.set_num_threads(1)
    where = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nets, _ = load_ensemble(list(models), encoder)
    return reg, BatchedValue([n.to(where) for n in nets], encoder, device=where)


def _games(directory: Path) -> Any:  # noqa: ANN401
    """(file name, line number, record) of every game in a match directory."""
    for path in sorted(directory.glob("games-worker*.jsonl")):
        with path.open("rb") as handle:
            for number, line in enumerate(handle):
                if line.strip():
                    yield path.name, number, line


def compute(args: argparse.Namespace) -> None:
    models = args.value or [
        ROOT / "data" / "models" / "value-mc0.pt", ROOT / "data" / "models" / "value-mc0-s1.pt"
    ]
    reg, evaluate = _evaluator(models, args.device)
    k, n = (int(x) for x in args.shard.split("/"))
    done: set[tuple[str, int]] = set()
    if args.out.exists():
        with args.out.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                done.add((row["file"], row["line"]))
    count = 0
    with args.out.open("ab") as out:
        for name, number, line in _games(args.dir):
            if (name, number) in done:
                continue
            game = json.loads(line)
            # Sharded by the game index, so both seats of a game land in one shard.
            index = game.get("gameIndex")
            if index is not None and int(index) % n != k:
                continue
            if args.limit and count >= args.limit:
                break
            terms = game_terms(reg, game, evaluate)
            row = {"file": name, "line": number, **terms.to_json()}
            out.write((json.dumps(row) + "\n").encode("utf-8"))
            out.flush()
            count += 1
    print(f"{args.dir.name} shard {k}/{n}: {count} games written to {args.out}")


# -- report ----------------------------------------------------------------------------------


def arm_score(row: dict[str, Any]) -> tuple[int, int, float, float] | None:
    """(game index, seat, raw, corrected) for the tested arm, as `paired_result` pairs them;
    None for a game with no outcome or index. The correction is in side 0's units, so it is
    subtracted where the arm sat on side 0 and added where it sat on side 1."""
    outcome, index = row.get("outcome"), row.get("gameIndex")
    if outcome is None or index is None:
        return None
    at0 = str(row.get("seat", "")).endswith("side 0")
    z = float(outcome)
    raw = float(z > 0.5) if at0 else float(z < 0.5)
    corrected = raw - row["C"] if at0 else raw + row["C"]
    return int(index), 0 if at0 else 1, raw, corrected


def pair_scores(rows: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(game indices, raw pair scores, corrected pair scores), in game-index order."""
    seats: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        got = arm_score(row)
        if got is not None:
            index, seat, raw, corrected = got
            seats[index][seat] = (raw, corrected)
    indices = sorted(i for i, v in seats.items() if len(v) == 2)
    raw = np.array([(seats[i][0][0] + seats[i][1][0]) / 2 for i in indices])
    corrected = np.array([(seats[i][0][1] + seats[i][1][1]) / 2 for i in indices])
    return np.array(indices), raw, corrected


def llr_normal_path(scores: np.ndarray, elo0: float, elo1: float) -> np.ndarray:
    """The normal-approximation GSPRT (fishtest's `LLR_alt2`) on continuous pair scores:
    `n (s1 - s0) (2 mean - s0 - s1) / (2 var)` after each pair, with the running mean and
    (population) variance. Zero while the variance is still zero."""
    from pokeuraou.sprt import expected_score

    s0, s1 = expected_score(elo0), expected_score(elo1)
    x = np.asarray(scores, dtype=np.float64)
    n = np.arange(1, len(x) + 1, dtype=np.float64)
    mean = np.cumsum(x) / n
    var = np.cumsum(x * x) / n - mean * mean
    with np.errstate(divide="ignore", invalid="ignore"):
        path = n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)
    return np.where(var > 1e-12, path, 0.0)


def _stop(path: np.ndarray) -> tuple[int, str | None]:
    from pokeuraou.sprt import bounds, first_crossing

    lower, upper = bounds(0.05, 0.05)
    stop = first_crossing(path, lower, upper)
    return stop.pairs, stop.decision


def _summary(x: np.ndarray) -> tuple[float, float, float]:
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else math.nan
    return mean, sd, 1.96 * sd / math.sqrt(n) if n > 1 else math.nan


def report(args: argparse.Namespace) -> None:
    from pokeuraou.sprt import cumulative_counts, elo_of, llr

    elo0, elo1 = args.sprt
    for path in args.jsonl:
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        indices, raw, corrected = pair_scores(rows)
        n = len(raw)
        print(f"\n{path}  ({len(rows)} games, {n} complete pairs)")
        if not n:
            continue
        # -- stages
        stages = [s for r in rows for s in r["stages"]]
        first = [s for s in stages if s["st"] == 0]
        status = Counter(s["s"] for s in first)
        print(
            f"  move decisions {len(first)}: "
            + ", ".join(f"{k} {v} ({v / len(first):.2%})" for k, v in status.most_common())
        )
        later = Counter(s["s"] for s in stages if s["st"] > 0)
        if later:
            print(f"  resumed stages after a pause: {dict(later)}")
        corrected_stages = [s for s in stages if s["s"] in ("matched", "paused")]
        inexact = [s for s in corrected_stages if not s["x"]]
        print(
            f"  corrected stages {len(corrected_stages)}, inexact enumeration (max_branches"
            f" bound) {len(inexact)} ({len(inexact) / max(len(corrected_stages), 1):.2%})"
        )
        for label, group in (("all", corrected_stages), ("exact", [s for s in corrected_stages if s["x"]]),
                             ("inexact", inexact),
                             ("after a pause", [s for s in corrected_stages if s["st"] > 0])):
            cs = np.array([s["c"] for s in group], dtype=np.float64)
            if len(cs) > 1:
                se = cs.std(ddof=1) / math.sqrt(len(cs))
                print(
                    f"  c over {label:13s} n={len(cs):6d}  mean {cs.mean():+.5f} +-{1.96 * se:.5f}"
                    f"  (z {cs.mean() / se if se else 0:+.2f})  sd {cs.std(ddof=1):.4f}"
                )
        # -- the pairs
        for label, x in (("raw", raw), ("corrected", corrected)):
            mean, sd, half = _summary(x)
            print(
                f"  {label:9s} mean {mean:.4f} +-{half:.4f}  sd {sd:.4f}  "
                f"Elo {elo_of(mean):+.1f} [{elo_of(max(mean - half, 1e-6)):+.1f},"
                f" {elo_of(min(mean + half, 1 - 1e-6)):+.1f}]"
            )
        diff = raw - corrected
        dmean, dsd, dhalf = _summary(diff)
        ratio = float(corrected.var(ddof=1) / raw.var(ddof=1)) if raw.var() > 0 else math.nan
        print(
            f"  raw - corrected per pair: mean {dmean:+.5f} +-{dhalf:.5f}"
            f"  (z {dmean / (dsd / math.sqrt(n)) if dsd else 0:+.2f})"
        )
        print(
            f"  variance ratio {ratio:.3f}  (sd x{math.sqrt(ratio):.3f}); pairs for the same"
            f" interval: x{ratio:.3f} ({n} -> {n * ratio:.0f})"
            f"  corr(raw, corrected) {np.corrcoef(raw, corrected)[0, 1]:.3f}"
        )
        flat = float(np.max(np.abs(corrected - 0.5)))
        print(f"  max |corrected pair - 1/2| {flat:.3e}   pairs split 1-1 raw: {int((raw == 0.5).sum())}")
        # -- sequential tests, in game-index order
        tri = _stop(llr(cumulative_counts(list(raw)), elo0, elo1))
        nraw = _stop(llr_normal_path(raw, elo0, elo1))
        ncor = _stop(llr_normal_path(corrected, elo0, elo1))
        print(
            f"  SPRT({elo0:+g},{elo1:+g}) in game-index order: trinomial raw {tri[1]} at {tri[0]},"
            f"  normal raw {nraw[1]} at {nraw[0]},  normal corrected {ncor[1]} at {ncor[0]}"
        )
        if args.bootstrap:
            rng = np.random.default_rng(args.seed)
            cap = args.cap
            got: dict[str, list[tuple[int, str | None]]] = {"raw": [], "corrected": []}
            for _ in range(args.bootstrap):
                pick = rng.integers(n, size=cap)
                got["raw"].append(_stop(llr_normal_path(raw[pick], elo0, elo1)))
                got["corrected"].append(_stop(llr_normal_path(corrected[pick], elo0, elo1)))
            for label, runs in got.items():
                pairs = np.array([p for p, _ in runs])
                decisions = Counter(d for _, d in runs)
                print(
                    f"  bootstrap {label:9s} ({args.bootstrap} paths, cap {cap}): pairs mean"
                    f" {pairs.mean():.0f} median {np.median(pairs):.0f}; "
                    + ", ".join(f"{k} {v}" for k, v in decisions.most_common())
                )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compute")
    c.add_argument("dir", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--value", type=Path, nargs="+", help="the evaluator (default value-mc0x2)")
    c.add_argument("--device", default=None)
    c.add_argument("--shard", default="0/1", help="k/n: the games whose index is k mod n")
    c.add_argument("--limit", type=int, default=0, help="stop after this many games")
    r = sub.add_parser("report")
    r.add_argument("jsonl", type=Path, nargs="+")
    r.add_argument("--sprt", type=float, nargs=2, default=(0.0, 10.0))
    r.add_argument("--bootstrap", type=int, default=0)
    r.add_argument("--cap", type=int, default=3000)
    r.add_argument("--seed", type=int, default=193)
    args = ap.parse_args(argv)
    (compute if args.command == "compute" else report)(args)


if __name__ == "__main__":
    main()
