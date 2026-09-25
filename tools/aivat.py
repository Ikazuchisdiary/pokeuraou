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
#: by-outcome  the last turn of a record without finalPosition (before IKA-87): the landed
#:           branch is one the battle ended in with the recorded winner, all scored alike
#: no-next   the record has nothing to match the last turn against
#: empty     the turn had no outcome with weight (the game ended "unresolved")
STATUSES = (
    "matched", "paused", "by-outcome", "mismatch", "refused", "illegal", "no-next", "empty",
)


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
    #: A pause: how many pauses stood at the recorded position (resumed together).
    group: int = 1

    def to_json(self) -> dict[str, Any]:
        out = {
            "d": self.decision, "t": self.turn, "s": self.status, "st": self.stage,
            "nb": self.branches, "np": self.pauses, "x": self.exact, "c": self.c,
        }
        if self.group > 1:
            out["g"] = self.group
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


def _combined(parts: Sequence[tuple[float, Any]]) -> Any:  # noqa: ANN401
    """Several resumed turns as one: each outcome weighted by its pause's weight times its
    own normalised weight. A pause keeps the port's object in `.pause` to be resumed."""
    from types import SimpleNamespace

    total = sum(w for w, _ in parts)
    outcomes, pauses = [], []
    for w, result in parts:
        own = sum(result.branches) + sum(result.suspended)
        if own <= 0:
            continue
        scale = w / total / own
        outcomes += [
            SimpleNamespace(probability=o.probability * scale, position=o.position)
            for o in result.outcomes or []
        ]
        pauses += [
            SimpleNamespace(
                probability=p.probability * scale, position=p.position,
                pause=getattr(p, "pause", p),
            )
            for p in result.pauses or []
        ]
    return SimpleNamespace(
        outcomes=outcomes, pauses=pauses, exact=all(bool(r.exact) for _, r in parts)
    )


def _expectation(result: Any, values: np.ndarray) -> float:  # noqa: ANN401
    """The neutral expectation of a turn: `neutral_correction`'s E, NaN with no branch."""
    outcomes = list(result.outcomes or [])
    pauses = list(result.pauses or [])
    weights = [o.probability for o in outcomes] + [p.probability for p in pauses]
    paused = [False] * len(outcomes) + [True] * len(pauses)
    return neutral_correction(
        np.concatenate([values, np.zeros(len(pauses))]), weights, paused, None
    )[1]


@dataclass
class ActionTerm:
    """AIVAT's action term at one move decision (IKA-193 stage 2): the value of the pair
    the two mixtures drew, less the value of the mixtures, with Q(a, b) the neutral
    expectation of the exact turn under the same evaluator."""

    decision: int
    status: str
    pairs: int = 0
    c: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"d": self.decision, "s": self.status, "n": self.pairs, "c": self.c}


def action_terms(
    reg: Any,  # noqa: ANN401
    game: dict[str, Any],
    evaluate: Callable[[list[Any]], np.ndarray],
    *,
    budget: Any = None,  # noqa: ANN401
) -> list[ActionTerm]:
    """Every move decision's action correction, in side 0's units.

    Zero-mean because each side's action was drawn from the mixture the record keeps
    (`ownPolicy` / `foePolicy`, sampled by `_sample_index`) independently of the other, and
    Q is a fixed function of the pair. Replacement decisions are not here: in the records
    this was written for, every one was a pure strategy (no draw to correct). A decision
    whose mixtures put weight on a single pair contributes exactly zero.
    """
    from pokeuraou import port
    from pokeuraou.actions import side_actions
    from pokeuraou.budget import Budget
    from pokeuraou.position import Position

    budget = budget or Budget.exact()
    out: list[ActionTerm] = []
    for i, decision in enumerate(game["decisions"]):
        if decision["kind"] != "move":
            continue
        own = [(a, p) for a, p in zip(decision["ownActions"], decision["ownPolicy"], strict=True) if p > 0]
        foe = [(a, p) for a, p in zip(decision["foeActions"], decision["foePolicy"], strict=True) if p > 0]
        pairs = [(a, b, pa * pb) for a, pa in own for b, pb in foe]
        if len(pairs) == 1:
            out.append(ActionTerm(i, "pure", 1))
            continue
        pos = Position.from_json(decision["position"])
        legal = [{x.to_choice(): x for x in side_actions(reg, pos, s)} for s in (0, 1)]
        if any(a not in legal[0] for a, _ in own) or any(b not in legal[1] for b, _ in foe):
            out.append(ActionTerm(i, "illegal", len(pairs)))
            continue
        answers = port.turns(
            reg, [(pos, [legal[0][a], legal[1][b]]) for a, b, _ in pairs], budget, full=True
        )
        if any(isinstance(answer, Exception) for answer in answers):
            out.append(ActionTerm(i, "refused", len(pairs)))
            continue
        # A pair every one of whose outcomes is a pause (a faster U-turn: every roll stops
        # the turn) has no neutral value; its Q is the pauses' own positions, weighted --
        # any fixed function of the pair keeps the term's mean at zero.
        pause_only = [not (answer.outcomes or []) for answer in answers]
        flat = [
            o.position
            for answer, only in zip(answers, pause_only, strict=True)
            for o in (answer.pauses if only else answer.outcomes) or []
        ]
        values = np.asarray(evaluate(flat), dtype=np.float64) if flat else np.zeros(0)
        q, at = [], 0
        for answer, only in zip(answers, pause_only, strict=True):
            if only:
                w = np.asarray([p.probability for p in answer.pauses or []], dtype=np.float64)
                count = len(w)
                part = values[at : at + count]
                q.append(float((w * part).sum() / w.sum()) if count and w.sum() > 0 else math.nan)
            else:
                count = len(answer.outcomes or [])
                q.append(_expectation(answer, values[at : at + count]))
            at += count
        q = np.asarray(q)
        if np.isnan(q).any():
            out.append(ActionTerm(i, "empty", len(pairs)))
            continue
        drawn = [
            k for k, (a, b, _) in enumerate(pairs)
            if a == decision["ownChosen"] and b == decision["foeChosen"]
        ]
        if len(drawn) != 1:
            out.append(ActionTerm(i, "illegal", len(pairs)))
            continue
        p = np.asarray([w for _, _, w in pairs])
        p = p / p.sum()
        out.append(ActionTerm(i, "corrected", len(pairs), float(q[drawn[0]] - (p * q).sum())))
    return out


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
                # A record from before IKA-87 has no final position, but a game with an
                # outcome ended on its last turn: every branch it can have landed on is one
                # the battle ended in with that winner, and the evaluator scores each of
                # them as that result -- so the correction is exact without the position.
                won = [
                    k for k, o in enumerate(outcomes)
                    if o.position.ended and o.position.winner is not None
                    and float(o.position.winner == o.position.sides[0].id) == terms.outcome
                ]
                if terms.outcome is None or not won:
                    stage.status = "no-next"
                    _, stage.expected = neutral_correction(values, weights, paused, None)
                else:
                    stage.status = "by-outcome"
                    stage.c, stage.expected = neutral_correction(values, weights, paused, won[0])
                    stage.landed_value = float(values[won[0]])
                    stage.p_landed = sum(weights[k] for k in won) / sum(weights)
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
                stage.p_landed = sum(weights[len(outcomes) + k] for k in landed) / sum(weights)
                stage.group = len(landed)
                terms.stages.append(stage)
                # Pauses can share a position and differ in the rest of the turn (a
                # `moveLastTurnFailed` still to be written, say): the record cannot tell
                # which was drawn, so the next stage is all of them, each resumed and
                # weighted by its own chance -- the distribution of the rest of the turn
                # given the position the record shows.
                chooser_is_0 = there.get("ownChosen") not in (None, "pass")
                answer = there["ownChosen"] if chooser_is_0 else there["foeChosen"]
                parts = []
                status = None
                for k in landed:
                    pause = getattr(pauses[k], "pause", pauses[k])
                    try:
                        chooser, alternatives = port.resume_alternatives(reg, pause)
                    except port.PortRefused:
                        status = "refused"
                        break
                    picked = next(
                        (a for a, _r in alternatives if a.to_choice() == answer), None
                    )
                    if chooser is None or picked is None or chooser != (0 if chooser_is_0 else 1):
                        status = "illegal"
                        break
                    other = 1 - chooser
                    passes = SideAction(
                        slots=tuple(
                            PassAction(slot=s)
                            for s in range(len(pause.position.sides[other].active))
                        )
                    )
                    try:
                        resumed = port.resume(
                            reg, pause, [picked, passes] if chooser == 0 else [passes, picked]
                        )
                    except port.PortRefused:
                        status = "refused"
                        break
                    parts.append((weights[len(outcomes) + k], resumed))
                if status is not None:
                    terms.stages.append(Stage(i, turn_no, status, stage=stage_no + 1))
                    break
                result = _combined(parts)
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


def _evaluator(
    models: Sequence[Path], device: str | None, format_id: str = "gen9championsvgc2026regmc"
) -> tuple[Any, Any]:  # noqa: ANN401
    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.regulation import load_regulation
    from pokeuraou.value import BatchedValue, load_ensemble

    reg = load_regulation(format_id)
    if format_id.endswith("regmb"):
        from pokeuraou.damage import register_mega_stones

        register_mega_stones(reg)
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
    reg, evaluate = _evaluator(models, args.device, args.format)
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
            if args.actions:
                acted = action_terms(reg, game, evaluate)
                row["A"] = float(sum(a.c for a in acted))
                row["actions"] = [a.to_json() for a in acted]
            out.write((json.dumps(row) + "\n").encode("utf-8"))
            out.flush()
            count += 1
    print(f"{args.dir.name} shard {k}/{n}: {count} games written to {args.out}")


# -- report ----------------------------------------------------------------------------------


def arm_score(
    row: dict[str, Any], terms: str = "chance"
) -> tuple[int, int, float, float] | None:
    """(game index, seat, raw, corrected) for the tested arm, as `paired_result` pairs them;
    None for a game with no outcome or index. The correction is in side 0's units, so it is
    subtracted where the arm sat on side 0 and added where it sat on side 1."""
    outcome, index = row.get("outcome"), row.get("gameIndex")
    if outcome is None or index is None:
        return None
    at0 = str(row.get("seat", "")).endswith("side 0")
    z = float(outcome)
    raw = float(z > 0.5) if at0 else float(z < 0.5)
    total = {
        "chance": row["C"], "actions": row.get("A", 0.0),
        "both": row["C"] + row.get("A", 0.0),
    }[terms]
    corrected = raw - total if at0 else raw + total
    return int(index), 0 if at0 else 1, raw, corrected


def pair_scores(
    rows: Sequence[dict[str, Any]], terms: str = "chance"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(game indices, raw pair scores, corrected pair scores), in game-index order."""
    seats: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        got = arm_score(row, terms)
        if got is not None:
            index, seat, raw, corrected = got
            seats[index][seat] = (raw, corrected)
    indices = sorted(i for i, v in seats.items() if len(v) == 2)
    raw = np.array([(seats[i][0][0] + seats[i][1][0]) / 2 for i in indices])
    corrected = np.array([(seats[i][0][1] + seats[i][1][1]) / 2 for i in indices])
    return np.array(indices), raw, corrected


def llr_normal_path(
    scores: np.ndarray, elo0: float, elo1: float, *, burn: int = 0
) -> np.ndarray:
    """The normal-approximation GSPRT (fishtest's `LLR_alt2`) on continuous pair scores:
    `n (s1 - s0) (2 mean - s0 - s1) / (2 var)` after each pair, with the running mean and
    (population) variance. Zero while the variance is still zero, and over the first
    `burn` pairs: the rule registered first had no burn-in, and the stage-1 bootstrap
    found it stopping at the second pair on two near-equal corrected scores (IKA-193)."""
    from pokeuraou.sprt import expected_score

    s0, s1 = expected_score(elo0), expected_score(elo1)
    x = np.asarray(scores, dtype=np.float64)
    n = np.arange(1, len(x) + 1, dtype=np.float64)
    mean = np.cumsum(x) / n
    var = np.cumsum(x * x) / n - mean * mean
    with np.errstate(divide="ignore", invalid="ignore"):
        path = n * (s1 - s0) * (2 * mean - s0 - s1) / (2 * var)
    path = np.where(var > 1e-12, path, 0.0)
    path[: max(0, burn)] = 0.0
    return path


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
    # c by kind over every file, and raw - corrected per pair: the bias checks.
    pooled: dict[str, list[float]] = defaultdict(list)
    for path in args.jsonl:
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if args.terms != "chance" and not any("A" in row for row in rows):
            print(f"\n{path}: no action terms (compute --actions)")
            continue
        indices, raw, corrected = pair_scores(rows, args.terms)
        n = len(raw)
        print(f"\n{path}  ({len(rows)} games, {n} complete pairs)")
        if not n:
            continue
        # -- the action terms, when there are any
        acted = [a for r in rows for a in r.get("actions", [])]
        if acted:
            kinds = Counter(a["s"] for a in acted)
            cs = np.array([a["c"] for a in acted if a["s"] == "corrected"])
            se = cs.std(ddof=1) / math.sqrt(len(cs)) if len(cs) > 1 else math.nan
            print(
                f"  action terms {len(acted)}: {dict(kinds)}; corrected mean {cs.mean():+.5f}"
                f" +-{1.96 * se:.5f} (z {cs.mean() / se:+.2f}) sd {cs.std(ddof=1):.4f}"
            )
            pooled["action"].extend(cs.tolist())
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
        corrected_stages = [s for s in stages if s["s"] in ("matched", "paused", "by-outcome")]
        inexact = [s for s in corrected_stages if not s["x"]]
        print(
            f"  corrected stages {len(corrected_stages)}, inexact enumeration (max_branches"
            f" bound) {len(inexact)} ({len(inexact) / max(len(corrected_stages), 1):.2%})"
        )
        for label, group in (("all", corrected_stages), ("exact", [s for s in corrected_stages if s["x"]]),
                             ("inexact", inexact),
                             ("after a pause", [s for s in corrected_stages if s["st"] > 0])):
            cs = np.array([s["c"] for s in group], dtype=np.float64)
            pooled[label].extend(cs.tolist())
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
        pooled["pair"].extend(diff.tolist())
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
        nraw = _stop(llr_normal_path(raw, elo0, elo1, burn=args.burn))
        ncor = _stop(llr_normal_path(corrected, elo0, elo1, burn=args.burn))
        print(
            f"  SPRT({elo0:+g},{elo1:+g}) in game-index order (normal burn-in {args.burn}):"
            f" trinomial raw {tri[1]} at {tri[0]},"
            f"  normal raw {nraw[1]} at {nraw[0]},  normal corrected {ncor[1]} at {ncor[0]}"
        )
        if args.bootstrap:
            rng = np.random.default_rng(args.seed)
            cap = args.cap
            recentred = corrected - corrected.mean() + raw.mean()
            series = {"raw": raw, "corrected": corrected, "recentred": recentred}
            got: dict[str, list[tuple[int, str | None]]] = {k: [] for k in series}
            for _ in range(args.bootstrap):
                pick = rng.integers(n, size=cap)
                for label, x in series.items():
                    got[label].append(_stop(llr_normal_path(x[pick], elo0, elo1, burn=args.burn)))
            for label, runs in got.items():
                pairs = np.array([p for p, _ in runs])
                decisions = Counter(d for _, d in runs)
                print(
                    f"  bootstrap {label:9s} ({args.bootstrap} paths, cap {cap}, burn-in"
                    f" {args.burn}): pairs mean {pairs.mean():.0f} median {np.median(pairs):.0f}; "
                    + ", ".join(f"{k} {v}" for k, v in decisions.most_common())
                )
    if len(args.jsonl) > 1:
        print(f"\npooled over {len(args.jsonl)} files")
        for label, values in pooled.items():
            cs = np.asarray(values, dtype=np.float64)
            if len(cs) > 1:
                se = cs.std(ddof=1) / math.sqrt(len(cs))
                print(
                    f"  {label:13s} n={len(cs):6d}  mean {cs.mean():+.5f} +-{1.96 * se:.5f}"
                    f"  (z {cs.mean() / se if se else 0:+.2f})"
                )

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compute")
    c.add_argument("dir", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--value", type=Path, nargs="+", help="the evaluator (default value-mc0x2)")
    c.add_argument("--device", default=None)
    c.add_argument("--format", default="gen9championsvgc2026regmc")
    c.add_argument("--shard", default="0/1", help="k/n: the games whose index is k mod n")
    c.add_argument("--limit", type=int, default=0, help="stop after this many games")
    c.add_argument("--actions", action="store_true", help="the action term too (stage 2)")
    r = sub.add_parser("report")
    r.add_argument("jsonl", type=Path, nargs="+")
    r.add_argument("--sprt", type=float, nargs=2, default=(0.0, 10.0))
    r.add_argument("--bootstrap", type=int, default=0)
    r.add_argument("--cap", type=int, default=3000)
    r.add_argument("--seed", type=int, default=193)
    r.add_argument("--terms", choices=("chance", "actions", "both"), default="chance")
    r.add_argument(
        "--burn", type=int, default=0,
        help="pairs before the normal test may stop (0: the rule registered first)",
    )
    args = ap.parse_args(argv)
    (compute if args.command == "compute" else report)(args)


if __name__ == "__main__":
    main()
