"""Positions where a person knows the answer, checked against a leaf.

This project has never had one. The record says so in as many words -- "尺度に人間基準が
1つも無い" -- and the consequence has been visible all week: every measurement is one
agent against another, so a whole family of agents can be wrong together and the scale
will not notice. Three of the defects found on 2026-09-19 were found by a human reading a
single game, and none by seven hours of automated measurement.

So: a small set of positions, each with a claim a person can defend, each checkable
against any leaf. Small on purpose. The value of a case is that someone can say why the
answer is what it is, and a hundred cases nobody can defend is a worse instrument than
three that anyone can.

A case never says "this move is best". It says one action's value must be at least
another's, and it says why -- because a claim that does not carry its reason cannot be
argued with when a future model fails it, and a failing test nobody can argue with gets
deleted.

    uv run python tools/human_baseline.py --value data/models/value-gen11L.pt
    uv run python tools/human_baseline.py --value data/models/value-gen11L.pt \\
        data/models/value-gen11L-s1.pt --case sash-ko
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn  # noqa: E402
from pokeuraou.search import leaf_ranking, search  # noqa: E402


@dataclass(frozen=True)
class Case:
    """One position, one claim, and the reason the claim is defensible."""

    name: str
    #: Where the position came from, exactly, so it can be looked at again.
    source: str
    game_index: int
    turn: int
    #: `at_least` must score at least as well as `at_most`.
    at_least: str
    at_most: str
    why: str
    #: What the resolver must do for the claim to make sense, checked before the leaf is
    #: blamed. `(action, species, "dead"|"alive")`.
    resolver_check: tuple[str, str, str] | None = None
    #: The opponent slot the two actions contest. When set, `at_least` is checked against
    #: EVERY legal opponent reply, not the modal one: it must never do worse than
    #: `at_most` on faint probability in that slot or on damage dealt, with damage taken
    #: equal. This is the part of a human claim a machine can own. A person reading one
    #: game can see that an action looks wrong and can name the alternative worth
    #: checking -- and on 2026-09-19 a person did both, and was right both times -- but
    #: "and there is no reply that makes it right" is a statement with a quantifier over
    #: 79 columns, and nobody should be asked to hold 79 columns in their head. The
    #: division that survives a complicated position is: the human picks the question,
    #: the resolver answers it.
    #:
    #: What it does not cover: the three statistics price damage and faints, and not
    #: board state. A Yawn that leaves the target alive and drowsy scores zero here, so
    #: passing this check says no reply makes the rejected action hit harder -- not that
    #: what it does instead is worthless. That valuation belongs to the leaf, which is
    #: what the rest of the case tests.
    dominance_slot: int | None = None
    #: A claim two independent models have rejected. Reported, never counted -- a
    #: case is only as good as its reason, and a reason the models argue with is
    #: evidence about the case rather than about them.
    contested: bool = False


CASES: tuple[Case, ...] = (
    Case(
        name="sash-ko",
        source="data/matches/anchor-gen11Lx2-book9-hidden-vs-hpshare-m2/games-worker21.jsonl",
        game_index=12,
        turn=2,
        at_least="move 1, move 1 2",
        at_most="move 3 2, move 1 2",
        why=(
            "REASON REPLACED 2026-09-19. The first version argued that the two actions "
            "share a second half, so no switching reason could separate them -- and it "
            "counted only OUR switches. The opponent's switches are the whole position, "
            "and it never mentioned them, nor the two facts that generate them: their "
            "Mega Charizard Y is ALREADY drowsy from turn 1's Yawn and falls asleep at "
            "the end of this turn, and Venusaur has Chlorophyll under that Charizard's "
            "own sun, which makes it the fast threat rather than a spare target. A "
            "reason that omits the drowsy Pokemon on the field is not a reason. "
            "The claim survived the correction and the correction made it stronger. "
            "Venusaur is at 187/187 behind a Focus Sash. Flare Blitz alone would be "
            "lethal and the Sash holds it at 1; Hyper Voice does 25-30, which is not "
            "lethal, so the Sash never fires and Flare Blitz kills -- 100% of the branch "
            "mass. What the first version missed is that Hyper Voice is the answer to "
            "the switches too. Sixteen legal replies retreat Venusaur; Hyper Voice plus "
            "Flare Blitz kills whatever replaces it in 95-100% of the mass in every one "
            "of them, while Yawn does land on the switch-in in all sixteen and leaves it "
            "alive at 30-77. THAT PAIR OF NUMBERS KNOWS THE BENCH AND THE PLAYERS DID "
            "NOT -- this game ran with `information: [hidden-bench, hidden-bench]`, and "
            "the sheet also listed Swampert, which takes Flare Blitz at 0.25x, and "
            "Pelipper, whose Drizzle replaces the sun as it lands and halves the Fire "
            "damage again. Over the six benches consistent with the sheet the kill lands "
            "52.5% of the time, and 9.6% in the world where both of those are the back "
            "two (`tools/hidden_dominance.py`). So the claim is NOT that Hyper Voice "
            "kills what comes in; it is that Hyper Voice is never worse, which survives: "
            "zero columns out of 79 in each of the six completions. "
            "The opponent's real line is the other retreat -- Charizard "
            "leaving to shed its drowsiness while Venusaur Protects, which the "
            "equilibrium plays at 38.3% -- and neither action kills there, but Hyper "
            "Voice still puts 105 into the Grimmsnarl that comes in and Yawn puts 0. "
            "Over all 79 legal replies there is no column where declining the kill does "
            "more damage or wins more faints, and the damage taken is identical because "
            "Incineroar's half does not change. "
            "What this does not settle is whether a second sleep is worth more than the "
            "extra damage, because drowsiness is not damage and the dominance check "
            "cannot see it. That is the leaf's judgement, and it is what the case tests."
        ),
        resolver_check=("move 1, move 1 2", "venusaur", "dead"),
        dominance_slot=1,
    ),
    Case(
        name="sash-ko-detect",
        source="data/matches/anchor-gen11Lx2-book9-hidden-vs-hpshare-m2/games-worker21.jsonl",
        game_index=12,
        turn=2,
        at_least="move 4, move 1 2",
        at_most="move 3 2, move 1 2",
        contested=True,
        why=(
            "CONTESTED as of 2026-09-19: value-gen10 and value-gen11L both rank Yawn "
            "above Detect, and gen10 gets the kill above right -- 61.5% on it -- so its "
            "judgement on this position is not obviously poor. gen10's mixture suggests "
            "why: the kill is its MAIN line and Yawn the one it mixes in, and 'if the "
            "kill is declined, protect instead' assumes declining is the plan. Kept as a "
            "case, not counted as a model failure. "
            "Measured both directions over all 79 legal replies: Detect does better on "
            "40 of them, Yawn on 4, and neither dominates -- so the position genuinely "
            "is mixed and the case stays contested rather than resolving either way. "
            "No `dominance_slot` is set here on purpose. The check prices Detect's "
            "protection, because avoided damage is damage, and cannot price Yawn's "
            "drowsiness at all, so on THIS pair it would be an instrument with a thumb "
            "on the scale -- it would report the claim as proven while measuring only "
            "the half of the trade that it can see. That asymmetry is harmless for "
            "`sash-ko`, where the accepted action's advantage is the visible one. "
            "The same position, from the other side. If the kill is declined, Sylveon's "
            "turn is better spent on Detect than on Yawn: Charizard's Heat Wave is a "
            "spread move that hits Sylveon, and Yawn's payoff is putting a Venusaur that "
            "is about to be at 1 HP to sleep two turns from now. So Yawn is dominated "
            "whichever way the kill is decided -- by Hyper Voice if it is taken, by "
            "Detect if it is not. That is what makes this a case rather than a judgement "
            "call about tempo."
        ),
    ),
)


def load_leaf(paths: list[Path], encoder: Encoder):  # noqa: ANN201
    import torch

    from pokeuraou.value import BatchedValue, load_ensemble

    nets, _ = load_ensemble(paths, encoder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return BatchedValue([n.to(device) for n in nets], encoder, device=device)


def position_of(case: Case) -> tuple[Position, dict]:
    with open(case.source, encoding="utf-8") as handle:
        games = [json.loads(line) for line in handle]
    game = games[case.game_index]
    decision = next(
        d for d in game["decisions"] if d["kind"] == "move" and d["turn"] == case.turn
    )
    return Position.from_json(decision["position"]), decision


def check_resolver(case: Case, reg, pos: Position, decision: dict) -> str | None:
    """Whether the turn does what the claim says it does, before the leaf is blamed."""
    if case.resolver_check is None:
        return None
    action, species, expected = case.resolver_check
    foe_mix = np.asarray(decision["foePolicy"], dtype=np.float64)
    their = decision["foeActions"][int(np.argmax(foe_mix))]
    legal = {
        side: {a.to_choice(): a for a in side_actions(reg, pos, side)} for side in (0, 1)
    }
    if action not in legal[0] or their not in legal[1]:
        return f"{action!r} or {their!r} is not legal in this position"
    result = resolve_turn(reg, pos, [legal[0][action], legal[1][their]], budget=Budget())
    dead = Counter()
    total = 0.0
    for branch in result.branches:
        mon = next(
            (m for m in branch.position.sides[1].pokemon if m.species == species), None
        )
        dead[bool(mon is None or mon.fainted)] += branch.probability
        total += branch.probability
    share = dead[True] / total if total else 0.0
    want = expected == "dead"
    if (share > 0.99) != want:
        return (
            f"the resolver disagrees with the case: {species} is dead in {share:.1%} of "
            f"the mass after {action!r}, and the case says {expected}"
        )
    return None


def check_dominance(case: Case, reg, pos: Position) -> str | None:  # noqa: ANN001
    """Whether `at_least` ever does worse than `at_most`, over every legal reply."""
    if case.dominance_slot is None:
        return None
    legal = {
        side: {a.to_choice(): a for a in side_actions(reg, pos, side)} for side in (0, 1)
    }
    for choice in (case.at_least, case.at_most):
        if choice not in legal[0]:
            return f"{choice!r} is not legal in this position"

    def score(ours: str, theirs: str) -> tuple[float, float, float]:
        res = resolve_turn(reg, pos, [legal[0][ours], legal[1][theirs]], budget=Budget())
        before_them = {m.species: m.hp for m in pos.sides[1].pokemon}
        before_us = {m.species: m.hp for m in pos.sides[0].pokemon}
        dead = total = dealt = taken = 0.0
        for branch in res.branches:
            total += branch.probability
            occupant = branch.position.sides[1].active_pokemon()[case.dominance_slot]
            if occupant is not None and occupant.fainted:
                dead += branch.probability
            dealt += branch.probability * sum(
                max(0, before_them.get(m.species, m.hp) - m.hp)
                for m in branch.position.sides[1].pokemon
            )
            taken += branch.probability * sum(
                max(0, before_us.get(m.species, m.hp) - m.hp)
                for m in branch.position.sides[0].pokemon
            )
        n = total or 1.0
        return dead / n, dealt / n, taken / n

    worse = []
    for theirs in legal[1]:
        hi, lo = score(case.at_least, theirs), score(case.at_most, theirs)
        # A tolerance on damage, none on faints: a faint is a discrete event the resolver
        # either produced or did not, while damage is an average over rolls and half a
        # point of it is not a counterexample to anything.
        if hi[0] + 1e-9 < lo[0] or hi[1] + 0.5 < lo[1] or hi[2] > lo[2] + 0.5:
            worse.append(theirs)
    print(f"    支配: {len(legal[1])} 列中 {len(worse)} 列で {case.at_least!r} が劣る")
    if worse:
        return (
            f"{case.at_most!r} does better against {len(worse)} legal replies, "
            f"e.g. {worse[0]!r} -- the case claims there are none"
        )
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--value", type=Path, nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--case", default=None, help="run one case by name")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    args = ap.parse_args()

    reg = load_regulation(args.format)
    register_mega_stones(reg)
    encoder = Encoder(reg)
    leaf = load_leaf(list(args.value), encoder)
    name = "+".join(p.stem for p in args.value)

    cases = [c for c in CASES if args.case is None or c.name == args.case]
    if not cases:
        raise SystemExit(f"no case named {args.case!r}; have {[c.name for c in CASES]}")

    budget = Budget()
    failures = 0
    contested_failures = 0
    for case in cases:
        pos, decision = position_of(case)
        print(f"\n=== {case.name}  ({name})")
        complaint = check_resolver(case, reg, pos, decision) or check_dominance(
            case, reg, pos
        )
        if complaint:
            print(f"  ! {complaint}")
            print("    The case is about the leaf, so this is a different bug and the")
            print("    claim below is not evidence about the value function.")
            failures += 1
            continue
        ours = narrow(
            reg, pos, 0, limit=args.limit,
            rank=leaf_ranking(reg, pos, 0, leaf, budget=budget),
        ).actions
        theirs = narrow(
            reg, pos, 1, limit=args.limit,
            rank=leaf_ranking(reg, pos, 1, leaf, budget=budget),
        ).actions
        mine = [a.to_choice() for a in ours]
        missing = [c for c in (case.at_least, case.at_most) if c not in mine]
        if missing:
            print(f"  ! {missing} not in this leaf's {len(mine)}-action menu")
            print("    The narrowing dropped it, which is a different failure from")
            print("    valuing it wrongly -- and it is not the one this case tests.")
            failures += 1
            continue
        result = search(reg, pos, ours, theirs, leaf, budget=budget, depth=1)
        eq = result.equilibrium
        ev = np.asarray(eq.row_ev, dtype=np.float64)
        row = np.asarray(eq.row_strategy, dtype=np.float64)
        hi, lo = mine.index(case.at_least), mine.index(case.at_most)
        ok = ev[hi] >= ev[lo]
        mark = "PASS" if ok else ("CONTESTED" if case.contested else "FAIL")
        print(f"  {mark}  "
              f"{case.at_least} {ev[hi]:.4f} (mass {row[hi]:.1%})  "
              f"{'>=' if ok else '<'}  "
              f"{case.at_most} {ev[lo]:.4f} (mass {row[lo]:.1%})")
        if not ok:
            if case.contested:
                contested_failures += 1
            else:
                failures += 1
            print(f"    gap {ev[lo] - ev[hi]:+.4f} in win probability")
            for line in case.why.split(". "):
                print(f"    {line.strip()}")

    # Three counts, not two. A contested case that a model rejects is not a pass, and
    # folding it into one would make the summary say "2/2" for a run where a model
    # disagreed with half the cases -- which is the shape of reporting this project has
    # spent a day removing.
    passed = len(cases) - failures - contested_failures
    parts = [f"{passed} pass"]
    if failures:
        parts.append(f"{failures} FAIL")
    if contested_failures:
        parts.append(f"{contested_failures} rejected-but-contested")
    print(f"\n{', '.join(parts)} of {len(cases)} for {name}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
