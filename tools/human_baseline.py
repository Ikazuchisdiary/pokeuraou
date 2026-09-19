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


CASES: tuple[Case, ...] = (
    Case(
        name="sash-ko",
        source="data/matches/anchor-gen11Lx2-book9-hidden-vs-hpshare-m2/games-worker21.jsonl",
        game_index=12,
        turn=2,
        at_least="move 1, move 1 2",
        at_most="move 3 2, move 1 2",
        why=(
            "Venusaur is at 187/187 behind a Focus Sash. Flare Blitz alone is lethal and "
            "the Sash holds it at 1. Hyper Voice does 25-30, which is not lethal, so the "
            "Sash never fires and Flare Blitz then kills: the resolver confirms Venusaur "
            "dies in 100% of the branches. The two actions have the SAME second half -- "
            "Incineroar uses Flare Blitz either way and takes the same recoil -- so "
            "neither the 'switch out first' nor the 'let the weather run down' reason for "
            "declining a kill can distinguish them. The equilibrium put 65.2% on the one "
            "that leaves Venusaur alive at 1 HP and 0.0% on the kill."
        ),
        resolver_check=("move 1, move 1 2", "venusaur", "dead"),
    ),
    Case(
        name="sash-ko-detect",
        source="data/matches/anchor-gen11Lx2-book9-hidden-vs-hpshare-m2/games-worker21.jsonl",
        game_index=12,
        turn=2,
        at_least="move 4, move 1 2",
        at_most="move 3 2, move 1 2",
        why=(
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
    games = [json.loads(line) for line in open(case.source, encoding="utf-8")]
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
    for case in cases:
        pos, decision = position_of(case)
        print(f"\n=== {case.name}  ({name})")
        complaint = check_resolver(case, reg, pos, decision)
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
        print(f"  {'PASS' if ok else 'FAIL'}  "
              f"{case.at_least} {ev[hi]:.4f} (mass {row[hi]:.1%})  "
              f"{'>=' if ok else '<'}  "
              f"{case.at_most} {ev[lo]:.4f} (mass {row[lo]:.1%})")
        if not ok:
            failures += 1
            print(f"    gap {ev[lo] - ev[hi]:+.4f} in win probability")
            for line in case.why.split(". "):
                print(f"    {line.strip()}")

    print(f"\n{len(cases) - failures}/{len(cases)} cases pass for {name}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
