"""Differential test of the whole turn resolver against Showdown.

For each turn: take the pre-turn position, run our resolver with every source of chance
pinned to the same outcome Showdown's policy forces, step Showdown, and compare the two
resulting positions field by field.

Under ``Budget.deterministic`` our resolver must produce exactly one branch, so this is an
equality test on states rather than a comparison of distributions. Divergences are counted
per field and attributed to the moves, abilities and items in play, so "what to fix next"
is a measurement.

    uv run python tools/diff_turn.py --battles 40 --roll 8
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import MoveAction, SideAction, side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn  # noqa: E402
from pokeuraou.speed import action_overriding_effects  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"

#: Position fields compared, and how much each matters. HP and fainted decide the game;
#: the rest shape the next turn.
COMPARED_FIELDS = (
    "hp",
    "fainted",
    "status",
    "boosts",
    "species",
    "item",
    "active",
    "side_conditions",
    "weather",
    "terrain",
    "pseudo_weather",
    "mega_used",
)


def canonical(pos: Position) -> dict[str, Any]:
    """The comparable projection of a position.

    Deliberately not everything: volatile bookkeeping and effect durations are compared
    only where the resolver claims to model them, and PP is left out because Showdown
    spends it in places the resolver does not yet reach (called moves, Pressure).
    """
    out: dict[str, Any] = {
        "weather": pos.field.weather,
        "terrain": pos.field.terrain,
        "pseudo_weather": sorted(p.id for p in pos.field.pseudo_weather),
    }
    for side_index, side in enumerate(pos.sides):
        prefix = f"p{side_index + 1}"
        out[f"{prefix}.active"] = list(side.active)
        out[f"{prefix}.side_conditions"] = sorted(c.id for c in side.side_conditions)
        out[f"{prefix}.mega_used"] = side.mega_used
        for mon in side.pokemon:
            key = f"{prefix}.{mon.slot}"
            out[f"{key}.hp"] = mon.hp
            out[f"{key}.fainted"] = mon.fainted
            # A fainted Pokemon's status is bookkeeping, not mechanics: Showdown writes
            # 'fnt' when it falls and clears it again when its replacement swaps in, so
            # comparing the field would report an artefact as a divergence.
            out[f"{key}.status"] = "fnt" if mon.fainted else mon.status
            out[f"{key}.boosts"] = dict(sorted(mon.boosts.items()))
            out[f"{key}.species"] = mon.species
            out[f"{key}.item"] = mon.item
    return out


def field_kind(key: str) -> str:
    return key.rsplit(".", 1)[-1]


@dataclass
class Report:
    compared: int = 0
    matched: int = 0
    #: Divergent turns where the resolver had already reported an unmodelled effect. These
    #: are documented gaps, not wrong answers.
    flagged_divergences: int = 0
    #: Divergent turns where the resolver reported nothing. This is the number that
    #: matters: a silent divergence means a printed value is quietly wrong.
    silent_divergences: int = 0
    #: Divergences per compared field kind.
    by_field: Counter[str] = field(default_factory=Counter)
    #: (field, move, ability, item) -> count.
    attributed: Counter[tuple[str, ...]] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    unmodelled: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)
    branch_counts: Counter[int] = field(default_factory=Counter)

    @property
    def divergence_rate(self) -> float:
        return 0.0 if not self.compared else 1 - self.matched / self.compared

    @property
    def silent_rate(self) -> float:
        """The rate that matters: divergence with no warning attached."""
        return 0.0 if not self.compared else self.silent_divergences / self.compared

    def render(self) -> str:
        out = [
            f"compared {self.compared} turns, matched {self.matched}, "
            f"divergence rate {self.divergence_rate * 100:.3f}% "
            f"(silent {self.silent_rate * 100:.3f}%, "
            f"flagged {self.flagged_divergences})"
        ]
        if self.branch_counts:
            out.append(
                "  branches produced: "
                + ", ".join(f"{k}x{v}" for k, v in sorted(self.branch_counts.items()))
            )
        if self.by_field:
            out.append(
                "  divergent fields: "
                + ", ".join(f"{k} x{v}" for k, v in self.by_field.most_common(12))
            )
        if self.attributed:
            out.append("  attributed (field / move / ability / item):")
            for key, count in self.attributed.most_common(18):
                out.append(f"    {count:5d}  {' / '.join(k or '-' for k in key)}")
        if self.skipped:
            out.append(
                "  skipped: " + ", ".join(f"{k} x{v}" for k, v in self.skipped.most_common(8))
            )
        if self.unmodelled:
            out.append("  resolver reported unmodelled:")
            for name, count in self.unmodelled.most_common(15):
                out.append(f"    {count:5d}  {name}")
        for line in self.examples[:10]:
            out.append("  " + line)
        return "\n".join(out)


def describe_actions(reg: Regulation, chosen: list[SideAction]) -> str:
    return " | ".join(a.describe(reg) for a in chosen)


def compare_turn(
    reg: Regulation,
    before: Position,
    chosen: list[SideAction],
    after_json: dict[str, Any],
    lines: list[str],
    roll: int,
    report: Report,
) -> None:
    # An Encore-style action override replaces a queued action after the turn starts,
    # which the resolver does not model; those turns are named rather than scored.
    if action_overriding_effects(lines):
        report.skipped["action overridden mid-turn"] += 1
        return

    result = resolve_turn(reg, before, chosen, budget=Budget.deterministic(roll))
    report.branch_counts[len(result.branches)] += 1
    for name in result.unmodelled:
        report.unmodelled[name] += 1
    if len(result.branches) != 1:
        report.skipped[f"{len(result.branches)} branches under a deterministic budget"] += 1
        return

    # The resolver deliberately stops at a pending replacement: which Pokemon comes in
    # after a self-switch is the player's choice, and after a forced switch it is drawn at
    # random. Showdown resolves both immediately, so these turns are classified rather
    # than scored -- the resolver is not wrong, it has handed the decision back.
    undetermined = [
        name for name in result.unmodelled if name.startswith(("selfSwitch", "forceSwitch"))
    ]
    if undetermined:
        report.skipped["pending replacement (resolver defers to a choice)"] += 1
        return

    ours = canonical(result.branches[0].position)
    theirs = canonical(Position.from_json(after_json))

    report.compared += 1
    differences = [k for k in ours if ours[k] != theirs.get(k)]
    if not differences:
        report.matched += 1
        return

    if result.unmodelled:
        report.flagged_divergences += 1
    else:
        report.silent_divergences += 1

    moves = [
        s.move_id
        for a in chosen
        for s in a.slots
        if isinstance(s, MoveAction)
    ]
    abilities = sorted(
        {
            m.ability
            for side in before.sides
            for m in side.active_pokemon()
            if m is not None
        }
    )
    items = sorted(
        {
            m.item or ""
            for side in before.sides
            for m in side.active_pokemon()
            if m is not None
        }
    )
    for key in differences:
        kind = field_kind(key)
        report.by_field[kind] += 1
        report.attributed[(kind, ",".join(sorted(moves)), ",".join(abilities), ",".join(items))] += 1

    if not result.unmodelled and len(report.examples) < 10:
        detail = "; ".join(
            f"{k}: ours {ours[k]!r} vs showdown {theirs.get(k)!r}" for k in differences[:5]
        )
        report.examples.append(
            f"turn {before.turn} [{describe_actions(reg, chosen)}] -> {detail}\n"
            f"      our events: {' / '.join(result.branches[0].events[:12])}"
        )


def run(
    battles: int, roll: int, seed: int, max_turns: int, quiet: bool = True
) -> Report:
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    register_mega_stones(reg)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    report = Report()
    policy = RandomnessPolicy(
        damage_roll=roll,
        accuracy="hit",
        crit=False,
        secondary=False,
        multihit="min",
        speed_tie="keep",
    )

    with Oracle() as oracle:
        for _ in range(battles):
            teams = [
                [TeamSet.from_json(s.to_team_set_json(reg)) for s in sample_team(rng, reg, prior)]
                for _ in range(2)
            ]
            handle = oracle.create(
                FORMAT_ID,
                teams[0],
                teams[1],
                seed=tuple(int(rng.integers(1, 60000)) for _ in range(4)),  # type: ignore[arg-type]
                policy=policy,
            )
            handle.step(["team 1,2,3,4", "team 1,2,3,4"])

            for _ in range(max_turns):
                before = Position.from_json(handle.position)
                if before.ended:
                    break
                chosen: list[SideAction] = []
                choices: list[str | None] = []
                forced = False
                for side_index in range(2):
                    request = handle.requests[side_index]
                    if request and request.get("forceSwitch"):
                        forced = True
                        choices.append("default")
                        continue
                    if not request or request.get("wait"):
                        choices.append(None)
                        continue
                    pick = py_rng.choice(side_actions(reg, before, side_index))
                    chosen.append(pick)
                    choices.append(pick.to_choice())
                if all(c is None for c in choices):
                    break
                handle.step(choices)
                if handle.choice_errors:
                    break
                if forced or len(chosen) != 2:
                    report.skipped["replacement turn"] += 1
                    continue
                compare_turn(
                    reg, before, chosen, handle.position, handle.log, roll, report
                )
            handle.close()

    if not quiet:
        print(report.render())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battles", type=int, default=25)
    ap.add_argument("--roll", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=10)
    args = ap.parse_args()
    run(args.battles, args.roll, args.seed, args.max_turns, quiet=False)


if __name__ == "__main__":
    main()
