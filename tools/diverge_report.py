"""Ranks the causes of silent resolver divergence by statistical lift.

The per-turn report names the whole cast of a divergent turn -- four moves, four abilities,
four items -- which makes a common effect look guilty just for being common. This tool
runs many seeds and, for each individual effect, compares how often it appears in divergent
turns with how often it appears in matching ones:

    lift = P(effect | divergent) / P(effect | matched)

An effect with high lift and a decent count is a cause. An effect with lift near 1 is
merely popular. That turns "what to fix next" into a ranking rather than a guess.

    uv run python tools/diverge_report.py --seeds 12 --battles 10
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from diff_turn import canonical, field_kind  # noqa: E402

from pokeuraou.actions import MoveAction, SideAction, side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn  # noqa: E402
from pokeuraou.speed import action_overriding_effects  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


def effects_in_play(
    reg: Regulation, pos: Position, chosen: list[SideAction]
) -> set[str]:
    """Every named effect that could plausibly be responsible for one turn."""
    out: set[str] = set()
    for action in chosen:
        for slot_action in action.slots:
            if isinstance(slot_action, MoveAction):
                out.add(f"move:{slot_action.move_id}")
    for side in pos.sides:
        for mon in side.active_pokemon():
            if mon is None:
                continue
            out.add(f"ability:{mon.ability}")
            if mon.item:
                out.add(f"item:{mon.item}")
            if mon.status:
                out.add(f"status:{mon.status}")
            for volatile in mon.volatiles:
                out.add(f"volatile:{volatile.id}")
            for volatile in mon.unmodelled_volatiles:
                out.add(f"unmodelled-volatile:{volatile}")
        for condition in side.side_conditions:
            out.add(f"side:{condition.id}")
    if pos.field.weather:
        out.add(f"weather:{pos.field.weather}")
    if pos.field.terrain:
        out.add(f"terrain:{pos.field.terrain}")
    for pseudo in pos.field.pseudo_weather:
        out.add(f"pseudo:{pseudo.id}")
    del reg
    return out


@dataclass
class Aggregate:
    matched: int = 0
    silent: int = 0
    flagged: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    #: effect -> how many matching turns it appeared in.
    in_matched: Counter[str] = field(default_factory=Counter)
    #: effect -> how many silently divergent turns it appeared in.
    in_silent: Counter[str] = field(default_factory=Counter)
    by_field: Counter[str] = field(default_factory=Counter)
    #: Bucketed ratio of the damage we dealt to the damage Showdown dealt. A missing
    #: modifier shows up as a pile at 0.67 (a missing 1.5x), 0.77 (1.3x), 0.5 or 2.0,
    #: which names the mechanic without reading examples one at a time.
    damage_ratio: Counter[str] = field(default_factory=Counter)
    #: effect -> the fields that diverged alongside it, for reading the mechanism off.
    fields_with: dict[str, Counter[str]] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def compared(self) -> int:
        return self.matched + self.silent + self.flagged

    def lift(self, effect: str) -> float:
        """P(effect | silent divergence) / P(effect | match)."""
        if not self.silent or not self.matched:
            return 0.0
        p_silent = self.in_silent[effect] / self.silent
        p_matched = self.in_matched[effect] / self.matched
        # An effect never seen in a matching turn is maximally suspicious, but only if it
        # occurred often enough to mean anything; the caller filters on count.
        return p_silent / p_matched if p_matched > 0 else float(p_silent * self.matched)

    def render(self, top: int, min_count: int) -> str:
        out = [
            f"compared {self.compared} turns: {self.matched} matched, "
            f"{self.silent} silent ({self.silent / max(self.compared, 1) * 100:.2f}%), "
            f"{self.flagged} flagged"
        ]
        if self.by_field:
            out.append(
                "  divergent fields: "
                + ", ".join(f"{k} x{v}" for k, v in self.by_field.most_common(10))
            )
        if self.damage_ratio:
            out.append(
                "  our damage / Showdown's, bucketed: "
                + ", ".join(f"{k} x{v}" for k, v in self.damage_ratio.most_common(12))
            )
        out.append("")
        out.append(
            f"  causes ranked by lift (count >= {min_count}); lift near 1 means the effect "
            "is merely common"
        )
        out.append(f"  {'effect':<42} {'silent':>7} {'matched':>8} {'lift':>7}  fields")
        ranked = [
            (effect, self.lift(effect))
            for effect, count in self.in_silent.items()
            if count >= min_count
        ]
        ranked.sort(key=lambda pair: -pair[1])
        for effect, lift in ranked[:top]:
            fields = self.fields_with.get(effect, Counter())
            field_text = ", ".join(f"{k}x{v}" for k, v in fields.most_common(3))
            out.append(
                f"  {effect:<42} {self.in_silent[effect]:>7} {self.in_matched[effect]:>8} "
                f"{lift:>7.2f}  {field_text}"
            )
        if self.skipped:
            out.append("")
            out.append(
                "  skipped: "
                + ", ".join(f"{k} x{v}" for k, v in self.skipped.most_common(6))
            )
        for effect, lines in list(self.examples.items())[:6]:
            out.append("")
            out.append(f"  -- {effect}")
            for line in lines[:2]:
                out.append(f"     {line}")
        return "\n".join(out)


def run(seeds: int, battles: int, roll: int, max_turns: int) -> Aggregate:
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    register_mega_stones(reg)
    agg = Aggregate()
    policy = RandomnessPolicy(
        damage_roll=roll, accuracy="hit", crit=False, secondary=False,
        multihit="min", speed_tie="keep",
    )
    budget = Budget.deterministic(roll)

    with Oracle() as oracle:
        for seed in range(1, seeds + 1):
            rng = np.random.default_rng(seed)
            py_rng = random.Random(seed)
            for _ in range(battles):
                teams = [
                    [
                        TeamSet.from_json(s.to_team_set_json(reg))
                        for s in sample_team(rng, reg, prior)
                    ]
                    for _ in range(2)
                ]
                handle = oracle.create(
                    FORMAT_ID, teams[0], teams[1],
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
                        agg.skipped["replacement turn"] += 1
                        continue
                    _score(reg, before, chosen, handle.position, handle.log, budget, agg)
                handle.close()
    return agg


def _score(
    reg: Regulation,
    before: Position,
    chosen: list[SideAction],
    after_json: dict[str, Any],
    lines: list[str],
    budget: Budget,
    agg: Aggregate,
) -> None:
    if action_overriding_effects(lines):
        agg.skipped["action overridden mid-turn"] += 1
        return
    result = resolve_turn(reg, before, chosen, budget=budget)
    if len(result.branches) != 1:
        agg.skipped["not a single branch"] += 1
        return
    if any(n.startswith(("selfSwitch", "forceSwitch")) for n in result.unmodelled):
        agg.skipped["pending replacement"] += 1
        return

    effects = effects_in_play(reg, before, chosen)
    ours = canonical(result.branches[0].position)
    theirs = canonical(Position.from_json(after_json))
    differences = [k for k in ours if ours[k] != theirs.get(k)]

    if not differences:
        agg.matched += 1
        for effect in effects:
            agg.in_matched[effect] += 1
        return
    if result.unmodelled:
        agg.flagged += 1
        return

    agg.silent += 1
    _record_damage_ratios(before, ours, theirs, differences, agg)
    kinds = {field_kind(k) for k in differences}
    for kind in kinds:
        agg.by_field[kind] += 1
    for effect in effects:
        agg.in_silent[effect] += 1
        agg.fields_with.setdefault(effect, Counter()).update(kinds)
        bucket = agg.examples.setdefault(effect, [])
        if len(bucket) < 2:
            detail = "; ".join(
                f"{k}: ours {ours[k]!r} vs sd {theirs.get(k)!r}" for k in differences[:3]
            )
            bucket.append(
                f"turn {before.turn} [{' | '.join(a.describe(reg) for a in chosen)}] {detail}"
            )


#: Ratios worth naming, and the modifier each implies is missing.
_NAMED_RATIOS: tuple[tuple[float, str], ...] = (
    (0.5, "we are half (missing a 2x)"),
    (0.67, "we are two thirds (missing a 1.5x)"),
    (0.77, "we are ~0.77 (missing a 1.3x)"),
    (0.83, "we are ~0.83 (missing a 1.2x)"),
    (1.2, "we are 1.2x too high"),
    (1.3, "we are 1.3x too high"),
    (1.5, "we are 1.5x too high"),
    (2.0, "we are 2x too high"),
)


def _record_damage_ratios(
    before: Position,
    ours: dict[str, Any],
    theirs: dict[str, Any],
    differences: list[str],
    agg: Aggregate,
) -> None:
    """Buckets how far off each HP divergence is, as a ratio of damage dealt."""
    start = {}
    for side_index, side in enumerate(before.sides):
        for mon in side.pokemon:
            start[f"p{side_index + 1}.{mon.slot}.hp"] = mon.hp
    for key in differences:
        if not key.endswith(".hp") or key not in start:
            continue
        our_damage = start[key] - ours[key]
        their_damage = start[key] - theirs.get(key, ours[key])
        if their_damage == 0:
            agg.damage_ratio["Showdown dealt none, we dealt some"] += 1
            continue
        if our_damage == 0:
            agg.damage_ratio["we dealt none, Showdown dealt some"] += 1
            continue
        ratio = our_damage / their_damage
        label = f"other ({ratio:.2f})"
        for value, name in _NAMED_RATIOS:
            if abs(ratio - value) <= 0.04:
                label = name
                break
        agg.damage_ratio[label] += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--battles", type=int, default=10)
    ap.add_argument("--roll", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()
    agg = run(args.seeds, args.battles, args.roll, args.max_turns)
    print(agg.render(args.top, args.min_count))


if __name__ == "__main__":
    main()
