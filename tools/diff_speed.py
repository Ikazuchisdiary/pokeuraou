"""Differential test of effective Speed against Showdown.

Compares ``pokeuraou.speed.effective_speed`` with Showdown's own ``getStat('spe')`` for
every active Pokemon on every turn. Checking Speed directly rather than inferring it from
a wrong turn order keeps the two apart: an order divergence could be a Speed bug or an
ordering bug, and this says which.

    uv run python tools/diff_speed.py --battles 60
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos, sample_team  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.speed import effective_speed  # noqa: E402
from pokeuraou.view import active_battlers, field_state  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


@dataclass
class Report:
    compared: int = 0
    matched: int = 0
    #: (ability, item, status, side conditions) -> count, for attributing a divergence.
    divergences: Counter[tuple[str, ...]] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)

    @property
    def divergence_rate(self) -> float:
        return 0.0 if not self.compared else 1 - self.matched / self.compared

    def render(self) -> str:
        out = [
            f"compared {self.compared} Speed values, matched {self.matched}, "
            f"divergence rate {self.divergence_rate * 100:.3f}%"
        ]
        if self.divergences:
            out.append("  divergences by (ability / item / status / side conditions / weather):")
            for key, count in self.divergences.most_common(15):
                out.append(f"    {count:5d}  {' / '.join(k or '-' for k in key)}")
        for line in self.examples[:15]:
            out.append("  " + line)
        return "\n".join(out)


def compare(reg: Regulation, pos: Position, speeds: list[dict], report: Report) -> None:
    fs = field_state(pos, reg)
    battlers = [active_battlers(reg, side) for side in pos.sides]
    for row in speeds:
        side_index, slot = row["side"], row["slot"]
        mon = battlers[side_index][slot] if slot < len(battlers[side_index]) else None
        if mon is None:
            continue
        conditions = frozenset(c.id for c in pos.sides[side_index].side_conditions)
        ours = int(effective_speed(reg, mon, fs, conditions)[0])
        theirs = int(row["effectiveSpe"])
        report.compared += 1
        if ours == theirs:
            report.matched += 1
            continue
        key = (
            row["ability"] or "",
            row["item"] or "",
            row["status"] or "",
            ",".join(sorted(conditions)),
            fs.weather or "",
        )
        report.divergences[key] += 1
        if len(report.examples) < 15:
            report.examples.append(
                f"turn {pos.turn} p{side_index + 1}{'ab'[slot]} {row['species']} "
                f"({row['ability']},{row['item']},{row['status']}) stored={row['storedSpe']} "
                f"boost={row['boostSpe']} cond={sorted(conditions)} weather={fs.weather}: "
                f"ours {ours}, showdown {theirs} (x{theirs / max(ours, 1):.3f})"
            )


def run(battles: int, seed: int, max_turns: int, quiet: bool = True) -> Report:
    reg = load_regulation(FORMAT_ID)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    report = Report()
    policy = RandomnessPolicy(damage_roll=8, accuracy="hit", crit=False, secondary=True)

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
                pos = Position.from_json(handle.position)
                if pos.ended:
                    break
                compare(reg, pos, handle.action_speeds(), report)
                choices: list[str | None] = []
                for side_index in range(2):
                    request = handle.requests[side_index]
                    if request and request.get("forceSwitch"):
                        choices.append("default")
                        continue
                    if not request or request.get("wait"):
                        choices.append(None)
                        continue
                    choices.append(py_rng.choice(side_actions(reg, pos, side_index)).to_choice())
                if all(c is None for c in choices):
                    break
                handle.step(choices)
                if handle.choice_errors:
                    break
            handle.close()

    if not quiet:
        print(report.render())
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battles", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args()
    run(args.battles, args.seed, args.max_turns, quiet=False)


if __name__ == "__main__":
    main()
