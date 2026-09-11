"""Python's own time on the turns the Rust port is measured against.

Same positions, same actions, same recorded budget, so the two numbers are comparable.
Everything that is not the resolver -- reading the fixture, rebuilding positions, building
the action objects -- happens before the clock starts, because the Rust harness does the
same.

    uv run python tools/bench_turn_cases.py rust/turns.json --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import MoveAction, PassAction, SideAction, SwitchAction  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn  # noqa: E402


def build_side(entries: list[dict]) -> SideAction:
    slots = []
    for entry in entries:
        if entry["kind"] == "move":
            slots.append(
                MoveAction(
                    slot=entry["slot"],
                    move_index=entry["moveIndex"],
                    move_id=entry["moveId"],
                    target=entry["target"],
                    mega=entry["mega"],
                )
            )
        elif entry["kind"] == "switch":
            slots.append(
                SwitchAction(
                    slot=entry["slot"],
                    party_index=entry["partyIndex"],
                    species=entry["species"],
                )
            )
        else:
            slots.append(PassAction(slot=entry["slot"]))
    return SideAction(slots=tuple(slots))


def build_budget(raw: dict) -> Budget:
    return Budget(
        damage_rolls=raw["damageRolls"],
        enumerate_crit=raw["enumerateCrit"],
        enumerate_accuracy=raw["enumerateAccuracy"],
        enumerate_status_checks=raw["enumerateStatusChecks"],
        enumerate_secondary=raw["enumerateSecondary"],
        enumerate_speed_ties=raw["enumerateSpeedTies"],
        pinned_policy=raw["pinnedPolicy"],
        max_branches=raw["maxBranches"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("turns")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument(
        "--refused",
        default=None,
        help="JSON list of case indices the Rust port refused, from `pokeuraou-damage turns`",
    )
    args = ap.parse_args()

    doc = json.loads(Path(args.turns).read_text(encoding="utf-8"))
    reg = load_regulation(doc["format_id"])
    register_mega_stones(reg)
    positions = [Position.from_json(p) for p in doc["positions"]]

    prepared = [
        (
            positions[case["position"]],
            [build_side(case["ours"]), build_side(case["theirs"])],
            build_budget(case["budget"]),
        )
        for case in doc["cases"]
    ]

    # The resolver mutates a copy, never the position it is handed, so one prepared list
    # can be replayed -- but check that on the first pass rather than assuming it.
    before = [json.dumps(pos.to_json(), sort_keys=True) for pos, _a, _b in prepared]
    for pos, actions, budget in prepared:
        resolve_turn(reg, pos, actions, budget=budget)
    after = [json.dumps(pos.to_json(), sort_keys=True) for pos, _a, _b in prepared]
    changed = sum(1 for x, y in zip(before, after, strict=True) if x != y)
    print(f"positions mutated by resolving: {changed} (must be 0)")

    def timed(cases: list, label: str) -> float:
        if not cases:
            print(f"{label}: none")
            return 0.0
        started = time.perf_counter()
        for _ in range(args.repeats):
            for pos, actions, budget in cases:
                resolve_turn(reg, pos, actions, budget=budget)
        elapsed = (time.perf_counter() - started) / args.repeats
        print(
            f"{label}: {len(cases)} turns in {elapsed:.3f} s = "
            f"{elapsed / len(cases) * 1e6:.1f} us/turn"
        )
        return elapsed

    total = timed(prepared, "python, all turns")

    if args.refused:
        indices = set(json.loads(Path(args.refused).read_text(encoding="utf-8")))
        refused = [case for index, case in enumerate(prepared) if index in indices]
        resolved = [case for index, case in enumerate(prepared) if index not in indices]
        print()
        refused_time = timed(refused, "python, the turns Rust refuses")
        timed(resolved, "python, the turns Rust resolves")
        print(
            f"\nhybrid projection: Python pays {refused_time:.3f} s of the {total:.3f} s "
            f"whatever Rust does, so the ceiling is {total / max(refused_time, 1e-9):.1f}x"
        )


if __name__ == "__main__":
    main()
