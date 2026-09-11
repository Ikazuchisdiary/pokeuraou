"""Python's own time on the cases the Rust port is measured against.

Same inputs, same call, so the two numbers are comparable. Reconstructs each Battler from
the dump rather than re-running the resolver, because the resolver's share of the time is
a separate question -- this measures `damage.calculate` alone.

    uv run python tools/bench_damage_cases.py rust/cases.json --repeats 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.battler import Battler, FieldState  # noqa: E402
from pokeuraou.damage import calculate, register_mega_stones  # noqa: E402
from pokeuraou.moveinfo import MoveContext  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402


def load_battler(d: dict) -> Battler:
    return Battler(
        species=d["species"],
        types=tuple(d["types"]),
        ability=d["ability"],
        item=d["item"],
        level=d["level"],
        stats=np.array([d["stats"]], dtype=np.int64),
        hp=np.array([d["hp"]], dtype=np.int64),
        maxhp=np.array([d["maxhp"]], dtype=np.int64),
        boosts=dict(d["boosts"]),
        status=d["status"],
        volatiles=frozenset(d["volatiles"]),
        gender=d["gender"],
        ability_state={"protean": True} if d["protean_fired"] else {},
    )


def load_field(d: dict) -> FieldState:
    return FieldState(
        weather=d["weather"],
        terrain=d["terrain"],
        pseudo_weather=frozenset(d["pseudo_weather"]),
        side_conditions=(
            frozenset(d["side_conditions"][0]),
            frozenset(d["side_conditions"][1]),
        ),
        active_abilities=(
            tuple(d["active_abilities"][0]),
            tuple(d["active_abilities"][1]),
        ),
        active_per_half=d["active_per_half"],
    )


def load_ctx(d: dict | None) -> MoveContext | None:
    return None if d is None else MoveContext(**d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()

    doc = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    reg = load_regulation(doc["format_id"])
    register_mega_stones(reg)

    prepared = []
    for case in doc["cases"]:
        prepared.append(
            (
                load_battler(case["attacker"]),
                load_battler(case["defender"]),
                case["move"],
                load_field(case["field"]),
                {
                    "defender_side": case["defender_side"],
                    "spread": case["spread"],
                    "crit": case["crit"],
                    "move_ctx": load_ctx(case["move_ctx"]),
                    "base_power_override": case["base_power_override"],
                },
                case["expect"],
            )
        )

    # The dump is the oracle for the port; check Python still reproduces it here too, so a
    # timing run cannot quietly be timing the wrong thing.
    wrong = 0
    for attacker, defender, move_id, field, kw, expect in prepared:
        result = calculate(reg, attacker, defender, move_id, field, **kw)
        if [int(v) for v in np.asarray(result.rolls)[0]] != expect["rolls"]:
            wrong += 1
    print(f"self-check: {len(prepared) - wrong}/{len(prepared)} reproduce the dump")

    started = time.perf_counter()
    for _ in range(args.repeats):
        for attacker, defender, move_id, field, kw, _expect in prepared:
            calculate(reg, attacker, defender, move_id, field, **kw)
    elapsed = time.perf_counter() - started
    calls = args.repeats * len(prepared)
    print(
        f"speed: {calls} calls in {elapsed:.3f} s = {elapsed / calls * 1e6:.3f} us/call "
        f"({calls / elapsed:,.0f} calls/s)"
    )


if __name__ == "__main__":
    main()
