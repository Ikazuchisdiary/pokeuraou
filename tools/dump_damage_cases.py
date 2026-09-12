"""Every damage call a real matrix fill makes, with the answer Python gave.

A port of the calculator to another language is only worth discussing if it can be held to
the same number on the same inputs. This writes those inputs out: it hooks
:func:`pokeuraou.damage.calculate`, fills the matrix the CLI would fill, and records each
call's arguments next to the 16 rolls it returned.

    uv run python tools/dump_damage_cases.py --out cases.json

The file is the differential test's fixture: Python is the oracle here, and Showdown stays
the oracle for Python (`tools/diff_damage.py`). Nothing in it is a guess -- every field is
read off the Battler the resolver actually built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import damage as damage_mod  # noqa: E402
from pokeuraou import resolve as resolve_mod  # noqa: E402
from pokeuraou.battler import Battler, FieldState  # noqa: E402
from pokeuraou.belief import reduce_for_position  # noqa: E402
from pokeuraou.cli import _modal, build_beliefs, joint_classes  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.moveinfo import MoveContext  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs  # noqa: E402
from pokeuraou.setup import load_scenario, with_spreads  # noqa: E402


def dump_battler(b: Battler) -> dict:
    if b.n != 1:
        raise ValueError(f"expected one particle, got {b.n}")
    return {
        "species": b.species,
        "types": list(b.types),
        "ability": b.ability,
        "item": b.item,
        "level": int(b.level),
        "stats": [int(v) for v in b.stats[0]],
        "hp": int(b.hp[0]),
        "maxhp": int(b.maxhp[0]),
        "boosts": {k: int(v) for k, v in b.boosts.items() if v},
        "status": b.status,
        "volatiles": sorted(b.volatiles),
        "gender": b.gender,
        # Only one key of ability_state reaches the calculator: Protean's "already fired".
        "protean_fired": bool(b.ability_state.get("protean")),
    }


def dump_field(f: FieldState) -> dict:
    return {
        "weather": f.weather,
        "terrain": f.terrain,
        "pseudo_weather": sorted(f.pseudo_weather),
        "side_conditions": [sorted(f.side_conditions[0]), sorted(f.side_conditions[1])],
        "active_abilities": [list(f.active_abilities[0]), list(f.active_abilities[1])],
        "active_per_half": int(f.active_per_half),
    }


def dump_move_ctx(c: MoveContext | None) -> dict | None:
    if c is None:
        return None
    return {
        "weather": c.weather,
        "terrain": c.terrain,
        "side_total_fainted": int(c.side_total_fainted),
        "times_attacked": int(c.times_attacked),
        "target_hurt_this_turn": bool(c.target_hurt_this_turn),
        "damaged_by_target": bool(c.damaged_by_target),
        "previous_move_failed": bool(c.previous_move_failed),
        "hit_index": int(c.hit_index),
        "moving_last": bool(c.moving_last),
        "ally_used_same_move": bool(c.ally_used_same_move),
    }


def build(scenario_path: Path, limit: int, classes_n: int):
    scenario = load_scenario(scenario_path)
    reg = scenario.reg
    register_mega_stones(reg)
    beliefs = build_beliefs(scenario)
    modal = {k: _modal(b) for k, b in beliefs.items()}
    base = with_spreads(scenario, modal) if beliefs else scenario.position.copy()
    active = {}
    for side_index, side in enumerate(base.sides):
        for slot, party in enumerate(side.active):
            if (side_index, party) in beliefs:
                active[(side_index, slot)] = beliefs[(side_index, party)]
    reductions = {k: reduce_for_position(reg, base, k, active) for k in active}
    classes, _covered, _fixed = joint_classes(
        scenario, reductions, beliefs, limit=classes_n
    )
    row = narrow(reg, base, 0, limit=limit).actions
    col = narrow(reg, base, 1, limit=limit).actions
    positions = [with_spreads(scenario, c.assignment, c.hp) for c in classes] or [base]
    return reg, positions, row, col


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="examples/scenario-turn5.json")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--out", default="cases.json")
    args = ap.parse_args()

    reg, positions, row, col = build(Path(args.scenario), args.limit, args.classes)

    cases: list[dict] = []
    seen: set[str] = set()
    real = damage_mod.calculate

    def recording(reg_, attacker, defender, move_id, field_state, **kw):  # noqa: ANN001
        result = real(reg_, attacker, defender, move_id, field_state, **kw)
        case = {
            "move": move_id,
            "defender_side": int(kw.get("defender_side", 1)),
            "spread": bool(kw.get("spread", False)),
            "crit": bool(kw.get("crit", False)),
            "base_power_override": kw.get("base_power_override"),
            "attacker": dump_battler(attacker),
            "defender": dump_battler(defender),
            "attacker_is_defender": attacker is defender,
            "field": dump_field(field_state),
            "move_ctx": dump_move_ctx(kw.get("move_ctx")),
            "expect": {
                "rolls": [int(v) for v in np.asarray(result.rolls)[0]],
                "effectiveness": float(result.effectiveness),
                "type_mod": int(result.type_mod),
                "immune": bool(result.immune),
            },
        }
        key = json.dumps(case, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            cases.append(case)
        return result

    damage_mod.calculate = recording
    resolve_mod.calculate = recording

    objective = OBJECTIVES["hp-share"]
    cross = OBJECTIVES["faints"]
    for pos in positions:
        batched_payoffs(
            reg, pos, row, col, [objective.batch, cross.batch], budget=Budget.matrix()
        )

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {"format_id": reg.meta.format_id, "cases": cases},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    moves = sorted({c["move"] for c in cases})
    print(f"{len(cases)} distinct cases over {len(moves)} moves -> {out}")
    abilities = {c["attacker"]["ability"] for c in cases} | {
        c["defender"]["ability"] for c in cases
    }
    items = {str(c["attacker"]["item"]) for c in cases} | {
        str(c["defender"]["item"]) for c in cases
    }
    print(f"  abilities: {sorted(abilities)}")
    print(f"  items: {sorted(items)}")


if __name__ == "__main__":
    main()
