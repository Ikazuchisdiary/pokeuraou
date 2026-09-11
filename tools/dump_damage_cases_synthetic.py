"""Cases built to make every modifier in the effect table fire at least once.

Self-play reaches the metagame's abilities, which is a small slice of the table: 6 games
touched 4 of the 48 abilities that carry a damage modifier. The rest would be transcribed
and never checked, and an unexercised entry is exactly where a transcription error lives.

So these cases are constructed rather than played: each ability and item is placed on the
attacker and on the defender in turn, against a grid of moves chosen to cover the flags
the predicates read (contact, sound, punch, bite, slicing, pulse, recoil, secondary, low
base power), with weather, terrain, status, screens and HP varied around it.

They are not all reachable positions -- Huge Power on a Charizard is not a legal battle.
That is deliberate and it is what makes them a transcription test: the question here is
whether two implementations of the same table agree on the same input, and Python stays
the oracle. Reachability is `tools/diff_damage.py`'s question, against Showdown.

    uv run python tools/dump_damage_cases_synthetic.py --out rust/cases-synthetic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pokeuraou.battler import Battler, FieldState  # noqa: E402
from pokeuraou.damage import calculate, register_mega_stones  # noqa: E402
from pokeuraou.effects import (  # noqa: E402
    ABILITY_MODIFIERS,
    ITEM_MODIFIERS,
    RESIST_BERRIES,
    TYPE_BOOST_ITEMS,
    TYPE_CHANGING_ABILITIES,
    TYPE_IMMUNITY_ABILITIES,
)
from pokeuraou.moveinfo import MoveContext  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from tools.dump_damage_cases import dump_battler, dump_field, dump_move_ctx  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"

#: Flags whose predicates the table reads, plus the shapes that are not flags.
WANTED_FLAGS = ("contact", "sound", "punch", "bite", "slicing", "pulse", "bullet", "powder")


def pick_moves(reg) -> list[str]:  # noqa: ANN001
    """One move per flag the predicates read, plus the non-flag shapes."""
    chosen: dict[str, str] = {}
    for flag in WANTED_FLAGS:
        for move in reg.moves.values():
            if move.base_power > 0 and flag in move.flags:
                chosen.setdefault(flag, move.id)
                break
    for label, test in (
        ("physical", lambda m: m.category == "Physical" and m.base_power >= 80),
        ("special", lambda m: m.category == "Special" and m.base_power >= 80),
        ("lowbp", lambda m: 0 < m.base_power <= 60),
        ("recoil", lambda m: bool(m.raw.get("recoil"))),
        ("secondary", lambda m: bool(m.raw.get("secondaries"))),
        ("status", lambda m: m.category == "Status"),
    ):
        for move in reg.moves.values():
            if test(move):
                chosen.setdefault(label, move.id)
                break
    # A few named ones whose own code the port reimplements.
    for move_id in ("facade", "knockoff", "weatherball", "freezedry", "lowkick", "electroball"):
        if move_id in reg.moves:
            chosen[move_id] = move_id
    return sorted(set(chosen.values()))


def base_battler(reg, species: str, ability: str, item: str | None, **kw) -> Battler:  # noqa: ANN001
    spec = reg.species[species]
    stats = np.array([[175, 150, 120, 130, 110, 100]], dtype=np.int64)
    return Battler(
        species=species,
        types=tuple(kw.pop("types", spec.types)),
        ability=ability,
        item=item,
        level=50,
        stats=stats,
        hp=np.array([kw.pop("hp", 175)], dtype=np.int64),
        maxhp=np.array([175], dtype=np.int64),
        boosts=kw.pop("boosts", {}),
        status=kw.pop("status", None),
        volatiles=frozenset(kw.pop("volatiles", ())),
        gender=kw.pop("gender", "M"),
        ability_state=kw.pop("ability_state", {}),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="rust/cases-synthetic.json")
    args = ap.parse_args()

    reg = load_regulation(FORMAT_ID)
    register_mega_stones(reg)
    moves = pick_moves(reg)

    abilities = sorted(
        set(ABILITY_MODIFIERS) | set(TYPE_CHANGING_ABILITIES) | set(TYPE_IMMUNITY_ABILITIES)
    )
    items = sorted(set(ITEM_MODIFIERS) | set(TYPE_BOOST_ITEMS) | set(RESIST_BERRIES))

    fields = [
        FieldState(weather=w, terrain=t, active_abilities=(("intimidate",), ("blaze",)))
        for w in (None, "sunnyday", "raindance", "sandstorm", "snowscape")
        for t in (None, "electricterrain", "grassyterrain", "psychicterrain", "mistyterrain")
    ]
    fields.append(
        FieldState(
            weather=None,
            terrain=None,
            side_conditions=(frozenset(), frozenset({"reflect", "lightscreen"})),
            active_abilities=(("friendguard",), ("friendguard",)),
        )
    )
    fields.append(
        FieldState(weather=None, terrain=None, active_abilities=(("fairyaura",), ("aurabreak",)))
    )

    # Two species with different types, so STAB and effectiveness both vary.
    attacker_species = "incineroar"
    defender_species = "sneasler"

    cases: list[dict] = []
    seen: set[str] = set()

    def record(attacker, defender, move_id, field, **kw):  # noqa: ANN001
        result = calculate(reg, attacker, defender, move_id, field, **kw)
        case = {
            "move": move_id,
            "defender_side": int(kw.get("defender_side", 1)),
            "spread": bool(kw.get("spread", False)),
            "crit": bool(kw.get("crit", False)),
            "base_power_override": kw.get("base_power_override"),
            "attacker": dump_battler(attacker),
            "defender": dump_battler(defender),
            "attacker_is_defender": False,
            "field": dump_field(field),
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

    ctx = MoveContext()
    # -- abilities, on each side, against the move grid ---------------------
    for ability in abilities:
        for on_attacker in (True, False):
            for move_id in moves:
                for field in fields[:6] + fields[-2:]:
                    attacker = base_battler(
                        reg,
                        attacker_species,
                        ability if on_attacker else "blaze",
                        None,
                        hp=58 if ability in ("overgrow", "blaze", "torrent", "swarm", "defeatist") else 175,
                    )
                    defender = base_battler(
                        reg, defender_species, "blaze" if on_attacker else ability, None
                    )
                    record(attacker, defender, move_id, field, move_ctx=ctx)

    # -- items, likewise ----------------------------------------------------
    for item in items:
        for on_attacker in (True, False):
            for move_id in moves:
                for field in fields[:3]:
                    attacker = base_battler(
                        reg, attacker_species, "blaze", item if on_attacker else None
                    )
                    defender = base_battler(
                        reg, defender_species, "blaze", None if on_attacker else item
                    )
                    record(attacker, defender, move_id, field, move_ctx=ctx)

    # -- the shapes that are not ability or item ----------------------------
    for move_id in moves:
        for crit in (False, True):
            for spread in (False, True):
                for status in (None, "brn", "psn", "par"):
                    for boosts in ({}, {"atk": 2, "spa": -2}, {"def": 3, "spd": -1}):
                        attacker = base_battler(
                            reg, attacker_species, "blaze", None, status=status, boosts=boosts
                        )
                        defender = base_battler(
                            reg, defender_species, "blaze", None, boosts=boosts, hp=90
                        )
                        record(
                            attacker,
                            defender,
                            move_id,
                            fields[0],
                            crit=crit,
                            spread=spread,
                            move_ctx=ctx,
                        )

    out = Path(args.out)
    out.write_text(
        json.dumps({"format_id": reg.meta.format_id, "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{len(cases)} synthetic cases -> {out}")
    print(f"  moves      {len(moves)}: {', '.join(moves)}")
    print(f"  abilities  {len(abilities)}")
    print(f"  items      {len(items)}")


if __name__ == "__main__":
    main()
