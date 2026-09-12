"""The same dump as `tools/dump_damage_cases.py`, but from self-play against the field.

One scenario exercises the abilities and items of six Pokemon. Self-play draws the
opponent from the tournament standings, so a short run reaches many more of the effect
table's entries -- which is what a port has to be measured on, not one team's worth.

    uv run python tools/dump_damage_cases_selfplay.py --games 6 --out rust/cases-field.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pokeuraou import damage as damage_mod  # noqa: E402
from pokeuraou import resolve as resolve_mod  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.selfplay import generate  # noqa: E402
from pokeuraou.standings import find_cached_standings, load_standings  # noqa: E402
from pokeuraou.teams import load_archetypes, load_roster, usable_archetypes  # noqa: E402
from tools.dump_damage_cases import dump_battler, dump_field, dump_move_ctx  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--archetypes", default="wcs2026-regmb")
    ap.add_argument("--out", default="rust/cases-field.json")
    args = ap.parse_args()

    reg = load_regulation(FORMAT_ID)
    register_mega_stones(reg)
    chaos = find_cached_chaos(FORMAT_ID)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    roster = load_roster(args.roster)
    _reg, declared = load_archetypes(args.archetypes)
    archetypes, _blocked = usable_archetypes(prior, declared)
    standings_path = find_cached_standings(2026, "worlds")
    standings = load_standings(standings_path, reg) if standings_path else None

    cases: list[dict] = []
    seen: set[str] = set()
    real = damage_mod.calculate

    def recording(reg_, attacker, defender, move_id, field_state, **kw):  # noqa: ANN001
        result = real(reg_, attacker, defender, move_id, field_state, **kw)
        if attacker.n != 1 or defender.n != 1:
            return result
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

    generate(
        reg,
        prior,
        roster,
        archetypes,
        games=args.games,
        seed=args.seed,
        objective=OBJECTIVES["hp-share"],
        search_limit=args.limit,
        max_turns=args.max_turns,
        standings=standings,
    )

    out = Path(args.out)
    out.write_text(
        json.dumps({"format_id": reg.meta.format_id, "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )
    abilities = {c["attacker"]["ability"] for c in cases} | {
        c["defender"]["ability"] for c in cases
    }
    items = {c["attacker"]["item"] for c in cases} | {c["defender"]["item"] for c in cases}
    print(f"\n{len(cases)} distinct cases -> {out}")
    print(f"  moves     {len({c['move'] for c in cases})}")
    species = {c["attacker"]["species"] for c in cases} | {
        c["defender"]["species"] for c in cases
    }
    print(f"  species   {len(species)}")
    print(f"  abilities {len(abilities)}")
    print(f"  items     {len(items - {None})}")


if __name__ == "__main__":
    main()
