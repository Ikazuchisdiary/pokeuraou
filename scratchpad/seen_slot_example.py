# 出どころ: IKA-116 の棚卸しセッション（9/23）の一時 scratchpad にあった seen_slot_example.py （sha256 ff468c72e66dd840、CRLF）を、改行を LF にしただけで写した。IKA-117/118 の受け入れの再生に使う。
"""One concrete game where the carried slot set forgets a shown Pokemon, end to end.

Prints the decision sequence for the side concerned (who is in which slot, what `shown`
holds) and then runs the real `completions` on that decision, with the same `seen` that
`play_game` would have passed, to show the belief offering benches without the Pokemon
that has already been on the field.
"""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
POOL = Path(sys.argv[2])
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.hidden import completions, seen_slots  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import SampledSet  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

reg = load_regulation("gen9championsvgc2026regmb")
register_mega_stones(reg)


def sheet_of(team_json):
    return [
        SampledSet(
            species=m["species"], ability=m["ability"], item=m["item"],
            nature=m["nature"], moves=tuple(m["moves"]),
            sp={k: m["sp"].get(k, 0) for k in ("hp", "atk", "def", "spa", "spd", "spe")},
        )
        for m in team_json
    ]


for path in sorted(POOL.glob("*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            game = json.loads(line)
            shown = [frozenset(), frozenset()]
            seen_species = [set(), set()]
            trail = []
            for d in game["decisions"]:
                pos = Position.from_json(d["position"])
                shown = [seen_slots(pos, i, shown[i]) for i in (0, 1)]
                side = 1
                mons = pos.sides[side].pokemon
                for mon in mons:
                    if mon.active_index is not None or mon.fainted:
                        seen_species[side].add(mon.species)
                trail.append(
                    f"  turn {d['turn']:>2} {d['kind']:<11} slots "
                    + " | ".join(
                        f"{s}:{m.species}{'*' if m.active_index is not None else ''}"
                        f"{' hp' + str(m.hp) + '/' + str(m.maxhp) if m.hp != m.maxhp else ''}"
                        for s, m in enumerate(mons)
                    )
                    + f"   shown={sorted(shown[side])}"
                )
                covered = {mons[s].species for s in shown[side]}
                missing = [m.species for m in mons
                           if m.species in seen_species[side] and m.species not in covered]
                if missing and d["kind"] == "move":
                    print(f"{path.name} gameIndex={game.get('gameIndex')} side {side}")
                    print("\n".join(trail))
                    print(f"\n  already on the field earlier, treated as unseen now: {missing}")
                    foe_sets = {m['species']: m for m in game['foeTeam']}
                    six = game["foeSix"]
                    sheet = [s for s in sheet_of(game["foeTeam"])]
                    # foeTeam is the four brought; the sheet of six is foeSix (species only).
                    # Rebuild the six from the four plus sheet-only species with dummy sets
                    # is not needed for the point: show the candidate list completions uses.
                    from pokeuraou.hidden import shown_species
                    on_board = shown_species(pos, side, shown[side])
                    print(f"  shown_species (what completions conditions on): {sorted(on_board)}")
                    print(f"  sheet of six: {six}")
                    cand = [s for s in six if s not in on_board]
                    print(f"  candidates completions would draw the unseen slots from: {cand}")
                    raise SystemExit(0)
