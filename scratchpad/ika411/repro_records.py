"""IKA-411 step 1: the three IKA-410 examples, seen sets and completions before/after the Mega.

    python repro_records.py

For each game: at every decision up to d1 prints, for the side whose bench is hidden from
the example's seat (the foe), the party (species, base_species, is_mega, active), the
carried identities, seen slots, shown_species and the number of completions (no weights).
"""

import json
from pathlib import Path

from pokeuraou.hidden import completions, seen_identities, seen_slots, shown_species
from pokeuraou.pool import load_pool
from pokeuraou.position import Position
from pokeuraou.teams import SampledSet  # noqa: F401  (type only)

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
pool = load_pool(str(M / "data/pool/regmc-matchupweb.json"))
reg = pool.reg

WANT = {("games-b11-worker15.jsonl", 321): 0, ("games-b07-worker9.jsonl", 11): 1,
        ("games-b04-worker18.jsonl", 670): 0}

games = {}
for ln in Path("C:/tmp/ika410/games.jsonl").read_bytes().splitlines():
    key, body = ln.split(b"\t", 1)
    k = json.loads(key)
    if (k["file"], k["line"]) in WANT:
        games[(k["file"], k["line"])] = json.loads(body)


def sheet_of(game, side):
    from pokeuraou.teams import SampledSet as S
    six = game["ownSix"] if side == 0 else game["foeSix"]
    return [S(**{k: v for k, v in s.items()}) if isinstance(s, dict) else s for s in six]


for key, seat in WANT.items():
    g = games[key]
    foe = 1 - seat
    print("=" * 100)
    print(key, "game", g["gameIndex"], "seat", seat, "foe side", foe)
    six = g["ownSix"] if foe == 0 else g["foeSix"]
    print("foe six:", [s["species"] if isinstance(s, dict) else s for s in six])
    carried = frozenset()
    for n, d in enumerate(g["decisions"][:4]):
        pos = Position.from_json(d["position"])
        carried = seen_identities(pos, foe, carried)
        slots = seen_slots(pos, foe, carried)
        print(f"-- decision {n} turn {d['turn']} kind {d['kind']}")
        for mon in pos.sides[foe].pokemon:
            print(f"   slot {mon.slot} species={mon.species:16s} base_species={mon.base_species:12s} "
                  f"mega={mon.is_mega} active={mon.active_index} hp={mon.hp}/{mon.maxhp}")
        print("   recorded shownIdentities:", d.get("shownIdentities"))
        print("   carried identities:", sorted(carried), " seen slots:", sorted(slots))
        on_board = shown_species(pos, foe, slots)
        print("   shown_species:", sorted(on_board))
        print("   rankViews:", d.get("rankViews"))
