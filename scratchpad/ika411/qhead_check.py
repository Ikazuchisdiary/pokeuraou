"""IKA-411: qhead.decision_views still rebuilds the recorded completion on a pre-fix record
whose Floette-Eternal Mega Evolved (the record's own base_species, "Floette", is used)."""

import json
from pathlib import Path

from pokeuraou.pool import load_pool
from pokeuraou.qhead import decision_views

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
pool = load_pool(str(M / "data/pool/regmc-matchupweb.json"))
lines = (M / "data/selfplay-mc3/games-b07-worker9.jsonl").read_bytes().splitlines()
ok = bad = 0
for n in (11,):
    g = json.loads(lines[n])
    for d in g["decisions"]:
        if d["kind"] != "move":
            continue
        try:
            views = decision_views(pool.reg, pool, g, d)
            ok += 1
            print("turn", d["turn"], [(v.side, v.species) for v in views])
        except ValueError as problem:
            bad += 1
            print("turn", d["turn"], "ERROR", problem)
print("ok", ok, "bad", bad)
