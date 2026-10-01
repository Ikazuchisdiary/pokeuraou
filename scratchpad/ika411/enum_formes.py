"""IKA-411: which M-C species change species mid-battle while their set id is not to_id(baseSpecies).

For each legal species S: the in-battle species it can become (Mega via mega_by_species, and the
dump's other battle-only formes), and whether `shown_species` as it stands (species + base_species
as `_make_pokemon` sets it, i.e. the dex's baseSpecies) would still contain to_id(S) afterwards.
"""

import json
import sys
from collections import Counter

from pokeuraou.pool import load_pool
from pokeuraou.regulation import load_regulation, to_id

reg = load_regulation("gen9championsvgc2026regmc")
raw = json.loads(open(reg.source, "rb").read())

pool = load_pool(sys.argv[1]) if len(sys.argv) > 1 else None

# mega targets
megas = {}
for (sid, item), target in reg.mega_by_species.items():
    megas.setdefault(sid, set()).add((item, target))

print("mega-capable species:", len(megas))
bad = []
for sid, pairs in sorted(megas.items()):
    base = to_id(reg.species[sid].base_species)
    for item, target in sorted(pairs):
        tbase = to_id(reg.species[target].base_species) if target in reg.species else None
        after = {target, base}  # _make_pokemon's base_species is the dex baseSpecies of the SET
        ok = sid in after
        print(f"  {sid:20s} --{item}--> {target:22s} set-base={base:14s} mega-base={tbase} {'ok' if ok else 'FORGETS'}")
        if not ok:
            bad.append((sid, item, target))
print("forgets after mega:", bad)

# other battle-only forme changes in the dump
keys = set()
for s in raw["species"] if isinstance(raw["species"], list) else raw["species"].values():
    keys |= set(s)
print("species fields:", sorted(keys))

if pool is not None:
    c = Counter()
    teams = 0
    for team in pool.teams:
        teams += 1
        for s in team.sets:
            if to_id(reg.species[s.species].base_species) != s.species:
                c[(s.species, s.item, reg.mega_target(s.species, s.item))] += 1
    print("pool teams:", teams)
    print("pool sets whose set id != dex baseSpecies id:", sorted(c.items()))
