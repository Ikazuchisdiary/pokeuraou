import json, re
from collections import Counter
reg = json.load(open("C:/tmp/ika413/wt/configs/regulations/gen9championsvgc2026regmc.json"))
moves = {m["id"]: m for m in reg["moves"]}
sp = {s["id"]: s for s in reg["species"]}
def tid(x): return re.sub(r"[^a-z0-9]", "", x.lower())
print([i["id"] for i in reg["items"] if "floet" in i["id"]], [s["id"] for s in reg["species"] if "floette" in s["id"]])
pool = json.load(open("C:/Users/Ikazuchi/repos/pokeuraou/data/pool/pastes.json"))
HOLD = {"friendguard", "cloudnine", "fairyaura", "darkaura", "aurabreak", "airlock"}
def mega_ability(m):
    it = tid(m.get("item") or "")
    mm = reg["megaMap"].get(it)
    if not mm: return None, None
    base = tid(m["species"].split("-Mega")[0]) if False else None
    for b, mega in mm.items():
        if tid(m["species"]) == b or tid(m.get("baseSpecies", "")) == b:
            return mega, [tid(a) for a in sp[mega]["abilities"]]
    return None, None
rows = []
for t in pool:
    for m in t["team"]:
        ab = tid(m["ability"])
        mega, mab = mega_ability(m)
        if ab in HOLD or (mab and set(mab) & HOLD):
            rows.append((t["name"], m["species"], ab, mega, mab))
for r in rows: print(r)
print(len({r[0] for r in rows}), "teams of", len(pool))
