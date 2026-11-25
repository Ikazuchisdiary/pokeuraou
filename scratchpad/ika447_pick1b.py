"""IKA-447 stage 1b: rank the 16-team sieve by the raw value (no correction) and pick the
top 200 plus 50 random candidates (seed 447) to recheck the sieve's r on all 65 teams."""
import json
import random
from pathlib import Path

D = Path("C:/tmp/ika447")
cands = json.loads((D / "cands2.json").read_bytes())["candidates"]
win = json.loads((D / "v16-winner/xd11eafad0445edc.json").read_bytes())["mean"]
rows = []
for c in cands:
    p = D / "v16" / f"{c['id']}.json"
    if not p.exists():
        continue
    rows.append((json.loads(p.read_bytes())["mean"], c["id"], c.get("base", ""), c["name"]))
rows.sort(reverse=True)
print(len(rows), "valued of", len(cands), "winner v16", round(win, 4))
print("above winner:", sum(r[0] > win for r in rows))
top = [r[1] for r in rows[:200]]
rnd = random.Random(447).sample([r[1] for r in rows[200:]], 50)
(D / "stage1b-ids.txt").write_bytes(",".join(top + rnd).encode())
(D / "stage1b-ids.json").write_bytes(json.dumps({"top": top, "random": rnd}).encode())
for r in rows[:20]:
    print(f"{r[0]:.4f} {r[1]} {r[3][:100]}")
print("200th", round(rows[199][0], 4), "median", round(rows[len(rows) // 2][0], 4))
bases = {}
for r in rows[:200]:
    bases[r[3].split(" / ")[0]] = bases.get(r[3].split(" / ")[0], 0) + 1
print(bases)
