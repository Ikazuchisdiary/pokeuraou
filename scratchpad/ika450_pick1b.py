"""IKA-450 stage 1b: rank the 16-team sieve with the IKA-448 correction (+0.012 per id no
pool team has) and pick the top 200 plus 50 random candidates (to recheck the sieve's r)."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, "C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/pokeuraou-priority-tasks-27bb3d/src")
from pokeuraou.pool import load_pool
pool = load_pool("regmc-matchupweb")
known = {"species": set(), "moves": set(), "items": set(), "abilities": set()}
for r in pool.teams:
    for s in r.sets:
        known["species"].add(s.species); known["moves"].update(s.moves)
        if s.item: known["items"].add(s.item)
        if s.ability: known["abilities"].add(s.ability)
def sid(x): return "".join(c for c in str(x).lower() if c.isalnum())
def unseen(team):
    sp = {sid(m["species"]) for m in team} - known["species"]
    mv = {sid(x) for m in team for x in m["moves"]} - known["moves"]
    it = {sid(m["item"]) for m in team if m.get("item")} - known["items"]
    ab = {sid(m["ability"]) for m in team} - known["abilities"]
    return len(sp) + len(mv) + len(it) + len(ab)
cands = json.load(open("C:/tmp/ika450/cands.json", encoding="utf-8"))["candidates"]
rows = []
for c in cands:
    p = Path("C:/tmp/ika450/v16") / f"{c['id']}.json"
    if not p.exists(): continue
    v = json.loads(p.read_bytes())["mean"]
    u = unseen(c["team"])
    rows.append((v + 0.012 * u, v, u, c["id"], c["name"]))
rows.sort(reverse=True)
print(len(rows), "valued of", len(cands))
# the winner's own 16-team value, for reference
top = [r[3] for r in rows[:200]]
rest = [r[3] for r in rows[200:]]
rnd = random.Random(450).sample(rest, 50)
Path("C:/tmp/ika450/stage1b-ids.txt").write_bytes((",".join(top + rnd)).encode())
json.dump({"top": top, "random": rnd}, open("C:/tmp/ika450/stage1b-ids.json", "w"))
for r in rows[:20]:
    print(f"{r[0]:.4f} {r[1]:.4f} u{r[2]}  {r[3]}  {r[4][:90]}")
print("200th adjusted", rows[199][0], "median", rows[len(rows)//2][0])
