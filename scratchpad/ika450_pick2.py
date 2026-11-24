"""IKA-450 stage 2 pick: rank the 65-team values (+0.012 per unseen id), check the 16-vs-65
agreement on the 50 random candidates, write stage2-teams.json and pool-stage2.json."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
H = Path("C:/tmp/ika450")
sys.path.insert(0, "C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/pokeuraou-priority-tasks-27bb3d/src")
exec(open(H / "pick1b.py").read().split("cands = json.load")[0])  # known, sid, unseen
cands = {c["id"]: c for c in json.load(open(H / "cands.json", encoding="utf-8"))["candidates"]}
win = json.load(open(H / "winner.json", encoding="utf-8"))["candidates"][0]
ids = json.load(open(H / "stage1b-ids.json"))
def mean(d, i): return json.loads((H / d / f"{i}.json").read_bytes())["mean"]
v16 = np.array([mean("v16", i) for i in ids["random"]]); v65 = np.array([mean("v65", i) for i in ids["random"]])
print("random 50: r(16, 65) =", round(float(np.corrcoef(v16, v65)[0, 1]), 3))
t16 = np.array([mean("v16", i) for i in ids["top"]]); t65 = np.array([mean("v65", i) for i in ids["top"]])
print("top 200: r(16, 65) =", round(float(np.corrcoef(t16, t65)[0, 1]), 3), "mean drop", round(float((t16 - t65).mean()), 4))
wv = json.loads(Path("C:/tmp/ika443/v65/xd11eafad0445edc.json").read_bytes())["mean"]
print(f"winner xd11 v65 {wv:.4f} unseen {unseen(win['team'])}")
rows = sorted(((mean("v65", i) + 0.012 * unseen(cands[i]["team"]), mean("v65", i), unseen(cands[i]["team"]), i)
               for i in ids["top"] + ids["random"]), reverse=True)
for r in rows[:25]:
    print(f"{r[0]:.4f} {r[1]:.4f} u{r[2]} {r[3]} {cands[r[3]]['name'][:100]}")
top = [r[3] for r in rows[:10]]
json.dump({"candidates": [win["id"], *top], "rows": rows[:25]}, open(H / "stage2-teams.json", "w"))
json.dump({"candidates": [win, *cands.values()]}, open(H / "cands-all.json", "w", encoding="utf-8"), ensure_ascii=False)
PY = "C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe"
WT = "C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/pokeuraou-priority-tasks-27bb3d"
subprocess.run([PY, f"{WT}/tools/team_search.py", "pool-with", "--candidates", str(H / "cands-all.json"),
                "--ids", win["id"], *top, "--name", "ika450-stage2", "--out", str(H / "pool-stage2.json")],
               check=True, env={**__import__("os").environ, "PYTHONPATH": f"{WT}/src"})
print("wrote pool-stage2.json")
