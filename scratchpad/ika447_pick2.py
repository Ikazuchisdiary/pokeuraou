"""IKA-447 stage 2 pick: rank the 65-team values (raw, no correction), check the 16-vs-65
agreement on the 50 random candidates, write stage2-teams.json and pool-stage2.json."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

H = Path("C:/tmp/ika447")
cands = {c["id"]: c for c in json.loads((H / "cands2.json").read_bytes())["candidates"]}
win = json.loads(Path("C:/tmp/ika450/winner.json").read_bytes())["candidates"][0]
assert win["id"] == "xd11eafad0445edc" and win["id"] not in cands
ids = json.loads((H / "stage1b-ids.json").read_bytes())


def mean(d, i):
    return json.loads((H / d / f"{i}.json").read_bytes())["mean"]


v16 = np.array([mean("v16", i) for i in ids["random"]])
v65 = np.array([mean("v65", i) for i in ids["random"]])
print("random 50: r(16, 65) =", round(float(np.corrcoef(v16, v65)[0, 1]), 3))
t16 = np.array([mean("v16", i) for i in ids["top"]])
t65 = np.array([mean("v65", i) for i in ids["top"]])
print("top 200: r(16, 65) =", round(float(np.corrcoef(t16, t65)[0, 1]), 3),
      "mean drop", round(float((t16 - t65).mean()), 4))
wv = json.loads(Path("C:/tmp/ika443/v65/xd11eafad0445edc.json").read_bytes())["mean"]
print(f"winner xd11 v65 {wv:.4f}")
rows = sorted(((mean("v65", i), i) for i in ids["top"] + ids["random"]), reverse=True)
print("above winner on 65:", sum(r[0] > wv for r in rows))
for r in rows[:25]:
    print(f"{r[0]:.4f} {r[1]} {cands[r[1]]['name'][:110]}")
top = [r[1] for r in rows[:10]]
(H / "stage2-teams.json").write_bytes(json.dumps({"candidates": [win["id"], *top], "rows": rows[:25]}).encode())
(H / "cands-all.json").write_bytes(
    json.dumps({"candidates": [win, *cands.values()]}, ensure_ascii=False).encode("utf-8"))
PY = "C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe"
WT = "C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/pokeuraou-priority-tasks-27bb3d"
subprocess.run([PY, f"{WT}/tools/team_search.py", "pool-with", "--candidates", str(H / "cands-all.json"),
                "--ids", win["id"], *top, "--name", "ika447-stage2", "--out", str(H / "pool-stage2.json")],
               check=True, env={**os.environ, "PYTHONPATH": f"{WT}/src"})
print("wrote pool-stage2.json")
