"""IKA-434 stage 1b: pick top-5% rows where the search was right and the model wrong, from the
earliest games (so that few record files are read), and find each game's file and line."""
import json
from pathlib import Path

import numpy as np

HERE = Path("C:/tmp/ika434")
DATA = Path("C:/Users/Ikazuchi/repos/pokeuraou/data")
t = np.load(HERE / "top.npz")
right = (np.abs(t["y"] - t["p_search"]) < 0.25) & (np.abs(t["y"] - t["p_model"]) > 0.5)
print("top rows", len(t["row"]), "search right and model wrong", int(right.sum()))
idx = np.flatnonzero(right)
# One row per game, the earliest games first.
seen, chosen = set(), []
for i in idx[np.argsort(t["game"][idx], kind="stable")]:
    g = int(t["game"][i])
    if g in seen:
        continue
    seen.add(g)
    chosen.append(i)
    if len(chosen) == 14:
        break
games = sorted(int(t["game"][i]) for i in chosen)
meta = json.loads(str(np.load(DATA / "selfplay-mc4-encoded.npz")["meta_json"]))
sources = [s[0] for s in meta["sources"]]
where = {}
gid = 0
for name in sources:
    with (DATA / "selfplay-mc4" / name).open("rb") as fh:
        k = 0
        for line in fh:
            if not line.strip():
                continue
            if gid in games:
                where[gid] = (name, k)
            gid += 1
            k += 1
    if len(where) == len(games):
        break
rows = []
for i in chosen:
    g = int(t["game"][i])
    rows.append({"row": int(t["row"][i]), "game": g, "file": where[g][0], "line": where[g][1],
                 "turn": int(t["turn"][i]), "y": float(t["y"][i]), "model": float(t["p_model"][i]),
                 "search": float(t["p_search"][i]), "mc3": float(t["p_mc3"][i]),
                 "ko": t["ko"][i].tolist()})
    print(rows[-1])
(HERE / "examples.json").write_bytes(json.dumps(rows, indent=1).encode("utf-8"))
