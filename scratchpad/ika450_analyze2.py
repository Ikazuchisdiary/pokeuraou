"""IKA-450: each focus team's score vs the 65 pool teams, and its paired difference from the
IKA-443 winner (per-opponent means, se over opponents).   analyze2.py <matches dir>"""
import glob, json, sys
from collections import defaultdict
import numpy as np
H = "C:/tmp/ika450"
st = json.load(open(f"{H}/stage2-teams.json"))
pool = json.load(open(f"{H}/pool-stage2.json", encoding="utf-8"))
name = {t["id"]: t["name"] for t in pool["teams"]}
focus = st["candidates"]; win = focus[0]
s = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{sys.argv[1]}/games-worker*.jsonl"):
    for line in open(f, encoding="utf-8"):
        g = json.loads(line)
        # the record is side 0's: ownSix and outcome belong to pool.teams[0] whatever the
        # seatIndex (which names the tested arm's side)
        a, b = g["pool"]["teams"]; o = g["outcome"]
        if a in focus and b not in focus: s[a][b].append(o)
        if b in focus and a not in focus: s[b][a].append(1 - o)
opps = sorted(set.intersection(*(set(s[t]) for t in focus)))
w = np.array([np.mean(s[win][o]) for o in opps])
print(f"{len(opps)} common opponents")
for t in focus:
    x = np.array([np.mean(s[t][o]) for o in opps]); n = sum(len(v) for v in s[t].values())
    d = x - w
    print(f"{t} n={n:5d} score {x.mean():.3f}  diff {d.mean():+.3f} [{d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(d)):+.3f}, {d.mean()+1.96*d.std(ddof=1)/np.sqrt(len(d)):+.3f}]  {name[t][:90]}")
