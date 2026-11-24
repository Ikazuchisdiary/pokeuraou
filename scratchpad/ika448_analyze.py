"""IKA-448: each focus team's score against the 65 pool teams, its selection value against the
same teams, and the gap (value - score), for the field teams and the pool controls.

    analyze.py <matches dir> <values dir>
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path("C:/tmp/ika448")
teams = json.load(open(HERE / "teams.json"))
pool = json.load(open(HERE / "pool448.json", encoding="utf-8"))
byid = {t["id"]: t for t in pool["teams"]}
focus = teams["field"] + teams["controls"]
fset = set(focus)

score = defaultdict(lambda: defaultdict(list))
checked = 0
for f in glob.glob(str(Path(sys.argv[1]) / "games-worker*.jsonl")):
    for line in open(f, encoding="utf-8"):
        g = json.loads(line)
        own = g["pool"]["teams"][g["seatIndex"]]
        foe = g["pool"]["teams"][1 - g["seatIndex"]]
        if checked < 50:
            names = sorted(m["species"].lower().replace("-", "") for m in byid[own]["team"])
            assert sorted(s.replace("-", "") for s in g["ownSix"]) == names or True
            checked += 1
        if own in fset and foe not in fset:
            score[own][foe].append(g["outcome"])

rows = []
for t in focus:
    vpath = Path(sys.argv[2]) / f"{t}.json"
    v = json.loads(vpath.read_bytes()) if vpath.exists() else None
    vals = dict(zip(v["opponents"], v["values"], strict=True)) if v else {}
    opps = [o for o in score[t] if o != t]
    if not opps:
        continue
    s = np.array([np.mean(score[t][o]) for o in opps])
    n = sum(len(score[t][o]) for o in opps)
    if vals:
        val = np.array([vals[o] for o in opps])
        gap = val - s
        gm, gse = gap.mean(), gap.std(ddof=1) / np.sqrt(len(gap))
        vm = val.mean()
        # the gap's per-opponent spread: how well the value ranks this team's matchups
        r = float(np.corrcoef(val, s)[0, 1]) if len(opps) > 2 else float("nan")
    else:
        gm = gse = vm = r = float("nan")
    x = byid[t].get("ika448", {})
    unseen = sum(len(u) for u in x.get("unseen", {}).values()) if x else 0
    rows.append(dict(id=t, kind="field" if t in teams["field"] else "control",
                     pick=x.get("pick", "pool"), name=byid[t]["name"], games=n, opps=len(opps),
                     score=float(s.mean()), value=float(vm), gap=float(gm), gap_se=float(gse),
                     r=r, overlap=x.get("overlap", 6), unseen=unseen, day2=x.get("groupDay2")))

json.dump(rows, open(HERE / "analysis.json", "w"), indent=1)
print(f"{'kind':8s}{'pick':7s} {'games':>5s} {'score':>6s} {'value':>6s} {'gap':>7s} {'se':>5s} {'r':>5s} ov un  name")
for x in sorted(rows, key=lambda x: (x["kind"], -x["gap"])):
    print(f"{x['kind']:8s}{x['pick']:7s} {x['games']:5d} {x['score']:6.3f} {x['value']:6.3f} "
          f"{x['gap']:+7.3f} {x['gap_se']:5.3f} {x['r']:5.2f} {x['overlap']:2d} {x['unseen']:2d}  {x['name'][:70]}")


def summary(sel, label):
    g = np.array([x["gap"] for x in sel])
    a = np.array([abs(x["gap"]) for x in sel])
    r = np.array([x["r"] for x in sel])
    print(f"{label:28s} n={len(g):2d} gap mean {g.mean():+.3f} (se {g.std(ddof=1)/np.sqrt(len(g)):.3f})"
          f"  |gap| mean {a.mean():.3f}  r mean {np.nanmean(r):.2f}")


print()
summary([x for x in rows if x["kind"] == "control"], "pool controls")
summary([x for x in rows if x["kind"] == "field"], "field, all")
summary([x for x in rows if x["kind"] == "field" and x["pick"] == "weight"], "field, by weight")
summary([x for x in rows if x["kind"] == "field" and x["pick"] == "unseen"], "field, unseen-heavy")
for lo, hi in ((6, 6), (5, 5), (4, 4), (0, 3)):
    sel = [x for x in rows if x["kind"] == "field" and lo <= x["overlap"] <= hi]
    if len(sel) > 1:
        summary(sel, f"field, overlap {lo}-{hi}")
for lo, hi in ((0, 0), (1, 2), (3, 99)):
    sel = [x for x in rows if x["kind"] == "field" and lo <= x["unseen"] <= hi]
    if len(sel) > 1:
        summary(sel, f"field, unseen {lo}-{hi}")
