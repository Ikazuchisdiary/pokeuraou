import json, sys
d = json.load(open(sys.argv[1]))
rs = {l: d[l]["divergences"] for l in d}
print(json.dumps(rs["old"][0], indent=0)[:600])
def key(r): return (r["where"], r["turn"])
old = {key(r): r for r in rs["old"]}; new = {key(r): r for r in rs["new"]}
print(len(old), len(new))
fixed = sorted(set(old) - set(new), key=str); added = sorted(set(new) - set(old), key=str)
print("fixed (old diverges, new matches):", len(fixed), "added (new diverges, old matched):", len(added))
for k in fixed:
    r = old[k]; print("FIXED", k, r["actions"], "|", r["unmodelled"], "|", str(r["differences"])[:300])
for k in added:
    r = new[k]; print("ADDED", k, r["actions"], "|", str(r["differences"])[:300])
chg = [k for k in set(old) & set(new) if json.dumps(old[k]["differences"], sort_keys=True) != json.dumps(new[k]["differences"], sort_keys=True)]
print("same turn, different differences:", len(chg))
for k in chg[:10]:
    print("CHG", k, old[k]["actions"], "\n   old", str(old[k]["differences"])[:300], "\n   new", str(new[k]["differences"])[:300])
