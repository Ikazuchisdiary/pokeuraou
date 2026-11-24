import collections, glob, json, re
teams = json.load(open("C:/tmp/ika448/teams.json"))
field = set(teams["field"])
pool = json.load(open("C:/tmp/ika448/pool448.json", encoding="utf-8"))
names = {t["id"]: t["name"] for t in pool["teams"]}
c = collections.Counter(); ctrl = collections.Counter(); who = collections.defaultdict(collections.Counter)
ng = collections.Counter()
for f in glob.glob("C:/tmp/ika448/matches/main/games-worker*.jsonl"):
    for line in open(f, encoding="utf-8"):
        g = json.loads(line)
        if g["seatIndex"] != 0:
            pass
        ids = g["pool"]["teams"]
        focus = ids[0] if ids[0] in field or ids[0] in teams["controls"] else ids[1]
        isfield = focus in field
        ng[isfield] += 1
        notes = set()
        for m in re.finditer(r'"unmodelled": \[([^\]]*)\]', line):
            for n in re.findall(r'"([^"]*)"', m.group(1)):
                notes.add(re.sub(r"\d+", "N", n))
        for n in notes:
            c[n] += 1
            if not isfield: ctrl[n] += 1
            else: who[n][focus] += 1
print("games field", ng[True], "control", ng[False])
print(" all   ctrl  note  (top field teams)")
for k, v in c.most_common(60):
    rate_f = (v - ctrl[k]) / ng[True]; rate_c = ctrl[k] / ng[False]
    if rate_f > 2 * rate_c + 0.002:
        top = ", ".join(f"{names[t][:28]} {n}" for t, n in who[k].most_common(3))
        print(f"{rate_f:6.2%} {rate_c:6.2%}  {k}  | {top}")
