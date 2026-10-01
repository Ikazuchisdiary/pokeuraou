"""IKA-181: the silent divergences of ally-target turns, by the `normal` move aimed at the ally."""
import json
import re
import sys
from collections import Counter

from pokeuraou.regulation import load_regulation

reg = load_regulation("gen9championsvgc2026regmc")
by_name = {m.name: m for m in reg.moves.values()}
for path in sys.argv[1:]:
    ex = json.load(open(path, encoding="utf-8"))
    silent = [e for e in ex if e.get("allyTurn") and not e["unmodelled"]]
    moves = Counter()
    fields = Counter()
    for e in silent:
        named = set()
        for part in e["actions"].split(" | "):
            m = re.match(r"(.+?) -> ally\d", part)
            if m and m.group(1) in by_name and by_name[m.group(1)].target == "normal":
                named.add(m.group(1))
        for n in named:
            moves[n] += 1
        for k in e["differences"]:
            fields[k.split(".")[-1]] += 1
    print("==", path, "silent ally-turn divergences", len(silent))
    print("  ally moves:", moves.most_common(25))
    print("  fields:", fields.most_common())
