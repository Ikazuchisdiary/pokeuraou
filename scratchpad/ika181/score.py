"""IKA-181: a match directory's tested-arm score by pairs (the two seats of one game).

Per game index the tested arm's mean score over its two seats; the mean of those, its 95%
interval (normal, by the pairs' spread), and Elo. Pair counts won / split / lost.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

seats = defaultdict(dict)
for f in Path(sys.argv[1]).glob("games-*.jsonl"):
    for line in f.read_bytes().splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        which = g["seatIndex"]
        out = g["outcome"]
        if out is None:
            continue
        seats[g["gameIndex"]][which] = out if which == 0 else 1.0 - out
pairs = [sum(s.values()) / 2 for s in seats.values() if len(s) == 2]
n = len(pairs)
mean = sum(pairs) / n
sd = math.sqrt(sum((p - mean) ** 2 for p in pairs) / (n - 1))
half = 1.96 * sd / math.sqrt(n)


def elo(s: float) -> float:
    return -400 * math.log10(1 / s - 1)


won = sum(p > 0.5 for p in pairs)
lost = sum(p < 0.5 for p in pairs)
print(json.dumps({"pairs": n, "won": won, "even": n - won - lost, "lost": lost,
                  "score": round(mean, 4), "half95": round(half, 4),
                  "elo": round(elo(mean), 1),
                  "elo95": [round(elo(mean - half), 1), round(elo(mean + half), 1)]}))
