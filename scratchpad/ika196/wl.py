"""Won / split / lost pairs for the tested arm, from the games (paired_result's reading)."""

import json
import sys
from collections import defaultdict
from pathlib import Path

for name in sys.argv[1:]:
    pairs = defaultdict(dict)
    for path in Path("C:/tmp/ika196/matches", name).glob("games-worker*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            g = json.loads(line)
            seat = g["seatIndex"]
            out = g["outcome"]
            tested_won = out > 0.5 if seat == 0 else out < 0.5
            pairs[g["gameIndex"]][seat] = int(tested_won)
    full = [p for p in pairs.values() if len(p) == 2]
    won = sum(1 for p in full if p[0] + p[1] == 2)
    lost = sum(1 for p in full if p[0] + p[1] == 0)
    print(f"{name}: {len(full)} pairs, won {won}, split {len(full) - won - lost}, lost {lost}")
