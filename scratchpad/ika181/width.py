"""IKA-181 step 3: how much the legal pools (what the Q ranks, `qhead.legal_pool`) grow.

On the move decisions of recorded gen-4 games (data/selfplay-mc4, the true position), each
side's legal pool under ally-target modes off / benefit / all. Prints the mean pool, the
mean Q matrix (own x foe), and the share of decisions whose pool grew.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from pokeuraou import actions
from pokeuraou.position import Position
from pokeuraou.qhead import legal_pool
from pokeuraou.regulation import load_regulation

reg = load_regulation("gen9championsvgc2026regmc")
files = sorted(Path(sys.argv[1]).glob("*.jsonl"))[: int(sys.argv[2])]
games_per_file = int(sys.argv[3])
modes = ("off", "benefit", "all")
size = {m: 0 for m in modes}
cells = {m: 0 for m in modes}
grew = Counter()
grew_sides = Counter()
n = 0
sides = 0
kinds = Counter()
for path in files:
    with path.open(encoding="utf-8") as fh:
        for g, line in enumerate(fh):
            if g >= games_per_file:
                break
            game = json.loads(line)
            for d in game["decisions"]:
                kinds[d.get("kind")] += 1
                if d.get("kind") != "move":
                    continue
                pos = Position.from_json(d["position"])
                n += 1
                per = {}
                for m in modes:
                    with actions.ally_targets(m):
                        per[m] = (len(legal_pool(reg, pos, 0)), len(legal_pool(reg, pos, 1)))
                    size[m] += per[m][0] + per[m][1]
                    cells[m] += per[m][0] * per[m][1]
                for m in ("benefit", "all"):
                    if per[m] != per["off"]:
                        grew[m] += 1
                    grew_sides[m] += sum(int(per[m][s] != per["off"][s]) for s in (0, 1))
sides = 2 * n
print(json.dumps({
    "files": len(files), "games_per_file": games_per_file, "kinds": dict(kinds),
    "move_decisions": n,
    "mean_pool_per_side": {m: size[m] / sides for m in modes},
    "mean_q_cells": {m: cells[m] / n for m in modes},
    "decisions_grown": {m: grew[m] for m in ("benefit", "all")},
    "sides_grown": {m: grew_sides[m] for m in ("benefit", "all")},
}, indent=1))
