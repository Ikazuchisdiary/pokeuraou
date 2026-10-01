"""IKA-181: what the M-C pool's sets carry -- normal-target moves, abilities, items."""
import sys
from collections import Counter

from pokeuraou.pool import load_pool

pool = load_pool(sys.argv[1])
reg = pool.reg
moves, abilities, items = Counter(), Counter(), Counter()
targets = Counter()
for team in pool.teams:
    for s in team.sets:
        abilities[s.ability] += 1
        items[s.item] += 1
        for m in s.moves:
            mv = reg.moves[m]
            targets[mv.target] += 1
            if mv.target == "normal":
                moves[(m, mv.type, mv.category)] += 1
print("teams", len(pool.teams), "pairs", len(pool.pairs))
print("targets", dict(targets))
print("normal moves:")
for (m, t, c), n in sorted(moves.items(), key=lambda kv: -kv[1]):
    print(f"  {m:18s} {t:9s} {c:8s} {n}")
print("abilities:", dict(sorted(abilities.items(), key=lambda kv: -kv[1])))
print("items:", dict(sorted(items.items(), key=lambda kv: -kv[1])))
