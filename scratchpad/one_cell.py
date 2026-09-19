"""Resolve the one cell the equilibrium refused, and see whether Venusaur dies in it.

The equilibrium gave `move 1, move 1 2` -- Hyper Voice then Flare Blitz into a Venusaur
that only a Focus Sash is keeping alive -- zero mass. In a zero-sum matrix a zero-mass
action is one whose value against the opponent's mixture is at most the game value, so
the solver is reporting the matrix rather than losing the move. The cell is what is
wrong, and a cell is wrong for one of two reasons:

  the resolver     the turn does not play out the way the damage calculator says
  the leaf         the turn plays out right and the value function prefers the position
                   where the opponent still has a Pokemon

One resolve tells them apart, and one resolve is cheap even with a generation running.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("C:/Users/Ikazuchi/repos/pokeuraou/src")))

from pokeuraou.actions import side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.position import Position
from pokeuraou.regulation import load_regulation
from pokeuraou.resolve import Budget, resolve_turn

reg = load_regulation("gen9championsvgc2026regmb")
register_mega_stones(reg)

path = "data/matches/anchor-gen11Lx2-book9-hidden-vs-hpshare-m2/games-worker21.jsonl"
game = [json.loads(line) for line in open(path, encoding="utf-8")][12]
turn2 = next(d for d in game["decisions"] if d["kind"] == "move" and d["turn"] == 2)
pos = Position.from_json(turn2["position"])

foe_mix = np.asarray(turn2["foePolicy"], dtype=np.float64)
their_best = turn2["foeActions"][int(np.argmax(foe_mix))]
print(f"their most likely reply ({foe_mix.max():.1%}): {their_best}")

legal = {
    side: {a.to_choice(): a for a in side_actions(reg, pos, side)} for side in (0, 1)
}
if their_best not in legal[1]:
    raise SystemExit(f"{their_best!r} is not legal here; the record and the rules differ")

for label, mine in (
    ("the KO the equilibrium refused", "move 1, move 1 2"),
    ("what it played instead       ", "move 3 2, move 1 2"),
):
    if mine not in legal[0]:
        print(f"\n--- {label}: {mine} IS NOT LEGAL HERE")
        continue
    result = resolve_turn(
        reg, pos, [legal[0][mine], legal[1][their_best]], budget=Budget()
    )
    alive: Counter[tuple[str, ...]] = Counter()
    hp: Counter[int] = Counter()
    total = 0.0
    for branch in list(result.branches) + [s.position for s in ()]:
        after = branch.position
        foe = after.sides[1]
        living = tuple(sorted(m.species for m in foe.pokemon if not m.fainted))
        alive[living] += branch.probability
        venu = next((m for m in foe.pokemon if m.species == "venusaur"), None)
        hp[venu.hp if venu else -1] += branch.probability
        total += branch.probability
    print(f"\n--- {label}: {mine}")
    print(f"    {len(result.branches)} branches, exact={result.exact}"
          + (f", {len(result.suspended)} suspended" if result.suspended else ""))
    for who, weight in alive.most_common(3):
        print(f"    {weight / total:6.1%}  their side still standing: {', '.join(who)}")
    dead = sum(w for h, w in hp.items() if h == 0 or h == -1) / total
    print(f"    venusaur dead in {dead:.1%} of the mass")
