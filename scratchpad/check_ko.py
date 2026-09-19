"""Does Hyper Voice + Flare Blitz actually kill that Venusaur through its Sash?

The user read the log and said Yawn is wasted there because Venusaur falls to Flare
Blitz, or to Hyper Voice plus Flare Blitz. The equilibrium put 100% of its mass on Yawn
and the line they named is in the same 24-action menu at 0.000%.

Whether that is an error depends on the damage, so the damage is computed from the
recorded position rather than reasoned about. Every roll, both moves, against the
defender the record says was standing there.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("C:/Users/Ikazuchi/repos/pokeuraou/src")))

import numpy as np

from pokeuraou.view import battler, field_state
from pokeuraou.damage import calculate, effective_damage, register_mega_stones
from pokeuraou.position import Position
from pokeuraou.regulation import load_regulation

reg = load_regulation("gen9championsvgc2026regmb")
register_mega_stones(reg)

path = "C:/Users/Ikazuchi/repos/pokeuraou/data/matches/anchor-gen11Lx2-book9-hidden-vs-hpshare-m2/games-worker21.jsonl"
game = [json.loads(line) for line in open(path, encoding="utf-8")][12]
turn2 = next(
    d for d in game["decisions"] if d["kind"] == "move" and d["turn"] == 2
)
pos = Position.from_json(turn2["position"])

ours = pos.sides[0]
theirs = pos.sides[1]
print("our actives  :", [m.species for m in ours.pokemon if m.slot in ours.active])
print("their actives:", [m.species for m in theirs.pokemon if m.slot in theirs.active])

by_slot = {m.slot: m for m in theirs.pokemon}
venusaur = by_slot[theirs.active[1]]
print(f"\ndefender {venusaur.species}  hp {venusaur.hp}/{venusaur.maxhp}  item {venusaur.item}")

ours_by_slot = {m.slot: m for m in ours.pokemon}
sylveon = ours_by_slot[ours.active[0]]
incineroar = ours_by_slot[ours.active[1]]
print(f"attackers: {sylveon.species} boosts {sylveon.boosts}  /  {incineroar.species}")

defender = battler(reg, venusaur)
field = field_state(pos, reg)


def rolls(attacker_mon, move_id: str, *, spread: bool) -> np.ndarray:
    result = calculate(
        reg,
        battler(reg, attacker_mon),
        defender,
        move_id,
        field,
        defender_side=1,
        spread=spread,
    )
    return np.asarray(effective_damage(result, defender)).reshape(-1)


hv = rolls(sylveon, "hypervoice", spread=True)
fb = rolls(incineroar, "flareblitz", spread=False)
hp = venusaur.maxhp
print(f"\nhyper voice  {hv.min():.0f}-{hv.max():.0f}   ({hv.min() / hp:.0%}-{hv.max() / hp:.0%} of {hp})")
print(f"flare blitz  {fb.min():.0f}-{fb.max():.0f}   ({fb.min() / hp:.0%}-{fb.max() / hp:.0%} of {hp})")
print(f"\nflare blitz alone KOs from full: {(fb >= hp).mean():.0%} of rolls"
      "  -- but a Focus Sash survives any single hit from full")
both = hv[:, None] + fb[None, :]
print(f"hyper voice THEN flare blitz, both landing: {(both >= hp).mean():.0%} of the 256 roll pairs reach {hp}")
survive = (hv < hp) & True
print(f"hyper voice alone is non-lethal (so the Sash does not fire on it): "
      f"{(hv < hp).mean():.0%} of rolls")
