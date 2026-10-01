"""IKA-181 step 2: in the M-C pool, which (user, ally, move) gain from naming the ally.

Per team: every ordered pair of distinct members (any two of six can stand together) and
each `normal` move of the user, `actions.ally_benefits` on the sets' abilities and items.
Also what each rule fires on, and the teams and pairs of the pool it touches.
"""
import json
import sys
from collections import Counter

from pokeuraou.actions import (
    ALLY_ABSORBING, ALLY_HIT_ITEMS, ALLY_HIT_TYPES, ALLY_STATUS_ABILITIES, ALLY_USE_MOVES,
    ITEM_TAKING_MOVES, team_ally_benefits,
)
from pokeuraou.pool import load_pool

pool = load_pool(sys.argv[1])
reg = pool.reg


def reason(user, ally, move_id):
    move = reg.moves[move_id]
    if move_id in ALLY_USE_MOVES:
        return f"move:{move_id}"
    if ally.ability in ALLY_ABSORBING or ally.ability in ALLY_HIT_TYPES:
        return f"ability:{ally.ability}"
    if ally.ability in ALLY_STATUS_ABILITIES or ally.ability in ("contrary", "owntempo", "weakarmor"):
        return f"ability:{ally.ability}"
    if ally.ability == "unburden" and move_id in ITEM_TAKING_MOVES:
        return "ability:unburden"
    if ally.item in ALLY_HIT_ITEMS or ally.item == "weaknesspolicy":
        return f"item:{ally.item}"
    return f"other:{move.type}"


by_team = {}
reasons = Counter()
combos = Counter()
normal_slots = 0
for t, team in enumerate(pool.teams):
    found = team_ally_benefits(reg, team.sets)
    by_team[t] = found
    for u, a, m in found:
        user, ally = team.sets[u], team.sets[a]
        r = reason(user, ally, m)
        reasons[r] += 1
        combos[(team.id, user.species, ally.species, m, r)] += 1
    for s in team.sets:
        normal_slots += sum(1 for m in s.moves if reg.moves[m].target == "normal")

teams_with = [t for t, f in by_team.items() if f]
pairs_with = [(a, b) for a, b in pool.pairs if a in teams_with or b in teams_with]
out = {
    "teams": len(pool.teams),
    "pairs": len(pool.pairs),
    "teams_with_benefit": len(teams_with),
    "pairs_with_benefit": len(pairs_with),
    "triples": sum(len(f) for f in by_team.values()),
    "normal_move_slots": normal_slots,
    "reasons": dict(reasons.most_common()),
    "combos": [list(k) for k in sorted(combos)],
}
print(json.dumps(out, indent=1, ensure_ascii=False))
