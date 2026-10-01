"""IKA-411 step 1: every M-C Mega, Mega Evolved through the port, before/after counts.

    python sweep_megas.py <out.jsonl>

For each (species, stone) -> mega in the M-C dump whose base is team-legal: side 1's six are
that set (from the pool when a pool set has it, else species' first ability + Protect) in
slot 0 and five fixed fillers. Turn 1 through the port with side 1's slot 0 Mega Evolving
(the port's own outcome, branch 0). Prints, before and after the turn:
  - the carried identities and seen slots of side 1
  - shown_species
  - the number of completions (no weights) and whether any holds the Mega's own sheet member
  - with a uniform book over all 360 ordered selections of the six: whether bench_weights
    prices any completion (the weighted belief) or falls back to uniform
"""

import itertools
import json
import sys
from pathlib import Path

from pokeuraou import port
from pokeuraou.actions import side_actions
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions, seen_identities, seen_slots, shown_species
from pokeuraou.pool import load_pool
from pokeuraou.priors import SampledSet
from pokeuraou.regulation import to_id
from pokeuraou.selection_book import bench_weights
from pokeuraou.selfplay import position_from_sets

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
pool = load_pool(str(M / "data/pool/regmc-matchupweb.json"))
reg = pool.reg
register_mega_stones(reg)

pool_sets = {}
for team in pool.teams:
    for s in team.sets:
        pool_sets.setdefault((s.species, s.item), s)


def fabricate(species: str, item: str) -> SampledSet:
    sp = reg.species[species]
    return SampledSet(species=species, ability=to_id(sp.abilities[0]), item=item,
                      nature="Hardy", sp={}, moves=["protect"])


FILLER_POOL = ["incineroar", "rillaboom", "amoonguss", "kingambit", "sneasler", "gholdengo",
               "whimsicott", "basculegion", "archaludon", "farigiraf"]


def filler(exclude_base: str, n: int) -> list[SampledSet]:
    out = []
    for sid in FILLER_POOL:
        if sid not in reg.species:
            continue
        if to_id(reg.species[sid].base_species) == exclude_base:
            continue
        found = next((s for (sp, _i), s in pool_sets.items() if sp == sid and reg.mega_target(sp, s.item) is None), None)
        out.append(found or fabricate(sid, None))
        if len(out) == n:
            break
    return out


own_six = filler("", 6)
rows = []
for (sid, item), target in sorted(reg.mega_by_species.items()):
    if sid not in reg.species or not reg.species[sid].team_legal:
        continue
    lead = pool_sets.get((sid, item)) or fabricate(sid, item)
    six = [lead] + filler(to_id(reg.species[sid].base_species), 5)
    pos = position_from_sets(reg, own_six[:4], six[:4])
    before_ids = seen_identities(pos, 1)
    before_slots = seen_slots(pos, 1, before_ids)
    before = completions(reg, pos, 1, six, seen=before_slots)
    acts0 = [a for a in side_actions(reg, pos, 0) if not a.declares_mega]
    acts1 = [a for a in side_actions(reg, pos, 1) if a.declares_mega]
    if not acts1:
        rows.append({"species": sid, "item": item, "target": target, "error": "no mega action"})
        continue
    t = port.turn(reg, pos, [acts0[0], acts1[0]], Budget.exact(), full=True)
    after_pos = t.outcomes[0].position
    mon = after_pos.sides[1].pokemon[0]
    after_ids = seen_identities(after_pos, 1, before_ids)
    after_slots = seen_slots(after_pos, 1, after_ids)
    after = completions(reg, after_pos, 1, six, seen=after_slots)
    # uniform book over every ordered selection of four from six
    sels = list(itertools.permutations(range(6), 4))
    probs = [1.0 / len(sels)] * len(sels)
    names = [s.species for s in six]
    leads = shown_species(pos, 1, frozenset(m.slot for m in pos.sides[1].pokemon if m.active_index is not None))
    w_before = bench_weights(sels, probs, names, shown_species(pos, 1, before_slots), leads=leads)
    w_after = bench_weights(sels, probs, names, shown_species(after_pos, 1, after_slots), leads=leads)
    priced_after = sum(w_after.get(tuple(sorted(c.species)), 0.0) for c in after)
    row = {
        "species": sid, "item": item, "target": target, "from_pool": (sid, item) in pool_sets,
        "after_species": mon.species, "after_is_mega": mon.is_mega, "base_species": mon.base_species,
        "ids_before": sorted(before_ids), "ids_after": sorted(after_ids),
        "slots_before": sorted(before_slots), "slots_after": sorted(after_slots),
        "shown_after": sorted(shown_species(after_pos, 1, after_slots)),
        "completions_before": len(before), "completions_after": len(after),
        "self_in_bench_after": sum(sid in c.species for c in after),
        "weight_keys_after": sorted({len(k) for k in w_after}),
        "weights_price_after": priced_after,
        "weights_price_before": sum(w_before.get(tuple(sorted(c.species)), 0.0) for c in before),
    }
    rows.append(row)
    flag = "BAD" if row["completions_after"] > row["completions_before"] or row["self_in_bench_after"] else "ok"
    print(f"{flag:3s} {sid:16s} {item:15s} -> {mon.species:16s} mega={mon.is_mega} base={mon.base_species:12s} "
          f"compl {len(before)} -> {len(after)}  self-in-bench {row['self_in_bench_after']}  "
          f"weights priced {row['weights_price_before']:.2f} -> {priced_after:.2f}")

Path(sys.argv[1]).write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in rows))
print("rows", len(rows), "bad", sum(1 for r in rows if r.get("completions_after", 0) > r.get("completions_before", 0)
                                    or r.get("self_in_bench_after")))
