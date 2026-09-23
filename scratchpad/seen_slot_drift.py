# 出どころ: IKA-116 の棚卸しセッション（9/23）の一時 scratchpad にあった seen_slot_drift.py （sha256 138b649e8ce57989、CRLF）を、改行を LF にしただけで写した。IKA-117/118 の受け入れの再生に使う。
"""Is `shown` (party slot numbers carried across turns) the set of Pokemon actually seen?

`play_game` carries `shown` as slot indices from one decision to the next, and
`seen_slots` adds `mon.slot`. But `_do_switch` renumbers `slot` on every switch -- the
incoming Pokemon takes the active index and the outgoing one takes the index it vacated
-- so a slot number names a *position*, not a Pokemon. A lead that switched out at full
HP with no status sits at a back index the carried set has never marked, and the next
completion treats it as unseen.

Replays the recorded decision positions exactly as `play_game` does, and in parallel
tracks seen Pokemon by species (unique by Species Clause). Counts decisions where some
species that was on the field earlier is benched now in a slot the carried set does not
cover -- i.e. where the completions would offer worlds without a Pokemon already shown.
"""

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
POOL = Path(sys.argv[2])
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.hidden import seen_slots  # noqa: E402
from pokeuraou.position import Position  # noqa: E402

games = decisions = drifted = drifted_games = 0
by_kind: dict[str, int] = {}
for path in sorted(POOL.glob("*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if games >= LIMIT:
                break
            game = json.loads(line)
            if game.get("information") != "hidden-bench":
                continue
            games += 1
            shown = [frozenset(), frozenset()]
            seen_species = [set(), set()]
            hit = False
            for d in game["decisions"]:
                pos = Position.from_json(d["position"])
                shown = [seen_slots(pos, i, shown[i]) for i in (0, 1)]
                decisions += 1
                for side in (0, 1):
                    mons = pos.sides[side].pokemon
                    for mon in mons:
                        if mon.active_index is not None or mon.fainted:
                            seen_species[side].add(mon.species)
                    covered = {mons[s].species for s in shown[side] if s < len(mons)}
                    missing = [
                        m.species for m in mons
                        if m.species in seen_species[side] and m.species not in covered
                    ]
                    if missing:
                        drifted += 1
                        by_kind[d["kind"]] = by_kind.get(d["kind"], 0) + 1
                        hit = True
                        break
            drifted_games += hit
    if games >= LIMIT:
        break

print(f"pool {POOL}: {games} hidden-bench games, {decisions} decisions")
print(f"decisions where a Pokemon already on the field is treated as unseen: {drifted} "
      f"({drifted / max(decisions, 1):.2%}) by kind {by_kind}")
print(f"games with at least one: {drifted_games} ({drifted_games / max(games, 1):.1%})")
