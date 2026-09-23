# 出どころ: IKA-116 の棚卸しセッション（9/23）の一時 scratchpad にあった hidden_share_by_identity.py （sha256 f87e5301210c62c3、CRLF）を、改行を LF にしただけで写した。IKA-117/118 の受け入れの再生に使う。
"""The recorded "how much is hidden" figures, recomputed by identity instead of by slot.

G2's 47.4% / 52.6%, IKA-104's 37% "both sides seen", beliefnode's 2.84 / 3.16
completions -- all count unseen slots by replaying `seen_slots` with the carried slot set,
which `_do_switch` renumbers. This counts both ways on the same move decisions.
"""

import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(sys.argv[1])
POOL = Path(sys.argv[2])
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.hidden import seen_slots  # noqa: E402
from pokeuraou.position import Position  # noqa: E402

SHEET = 6
games = 0
by_slot = Counter()
by_identity = Counter()
completions_slot = [0, 0]
completions_identity = [0, 0]
moves = 0
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
            seen = [set(), set()]
            for d in game["decisions"]:
                pos = Position.from_json(d["position"])
                shown = [seen_slots(pos, i, shown[i]) for i in (0, 1)]
                for side in (0, 1):
                    for mon in pos.sides[side].pokemon:
                        if (mon.active_index is not None or mon.fainted
                                or mon.hp != mon.maxhp or mon.status is not None
                                or mon.is_mega or mon.volatiles or any(mon.boosts.values())):
                            seen[side].add(mon.species)
                if d["kind"] != "move":
                    continue
                moves += 1
                hs = [len([m for m in pos.sides[s].pokemon if m.slot not in shown[s]])
                      for s in (0, 1)]
                hi = [len([m for m in pos.sides[s].pokemon if m.species not in seen[s]])
                      for s in (0, 1)]
                key = lambda h: "both seen" if h == [0, 0] else (
                    "one side hidden" if 0 in h else "both hidden")
                by_slot[key(hs)] += 1
                by_identity[key(hi)] += 1
                for s in (0, 1):
                    # completions for side s's bench: C(candidates, hidden), candidates =
                    # 6 minus the species accounted for on the board
                    completions_slot[s] += comb(SHEET - (4 - hs[s]), hs[s])
                    completions_identity[s] += comb(SHEET - (4 - hi[s]), hi[s])
    if games >= LIMIT:
        break

print(f"{games} hidden-bench games, {moves} move decisions ({POOL.name})")
for name, c in (("carried slots (as play_game)", by_slot), ("by identity", by_identity)):
    print(f"  {name:30s} " + "   ".join(f"{k} {c[k] / moves:.1%}"
          for k in ("both seen", "one side hidden", "both hidden")))
print(f"  mean completions per side (slot):     side0 {completions_slot[0] / moves:.2f}  "
      f"side1 {completions_slot[1] / moves:.2f}")
print(f"  mean completions per side (identity): side0 {completions_identity[0] / moves:.2f}  "
      f"side1 {completions_identity[1] / moves:.2f}")
