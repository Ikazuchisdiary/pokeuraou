"""IKA-117 acceptance: does the belief still forget a Pokemon that has been on the field?

Replays each recorded hidden-bench game's decisions in order and carries "seen" exactly
the way `selfplay.play_game` does in the tree named by ROOT -- by identity when that tree
has `hidden.seen_identities` (IKA-117), by party slot when it does not (before it) -- and
updates it only at move and replacement decisions, which are the only positions
`play_game` looks at. At each of those decisions it asks whether some Pokemon that an
independent tracker has already seen sits in a slot the carried set does not cover.

The independent tracker is the one `seen_slot_drift.py` used, widened to the same
criteria `seen_slots` applies: a species is seen once it was active, fainted, damaged,
statused, boosted, carrying a volatile or Mega Evolved at any decision so far. Keyed on
the current `species` string, not on `hidden.identity`, so it does not share the code it
is checking. Two versions of it:

  play   positions play_game itself sees (move and replacement decisions)
  all    every recorded decision, self-switch pauses included

`all` can see a Pokemon that was out only mid-turn -- in at a self-switch pause and gone
before the next decision -- which play_game never looks at. A count there after the fix
is that, not the slot bug.

Also printed, for the records this changes: the both-seen / one-hidden / both-hidden
split of move decisions (IKA-104's 37%), and completions per side (beliefnode's docstring),
counted as C(6 - shown, 4 - shown).

    python scratchpad/ika117_forgotten.py <ROOT> <POOL> [LIMIT]
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

import pokeuraou.hidden as hidden  # noqa: E402
from pokeuraou.position import Position  # noqa: E402

BY_IDENTITY = hasattr(hidden, "seen_identities")
print(f"tree {ROOT} ({hidden.__file__})")
print(f"carrying: {'by identity (seen_identities)' if BY_IDENTITY else 'by party slot'}")


def carry(position, carried):
    """One decision's update, as play_game writes it in that tree."""
    if BY_IDENTITY:
        new = [hidden.seen_identities(position, i, carried[i]) for i in (0, 1)]
        return new, [hidden.seen_slots(position, i, new[i]) for i in (0, 1)]
    shown = [hidden.seen_slots(position, i, carried[i]) for i in (0, 1)]
    return shown, shown


def reveals(mon) -> bool:
    return bool(
        mon.active_index is not None or mon.fainted or mon.hp != mon.maxhp
        or mon.status is not None or any(mon.boosts.values()) or mon.volatiles
        or mon.is_mega
    )


games = 0
decisions = Counter()
forgotten = {"play": Counter(), "all": Counter()}
games_hit = {"play": 0, "all": 0}
split = Counter()
completions = [0, 0]
moves = 0
example = None
for path in sorted(POOL.glob("*.jsonl")):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if games >= LIMIT:
                break
            game = json.loads(line)
            if game.get("information") != "hidden-bench":
                continue
            games += 1
            carried = [frozenset(), frozenset()]
            known = {"play": [set(), set()], "all": [set(), set()]}
            hit = {"play": False, "all": False}
            for d in game["decisions"]:
                pos = Position.from_json(d["position"])
                kind = d["kind"]
                for side in (0, 1):
                    for mon in pos.sides[side].pokemon:
                        if reveals(mon):
                            known["all"][side].add(mon.species)
                            if kind != "selfswitch":
                                known["play"][side].add(mon.species)
                if kind == "selfswitch":
                    continue
                carried, shown = carry(pos, carried)
                decisions[kind] += 1
                for ref in ("play", "all"):
                    lost = [
                        (side, mon.slot, mon.species)
                        for side in (0, 1)
                        for mon in pos.sides[side].pokemon
                        if mon.species in known[ref][side] and mon.slot not in shown[side]
                    ]
                    if lost:
                        forgotten[ref][kind] += 1
                        hit[ref] = True
                        if example is None and ref == "play":
                            example = (path.name, game.get("gameIndex"), d["turn"], kind, lost)
                if kind == "move":
                    moves += 1
                    hidden_count = [
                        sum(1 for mon in pos.sides[s].pokemon if mon.slot not in shown[s])
                        for s in (0, 1)
                    ]
                    split[
                        "both seen" if hidden_count == [0, 0]
                        else "both hidden" if all(hidden_count)
                        else "one side hidden"
                    ] += 1
                    for s in (0, 1):
                        completions[s] += comb(6 - (4 - hidden_count[s]), hidden_count[s])
            for ref in ("play", "all"):
                games_hit[ref] += hit[ref]
    if games >= LIMIT:
        break

print(f"pool {POOL}: {games} hidden-bench games; decisions {dict(decisions)}")
for ref in ("play", "all"):
    parts = "   ".join(
        f"{kind} {forgotten[ref][kind]:,} / {decisions[kind]:,} "
        f"({forgotten[ref][kind] / max(decisions[kind], 1):.1%})"
        for kind in ("move", "replacement")
    )
    print(f"  forgotten ({ref:4s} reference): {parts}   games {games_hit[ref]:,} / {games:,} "
          f"({games_hit[ref] / max(games, 1):.1%})")
print(f"  first example (play reference): {example}")
print(f"  move decisions {moves:,}: " + "   ".join(
    f"{key} {split[key] / max(moves, 1):.1%}" for key in ("both seen", "one side hidden", "both hidden")
))
print(f"  completions per side: side0 {completions[0] / max(moves, 1):.2f}  "
      f"side1 {completions[1] / max(moves, 1):.2f}")
