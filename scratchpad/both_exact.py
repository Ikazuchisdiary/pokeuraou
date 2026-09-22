"""How many hidden-bench move decisions have nothing hidden on either side?

Asked on 2026-09-23 while reading `beliefnode.belief_payoffs` for IKA-97. When no side has
an unseen slot it returns `_per_completion`, and `_per_completion` calls `batched_payoff`
once per side -- with the same Position object, the same menus and the same leaf both
times, because each side's only completion is the position itself. So the share of such
decisions is the share of decisions whose node is resolved, encoded and scored twice.

Counts only, no timing: replays `seen_slots` over each game's recorded decisions in order,
the way `selfplay.play_game` carries `shown` from one decision to the next, and asks at
every move decision whether each side still has an unseen slot.

    uv run python scratchpad/both_exact.py data/ika73/w12/games-worker0.jsonl 500

Answer that day (IKA-73's production pool, width 12, the shipping generation settings):

    500 games, 4862 move decisions
      both exact          1801   37.0%
      one side hidden     1036   21.3%
      both hidden         2025   41.6%
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

from pokeuraou.hidden import seen_slots  # noqa: E402
from pokeuraou.position import Position  # noqa: E402

path = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 500

kinds = Counter()
hidden_counts = Counter()
games = 0
with path.open(encoding="utf-8") as handle:
    for line in handle:
        if games >= limit:
            break
        game = json.loads(line)
        # The field is one string for both sides in these pools.
        if game.get("information") not in ("hidden-bench", ["hidden-bench", "hidden-bench"]):
            kinds["not hidden-bench"] += 1
            continue
        games += 1
        shown = [frozenset(), frozenset()]
        for decision in game["decisions"]:
            pos = Position.from_json(decision["position"])
            shown = [seen_slots(pos, i, shown[i]) for i in (0, 1)]
            if decision.get("kind") != "move":
                kinds[f"kind {decision.get('kind')}"] += 1
                continue
            hidden = [
                sum(1 for mon in pos.sides[i].pokemon if mon.slot not in shown[i])
                for i in (0, 1)
            ]
            state = (
                "both exact" if hidden == [0, 0]
                else "both hidden" if hidden[0] and hidden[1]
                else "one side hidden"
            )
            kinds[state] += 1
            hidden_counts[tuple(hidden)] += 1

moves = sum(v for k, v in kinds.items() if k in ("both exact", "both hidden", "one side hidden"))
print(f"{games} games, {moves} move decisions")
for key in ("both exact", "one side hidden", "both hidden"):
    print(f"  {key:<16} {kinds[key]:>7}  {100.0 * kinds[key] / max(moves, 1):5.1f}%")
for key, value in sorted(kinds.items()):
    if key not in ("both exact", "one side hidden", "both hidden"):
        print(f"  ({key}: {value})")
print("hidden slots (side0, side1):", dict(sorted(hidden_counts.items())))
