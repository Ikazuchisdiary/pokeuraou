"""When the chip is done, is the switch to the sweeper even on the menu?

The plan the human repertoire plays is chip with Incineroar and Toxapex, then bring
Venusaur in and sweep. The data says the model does the chipping and then wins with the
chippers: in games where that pair led, the sweeps were Toxapex 73 and Incineroar 35
against Venusaur almost never. So the handoff is what is missing, and there are three
different reasons it could be missing, which want different fixes:

  not in the menu   `narrow` cut the switch before the search ever saw it. Then the fix is
                    the ordering or the width, and no amount of teacher data helps.
  in the menu, no weight
                    the equilibrium looked at it and put zero on it. Then the leaf thinks
                    the position after the switch is bad, and teacher data is the fix.
  weight, not drawn the equilibrium wanted it and the draw went elsewhere. Then nothing is
                    wrong and the plan is simply a minority line.

Told apart by reading the recorded decisions, which carry the menu and the mixture as well
as the action taken. No games are played.

    uv run python tools/why_no_handoff.py data/selfplay-gen9 --sweeper venusaur
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: The chippers, as the roster spells them.
CHIPPERS = {"toxapex", "incineroar"}


def _active(position: dict, side: int) -> list[str]:
    sides = position.get("sides") or []
    if side >= len(sides):
        return []
    return [
        str(mon.get("species") or "")
        for mon in sides[side].get("pokemon") or []
        if mon.get("activeIndex") is not None
    ]


def _bench_slot(position: dict, side: int, species: str) -> int | None:
    """The 1-based party slot of a healthy benched `species`, as a switch names it."""
    sides = position.get("sides") or []
    if side >= len(sides):
        return None
    for index, mon in enumerate(sides[side].get("pokemon") or [], start=1):
        if (
            str(mon.get("species") or "") == species
            and mon.get("activeIndex") is None
            and not mon.get("fainted")
        ):
            return index
    return None


def main() -> None:
    directories = [Path(a) for a in sys.argv[1:] if not a.startswith("--")]
    sweeper = "venusaur"
    limit = 3000
    for argument in sys.argv[1:]:
        if argument.startswith("--sweeper="):
            sweeper = argument.split("=", 1)[1]
        if argument.startswith("--games="):
            limit = int(argument.split("=", 1)[1])
    if not directories:
        raise SystemExit(__doc__)

    for directory in directories:
        games = 0
        chances = 0          # a chipper is out and the sweeper is healthy on the bench
        on_menu = 0          # the switch survived `narrow`
        had_weight = 0       # the equilibrium put something on it
        was_drawn = 0        # and the draw took it
        weights: list[float] = []
        # And the question the three-way split leads to: when it *is* taken, does it win?
        # Zero weight in 60% of chances says the leaf dislikes the position after the
        # switch. Whether the leaf is wrong about that is a different claim, and the games
        # where the draw took it anyway are the evidence for it.
        took_it = 0
        took_and_won = 0
        chance_games = 0
        chance_won = 0
        cut_when: Counter[int] = Counter()

        for path in sorted(directory.glob("*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if games >= limit:
                        break
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("outcome") is None:
                        continue
                    games += 1
                    ours_won = record["outcome"] > 0.5
                    had_chance = False
                    handed_off = False
                    for decision in record.get("decisions", ()):
                        if decision.get("kind") != "move":
                            continue
                        position = decision.get("position") or {}
                        actives = _active(position, 0)
                        if not (CHIPPERS & set(actives)):
                            continue
                        slot = _bench_slot(position, 0, sweeper)
                        if slot is None:
                            continue
                        chances += 1
                        had_chance = True

                        names = decision.get("ownActions") or []
                        policy = np.asarray(
                            decision.get("ownPolicy") or [], dtype=np.float64
                        )
                        if len(names) != len(policy) or not len(names):
                            continue
                        total = policy.sum()
                        if total <= 0:
                            continue
                        policy = policy / total

                        want = f"switch {slot}"
                        found = [
                            i for i, n in enumerate(names)
                            if any(part.strip() == want for part in n.split(","))
                        ]
                        if not found:
                            cut_when[int(decision.get("turn", 0))] += 1
                            continue
                        on_menu += 1
                        mass = float(policy[found].sum())
                        weights.append(mass)
                        if mass > 1e-9:
                            had_weight += 1
                        chosen = decision.get("ownChosen") or ""
                        if any(part.strip() == want for part in chosen.split(",")):
                            was_drawn += 1
                            handed_off = True
                    if had_chance:
                        chance_games += 1
                        chance_won += int(ours_won)
                        if handed_off:
                            took_it += 1
                            took_and_won += int(ours_won)
            if games >= limit:
                break

        print(f"\n  {directory.name}: {games:,} games, {chances:,} decisions with a "
              f"chipper out and {sweeper} healthy on the bench")
        if not chances:
            continue
        print(f"    switch was on the menu      {on_menu:>7,}  {on_menu / chances:>6.1%}")
        print(f"    ...equilibrium gave it mass {had_weight:>7,}  "
              f"{had_weight / chances:>6.1%}")
        print(f"    ...and the draw took it     {was_drawn:>7,}  "
              f"{was_drawn / chances:>6.1%}")
        if weights:
            array = np.asarray(weights)
            print(f"    mean weight when on the menu  {array.mean():.4f}, "
                  f"median {np.median(array):.4f}, max {array.max():.4f}")
        if chance_games:
            base = chance_won / chance_games
            with_it = took_and_won / max(took_it, 1)
            print(f"    games with a chance          {chance_games:>7,}  "
                  f"won {base:>6.1%}")
            print(f"    ...where it handed off       {took_it:>7,}  "
                  f"won {with_it:>6.1%}  ({100 * (with_it - base):+.1f} points)")
        if cut_when:
            common = ", ".join(f"turn {t}: {n}" for t, n in cut_when.most_common(4))
            print(f"    cut by narrow, by turn: {common}")


if __name__ == "__main__":
    main()
