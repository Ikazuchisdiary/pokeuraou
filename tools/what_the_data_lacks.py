"""Does the training data contain games a slow plan won?

The standing suspicion is that it does not: the model never leads Incineroar and Toxapex
into a stall, so self-play never records a stall winning, so the value function never
learns that a half-built stall is worth anything -- and a depth-1 search can only execute a
long plan through a value function that recognises it. The loop closes on itself.

That is a claim about what is *in* the data, and it has not been measured. This measures
it, before anything is built to fix it, because "there is nothing here" is exactly the kind
of premise that turns out to be wrong after the machine time is spent.

Four things, each chosen because a stall would move it:

  long games        a stall wins late or not at all
  status moves      Toxic, Will-O-Wisp and Yawn are how a stall starts
  status then win   the share of games where a side used status and went on to win, which
                    is the one that matters: using it and losing teaches the opposite
  the slow win      status, *and* the game ran long, *and* that side won. This is the one
                    the claim is about. The plain "used status and won" conflates a Yawn
                    played for tempo in a nine-turn race with a Toxic that decided a
                    twenty-turn one, and on gen-8 it is mostly the former: 741 of the
                    status moves came from Sylveon and 528 from Toxapex
  trap and poison   the plan rather than its parts: Infestation *and* Toxic from the same
                    side in the same game. Toxapex's four moves are one plan with an
                    order -- trap so it cannot switch, poison, outlast -- and a Toxic with
                    no trap is a move the opponent simply walks away from
  comebacks         won from behind on HP at the midpoint, which is what a stall looks
                    like from outside -- behind on damage, ahead on the clock

Compared across generations, because the settings changed: gen-8 was width 48 with a
uniform selection, gen-9 is width 24 drawing from the book.

    uv run python tools/what_the_data_lacks.py data/selfplay-gen8 data/selfplay-gen9
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Moves whose whole value is on a later turn. Protect and Recover are excluded: both are
#: played in fast games too, so counting them would say nothing.
STALL_MOVES = {"toxic", "willowisp", "yawn", "leechseed", "infestation"}

#: The plan itself, rather than its parts. Toxapex on this roster runs Infestation, Toxic,
#: Wide Guard and Baneful Bunker with Leftovers, and those are not four moves -- they are
#: one plan with an order: trap so the target cannot switch, poison, then outlast it. A
#: Toxic without the trap is a move the opponent walks away from, which is why counting
#: poison alone counts something else.
TRAP = "infestation"
POISON = "toxic"


def summarise(directory: Path, limit: int) -> dict:
    games = 0
    turns: list[int] = []
    stall_used = 0
    stall_and_won = 0
    slow_win = 0
    poison_used = 0
    poison_and_won = 0
    plan_used = 0
    plan_and_won = 0
    plan_and_won_long = 0
    # Side 0 alone, because Toxapex is on side 0's team and nobody else's, so "the side
    # that played the plan" is almost always side 0 -- and side 0 wins about 53% of
    # everything on team strength. Comparing the plan against the field would credit it
    # with the team's advantage.
    ours_games = 0
    ours_won = 0
    ours_planned = 0
    ours_planned_won = 0
    behind_and_won = 0
    scored = 0
    species_leading_stall: Counter[str] = Counter()

    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if games >= limit:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                outcome = record.get("outcome")
                if outcome is None:
                    continue
                games += 1
                turns.append(int(record.get("turns", 0)))

                decisions = [
                    d for d in record.get("decisions", ()) if d.get("kind") == "move"
                ]
                if not decisions:
                    continue

                # Which side used a stall move, from the action each side actually drew.
                used = [False, False]
                # Toxic apart from the rest: its whole value is on a later turn, where
                # Yawn buys a switch now. They are not the same plan.
                poisoned = [False, False]
                trapped = [False, False]
                for decision in decisions:
                    for side, chosen in enumerate(
                        (decision.get("ownChosen"), decision.get("foeChosen"))
                    ):
                        if not chosen:
                            continue
                        position = decision.get("position") or {}
                        for slot in chosen.split(","):
                            slot = slot.strip()
                            if not slot.startswith("move "):
                                continue
                            index = slot.split()[1]
                            if not index.isdigit():
                                continue
                            actives = _active_moves(position, side)
                            move = actives[int(index) - 1] if len(actives) >= int(index) else None
                            if move in STALL_MOVES:
                                used[side] = True
                                species_leading_stall[_active_species(position, side)] += 1
                            if move == POISON:
                                poisoned[side] = True
                            if move == TRAP:
                                trapped[side] = True

                winner = 0 if outcome > 0.5 else 1
                long_game = int(record.get("turns", 0)) >= 15
                if any(used):
                    stall_used += 1
                    if used[winner]:
                        stall_and_won += 1
                        if long_game:
                            # The shape the claim is about: a status move, a long game,
                            # and the side that played it winning. A Yawn in a nine-turn
                            # race is none of these things.
                            slow_win += 1
                if any(poisoned):
                    poison_used += 1
                    if poisoned[winner]:
                        poison_and_won += 1
                ours_games += 1
                ours_won += int(winner == 0)
                planned = [poisoned[i] and trapped[i] for i in (0, 1)]
                if planned[0]:
                    ours_planned += 1
                    ours_planned_won += int(winner == 0)
                if any(planned):
                    plan_used += 1
                    if planned[winner]:
                        plan_and_won += 1
                        if long_game:
                            plan_and_won_long += 1

                # Behind on HP at the midpoint and won anyway.
                middle = decisions[len(decisions) // 2].get("position") or {}
                shares = _hp_shares(middle)
                if shares is not None:
                    scored += 1
                    if shares[winner] < shares[1 - winner]:
                        behind_and_won += 1
        if games >= limit:
            break

    turns_a = np.asarray(turns) if turns else np.zeros(1)
    return {
        "games": games,
        "mean_turns": float(turns_a.mean()),
        "reached_15": float((turns_a >= 15).mean()),
        "reached_20": float((turns_a >= 20).mean()),
        "stall_used": stall_used / max(games, 1),
        "stall_and_won": stall_and_won / max(games, 1),
        "slow_win": slow_win / max(games, 1),
        "slow_win_count": slow_win,
        "poison_used": poison_used / max(games, 1),
        "poison_and_won": poison_and_won / max(games, 1),
        "plan_used": plan_used / max(games, 1),
        "plan_and_won": plan_and_won / max(games, 1),
        "plan_and_won_long": plan_and_won_long / max(games, 1),
        "plan_count": plan_and_won_long,
        "ours_base": ours_won / max(ours_games, 1),
        "ours_planned": ours_planned / max(ours_games, 1),
        "ours_planned_won": ours_planned_won / max(ours_planned, 1),
        "ours_planned_count": ours_planned,
        "behind_and_won": behind_and_won / max(scored, 1),
        "scored": scored,
        "stall_species": species_leading_stall.most_common(5),
    }


def _active_moves(position: dict, side: int) -> list[str]:
    sides = position.get("sides") or []
    if side >= len(sides):
        return []
    for mon in sides[side].get("pokemon") or []:
        if mon.get("activeIndex") is not None:
            return [str(m.get("id") or "") for m in mon.get("moves") or []]
    return []


def _active_species(position: dict, side: int) -> str:
    sides = position.get("sides") or []
    if side >= len(sides):
        return "?"
    for mon in sides[side].get("pokemon") or []:
        if mon.get("activeIndex") is not None:
            return str(mon.get("species") or "?")
    return "?"


def _hp_shares(position: dict) -> tuple[float, float] | None:
    sides = position.get("sides") or []
    if len(sides) < 2:
        return None
    out = []
    for side in sides[:2]:
        total = alive = 0.0
        for mon in side.get("pokemon") or []:
            maximum = float(mon.get("maxhp") or 0)
            if maximum <= 0:
                continue
            total += maximum
            alive += float(mon.get("hp") or 0)
        out.append(alive / total if total else 0.0)
    return (out[0], out[1])


def main() -> None:
    directories = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    limit = 4000
    for argument in sys.argv[1:]:
        if argument.startswith("--games="):
            limit = int(argument.split("=", 1)[1])
    if not directories:
        raise SystemExit(__doc__)

    rows = [(d.name, summarise(d, limit)) for d in directories]
    print(f"\n  {'':<22}" + "".join(f"{name:>20}" for name, _ in rows))
    def line(label: str, key: str, form: str = "{:.1%}") -> None:
        print(f"  {label:<22}"
              + "".join(f"{form.format(row[key]):>20}" for _, row in rows))

    line("games read", "games", "{:,}")
    line("mean turns", "mean_turns", "{:.1f}")
    line("reached turn 15", "reached_15")
    line("reached turn 20", "reached_20")
    line("a side used status", "stall_used")
    line("...and that side won", "stall_and_won")
    line("...and it ran long too", "slow_win")
    line("   (that, in games)", "slow_win_count", "{:,}")
    line("a side used Toxic", "poison_used")
    line("...and that side won", "poison_and_won")
    line("TRAPPED and poisoned", "plan_used")
    line("...and that side won", "plan_and_won")
    line("...and it ran long too", "plan_and_won_long")
    line("   (that, in games)", "plan_count", "{:,}")
    print()
    line("side 0 wins, all games", "ours_base")
    line("side 0 played the plan", "ours_planned")
    line("   (that, in games)", "ours_planned_count", "{:,}")
    line("side 0 wins when it did", "ours_planned_won")
    line("behind at half, won", "behind_and_won")
    print()
    for name, row in rows:
        top = ", ".join(f"{s} {n}" for s, n in row["stall_species"]) or "none"
        print(f"  {name}: status moves came from {top}")


if __name__ == "__main__":
    main()
