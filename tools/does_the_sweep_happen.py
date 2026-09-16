"""Does the plan the humans actually play appear in the data?

The plan is not "stall until the opponent dies". It is: chip with Incineroar and Toxapex
-- Intimidate, Fake Out, Parting Shot, Toxic -- and then bring Venusaur in and sweep. The
stall is the setup; the sweep is the win. `what_the_data_lacks.py` counted the setup and
never once counted the sweep, which is why it could say "the plan is good" from games that
may have won some other way.

So this counts the shape:

  swept            a Pokemon of ours was active while two or more of theirs fainted
  swept from back  the same, by a Pokemon that was not in our lead pair -- which is what
                   "chip, then bring it in" looks like from the record
  chip then sweep  our lead used a status or pivot move, and then someone else swept

Counted per species, because "who sweeps" is the question. If the data says Charizard
sweeps and Venusaur never does, that is a different agent from the one the article
describes, and the disagreement is about which Pokemon wins the game rather than about
how long it takes.

    uv run python tools/does_the_sweep_happen.py data/selfplay-gen7 data/selfplay-gen9
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

#: Moves whose value is setting someone else up rather than winning themselves.
SETUP_MOVES = {"toxic", "infestation", "partingshot", "fakeout", "willowisp", "yawn"}


def _side_view(position: dict, side: int) -> tuple[str | None, int]:
    """(species active for `side`, how many of `side`'s Pokemon have fainted)."""
    sides = position.get("sides") or []
    if side >= len(sides):
        return None, 0
    active = None
    fainted = 0
    for mon in sides[side].get("pokemon") or []:
        if mon.get("fainted"):
            fainted += 1
        if mon.get("activeIndex") is not None and active is None:
            active = str(mon.get("species") or "")
    return active, fainted


def summarise(directory: Path, limit: int) -> dict:
    games = won = 0
    swept = swept_from_back = chip_then_sweep = 0
    swept_and_won = chip_sweep_and_won = 0
    sweeper: Counter[str] = Counter()
    back_sweeper: Counter[str] = Counter()

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
                won += int(ours_won)

                six = record.get("ownSix") or []
                pick = record.get("ownPick") or []
                lead = {six[i] for i in pick[:2] if i < len(six)}

                decisions = [
                    d for d in record.get("decisions", ()) if d.get("kind") == "move"
                ]
                if not decisions:
                    continue

                # Faints of *theirs* credited to whoever of ours was on the field when the
                # count went up. Credit is coarse on purpose: in doubles two of ours are
                # out, so this reads the first active slot and will miss a partner's kill.
                # It is the same coarseness for every species, which is what a comparison
                # between them needs.
                kills: Counter[str] = Counter()
                previous = 0
                setup_used = False
                for decision in decisions:
                    position = decision.get("position") or {}
                    ours_active, _ = _side_view(position, 0)
                    _theirs, theirs_fainted = _side_view(position, 1)
                    if theirs_fainted > previous and ours_active:
                        kills[ours_active] += theirs_fainted - previous
                    previous = theirs_fainted
                    chosen = decision.get("ownChosen") or ""
                    if ours_active in lead and _used_setup(position, chosen):
                        setup_used = True

                best = kills.most_common(1)
                if best and best[0][1] >= 2:
                    name, _count = best[0]
                    swept += 1
                    sweeper[name] += 1
                    swept_and_won += int(ours_won)
                    if name not in lead:
                        swept_from_back += 1
                        back_sweeper[name] += 1
                        if setup_used:
                            chip_then_sweep += 1
                            chip_sweep_and_won += int(ours_won)
        if games >= limit:
            break

    return {
        "games": games,
        "won": won / max(games, 1),
        "swept": swept / max(games, 1),
        "swept_and_won": swept_and_won / max(swept, 1),
        "swept_from_back": swept_from_back / max(games, 1),
        "chip_then_sweep": chip_then_sweep / max(games, 1),
        "chip_sweep_count": chip_then_sweep,
        "chip_sweep_won": chip_sweep_and_won / max(chip_then_sweep, 1),
        "sweeper": sweeper.most_common(4),
        "back_sweeper": back_sweeper.most_common(4),
    }


def _used_setup(position: dict, chosen: str) -> bool:
    sides = position.get("sides") or []
    if not sides:
        return False
    moves: list[str] = []
    for mon in sides[0].get("pokemon") or []:
        if mon.get("activeIndex") is not None:
            moves = [str(m.get("id") or "") for m in mon.get("moves") or []]
            break
    for slot in chosen.split(","):
        slot = slot.strip()
        if not slot.startswith("move "):
            continue
        index = slot.split()[1]
        if (
            index.isdigit()
            and len(moves) >= int(index)
            and moves[int(index) - 1] in SETUP_MOVES
        ):
            return True
    return False


def main() -> None:
    directories = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    limit = 4000
    for argument in sys.argv[1:]:
        if argument.startswith("--games="):
            limit = int(argument.split("=", 1)[1])
    if not directories:
        raise SystemExit(__doc__)

    rows = [(d.name, summarise(d, limit)) for d in directories]
    print(f"\n  {'':<26}" + "".join(f"{name:>18}" for name, _ in rows))

    def line(label: str, key: str, form: str = "{:.1%}") -> None:
        print(f"  {label:<26}"
              + "".join(f"{form.format(row[key]):>18}" for _, row in rows))

    line("games", "games", "{:,}")
    line("side 0 wins", "won")
    line("someone swept (2+ KOs)", "swept")
    line("  ...and we won", "swept_and_won")
    line("swept from the back", "swept_from_back")
    line("chip, then swept from back", "chip_then_sweep")
    line("   (that, in games)", "chip_sweep_count", "{:,}")
    line("   ...and we won", "chip_sweep_won")
    print()
    for name, row in rows:
        top = ", ".join(f"{s} {n}" for s, n in row["sweeper"]) or "nobody"
        back = ", ".join(f"{s} {n}" for s, n in row["back_sweeper"]) or "nobody"
        print(f"  {name}\n    sweeps by: {top}\n    from the back: {back}")


if __name__ == "__main__":
    main()
