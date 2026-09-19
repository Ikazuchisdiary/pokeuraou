"""Did the setting under test change the game, or only the label on it?

A queued match plays game `k` twice: once with the tested arm at side 0 and once with it
at side 1, same teams, same seed, same battle stream. With identical arms the two games
are the same game, every pair splits 1-1, and the paired interval is exactly zero -- which
is the null control this harness has and nothing was reading.

That control has a twin nobody had either. When a pair splits 1-1 it can mean two very
different things:

  the two arms played the same moves and the matchup decided it; or
  they played differently and it came out the same anyway.

An arm that never changes a move is an arm whose win rate is 50% by construction, and it
will come back 50% with a beautifully tight interval however long it runs. So the rate is
not reportable on its own: it needs the number of games the setting actually moved.

This counts that, from the recorded games alone:

  moved      the two seats chose a different action somewhere
  first      the turn it first happened, so "changes the endgame only" is visible
  identical  the setting made no difference to this game at all

    uv run python tools/pair_divergence.py data/matches/ika12-depth2-vs-depth1-w24
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def moves_of(game: dict) -> list[tuple]:
    """The pair of choices at each decision, in order."""
    return [
        (d.get("turn"), d.get("kind"), d.get("ownChosen"), d.get("foeChosen"))
        for d in game.get("decisions", ())
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", type=Path, nargs="+")
    args = ap.parse_args()

    for directory in args.dirs:
        # The moves and the outcome, never the game. A record is about 130 KB -- the
        # positions are most of it -- so a 16,000-game run held whole is several GB of
        # Python objects for two fields per decision.
        pairs: dict[int, dict[int, tuple[list[tuple], float | None]]] = defaultdict(dict)
        unindexed = 0
        for path in sorted(directory.glob("games-*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    game = json.loads(line)
                    # Top level, beside `outcome`: `write_game` takes them as `extra`
                    # and they are not part of the provenance block.
                    index = game.get("gameIndex")
                    seat = game.get("seatIndex")
                    if index is None or seat is None:
                        unindexed += 1
                        continue
                    pairs[int(index)][int(seat)] = (moves_of(game), game.get("outcome"))

        complete = {k: v for k, v in pairs.items() if len(v) == 2}
        print(f"\n{directory}")
        if unindexed:
            print(f"  {unindexed} games carry no gameIndex and are not pairable")
        if not complete:
            print("  no complete pairs")
            continue

        moved = 0
        first_turns: Counter[int] = Counter()
        same_outcome = 0
        moved_and_same_outcome = 0
        for _index, sides in sorted(complete.items()):
            (a, outcome_a), (b, outcome_b) = sides[0], sides[1]
            where = next(
                (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                None,
            )
            if where is None and len(a) != len(b):
                where = min(len(a), len(b))
            agreed = outcome_a == outcome_b
            same_outcome += int(agreed)
            if where is not None:
                moved += 1
                first_turns[int(a[where][0]) if where < len(a) else -1] += 1
                moved_and_same_outcome += int(agreed)

        total = len(complete)
        print(f"  {total} complete pairs")
        print(
            f"  moved the game: {moved} ({moved / total:.1%})   "
            f"identical play: {total - moved} ({1 - moved / total:.1%})"
        )
        print(
            f"  same side won both seats: {same_outcome} ({same_outcome / total:.1%}), "
            f"of which {moved_and_same_outcome} had different play"
        )
        if first_turns:
            head = ", ".join(
                f"turn {turn}: {count}" for turn, count in sorted(first_turns.items())[:8]
            )
            print(f"  first difference at {head}")
        if not moved:
            print(
                "  ! the setting never changed a move. A 50% win rate here is arithmetic,"
                "\n    not a measurement, and no number of games will make it one."
            )


if __name__ == "__main__":
    main()
