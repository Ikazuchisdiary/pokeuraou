"""What the refine does to the number the search believes, on the same position.

A queued match plays game k twice with the arms swapped, and side 0 is our roster in both
seats. `searchValue` is recorded for side 0 only, so at turn 1 -- where the position is
identical in the two seats, same teams, same selection, same seed -- the pair is:

    seat 0   our roster's own equilibrium value, solved at depth 2
    seat 1   our roster's own equilibrium value, solved at depth 1

and the difference is the refine, on one position, with nothing else moved. Later turns
are not comparable: by then the two games have diverged and the positions differ.

    uv run python scratchpad/ika12_value_shift.py data/matches/ika12-depth2-vs-depth1-w24
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    for name in sys.argv[1:]:
        directory = Path(name)
        first: dict[int, dict[int, float]] = defaultdict(dict)
        for path in sorted(directory.glob("games-*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    game = json.loads(line)
                    index, seat = game.get("gameIndex"), game.get("seatIndex")
                    decisions = game.get("decisions") or []
                    if index is None or seat is None or not decisions:
                        continue
                    opening = decisions[0]
                    if opening.get("kind") != "move" or opening.get("turn") != 1:
                        continue
                    first[int(index)][int(seat)] = float(opening["searchValue"])

        pairs = [(v[0], v[1]) for v in first.values() if len(v) == 2]
        print(f"\n{directory}")
        if not pairs:
            print("  no turn-1 pairs")
            continue
        shifts = [deep - shallow for deep, shallow in pairs]
        n = len(shifts)
        mean = sum(shifts) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in shifts) / max(n - 1, 1))
        half = 1.96 * sd / math.sqrt(n)
        moved = sum(1 for x in shifts if abs(x) > 1e-9)
        down = sum(1 for x in shifts if x < -1e-9)
        biggest = max(shifts, key=abs)
        print(f"  {n} turn-1 positions, both depths")
        print(
            f"  refine moves side 0's own equilibrium value by {mean:+.5f}"
            f" +-{half:.5f} (sd {sd:.5f})"
        )
        print(
            f"  moved at all: {moved} ({moved / n:.1%}),"
            f" downward: {down} ({down / n:.1%}), largest {biggest:+.4f}"
        )
        print(
            "  → depth 1 was optimistic about its own position"
            if mean < -1e-6
            else "  → depth 1 was pessimistic about its own position"
            if mean > 1e-6
            else "  → no systematic shift"
        )


if __name__ == "__main__":
    main()
