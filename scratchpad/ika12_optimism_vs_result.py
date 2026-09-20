"""Does the arm lose where its own re-estimate was most optimistic?

`ika12_value_shift.py` says the refine raises side 0's own turn-1 equilibrium value by
+0.007 on average while the arm loses 1.7 points on the board. Two separate facts. This
asks whether they are the same fact, by bucketing the games on the shift.

Seat 0 only, where our roster IS the tested arm: in seat 1 side 0 is the other arm and the
shift there is its optimism, not the tested arm's. Seat 0 alone carries the roster's own
advantage, so the LEVELS below are not win rates against a fair opponent -- only the
differences between buckets mean anything, and they are all in the same seat.

    uv run python scratchpad/ika12_optimism_vs_result.py data/matches/ika12-depth2-...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> None:
    for name in sys.argv[1:]:
        directory = Path(name)
        rows: list[tuple[float, float]] = []  # (shift, tested arm won in seat 0)
        seat1_value: dict[int, float] = {}
        seat0: dict[int, tuple[float, float]] = {}  # index -> (value, outcome)
        for path in sorted(directory.glob("games-*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    game = json.loads(line)
                    index, seat = game.get("gameIndex"), game.get("seatIndex")
                    decisions = game.get("decisions") or []
                    outcome = game.get("outcome")
                    if index is None or seat is None or not decisions or outcome is None:
                        continue
                    opening = decisions[0]
                    if opening.get("kind") != "move" or opening.get("turn") != 1:
                        continue
                    value = float(opening["searchValue"])
                    if int(seat) == 0:
                        seat0[int(index)] = (value, float(outcome))
                    else:
                        seat1_value[int(index)] = value

        for index, (value, outcome) in seat0.items():
            if index in seat1_value:
                rows.append((value - seat1_value[index], 1.0 if outcome > 0.5 else 0.0))

        print(f"\n{directory}")
        if not rows:
            print("  no seat-0 games with a partner")
            continue
        rows.sort(key=lambda r: r[0])
        buckets = 5
        size = len(rows) // buckets
        print(f"  {len(rows)} seat-0 games, bucketed by the tested arm's turn-1 optimism")
        print(f"  {'bucket':>8}  {'shift range':>22}  {'n':>5}  {'won (seat 0)':>13}")
        for b in range(buckets):
            chunk = rows[b * size : (b + 1) * size if b < buckets - 1 else len(rows)]
            won = sum(w for _s, w in chunk) / len(chunk)
            half = 1.96 * math.sqrt(max(won * (1 - won), 1e-9) / len(chunk))
            print(
                f"  {b + 1:>8}  {chunk[0][0]:>+10.4f}..{chunk[-1][0]:>+9.4f}  "
                f"{len(chunk):>5}  {won:>8.2%} +-{half:.2%}"
            )
        # One number for the whole thing: the correlation between optimism and losing.
        shifts = [s for s, _w in rows]
        wins = [w for _s, w in rows]
        n = len(rows)
        ms, mw = sum(shifts) / n, sum(wins) / n
        cov = sum((s - ms) * (w - mw) for s, w in rows) / n
        sds = math.sqrt(sum((s - ms) ** 2 for s in shifts) / n)
        sdw = math.sqrt(sum((w - mw) ** 2 for w in wins) / n)
        r = cov / (sds * sdw) if sds and sdw else float("nan")
        print(f"  correlation(optimism, won) = {r:+.4f}   (n={n}, +-{1.96 / math.sqrt(n):.4f})")


if __name__ == "__main__":
    main()
