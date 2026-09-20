"""Is the deeper search's own number better, even though its play is worse?

Each seat records `searchValue`: side 0's equilibrium value at turn 1, as the agent
holding side 0 in THAT seat computed it. Seat 0's is the depth-2 arm's, seat 1's is the
depth-1 arm's, and each seat has its own outcome -- the game that agent actually played.

So the two seats give two (forecast, outcome) samples over the same distribution of
turn-1 positions, one per depth. Brier score is the answer to "was the number better":
mean squared error of a probability forecast, lower is better.

This separates two claims that the win rate cannot:

  the refined numbers are worse  -> Brier goes up, and the losses are explained
  the refined numbers are better -> Brier goes down, and the losses are NOT the numbers
                                    but the strategy read off a matrix of mixed depths,
                                    which is what search.py's own "What it does not
                                    claim" warns about

    uv run python scratchpad/ika12_calibration.py data/matches/ika12-depth2-vs-depth1-w24
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> None:
    for name in sys.argv[1:]:
        directory = Path(name)
        seats: dict[int, list[tuple[float, float]]] = {0: [], 1: []}
        by_index: dict[int, dict[int, tuple[float, float]]] = {}
        for path in sorted(directory.glob("games-*.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    game = json.loads(line)
                    seat, outcome = game.get("seatIndex"), game.get("outcome")
                    decisions = game.get("decisions") or []
                    if seat is None or outcome is None or not decisions:
                        continue
                    opening = decisions[0]
                    if opening.get("kind") != "move" or opening.get("turn") != 1:
                        continue
                    pair = (float(opening["searchValue"]), float(outcome))
                    seats[int(seat)].append(pair)
                    index = game.get("gameIndex")
                    if index is not None:
                        by_index.setdefault(int(index), {})[int(seat)] = pair

        print(f"\n{directory}")
        print(f"  {'seat':>6}  {'n':>6}  {'mean forecast':>14}  {'outcome':>8}  "
              f"{'Brier':>8}  {'+-':>7}")
        for seat, rows in seats.items():
            if not rows:
                continue
            n = len(rows)
            forecast = sum(v for v, _o in rows) / n
            realised = sum(o for _v, o in rows) / n
            squares = [(v - o) ** 2 for v, o in rows]
            brier = sum(squares) / n
            sd = math.sqrt(sum((x - brier) ** 2 for x in squares) / max(n - 1, 1))
            print(
                f"  {seat:>6}  {n:>6}  {forecast:>14.4f}  {realised:>8.4f}  "
                f"{brier:>8.4f}  {1.96 * sd / math.sqrt(n):>7.4f}"
            )
        both = seats[0] and seats[1]
        if both:
            b0 = sum((v - o) ** 2 for v, o in seats[0]) / len(seats[0])
            b1 = sum((v - o) ** 2 for v, o in seats[1]) / len(seats[1])
            # Paired on the position, which is the same in the two seats.
            diffs = [
                (v[0][0] - v[0][1]) ** 2 - (v[1][0] - v[1][1]) ** 2
                for v in by_index.values()
                if len(v) == 2
            ]
            n = len(diffs)
            mean = sum(diffs) / n if n else float("nan")
            var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
            half = 1.96 * math.sqrt(var / n) if n else float("nan")
            print(
                f"  seat 0 minus seat 1: {b0 - b1:+.5f} unpaired, "
                f"{mean:+.5f} +-{half:.5f} paired on the position ({n} pairs)"
                f"   ({'the deeper number is better' if mean < 0 else 'the deeper number is worse'})"
            )
            print(
                "  ! the two seats are different GAMES -- each agent's forecast is scored\n"
                "    against the game IT played, and its opponent is the other arm. An arm\n"
                "    facing a weaker opponent realises more than its forecast for that\n"
                "    reason alone, so part of the gap below is the win rate, not the\n"
                "    calibration. Same positions, swapped roles."
            )


if __name__ == "__main__":
    main()
