"""Read a two-seat match as pairs, and compare the two solvers' answers directly.

`generation_match.py` plays each seat as its own run, so a seat-0 game and a seat-1 game
with the same index are usually different matchups -- 69% of them in the first match this
was run on. Matching on the teams instead recovers the pairs, and a pair is worth far more
than two games: when both solvers pick the same moves the matchup plays out identically
whichever seat is which, the pair scores exactly 1.0, and it contributes no variance at
all. Treating those games as independent coin flips spends a confidence interval on games
that could not have disagreed.

The win rate is not the sharpest instrument here either. Every decision record carries the
policy and the node's value, and at a position both seats reached, seat 0's `ownPolicy` is
one solver's answer and seat 1's is the other's -- the same question asked twice. That
compares the solvers on thousands of positions rather than on a few hundred game outcomes,
and it separates the two claims worth keeping apart: that the value is the same (which
double oracle proves, so a difference there is a bug) and that the strategy is the same
(which it does not promise, and which is the thing worth measuring).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
from typing import Any

Game = dict[str, Any]


def _key(game: Game) -> tuple[str, str]:
    return (
        json.dumps(game["ownTeam"], sort_keys=True),
        json.dumps(game["foeTeam"], sort_keys=True),
    )


def _chosen(game: Game) -> list[tuple[Any, ...]]:
    return [(t["turn"], t["kind"], t["ownChosen"], t["foeChosen"]) for t in game["decisions"]]


def _elo(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -400.0 * math.log10(1.0 / p - 1.0)


def _interval(scores: list[float]) -> tuple[float, float]:
    """Win rate and its half-width, over pair scores whose null value is 1.0."""
    n = len(scores)
    mean = sum(scores) / n
    if n < 2:
        return mean / 2, 0.0
    var = sum((x - mean) ** 2 for x in scores) / (n - 1)
    return mean / 2, 1.96 * math.sqrt(var / n) / 2


def pair_up(paths: list[str]) -> tuple[list[tuple[bool, float]], list[tuple[float, float]]]:
    """Pairs of the same matchup played from both seats, and the searches they share."""
    pairs: list[tuple[bool, float]] = []
    searches: list[tuple[float, float]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            games = [json.loads(line) for line in fh if line.strip()]
        seats: dict[int, dict[tuple[str, str], list[Game]]] = {
            0: collections.defaultdict(list),
            1: collections.defaultdict(list),
        }
        for game in games:
            # The seat label ends in the side the new solver played.
            seats[int(game["foeArchetype"].strip()[-1])][_key(game)].append(game)
        for key, first in seats[0].items():
            for a, b in zip(first, seats[1].get(key, []), strict=False):
                # `outcome` is always side 0's, so the new solver's score is `outcome` in
                # the seat where it played side 0 and the complement in the other.
                pairs.append((_chosen(a) == _chosen(b), a["outcome"] + (1.0 - b["outcome"])))
                for ta, tb in zip(a["decisions"], b["decisions"], strict=False):
                    if ta["position"] != tb["position"] or ta["ownActions"] != tb["ownActions"]:
                        break  # the games have parted; later turns are different questions
                    searches.append(
                        (
                            0.5
                            * sum(
                                abs(x - y)
                                for x, y in zip(
                                    ta["ownPolicy"], tb["ownPolicy"], strict=False
                                )
                            ),
                            abs(ta["searchValue"] - tb["searchValue"]),
                        )
                    )
    return pairs, searches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", help="a match directory holding games-seed*.jsonl")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.directory, "games-seed*.jsonl")))
    if not paths:
        raise SystemExit(f"no games-seed*.jsonl under {args.directory}")
    pairs, searches = pair_up(paths)
    if not pairs:
        raise SystemExit("no matchup was played from both seats -- nothing to pair")

    print(f"{len(pairs)} matchups played from both seats, over {len(paths)} seeds")
    for label, keep in (
        ("all pairs", lambda same: True),
        ("  played the same game", lambda same: same),
        ("  the games diverged", lambda same: not same),
    ):
        scores = [score for same, score in pairs if keep(same)]
        if not scores:
            continue
        p, half = _interval(scores)
        print(
            f"  {label:<24} {len(scores):>4} pairs   new side {p:>6.2%} "
            f"+-{half:>5.2%}   Elo {_elo(p):+5.1f} "
            f"[{_elo(p - half):+6.1f}, {_elo(p + half):+6.1f}]"
        )

    if not searches:
        return
    distance = sorted(d for d, _ in searches)
    value = sorted(v for _, v in searches)
    n = len(distance)
    print(f"\n{n} searches the two seats both ran on the same position")
    print("  how far apart the two strategies are (total variation):")
    for threshold in (1e-9, 1e-6, 1e-3, 1e-2, 0.1, 0.5):
        count = sum(1 for d in distance if d > threshold)
        print(f"    more than {threshold:<8g} {count:>6}  ({count / n:>5.1%})")
    print(
        f"  how far apart the node's value is: "
        f"p50 {value[n // 2]:.1e}  p99 {value[int(n * 0.99)]:.1e}  max {value[-1]:.1e}"
    )


if __name__ == "__main__":
    main()
