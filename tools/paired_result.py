"""A queued match read as the paired design it is.

Index `i` is game `i // 2` in seat `i % 2`, and every game seeds from its own index -- so
game `k` in the two seats is the same opponent, the same spread class and the same battle
stream, with the arms swapped. That is a paired comparison, and every interval this
project has reported on a queued match was computed as though the 2N games were
independent.

The bias is conservative on the data so far (the interval comes out too wide), which is
why nothing caught it, and conservative is not a property of the design -- it depends on
how correlated the two seats are. On `gen10-vs-gen9`, 516 of 848 pairs score exactly 0.5,
meaning the same side won both times and the pair contributes no variance at all; the
paired interval is +-2.10 against the +-2.38 reported.

Pairs are found by `gameIndex`, which `write_game` records from 2026-09-19. A match run
before that has no index and is reported unpaired, with a line saying so rather than a
number that quietly means something else.

    uv run python tools/paired_result.py data/matches/gen11L-vs-gen10-ownbooks-m2
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def elo(p: float) -> float:
    if not 0.0 < p < 1.0:
        return math.copysign(float("inf"), p - 0.5)
    return -400.0 * math.log10(1.0 / p - 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", type=Path, nargs="+")
    args = ap.parse_args()

    for directory in args.dirs:
        # The tested arm's result per (game index, seat). `provenance.seat` names the arm
        # and which side it sat on, which is what decides whether `outcome` is its win.
        by_index: dict[int, dict[int, float]] = defaultdict(dict)
        loose = 0
        total = 0
        for path in sorted(directory.glob("games-worker*.jsonl")) + sorted(
            directory.glob("games-seed*.jsonl")
        ):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    game = json.loads(line)
                    outcome = game.get("outcome")
                    if outcome is None:
                        continue
                    total += 1
                    index = game.get("gameIndex")
                    if index is None:
                        loose += 1
                        continue
                    seat = str((game.get("provenance") or {}).get("seat", ""))
                    tested_at_side0 = seat.endswith("side 0")
                    won = (
                        float(outcome) > 0.5 if tested_at_side0 else float(outcome) < 0.5
                    )
                    by_index[int(index)][0 if tested_at_side0 else 1] = float(won)

        print(f"\n{directory}")
        if loose:
            print(
                f"  {loose} of {total} games have no gameIndex -- run before 2026-09-19."
                "\n  Reported unpaired, which is what every earlier match reported."
            )
        pairs = [v for v in by_index.values() if len(v) == 2]
        if not pairs:
            print("  no complete pairs")
            continue
        diffs = [sum(v.values()) / 2.0 for v in pairs]
        n = len(diffs)
        mean = sum(diffs) / n
        var = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
        half = 1.96 * math.sqrt(var / n)
        singles = sum(1 for v in by_index.values() if len(v) == 1)
        flat = sum(1 for d in diffs if d == 0.5)
        print(
            f"  {n} complete pairs ({2 * n} games)"
            + (f", {singles} games whose partner is missing" if singles else "")
        )
        print(
            f"  {flat} pairs ({flat / n:.0%}) split 1-1 -- the same side won both times,"
            "\n  which is a pair contributing no variance and the reason pairing helps."
        )
        unpaired_half = 1.96 * math.sqrt(max(mean * (1 - mean), 1e-9) / (2 * n))
        print(
            f"  tested arm {mean:.2%} +-{half:.2%} paired"
            f"   (+-{unpaired_half:.2%} if the games were independent)"
        )
        print(
            f"  Elo {elo(mean):+.1f} [{elo(max(mean - half, 1e-6)):+.1f},"
            f" {elo(min(mean + half, 1 - 1e-6)):+.1f}]"
        )


if __name__ == "__main__":
    main()
