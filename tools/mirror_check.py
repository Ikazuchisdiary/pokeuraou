"""The mirror row for every pool, recomputed from the games rather than remembered.

A true mirror is antisymmetric: our six against our six, the same spreads, both sides
played by the same agent. Its win rate must come out at 50%, and nothing has to be trusted
for that to be true -- not the value function, not the ranking, not the search width. It is
the project's only zero-noise instrument, and it checks search, resolver and evaluator
together.

Which is why three generations sat recorded below 0.500 with no interval beside them and no
way to tell a finding from a sample size. Recomputed over every pool that has mirror games:

    gen-3   551   46.28%   [0.422, 0.505]
    gen-4   703   50.36%   [0.467, 0.540]
    gen-5   649   48.38%   [0.446, 0.522]
    gen-6   666   48.95%   [0.452, 0.527]
    gen-7  1202   47.84%   [0.450, 0.507]
    gen-8   696   52.30%   [0.486, 0.560]
    gen-9  1209   49.30%   [0.465, 0.521]
    gen-10 1186   49.41%   [0.466, 0.523]

Every interval contains 0.500. The sub-0.500 rows close as sample size.

Generations 11h and 11L hold no mirror games at all -- they were generated without
`--mirror-share`, which is also why they are not a paired twin of generation 10 (the mirror
draw consumes a random number, so the two runs' streams diverge; measured at 49.1% of games
drawing the same setup). A pool with no mirror row has no calibration check.

    uv run python tools/mirror_check.py
    uv run python tools/mirror_check.py data/selfplay-gen12
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if not n:
        return (float("nan"),) * 3
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def row(directory: str) -> tuple[int, int] | None:
    """(mirror wins, mirror games) in one pool, or None when it has none."""
    wins = played = 0
    for path in glob.glob(os.path.join(directory, "*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("foeArchetype") != "mirror":
                    continue
                if record.get("outcome") is None:
                    continue
                played += 1
                wins += int(record["outcome"] > 0.5)
    return (wins, played) if played else None


def main() -> None:
    wanted = sys.argv[1:] or sorted(
        d for d in glob.glob("data/selfplay-*") if os.path.isdir(d)
    )
    print(f"  {'pool':<28} {'n':>6}  {'win':>7}   95% Wilson        50% inside")
    checked = failed = empty = 0
    for directory in wanted:
        got = row(directory)
        if got is None:
            empty += 1
            print(f"  {os.path.basename(directory):<28} {'-':>6}  "
                  f"{'-':>7}   no mirror games      --")
            continue
        wins, played = got
        rate, low, high = wilson(wins, played)
        inside = low <= 0.5 <= high
        checked += 1
        failed += int(not inside)
        print(f"  {os.path.basename(directory):<28} {played:>6}  {rate:>7.2%}   "
              f"[{low:.3f}, {high:.3f}]     {'yes' if inside else 'NO'}")
    print(f"\n  {checked} pools with a mirror row, {failed} whose interval excludes 50%, "
          f"{empty} with no mirror games.")
    if failed:
        print("  A mirror that is not 50% is a defect in search, resolver or evaluator --\n"
              "  not in the value function, which cancels.")


if __name__ == "__main__":
    main()
