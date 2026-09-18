"""One match, read back: the win rate per seat, the pair, and what it was played under.

Every match in `data/matches/` is read the same way and the arithmetic is three lines, so
it has been written inline each time it was wanted -- which is how a seat label gets read
as the wrong arm, and how a number ends up quoted without the condition it was measured
in. Both have happened here.

The seat split is printed and not just the total, because it is the check that costs
nothing: the two seats of a fair match differ by the seat advantage and no more, and a gap
much larger than that is a defect in the harness rather than a fact about the agents.

    uv run python tools/match_result.py data/matches/gen11L-vs-gen10-ownbooks
    uv run python tools/match_result.py data/matches/*-vs-hpshare
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def elo(p: float) -> float:
    if not 0.0 < p < 1.0:
        return math.copysign(float("inf"), p - 0.5)
    return -400.0 * math.log10(1.0 / p - 1.0)


def read(directory: Path) -> tuple[dict[str, tuple[int, int]], dict, int]:
    """Per-seat (wins, played) for the named arm, the provenance, and unfinished games."""
    seats: dict[str, tuple[int, int]] = {}
    unfinished = 0
    for path in sorted(directory.glob("worker*.jsonl")) + sorted(
        directory.glob("seed*.jsonl")
    ):
        if path.name.startswith("games-"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "gen2_wins" not in row and "wins" not in row:
                continue
            seat = str(row.get("seat", "?"))
            wins = int(row.get("gen2_wins", row.get("wins", 0)))
            played = int(row.get("played", 0))
            a, b = seats.get(seat, (0, 0))
            seats[seat] = (a + wins, b + played)
            unfinished += int(row.get("unfinished", 0))
    provenance: dict = {}
    for path in sorted(directory.glob("games-worker*.jsonl")) + sorted(
        directory.glob("games-seed*.jsonl")
    ):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    provenance = json.loads(line).get("provenance", {})
                    break
        if provenance:
            break
    return seats, provenance, unfinished


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", type=Path, nargs="+")
    args = ap.parse_args()

    for directory in args.dirs:
        seats, provenance, unfinished = read(directory)
        total_w = sum(w for w, _n in seats.values())
        total_n = sum(n for _w, n in seats.values())
        print(f"\n{directory}")
        if not total_n:
            print("  no completed seat rows")
            continue
        if provenance:
            leaves = provenance.get("leaves") or ["?", "?"]
            books = provenance.get("books") or ["?", "?"]
            info = (provenance.get("information") or ["?", "?"])[0]
            ranks = provenance.get("rankings") or ["?", "?"]
            limits = provenance.get("limits") or ["?", "?"]
            print(f"  leaf   {leaves[0]}  vs  {leaves[1]}")
            print(f"  book   {books[0]}  vs  {books[1]}")
            print(f"  width  {limits[0]}/{limits[1]}   order {ranks[0]}/{ranks[1]}"
                  f"   bench {info}")
        for seat in sorted(seats):
            wins, played = seats[seat]
            print(f"  {seat:<40} {wins:>5}/{played:<5} = {wins / played:6.2%}")
        rate = total_w / total_n
        half = 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / total_n)
        # The named arm is whichever the seat labels name; both seats report ITS wins, so
        # the total is already the arm's rate and needs no flip. The flip belongs in the
        # writer, and putting a second one here is how a sign gets lost.
        print(
            f"  total  {total_w}/{total_n} = {rate:.2%} +-{half:.2%}"
            f"   Elo {elo(rate):+.1f} [{elo(max(rate - half, 1e-6)):+.1f},"
            f" {elo(min(rate + half, 1 - 1e-6)):+.1f}]"
        )
        if unfinished:
            print(f"  ! {unfinished} games did not finish and are not in the total")
        if len(seats) == 2:
            rates = [w / n for w, n in seats.values()]
            gap = abs(rates[0] - rates[1])
            if gap > 4 * half:
                print(
                    f"  ! the two seats differ by {gap:.1%}, more than four times the "
                    "interval.\n    A fair match's seats differ by the seat advantage; "
                    "this is a harness question,\n    not a fact about the agents."
                )


if __name__ == "__main__":
    main()
