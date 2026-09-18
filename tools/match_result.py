"""One match, read back: the win rate per seat, the pair, and what it was played under.

Every match in `data/matches/` is read the same way and the arithmetic is three lines, so
it has been written inline each time it was wanted -- which is how a seat label gets read
as the wrong arm, and how a number ends up quoted without the condition it was measured
in. Both have happened here.

The configuration is printed PER SEAT. Everything in a provenance block is ordered by side
and the two arms swap sides halfway, so one sampled record prints one seat's ordering as
if it were the arms' -- which read as "gen10's book against gen11L's" for a match whose
tested arm held gen11L's. It is worse when both arms share a leaf name, because then
nothing else in the line tells them apart.

    uv run python tools/match_result.py data/matches/gen11L-vs-gen10-ownbooks-fixed
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


def read(directory: Path) -> tuple[dict[str, tuple[int, int]], dict[str, dict], int]:
    """Per-seat (wins, played) for the named arm, a provenance per seat, and unfinished."""
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
    provenance: dict[str, dict] = {}
    for path in sorted(directory.glob("games-worker*.jsonl")) + sorted(
        directory.glob("games-seed*.jsonl")
    ):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line).get("provenance", {})
                provenance.setdefault(str(source.get("seat", "?")), source)
        if seats and len(provenance) >= len(seats):
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
        sample = next(iter(provenance.values()), {})
        if sample:
            info = (sample.get("information") or ["?", "?"])[0]
            print(f"  bench {info}    side 0 is our roster, side 1 a tournament team")
        for seat in sorted(seats):
            wins, played = seats[seat]
            source = provenance.get(seat, {})
            leaves = source.get("leaves") or ["?", "?"]
            books = source.get("books") or ["?", "?"]
            ranks = source.get("rankings") or ["?", "?"]
            limits = source.get("limits") or ["?", "?"]
            print(f"  {seat:<38} {wins:>5}/{played:<5} = {wins / played:6.2%}")
            for side in (0, 1):
                print(
                    f"      side {side}  {leaves[side]} / {books[side]}"
                    f" / w{limits[side]} / {ranks[side]}"
                )
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
                    f"  ! the seats differ by {gap:.1%}, over four times the interval.\n"
                    "    Not automatically a defect. Side 0 is our roster and side 1 a\n"
                    "    tournament team, and the roster is slightly behind that field:\n"
                    "    gen11L's book puts our mean equilibrium value at 0.4965 and\n"
                    "    gen10's at 0.4899, with over half the teams under 50%. Whichever\n"
                    "    arm sits at side 0 therefore wins under half, in both seats, and\n"
                    "    swapping the seats is what cancels it -- which the total does.\n"
                    "    Look here for a defect when the gap survives the swap, or when a\n"
                    "    uniform-draw match shows one (the anchor was 62.62% / 62.97%)."
                )


if __name__ == "__main__":
    main()
