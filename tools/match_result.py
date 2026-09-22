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


def read(
    directory: Path,
) -> tuple[dict[str, tuple[int, int]], dict[str, dict], int, int, list[float]]:
    """Per-seat (wins, played), a provenance per seat, unfinished, games, and the clock.

    The clock is `[tested arm seconds, other arm seconds, move decisions]`, summed over
    every row. It is what an "at equal wall clock" claim is made of and it is absent from
    every row written before it existed, which is why it is summed rather than averaged:
    a run half of whose workers predate the field would otherwise report the half.
    """
    seats: dict[str, tuple[int, int]] = {}
    clock = [0.0, 0.0, 0.0]
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
            clock[0] += float(row.get("armSeconds", 0.0))
            clock[1] += float(row.get("otherArmSeconds", 0.0))
            clock[2] += float(row.get("moveDecisions", 0.0))
    # Every game record on disk, counted separately from the summary rows.
    #
    # A summary row is written by a worker when it FINISHES. A worker that dies has
    # written its games and no row, so the rate divides the survivors' wins by the
    # survivors' games and reads as a complete run: `width-12-vs-48` prints 1,489 games
    # with 1,695 on disk, and `genmatch-value-gen9x2-vs-value-allx2` prints 1,635 with
    # 1,695 and an interval that excludes zero. Both directories have DONE files saying
    # FAILED, which this tool never reads either.
    written = 0
    provenance: dict[str, dict] = {}
    for path in sorted(directory.glob("games-worker*.jsonl")) + sorted(
        directory.glob("games-seed*.jsonl")
    ):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                written += 1
                source = json.loads(line).get("provenance", {})
                provenance.setdefault(str(source.get("seat", "?")), source)
    return seats, provenance, unfinished, written, clock


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", type=Path, nargs="+")
    args = ap.parse_args()

    for directory in args.dirs:
        seats, provenance, unfinished, written, clock = read(directory)
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
        # What each arm spent per move decision, from inside this run. Both arms are
        # timed on the same machine in the same minute over the same positions, which is
        # what the previous depth-2 cost -- `s/game` from one run divided by decisions
        # from another -- was not.
        if clock[2]:
            tested, other = clock[0] / clock[2], clock[1] / clock[2]
            names = (next(iter(provenance.values()), {}).get("leaves") or ["?", "?"])[:2]
            print(
                f"  per move decision: tested {tested:.4f}s, other {other:.4f}s"
                f" = {tested / other if other else float('nan'):.2f}x"
                f"   ({int(clock[2]):,} move decisions, leaves {names[0]}/{names[1]})"
            )
        if unfinished:
            print(f"  ! {unfinished} games did not finish and are not in the total")
        marker = directory / "DONE"
        said = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
        if written and written != total_n:
            print(
                f"  ! {written} games are on disk and only {total_n} are in this total.\n"
                "    A summary row is written when a worker FINISHES, so a worker that\n"
                "    died left its games behind and no row -- and the rate then divides\n"
                "    the survivors' wins by the survivors' games and reads as a\n"
                "    complete run."
            )
        if "FAIL" in said.upper():
            print(f"  ! this run's own DONE marker says: {said}")
        if len(seats) == 2:
            rates = [w / n for w, n in seats.values()]
            gap = abs(rates[0] - rates[1])
            if gap > 4 * half:
                # Side 0's own rate, computed rather than asserted. This note used to
                # say "whichever arm sits at side 0 wins under half, in both seats" and
                # quote the book's 0.4965 -- and on 2026-09-19 it printed that directly
                # under a table showing side 0 at 53.1% and 56.5%. An explanation that
                # contradicts the numbers beside it teaches the reader to skip the
                # warning, which is worse than having no explanation at all.
                # The key is the seat LABEL the writer chose -- "value-gen11L = side 0" --
                # not "0". Comparing it to "0" matched neither seat and quietly took the
                # other branch for both, printing 51.7% where the arithmetic gives 54.8%.
                # So it is parsed, and when it cannot be parsed the line is not printed:
                # a number nobody can derive is worse than a missing one.
                def at_side0(label: str) -> bool | None:
                    text = label.strip()
                    if text.endswith("side 0"):
                        return True
                    if text.endswith("side 1"):
                        return False
                    return None

                marks = {seat: at_side0(seat) for seat in seats}
                if None in marks.values():
                    side0_line = (
                        "    Side 0's own rate is not shown: these seat labels do not\n"
                        f"    end in 'side 0' or 'side 1' ({sorted(seats)}).\n"
                    )
                else:
                    side0 = sum(
                        w if marks[seat] else n - w for seat, (w, n) in seats.items()
                    )
                    played = sum(n for _w, n in seats.values())
                    side0_line = (
                        f"    Side 0 scored {side0 / played:.1%} across both seats.\n"
                    )
                print(
                    f"  ! the seats differ by {gap:.1%}, over four times the interval.\n"
                    f"{side0_line}"
                    "    Side 0 is\n"
                    "    our roster and side 1 a tournament team, so a gap is expected\n"
                    "    whenever the two are not evenly matched, and swapping the seats\n"
                    "    is what cancels it -- which the total above does.\n"
                    "    A defect looks different: the gap survives the swap, or the\n"
                    "    tested arm beats its own mirror. Compare side 0's rate here\n"
                    "    against the book's mean equilibrium value for the roster\n"
                    "    (`tools/ordering_result.py` prints that comparison per opponent);\n"
                    "    a large disagreement is the matrix, not the harness."
                )


if __name__ == "__main__":
    main()
