"""Pools independent head-to-head runs into one verdict, per seat and overall.

Fourteen runs of 400 games each answer the question far better than any one of them, but
only if they are combined properly. Two things this does that eyeballing the per-run lines
does not:

- **pools the counts, not the percentages.** Averaging fourteen rates weights a run that
  discarded more games equally with one that discarded fewer.
- **reports each seat separately, and decomposes the difference.** The two seats disagree
  hugely here -- 66.1% against 42.4% -- and that is expected rather than alarming: side 0
  always holds our roster and side 1 always holds a tournament team, so swapping which
  *generation* sits where does not swap which *team* does. Writing R for the roster's win
  rate and `a` for the new generation's edge:

      seat A:  roster+new vs field+old   ->  R + a
      seat B:  roster+old vs field+new   ->  (1 - R) + a
      average                            ->  0.5 + a

  so the mean of the two seats is the estimator of `a`, and the seats' difference recovers
  R. That R can then be checked against the independently measured roster-versus-field rate,
  which is the closest thing to a free validation of the whole setup.

Reads either the JSONL written by `--out` or the redirect files, so runs that predate the
JSONL option still count.

    uv run python tools/pool_matches.py --dir <scratch dir>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: The seat label is whatever the producing tool called its arm -- it used to be the
#: literal "gen2", and is now the model's filename -- so the name is matched loosely and
#: only its shape, `<something> = side N`, is relied on.
SEAT_LINE = re.compile(r"(\S[^\s]* = side \d)\s+(\d+)\s+([\d.]+)%")


def wilson(wins: int, played: int) -> tuple[float, float, float]:
    """Point estimate and a 95% Wilson interval, which behaves near 0 and 1.

    The normal approximation is fine at these counts, but Wilson costs nothing and never
    produces a bound outside [0, 1] when a seat happens to be lopsided.
    """
    if played == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.959963984540054
    p = wins / played
    denom = 1 + z * z / played
    centre = (p + z * z / (2 * played)) / denom
    half = z * ((p * (1 - p) / played + z * z / (4 * played * played)) ** 0.5) / denom
    return p, centre - half, centre + half


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--glob", default="genmatch-*.txt")
    ap.add_argument("--jsonl", default="*.jsonl")
    args = ap.parse_args()

    seats: dict[str, list[tuple[int, int]]] = {}

    for path in sorted(args.dir.glob(args.jsonl)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            # The win count's key names whichever arm is being measured -- `gen2_wins`
            # for a generation match, `wide_wins` for a candidate-width match. Pooling is
            # the same arithmetic either way, so the key is read from a list rather than
            # forcing every producer to call its arm "gen2".
            wins = next(
                (row[key] for key in ("gen2_wins", "wide_wins", "wins") if key in row),
                None,
            )
            if wins is not None and "played" in row:
                seats.setdefault(row["seat"], []).append((wins, row["played"]))

    # The text form carries the rate rather than the count, so wins are recovered from it.
    # Rounding costs at most half a game per row, which is nothing against 400.
    for path in sorted(args.dir.glob(args.glob)):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = SEAT_LINE.search(line)
            if match is None:
                continue
            seat, played, rate = match.group(1), int(match.group(2)), float(match.group(3))
            seats.setdefault(seat, []).append((round(played * rate / 100), played))

    if not seats:
        raise SystemExit(f"no results found in {args.dir}")

    print(f"  {'seat':>16}  {'runs':>5}  {'games':>7}  {'arm win':>9}  {'95% (Wilson)':>16}")
    total_wins = total_played = 0
    for seat, rows in sorted(seats.items()):
        wins = sum(w for w, _ in rows)
        played = sum(n for _, n in rows)
        p, low, high = wilson(wins, played)
        print(
            f"  {seat:>16}  {len(rows):>5}  {played:>7}  {p * 100:>8.1f}%  "
            f"[{low * 100:5.1f}, {high * 100:5.1f}]"
        )
        total_wins += wins
        total_played += played

    p, low, high = wilson(total_wins, total_played)
    print(
        f"\n  {'combined':>16}  {sum(len(r) for r in seats.values()):>5}  "
        f"{total_played:>7}  {p * 100:>8.1f}%  [{low * 100:5.1f}, {high * 100:5.1f}]"
    )
    if low > 0.5:
        print("  → gen2 が有意に強い。次世代のデータ生成に使える。")
    elif high < 0.5:
        print("  → gen2 が有意に弱い。生成する前に原因を出すべき。")
    else:
        print(
            f"  → 有意でない。幅が {(high - low) * 100:.1f} ポイントなので、"
            "この標本では差を判定できない。"
        )
    if len(seats) == 2:
        # seat A is "new generation on side 0", i.e. on our roster.
        by_seat = sorted(seats.items())
        rates = [
            sum(w for w, _ in rows) / max(sum(n for _, n in rows), 1)
            for _, rows in by_seat
        ]
        # R + a = rates[0] and (1 - R) + a = rates[1], so:
        edge = (rates[0] + rates[1]) / 2 - 0.5
        roster = (rates[0] - rates[1] + 1) / 2
        print("")
        print(
            f"  分解: 自陣チームの勝率 R = {roster * 100:.1f}%、"
            f"新世代の上積み a = {edge * 100:+.1f} ポイント"
        )
        print(
            "  席差はチーム強さの差なので大きくて当然"
            "（side 0 は常に自陣、side 1 は常に大会チーム）。"
        )
        print(
            "  R を独立測定の自陣 vs 大会（13,395 ゲームで 60.9%）と比べると、"
            "この実験構成そのものの検算になる。"
        )


if __name__ == "__main__":
    main()
