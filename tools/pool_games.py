"""Pool a match from the recorded games rather than from the workers' summaries.

`tools/pool_matches.py` reads the per-seat line each worker writes when it finishes, which
is the right thing when every worker finishes. A worker that died, or one still grinding a
pathological node while the rest are done, never writes it -- and its games, which are on
disk and perfectly good, are then invisible.

The games carry everything needed. Each line has the outcome and a provenance block naming
both sides' agents, so which arm won is read off the record rather than inferred from which
file it landed in.

    uv run python tools/pool_games.py --dir data/matches/width-12-vs-48 --arm value-allx2@w12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.provenance import agent_name  # noqa: E402


def wilson(wins: int, games: int) -> tuple[float, float]:
    """95% interval. Wilson rather than normal: seats here can be small and lopsided."""
    if games == 0:
        return (0.0, 0.0)
    z = 1.959963985
    p = wins / games
    centre = (p + z * z / (2 * games)) / (1 + z * z / games)
    half = (
        z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games))
    ) / (1 + z * z / games)
    return (100 * (centre - half), 100 * (centre + half))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--glob", default="games-*.jsonl")
    ap.add_argument(
        "--arm",
        default=None,
        help="the agent whose win rate to report; default is whichever agent sits on "
        "side 0 in the first record",
    )
    args = ap.parse_args()

    games: list[tuple[tuple[str, str], float]] = []
    agents: Counter[str] = Counter()
    unlabelled = 0
    for path in sorted(args.dir.glob(args.glob)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            source = record.get("provenance")
            outcome = record.get("outcome")
            if source is None or outcome is None:
                unlabelled += 1
                continue
            names = (agent_name(source, 0), agent_name(source, 1))
            agents[names[0]] += 1
            agents[names[1]] += 1
            games.append((names, float(outcome)))

    if not games:
        raise SystemExit(f"no usable games under {args.dir}")
    # Fixed once, for every record. Choosing it per record is the mistake this tool made
    # first: the seats swap halfway, so "whoever is on side 0 here" is one arm in half the
    # games and the other arm in the rest, and the two get pooled into a number that is
    # neither. It read +4.0 on a run that was really -5.6.
    arm = args.arm or games[0][0][0]

    seats: dict[str, list[int]] = {}
    for names, outcome in games:
        if arm not in names:
            continue
        side = 0 if names[0] == arm else 1
        # `outcome` is side 0's result, so flip it when the arm sits at 1.
        won = outcome > 0.5 if side == 0 else outcome < 0.5
        tally = seats.setdefault(f"{arm} = side {side}", [0, 0])
        tally[0] += int(won)
        tally[1] += 1

    if not seats:
        raise SystemExit(
            f"no games naming {arm!r}. Agents present: {', '.join(sorted(agents))}"
        )
    other = next((n for names, _ in games for n in names if n != arm), "?")
    # Named, because the win rate of one arm is the complement of the other's and a table
    # that does not say which is which is a table that will be read the wrong way round.
    print(f"\n  {arm}  vs  {other}\n")
    print(f"  {'seat':>34}  {'games':>6}  {'arm win':>8}  {'95% (Wilson)':>16}")
    total_wins = total_games = 0
    for label in sorted(seats):
        wins, games = seats[label]
        low, high = wilson(wins, games)
        print(f"  {label:>34}  {games:>6}  {100 * wins / games:>7.1f}%  "
              f"[{low:>5.1f}, {high:>5.1f}]")
        total_wins += wins
        total_games += games
    low, high = wilson(total_wins, total_games)
    rate = 100 * total_wins / total_games
    print(f"\n  {'combined':>34}  {total_games:>6}  {rate:>7.1f}%  [{low:>5.1f}, {high:>5.1f}]")
    print(f"  a = {rate - 50:+.1f} points"
          + ("  (significant)" if low > 50 or high < 50 else "  (not significant)"))
    if unlabelled:
        print(f"  {unlabelled} record(s) had no outcome or no provenance and were skipped")


if __name__ == "__main__":
    main()
