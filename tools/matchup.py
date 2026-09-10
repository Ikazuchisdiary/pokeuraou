"""Is the win rate the matchup, or the machinery?

Our team wins 62.3% against the tournament field, which is high enough to keep suspecting
a bias rather than a matchup. The two are separable, and neither needs a trained value
function to separate them:

- **field vs field**: draw *both* sides from the standings. Two random tournament entries
  have no reason to average anything but 50%, so whatever this comes out as is the
  machinery's own bias -- side-0 advantage, a resolver asymmetry, anything.
- **our team as side 1**: if side 0 and side 1 are symmetric, playing our six from the
  other seat should give exactly one minus its side-0 rate.
- **against the top cut instead of the field**: the field includes entries that went 1-6.
  If our rate against the 57 who reached day two is much lower, the 62.3% is diluted by
  weak opposition, which makes it a real matchup number rather than an artefact.

    uv run python tools/matchup.py --games 400 --pairs roster:field,field:field,field:roster
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.selfplay import play_game
from pokeuraou.standings import find_cached_standings, load_standings, sample_standings_team
from pokeuraou.teams import load_roster, pick_four

SOURCES = ("roster", "field", "phase2", "cut")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=400, help="games per pair")
    ap.add_argument("--pairs", default="roster:field,field:field,field:roster,roster:phase2")
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--objective", default="hp-share")
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    chaos = find_cached_chaos(reg.meta.format_id)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    cached = find_cached_standings()
    if cached is None:
        raise SystemExit("no cached standings; run tools/fetch_standings.py 2026 worlds")
    standings = load_standings(cached, reg)
    objective = OBJECTIVES[args.objective]

    pools = {
        "field": standings.pool("all"),
        "phase2": standings.pool("phase2"),
        "cut": standings.pool("cut"),
    }

    def draw(source: str, rng: np.random.Generator) -> list:
        if source == "roster":
            return roster.sets
        pool = pools[source]
        return sample_standings_team(rng, reg, prior, pool[int(rng.integers(len(pool)))])

    print(
        f"  {'side0 vs side1':>22}  {'games':>6}  {'side0 win':>9}  "
        f"{'unfinished':>10}  {'s/game':>6}"
    )
    for token in args.pairs.split(","):
        left, right = (s.strip() for s in token.strip().split(":"))
        for name in (left, right):
            if name not in SOURCES:
                raise SystemExit(f"unknown source {name!r}; use one of {SOURCES}")
        # Same seed for every row, so the only difference between rows is who plays.
        rng = np.random.default_rng(args.seed)
        wins = finished = unfinished = 0
        started = time.perf_counter()
        for _ in range(args.games):
            own_six = draw(left, rng)
            foe_six = draw(right, rng)
            record = play_game(
                reg,
                rng,
                pick_four(rng, own_six, size=reg.meta.picked_team_size),
                pick_four(rng, foe_six, size=reg.meta.picked_team_size),
                f"{left}:{right}",
                objective=objective,
                search_limit=args.limit,
                max_turns=args.max_turns,
            )
            if record.outcome is None:
                unfinished += 1
                continue
            finished += 1
            wins += int(record.outcome > 0.5)
        elapsed = time.perf_counter() - started
        rate = wins / finished * 100 if finished else float("nan")
        # A 95% interval, so a difference between rows can be read as real or not.
        half = (
            1.96 * (rate / 100 * (1 - rate / 100) / finished) ** 0.5 * 100
            if finished
            else float("nan")
        )
        print(
            f"  {left + ' vs ' + right:>22}  {finished:>6}  {rate:>7.1f}%  "
            f"+-{half:.1f}  {unfinished:>6}  {elapsed / args.games:>6.2f}"
        )


if __name__ == "__main__":
    main()
