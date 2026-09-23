"""Is the win rate about the teams, or about how well each side is searched?

Self-play reported 72.4% for our team, which is high enough to suspect the machinery
rather than the matchup. Both sides are solved by the same LP over the same kind of
candidate set, so there is no *code* asymmetry -- but both sides are also narrowed to the
same small number of candidates, and a small budget does not cost every team equally. A
team whose good turns need an unusual combination (a switch plus a redirect, a setup under
a Protect) loses more by being cut to eight candidates than a team that just attacks.

So vary the budget on one side only and watch the win rate. If our advantage survives the
opponent being searched twice as widely, it is the matchup. If it does not, the earlier
number was measuring our own search.

    uv run python tools/asymmetry.py --games 150 --pairs 8x8,8x16,8x24,16x8

**Open game only: a reference** (IKA-123). This tool has no hidden-bench path, so its
search is shown the opponent's four -- not the game that ships. It says so on stderr
when it runs, and `tools/agent_drift.py` lists it under KNOWN_DRIFT.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.benchflags import say_open_reference
from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.selfplay import play_game
from pokeuraou.teams import load_archetypes, load_roster, pick_four, sample_archetype, usable_archetypes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=150, help="games per pair")
    ap.add_argument("--pairs", default="8x8,8x16,16x8")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--archetypes", default="wcs2026-regmb")
    ap.add_argument("--objective", default="hp-share")
    args = ap.parse_args()
    say_open_reference()

    roster = load_roster(args.roster)
    reg, archetypes = load_archetypes(args.archetypes)
    register_mega_stones(reg)
    chaos = find_cached_chaos(reg.meta.format_id)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    usable, _ = usable_archetypes(prior, archetypes)
    objective = OBJECTIVES[args.objective]

    pairs = []
    for token in args.pairs.split(","):
        a, b = token.strip().split("x")
        pairs.append((int(a), int(b)))

    print("  {:>9}  {:>6}  {:>7}  {:>9}  {:>6}".format(
        "ours x foe", "games", "our win", "unfinished", "s/game"))
    for ours_limit, foe_limit in pairs:
        # The same seed for every pair, so the teams and the RNG draws line up and the
        # only difference between rows is the budget.
        rng = np.random.default_rng(args.seed)
        wins = 0
        finished = 0
        unfinished = 0
        started = time.perf_counter()
        for _ in range(args.games):
            archetype = usable[int(rng.integers(len(usable)))]
            foe_six = sample_archetype(rng, reg, prior, archetype)
            own_four = pick_four(rng, roster.sets, size=reg.meta.picked_team_size)
            foe_four = pick_four(rng, foe_six, size=reg.meta.picked_team_size)
            record = play_game(
                reg, rng, own_four, foe_four, archetype.id,
                objective=objective,
                search_limit=(ours_limit, foe_limit),
                max_turns=args.max_turns,
                open_information=True,
            )
            if record.outcome is None:
                unfinished += 1
                continue
            finished += 1
            wins += int(record.outcome > 0.5)
        elapsed = time.perf_counter() - started
        rate = wins / finished * 100 if finished else float("nan")
        print("  {:>9}  {:>6}  {:>6.1f}%  {:>9}  {:>6.2f}".format(
            f"{ours_limit}x{foe_limit}", finished, rate, unfinished, elapsed / args.games))


if __name__ == "__main__":
    main()
