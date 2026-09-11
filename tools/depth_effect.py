"""Does the second ply actually change the answer, and how often?

A head-to-head takes an hour and comes back 50% either because depth does not help or
because depth did nothing at all -- a matrix whose supported cells cannot be refined, an
equilibrium so pure that one cell is the whole game, a refinement that lands on the same
number. Those are different failures with the same win rate, and this separates them
before the hour is spent.

Reports, over positions from real games:

    refined      cells that got a depth-2 value
    moved        how far the equilibrium value moved
    strategy     how far the mixture moved, in total variation
    reordered    how often the most-played action changed

The last is the one that matters for play: a value that shifts by a hundredth while the
strategy is identical cannot win a single game.

    uv run --group learn python tools/depth_effect.py --value data/models/value-gen234.pt
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.resolve import Budget, resolve_turn
from pokeuraou.search import search
from pokeuraou.selfplay import position_from_sets
from pokeuraou.standings import find_cached_standings, load_standings, sample_standings_team
from pokeuraou.teams import all_selections, load_roster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--value", type=Path, default=None, help="omit for the hp-share leaf")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    evaluate = OBJECTIVES["hp-share"].batch
    if args.value is not None:
        import torch

        torch.set_num_threads(1)
        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        encoder = Encoder(load_roster(args.roster).reg)
        net, _meta = load_model(args.value, encoder)
        evaluate = BatchedValue(net.to(torch.device(args.device)), encoder).__call__

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
    rng = np.random.default_rng(args.seed)

    refined: list[int] = []
    subgames: list[int] = []
    moved: list[float] = []
    drift: list[float] = []
    reordered = decisions = unrefinable = 0
    shallow_seconds = deep_seconds = 0.0

    for _ in range(args.games):
        team = pool[int(rng.integers(len(pool)))]
        foe_six = sample_standings_team(rng, reg, prior, team)
        own_pick = selections[int(rng.integers(len(selections)))]
        foe_pick = selections[int(rng.integers(len(selections)))]
        pos = position_from_sets(
            reg, [roster.sets[i] for i in own_pick], [foe_six[j] for j in foe_pick]
        )
        for _turn in range(12):
            if pos.ended:
                break
            ours = narrow(reg, pos, 0, limit=args.limit).actions
            theirs = narrow(reg, pos, 1, limit=args.limit).actions
            if not ours or not theirs:
                break
            started = time.perf_counter()
            shallow = search(reg, pos, ours, theirs, evaluate, budget=Budget.matrix())
            shallow_seconds += time.perf_counter() - started
            started = time.perf_counter()
            deep = search(
                reg, pos, ours, theirs, evaluate, budget=Budget.matrix(), depth=2
            )
            deep_seconds += time.perf_counter() - started

            decisions += 1
            refined.append(deep.refined)
            subgames.append(deep.subgames)
            if deep.refined == 0:
                unrefinable += 1
            moved.append(abs(deep.equilibrium.value - shallow.equilibrium.value))
            drift.append(
                0.5
                * float(
                    np.abs(
                        deep.equilibrium.row_strategy - shallow.equilibrium.row_strategy
                    ).sum()
                )
            )
            if int(np.argmax(deep.equilibrium.row_strategy)) != int(
                np.argmax(shallow.equilibrium.row_strategy)
            ):
                reordered += 1

            chosen = [
                ours[int(np.argmax(deep.equilibrium.row_strategy))],
                theirs[int(np.argmax(deep.equilibrium.col_strategy))],
            ]
            result = resolve_turn(reg, pos, chosen, budget=Budget.exact())
            if result.suspended or not result.branches:
                break
            weights = np.array([b.probability for b in result.branches])
            pos = result.branches[
                int(rng.choice(len(weights), p=weights / weights.sum()))
            ].position

    if not decisions:
        raise SystemExit("no decisions searched")
    print(f"{decisions} decisions over {args.games} games, width {args.limit}")
    print(
        f"  refined cells   mean {statistics.mean(refined):.1f}  "
        f"max {max(refined)}  zero on {unrefinable} of {decisions} decisions"
    )
    print(f"  subgames solved mean {statistics.mean(subgames):.1f}  max {max(subgames)}")
    print(
        f"  value moved     mean {statistics.mean(moved):.4f}  "
        f"max {max(moved):.4f}  median {statistics.median(moved):.4f}"
    )
    print(
        f"  strategy moved  mean {statistics.mean(drift):.4f}  "
        f"max {max(drift):.4f}  (total variation, 0 = identical mixture)"
    )
    print(
        f"  top action changed on {reordered} of {decisions} decisions "
        f"({100 * reordered / decisions:.1f}%)"
    )
    print(
        f"  cost  depth 1 {shallow_seconds / decisions:.2f} s/decision, "
        f"depth 2 {deep_seconds / decisions:.2f} s/decision "
        f"({deep_seconds / max(shallow_seconds, 1e-9):.2f}x)"
    )


if __name__ == "__main__":
    main()
