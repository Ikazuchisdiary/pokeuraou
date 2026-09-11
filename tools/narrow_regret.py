"""How much the candidate menu costs, measured against the search's own evaluation.

A doubles action is a *pair* of slot choices, and there are a few hundred legal pairs per
side. `narrow` keeps 24: every individual slot option appears somewhere (so nothing is
eliminated outright), and the rest of the budget goes to the highest-scoring combinations,
scored by average damage fraction.

That score is not the thing the search evaluates with. The matrix is filled by a learned
value function, and the two can disagree: a weak move that leads somewhere the value
function likes ranks low on damage and is only ever offered next to whatever partner the
coverage rule happened to pair it with. The combination of the two individually best
choices is not guaranteed to be on the menu at all.

So this measures the disagreement in the only unit that matters. For a recorded decision:
solve the matrix the search actually solved, then score *every legal pair* for one side
against that same opponent strategy, with the same leaf and the same budget. The gap
between the best legal reply and the best reply that was on the menu is what the narrowing
cost, in win probability.

It is a *best-response* gap, not a change in the equilibrium value: if the missing action
were offered, the other side would answer it too. What it does establish is whether the
answer the search prints is an equilibrium of the game or of the menu -- a side holding a
deviation worth ten points is not at equilibrium in any useful sense.

    uv run --group learn python tools/narrow_regret.py --dir data/selfplay-gen4 --games 40
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.actions import side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow
from pokeuraou.position import Position
from pokeuraou.resolve import Budget, batched_payoff
from pokeuraou.search import leaf_ranking
from pokeuraou.teams import load_roster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/selfplay-gen4"))
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--value", type=Path, default=None, help="default: the game's own leaf")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--per-game", type=int, default=3, help="decisions sampled per game")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument(
        "--leaf-rank",
        action="store_true",
        help="rank candidates with the leaf instead of the damage score. The regret this "
        "tool measures is the gap between the menu and the full legal set *by the leaf*, "
        "so ranking by the leaf should shrink it -- and if it does not, the ranking is "
        "not doing what it was built to do.",
    )
    args = ap.parse_args()

    reg = load_roster(args.roster).reg
    register_mega_stones(reg)

    import torch

    torch.set_num_threads(1)
    from pokeuraou.encode import Encoder
    from pokeuraou.value import BatchedValue, load_model

    files = sorted(glob.glob(str(args.dir / "games-seed*.jsonl")))
    if not files:
        raise SystemExit(f"no games under {args.dir}")

    records = []
    with open(files[0], encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= args.games:
                break
            records.append(json.loads(line))

    leaf_name = args.value.stem if args.value else records[0]["searchObjective"].split(":", 1)[1]
    encoder = Encoder(reg)
    net, _meta = load_model(Path("data/models") / f"{leaf_name}.pt", encoder)
    evaluate = BatchedValue(net, encoder).__call__
    ranking = "leaf" if args.leaf_rank else "damage"
    print(f"leaf {leaf_name}, width {args.limit}, ranking by {ranking}, {len(records)} games")

    rng = np.random.default_rng(args.seed)
    regrets: list[float] = []
    offered_sizes: list[int] = []
    legal_sizes: list[int] = []
    already_best = 0
    checked = 0

    for record in records:
        moves = [d for d in record["decisions"] if d.get("kind") == "move"]
        if not moves:
            continue
        pick = rng.choice(len(moves), size=min(args.per_game, len(moves)), replace=False)
        for which in pick:
            decision = moves[int(which)]
            pos = Position.from_json(decision["position"])
            ranks = (
                {
                    side: leaf_ranking(
                        reg, pos, side, evaluate, budget=Budget.matrix()
                    )
                    for side in (0, 1)
                }
                if args.leaf_rank
                else {0: None, 1: None}
            )
            row = narrow(reg, pos, 0, limit=args.limit, rank=ranks[0]).actions
            col = narrow(reg, pos, 1, limit=args.limit, rank=ranks[1]).actions
            if not row or not col:
                continue
            payoff, _n = batched_payoff(
                reg, pos, row, col, evaluate, budget=Budget.matrix()
            )
            equilibrium = solve(payoff)

            everything = side_actions(reg, pos, 1)
            full, _n = batched_payoff(
                reg, pos, row, everything, evaluate, budget=Budget.matrix()
            )
            # Row player's payoff per column action, so the column player wants it low.
            ev = equilibrium.row_strategy @ full
            offered = {a.to_choice() for a in col}
            on_menu = [i for i, a in enumerate(everything) if a.to_choice() in offered]
            if not on_menu:
                continue
            best_offered = float(ev[on_menu].min())
            best_legal = float(ev.min())
            regrets.append(best_offered - best_legal)
            offered_sizes.append(len(col))
            legal_sizes.append(len(everything))
            already_best += int(best_offered - best_legal < 1e-9)
            checked += 1

    if not checked:
        raise SystemExit("no decisions measured")
    regrets.sort()
    print(f"\n{checked} decisions measured")
    print(
        f"  menu {statistics.mean(offered_sizes):.0f} of "
        f"{statistics.mean(legal_sizes):.0f} legal pairs on average"
    )
    print(f"  the best legal reply was already on the menu: {already_best} of {checked}")
    print(
        f"  regret  mean {statistics.mean(regrets):.4f}  median {regrets[len(regrets) // 2]:.4f}"
        f"  p90 {regrets[int(len(regrets) * 0.9)]:.4f}  max {regrets[-1]:.4f}"
    )
    for threshold in (0.02, 0.05, 0.10):
        n = sum(1 for r in regrets if r > threshold)
        print(f"  over {threshold:.2f}: {n} of {checked} ({100 * n / checked:.0f}%)")


if __name__ == "__main__":
    main()
