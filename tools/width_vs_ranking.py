"""Is a wide menu buying anything a better-ordered narrow one would not?

The matrix costs `limit**2`, so width 48 is four times width 24 and is most of what
generation spends. It is wide because `narrow`'s ordering is a heuristic and a good action
ranked 30th is an action the search cannot play. That argument has a corollary nobody has
tested: **if the ordering improves, the width can come down**, and the saving is quadratic
where every other saving discussed here is a few percent.

So this measures the ordering and the width together, on positions the search actually
meets, in the unit that decides it.

For each recorded position and each ordering (expected damage, or the leaf):

  excluded mass     how much of the wide equilibrium's weight sits on actions the narrow
                    menu does not contain. Zero means the narrow menu already holds
                    everything the wide solve wanted to play.
  what it costs     the narrow side plays its own equilibrium; the wide side answers with
                    its best reply over the whole wide menu. The gap from the wide game's
                    value is what the narrowing costs in win probability -- a
                    best-response gap, which is the honest way to price a restriction,
                    because a real opponent is not obliged to stay on our menu.

Both menus are solved as one matrix over their union, so the two answers are comparable
cell by cell rather than being two separate searches that happen to overlap.

    uv run --group learn python tools/width_vs_ranking.py --positions 40 \\
        --value data/models/value-all.pt --narrow 24 --wide 48
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from budget_effect import load_positions  # noqa: E402

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError, solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs  # noqa: E402
from pokeuraou.search import leaf_ranking  # noqa: E402


def _index(actions: list, lookup: dict) -> list[int]:
    return [lookup[str(a)] for a in actions]


def measure(reg, pos, evaluate, budget, narrow_width: int, wide_width: int, rank_by_leaf):
    """One position, one ordering. Returns (excluded mass, cost) per side, or None."""
    ranks = (
        [leaf_ranking(reg, pos, side, evaluate, budget=budget) for side in (0, 1)]
        if rank_by_leaf
        else [None, None]
    )
    wide = [narrow(reg, pos, s, limit=wide_width, rank=ranks[s]).actions for s in (0, 1)]
    tight = [narrow(reg, pos, s, limit=narrow_width, rank=ranks[s]).actions for s in (0, 1)]

    # One matrix over the union, so the narrow game is a submatrix of the wide one rather
    # than a separate search. A tight menu is usually a subset, but the coverage rule can
    # pick differently at a different limit, so it is taken as a union rather than assumed.
    union = []
    for side in (0, 1):
        merged = []
        for action in [*wide[side], *tight[side]]:
            if str(action) not in {str(a) for a in merged}:
                merged.append(action)
        union.append(merged)
    payoffs, _notes, _exact = batched_payoffs(
        reg, pos, union[0], union[1], [evaluate], budget=budget
    )
    matrix = payoffs[0]
    lookup = [{str(a): i for i, a in enumerate(union[s])} for s in (0, 1)]

    wide_rows, wide_cols = _index(wide[0], lookup[0]), _index(wide[1], lookup[1])
    tight_rows, tight_cols = _index(tight[0], lookup[0]), _index(tight[1], lookup[1])
    try:
        wide_eq = solve(matrix[np.ix_(wide_rows, wide_cols)])
        tight_eq = solve(matrix[np.ix_(tight_rows, tight_cols)])
    except EquilibriumError:
        return None

    out = {}
    for side in (0, 1):
        strategy = np.asarray(
            wide_eq.row_strategy if side == 0 else wide_eq.col_strategy, dtype=np.float64
        )
        kept = {str(a) for a in tight[side]}
        wide_actions = wide[side]
        out[f"excluded{side}"] = float(
            sum(w for a, w in zip(wide_actions, strategy, strict=True) if str(a) not in kept)
        )

    # Row player restricted to the tight menu, column player answering over the wide one.
    tight_x = np.asarray(tight_eq.row_strategy, dtype=np.float64)
    against = tight_x @ matrix[np.ix_(tight_rows, wide_cols)]
    out["cost0"] = float(wide_eq.value - against.min())
    tight_y = np.asarray(tight_eq.col_strategy, dtype=np.float64)
    against = matrix[np.ix_(wide_rows, tight_cols)] @ tight_y
    out["cost1"] = float(against.max() - wide_eq.value)
    out["pool"] = (len(union[0]), len(union[1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=24)
    ap.add_argument("--narrow", type=int, default=24, help="the width being tested")
    ap.add_argument("--wide", type=int, default=48, help="the width it is judged against")
    ap.add_argument("--games-dir", type=Path, default=ROOT / "data" / "selfplay-gen7")
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--value", type=Path, default=None, help="omit for the hp-share leaf")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = ap.parse_args()

    # The regulation comes from the positions, not from a flag: a recorded position
    # carries the format it was played under.
    positions = load_positions(args.games_dir, args.min_turn, args.seed)[: args.positions]
    if not positions:
        raise SystemExit(f"no recorded positions under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)
    if args.value is None:
        objective = OBJECTIVES["hp-share"]

        def evaluate(positions):
            return np.array([objective.batch([p])[0] for p in positions], dtype=np.float64)
    else:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        torch.set_num_threads(1)
        encoder = Encoder(reg)
        net, _meta = load_model(args.value, encoder)
        device = torch.device(args.device)
        evaluate = BatchedValue(net.to(device), encoder, device=device)

    budget = Budget.matrix()

    print(f"{len(positions)} recorded positions, width {args.narrow} judged against "
          f"{args.wide}, leaf {args.value.stem if args.value else 'hp-share'}")
    rows: dict[str, list[dict]] = {"damage": [], "leaf": []}
    started = time.perf_counter()
    for index, pos in enumerate(positions, 1):
        for name, by_leaf in (("damage", False), ("leaf", True)):
            got = measure(reg, pos, evaluate, budget, args.narrow, args.wide, by_leaf)
            if got is not None:
                rows[name].append(got)
        if index % 5 == 0:
            print(f"  {index}/{len(positions)}  ({time.perf_counter() - started:.0f}s)",
                  flush=True)

    print(f"\n{'ordering':<10} {'excluded mass':>14} {'>1% of it':>10} "
          f"{'cost (win prob)':>16} {'p90':>8} {'worst':>8}")
    for name in ("damage", "leaf"):
        got = rows[name]
        if not got:
            continue
        excluded = [g[f"excluded{s}"] for g in got for s in (0, 1)]
        cost = sorted(g[f"cost{s}"] for g in got for s in (0, 1))
        print(f"{name:<10} {float(np.mean(excluded)):>14.4f} "
              f"{sum(1 for e in excluded if e > 0.01) / len(excluded):>9.0%} "
              f"{float(np.mean(cost)):>16.4f} "
              f"{cost[int(len(cost) * 0.9)]:>8.4f} {cost[-1]:>8.4f}")
    print(f"\n  the matrix is limit^2, so width {args.narrow} costs "
          f"{(args.narrow / args.wide) ** 2:.2f}x of width {args.wide}")


if __name__ == "__main__":
    main()
