"""What a narrowing costs, on the menu `narrow` actually builds.

`tools/policy_ceiling.py` compared orderings by reordering a 48-wide menu and keeping the
top W. That is not what happens. `narrow` is not a top-W: it first makes sure every
individual slot option appears somewhere, so that nothing is eliminated outright, and only
then spends what is left on the highest-scoring combinations. An ordering changes which
combinations win that second part; it does not get to skip the first.

Worse, the two orderings were not treated alike. The leaf's menu came from `narrow` -- so
it carried the coverage rule -- and the learned ordering's came from `argsort(-score)[:W]`,
which is the best menu its scores could possibly produce and one it is never given. The
comparison flattered the policy, and the board disagreed with it by 4.1 points.

So here every menu is built by calling `narrow` with that ordering, exactly as a search
does, and the reference is the whole legal pool rather than a 48-wide menu of it -- which
is nearly the same thing, since the midgame pool averages 42 legal pairs, but it is the
same thing *by construction* rather than by luck.

    uv run --group learn python tools/narrow_as_built.py --positions 60 \\
        --value data/models/value-all.pt --policy data/models/policy-gen7.pt
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
from pokeuraou.policy import load_policy  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs  # noqa: E402
from pokeuraou.search import leaf_ranking  # noqa: E402

#: Wider than any legal pool, so `narrow` returns all of it and the reference is the whole
#: game rather than a wide menu of it.
EVERYTHING = 4096


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--widths", default="6,8,12,16,24,32")
    ap.add_argument("--games-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data" / "selfplay-gen8")
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--value", type=Path, required=True)
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",") if w]
    positions = load_positions(args.games_dir, args.min_turn, args.seed)[: args.positions]
    if not positions:
        raise SystemExit(f"no recorded positions under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)

    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.value import BatchedValue, load_model

    torch.set_num_threads(1)
    encoder = Encoder(reg)
    net, _meta = load_model(args.value, encoder)
    device = torch.device(args.device)
    evaluate = BatchedValue(net.to(device), encoder, device=device)
    policy = (
        load_policy(args.policy, reg, "cpu", value_net=evaluate.net)
        if args.policy
        else None
    )
    budget = Budget.matrix()

    names = ["leaf"] + (["policy"] if policy else []) + ["damage"]
    costs: dict[str, dict[int, list[float]]] = {n: {} for n in names}
    pools: list[int] = []
    started = time.perf_counter()
    for index, pos in enumerate(positions, 1):
        ranks = {
            "leaf": [leaf_ranking(reg, pos, s, evaluate, budget=budget) for s in (0, 1)],
            "damage": [None, None],
        }
        if policy:
            # `at` binds the position as well as the side. These closures are rebuilt
            # every iteration and used only within it, so a late-bound `pos` happens to
            # be the right one -- but that is an accident of the loop's shape rather than
            # a property of the closure, and moving one line would end it quietly.
            ranks["policy"] = [
                (lambda pool, scored=None, side=s, at=pos: policy(at, side, pool, scored))
                for s in (0, 1)
            ]
        # The reference is the whole pool, ordered however -- the set is what matters.
        whole = [
            narrow(reg, pos, s, limit=EVERYTHING, rank=ranks["damage"][s]).actions
            for s in (0, 1)
        ]
        pools.extend(len(w) for w in whole)
        payoffs, _notes, _exact = batched_payoffs(
            reg, pos, whole[0], whole[1], [evaluate], budget=budget
        )
        matrix = payoffs[0]
        try:
            reference = solve(matrix)
        except EquilibriumError:
            continue
        lookup = [{a.to_choice(): i for i, a in enumerate(whole[s])} for s in (0, 1)]

        for name in names:
            for width in widths:
                # Built by `narrow`, with this ordering, exactly as a search would.
                menus = [
                    narrow(reg, pos, s, limit=width, rank=ranks[name][s]).actions
                    for s in (0, 1)
                ]
                rows = sorted(lookup[0][a.to_choice()] for a in menus[0])
                cols = sorted(lookup[1][a.to_choice()] for a in menus[1])
                try:
                    tight = solve(matrix[np.ix_(rows, cols)])
                except EquilibriumError:
                    continue
                x = np.asarray(tight.row_strategy, dtype=np.float64)
                y = np.asarray(tight.col_strategy, dtype=np.float64)
                costs[name].setdefault(width, []).extend([
                    reference.value - float((x @ matrix[rows, :]).min()),
                    float((matrix[:, cols] @ y).max()) - reference.value,
                ])
        if index % 20 == 0:
            print(f"  {index}/{len(positions)} ({time.perf_counter() - started:.0f}s)",
                  flush=True)

    print(f"\n{len(positions)} positions, pool {np.mean(pools):.1f} legal pairs on average")
    print(f"  {'width':>6} " + " ".join(f"{n:>12}" for n in names))
    for width in widths:
        line = f"  {width:>6}"
        for name in names:
            values = costs[name].get(width)
            line += f" {np.mean(values):>12.4f}" if values else f" {'-':>12}"
        print(line)

    if policy:
        print("\n  the menu a search really gets, learned against leaf:")
        for width in widths:
            leaf = float(np.mean(costs["leaf"][width]))
            learned = float(np.mean(costs["policy"][width]))
            verdict = "learned" if learned < leaf else "leaf"
            print(f"    width {width:>2}: {leaf:.4f} vs {learned:.4f}  -> {verdict} "
                  f"({max(leaf, learned) / max(min(leaf, learned), 1e-9):.2f}x)")


if __name__ == "__main__":
    main()
