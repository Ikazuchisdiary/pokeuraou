"""Does the learned ordering still work when it has to look at everything?

`tools/policy_ceiling.py` measures the policy inside a menu the *leaf* ordering built: the
48 candidates it kept out of several hundred legal pairs. That is how the policy was
trained too -- a recorded decision's `ownActions` is already a narrowed menu, so every row
the model ever saw had survived the leaf's cut.

`narrow` does not work that way. It hands its ranker the whole legal pool and keeps the top
W of whatever comes back. So at search time the model is asked about a few hundred actions
it has never scored, and the board says something changed: at width 16, the same policy
that is 2.4x better than the leaf ordering inside the leaf's menu loses 4.1 points playing
real games.

This measures the gap directly. Ground truth is the 48-wide equilibrium's support -- the
actions that actually get played. Each ordering then ranks the *full legal pool*, and the
question is how much of that support its top W contains.

  in-menu     the policy reorders the leaf's 48, which is what the ceiling measures
  from-pool   the policy ranks every legal pair, which is what a game does

If the two differ, the ceiling was measuring a question the search never asks.

    uv run --group learn python tools/policy_pool_recall.py --positions 60 \\
        --value data/models/value-all.pt --policy data/models/policy-gen7.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from budget_effect import load_positions  # noqa: E402

from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError, solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.policy import load_policy  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs  # noqa: E402
from pokeuraou.search import leaf_ranking  # noqa: E402

SUPPORT_FLOOR = 1e-9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--wide", type=int, default=48)
    ap.add_argument("--widths", default="8,12,16,24")
    ap.add_argument("--games-dir", type=Path, default=ROOT / "data" / "selfplay-gen8")
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--value", type=Path, required=True)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--policy-value", type=Path, default=None)
    ap.add_argument("--policy-device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = ap.parse_args()

    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.value import BatchedValue, load_model

    torch.set_num_threads(1)
    widths = [int(w) for w in args.widths.split(",") if w]
    positions = load_positions(args.games_dir, args.min_turn, args.seed)[: args.positions]
    if not positions:
        raise SystemExit(f"no recorded positions under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)

    encoder = Encoder(reg)
    device = torch.device(args.device)
    net, _meta = load_model(args.value, encoder)
    net = net.to(device)
    evaluate = BatchedValue(net, encoder, device=device)
    policy_net = net
    if args.policy_value is not None:
        loaded, _ = load_model(args.policy_value, encoder)
        policy_net = loaded.to(device).eval()
    policy = load_policy(args.policy, reg, args.policy_device, value_net=policy_net)

    budget = Budget.matrix()
    # Equilibrium weight the top W captures, per ordering. Weight rather than a count of
    # support actions, because the support is 2.4 actions on average and missing the one
    # that carries 0.9 of the weight is not the same as missing the one that carries 0.05.
    got: dict[str, dict[int, list[float]]] = {
        "leaf": {w: [] for w in widths},
        "policy in-menu": {w: [] for w in widths},
        "policy from-pool": {w: [] for w in widths},
    }
    pool_sizes: list[int] = []
    kept = 0

    for index, pos in enumerate(positions, 1):
        ranks = [leaf_ranking(reg, pos, side, evaluate, budget=budget) for side in (0, 1)]
        menus = [narrow(reg, pos, s, limit=args.wide, rank=ranks[s]).actions for s in (0, 1)]
        payoffs, _notes, _exact = batched_payoffs(
            reg, pos, menus[0], menus[1], [evaluate], budget=budget
        )
        try:
            equilibrium = solve(payoffs[0])
        except EquilibriumError:
            continue
        kept += 1
        strategies = [
            np.asarray(equilibrium.row_strategy, dtype=np.float64),
            np.asarray(equilibrium.col_strategy, dtype=np.float64),
        ]
        for side in (0, 1):
            weight = strategies[side]
            total = float(weight.sum()) or 1.0
            menu = menus[side]
            pool = side_actions(reg, pos, side)
            pool_sizes.append(len(pool))
            # Where each menu action sits in an ordering of the whole pool. An action the
            # ordering ranks below W is not in the menu it would have built, so its weight
            # is lost -- which is the quantity a game pays and the ceiling never charges.
            by_choice = {a.to_choice(): i for i, a in enumerate(menu)}
            in_menu = np.argsort(-policy(pos, side, menu))
            from_pool = np.argsort(-policy(pos, side, pool))
            for width in widths:
                # The leaf's own top W, which is the first W of the menu it handed over.
                got["leaf"][width].append(float(weight[:width].sum()) / total)
                got["policy in-menu"][width].append(
                    float(weight[in_menu[:width]].sum()) / total
                )
                chosen = {pool[i].to_choice() for i in from_pool[:width]}
                rows = [by_choice[c] for c in chosen if c in by_choice]
                got["policy from-pool"][width].append(
                    float(weight[rows].sum()) / total if rows else 0.0
                )
        if index % 20 == 0:
            print(f"  {index}/{len(positions)}", flush=True)

    print(
        f"\n{kept} positions, {2 * kept} sides, {args.wide}-wide menus out of "
        f"{np.mean(pool_sizes):.0f} legal pairs on average"
    )
    print(f"\n  {'width':>6}  {'leaf':>8}  {'policy in-menu':>15}  {'policy from-pool':>17}")
    for width in widths:
        print(
            f"  {width:>6}  {np.mean(got['leaf'][width]):>7.1%}  "
            f"{np.mean(got['policy in-menu'][width]):>14.1%}  "
            f"{np.mean(got['policy from-pool'][width]):>16.1%}"
        )


if __name__ == "__main__":
    main()
