"""How narrow could the menu be if the ordering were perfect?

`tools/width_vs_ranking.py` prices two orderings that exist. This prices the one that does
not: rank the candidates by the width-48 equilibrium's own weight -- which no ranking can
know in advance, since knowing it is solving the game -- and ask what restricting to the
top W then costs. That is the ceiling on what any ordering could ever buy, learned or not.

It matters because the two levers are different work. Narrowing the width is a setting;
learning a policy to order candidates is a model, a training run and an integration. The
second is only worth starting if the ceiling is far above where the leaf ranking sits, and
only worth finishing if it fixes the tail -- the leaf ordering's worst case does not move
between width 8 and width 32, so on that position widening is pure loss, and the question
is whether a *better* ordering would have helped or whether nothing would.

Three numbers per position:

  support        how many actions the 48-wide equilibrium actually plays. A perfect
                 ordering needs no more width than this, so it is the floor on the width
                 and the cheapest possible statement of the whole argument.
  oracle cost    restricting to the equilibrium's own top W, judged against the full
                 48-wide menu -- the ceiling.
  leaf cost      the same for the leaf ordering, so the gap between what we have and what
                 is available is visible in one line.

    uv run --group learn python tools/policy_ceiling.py --positions 100 \\
        --value data/models/value-all.pt
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

#: Weight below which an action is not being played. The LP returns exact zeros outside
#: the support, so this only guards against a solver's dust.
SUPPORT_FLOOR = 1e-9


def measure(
    reg, pos, evaluate, budget, wide: int, widths: list[int], policy=None
) -> dict | None:
    """One position: the support of the wide equilibrium, and what each top-W costs."""
    ranks = [leaf_ranking(reg, pos, side, evaluate, budget=budget) for side in (0, 1)]
    menus = [narrow(reg, pos, s, limit=wide, rank=ranks[s]).actions for s in (0, 1)]
    payoffs, _notes, _exact = batched_payoffs(
        reg, pos, menus[0], menus[1], [evaluate], budget=budget
    )
    matrix = payoffs[0]
    try:
        equilibrium = solve(matrix)
    except EquilibriumError:
        return None

    strategies = [
        np.asarray(equilibrium.row_strategy, dtype=np.float64),
        np.asarray(equilibrium.col_strategy, dtype=np.float64),
    ]
    out: dict = {
        "support": [int((s > SUPPORT_FLOOR).sum()) for s in strategies],
        "rows": len(menus[0]),
        "cols": len(menus[1]),
        "oracle": {},
        "leaf": {},
        "policy": {},
    }
    # The learned ordering scores the same menu, so it is compared cell for cell against
    # the other two rather than against a menu of its own.
    learned = (
        [policy(pos, side, menus[side]) for side in (0, 1)] if policy is not None else None
    )
    for width in widths:
        # The oracle keeps the top W by equilibrium weight; the leaf ordering keeps the
        # first W of the menu, which `narrow` already handed over best-first.
        orderings = [
            ("oracle", [np.argsort(-s)[:width] for s in strategies]),
            ("leaf", [np.arange(min(width, len(s))) for s in strategies]),
        ]
        if learned is not None:
            orderings.append(("policy", [np.argsort(-s)[:width] for s in learned]))
        for name, keep in orderings:
            rows, cols = sorted(keep[0]), sorted(keep[1])
            try:
                tight = solve(matrix[np.ix_(rows, cols)])
            except EquilibriumError:
                continue
            x = np.asarray(tight.row_strategy, dtype=np.float64)
            y = np.asarray(tight.col_strategy, dtype=np.float64)
            # Restricted side plays its own equilibrium; the other answers over everything.
            row_cost = equilibrium.value - float((x @ matrix[rows, :]).min())
            col_cost = float((matrix[:, cols] @ y).max()) - equilibrium.value
            out[name].setdefault(width, []).extend([row_cost, col_cost])
    return out


def _load_policy(path, reg, device_name: str = "cpu"):
    """The learned ordering, as a function from (position, side, actions) to scores.

    On the CPU, whatever the value function is using. The model is small and a menu is
    about thirty-five rows, which on CUDA is 2.25 ms of pure launch latency -- flat from 34
    rows to 138 -- against 0.14 ms on the CPU. Sixteen times, for the same arithmetic, and
    it is the same fact as everything else in `rust/README.md`: a small batch never reaches
    the card's arithmetic at all.
    """
def _load_policy(path, reg, device_name: str):
    """The learned ordering, as a function from (position, side, actions) to scores."""
    import torch

    sys.path.insert(0, str(ROOT / "tools"))
    from policy_dataset import features_for, ids_for

    from pokeuraou.encode import Encoder
    from pokeuraou.narrow import _bridged_scores, score_action

    blob = torch.load(path, map_location="cpu", weights_only=False)
    encoder = Encoder(reg)
    device = torch.device(device_name)

    from torch import nn

    embed = 24
    species = nn.Embedding(int(blob["species_vocab"]), embed, padding_idx=0)
    moves = nn.Embedding(int(blob["move_vocab"]), embed, padding_idx=0)
    width = int(blob["hidden"])
    trunk = nn.Sequential(
        nn.Linear(int(blob["features"]) + 2 * 4 * embed, width), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(width, width), nn.ReLU(), nn.Dropout(0.0),
        nn.Linear(width, 1),
    )
    state = blob["state"]
    species.load_state_dict({"weight": state["species.weight"]})
    moves.load_state_dict({"weight": state["moves.weight"]})
    trunk.load_state_dict(
        {k[len("trunk.") :]: v for k, v in state.items() if k.startswith("trunk.")}
    )
    for module in (species, moves, trunk):
        module.to(device).eval()

    @torch.no_grad()
    def rank(pos, side: int, actions, scored=None):
        # `narrow` computes the damage candidates before it calls a ranker and now hands
        # them over, so there is no second crossing to the port for them.
        if scored is None:
            from pokeuraou.narrow import _bridged_scores, score_action

            scored = _bridged_scores(reg, pos, side, list(actions))
            if scored is None:
                scored = [score_action(reg, pos, side, a, battlers=None) for a in actions]
    def rank(pos, side: int, actions):
        scored = _bridged_scores(reg, pos, side, list(actions))
        if scored is None:
            scored = [score_action(reg, pos, side, a, battlers=None) for a in actions]
        by_choice = {c.action.to_choice(): c.score for c in scored}
        scores = [by_choice.get(a.to_choice(), 0.0) for a in actions]
        x = torch.from_numpy(
            features_for(reg, pos, side, list(actions), scores)
        ).to(device)
        k = torch.from_numpy(
            ids_for(encoder, pos, side, list(actions)).astype("int64")
        ).to(device)
        flat = torch.cat(
            [x, species(k[:, [0, 2, 3, 4, 6, 7]]).flatten(-2),
             moves(k[:, [1, 5]]).flatten(-2)],
            dim=-1,
        )
        return trunk(flat).squeeze(-1).cpu().numpy()

    return rank


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=100)
    ap.add_argument("--wide", type=int, default=48)
    ap.add_argument("--widths", default="4,8,12,16,24")
    ap.add_argument("--games-dir", type=Path, default=ROOT / "data" / "selfplay-gen7")
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--value", type=Path, default=None)
    ap.add_argument("--policy", type=Path, default=None,
                    help="a model from tools/policy_train.py, ranked alongside the others")
    ap.add_argument("--policy-device", default="cpu", choices=("cpu", "cuda"),
                    help="the CPU is 16x quicker on a menu-sized batch")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = ap.parse_args()

    widths = [int(w) for w in args.widths.split(",") if w]
    positions = load_positions(args.games_dir, args.min_turn, args.seed)[: args.positions]
    if not positions:
        raise SystemExit(f"no recorded positions under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)

    if args.value is None:
        objective = OBJECTIVES["hp-share"]

        def evaluate(items):
            return np.array([objective.batch([p])[0] for p in items], dtype=np.float64)
    else:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        torch.set_num_threads(1)
        encoder = Encoder(reg)
        net, _meta = load_model(args.value, encoder)
        evaluate = BatchedValue(net.to(torch.device(args.device)), encoder,
                                device=torch.device(args.device))

    policy = (
        _load_policy(args.policy, reg, args.policy_device) if args.policy else None
    )
    policy = _load_policy(args.policy, reg, args.device) if args.policy else None

    budget = Budget.matrix()
    print(f"{len(positions)} recorded positions, {args.wide}-wide menus, "
          f"leaf {args.value.stem if args.value else 'hp-share'}"
          f"{', policy ' + args.policy.stem if args.policy else ''}")

    support: list[int] = []
    costs: dict[str, dict[int, list[float]]] = {"oracle": {}, "leaf": {}, "policy": {}}
    started = time.perf_counter()
    for index, pos in enumerate(positions, 1):
        got = measure(reg, pos, evaluate, budget, args.wide, widths, policy)
        if got is None:
            continue
        support.extend(got["support"])
        for name in ("oracle", "leaf", "policy"):
            for width, values in got[name].items():
                costs[name].setdefault(width, []).extend(values)
        if index % 20 == 0:
            print(f"  {index}/{len(positions)} ({time.perf_counter() - started:.0f}s)",
                  flush=True)

    support_array = np.array(support)
    print(f"\nthe {args.wide}-wide equilibrium plays "
          f"{support_array.mean():.1f} actions on average "
          f"(median {int(np.median(support_array))}, "
          f"p90 {int(np.percentile(support_array, 90))}, max {support_array.max()})")
    print(f"  never more than {int(np.percentile(support_array, 99))} in 99% of sides, "
          f"out of {args.wide}")

    names = ["oracle", "leaf"] + (["policy"] if policy is not None else [])
    labels = {"oracle": "perfect", "leaf": "leaf", "policy": "learned"}
    header = f"{'width':>6}"
    for name in names:
        header += f" {labels[name] + ' mean':>13} {'p90':>8}"
    print("\n" + header)
    for width in widths:
        line = f"{width:>6}"
        for name in names:
            values = np.array(costs[name].get(width, [0.0]))
            line += f" {values.mean():>13.4f} {np.percentile(values, 90):>8.4f}"
        print(line)


if __name__ == "__main__":
    main()
