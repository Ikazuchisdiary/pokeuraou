"""What narrowing throws away, measured as equilibrium mass rather than guessed at.

A doubles turn has a few hundred legal choices a side and the search solves a 16x16 matrix,
so the candidate set is chosen by `narrow` -- and its ordering is a damage heuristic. Its own
docstring is honest about the risk: "a mediocre ordering costs accuracy in the *choice of
candidates*". This measures that cost, because it decides what to build next.

If a wide solve puts real equilibrium weight on actions the production width excluded, then
the ordering is the binding constraint: a better value function cannot recover an action the
search never considered, and neither can a deeper search. That would make a learned policy
(which can order candidates from the recorded equilibria, with no new self-play) the highest
value work. If the excluded mass is negligible, narrowing is fine and the effort belongs in
the value function or in depth.

Three numbers per position, comparing a wide solve against the production one:

  excluded mass   the share of the wide equilibrium's weight on actions the narrow
                  candidate set does not contain -- the number that matters
  value shift     how far the game's value moves, in the objective's units
  policy TV       total-variation distance on the shared support, i.e. how differently the
                  two would play among the actions both considered

    uv run python tools/narrow_effect.py --positions 8 --wide 48
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from budget_effect import load_positions  # noqa: E402

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError, solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn, turn_expectation  # noqa: E402


def matrix(
    reg: Regulation,
    pos: Position,
    ours: list,
    theirs: list,
    budget: Budget,
    objective,  # noqa: ANN001
) -> np.ndarray:
    payoff = np.zeros((len(ours), len(theirs)), dtype=np.float64)
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            value, _flags = turn_expectation(
                reg, resolve_turn(reg, pos, [a, b], budget=budget), objective
            )
            payoff[i, j] = value
    return payoff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=8)
    ap.add_argument("--narrow", type=int, default=16, help="the production width")
    ap.add_argument("--wide", type=int, default=48, help="the width to compare against")
    ap.add_argument("--games-dir", type=Path, default=Path("data/selfplay-gen2"))
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--objective", default="hp-share", choices=sorted(OBJECTIVES))
    args = ap.parse_args()

    positions = load_positions(args.games_dir, args.min_turn, args.seed)
    if not positions:
        raise SystemExit(f"no recorded positions under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)
    objective = OBJECTIVES[args.objective]
    budget = Budget.matrix()

    print(
        f"narrow {args.narrow} against wide {args.wide}, {args.positions} recorded "
        f"positions, objective {args.objective}\n"
    )
    print(
        f"  {'#':>3} {'legal':>7} {'excluded mass':>14} {'value shift':>12} "
        f"{'policy TV':>10} {'seconds':>8}"
    )

    excluded: list[float] = []
    shifts: list[float] = []
    tvs: list[float] = []
    taken = 0
    done = 0
    while done < args.positions and taken < len(positions):
        pos = positions[taken]
        taken += 1
        wide = narrow(reg, pos, 0, limit=args.wide)
        tight = narrow(reg, pos, 0, limit=args.narrow)
        foes = narrow(reg, pos, 1, limit=args.narrow).actions
        if len(wide.actions) <= len(tight.actions) or not foes:
            continue

        started = time.perf_counter()
        try:
            wide_eq = solve(matrix(reg, pos, wide.actions, foes, budget, objective))
            tight_eq = solve(matrix(reg, pos, tight.actions, foes, budget, objective))
        except EquilibriumError:
            continue
        seconds = time.perf_counter() - started
        done += 1

        kept = {a.to_choice() for a in tight.actions}
        # The mass the wide equilibrium puts where the production width cannot look.
        outside = sum(
            float(w)
            for a, w in zip(wide.actions, wide_eq.row_strategy, strict=True)
            if a.to_choice() not in kept
        )
        # And how differently they play among the actions both of them considered.
        wide_on_kept = {
            a.to_choice(): float(w)
            for a, w in zip(wide.actions, wide_eq.row_strategy, strict=True)
            if a.to_choice() in kept
        }
        tight_on_kept = {
            a.to_choice(): float(w)
            for a, w in zip(tight.actions, tight_eq.row_strategy, strict=True)
        }
        shared = set(wide_on_kept) | set(tight_on_kept)
        tv = 0.5 * sum(
            abs(wide_on_kept.get(k, 0.0) - tight_on_kept.get(k, 0.0)) for k in shared
        )
        shift = abs(wide_eq.value - tight_eq.value)

        excluded.append(outside)
        shifts.append(shift)
        tvs.append(tv)
        print(
            f"  {done:>3} {len(wide.actions):>7} {outside * 100:>13.1f}% "
            f"{shift:>12.4f} {tv:>10.3f} {seconds:>8.1f}",
            flush=True,
        )

    if not excluded:
        raise SystemExit("no position had more candidates at the wide width")
    print(
        f"\nmean excluded mass {np.mean(excluded) * 100:.1f}% "
        f"(worst {np.max(excluded) * 100:.1f}%), "
        f"mean value shift {np.mean(shifts):.4f} (worst {np.max(shifts):.4f}), "
        f"mean policy TV {np.mean(tvs):.3f}"
    )
    print(
        "\nexcluded mass is the number that decides this: it is equilibrium weight the\n"
        "production search cannot place, and no value function or extra depth recovers it."
    )


if __name__ == "__main__":
    main()
