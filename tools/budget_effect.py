"""What each collapse in the search budget costs, and what it buys.

`Budget.matrix()` is the search's budget, and it collapses four independent sources of
chance. Written out, the model the search actually reasons in is:

  * every move hits            (`enumerate_accuracy=False`)   -- Hurricane at 70% priced at 100%
  * nothing crits              (`enumerate_crit=False`)
  * no sub-100% secondary      (`enumerate_secondary=False`)  -- Heat Wave's burn never happens
  * no full paralysis, no confusion self-hit, Protect always works
                               (`enumerate_status_checks=False`)
  * a multi-hit move hits the minimum number of times

Each is reported as an unmodelled effect, so none of it is hidden -- but "declared" is not
"free", and which of them is worth paying for is a measurement rather than a judgement. What
matters is not the cost in milliseconds alone but whether the *equilibrium* moves: a collapse
that changes the payoff matrix uniformly changes no decision.

So for each knob this measures, against the same positions:

  value      how far the game's value moves, in win-probability points
  policy     total-variation distance between the two equilibrium mixtures, which is the
             share of the time the two strategies would play differently
  cost       ms per cell, and the branch count, because that is what has to be afforded

    uv run python tools/budget_effect.py --positions 12
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError, solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.regulation import Regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn, turn_expectation  # noqa: E402
from pokeuraou.selfplay import position_from_sets  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402

#: The knobs, each turned on alone so the effects do not mask one another.
KNOBS: tuple[tuple[str, dict[str, object]], ...] = (
    ("accuracy", {"enumerate_accuracy": True}),
    ("crit", {"enumerate_crit": True}),
    ("secondary", {"enumerate_secondary": True}),
    ("status checks", {"enumerate_status_checks": True}),
    ("all four", {
        "enumerate_accuracy": True,
        "enumerate_crit": True,
        "enumerate_secondary": True,
        "enumerate_status_checks": True,
    }),
)


def matrix_for(
    reg: Regulation,
    pos: Position,
    ours: list,
    theirs: list,
    budget: Budget,
    objective,  # noqa: ANN001
) -> tuple[np.ndarray, float, int]:
    """The payoff matrix under one budget, with the time it took and the leaves it used."""
    payoff = np.zeros((len(ours), len(theirs)), dtype=np.float64)
    leaves = 0
    started = time.perf_counter()
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            result = resolve_turn(reg, pos, [a, b], budget=budget)
            value, _flags = turn_expectation(reg, result, objective)
            payoff[i, j] = value
            leaves += len(result.branches) + len(result.suspended)
    return payoff, time.perf_counter() - started, leaves


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    """Share of the time two mixtures would play differently."""
    return 0.5 * float(np.abs(a - b).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument("--limit", type=int, default=8, help="candidate actions per side")
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--objective", default="hp-share", choices=sorted(OBJECTIVES))
    ap.add_argument(
        "--branch-cap",
        type=int,
        default=256,
        help="max_branches for the richer budgets. The default matrix budget's 16 would "
        "truncate the very branches being measured, which would report the collapse as "
        "free.",
    )
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    objective = OBJECTIVES[args.objective]
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))
    rng = np.random.default_rng(args.seed)

    base = Budget.matrix()
    print(f"{args.limit}x{args.limit} matrices on {args.positions} positions, "
          f"objective {args.objective}\n")

    stats: dict[str, list[tuple[float, float, float, int]]] = {name: [] for name, _ in KNOBS}
    base_cost: list[tuple[float, int]] = []

    built = 0
    while built < args.positions:
        team = pool[int(rng.integers(len(pool)))]
        foe_six = sample_standings_team(rng, reg, prior, team)
        own_pick = selections[int(rng.integers(len(selections)))]
        foe_pick = selections[int(rng.integers(len(selections)))]
        pos = position_from_sets(
            reg, [roster.sets[i] for i in own_pick], [foe_six[j] for j in foe_pick]
        )
        ours = narrow(reg, pos, 0, limit=args.limit).actions
        theirs = narrow(reg, pos, 1, limit=args.limit).actions
        if not ours or not theirs:
            continue

        payoff, seconds, leaves = matrix_for(reg, pos, ours, theirs, base, objective)
        try:
            reference = solve(payoff)
        except EquilibriumError:
            continue
        base_cost.append((seconds, leaves))
        built += 1

        for name, overrides in KNOBS:
            richer = replace(base, max_branches=args.branch_cap, **overrides)  # type: ignore[arg-type]
            other, other_seconds, other_leaves = matrix_for(
                reg, pos, ours, theirs, richer, objective
            )
            try:
                equilibrium = solve(other)
            except EquilibriumError:
                continue
            stats[name].append(
                (
                    abs(equilibrium.value - reference.value),
                    total_variation(equilibrium.row_strategy, reference.row_strategy),
                    other_seconds / max(seconds, 1e-9),
                    other_leaves,
                )
            )
        print(f"  position {built}/{args.positions} done", flush=True)

    cells = args.limit * args.limit
    mean_seconds = float(np.mean([s for s, _ in base_cost]))
    mean_leaves = float(np.mean([n for _, n in base_cost]))
    print(
        f"\nbaseline Budget.matrix(): {mean_seconds / cells * 1e3:.2f} ms/cell, "
        f"{mean_leaves / cells:.1f} leaves/cell"
    )
    print(f"\n{'knob':>14}  {'value shift':>12}  {'policy TV':>10}  {'cost':>7}  {'leaves/cell':>11}")
    for name, _ in KNOBS:
        rows = stats[name]
        if not rows:
            print(f"{name:>14}  {'no data':>12}")
            continue
        value = float(np.mean([r[0] for r in rows]))
        worst = float(np.max([r[0] for r in rows]))
        tv = float(np.mean([r[1] for r in rows]))
        cost = float(np.mean([r[2] for r in rows]))
        leaves = float(np.mean([r[3] for r in rows])) / cells
        print(
            f"{name:>14}  {value:>11.4f}  {tv:>10.3f}  {cost:>6.2f}x  {leaves:>11.1f}"
            f"   (worst value shift {worst:.4f})"
        )
    print(
        "\nvalue shift is in the objective's units; policy TV is the share of the time the "
        "\ntwo equilibria would choose differently, which is what actually costs games."
    )


if __name__ == "__main__":
    main()
