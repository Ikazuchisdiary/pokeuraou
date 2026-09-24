"""Does the branch merge change the move, and does it change it towards the right one?

The merge is lossless, so on a turn the budget resolves in full it cannot move anything.
What it does move is the turns where the cap binds: fewer live branches means fewer are
dropped and each action gets more of the resolution budget, so the *cell* is a different
number and the equilibrium over those cells can be a different mixture.

"Different" is the easy half. The question worth answering is whether the difference goes
towards the truth, so three matrices are built on the same position with the same action
lists:

    off    Budget.matrix() with the merge off -- what the search used to see
    on     Budget.matrix() as it now ships
    ref    the same budget with the cap lifted, so nothing is dropped and nothing is
           narrowed. The merge is lossless, so this is the same game either way; it is
           what both arms are trying to approximate

and each arm is scored against `ref` two ways:

  policy TV     how much of the mixture moved. Sensitive to which vertex the LP picked
                when a game has more than one equilibrium, so it overstates disagreement
  guarantee     what the arm's own strategy is actually worth in the true game:
                `min_j (x @ ref)_j` against `ref`'s value. This is the decision-relevant
                number and it does not care which vertex was picked -- two strategies from
                the same equilibrium set score the same

Positions come from recorded games, mid-game, for the reason `budget_effect.py` gives: a
turn-1 sample has nothing statused, boosted or half-dead, so a collapse looks free because
it had no opportunity.

    uv run python tools/merge_effect.py --positions 12 --limit 12
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import port  # noqa: E402 - Python's resolver until IKA-212
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError, solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402


def matrix_for(
    reg: Regulation,
    pos: Position,
    ours: list,
    theirs: list,
    budget: Budget,
    objective,  # noqa: ANN001
) -> tuple[np.ndarray, float, int, int]:
    """The payoff matrix under one budget, with its time, leaves and exact-cell count."""
    payoff = np.zeros((len(ours), len(theirs)), dtype=np.float64)
    leaves = 0
    exact_cells = 0
    started = time.perf_counter()
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            result = port.turn(reg, pos, [a, b], budget, full=True)
            value, _flags = port.turn_expectation(reg, result, objective)
            payoff[i, j] = value
            leaves += len(result.branches) + len(result.suspended)
            exact_cells += int(result.exact)
    return payoff, time.perf_counter() - started, leaves, exact_cells


def load_positions(games_dir: Path, min_turn: int, seed: int, wanted: int) -> list[Position]:
    """Mid-game positions from recorded games, shuffled."""
    out: list[Position] = []
    for path in sorted(games_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for decision in record.get("decisions", ()):
                    if decision.get("kind") != "move":
                        continue
                    if int(decision.get("turn", 0)) < min_turn:
                        continue
                    out.append(Position.from_json(decision["position"]))
                if len(out) > 40 * wanted:
                    break
        if len(out) > 40 * wanted:
            break
    random.Random(seed).shuffle(out)
    return out


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def guarantee(matrix: np.ndarray, row_strategy: np.ndarray) -> float:
    """What this strategy is worth to the row player in this game, against best reply."""
    return float(np.min(row_strategy @ matrix))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--limit", type=int, default=12, help="candidate actions per side")
    ap.add_argument("--games-dir", type=Path, default=Path("data/selfplay-gen11L"))
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--objective", default="hp-share", choices=sorted(OBJECTIVES))
    ap.add_argument(
        "--branch-cap",
        type=int,
        default=4096,
        help="max_branches for the reference. It has to be past what the turn can reach, "
        "or the reference is capped too and both arms are compared against a third "
        "approximation.",
    )
    ap.add_argument(
        "--reference-rolls",
        action="store_true",
        help="make the reference enumerate all sixteen damage rolls rather than the "
        "single pinned one the matrix budget uses. This changes the question: with the "
        "default reference the only difference between it and the arms is the branch cap, "
        "which is what the merge acts on; with this one the reference also prices the "
        "damage distribution, and the gap includes the median-roll collapse -- which is "
        "what interval stratification (IKA-11) would be buying, not the merge.",
    )
    args = ap.parse_args()

    positions = load_positions(args.games_dir, args.min_turn, args.seed, args.positions)
    if not positions:
        raise SystemExit(f"no recorded positions found under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)
    objective = OBJECTIVES[args.objective]

    arms = {
        "off": replace(Budget.matrix(), merge_duplicates=False),
        "on": Budget.matrix(),
    }
    reference_budget = replace(Budget.matrix(), max_branches=args.branch_cap)
    if args.reference_rolls:
        reference_budget = replace(reference_budget, damage_rolls=16)

    print(
        f"  {args.limit}x{args.limit} matrices on {args.positions} positions from "
        f"{args.games_dir}, objective {args.objective}"
    )
    print(
        f"  reference: Budget.matrix() with max_branches={args.branch_cap}"
        + (", all 16 damage rolls" if args.reference_rolls else ", the pinned roll 8")
        + "\n"
    )

    rows: dict[str, list[dict]] = {name: [] for name in arms}
    direct: list[tuple[float, float, bool]] = []
    reference_exact: list[float] = []
    identical_reference = 0
    built = taken = 0
    while built < args.positions and taken < len(positions):
        pos = positions[taken]
        taken += 1
        ours = narrow(reg, pos, 0, limit=args.limit).actions
        theirs = narrow(reg, pos, 1, limit=args.limit).actions
        if len(ours) < 2 or len(theirs) < 2:
            continue
        cells = len(ours) * len(theirs)

        ref_matrix, ref_seconds, _ref_leaves, ref_exact = matrix_for(
            reg, pos, ours, theirs, reference_budget, objective
        )
        try:
            reference = solve(ref_matrix)
        except EquilibriumError:
            continue
        reference_exact.append(ref_exact / cells)
        # The reference has to be the same game whichever way it is built, or "closer to
        # the reference" means nothing. Checked rather than asserted from the theory.
        unmerged_reference, _s, _l, _e = matrix_for(
            reg, pos, ours, theirs,
            replace(reference_budget, merge_duplicates=False), objective,
        )
        if np.allclose(ref_matrix, unmerged_reference, atol=1e-12, rtol=0.0):
            identical_reference += 1

        built += 1
        solved: dict[str, object] = {}
        for name, budget in arms.items():
            matrix, seconds, leaves, exact_cells = matrix_for(
                reg, pos, ours, theirs, budget, objective
            )
            try:
                equilibrium = solve(matrix)
            except EquilibriumError:
                continue
            solved[name] = equilibrium
            rows[name].append(
                {
                    "cell": float(np.abs(matrix - ref_matrix).max()),
                    "value": abs(equilibrium.value - reference.value),
                    "tv": total_variation(equilibrium.row_strategy, reference.row_strategy),
                    "loss": reference.value - guarantee(ref_matrix, equilibrium.row_strategy),
                    "top": int(np.argmax(equilibrium.row_strategy))
                    != int(np.argmax(reference.row_strategy)),
                    "exact": exact_cells / cells,
                    "leaves": leaves / cells,
                    "ms": seconds / cells * 1e3,
                }
            )
        if len(solved) == 2:
            off, on = solved["off"], solved["on"]
            direct.append(
                (
                    total_variation(off.row_strategy, on.row_strategy),  # type: ignore[attr-defined]
                    abs(off.value - on.value),  # type: ignore[attr-defined]
                    int(np.argmax(off.row_strategy))  # type: ignore[attr-defined]
                    != int(np.argmax(on.row_strategy)),  # type: ignore[attr-defined]
                )
            )
        print(
            f"  position {built}/{args.positions} "
            f"(reference {ref_seconds / cells * 1e3:.1f} ms/cell)",
            flush=True,
        )

    print(
        f"\n  reference cells resolved exactly: {np.mean(reference_exact):.1%}  "
        f"(and merged == unmerged on {identical_reference}/{built} of them)"
    )
    print(
        f"\n  {'arm':>5}  {'exact cells':>11}  {'leaves/cell':>11}  {'ms/cell':>8}  "
        f"{'worst cell':>10}  {'policy TV':>9}  {'value':>7}  {'loss vs ref':>11}  {'top move':>9}"
    )
    for name in arms:
        data = rows[name]
        if not data:
            print(f"  {name:>5}  no data")
            continue
        print(
            f"  {name:>5}  {np.mean([r['exact'] for r in data]):>10.1%}  "
            f"{np.mean([r['leaves'] for r in data]):>11.1f}  "
            f"{np.mean([r['ms'] for r in data]):>8.2f}  "
            f"{np.mean([r['cell'] for r in data]):>10.4f}  "
            f"{np.mean([r['tv'] for r in data]):>9.3f}  "
            f"{np.mean([r['value'] for r in data]):>7.4f}  "
            f"{np.mean([r['loss'] for r in data]):>11.5f}  "
            f"{np.mean([r['top'] for r in data]):>8.0%}"
        )
    if direct:
        tv = float(np.mean([d[0] for d in direct]))
        value = float(np.mean([d[1] for d in direct]))
        top = float(np.mean([d[2] for d in direct]))
        print(
            f"\n  off against on directly: policy TV {tv:.3f}, value {value:.4f}, "
            f"top move differs on {top:.0%} of positions"
        )
    print(
        "\n  'loss vs ref' is the decision-relevant one: the arm's own mixture, scored in "
        "the\n  reference game against a best reply, against what the reference itself "
        "guarantees.\n  policy TV is an upper bound on disagreement -- a game with more "
        "than one equilibrium\n  can report a moved mixture that plays exactly as well."
    )


if __name__ == "__main__":
    main()
