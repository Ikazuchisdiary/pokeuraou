"""How much room is left in the damage roll -- what IKA-11 would be buying.

The search's budget pins roll 8 (`Budget.matrix()` has `damage_rolls = -9`), so every
attack is priced at its median damage and a knock-out is either certain or impossible.
That collapse is the one interval stratification would replace, and it is a different
thing from anything the lossless merge touched: the merge folds branches that were the
same state, this one never made the branches at all.

Isolated so the number means one thing: accuracy, status checks and secondaries are off in
*both* arms, so the only difference between them is the roll.

    pinned    roll 8, cap 16              what the search sees today
    rolls16   all 16 rolls, cap lifted    the distribution the calculator already returns

Reported per position rather than only as a mean, because the effect is not spread evenly
-- most positions are untouched and the tail is where the decision moves. The two columns
to read are `loss` (what the arm's own mixture is worth in the sixteen-roll game, against
what that game guarantees) and whether the top move changed.

Narrow on purpose: sixteen rolls across four damaging actions is 65,536 states before
anything folds, and one position in twelve took seven minutes at width 6. This is for
saying whether the effect is 0.001 or 0.05, not for a board-level claim.

    uv run python tools/roll_headroom.py --positions 12 --limit 6
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

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError, solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import Regulation, load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn, turn_expectation  # noqa: E402


def load_positions(games_dir: Path, min_turn: int, seed: int) -> list[Position]:
    out: list[Position] = []
    for path in sorted(games_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for decision in record.get("decisions", ()):
                    if decision.get("kind") == "move" and int(decision.get("turn", 0)) >= min_turn:
                        out.append(Position.from_json(decision["position"]))
                if len(out) > 400:
                    break
        break
    random.Random(seed).shuffle(out)
    return out


def matrix_for(
    reg: Regulation, pos: Position, ours: list, theirs: list, budget: Budget, objective
) -> tuple[np.ndarray, float, int]:  # noqa: ANN001
    out = np.zeros((len(ours), len(theirs)))
    leaves = 0
    started = time.perf_counter()
    for i, a in enumerate(ours):
        for j, b in enumerate(theirs):
            result = resolve_turn(reg, pos, [a, b], budget=budget)
            out[i, j], _flags = turn_expectation(reg, result, objective)
            leaves += len(result.branches) + len(result.suspended)
    return out, time.perf_counter() - started, leaves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--positions", type=int, default=12)
    ap.add_argument("--limit", type=int, default=6, help="candidate actions per side")
    ap.add_argument("--games-dir", type=Path, default=Path("data/selfplay-gen11L"))
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--objective", default="hp-share", choices=sorted(OBJECTIVES))
    args = ap.parse_args()

    positions = load_positions(args.games_dir, args.min_turn, args.seed)
    if not positions:
        raise SystemExit(f"no recorded positions found under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)
    objective = OBJECTIVES[args.objective]

    pinned = replace(
        Budget.matrix(),
        enumerate_accuracy=False,
        enumerate_status_checks=False,
        enumerate_secondary=False,
    )
    rolls16 = replace(pinned, damage_rolls=16, max_branches=200_000)

    print(f"  {args.limit}x{args.limit}, rolls isolated (accuracy/status/secondary off "
          f"in both arms), objective {args.objective}\n")
    print(f"  {'#':>2}  {'cell max':>9}  {'cell mean':>9}  {'TV':>6}  {'loss':>8}  "
          f"{'top':>4}  {'leaves/cell':>11}  {'ref s':>7}")
    rows: list[tuple[float, float, float, float, bool]] = []
    built = taken = 0
    while built < args.positions and taken < len(positions):
        pos = positions[taken]
        taken += 1
        ours = narrow(reg, pos, 0, limit=args.limit).actions
        theirs = narrow(reg, pos, 1, limit=args.limit).actions
        if len(ours) < 2 or len(theirs) < 2:
            continue
        cells = len(ours) * len(theirs)
        try:
            reference_matrix, ref_seconds, ref_leaves = matrix_for(
                reg, pos, ours, theirs, rolls16, objective
            )
            arm_matrix, _seconds, _leaves = matrix_for(
                reg, pos, ours, theirs, pinned, objective
            )
            reference = solve(reference_matrix)
            arm = solve(arm_matrix)
        except EquilibriumError:
            continue
        built += 1
        loss = reference.value - float(np.min(arm.row_strategy @ reference_matrix))
        tv = 0.5 * float(np.abs(arm.row_strategy - reference.row_strategy).sum())
        top = int(np.argmax(arm.row_strategy)) != int(np.argmax(reference.row_strategy))
        gap = np.abs(arm_matrix - reference_matrix)
        rows.append((float(gap.max()), float(gap.mean()), tv, loss, top))
        print(
            f"  {built:>2}  {rows[-1][0]:>9.4f}  {rows[-1][1]:>9.4f}  {tv:>6.3f}  "
            f"{loss:>8.5f}  {'yes' if top else 'no':>4}  {ref_leaves / cells:>11.1f}  "
            f"{ref_seconds:>7.1f}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no position produced a solvable pair of matrices")
    print(
        f"\n  mean over {len(rows)}: cell max {np.mean([r[0] for r in rows]):.4f}, "
        f"cell mean {np.mean([r[1] for r in rows]):.4f}, "
        f"TV {np.mean([r[2] for r in rows]):.3f}, "
        f"loss {np.mean([r[3] for r in rows]):.5f}, "
        f"top differs {np.mean([r[4] for r in rows]):.0%}"
    )
    print(
        "\n  the mean is the wrong summary on its own: most positions are untouched and "
        "the\n  effect lives in the tail, so read the per-position rows."
    )


if __name__ == "__main__":
    main()
