"""How much self-play the current resolver can actually afford.

The README used to say the resolver needed "about 100x" before milestone 3's value
function could be trained. That was an estimate written before anything downstream
existed, and estimates like it are exactly what this project is supposed to measure
instead. So this computes the budget from measurements: the cost of one resolved turn, the
cost of one solved position at several search sizes, and what that implies for games per
hour.

The point it tends to make is that the lever is not microseconds per turn. A solved
position costs (rows x columns x belief classes) resolves, so halving the search size
divides the cost by four -- far more than any plausible rewrite of the inner loop. What
the rewrite would buy is depth at a fixed budget, which is a different and later question.

    uv run python tools/selfplay_budget.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.cli import _modal, build_beliefs
from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve_bayesian
from pokeuraou.narrow import narrow
from pokeuraou.resolve import Budget, resolve_turn
from pokeuraou.setup import load_scenario, with_spreads

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn1.json"

#: Turns in a Champions doubles game, for turning per-position costs into per-game ones.
TURNS_PER_GAME = 12


def timed(fn, repeat: int) -> float:  # noqa: ANN001
    fn()
    started = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - started) / repeat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=50_000, help="学習に使う自己対戦数")
    ap.add_argument("--cores", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args()

    scenario = load_scenario(EXAMPLE)
    reg = scenario.reg
    register_mega_stones(reg)
    beliefs = build_beliefs(scenario)
    base = with_spreads(scenario, {key: _modal(b) for key, b in beliefs.items()})

    budget = Budget.matrix()
    ours_all = narrow(reg, base, 0, limit=24).actions
    theirs_all = narrow(reg, base, 1, limit=24).actions

    per_turn = timed(
        lambda: resolve_turn(reg, base, [ours_all[0], theirs_all[0]], budget=budget),
        repeat=300,
    )
    print(f"resolve_turn（Budget.matrix()）: {per_turn * 1e6:.0f} us\n")

    print("1 局面を解くコスト（行列を埋める + ベイズ LP）")
    header = f"  {'片側手数':>8}  {'同値類':>6}  {'セル数':>7}  {'行列':>9}  {'LP':>8}  {'合計':>9}"
    print(header)

    rows = []
    for limit in (6, 8, 12, 24):
        ours = ours_all[:limit]
        theirs = theirs_all[:limit]
        for classes in (1, 4):
            cells = len(ours) * len(theirs)
            matrix_cost = cells * classes * per_turn
            matrices = [
                np.random.default_rng(0).random((len(ours), len(theirs)))
                for _ in range(classes)
            ]
            weights = np.full(classes, 1.0 / classes)
            lp = timed(lambda m=matrices, w=weights: solve_bayesian(m, w), repeat=20)
            total = matrix_cost + lp
            rows.append((limit, classes, cells, matrix_cost, lp, total))
            print(
                f"  {limit:>8}  {classes:>6}  {cells:>7}  "
                f"{matrix_cost:>8.3f}s  {lp * 1000:>6.1f}ms  {total:>8.3f}s"
            )

    print()
    print(f"自己対戦の予算（1 ゲーム {TURNS_PER_GAME} ターン、{args.games:,} ゲーム）")
    print(
        f"  {'設定':>12}  {'1ゲーム':>9}  {'1コア':>11}  "
        f"{args.cores} コア"
    )
    for limit, classes, _cells, _matrix, _lp, total in rows:
        game = total * TURNS_PER_GAME
        single = game * args.games / 3600
        parallel = single / max(args.cores, 1)
        label = f"{limit}x{limit} x{classes}"
        print(
            f"  {label:>12}  {game:>8.2f}s  {single:>9.1f}h  {parallel:>9.1f}h"
        )

    print()
    print(
        "自己対戦の探索は根を厳密に解くだけでよく、行列のサイズは検討用と同じである"
        "必要はありません。コストはセル数に比例するので、片側 24 手 → 8 手で 9 倍"
        "安くなります。"
    )
    print(
        "内側のループを書き換えて得られるのは「同じ予算での深さ」であって、"
        "自己対戦が回るかどうかはサイズの選択で決まります。"
    )


if __name__ == "__main__":
    main()
