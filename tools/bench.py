"""Benchmarks for the pieces whose speed decides what the tool can do.

Every number the project quotes comes from here, so it is a command rather than a claim.

    uv run python tools/bench.py
    uv run python tools/bench.py --matrix 16
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import calculate, register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.oracle import load_team  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import STAT_IDS, load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, resolve_turn  # noqa: E402
from pokeuraou.speed import build_queue, effective_speed, order_groups  # noqa: E402
from pokeuraou.stats import nature_multipliers, stats_from_sp  # noqa: E402
from pokeuraou.view import active_battlers, field_state  # noqa: E402

FORMAT_ID = "gen9championsvgc2026regmc"


def timed(label: str, repeats: int, fn) -> float:  # noqa: ANN001
    fn()  # warm up
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    per = (time.perf_counter() - start) / repeats
    unit = "us" if per < 1e-3 else "ms"
    shown = per * 1e6 if unit == "us" else per * 1e3
    print(f"  {label:<44} {shown:9.2f} {unit}   {1 / per:12,.0f} /s")
    return per


def bench_stats(reg) -> None:  # noqa: ANN001
    print("stat computation (vectorised over the belief axis)")
    base = np.array(reg.species["incineroar"].base_stats, dtype=np.int64)
    for n in (1, 100, 10_000):
        sp = np.random.default_rng(0).integers(0, 33, size=(n, len(STAT_IDS)))
        num = nature_multipliers(reg, ["Impish"] * n)
        timed(f"stats_from_sp, {n:,} spreads", 200 if n < 10_000 else 50,
              lambda sp=sp, num=num: stats_from_sp(reg, base, sp, num))


def bench_damage(reg, pos: Position) -> None:  # noqa: ANN001
    print("damage (all 16 rolls per call)")
    fs = field_state(pos, reg)
    ours = active_battlers(reg, pos.sides[0])
    theirs = active_battlers(reg, pos.sides[1])
    attacker, defender = ours[0], theirs[0]
    assert attacker is not None and defender is not None
    move = next(
        m.id for m in pos.sides[0].pokemon[0].moves if reg.moves[m.id].category != "Status"
    )
    timed(
        "calculate, 1 particle",
        2000,
        lambda: calculate(reg, attacker, defender, move, fs, defender_side=1),
    )

    # The belief axis is what this is vectorised for: many candidate spreads at once.
    species = reg.species[defender.species]
    rng = np.random.default_rng(0)
    for n in (100, 10_000):
        sp = rng.integers(0, 33, size=(n, len(STAT_IDS)))
        num = nature_multipliers(reg, ["Impish"] * n)
        stats = stats_from_sp(reg, np.array(species.base_stats, dtype=np.int64), sp, num)
        batched = copy.copy(defender)
        batched.stats = stats
        batched.hp = stats[:, 0].copy()
        batched.maxhp = stats[:, 0].copy()
        timed(
            f"calculate, {n:,} particles",
            200 if n < 10_000 else 20,
            lambda b=batched: calculate(reg, attacker, b, move, fs, defender_side=1),
        )


def bench_speed(reg, pos: Position) -> None:  # noqa: ANN001
    print("turn order")
    fs = field_state(pos, reg)
    battlers = [active_battlers(reg, side) for side in pos.sides]
    mon = battlers[0][0]
    assert mon is not None
    timed("effective_speed", 20_000, lambda: effective_speed(reg, mon, fs, frozenset()))
    chosen = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
    queue = build_queue(reg, pos, chosen, battlers, fs)[0]
    timed("order_groups, 4 actions", 5000, lambda: order_groups(queue, trick_room=False))


def bench_actions(reg, pos: Position) -> None:  # noqa: ANN001
    print("action enumeration")
    for side in (0, 1):
        count = len(side_actions(reg, pos, side))
        timed(f"side_actions (side {side + 1}, {count} actions)", 500,
              lambda side=side: side_actions(reg, pos, side))


def bench_resolve(reg, pos: Position) -> None:  # noqa: ANN001
    print("turn resolution")
    chosen = [side_actions(reg, pos, 0)[0], side_actions(reg, pos, 1)[0]]
    timed("Position.copy", 5000, pos.copy)
    timed("copy.deepcopy(Position), for comparison", 500, lambda: copy.deepcopy(pos))
    for label, budget in (
        ("resolve_turn, Budget.matrix() (1 branch)", Budget.matrix()),
        ("resolve_turn, Budget.fast() (2 rolls)", Budget.fast()),
    ):
        result = resolve_turn(reg, pos, chosen, budget=budget)
        timed(f"{label} -> {len(result.branches)} branches", 20,
              lambda b=budget: resolve_turn(reg, pos, chosen, budget=b))


def bench_equilibrium() -> None:
    print("equilibrium (HiGHS)")
    rng = np.random.default_rng(0)
    for n in (8, 16, 24, 48):
        payoff = rng.random((n, n))
        timed(f"solve, {n}x{n}", 50, lambda payoff=payoff: solve(payoff))


def bench_matrix(reg, pos: Position, size: int) -> None:  # noqa: ANN001
    """The end-to-end cost of one matrix, which is what the tool has to pay per position."""
    print(f"one {size}x{size} matrix, Budget.matrix()")
    ours = side_actions(reg, pos, 0)[:size]
    theirs = side_actions(reg, pos, 1)[:size]
    budget = Budget.matrix()
    start = time.perf_counter()
    cells = 0
    for a in ours:
        for b in theirs:
            resolve_turn(reg, pos, [a, b], budget=budget)
            cells += 1
    total = time.perf_counter() - start
    print(
        f"  {cells} cells in {total:.2f} s  ({total / cells * 1e3:.2f} ms/cell, "
        f"{cells / total:,.0f} cells/s)"
    )
    print(
        f"  a 24x24 matrix would be {576 * total / cells:.1f} s on one core, "
        f"{576 * total / cells / 16:.1f} s across 16"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=int, default=8, help="matrix size to time end to end")
    ap.add_argument("--skip-matrix", action="store_true")
    args = ap.parse_args()

    reg = load_regulation(FORMAT_ID)
    register_mega_stones(reg)
    from tests.test_actions import _synthetic_position

    pos = _synthetic_position(reg, load_team("tests/fixtures/team_a.json"))

    print(f"pokeuraou benchmarks  ({FORMAT_ID}, commit {reg.meta.showdown_commit[:8]})\n")
    bench_stats(reg)
    print()
    bench_damage(reg, pos)
    print()
    bench_speed(reg, pos)
    print()
    bench_actions(reg, pos)
    print()
    bench_resolve(reg, pos)
    print()
    bench_equilibrium()
    if not args.skip_matrix:
        print()
        bench_matrix(reg, pos, args.matrix)


if __name__ == "__main__":
    main()
