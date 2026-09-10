"""Where the turn resolver's time actually goes.

Self-play for a learned value function needs something like a hundred times the current
1.6 ms per turn, and there are two very different ways to chase that: rewrite the resolver
around batched arrays, or find out that a handful of call sites dominate. Guessing which
would be expensive to get wrong -- an array rewrite is weeks of work and puts the 3.4%
silent divergence rate at risk -- so this profiles the real workload first.

Two workloads, because they stress different things:

    turn      one resolve_turn at the matrix budget, the unit of self-play
    matrix    a full 24x24 fill, the unit the CLI pays for

    uv run python tools/profile_resolve.py --workload matrix --top 25
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.cli import _modal, build_beliefs
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.resolve import Budget, resolve_turn
from pokeuraou.setup import load_scenario, with_spreads

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn1.json"


def build() -> tuple[object, object, list, list]:  # noqa: ANN401
    scenario = load_scenario(EXAMPLE)
    reg = scenario.reg
    register_mega_stones(reg)
    beliefs = build_beliefs(scenario)
    base = with_spreads(scenario, {key: _modal(b) for key, b in beliefs.items()})
    ours = narrow(reg, base, 0, limit=24).actions
    theirs = narrow(reg, base, 1, limit=24).actions
    return reg, base, ours, theirs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", choices=("turn", "matrix"), default="turn")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=200)
    args = ap.parse_args()

    reg, base, ours, theirs = build()
    budget = Budget.matrix()

    if args.workload == "turn":
        def work() -> None:
            for _ in range(args.repeat):
                resolve_turn(reg, base, [ours[0], theirs[0]], budget=budget)
        unit, count = "turn", args.repeat
    else:
        def work() -> None:
            for a in ours:
                for b in theirs:
                    result = resolve_turn(reg, base, [a, b], budget=budget)
                    result.expected(HP_SHARE)
        unit, count = "cell", len(ours) * len(theirs)

    work()  # warm up: first-call imports and caches are not what is being measured
    started = time.perf_counter()
    work()
    elapsed = time.perf_counter() - started
    print(f"{unit}: {count} x {elapsed / count * 1e6:.0f} us = {elapsed:.2f} s\n")

    profiler = cProfile.Profile()
    profiler.enable()
    work()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs()
    stats.sort_stats("tottime").print_stats(args.top)
    text = stream.getvalue()
    # Drop pstats' preamble; the table is what matters.
    marker = "ncalls"
    index = text.find(marker)
    print(text[index:] if index >= 0 else text)


if __name__ == "__main__":
    main()
