"""How many generation processes should this machine run at once?

`bench_generation.py` measures one process. Generation is throughput, not latency, so the
number that decides how long a generation takes is the *aggregate* -- and it stops rising
well before the process count does, because eight cores are eight cores and every worker
now drives a Rust process beside it.

This launches N workers and reports games a minute for the whole machine, so the choice of
N is a measurement rather than a habit.

    uv run --group learn python tools/bench_workers.py --workers 1,2,4,8 --games 4
    uv run --group learn python tools/bench_workers.py --workers 1,4,8 --games 4 \
        --value data/models/value-gen234.pt --device cpu

**Every wave plays the same games, and there are several of them.** Both halves matter,
and the tool produced a table that had to be thrown away for want of each.

A game's cost varies by a factor of eight. Measured serially on an idle machine, one
process at a time, seed by seed: 1.77, 2.35, 2.78, 2.87, 3.81, 8.71, 12.61, 13.75 s/game.
Nothing is contending; that is simply how much the games differ. What varies is not their
length in turns but the number of leaves the search has to evaluate, since a position that
needs an exact budget expands one matrix cell into many branches. So three games is not a
workload, it is one draw from a distribution with an eight-fold spread, and a wave of them
can land anywhere. Give each wave several seeds.

And every wave must play the *identical* set. Giving each concurrency level its own seeds
compares game sets rather than concurrency: that mistake made a one-worker row look 3.4x
slower than the same configuration measured on another seed, which reads as the machine
collapsing when nothing is wrong with it.

Absolute numbers are only meaningful on an idle machine; the *shape* -- where the curve
flattens -- survives a bit of background load. On this machine (8 physical cores) the
answer came out 5.6x at eight workers, still rising, with cuda and cpu within 8%% of each
other at every count and the sign of the difference changing between them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_wave(workers: int, args: argparse.Namespace) -> tuple[float, float, int]:
    """Starts `workers` generation processes at once and waits for all of them.

    Returns wall clock, the sum of what the workers reported for themselves, and how many
    failed. The two rates answer different questions: wall clock includes loading the
    regulation, the standings and the priors in every worker, which a real run pays once
    and a benchmark of twelve games pays over and over.
    """
    env = dict(os.environ)
    env["POKEURAOU_RUST_NODE"] = "0" if args.no_bridge else "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "tools" / "bench_generation.py"),
        "--games",
        str(args.games),
        "--limit",
        str(args.limit),
        "--device",
        args.device,
    ]
    if args.value:
        command += ["--value", args.value]
    if args.rank_leaf:
        command += ["--rank-leaf"]

    seeds = [int(x) for x in str(args.seeds).split(",") if x]

    def one_worker() -> tuple[float, int]:
        """One worker's share: every seed in turn, which is its whole workload."""
        rate = 0.0
        failed = 0
        for seed in seeds:
            process = subprocess.Popen(  # noqa: S603
                command + ["--seed", str(seed)],
                env=env,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            out, _ = process.communicate()
            failed += int(process.returncode != 0)
            for line in (out or "").splitlines():
                if "games/min" in line:
                    rate += float(line.split("games/min")[0].split(",")[-1].strip())
        return rate / max(len(seeds), 1), failed

    started = time.perf_counter()
    # Every worker runs the same seeds, so each concurrency level plays an identical
    # workload and the only thing that changes between rows is how many run at once.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = [f.result() for f in [pool.submit(one_worker) for _ in range(workers)]]
    reported = sum(rate for rate, _ in results)
    failures = sum(failed for _, failed in results)
    return time.perf_counter() - started, reported, failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", default="1,2,4,8,12,16", help="comma-separated counts")
    ap.add_argument("--games", type=int, default=3, help="games per worker")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument(
        "--rank-leaf",
        action="store_true",
        help="rank the menu by the leaf, as generation does. Nearly free at width 48 and "
        "75%% at width 24, because its cost is the pool against the matrix's limit^2.",
    )
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--value", default=None, help="a trained value function, or hp-share")
    ap.add_argument(
        "--seeds",
        default="1000,1001,1002,1003,1004,1005,1006,1007",
        help="comma-separated seeds. Every worker plays all of them, so a wave is the "
        "same workload at every concurrency. Several, because one game's cost varies "
        "eight-fold and a single seed is a draw rather than a measurement.",
    )
    ap.add_argument("--no-bridge", action="store_true", help="measure without the Rust port")
    args = ap.parse_args()

    # One wave thrown away first. The very first measurement on an idle machine is taken
    # against cold file caches and a CPU that has not boosted yet, and it came out 43%
    # slower than the same wave run second -- which would have read as superlinear scaling.
    run_wave(1, args)

    counts = [int(n) for n in args.workers.split(",") if n]
    seeds = [x for x in str(args.seeds).split(",") if x]
    leaf = Path(args.value).stem if args.value else "hp-share"
    print(f"{leaf} on {args.device}, {args.games} games x {len(seeds)} seeds per worker, "
          f"bridge {'off' if args.no_bridge else 'on'}")
    print(
        f"{'workers':>8}  {'seconds':>8}  {'wall g/min':>11}  {'steady g/min':>13}  "
        f"{'per worker':>11}  {'scaling':>8}"
    )
    one = None
    for workers in counts:
        elapsed, reported, failures = run_wave(workers, args)
        total = workers * args.games * len(seeds)
        rate = total / elapsed * 60
        if one is None:
            one = reported / workers
        note = "" if not failures else f"  ({failures} failed)"
        print(
            f"{workers:>8}  {elapsed:>8.1f}  {rate:>11.1f}  {reported:>13.1f}  "
            f"{reported / workers:>11.1f}  {reported / (one * workers):>7.0%}{note}"
        )


if __name__ == "__main__":
    main()
