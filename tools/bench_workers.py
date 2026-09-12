"""How many generation processes should this machine run at once?

`bench_generation.py` measures one process. Generation is throughput, not latency, so the
number that decides how long a generation takes is the *aggregate* -- and it stops rising
well before the process count does, because eight cores are eight cores and every worker
now drives a Rust process beside it.

This launches N workers with distinct seeds and reports games a minute for the whole
machine, so the choice of N is a measurement rather than a habit.

    uv run --group learn python tools/bench_workers.py --workers 1,2,4,8,12,16 --games 3
    uv run --group learn python tools/bench_workers.py --workers 1,4,8 --games 3 \
        --value data/models/value-gen234.pt --device cpu

Absolute numbers are only meaningful on an idle machine; the *shape* -- where the curve
flattens -- survives a bit of background load.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
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

    started = time.perf_counter()
    running = [
        subprocess.Popen(  # noqa: S603
            # Every worker plays the *same* games. Different seeds would make the four-
            # worker wave a different workload from the one-worker wave, and twelve games
            # vary enough between seeds to swamp the scaling this is trying to measure.
            command + ["--seed", str(args.seed)],
            env=env,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for index in range(workers)
    ]
    reported = 0.0
    failures = 0
    for process in running:
        out, _ = process.communicate()
        failures += int(process.returncode != 0)
        for line in (out or "").splitlines():
            if "games/min" in line:
                reported += float(line.split("games/min")[0].split(",")[-1].strip())
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
        "75% at width 24, because its cost is the pool against the matrix's limit^2.",
    )
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--value", default=None, help="a trained value function, or hp-share")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-bridge", action="store_true", help="measure without the Rust port")
    args = ap.parse_args()

    # One wave thrown away first. The very first measurement on an idle machine is taken
    # against cold file caches and a CPU that has not boosted yet, and it came out 43%
    # slower than the same wave run second -- which would have read as superlinear scaling.
    run_wave(1, args)

    counts = [int(n) for n in args.workers.split(",") if n]
    leaf = Path(args.value).stem if args.value else "hp-share"
    print(f"{leaf} on {args.device}, {args.games} games per worker, bridge "
          f"{'off' if args.no_bridge else 'on'}")
    print(
        f"{'workers':>8}  {'seconds':>8}  {'wall g/min':>11}  {'steady g/min':>13}  "
        f"{'per worker':>11}  {'scaling':>8}"
    )
    one = None
    for workers in counts:
        elapsed, reported, failures = run_wave(workers, args)
        total = workers * args.games
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
