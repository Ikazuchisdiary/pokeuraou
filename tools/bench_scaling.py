"""How many workers this machine actually wants, measured rather than guessed.

Generation is the only bottleneck -- 11 core-hours against 11 seconds of training -- so the
worker count is worth getting right. It was guessed at 10, then raised to 14 because the
CPU counter read 35.6%, and neither number came from a measurement.

Guessing is particularly unreliable here because 8 physical cores presenting 16 threads do
not give 16 cores' throughput. Hyperthreading typically adds 20-30% on this kind of pure
Python work, so the throughput curve flattens somewhere above 8 and a counter reading 98.5%
can mean "saturated" while the extra workers are only stealing from each other. Total games
per minute is the thing to maximise, not utilisation.

Each worker plays the same fixed number of games, and the wall time of the *whole* group is
what counts -- a run where one worker finishes early and the rest crawl is not faster.

    uv run --group learn python tools/bench_scaling.py --counts 4,8,10,12,14,16 --games 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", default="4,8,12,16")
    ap.add_argument("--games", type=int, default=3, help="games per worker")
    ap.add_argument("--value", type=Path, default=None, help="omit for the hp-share leaf")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()

    print(f"logical cpus: {os.cpu_count()}")
    leaf = args.value.stem if args.value else "hp-share"
    print(f"leaf {leaf} on {args.device}, {args.games} games per worker")
    print(f"\n  {'workers':>8}  {'wall s':>7}  {'games/min':>10}  {'per worker':>11}  {'vs 1 worker':>12}")

    baseline = None
    for token in args.counts.split(","):
        workers = int(token.strip())
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        command = [
            sys.executable,
            "-u",
            str(ROOT / "tools" / "bench_generation.py"),
            "--games",
            str(args.games),
            "--limit",
            str(args.limit),
            "--device",
            args.device,
            "--torch-threads",
            "1",
        ]
        if args.value is not None:
            command += ["--value", str(args.value)]

        started = time.perf_counter()
        # Different seeds so the workers do not all play the identical game, which would
        # let the OS cache make the group look faster than a real run.
        procs = [
            subprocess.Popen(  # noqa: S603
                command + ["--seed", str(100 + i)],
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for i in range(workers)
        ]
        for proc in procs:
            proc.wait()
        elapsed = time.perf_counter() - started

        total = workers * args.games
        rate = total / elapsed * 60
        if baseline is None:
            baseline = rate / workers
        print(
            f"  {workers:>8}  {elapsed:>7.1f}  {rate:>10.1f}  "
            f"{rate / workers:>11.1f}  {rate / (baseline * workers):>11.2f}x"
        )

    print(
        "\n  'per worker' falling is the cost of contention; the count to use is where\n"
        "  games/min stops rising, not where the CPU counter reaches 100%."
    )


if __name__ == "__main__":
    main()
