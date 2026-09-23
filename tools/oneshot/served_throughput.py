"""Served against direct, at the same width, on the same machine, back to back.

The numbers floating around cannot be compared: 113 games/min came from a served run at
width 12 and 83.6 from a direct run at width 16, and width is most of that difference. The
machine's own load has moved too. So each configuration is run here, one after another,
within a few minutes of the others.

Startup is excluded rather than amortised. A direct worker loads torch and builds a CUDA
context; a served worker does neither, and the server loads once for all of them -- which
is a real difference, but it is a fixed cost and it would flatter the server in a short run
and vanish in a long one. Counting only what happens after `--warmup` seconds measures the
rate a long run would actually get.

    uv run --group learn python tools/oneshot/served_throughput.py --seconds 240
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def count(directory: Path) -> int:
    total = 0
    for path in directory.glob("games-worker*.jsonl"):
        try:
            with path.open(encoding="utf-8") as handle:
                total += sum(1 for line in handle if line.strip())
        except OSError:
            pass
    return total


def run(label: str, command: list[str], out: Path, warmup: float, seconds: float) -> dict:
    if out.exists():
        shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "driver.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=str(ROOT), stdout=log, stderr=log)  # noqa: S603
    try:
        time.sleep(warmup)
        first, started = count(out), time.perf_counter()
        time.sleep(seconds)
        last, ended = count(out), time.perf_counter()
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()
    # The driver's own children can outlive it on Windows, so they are ended by hand --
    # but by the whole output path, never by its last component. That was `out.name`, which
    # is the worker count, so the pattern was `*6*` and it matched any command line with a
    # six in it. This script's own, for one: the first run of it killed itself and left an
    # empty log, because stdout had not been flushed either.
    marker = str(out).replace("\\", "\\\\")
    subprocess.run(  # noqa: S603
        ["powershell", "-NoProfile", "-Command",
         f"Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         f"Where-Object {{ $_.ProcessId -ne {os.getpid()} -and "
         f"($_.CommandLine -like '*{marker}*' -or "
         f"$_.CommandLine -like '*inference_server.py*') }} | "
         f"ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }}"],
        check=False, capture_output=True,
    )
    time.sleep(5)
    rate = (last - first) * 60 / (ended - started)
    print(f"  {label:<26}{last - first:>6} games in {ended - started:>5.0f}s  "
          f"= {rate:>6.1f} games/min", flush=True)
    return {"label": label, "games": last - first, "rate": rate}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warmup", type=float, default=90.0, help="seconds to discard")
    ap.add_argument("--seconds", type=float, default=240.0, help="seconds to count over")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--baseline-limit", type=int, default=48)
    ap.add_argument("--seed", type=int, default=31000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "matches" / "_throughput")
    args = ap.parse_args()

    models = ["data/models/value-all.pt", "data/models/value-all-s1.pt"]
    tail = ["--limit", str(args.limit), "--baseline-limit", str(args.baseline_limit),
            "--rank-leaf", "--baseline-rank-leaf"]

    def command(workers: int, servers: int) -> list[str]:
        out = args.out / (f"served{servers}" if servers else "direct") / str(workers)
        base = [sys.executable, str(ROOT / "tools" / "match_queue.py"),
                "--out", str(out), "--games", "100000", "--workers", str(workers),
                "--seed", str(args.seed), "--device", "cuda",
                "--value", *models, "--baseline", *models,
                # What this sweep played before IKA-123 made the condition explicit.
                "--open-bench"]
        if servers:
            base += ["--served", "--servers", str(servers)]
        return base + ["--", *tail]

    # Flushed, all of it. The first run of this died mid-sweep and left an empty log --
    # not because nothing had been printed, but because nothing had been flushed.
    print(f"width {args.limit} against {args.baseline_limit}, "
          f"{args.warmup:.0f}s discarded then {args.seconds:.0f}s counted\n", flush=True)
    # Server processes as well as workers, because the servers turned out to matter more:
    # sixteen workers through one server ran 55.5 games/min and the same sixteen through
    # two ran 116.2. One server saturates on something inside one Python process, so a
    # sweep that varies only the worker count measures that ceiling and nothing else.
    plan = [
        ("direct, 6 workers", 6, 0),
        ("2 servers, 16 workers", 16, 2),
        ("2 servers, 24 workers", 24, 2),
        ("3 servers, 24 workers", 24, 3),
        ("4 servers, 32 workers", 32, 4),
    ]
    results = []
    for label, workers, servers in plan:
        out = args.out / (f"served{servers}" if servers else "direct") / str(workers)
        results.append(run(label, command(workers, servers), out, args.warmup, args.seconds))

    best = max(results, key=lambda r: r["rate"])
    reference = results[0]["rate"]
    print("\n  against direct at 6 workers:")
    for row in results:
        print(f"    {row['label']:<26}{row['rate'] / reference:>6.2f}x")
    print(f"\n  best: {best['label']} at {best['rate']:.1f} games/min")


if __name__ == "__main__":
    main()
