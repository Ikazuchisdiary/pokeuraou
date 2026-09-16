"""Holds the leaves so the workers do not have to.

A match worker is 4.0 GB of commit and 1.5 GB of VRAM, and 27 MB of that is this project's
data -- the rest is torch, a CUDA context and torch's arena, one set per process. Eight of
them do not fit on this machine; eight generation workers do, because a generation worker
carries one leaf and a match worker carries two. Moving the leaf out makes a worker about
half a gigabyte with no CUDA context, and the worker count stops being a memory question.

Arms are named, because a match has two and they must not be confused:

    uv run --group learn python tools/inference_server.py \\
        --arm value data/models/value-all.pt data/models/value-all-s1.pt \\
        --arm baseline data/models/value-gen8.pt \\
        --device cuda

It prints the address on stdout as its first line, so a launcher can read it and pass it to
the workers, and keeps running until interrupted.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.inference import load_models, serve  # noqa: E402
from pokeuraou.teams import load_roster  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        action="append",
        nargs="+",
        metavar=("NAME", "MODEL"),
        required=True,
        help="a named arm and its model files; several files are an ensemble, "
        "averaged in logit space",
    )
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 asks the OS for a free one")
    ap.add_argument("--report-every", type=float, default=60.0, help="seconds, 0 to hush")
    args = ap.parse_args()

    import torch

    from pokeuraou.encode import Encoder

    torch.set_num_threads(1)
    reg = load_roster(args.roster).reg
    register_mega_stones(reg)
    encoder = Encoder(reg)

    paths: dict[str, list[Path]] = {}
    for entry in args.arm:
        if len(entry) < 2:
            raise SystemExit(f"--arm needs a name and at least one model: {entry}")
        paths[entry[0]] = [Path(p) for p in entry[1:]]

    models = load_models(paths, encoder, args.device)
    server, address = serve(
        models,
        host=args.host,
        port=args.port,
        # So a client can ask what an arm *is*, rather than recording the name it was
        # given on its own command line.
        arms={name: [p.name for p in group] for name, group in paths.items()},
    )

    # First line of stdout, so a launcher can read it without parsing the prose.
    print(address, flush=True)
    for name, group in paths.items():
        print(f"  arm {name}: {', '.join(p.name for p in group)}"
              + (" (ensemble, logits averaged)" if len(group) > 1 else ""),
              file=sys.stderr)
    print(f"  on {args.device}; requests are served as they arrive and are never merged "
          f"across workers, so every answer is the one a worker would have computed itself",
          file=sys.stderr, flush=True)

    stopping = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    last = time.perf_counter()
    served = 0
    while not stopping.wait(timeout=1.0):
        if args.report_every and time.perf_counter() - last >= args.report_every:
            now = server.requests_served
            # Waiting and working, separately. Throughput falls as workers are added,
            # which is a queue rather than a latency, and this says whether the queue is
            # this lock: if waited climbs with the worker count while held stays put, it
            # is, and if neither moves the contention is somewhere else.
            waited = sum(getattr(m, "waited", 0.0) for m in models.values())
            held = sum(getattr(m, "held", 0.0) for m in models.values())
            calls = sum(getattr(m, "calls", 0) for m in models.values()) or 1
            print(f"  {now:,} requests, {server.rows_served:,} rows "
                  f"({now - served} since the last line); per call "
                  f"{1000 * waited / calls:.2f} ms waiting for the lock, "
                  f"{1000 * held / calls:.2f} ms holding it",
                  file=sys.stderr, flush=True)
            served, last = now, time.perf_counter()
    server.shutdown()
    print(f"stopped after {server.requests_served:,} requests, "
          f"{server.rows_served:,} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
