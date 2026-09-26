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
import faulthandler
import signal
import sys
import threading
import time
from pathlib import Path

# A server that dies without saying why takes every worker with it, and one did: no
# traceback, nothing in the log, and the workers saw a reset connection. `faulthandler`
# catches what Python's own handling cannot -- an access violation inside a native
# library -- and writes the stack to stderr on the way down.
faulthandler.enable()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import timing  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.inference import (  # noqa: E402
    load_models,
    scheduling,
    serve,
    wait_by_sleeping,
)
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
    ap.add_argument(
        "--q-arm",
        action="append",
        nargs=2,
        default=[],
        metavar=("NAME", "MODEL"),
        help="a named Q arm (IKA-274, pokeuraou.qrank): one Q file, asked by the q rank "
        "fills of the workers (--q-arm NAME there)",
    )
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--regulation",
        default=None,
        help="the format id the encoder is built for, instead of the roster's. M-C "
        "generation has no roster (IKA-81), so generate_queue.py --pool names the pool's.",
    )
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0, help="0 asks the OS for a free one")
    ap.add_argument("--report-every", type=float, default=60.0, help="seconds, 0 to hush")
    args = ap.parse_args()

    import torch

    from pokeuraou.encode import Encoder

    torch.set_num_threads(1)
    if args.regulation is not None:
        from pokeuraou.regulation import load_regulation

        reg = load_regulation(args.regulation)
    else:
        reg = load_roster(args.roster).reg
    register_mega_stones(reg)
    encoder = Encoder(reg)

    paths: dict[str, list[Path]] = {}
    for entry in args.arm:
        if len(entry) < 2:
            raise SystemExit(f"--arm needs a name and at least one model: {entry}")
        paths[entry[0]] = [Path(p) for p in entry[1:]]

    if args.device == "cuda":
        # Before the first CUDA call, or the context already exists with the default
        # (spinning) wait. IKA-106.
        wait_by_sleeping()
    models = load_models(paths, encoder, args.device)
    q_models = {}
    if args.q_arm:
        from pokeuraou.qrank import load_q_arms

        q_models = load_q_arms({name: Path(path) for name, path in args.q_arm}, args.device)
        for name, model in q_models.items():
            if model.fingerprint != encoder.vocab.fingerprint():
                raise SystemExit(f"Q arm {name!r} reads another vocabulary than this server's")
    server, address = serve(
        models,
        host=args.host,
        port=args.port,
        # So a client can ask what an arm *is*, rather than recording the name it was
        # given on its own command line.
        arms={name: [p.name for p in group] for name, group in paths.items()},
        q_models=q_models,
    )

    # First line of stdout, so a launcher can read it without parsing the prose.
    print(address, flush=True)
    for name, group in paths.items():
        print(f"  arm {name}: {', '.join(p.name for p in group)}"
              + (" (ensemble, logits averaged)" if len(group) > 1 else ""),
              file=sys.stderr)
    for name, model in q_models.items():
        print(f"  Q arm {name}: {', '.join(model.files)} (eager, no graphs)", file=sys.stderr)
    if args.device == "cuda":
        print(f"  cuda waits: {scheduling()}", file=sys.stderr)
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
    # The launcher stops a server with `Popen.terminate()`, which on Windows is
    # `TerminateProcess`: no signal, no `atexit`, no report. So the report is
    # written over and over while the server lives, and what survives the kill is
    # the last one. Cheap enough to do often -- it is one small JSON file.
    # `count` accumulates and `note` runs every five seconds, so rows go in as a delta.
    _reported_rows = [0]

    def note() -> None:
        if not timing.ON:
            return
        timing.set_total(
            "server.held",
            sum(getattr(m, "held", 0.0) for m in models.values()),
            calls=sum(getattr(m, "calls", 0) for m in models.values()),
        )
        timing.set_total(
            "server.queue",
            sum(getattr(m, "waited", 0.0) for m in models.values()),
            calls=sum(getattr(m, "calls", 0) for m in models.values()),
        )
        if q_models:
            timing.set_total(
                "server.q",
                sum(m.held for m in q_models.values()),
                calls=sum(m.calls for m in q_models.values()),
            )
        # Rows are a count, not a call count. Putting `rows_served` in the calls column
        # made the queueing row read as 3.2 million calls of 0.03 microseconds each.
        timing.count("server.rows", int(server.rows_served) - _reported_rows[0])
        _reported_rows[0] = int(server.rows_served)
        timing.write_report("inference-server")

    # Serving from here: what came before is startup, and IKA-98's spin ratio leaves it out.
    timing.ready()
    # One on the way in, so the file exists from the first second rather than the
    # fifth: a run shorter than the cadence produced no server report at all.
    note()
    noted = time.perf_counter()
    while not stopping.wait(timeout=1.0):
        if time.perf_counter() - noted >= 5.0:
            note()
            noted = time.perf_counter()
        if args.report_every and time.perf_counter() - last >= args.report_every:
            now = server.requests_served
            # Both halves of the same line, because they answer the two questions a
            # server gets asked when a run goes wrong.
            #
            # Waiting against working says whether requests are queueing. It is how the
            # contention was found: holding flat at 5.6 ms while waiting went 7.4 -> 32.0
            # as workers went 6 -> 14, which is a saturated lock and not a slow one. It
            # should now read near zero, and its climbing again would mean a new queue.
            #
            # What CUDA is holding says what the server had on it when it died. One died
            # three times with no traceback and nothing from `faulthandler`, which rules
            # out a native fault and leaves being killed from outside.
            waited = sum(getattr(m, "waited", 0.0) for m in models.values())
            held = sum(getattr(m, "held", 0.0) for m in models.values())
            calls = sum(getattr(m, "calls", 0) for m in models.values()) or 1
            reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            in_use = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"  {now:,} requests, {server.rows_served:,} rows "
                  f"({now - served} since the last line); per call "
                  f"{1000 * waited / calls:.2f} ms queued, {1000 * held / calls:.2f} ms "
                  f"working; cuda reserved {reserved:.2f} GB, in use {in_use:.2f} GB",
                  file=sys.stderr, flush=True)
            served, last = now, time.perf_counter()
    note()
    server.shutdown()
    print(f"stopped after {server.requests_served:,} requests, "
          f"{server.rows_served:,} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
