"""A two-seat match whose games are handed out rather than dealt.

The same argument `tools/generate_queue.py` makes for generation applies to matches, and
matches are where most of this project's machine time goes -- one is fifty minutes and the
answers it gives cost fifty minutes each. Generation 8 left 25% of the machine idle under a
fixed split, because a game's cost varies eightfold and a block ends when its unluckiest
worker does.

One index names a whole job: the seat *and* the game. Draining game indices per seat would
let the first seat empty the queue and leave the second with none, and the pairing between
the seats -- the entire reason for playing both -- would be gone. So index `i` is game
`i // 2` in seat `i % 2`, and a worker plays whichever it is handed.

Games are seeded from their index inside the worker, so the work is the same whoever plays
it and in whatever order. That is what makes a queue reproducible and a retried game the
game that was lost.

The driver is `pokeuraou.workqueue.run_workers`, shared with generation: the two had a copy
each, differing only in which program they start and how the indices are laid out, and the
second one written inherited none of the first one's reporting.

    uv run --group learn python tools/match_queue.py \\
        --out data/matches/foo --games 848 --workers 6 \\
        --value data/models/a.pt --baseline data/models/b.pt \\
        -- --limit 48 --rank-leaf --baseline-rank-leaf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.workqueue import run_workers  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, default=848, help="games *per seat*")
    # Six, not the eight generation uses, because a match worker carries two leaves and a
    # generation worker carries one. Measured on this machine (31.1 GB, RTX 5070 12.2 GB),
    # two-member ensembles on both arms:
    #
    #   one match worker   4.0 GB committed, 1.5 GB VRAM, and 27 MB of that is our data --
    #                      the rest is numpy 491 MB, torch 816, a CUDA context 446 and
    #                      torch's arena about 2,000, one set per process
    #   eight of them      31.8 GB committed against 31.1 physical, and 14.8 GB of VRAM
    #                      against 12.2. Neither fits. Eight died at 505 games with
    #                      `illegal memory access` on the first worker and `unknown error`
    #                      on the other seven, which is what the card reports when the
    #                      machine underneath it has run out
    #   six of them        completed 1,696 games in 31 minutes. Free memory did fall to
    #                      0.1 GB on the way and nothing died, so this is the ceiling
    #                      rather than a comfortable number -- a browser holding 3.5 GB is
    #                      enough to change the answer
    #
    # The real fix is tools/inference_server.py: through it a worker is 542 MB and holds no
    # CUDA context at all, and the worker count stops being a memory question. Pass
    # --workers explicitly when running that way.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--value", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", default=None)
    ap.add_argument(
        "--served",
        action="store_true",
        help="start one inference server holding both arms, point every worker at it, and "
        "shut it down at the end. --value and --baseline then say what the server loads "
        "rather than what each worker loads, and a worker holds no torch: 542 MB instead "
        "of 4.0 GB, and no CUDA context. The worker count stops being a memory question.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="after --, options passed to every worker unchanged",
    )
    args = ap.parse_args()
    extra = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["POKEURAOU_RUST_NODE"] = "0" if args.no_bridge else "1"
    env["PYTHONPATH"] = str(ROOT / "src")

    server: subprocess.Popen | None = None
    served_at: str | None = None
    if args.served:
        command = [
            sys.executable, str(ROOT / "tools" / "inference_server.py"),
            "--device", args.device, "--arm", "value", *args.value,
        ]
        if args.baseline:
            command += ["--arm", "baseline", *args.baseline]
        server_log = (out_dir / "logs")
        server_log.mkdir(parents=True, exist_ok=True)
        errors = (server_log / "inference.log").open("w", encoding="utf-8")
        server = subprocess.Popen(  # noqa: S603
            command, env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=errors,
            text=True,
        )
        # The address is the server's first line of stdout. Read it before starting any
        # worker: a worker that starts first has nothing to connect to, and the failure
        # would look like the run dying for a reason of its own.
        assert server.stdout is not None
        served_at = server.stdout.readline().strip()
        if not served_at:
            server.terminate()
            raise SystemExit(
                f"the inference server exited before naming an address; see "
                f"{server_log / 'inference.log'}"
            )
        print(f"  inference: {served_at} (workers hold no model)", file=sys.stderr,
              flush=True)

    def build(worker: int, address: str) -> list[str]:
        command = [
            sys.executable, str(ROOT / "tools" / "generation_match.py"),
            "--queue", address,
            "--seed", str(args.seed),
            "--games", str(args.games),
            "--device", args.device,
            "--torch-threads", "1",
            "--out", str(out_dir / f"worker{worker}.jsonl"),
            "--games-out", str(out_dir / f"games-worker{worker}.jsonl"),
        ]
        if served_at is not None:
            command += ["--inference", served_at, "--inference-arm", "value"]
            if args.baseline:
                command += ["--baseline-inference-arm", "baseline"]
        else:
            command += ["--value", *args.value]
            if args.baseline:
                command += ["--baseline", *args.baseline]
        return command + extra

    def written() -> int:
        return sum(
            1
            for path in out_dir.glob("games-worker*.jsonl")
            for line in path.open(encoding="utf-8")
            if line.strip()
        )

    print(
        f"match: {args.workers} workers sharing {2 * args.games} seat-games "
        f"-> {out_dir}\n  seed {args.seed}, device {args.device}",
        file=sys.stderr,
        flush=True,
    )
    # Two seats per game, interleaved, so a straggler cannot strand half of a pair.
    try:
        status = run_workers(
            range(2 * args.games), build,
            workers=args.workers, out_dir=out_dir, env=env, cwd=str(ROOT),
            label="match", counts=written,
        )
    finally:
        # The server outlives every worker and is ours to end, however the run ended --
        # a leftover one holds 1.5 GB of VRAM and answers the next run's questions with
        # the previous run's models.
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
