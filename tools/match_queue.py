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

A match that exists to decide something -- ship it or not -- can stop as soon as it has
decided (`--sprt`, IKA-89); `--games` is then the most it will play. A match whose number
is the point (a conversion rate, one side of a triangle) plays its count out.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.sprt import Sprt, StopWhenDecided  # noqa: E402
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
    # With --served a worker is 224 MB and holds no CUDA context, so this number stops
    # being a memory question and becomes a throughput one: 16 workers over 2 servers ran
    # 116.2 games/min against 82.5 for the best direct configuration. Raise it when serving.
    # Resolved after parsing, because the right number depends on where the leaf
    # lives: 6 when every worker carries one, 24 when they are served.
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--value", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", default=None)
    ap.add_argument(
        "--served",
        action="store_true",
        help="run the leaves in inference servers instead of in every worker. --value and "
        "--baseline then say what the servers load rather than what each worker loads, and "
        "a worker holds no torch at all: 224 MB against 4.0 GB, and no CUDA context. See "
        "--servers, which decides whether this is faster or slower than not using it.",
    )
    ap.add_argument(
        "--servers",
        type=int,
        default=2,
        help="inference server processes, with workers dealt round robin between them. "
        "Two, not one, and it is the largest single effect measured on this: sixteen "
        "workers through one server ran 55.5 games/min and the same sixteen through two "
        "ran 116.2, on the same card. One Python process serialises more than the card "
        "does -- the arithmetic releases the GIL but the JSON, the buffer views and the "
        "result do not -- so the server has to be more than one process before it beats "
        "a direct run at all. Answers do not depend on this: requests are never merged, "
        "so a batch gets the same numbers wherever it is served.",
    )
    ap.add_argument(
        "--hide-bench",
        action="store_true",
        help="pass --hide-bench to every worker. A model trained on hidden-bench games "
        "has to be judged in that condition; judging it in the open game measures which "
        "model suits the open game.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument(
        "--uniform-selection",
        action="store_true",
        help="both arms draw their four of six uniformly, and you mean it. Required when "
        "no book reaches the workers, because a uniform draw is worth about -141 Elo "
        "against the advice -- the whole distance from the parameter-free baseline to the "
        "best agent on the scale -- and an agent measured that way is an agent playing a "
        "selection nobody would play. `tools/generate_queue.py` has refused this since "
        "generation 9; this wrapper never did, and every width match and several "
        "generation matches recorded `books: [uniform, uniform]` as a result.",
    )
    ap.add_argument(
        "--sprt",
        nargs=2,
        type=float,
        metavar=("ELO0", "ELO1"),
        default=None,
        help="stop as soon as a sequential probability ratio test decides between 'the "
        "tested arm is worth ELO0' and 'it is worth ELO1' (logistic Elo per game, over "
        "game pairs; `pokeuraou.sprt`). --games becomes the cap: a run undecided there ends "
        "there, and its fixed count is the answer. The test is written to sprt.json before "
        "the first game and its outcome after the last; tools/sprt_replay.py replays it over "
        "a finished match. For a decision only -- a run stopped early reports a win rate "
        "biased away from 50%%, so a size needs a fixed count.",
    )
    ap.add_argument("--sprt-alpha", type=float, default=0.05,
                    help="chance of passing a change worth ELO0 or less")
    ap.add_argument("--sprt-beta", type=float, default=0.05,
                    help="chance of failing a change worth ELO1 or more")
    ap.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="after --, options passed to every worker unchanged",
    )
    args = ap.parse_args()
    if args.workers is None:
        # Swept on the board, width 16 against 48, startup discarded, one machine, back to
        # back:
        #
        #   direct,      6 workers    84.0 games/min   1.00x
        #   2 servers,  16 workers    97.4             1.16x
        #   2 servers,  24 workers   108.8             1.30x
        #   3 servers,  24 workers   109.4             1.30x  -- a third server buys 0.6%
        #   4 servers,  32 workers   112.7             1.34x  -- a third more workers, 3%
        #
        # So 24 served workers over two servers, and no more: past that the curve
        # flattens and the CPU is pegged. Direct stays at 6 because a direct worker is
        # 4.0 GB and eight of them did not fit in 31.1.
        args.workers = 24 if getattr(args, "served", False) else 6
    extra = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest

    # The guard `tools/generate_queue.py` has had since generation 9, and this wrapper
    # did not. A book reaches the workers only through the free-form tail, so leaving it
    # out looks exactly like not wanting one -- and the two differ by about 141 Elo, the
    # whole distance from the parameter-free baseline to the best agent on the scale.
    # Every width match and several generation matches recorded `books: [uniform,
    # uniform]` without anyone choosing that.
    if "--selection-book" not in extra and not args.uniform_selection:
        raise SystemExit(
            "no --selection-book in the options passed after `--`, so both arms will "
            "draw their four of six uniformly. That is worth about -141 Elo against the "
            "advice and is not a default: pass a book, or pass --uniform-selection to "
            "mean it (which the hp-share anchors do)."
        )
    if args.uniform_selection and "--selection-book" in extra:
        raise SystemExit(
            "--uniform-selection contradicts the --selection-book in the tail; one of "
            "them is not what you meant."
        )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    monitor = None
    if args.sprt is not None:
        record = out_dir / "sprt.json"
        # One registration per run. A test whose bounds could be rewritten after games
        # exist is not a test fixed in advance, and a leftover file from an earlier run in
        # the same directory would be read as this run's.
        if record.exists():
            raise SystemExit(
                f"{record} already exists: a registered test belongs to one run. Use a new "
                "--out, or remove the file if that run is being discarded."
            )
        test = Sprt(args.sprt[0], args.sprt[1], alpha=args.sprt_alpha, beta=args.sprt_beta)
        monitor = StopWhenDecided(out_dir, test, record=record)
        monitor.save()
        print(
            f"  registered SPRT({test.elo0:+g}, {test.elo1:+g}), alpha {test.alpha}, "
            f"beta {test.beta}: LLR bounds [{test.lower:+.3f}, {test.upper:+.3f}] -> {record}",
            file=sys.stderr,
            flush=True,
        )

    env = dict(os.environ)
    env["POKEURAOU_RUST_NODE"] = "0" if args.no_bridge else "1"
    env["PYTHONPATH"] = str(ROOT / "src")

    servers: list[subprocess.Popen] = []
    served_at: list[str] = []
    if args.served:
        server_log = out_dir / "logs"
        server_log.mkdir(parents=True, exist_ok=True)
        for index in range(args.servers):
            command = [
                sys.executable, str(ROOT / "tools" / "inference_server.py"),
                "--device", args.device, "--arm", "value", *args.value,
            ]
            if args.baseline:
                command += ["--arm", "baseline", *args.baseline]
            errors = (server_log / f"inference{index}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(  # noqa: S603
                command, env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=errors,
                text=True,
            )
            servers.append(process)
            # The address is the server's first line of stdout, read before any worker
            # starts: a worker that starts first has nothing to connect to, and that
            # failure reads as the run dying for a reason of its own.
            assert process.stdout is not None
            address = process.stdout.readline().strip()
            if not address:
                for other in servers:
                    other.terminate()
                raise SystemExit(
                    f"inference server {index} exited before naming an address; see "
                    f"{server_log / f'inference{index}.log'}"
                )
            served_at.append(address)
        print(f"  inference: {', '.join(served_at)} (workers hold no model)",
              file=sys.stderr, flush=True)

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
        if served_at:
            # Round robin. Which worker lands on which server does not affect any answer
            # -- requests are never merged, so a batch gets the same numbers wherever it
            # is served -- only how many threads each process has to interleave.
            command += ["--inference", served_at[worker % len(served_at)],
                        "--inference-arm", "value"]
            if args.baseline:
                command += ["--baseline-inference-arm", "baseline"]
        else:
            command += ["--value", *args.value]
            if args.baseline:
                command += ["--baseline", *args.baseline]
        if args.hide_bench:
            command += ["--hide-bench"]
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
            label="match", counts=written, monitor=monitor,
        )
        if monitor is not None:
            monitor.save()
            state = monitor.state()
            print(
                f"  SPRT: {state['decision'] or 'no decision'} after {state['pairs']} pairs, "
                f"LLR {state['llr']:+.3f}"
                + ("" if state["stoppedEarly"] else " -- ran to --games, so the fixed count "
                   "is the answer")
                + f"\n  {json.dumps(state['counts'])} -> {out_dir / 'sprt.json'}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        # The server outlives every worker and is ours to end, however the run ended --
        # a leftover one holds 1.5 GB of VRAM and answers the next run's questions with
        # the previous run's models.
        for process in servers:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
