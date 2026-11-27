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
        --value data/models/a.pt --baseline data/models/b.pt --hide-bench \\
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

# IKA-431: OpenBLAS commits a buffer per thread it may run when it loads -- numpy's copy
# and scipy's, about 0.5 GB each on this 16-thread machine, a gigabyte in every process
# that never touches it. Set before numpy loads, and inherited by the workers, servers and
# port nodes this starts. A worker runs on one core and no answer moves (records/IKA-431.md).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.benchflags import add_bench_flags, bench_argv, require_bench  # noqa: E402
from pokeuraou.sprt import Sprt, StopWhenDecided  # noqa: E402
from pokeuraou.workqueue import run_workers  # noqa: E402


def check_other_arm(args: argparse.Namespace, extra: list[str]) -> None:
    """The other arm is named, never defaulted (IKA-335).

    Without `--baseline` the other arm has no leaf: hp-share. Leaving it out used to mean
    that silently, and IKA-296's null control ran an arm against hp-share instead of
    against itself. So a match names its baseline leaf or says `--baseline-hp-share`.
    """
    named = bool(args.baseline) or any(
        flag in extra for flag in ("--baseline", "--baseline-inference-arm")
    )
    if args.baseline_hp_share and named:
        raise SystemExit(
            "--baseline-hp-share contradicts the baseline leaf also given; one of them is "
            "not what you meant."
        )
    if args.baseline_hp_share and not args.served:
        raise SystemExit(
            "--baseline-hp-share is the inference server's hp-share arm: it needs --served "
            "(IKA-460 step 4)"
        )
    if not args.baseline_hp_share and not named:
        raise SystemExit(
            "no --baseline: the other arm would be hp-share without anyone choosing it "
            "(IKA-296's null control ran against hp-share this way). Pass --baseline with "
            "the other arm's leaf -- the same files as --value for a null control -- or "
            "--baseline-hp-share to mean hp-share (IKA-335)."
        )


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
    ap.add_argument("--baseline", nargs="+", default=None,
                    help="the other arm's leaf. This or --baseline-hp-share is required")
    ap.add_argument(
        "--baseline-hp-share",
        action="store_true",
        help="the other arm has no leaf and you mean it: hp-share (the M-C origin, "
        "passed on to every worker). Leaving --baseline out used to mean this silently, "
        "and IKA-296's null control -- an arm against itself -- ran against hp-share "
        "because its driver left it out (IKA-335).",
    )
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
    add_bench_flags(ap)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument(
        "--port-threads",
        type=int,
        default=None,
        help="threads each worker's port resolves a node's cells on (IKA-32; "
        "POKEURAOU_PORT_THREADS). The games are the same at any count; only the wall clock "
        "moves. For a match shaped like play on a clock (one worker); with many workers the "
        "machine is already full. Default: the environment's, else 1.",
    )
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
    ap.add_argument(
        "--pool",
        required=True,
        help="an M-C match (IKA-259): both seats from this pool (data/pool/<id>.json), "
        "played by tools/pool_match.py. Each arm solves each pair's selection with its own "
        "leaf and believes the opponent's bench from that solve, so there is no book; "
        "--baseline-hp-share makes the other arm hp-share (the M-C origin). The servers load "
        "the pool's vocabulary.",
    )
    ap.add_argument(
        "--q-model",
        type=Path,
        default=None,
        help="the Q a q / q-nocover rank fill ranks by (IKA-274): with --served the servers "
        "load it as the Q arm `q` and every worker gets --q-arm q; otherwise every worker "
        "gets --q-model. Default with --pool: data/models/q-mc6.pt when an arm ranks by the "
        "default Q -- an arm with --rank-leaf and no --rank-fill after -- plays q-nocover, "
        "what M-C generation plays (IKA-338) -- and a stop if that file is not there",
    )
    ap.add_argument(
        "--q-model-named",
        nargs=2,
        action="append",
        default=[],
        metavar=("NAME", "MODEL"),
        help="the Q a q.NAME / q-nocover.NAME rank fill ranks by (IKA-274 stage 3), so two "
        "arms can rank by two Qs: with --served the servers load it as the Q arm NAME and "
        "every worker gets --q-arm-named NAME NAME; otherwise --q-model-named NAME MODEL",
    )
    ap.add_argument("--sprt-alpha", type=float, default=0.05,
                    help="chance of passing a change worth ELO0 or less")
    ap.add_argument("--sprt-beta", type=float, default=0.05,
                    help="chance of failing a change worth ELO1 or more")
    ap.add_argument(
        "--max-worker-failures",
        type=int,
        default=0,
        help="stop once more workers than this have failed (IKA-336); -1 never stops. "
        "Default 0: the first failure stops the board -- the queue would hand the dead "
        "worker's games to the others, so the games would all arrive and the run would "
        "look whole while its clock and CPU per game were wrong (IKA-307).",
    )
    ap.add_argument(
        "--python-worker",
        action="store_true",
        help="IKA-458 stage 3.7: the workers are tools/pool_match.py (Python), as before; the "
        "default is pokeuraou-damage pool-match-worker. For the comparison only",
    )
    ap.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="after --, options passed to every worker unchanged",
    )
    args = ap.parse_args()
    require_bench(args)
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
    pool_regulation = None
    if args.pool is not None:
        from pokeuraou.pool import find_pool

        if "--selection-book" in extra or "--roster" in extra or args.uniform_selection:
            raise SystemExit(
                "--pool has no book, no roster and no uniform switch: each arm solves its "
                "own selection (IKA-259)"
            )
        pool_regulation = json.loads(find_pool(args.pool).read_bytes().decode("utf-8"))[
            "regulation"
        ]

    check_other_arm(args, extra)
    if args.q_model is None and args.pool is not None:
        from pokeuraou import qrank

        # IKA-338: an arm ranks its leaf-ranked menus by the default Q unless the tail names
        # another fill -- the player M-C generation ships. Found here so that the servers
        # hold it and a missing file stops the board before any worker starts.
        if qrank.tail_wants_default_q(extra):
            q_path = qrank.default_q()
            if not q_path.exists():
                raise SystemExit(qrank.missing_q(q_path))
            args.q_model = q_path
    if args.pool is not None and args.baseline_hp_share:
        extra = [*extra, "--baseline-hp-share"]

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
    if args.port_threads is not None:
        if args.port_threads < 1:
            raise SystemExit("--port-threads must be at least 1")
        env["POKEURAOU_PORT_THREADS"] = str(args.port_threads)

    servers: list[subprocess.Popen] = []
    server_logs: list[Path] = []
    served_at: list[str] = []

    def stop_servers() -> None:
        # The server outlives every worker and is ours to end, however the run ended --
        # a leftover one holds 1.5 GB of VRAM and answers the next run's questions with
        # the previous run's models.
        for process in servers:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    def launch(arms: dict[str, list[str]], qs: dict[str, str], tag: str, count: int,
               hp_arms: tuple[str, ...] = ()) -> list[str]:
        """`count` Rust inference servers holding `arms` and the Qs `qs`; their addresses (each
        server's first line of stdout)."""
        server_log = out_dir / "logs"
        server_log.mkdir(parents=True, exist_ok=True)
        found: list[str] = []
        for index in range(count):
            # IKA-461: every arm and every Q in the Rust server.
            from pokeuraou.rustnet import server_command

            command = server_command(pool_regulation, arms, qs, out_dir, hp_arms)
            log = server_log / f"inference{tag}{index}.log"
            process = subprocess.Popen(  # noqa: S603
                command, env=env, cwd=str(ROOT), stdout=subprocess.PIPE,
                stderr=log.open("w", encoding="utf-8"), text=True,
            )
            servers.append(process)
            server_logs.append(log)
            # The address is the server's first line of stdout, read before any worker
            # starts: a worker that starts first has nothing to connect to, and that
            # failure reads as the run dying for a reason of its own.
            assert process.stdout is not None
            address = process.stdout.readline().strip()
            if not address:
                for other in servers:
                    other.terminate()
                raise SystemExit(f"inference server {tag}{index} exited before naming an address; see {log}")
            found.append(address)
        return found

    if args.served:
        for name, _model in args.q_model_named:
            if name == "q" and args.q_model is not None:
                raise SystemExit("--q-model-named q collides with --q-model's server arm q")
        arms = {"value": list(args.value)}
        if args.baseline:
            arms["baseline"] = list(args.baseline)
        qs = {name: model for name, model in args.q_model_named}
        if args.q_model is not None:
            qs["q"] = str(args.q_model)
        if pool_regulation is None or args.device != "cuda":
            raise SystemExit("the inference server is the Rust one: an M-C pool match (--pool) "
                             "on CUDA (IKA-460)")
        served_at = launch(arms, qs, "", args.servers,
                           ("hp-share",) if args.baseline_hp_share else ())
        print(f"  inference: {', '.join(served_at)} (workers hold no model)",
              file=sys.stderr, flush=True)

    def build(worker: int, address: str) -> list[str]:
        command = [
            sys.executable, str(ROOT / "tools" / "pool_match.py"),
            "--pool", args.pool,
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
        if args.q_model is not None:
            command += ["--q-arm", "q"] if served_at else ["--q-model", str(args.q_model)]
        for name, model in args.q_model_named:
            command += (
                ["--q-arm-named", name, name] if served_at else ["--q-model-named", name, model]
            )
        command += bench_argv(args.hide_bench)
        return command + extra

    python_build = build
    if not args.python_worker:
        # IKA-458 stage 3.7: the workers are `pokeuraou-damage pool-match-worker`. The flags are read
        # once, by tools/pool_match.py, into the config every worker reads.
        if not served_at:
            raise SystemExit("the Rust board worker plays the leaves of the inference servers: pass --served")
        from pokeuraou import rustworker

        config = out_dir / "worker-config.json"
        flags = python_build(0, "-")
        at = flags.index("--queue")
        del flags[at : at + 2]
        made = subprocess.run(  # noqa: S603
            [*flags, "--rust-config", str(config)], env=env, cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        )
        if made.returncode != 0:
            stop_servers()
            raise SystemExit(f"the Rust board worker's config was not written:\n{made.stdout}")
        print(f"  workers: pokeuraou-damage {rustworker.BOARD_SUBCOMMAND} (config {config})",
              file=sys.stderr, flush=True)
        baseline_server = (extra[extra.index("--baseline-inference") + 1]
                           if "--baseline-inference" in extra else None)

        def build(worker: int, address: str) -> list[str]:
            return rustworker.board_command(
                config, served_at[worker % len(served_at)],
                out_dir / f"games-worker{worker}.jsonl", out_dir / f"worker{worker}.jsonl",
                baseline_inference=baseline_server, queue=address,
            )

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
    outcome: dict = {}
    try:
        status = run_workers(
            range(2 * args.games), build,
            workers=args.workers, out_dir=out_dir, env=env, cwd=str(ROOT),
            label="match", counts=written, monitor=monitor,
            max_failures=None if args.max_worker_failures < 0 else args.max_worker_failures,
            server_logs=server_logs,
            outcome=outcome,
        )
        if outcome.get("stoppedForFailures"):
            # Before the SPRT's line, which would otherwise read "ran to --games" (IKA-336).
            print(
                f"  the board STOPPED because a worker failed ({outcome['stoppedForFailures']}): "
                f"its count and its test below are of the games before the stop, not an "
                f"answer -> {out_dir / 'workers.json'}",
                file=sys.stderr,
                flush=True,
            )
        if monitor is not None:
            monitor.save()
            state = monitor.state()
            print(
                f"  SPRT: {state['decision'] or 'no decision'} after {state['pairs']} pairs, "
                f"LLR {state['llr']:+.3f}"
                + ("" if state["stoppedEarly"]
                   else " -- stopped by a failed worker, no answer"
                   if outcome.get("stoppedForFailures")
                   else " -- ran to --games, so the fixed count is the answer")
                + f"\n  {json.dumps(state['counts'])} -> {out_dir / 'sprt.json'}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        stop_servers()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
