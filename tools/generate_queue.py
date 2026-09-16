"""Generation with the games in one queue instead of dealt out in blocks.

`generate_parallel.sh` gives every worker a fixed block, so a run ends when the unluckiest
worker does, and the cost is measured rather than supposed: from the real logs, idle cores
were 1.4% of gen2345, 16.4% of gen6, and 10.5% of gen7 once its two dead workers are set
aside, the last worker finishing 28 minutes after the first. Generation 8 was 25%.

The dead workers are the other half. gen7's seeds 7108 and 7112 stopped after 126 and 50
games of 1,098 -- one of them with `the Rust node sent 0 of 64219920 bytes` in its log --
and their remaining blocks died with them, about 2,100 games that nobody noticed were
missing until the files were counted. Here a worker's unplayed games are still in the
queue, and the game it was holding goes back to the front when its socket closes.

What a worker plays is decided by the queue, so what a game *is* must not be. Each game is
seeded from its own number, which makes a run reproducible again -- `--first-game` and
`--games` name exactly the same games every time, whatever the machine does with them --
and makes a retry after a death a retry of the game that was lost.

The driver itself lives in `pokeuraou.workqueue`, shared with `tools/match_queue.py`.

    uv run --group learn python tools/generate_queue.py --out data/selfplay-gen9 \\
        --games 12000 --workers 8 --value data/models/value-all.pt --limit 48

Everything after `--` goes to the workers untouched, for options this does not name:

    ... --workers 8 -- --mirror-share 0.1 --rank-leaf
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
    ap.add_argument("--out", type=Path, required=True, help="directory for the games")
    ap.add_argument("--games", type=int, required=True, help="games for the whole run")
    ap.add_argument("--first-game", type=int, default=0, help="first game number")
    # Resolved after parsing, because the right number depends on where the leaf
    # lives: 8 when every worker carries one, 24 when they are served.
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1, help="seeds the run, not a worker")
    ap.add_argument("--value", default=None)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    # 24, swept on the board against 48 with the same model on both sides: -1.3 [-3.7,
    # +1.1] at 2.03x the speed, while 16 is -5.6 and 12 is -7.5. Only 24 sits inside the
    # band, so it is a floor rather than a direction, and the doubling it buys is the one
    # thing that has reliably moved the leaf.
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument(
        "--served",
        action="store_true",
        help="run the leaf in inference servers instead of in every worker. A generation "
        "worker then holds no torch and no CUDA context -- 224 MB against about 1.7 GB -- "
        "so the worker count stops being a memory question. Measured on matches, which "
        "carry two leaves: 16 workers over 2 servers ran 116.2 games/min against 82.5 for "
        "the best direct configuration. Generation carries one leaf, so the memory saved "
        "is smaller and the contention is the same; it is worth measuring here rather "
        "than assuming the same 1.41x.",
    )
    ap.add_argument(
        "--servers",
        type=int,
        default=2,
        help="inference server processes, workers dealt round robin. Two, not one: the "
        "same sixteen workers ran 55.5 games/min through one server and 116.2 through "
        "two, because one Python process serialises the JSON, the buffer views and the "
        "result even though the arithmetic releases the GIL. One server is the only "
        "configuration that loses to not serving at all.",
    )
    ap.add_argument("--no-bridge", action="store_true")
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
        # flattens and the CPU is pegged. Direct stays at 8 because a direct worker is
        # 4.0 GB and eight of them did not fit in 31.1.
        args.workers = 24 if getattr(args, "served", False) else 8

    extra = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["POKEURAOU_RUST_NODE"] = "0" if args.no_bridge else "1"
    env["PYTHONPATH"] = str(ROOT / "src")

    servers: list[subprocess.Popen] = []
    served_at: list[str] = []
    if args.served:
        if not args.value:
            raise SystemExit("--served needs --value: the servers hold the leaf")
        server_log = out_dir / "logs"
        server_log.mkdir(parents=True, exist_ok=True)
        for index in range(args.servers):
            errors = (server_log / f"inference{index}.log").open("w", encoding="utf-8")
            process = subprocess.Popen(  # noqa: S603
                [sys.executable, str(ROOT / "tools" / "inference_server.py"),
                 "--device", args.device, "--arm", "value", args.value],
                env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=errors, text=True,
            )
            servers.append(process)
            # Read before any worker starts: one that starts first has nothing to connect
            # to, and that failure reads as the run dying for a reason of its own.
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

    # An index is a game. Generation has no seats to keep paired, so this is the simple
    # case of what `match_queue.py` does.
    numbers = range(args.first_game, args.first_game + args.games)

    def build(worker: int, address: str) -> list[str]:
        command = [
            sys.executable, str(ROOT / "tools" / "selfplay.py"),
            "--queue", address,
            "--seed", str(args.seed),
            "--limit", str(args.limit),
            "--device", args.device,
            "--torch-threads", "1",
            "--out", str(out_dir / f"games-worker{worker}.jsonl"),
        ]
        if served_at:
            # Which worker lands on which server changes no answer: requests are never
            # merged, so a batch gets the same numbers wherever it is served.
            command += ["--inference", served_at[worker % len(served_at)],
                        "--inference-arm", "value"]
        elif args.value:
            command += ["--value", args.value]
        return command + extra

    def written() -> int:
        return sum(
            1
            for path in out_dir.glob("games-worker*.jsonl")
            for line in path.open(encoding="utf-8")
            if line.strip()
        )

    print(
        f"generation: {args.workers} workers sharing {args.games} games "
        f"({numbers.start}..{numbers.stop - 1}) -> {out_dir}\n"
        f"  run seed {args.seed}, search {args.limit}, "
        f"leaf {args.value or 'hp-share'} on {args.device}",
        file=sys.stderr,
        flush=True,
    )
    try:
        status = run_workers(
            numbers, build,
            workers=args.workers, out_dir=out_dir, env=env, cwd=str(ROOT),
            label="generation", counts=written,
        )
    finally:
        # However the run ended. A leftover server holds VRAM and answers the next run's
        # questions with the previous run's model, so ending it is not conditional on
        # anything having gone well.
        for process in servers:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
