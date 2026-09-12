"""Generation with the games in one queue instead of dealt out in blocks.

`generate_parallel.sh` gives every worker a fixed block, so a run ends when the unluckiest
worker does, and the cost is measured rather than supposed: from the real logs, idle cores
were 1.4% of gen2345, 16.4% of gen6, and 10.5% of gen7 once its two dead workers are set
aside, the last worker finishing 28 minutes after the first.

The dead workers are the other half. gen7's seeds 7108 and 7112 stopped after 126 and 50
games of 1,098 -- one of them with `the Rust node sent 0 of 64219920 bytes` in its log --
and their remaining blocks died with them, about 2,100 games that nobody noticed were
missing until the files were counted. Here a worker's unplayed games are still in the
queue, and the game it was holding goes back to the front when its socket closes.

What a worker plays is decided by the queue, so what a game *is* must not be. Each game is
seeded from its own number, which makes a run reproducible again -- `--first-game` and
`--games` name exactly the same games every time, whatever the machine does with them --
and makes a retry after a death a retry of the game that was lost.

    uv run --group learn python tools/generate_queue.py --out data/selfplay-gen8 \\
        --games 12000 --workers 8 --value data/models/value-all.pt --limit 48

Everything after `--` goes to the workers untouched, for options this does not name:

    ... --workers 8 -- --mirror-share 0.1 --rank-leaf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.workqueue import WorkQueue, serve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="directory for the games")
    ap.add_argument("--games", type=int, required=True, help="games for the whole run")
    ap.add_argument("--first-game", type=int, default=0, help="first game number")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1, help="seeds the run, not a worker")
    ap.add_argument("--value", default=None)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="after --, options passed to every worker unchanged",
    )
    args = ap.parse_args()

    extra = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    out_dir: Path = args.out
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    numbers = range(args.first_game, args.first_game + args.games)
    queue = WorkQueue(numbers)
    returned: list[int] = []

    def note_return(indices: list[int]) -> None:
        returned.extend(indices)
        print(
            f"  a worker went away holding {len(indices)} game(s); back on the queue",
            file=sys.stderr,
            flush=True,
        )

    server, address = serve(queue, on_return=note_return)

    env = dict(os.environ)
    env["POKEURAOU_RUST_NODE"] = "0" if args.no_bridge else "1"
    env["PYTHONPATH"] = str(ROOT / "src")

    print(
        f"generation: {args.workers} workers sharing {args.games} games "
        f"({numbers.start}..{numbers.stop - 1}) -> {out_dir}\n"
        f"  queue at {address}, run seed {args.seed}, search {args.limit}, "
        f"leaf {args.value or 'hp-share'} on {args.device}",
        file=sys.stderr,
        flush=True,
    )

    started = time.perf_counter()
    workers = []
    for worker in range(args.workers):
        command = [
            sys.executable, str(ROOT / "tools" / "selfplay.py"),
            "--queue", address,
            "--seed", str(args.seed),
            "--limit", str(args.limit),
            "--device", args.device,
            "--torch-threads", "1",
            "--out", str(out_dir / f"games-worker{worker}.jsonl"),
        ]
        if args.value:
            command += ["--value", args.value]
        command += extra
        log = (out_dir / "logs" / f"worker{worker}.log").open("w", encoding="utf-8")
        workers.append(
            (
                worker,
                subprocess.Popen(  # noqa: S603
                    command, env=env, cwd=str(ROOT), stdout=log, stderr=log, text=True
                ),
                log,
            )
        )

    # A worker that exits before the queue is empty is a worker that died. Saying so while
    # the run is going is the whole difference from the old script, which reported the
    # count of failures after everything had already finished.
    finished_at: dict[int, float] = {}

    def watch(worker: int, process: subprocess.Popen) -> None:
        process.wait()
        finished_at[worker] = time.perf_counter() - started
        # Work nobody holds. A worker leaving while others still hold games has simply
        # run out of queue, which is what is supposed to happen at the end of a run.
        left = queue.pending
        if process.returncode != 0 or left:
            print(
                f"  worker {worker} exited with {process.returncode} after "
                f"{finished_at[worker]:.0f}s, {left} game(s) still unclaimed",
                file=sys.stderr,
                flush=True,
            )

    threads = [threading.Thread(target=watch, args=(w, p)) for w, p, _ in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for _worker, _process, log in workers:
        log.close()
    server.shutdown()

    elapsed = time.perf_counter() - started
    played, missing = len(queue.done), queue.remaining
    order = sorted(finished_at.values())
    idle = sum(order[-1] - t for t in order) if order else 0.0
    print(
        f"done in {elapsed / 60:.1f}m: {played} games played"
        + (f", {missing} left unplayed" if missing else "")
        + (f", {len(queue.abandoned)} abandoned after "
           f"{len(queue.abandoned) and 'repeated failures'}" if queue.abandoned else "")
        + (f", {len(returned)} replayed after a worker went away" if returned else "")
    )
    if order:
        print(
            f"  workers finished {order[-1] - order[0]:.0f}s apart, "
            f"{idle / (len(order) * elapsed):.1%} of the machine idle at the end"
        )
    raise SystemExit(1 if missing or queue.abandoned else 0)


if __name__ == "__main__":
    main()
