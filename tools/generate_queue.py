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
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["POKEURAOU_RUST_NODE"] = "0" if args.no_bridge else "1"
    env["PYTHONPATH"] = str(ROOT / "src")

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
        if args.value:
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
    raise SystemExit(
        run_workers(
            numbers, build,
            workers=args.workers, out_dir=out_dir, env=env, cwd=str(ROOT),
            label="generation", counts=written,
        )
    )


if __name__ == "__main__":
    main()
