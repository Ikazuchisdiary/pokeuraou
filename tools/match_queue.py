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
        --out data/matches/foo --games 848 --workers 8 \\
        --value data/models/a.pt --baseline data/models/b.pt \\
        -- --limit 48 --rank-leaf --baseline-rank-leaf
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
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--games", type=int, default=848, help="games *per seat*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--value", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", default=None)
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

    def build(worker: int, address: str) -> list[str]:
        command = [
            sys.executable, str(ROOT / "tools" / "generation_match.py"),
            "--queue", address,
            "--seed", str(args.seed),
            "--games", str(args.games),
            "--device", args.device,
            "--torch-threads", "1",
            "--value", *args.value,
            "--out", str(out_dir / f"worker{worker}.jsonl"),
            "--games-out", str(out_dir / f"games-worker{worker}.jsonl"),
        ]
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
    raise SystemExit(
        run_workers(
            range(2 * args.games), build,
            workers=args.workers, out_dir=out_dir, env=env, cwd=str(ROOT),
            label="match", counts=written,
        )
    )


if __name__ == "__main__":
    main()
