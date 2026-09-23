"""IKA-143: does `--baseline-rank-first-completion` reach the other arm in BOTH seats?

Runs `tools/generation_match.py`'s own `main` with `play_game` wrapped: the wrapper records
the `rank_view` pair each game was handed, which seat the tested arm sat in, and what the
record says each side's decisions were ranked from, then plays the real game with a
two-turn cap. A plumbing check, not a measurement: two games, width 4.

    PYTHONPATH=<tree>/src python scratchpad/ika143_seat_plumbing.py <model.pt> <book> \\
        [--first-for tested|baseline|none]
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))

import generation_match  # noqa: E402

SEEN: list[dict] = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("book")
    ap.add_argument("--first-for", choices=("tested", "baseline", "none"), default="baseline")
    args = ap.parse_args()
    real = generation_match.play_game

    def wrapped(*a, **kw):  # noqa: ANN002, ANN003, ANN202
        kw["max_turns"] = 2
        record = real(*a, **kw)
        views = [d.rank_views for d in record.decisions if d.kind == "move"]
        SEEN.append({
            "seat": a[4], "rank_view": kw.get("rank_view"), "record": record.rank_view,
            "move_decisions": len(views), "views": views,
        })
        return record

    generation_match.play_game = wrapped
    argv = [
        "generation_match.py", "--games", "1", "--limit", "4",
        "--value", args.model, "--baseline", args.model, "--hide-bench",
        "--rank-leaf", "--baseline-rank-leaf", "--selection-book", args.book,
        "--device", "cpu", "--max-turns", "2",
    ]
    if args.first_for == "tested":
        argv.append("--rank-first-completion")
    elif args.first_for == "baseline":
        argv.append("--baseline-rank-first-completion")
    sys.argv = argv
    with contextlib.suppress(SystemExit):
        generation_match.main()
    for row in SEEN:
        print(row)


if __name__ == "__main__":
    main()
