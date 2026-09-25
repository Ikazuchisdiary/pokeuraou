"""Holds a generation run's rank files against its games (IKA-278).

    uv run python tools/rank_scores_check.py data/selfplay-mcN

Each ``games-<name>.jsonl`` is read with its ``rank-<name>.jsonl.gz`` (`rank_scores.path_for`),
one game at a time -- a worker writes both in the same order, so nothing is held but the
current game. For every game: its line is there, and `rank_scores.check_game` finds the
recorded menus in the recorded scores' order, each score the fold of its fills to the bit,
and each completion the one ``rankViews`` names. A rank line whose game is not in the file
(a worker stopped between the two writes) is counted, not an error.

Prints the counts and the size: compressed bytes a game, as the files hold them, and
the rows and candidates a game.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.rank_scores import check_game, iter_file, path_for  # noqa: E402


def _games(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw)


def check_dir(directory: Path, *, show: int = 5) -> dict[str, Any]:
    out: dict[str, Any] = {
        "games": 0, "matched": 0, "missing": 0, "extra": 0, "problems": 0,
        "rankings": 0, "candidates": 0, "fills": 0, "guessed": 0,
        "rank_bytes": 0, "game_bytes": 0, "shown": [],
    }
    for games_path in sorted(directory.glob("games-*.jsonl")):
        rank_path = path_for(games_path)
        out["game_bytes"] += games_path.stat().st_size
        if not rank_path.exists():
            count = sum(1 for _ in _games(games_path))
            out["games"] += count
            out["missing"] += count
            continue
        out["rank_bytes"] += rank_path.stat().st_size
        lines = iter_file(rank_path)
        pending = next(lines, None)
        for game in _games(games_path):
            out["games"] += 1
            # Skip rank lines whose game was never written (a stop between the writes).
            while pending is not None and pending.get("gameIndex") != game.get("gameIndex"):
                out["extra"] += 1
                pending = next(lines, None)
            if pending is None:
                out["missing"] += 1
                continue
            out["matched"] += 1
            problems = check_game(game, pending)
            out["problems"] += len(problems)
            if problems and len(out["shown"]) < show:
                out["shown"].extend(problems[: show - len(out["shown"])])
            for row in pending["rankings"]:
                out["rankings"] += 1
                out["candidates"] += len(row["candidates"])
                out["fills"] += len(row["fills"])
                out["guessed"] += bool(row["completion"] and row["completion"]["slots"])
            pending = next(lines, None)
        out["extra"] += sum(1 for _ in lines) + (pending is not None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    found = check_dir(args.directory)
    games = found["games"] or 1
    rankings = found["rankings"] or 1
    print(f"{found['games']} games, {found['matched']} with their rankings, "
          f"{found['missing']} without, {found['extra']} rank lines without a game")
    print(f"  {found['rankings']} rankings ({found['rankings'] / games:.1f} a game), "
          f"{found['candidates'] / rankings:.1f} candidates and {found['fills'] / rankings:.2f} "
          f"fills a ranking, {found['guessed'] / rankings:.1%} read a guessed bench")
    print(f"  rank files {found['rank_bytes']:,} bytes = {found['rank_bytes'] / games:,.0f} a game; "
          f"games files {found['game_bytes'] / games:,.0f} a game")
    print(f"  problems {found['problems']}")
    for problem in found["shown"]:
        print(f"    {problem}")
    raise SystemExit(1 if found["problems"] or found["missing"] else 0)


if __name__ == "__main__":
    main()
