"""IKA-98's negative control: do two generation runs hold the same games, decision for decision?

One run with `POKEURAOU_TIMING` set and one without, same seed and same games. A game is
seeded from its own index, so which worker played it does not matter; games are matched
by `gameIndex` and compared whole, less `engine` (the commit and the dirty flag, which
say which checkout ran) and `searchSeconds` (a wall clock); neither is a decision.

The positive control is built in: the same comparison between game i and game i+1 of the
first run must find differences, or a "no difference" below would mean the comparison
cannot see one.

    python scratchpad/ika98_same_games.py <games dir A> <games dir B>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def games(directory: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in sorted(directory.glob("games-worker*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    record.pop("engine", None)
                    record.pop("searchSeconds", None)
                    out[int(record["gameIndex"])] = record
    return out


def differs(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """The top-level fields that differ; for `decisions`, the first index that does."""
    found = []
    for key in sorted(set(a) | set(b)):
        if key in ("gameIndex",):
            continue
        if a.get(key) != b.get(key):
            if key == "decisions":
                da, db = a.get(key) or [], b.get(key) or []
                first = next(
                    (i for i, (x, y) in enumerate(zip(da, db, strict=False)) if x != y),
                    min(len(da), len(db)),
                )
                found.append(f"decisions (first at {first}; {len(da)} vs {len(db)})")
            else:
                found.append(key)
    return found


a_dir, b_dir = Path(sys.argv[1]), Path(sys.argv[2])
a, b = games(a_dir), games(b_dir)
print(f"A {a_dir}: {len(a)} games, {sum(len(g['decisions']) for g in a.values())} decisions")
print(f"B {b_dir}: {len(b)} games, {sum(len(g['decisions']) for g in b.values())} decisions")
only = sorted(set(a) ^ set(b))
if only:
    print(f"games in one run only: {only}")
same = 0
for index in sorted(set(a) & set(b)):
    found = differs(a[index], b[index])
    if found:
        print(f"  game {index}: {', '.join(found)}")
    else:
        same += 1
print(f"identical: {same} of {len(set(a) & set(b))} shared games")

keys = sorted(a)
shifted = sum(1 for x, y in zip(keys, keys[1:], strict=False) if differs(a[x], a[y]))
print(f"positive control: game i against game i+1 of A differs in {shifted} of "
      f"{max(len(keys) - 1, 0)} pairs")
sys.exit(0 if same == len(set(a) & set(b)) and not only and shifted == len(keys) - 1 else 1)
