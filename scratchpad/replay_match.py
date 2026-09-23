"""Is a regenerated game the recorded game, to the bit?

Asked on 2026-09-23 for IKA-104, which changes how a node is computed and claims to change
no answer. IKA-73 settled the same kind of question the same way -- armA's games 0-23
played again on the day's tree -- with a script nobody kept, so this is that check written
down.

    python scratchpad/replay_match.py <recorded pool dir> <replay dir> [--first 0 --games 24]

The replay is `tools/generate_queue.py` at the pool's own seed and settings with
`--first-game` / `--games` naming the games: each game is seeded from `[seed, index]`, so
game k is the same six, the same four and the same random stream whichever worker plays it.
Games are matched by `gameIndex`, never by file order.

Everything a game record holds is compared except two fields that are not the game:
`searchSeconds` (the clock) and `engine` (which tree played it -- the thing being varied).
Compared as the JSON text each side wrote, so a float equal to the last bit is the only
kind that matches, and -0.0 is not 0.0.

A match means nothing without a replay that is made to differ and does: run the same
check on a replay at another width, and every game should come back different in its
`decisions`, not merely in `searchLimit` (which differs by construction).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

IGNORED = ("searchSeconds", "engine")
_INDEX = re.compile(r'"gameIndex":\s*(\d+)\s*}\s*$')


def _text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def recorded(pool: Path, wanted: set[int]) -> dict[int, dict]:
    """The recorded games with those indices, reading each worker file only as far as it
    has to: a worker takes indices from the queue in increasing order, so once a file is
    well past the largest one wanted the rest of it is later games."""
    found: dict[int, dict] = {}
    horizon = max(wanted) + 400
    for path in sorted(pool.glob("games-worker*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = _INDEX.search(line[-64:])
                if match is None:
                    continue
                index = int(match.group(1))
                if index in wanted and index not in found:
                    found[index] = json.loads(line)
                if index > horizon:
                    break
    return found


def replayed(directory: Path) -> dict[int, dict]:
    games: dict[int, dict] = {}
    for path in sorted(directory.glob("games-worker*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                game = json.loads(line)
                games[int(game["gameIndex"])] = game
    return games


def differences(old: dict, new: dict) -> list[str]:
    return [
        key
        for key in sorted(set(old) | set(new))
        if key not in IGNORED and _text(old.get(key)) != _text(new.get(key))
    ]


def first_divergence(old: list[dict], new: list[dict]) -> str:
    for step, (a, b) in enumerate(zip(old, new, strict=False)):
        fields = [
            key for key in sorted(set(a) | set(b)) if _text(a.get(key)) != _text(b.get(key))
        ]
        if fields:
            return f"decision {step} (turn {a.get('turn')}, {a.get('kind')}): {', '.join(fields)}"
    return f"{len(old)} decisions against {len(new)}, equal as far as both go"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path, help="the recorded pool (games-worker*.jsonl)")
    ap.add_argument("replay", type=Path, help="the regenerated games")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--games", type=int, default=24)
    args = ap.parse_args()

    wanted = set(range(args.first, args.first + args.games))
    old = recorded(args.pool, wanted)
    new = replayed(args.replay)
    print(
        f"{len(old)} of {len(wanted)} games found in the record, "
        f"{len(set(new) & wanted)} in the replay"
    )
    same = decided = 0
    engines: set[tuple[str, str]] = set()
    for index in sorted(wanted):
        if index not in old or index not in new:
            print(f"  game {index}: missing from the {'record' if index not in old else 'replay'}")
            continue
        engines.add((_text(old[index].get("engine")), _text(new[index].get("engine"))))
        fields = differences(old[index], new[index])
        if not fields:
            same += 1
            decided += 1
            continue
        if "decisions" not in fields:
            decided += 1
        where = (
            first_divergence(old[index]["decisions"], new[index]["decisions"])
            if "decisions" in fields
            else "decisions identical"
        )
        print(f"  game {index}: differs in {', '.join(fields)} -- {where}")
    print(f"identical in every field but {' and '.join(IGNORED)}: {same}/{len(wanted)}")
    print(f"decisions identical: {decided}/{len(wanted)}")
    for recorded_engine, replay_engine in sorted(engines):
        print(f"  engine  record {recorded_engine}\n          replay {replay_engine}")
    sys.exit(0 if same == len(wanted) else 1)


if __name__ == "__main__":
    main()
