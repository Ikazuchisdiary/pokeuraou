"""Is a generated game the same game after a change to the roads it is played through?

Plays one generation twice -- `tools/selfplay.py` with the same flags and seed, once from
another checkout (`--before`, e.g. the last version that resolved in Python) and once from
this one -- through the same Rust binary, and compares the records line for line. A record
is compared whole, as the JSON the generation run writes, less the two fields that are not
the game: `searchSeconds` (the clock) and `engine` (which code wrote it).

    python tools/diff_generation.py --before C:/tmp/ikaNNN/before --games 20 --seed 3 \\
        -- --pool regmc-matchupweb --uniform-selection
    python tools/diff_generation.py --before C:/tmp/ikaNNN/before --games 4 --seed 3 \\
        -- --pool regmc-matchupweb --value data/models/value-gen11L.pt --device cpu

Everything after `--` goes to `tools/selfplay.py` as it is (not `--games`, `--seed` or
`--out`, which this sets). The decisions are counted by kind, so a sample in which no turn
paused for a replacement says so instead of passing for one that did.

Until IKA-209 this compared a game played with `POKEURAOU_RUST_NODE=0` against the same
game through the port. The production roads have no Python resolver any more, so the
comparison is between checkouts instead: the one before a change and the one after it.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The fields of a record that are not the game.
NOT_THE_GAME = ("searchSeconds", "engine")


def default_binary() -> Path:
    name = "pokeuraou-damage.exe" if sys.platform == "win32" else "pokeuraou-damage"
    return ROOT / "rust" / "target" / "release" / name


def play(checkout: Path, args: argparse.Namespace, out: Path, binary: Path) -> float:
    """One run of `checkout`'s own `tools/selfplay.py`; its seconds."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(checkout / "src")
    env["POKEURAOU_RUST_NODE"] = "1"
    env["POKEURAOU_RUST_NODE_BIN"] = str(binary)
    env.setdefault("PYTHONHASHSEED", "0")
    command = [
        sys.executable,
        str(checkout / "tools" / "selfplay.py"),
        *args.selfplay,
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--out",
        str(out),
    ]
    # A fresh directory: a pool run keeps its selection solves beside `--out`, and a run
    # that read the last run's solves would not have solved anything.
    if out.parent.exists():
        shutil.rmtree(out.parent)
    out.parent.mkdir(parents=True)
    started = time.perf_counter()
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    seconds = time.perf_counter() - started
    if done.returncode != 0:
        sys.stderr.write(done.stdout + done.stderr)
        raise SystemExit(f"{checkout}: tools/selfplay.py exited {done.returncode}")
    return seconds


def the_game(line: str) -> str:
    record = json.loads(line)
    for key in NOT_THE_GAME:
        record.pop(key, None)
    return json.dumps(record, sort_keys=True, ensure_ascii=False)


def first_difference(before: dict, after: dict) -> str:
    """Where two records part, in the words of the game."""
    for step, (a, b) in enumerate(zip(before["decisions"], after["decisions"], strict=False)):
        if a != b:
            fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            return (
                f"decision {step} (turn {a.get('turn')}, {a.get('kind')}/{b.get('kind')}) "
                f"differs in {', '.join(fields)}: chose {a.get('ownChosen')!r}/"
                f"{a.get('foeChosen')!r} before, {b.get('ownChosen')!r}/{b.get('foeChosen')!r} after"
            )
    if len(before["decisions"]) != len(after["decisions"]):
        return f"{len(before['decisions'])} decisions before, {len(after['decisions'])} after"
    fields = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    return f"the decisions agree; the record differs in {', '.join(fields)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--before", type=Path, required=True, help="the other checkout")
    ap.add_argument(
        "--after", type=Path, default=ROOT, help="the checkout to compare (default: this one)"
    )
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--binary", type=Path, default=None, help="the Rust node both runs use (default: this "
        "checkout's rust/target/release)"
    )
    ap.add_argument("--work", type=Path, required=True, help="a directory for the two runs")
    ap.add_argument("selfplay", nargs=argparse.REMAINDER, help="-- then tools/selfplay.py's flags")
    args = ap.parse_args()
    if args.selfplay[:1] == ["--"]:
        args.selfplay = args.selfplay[1:]
    binary = args.binary or default_binary()
    if not binary.exists():
        raise SystemExit(f"no Rust binary at {binary}")
    args.work.mkdir(parents=True, exist_ok=True)

    seconds = {}
    lines = {}
    for name, checkout in (("before", args.before), ("after", args.after)):
        out = args.work / name / "games.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        seconds[name] = play(checkout.resolve(), args, out, binary)
        lines[name] = out.read_text(encoding="utf-8").splitlines()

    before, after = lines["before"], lines["after"]
    print(f"binary {binary}")
    print(f"before {args.before}  ({seconds['before']:.1f} s)")
    print(f"after  {args.after}  ({seconds['after']:.1f} s)")
    print(f"selfplay {' '.join(args.selfplay)} --games {args.games} --seed {args.seed}")
    if len(before) != len(after):
        print(f"  {len(before)} records before, {len(after)} after")
    kinds: Counter[str] = Counter()
    games_with: Counter[str] = Counter()
    same = 0
    differing: list[int] = []
    for index, (a, b) in enumerate(zip(before, after, strict=False)):
        record = json.loads(a)
        found = Counter(d["kind"] for d in record["decisions"])
        kinds.update(found)
        for kind in found:
            games_with[kind] += 1
        if the_game(a) == the_game(b):
            same += 1
        else:
            differing.append(index)
    count = min(len(before), len(after))
    print(f"  identical games {same}/{count}")
    print(
        "  decisions in the sample (before): "
        + ", ".join(f"{kind} {kinds[kind]} in {games_with[kind]} games" for kind in sorted(kinds))
    )
    for index in differing:
        print(
            f"  game {index}: "
            + first_difference(json.loads(before[index]), json.loads(after[index]))
        )


if __name__ == "__main__":
    main()
