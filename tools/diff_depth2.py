"""Does depth 2 decide the same on recorded positions after a change to its roads?

Solves recorded move positions with `search(depth=2)` -- the shipped reading and, with
`--restricted`, the restricted one -- under two checkouts, the one before a change
(`--before`) and this one, and compares the answers to the bit: the value, both mixtures,
how many cells were refined, the notes. Each checkout is its own process and solves every
position once, spread over `--jobs` workers.

    python tools/diff_depth2.py --before C:/tmp/ikaNNN/before --games-dir C:/tmp/ika77/out/g600 \\
        --positions 60 --jobs 8 --work C:/tmp/ikaNNN/depth2

IKA-209 moved `search._refined_value` from Python's `resolve_turn` to the port's `turn`
command; this is what says the decisions did not move.

`--null` solves this checkout twice, with `--jobs 1` and with `--jobs`, and the two must
agree to the byte. `--after-sub-branches K` changes this checkout's `sub_branches`, which
must show up as differences -- the positive control that the comparison can see a change.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_binary() -> Path:
    name = "pokeuraou-damage.exe" if sys.platform == "win32" else "pokeuraou-damage"
    return ROOT / "rust" / "target" / "release" / name


# -- The solving side: runs inside one checkout's src.

_STATE: dict = {}


def _init(src: str, limit: int, sub_branches: int | None) -> None:
    sys.path.insert(0, src)
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.pool import load_pool

    reg = load_pool("regmc-matchupweb").reg
    register_mega_stones(reg)
    _STATE.update(reg=reg, limit=limit, sub_branches=sub_branches)


def _solve(line: str) -> str:
    from pokeuraou.narrow import narrow
    from pokeuraou.payoff import HP_SHARE
    from pokeuraou.position import Position
    from pokeuraou.resolve import Budget  # re-exported after IKA-209; loads, calls nothing
    from pokeuraou.search import DEFAULT_SUB_BRANCHES, search

    reg, limit = _STATE["reg"], _STATE["limit"]
    sub = _STATE["sub_branches"] or DEFAULT_SUB_BRANCHES
    pos = Position.from_json(json.loads(line))
    row = narrow(reg, pos, 0, limit=limit).actions
    col = narrow(reg, pos, 1, limit=limit).actions
    out = {}
    for name, restricted in (("shipped", False), ("restricted", True)):
        result = search(
            reg, pos, row, col, HP_SHARE.batch, budget=Budget.matrix(), depth=2,
            solve_restricted=restricted, sub_branches=sub,
        )
        eq = result.equilibrium
        out[name] = {
            "value": float(eq.value).hex(),
            "row": [float(x).hex() for x in eq.row_strategy],
            "col": [float(x).hex() for x in eq.col_strategy],
            "refined": result.refined,
            "subgames": result.subgames,
            "unmodelled": sorted(result.unmodelled),
        }
    return json.dumps(out, sort_keys=True)


def solve_main(args: argparse.Namespace) -> None:
    import multiprocessing as mp

    lines = Path(args.positions_file).read_text(encoding="utf-8").splitlines()
    init = (args.src, args.limit, args.sub_branches)
    if args.jobs <= 1:
        _init(*init)
        answers = [_solve(line) for line in lines]
    else:
        with mp.get_context("spawn").Pool(args.jobs, initializer=_init, initargs=init) as pool:
            answers = pool.map(_solve, lines, chunksize=1)
    Path(args.out).write_text("\n".join(answers) + "\n", encoding="utf-8")


# -- The comparing side.


def pick_positions(games_dir: Path, count: int) -> list[str]:
    """Every move decision's position in file order, then `count` of them evenly spaced."""
    found: list[str] = []
    for path in sorted(games_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            for decision in json.loads(line)["decisions"]:
                if decision["kind"] == "move":
                    found.append(json.dumps(decision["position"], sort_keys=True))
    step = max(1, len(found) // count)
    return found[::step][:count]


def run(checkout: Path, args: argparse.Namespace, positions: Path, out: Path, jobs: int,
        sub_branches: int | None = None) -> float:
    env = dict(os.environ)
    env["POKEURAOU_RUST_NODE"] = "1"
    env["POKEURAOU_RUST_NODE_BIN"] = str(args.binary or default_binary())
    env.setdefault("PYTHONHASHSEED", "0")
    command = [
        sys.executable, str(Path(__file__).resolve()), "--solve",
        "--src", str(checkout / "src"), "--positions-file", str(positions), "--out", str(out),
        "--jobs", str(jobs), "--limit", str(args.limit),
    ]
    if sub_branches is not None:
        command += ["--sub-branches", str(sub_branches)]
    started = time.perf_counter()
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        sys.stderr.write(done.stdout + done.stderr)
        raise SystemExit(f"{checkout}: the solve exited {done.returncode}")
    return time.perf_counter() - started


def compare(a: Path, b: Path) -> tuple[int, int, list[str]]:
    left = a.read_text(encoding="utf-8").splitlines()
    right = b.read_text(encoding="utf-8").splitlines()
    notes = []
    same = 0
    for index, (x, y) in enumerate(zip(left, right, strict=True)):
        if x == y:
            same += 1
            continue
        dx, dy = json.loads(x), json.loads(y)
        parts = [
            f"{mode}.{key}"
            for mode in dx
            for key in dx[mode]
            if dx[mode][key] != dy[mode][key]
        ]
        notes.append(f"  position {index}: {', '.join(parts)}")
    return same, len(left), notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--solve", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--src", help=argparse.SUPPRESS)
    ap.add_argument("--positions-file", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    ap.add_argument("--sub-branches", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--before", type=Path)
    ap.add_argument("--games-dir", type=Path)
    ap.add_argument("--positions", type=int, default=60)
    ap.add_argument("--limit", type=int, default=8, help="candidates a side (M-C searches 8)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--binary", type=Path, default=None)
    ap.add_argument("--null", action="store_true", help="also solve this checkout with --jobs 1")
    ap.add_argument("--after-sub-branches", type=int, default=None)
    args = ap.parse_args()
    if args.solve:
        solve_main(args)
        return
    if args.before is None or args.games_dir is None or args.work is None:
        ap.error("--before, --games-dir and --work are required")
    args.work.mkdir(parents=True, exist_ok=True)
    positions = args.work / "positions.jsonl"
    picked = pick_positions(args.games_dir, args.positions)
    positions.write_text("\n".join(picked) + "\n", encoding="utf-8")
    print(f"{len(picked)} recorded move positions from {args.games_dir}, limit {args.limit}")

    before_out, after_out = args.work / "before.jsonl", args.work / "after.jsonl"
    seconds = run(args.before.resolve(), args, positions, before_out, args.jobs)
    print(f"before {args.before} ({seconds:.1f} s, --jobs {args.jobs})")
    seconds = run(ROOT, args, positions, after_out, args.jobs, args.after_sub_branches)
    label = "" if args.after_sub_branches is None else f", sub_branches {args.after_sub_branches}"
    print(f"after  {ROOT} ({seconds:.1f} s, --jobs {args.jobs}{label})")
    same, total, notes = compare(before_out, after_out)
    print(f"  the same answer, to the bit, on {same}/{total} positions")
    for line in notes[:20]:
        print(line)
    if args.null:
        serial = args.work / "after-jobs1.jsonl"
        seconds = run(ROOT, args, positions, serial, 1, args.after_sub_branches)
        same_null = serial.read_bytes() == after_out.read_bytes()
        print(f"null: --jobs 1 ({seconds:.1f} s) and --jobs {args.jobs} byte-identical: {same_null}")
    kinds: dict[str, int] = {}
    for line in after_out.read_text(encoding="utf-8").splitlines():
        answer = json.loads(line)
        kinds["refined cells"] = kinds.get("refined cells", 0) + answer["shipped"]["refined"]
        kinds["subgames"] = kinds.get("subgames", 0) + answer["shipped"]["subgames"]
    print(f"  shipped reading, after: {kinds}")


if __name__ == "__main__":
    main()
