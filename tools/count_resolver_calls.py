"""Does a production road call Python's resolver? Counted, with resolve.py imported.

IKA-209 moved generation, the analyser and the selection solve onto the port alone. That is
a claim about what runs, so this runs them and counts every call into a function whose code
lives in `src/pokeuraou/resolve.py` -- `sys.monitoring` on exactly those code objects, so a
call anywhere, from any module, by any name, is seen, and nothing else is slowed down.

resolve.py is imported first, on purpose: the claim is not "nobody imports it" but "nobody
calls it", and a module that is loaded and wired into nothing is the case to rule out.

    python tools/count_resolver_calls.py                  # every road, a few games each
    python tools/count_resolver_calls.py --roads cli,generation-hp --games 4

A positive control goes first: one `resolve_turn` through the same counter must be seen,
or the counter is not counting and every zero after it means nothing.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import tempfile
import time
import types
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pokeuraou.resolve as resolve  # noqa: E402 - loaded before any road, on purpose

TOOL = sys.monitoring.PROFILER_ID
CALLS: Counter[str] = Counter()


def resolver_codes() -> list[types.CodeType]:
    """Every code object defined in resolve.py: functions, methods, nested ones, lambdas."""
    path = Path(resolve.__file__).resolve()
    found: dict[int, types.CodeType] = {}

    def add(code: types.CodeType) -> None:
        if Path(code.co_filename).resolve() != path or id(code) in found:
            return
        found[id(code)] = code
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                add(const)

    def visit(obj: object, seen: set[int]) -> None:
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, (staticmethod, classmethod)):
            obj = obj.__func__
        if isinstance(obj, property):
            for part in (obj.fget, obj.fset, obj.fdel):
                if part is not None:
                    visit(part, seen)
            return
        code = getattr(obj, "__code__", None)
        if isinstance(code, types.CodeType):
            add(code)
        if isinstance(obj, type) and obj.__module__ == resolve.__name__:
            for member in vars(obj).values():
                visit(member, seen)

    seen: set[int] = set()
    for value in vars(resolve).values():
        visit(value, seen)
    return list(found.values())


def start() -> int:
    codes = resolver_codes()
    sys.monitoring.use_tool_id(TOOL, "count_resolver_calls")

    def on_start(code: types.CodeType, _offset: int) -> None:
        CALLS[code.co_qualname] += 1

    sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, on_start)
    for code in codes:
        sys.monitoring.set_local_events(TOOL, code, sys.monitoring.events.PY_START)
    return len(codes)


def run_tool(script: str, argv: list[str]) -> None:
    saved = sys.argv
    sys.argv = [script, *argv]
    try:
        runpy.run_path(str(ROOT / "tools" / script), run_name="__main__")
    finally:
        sys.argv = saved


def control() -> None:
    """One Python resolve through the counter: it has to be seen."""
    from pokeuraou.actions import SideAction
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.narrow import narrow
    from pokeuraou.pool import load_pool
    from pokeuraou.selfplay import position_from_sets

    pool = load_pool("regmc-matchupweb")
    reg = pool.reg
    register_mega_stones(reg)
    first, second = pool.teams[0], pool.teams[1]
    own = list(getattr(first, "sets", first))[:4]
    foe = list(getattr(second, "sets", second))[:4]
    pos = position_from_sets(reg, own, foe)
    actions: list[SideAction] = [narrow(reg, pos, s, limit=1).actions[0] for s in (0, 1)]
    resolve.resolve_turn(reg, pos, actions, budget=resolve.Budget.matrix())


def roads(args: argparse.Namespace, work: Path) -> dict[str, callable]:
    games = str(args.games)
    value = str(args.value)
    return {
        # M-C generation as it ships: pool against pool, hidden bench, the selection solved
        # with the learned leaf at the start of each game (the selection road).
        "generation-value": lambda: run_tool(
            "selfplay.py",
            ["--pool", "regmc-matchupweb", "--value", value, "--device", "cpu",
             "--games", games, "--seed", "7", "--out", str(work / "gv" / "g.jsonl")],
        ),
        # hp-share, the selection drawn uniformly (the solve needs a leaf).
        "generation-hp": lambda: run_tool(
            "selfplay.py",
            ["--pool", "regmc-matchupweb", "--uniform-selection", "--games", games,
             "--seed", "7", "--out", str(work / "gh" / "g.jsonl")],
        ),
        # The roster path (M-B), hidden bench and open, and depth 2 in the open game.
        "generation-roster-hidden": lambda: run_tool(
            "selfplay.py",
            ["--hide-bench", "--games", games, "--seed", "5", "--out", str(work / "rh.jsonl")],
        ),
        "generation-roster-depth2": lambda: run_tool(
            "selfplay.py",
            ["--open-bench", "--depth", "2", "--games", games, "--seed", "5",
             "--out", str(work / "rd.jsonl")],
        ),
        # The analyser, both examples, with the learned leaf and without.
        "cli": lambda: [
            __import__("pokeuraou.cli", fromlist=["main"]).main([str(ROOT / "examples" / name), "--json"])
            for name in ("scenario-turn1.json", "scenario-turn5.json")
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--value", type=Path, default=ROOT / "data" / "models" / "value-gen11L.pt")
    ap.add_argument("--roads", default=None, help="comma-separated; default all")
    args = ap.parse_args()

    watched = start()
    print(f"resolve.py: {watched} code objects watched", file=sys.stderr)
    CALLS.clear()
    control()
    control_calls = sum(CALLS.values())
    print(f"control: one resolve_turn -> {control_calls} calls into resolve.py "
          f"({CALLS['resolve_turn']} of them resolve_turn)", file=sys.stderr)
    if control_calls == 0 or CALLS["resolve_turn"] != 1:
        raise SystemExit("the counter did not see the control; nothing below would mean anything")

    results: list[tuple[str, int, float, Counter[str]]] = []
    with tempfile.TemporaryDirectory(prefix="count_resolver_") as scratch:
        work = Path(scratch)
        chosen = roads(args, work)
        names = list(chosen) if args.roads is None else args.roads.split(",")
        for name in names:
            CALLS.clear()
            started = time.perf_counter()
            chosen[name]()
            results.append((name, sum(CALLS.values()), time.perf_counter() - started, Counter(CALLS)))
    print()
    print(f"control   {control_calls:>8} calls into resolve.py (must be > 0)")
    for name, total, seconds, counted in results:
        top = ", ".join(f"{fn} {n}" for fn, n in counted.most_common(5))
        print(f"{name:<26} {total:>8} calls into resolve.py  ({seconds:.1f} s){'  ' + top if top else ''}")


if __name__ == "__main__":
    main()
