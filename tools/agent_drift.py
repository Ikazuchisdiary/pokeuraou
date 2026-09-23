"""Which tools build a different agent than generation does.

Generation calls `play_game` with a leaf, a width, a narrowing order, a depth, a policy, a
sparsity flag and the sheets that hide the bench. A tool that omits one of those is not
measuring the agent that ships -- it is measuring a different one and labelling the
difference as whatever it set out to study.

That is not hypothetical. On 2026-09-19 `selection_check` was found never to pass
`evaluate`, so every board number it had ever produced came from games the hp-share
heuristic played while the value function's claim sat beside them; it voided G2's recorded
-21.8 point miss, a six-team calibration panel and a 0-for-30 that looked like a
value-function collapse. `book_check` had the same gap on `rank_by_leaf`. The tool that
stayed correct, `generation_match`, is also the only one that wrote its agent into each
row's provenance -- writing it down is what makes the drift visible.

So this reads the call sites and prints the difference, rather than trusting anyone to
notice. Parsed with `ast`, because a grep for keyword names finds them in docstrings and
misses the ones spelled across a line break.

    uv run python tools/agent_drift.py            # the report
    uv run python tools/agent_drift.py --check    # and fail if the set below moved

Printing it was never enough on its own: this file existed on the day `selection_check`
was found, and it had not been run. `--check` is what CI calls, and it compares the
drifted set against `KNOWN_DRIFT` below rather than requiring it to be empty -- seven
tools drift today and fixing them is not this check's job. It fails in *both* directions:
a new tool that drifts is the thing to catch, and a tool that stopped drifting has to be
struck off the list, or the list stops describing anything.
"""

from __future__ import annotations

import argparse
import ast
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The arguments that decide which agent plays. `objective` is not here: it names the
#: payoff, not the player, and every caller passes it.
AGENT_ARGS = (
    "evaluate",
    "search_limit",
    "rank_by_leaf",
    "depth",
    "policy",
    "solve_sparsely",
    "solve_restricted",
    "sheets",
    # What the search believes the hidden bench holds. Generation has passed it since
    # IKA-5 and `generation_match` did not, so every hidden-bench match until IKA-122
    # played the uniform belief -- a different agent -- while this list, lacking the name,
    # reported the match as the agent that ships.
    "bench_prior",
    # Which completion a hidden-bench menu is ranked from (IKA-143). Its default is what
    # ships, so omitting it drifts nothing; a tool that passes "first" plays the old rule.
    "rank_view",
)
#: One argument a tool may omit for a reason, where excusing the whole file in `EXPECTED`
#: would also excuse every other argument it might stop passing later.
EXCUSED_ARGS = {
    "branch_dedup.py": {
        "bench_prior": "draws both fours uniformly with no book, and against a uniform "
        "draw the uniform belief is the true one -- what generation holds on a book miss",
    },
}
#: Tools that drive something other than a full game on purpose, so a missing argument is
#: a choice rather than a drift. Each one says why.
EXPECTED = {
    "diff_generation.py": "differential harness: compares two engines on identical inputs",
    "diff_narrow.py": "differential harness for narrowing itself",
    "diff_node.py": "differential harness for one node",
    "diff_solve_node.py": "differential harness for one node",
    "dump_turn_cases.py": "dumps positions; the agent is not the subject",
    "bench_generation.py": "throughput benchmark; the agent is the variable being swept",
    "profile_generation.py": "profiler",
    "worker_growth.py": "throughput benchmark",
    "cells_needed.py": "counts cells; plays nothing that is scored",
}
#: The tools that drift as of 2026-09-19, recorded so that `--check` can fail on an
#: eighth without first demanding these seven be fixed. Not an excuse: a board number from
#: any of them is still a number about a different player, which is what the report says.
#: Struck off by fixing the call site, which `--check` then insists on.
KNOWN_DRIFT = frozenset(
    {
        "asymmetry.py",
        "book_check.py",
        "cycle_match.py",
        "forced_handoff.py",
        "matchup.py",
        "resume_generate.py",
        # Since IKA-122 lists `bench_prior`: its named arms draw the opponent's four from
        # the book's column strategy and then search over a uniform belief.
        "selection_check.py",
        "width_match.py",
    }
)


def calls(path: Path) -> list[tuple[int, set[str]]]:
    """(line, keyword names) for every `play_game(...)` in this file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "play_game":
            continue
        out.append((node.lineno, {k.arg for k in node.keywords if k.arg}))
    return out


def shipping_args() -> set[str]:
    """The arguments generation passes whose default is not what ships."""
    reference: set[str] = set()
    for _, kwargs in calls(ROOT / "src/pokeuraou/selfplay.py"):
        reference |= kwargs
    reference &= set(AGENT_ARGS)
    # `depth`, `policy`, `solve_sparsely` and `solve_restricted` default to what
    # generation ships (1, None, False, False), so omitting them changes nothing. The five
    # that bite are the ones whose default is NOT what ships: the leaf (None means
    # hp-share), the narrowing order (False means damage, generation uses the leaf), the
    # sheets (None means the search is handed the opponent's four), the width (8, where
    # generation plays 12) and the bench prior (None means the uniform belief, where
    # generation weights the bench by the opponent's selection equilibrium).
    return reference & {"evaluate", "rank_by_leaf", "sheets", "search_limit", "bench_prior"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="which tools build a different agent")
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero unless the drifted set is exactly KNOWN_DRIFT",
    )
    args = ap.parse_args(argv)

    reference = shipping_args()
    print(f"  the arguments whose default is not what ships: {', '.join(sorted(reference))}")
    print("  (depth, policy, solve_sparsely and solve_restricted default to the shipped\n"
          "   setting, so an omission there is not a drift)\n")

    drifted: set[str] = set()
    # `tools/oneshot/` is in scope too. A tool is shelved there when its question was
    # asked once, not because it stopped being runnable -- and a shelved tool that builds
    # an agent the current generation would not recognise is exactly the thing someone
    # re-runs later and reads as a board number. Scanning only `tools/*.py` would also let
    # a drifting tool leave the check by being moved, which is how `matchup.py` fell out.
    for path in sorted(glob.glob(str(ROOT / "tools/**/*.py"), recursive=True)):
        p = Path(path)
        found = calls(p)
        if not found:
            continue
        passed: set[str] = set()
        for _, kwargs in found:
            passed |= kwargs
        excused = EXCUSED_ARGS.get(p.name, {})
        for name in sorted((reference - passed) & set(excused)):
            print(f"  excused   {p.name:<26} omits {name}  -- {excused[name]}")
        missing = sorted(reference - passed - set(excused))
        note = EXPECTED.get(p.name)
        if not missing:
            print(f"  ok        {p.name}")
            continue
        if note:
            print(f"  expected  {p.name:<26} missing {', '.join(missing)}  -- {note}")
            continue
        drifted.add(p.name)
        print(f"  DRIFTED   {p.name:<26} missing {', '.join(missing)}")

    print(f"\n  {len(drifted)} tool(s) build an agent generation would not recognise.")
    if drifted:
        print("  A board number from one of those is a number about a different player.")

    if not args.check:
        return 0

    appeared = sorted(drifted - KNOWN_DRIFT)
    fixed = sorted(KNOWN_DRIFT - drifted)
    if appeared:
        print(
            f"\n  NEW: {', '.join(appeared)} drifted since the list was written.\n"
            "  Pass the missing arguments, or add the file to EXPECTED with the reason it\n"
            "  drives something other than a full game."
        )
    if fixed:
        print(
            f"\n  STALE: {', '.join(fixed)} no longer drifts and is still in KNOWN_DRIFT.\n"
            "  Strike it off, so the list keeps meaning what it says."
        )
    return 1 if (appeared or fixed) else 0


if __name__ == "__main__":
    sys.exit(main())
