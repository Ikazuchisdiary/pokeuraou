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

    uv run python tools/agent_drift.py
"""

from __future__ import annotations

import ast
import glob
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
    "sheets",
)
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


def main() -> None:
    reference: set[str] = set()
    for _, kwargs in calls(ROOT / "src/pokeuraou/selfplay.py"):
        reference |= kwargs
    reference &= set(AGENT_ARGS)
    # `depth` and `policy` and `solve_sparsely` default to what generation ships (1, None,
    # False), so omitting them changes nothing. The three that bite are the ones whose
    # default is NOT what ships: the leaf (None means hp-share), the narrowing order
    # (False means damage, generation uses the leaf) and the sheets (None means the search
    # is handed the opponent's four).
    reference &= {"evaluate", "rank_by_leaf", "sheets", "search_limit"}
    print(f"  the arguments whose default is not what ships: {', '.join(sorted(reference))}")
    print("  (depth, policy and solve_sparsely default to the shipped setting, so an\n"
          "   omission there is not a drift)\n")

    drifted = 0
    for path in sorted(glob.glob(str(ROOT / "tools/*.py"))):
        p = Path(path)
        found = calls(p)
        if not found:
            continue
        passed: set[str] = set()
        for _, kwargs in found:
            passed |= kwargs
        missing = sorted(reference - passed)
        note = EXPECTED.get(p.name)
        if not missing:
            print(f"  ok        {p.name}")
            continue
        if note:
            print(f"  expected  {p.name:<26} missing {', '.join(missing)}  -- {note}")
            continue
        drifted += 1
        print(f"  DRIFTED   {p.name:<26} missing {', '.join(missing)}")

    print(f"\n  {drifted} tool(s) build an agent generation would not recognise.")
    if drifted:
        print("  A board number from one of those is a number about a different player.")


if __name__ == "__main__":
    main()
