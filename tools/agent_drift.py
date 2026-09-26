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
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.qrank import DEFAULT_Q  # noqa: E402
from pokeuraou.search import SHIPPED_RANK_FILL  # noqa: E402

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
    # How a leaf ranking fills its cells (IKA-268). `play_game`'s default (refs2) is what
    # M-B generation ships, so omitting it drifts nothing here. M-C generation plays
    # q-nocover since IKA-338, and passes it: its tools' fill is checked where they resolve
    # it, at their options (`resolution_drift` below), since no `play_game` call shows it.
    "rank_fill",
    # Which completions a hidden-bench belief leaves out (IKA-283). Its default is what
    # ships, so omitting it drifts nothing; a tool that passes a drop plays another game.
    "bench_drop",
    # A best-first deepening budget in cells (IKA-33). Its default is what ships (0, off),
    # so omitting it drifts nothing; a tool that passes a budget plays another search.
    "deepen",
)
#: One argument a tool may omit for a reason, where excusing the whole file in `EXPECTED`
#: would also excuse every other argument it might stop passing later.
EXCUSED_ARGS: dict[str, dict[str, str]] = {
    # `branch_dedup.py` stood here (bench_prior) until IKA-212 deleted it.
}
#: Tools that drive something other than a full game on purpose, so a missing argument is
#: a choice rather than a drift. Each one says why.
EXPECTED = {
    "diff_generation.py": "differential harness: compares two engines on identical inputs",
    # diff_narrow, diff_node, diff_solve_node, dump_turn_cases and cells_needed stood here
    # until IKA-212 deleted them with Python's resolver.
    "bench_generation.py": "throughput benchmark; the agent is the variable being swept",
    "profile_generation.py": "profiler",
    "worker_growth.py": "throughput benchmark",
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


#: The options that name a leaf-ranked menu's fill. M-C generation's worker leaves them
#: unset and resolves them with `search.resolve_rank_fill` (q-nocover for a leaf-ranked
#: menu, IKA-338); a tool that defaults one to a label of its own plays another menu than
#: generation whenever the flag is left off -- the IKA-338 switch would have left every
#: board at refs2 had `pool_match.py` kept `default=DEFAULT_RANK_FILL`.
FILL_OPTIONS = ("--rank-fill", "--baseline-rank-fill")
#: A launcher's `--q-model`: unset, the Q the workers' fills want (`qrank.tail_wants_default_q`).
Q_OPTIONS = ("--q-model",)
#: Tools that resolve an unnamed fill or Q their own way on purpose, and why.
OWN_RESOLUTION = {
    "play_human.py": "a person's game: q-nocover when the Q is there, else refs2 with a "
    "note, as for a missing leaf (IKA-330, IKA-338)",
    "analyze.py": "play_human's agent, for a person reading (IKA-330)",
    "profile_stages.py": "hands --q-model to generate_queue.py only when given, so the "
    "driver's own default runs (IKA-98)",
}


def resolution_drift(path: Path) -> list[str]:
    """How this tool's fill and Q options fail to resolve as M-C generation's do (IKA-338).

    A fill option (`FILL_OPTIONS`) must default to None and the file must resolve it with
    `resolve_rank_fill`; a `--q-model` must default to None and the file must ask
    `tail_wants_default_q`. Empty when the file declares neither.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Name, ast.Attribute))
    }
    problems: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in FILL_OPTIONS + Q_OPTIONS):
            continue
        option = node.args[0].value
        default = next((k.value for k in node.keywords if k.arg == "default"), None)
        if default is not None and not (isinstance(default, ast.Constant)
                                        and default.value is None):
            problems.append(f"{option} defaults to {ast.unparse(default)}")
        resolver = "resolve_rank_fill" if option in FILL_OPTIONS else "tail_wants_default_q"
        if resolver not in names:
            problems.append(f"{option} is not resolved by {resolver}")
    return problems


def calls(path: Path) -> list[tuple[int, set[str]]]:
    """(line, keyword names) for every `play_game(...)` in this file.

    A `**kwargs` splat is recorded as the name `**`, because what it passes cannot be read
    from here.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "play_game":
            continue
        out.append((node.lineno, {k.arg or "**" for k in node.keywords}))
    return out


def unstated(path: Path) -> list[int]:
    """Lines of the `play_game` calls here that name neither `sheets` nor
    `open_information` (IKA-123).

    `play_game` stops at run time on such a call, but a tool nobody runs finds out late,
    so it is read here too. Every tool is in scope, the excused ones included: a harness
    that plays the open game on purpose says so with `open_information=True`, which costs
    one argument and leaves nothing to infer.
    """
    return [
        line
        for line, kwargs in calls(path)
        if not kwargs & {"sheets", "open_information", "**"}
    ]


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


def pool_generation_missing(reference: set[str]) -> list[str]:
    """What M-B generation passes and the pool generator (`poolplay`, IKA-81) does not.

    M-C is generated by `poolplay.generate_pool`, a second caller of `play_game` with
    generation's job. It is held to generation's own set rather than listed as a tool:
    the two generators drifting apart is the "measurement lags generation" defect with the
    teacher on both ends.
    """
    # Per CALL, not the union of the calls: the module holds generation's call and the
    # pool match's (IKA-259), and a union would let one of them omit what the other
    # passes -- the match measuring a different agent from the one generation plays.
    missing: set[str] = set()
    for _, kwargs in calls(ROOT / "src/pokeuraou/poolplay.py"):
        missing |= reference - kwargs
    return sorted(missing)


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
    pool_missing = pool_generation_missing(reference)
    if pool_missing:
        print(f"  DRIFTED   poolplay.py (M-C generation) missing {', '.join(pool_missing)}")
    else:
        print("  ok        poolplay.py (M-C generation, IKA-81)")
    print("  (depth, policy, solve_sparsely and solve_restricted default to the shipped\n"
          "   setting, so an omission there is not a drift)\n")

    drifted: set[str] = set()
    silent: list[str] = []
    # IKA-338: the fill and the Q a tool plays when they are left off, beside the arguments
    # it passes. The reference is M-C generation's worker, held to the same rule.
    print(f"  an unnamed fill of a leaf-ranked menu (M-C): {SHIPPED_RANK_FILL} by {DEFAULT_Q} "
          "(search.resolve_rank_fill, qrank.tail_wants_default_q)")
    unresolved: list[str] = []
    for path in sorted(glob.glob(str(ROOT / "tools/**/*.py"), recursive=True)):
        p = Path(path)
        problems = resolution_drift(p)
        if not problems:
            continue
        if p.name in OWN_RESOLUTION:
            print(f"  own way   {p.name:<26} {'; '.join(problems)}  -- {OWN_RESOLUTION[p.name]}")
            continue
        unresolved.append(p.relative_to(ROOT).as_posix())
        print(f"  DEFAULT   {p.name:<26} {'; '.join(problems)}")
    worker = ROOT / "tools/selfplay.py"
    if "resolve_rank_fill" not in worker.read_text(encoding="utf-8"):
        unresolved.append("tools/selfplay.py")
        print("  DEFAULT   selfplay.py                M-C generation's worker does not "
              "resolve its fill with resolve_rank_fill")
    print()
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
        for line in unstated(p):
            silent.append(f"{p.relative_to(ROOT).as_posix()}:{line}")
            print(f"  UNSTATED  {p.name}:{line} names neither sheets nor open_information")
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
    if silent:
        print(
            f"\n  UNSTATED: {', '.join(silent)} call play_game without saying which game.\n"
            "  Pass the sixes as `sheets`, or `open_information=True` for the open game."
        )
    if unresolved:
        print(
            f"\n  DEFAULT: {', '.join(unresolved)} resolve an unnamed fill or Q apart from "
            "generation.\n  Default the option to None and resolve it with "
            "search.resolve_rank_fill / qrank.tail_wants_default_q, or add the file to "
            "OWN_RESOLUTION with the reason."
        )
    if pool_missing:
        print(
            f"\n  POOL: poolplay.py omits {', '.join(pool_missing)} that M-B generation passes.\n"
            "  The two generators have to build the same agent."
        )
    return 1 if (appeared or fixed or silent or pool_missing or unresolved) else 0


if __name__ == "__main__":
    sys.exit(main())
