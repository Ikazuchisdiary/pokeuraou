"""Which tools build a different agent than generation does.

Generation and the board are played by the Rust workers (IKA-458 stage 3.7); their agents are
the configs `rustworker.build_config` and `rustworker.board_config` write, and `worker_drift`
holds the two to the same functions. A tool that calls `play_game` must name a leaf, a width, a
narrowing order, the sheets that hide the bench, the bench prior and the fill. A tool that omits
one of those is not measuring the agent that ships -- it is measuring a different one and
labelling the difference as whatever it set out to study.

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
    # How a leaf ranking fills its cells (IKA-268). Since IKA-341 `play_game`'s default is
    # no longer what generation ships: unnamed, a leaf-ranked side plays q-nocover by
    # the process's Q. M-C generation passes it too: its tools' fill is checked where they
    # resolve it, at their options (`resolution_drift` below), since no `play_game` call shows it.
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
EXPECTED: dict[str, str] = {
    # diff_generation.py stood here until IKA-458 stage 3.7 deleted it with the Python worker.
    # diff_narrow, diff_node, diff_solve_node, dump_turn_cases and cells_needed stood here
    # until IKA-212 deleted them with Python's resolver.
}
#: The tools that drift as of 2026-09-19, recorded so that `--check` can fail on an
#: eighth without first demanding these seven be fixed. Not an excuse: a board number from
#: any of them is still a number about a different player, which is what the report says.
#: Struck off by fixing the call site, which `--check` then insists on.
KNOWN_DRIFT: frozenset[str] = frozenset()


#: The options that name a leaf-ranked menu's fill. M-C generation's worker leaves them
#: unset and resolves them with `search.resolve_rank_fill` (q-nocover for a leaf-ranked
#: menu, IKA-338); a tool that defaults one to a label of its own plays another menu than
#: generation whenever the flag is left off -- the IKA-338 switch would have left every
#: board at refs2 had `pool_match.py` kept `default=DEFAULT_RANK_FILL`.
FILL_OPTIONS = ("--rank-fill", "--baseline-rank-fill")
#: A launcher's `--q-model`: unset, the Q the workers' fills want (`qrank.tail_wants_default_q`).
Q_OPTIONS = ("--q-model",)
#: Where a person's game (`tools/play_human.py`) differs from generation and the time match on
#: purpose, and why. Not a drift: the check names it so that it is read as a decision.
PLAY_ONLY_DEFAULTS = (
    "the selection is read deeper than the leaf's one estimate of each cell, 90 s of wall time on "
    "every core, when the game is on the wall clock with a leaf (`--selection-reading none` "
    "turns it off; IKA-392: the reading beat the leaf's selection by +23.2 Elo [+8.0, +38.6] on "
    "the time match's board). Each cell is read at depth 1 (`humanplay.PLAY_SELECTION_READING`, "
    "IKA-440: +12.7 Elo [-0.1, +25.6] over the depth-2 cells at equal cost; "
    "`--selection-reading default` plays the depth-2 cells). Generation (`poolplay`, "
    "`selfplay`), the time match's conditions and the move's reading do not read it",
)
#: Tools that resolve an unnamed fill or Q their own way on purpose, and why.
OWN_RESOLUTION = {
    "play_human.py": "a person's game: q-nocover when the Q is there, else refs2 with a "
    "note, as for a missing leaf (IKA-330, IKA-338)",
    "analyze.py": "play_human's agent, for a person reading (IKA-330)",
    "profile_stages.py": "hands --q-model to generate_queue.py only when given, so the "
    "driver's own default runs (IKA-98)",
    "time_match.py": "play_human's agent in both seats: q-nocover by the default Q, and a "
    "match stops without it as a board does (IKA-333)",
    "position_set.py": "a verification position set read by time-match conditions: the menus "
    "are q-nocover by the Q it names (the default Q), the same for every reading (IKA-362)",
    "root_width.py": "reads a position set's recorded positions with a ladder at fixed or staged "
    "root widths: menus q-nocover by the Q it names (the default Q), the same for every arm; "
    "plays no game (IKA-418)",
    "depth_outcome.py": "reads recorded held-out positions with a ladder to score its values "
    "against the games' results: menus q-nocover by the Q it is given; plays no game (IKA-421)",
    "child_menus.py": "measures child menus built by damage and by the Q it names (the "
    "default Q); plays no game (IKA-362)",
    "selection_sweep.py": "reads selection cells as a person's move reads them, menus q-nocover "
    "by the Q it names (the default Q); plays no game (IKA-392)",
    "ranked_entry.py": "reads a ranked match's selection with play_human's own set-up "
    "(its install_menus: q-nocover by the Q it names, the default Q; its leaf, its 90 s "
    "default reading); plays no game (IKA-407)",
    "show_game.py": "--reread re-solves a recorded generation game with the Q the record names "
    "(its qModel, checked against qModelSha256), not the default; plays no game",
}


def agent_kwargs(path: Path) -> set[str]:
    """The keyword names of every `Agent(...)` / `humanplay.Agent(...)` call in this file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "Agent":
            names |= {k.arg or "**" for k in node.keywords}
    return names


def flag_defaults(path: Path) -> dict[str, str]:
    """``--flag``: its default as source text, for every `add_argument` in this file
    (``store_true`` flags as ``False``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant)):
            continue
        keywords = {k.arg: k.value for k in node.keywords}
        if "default" in keywords:
            out[node.args[0].value] = ast.unparse(keywords["default"])
        elif isinstance(keywords.get("action"), ast.Constant) and keywords["action"].value == "store_true":
            out[node.args[0].value] = "False"
    return out


def play_agent_drift() -> list[str]:
    """How the time match's agent (`timematch`, IKA-333) differs from a person's opponent
    (`tools/play_human.py`): an argument its `Agent(...)` leaves out, or a condition's
    default that is not the flag's default a person's game plays with."""
    from pokeuraou import humanplay, timematch

    problems: list[str] = []
    reference = agent_kwargs(ROOT / "tools/play_human.py")
    mine = agent_kwargs(ROOT / "src/pokeuraou/timematch.py")
    if not reference:
        problems.append("tools/play_human.py builds no Agent(...) to compare with")
    if reference - mine:
        problems.append(f"timematch's Agent(...) omits {', '.join(sorted(reference - mine))}")
    flags = flag_defaults(ROOT / "tools/play_human.py")
    base = timematch.Condition(name="default", seconds=1.0)
    wanted = {
        # flag: (what play_human's default resolves to, the condition's default)
        "--oracle": (timematch._oracle(eval(flags.get("--oracle", "None")) or "none"), base.oracle),  # noqa: S307
        "--clock": (eval(flags.get("--clock", "None")), base.clock),  # noqa: S307
        "--max-levels": (
            humanplay.PLAY_MAX_LEVELS if flags.get("--max-levels") == "humanplay.PLAY_MAX_LEVELS"
            else flags.get("--max-levels"),
            base.max_levels,
        ),
        "--threads": (humanplay.default_threads() if flags.get("--threads") == "None"
                      else flags.get("--threads"), base.threads),
        "--child-q": (eval(flags.get("--child-q", "0")), base.child_q),  # noqa: S307
        "--width-only": (eval(flags.get("--width-only", "None")), base.width_only),  # noqa: S307
    }
    for flag, (theirs, ours) in wanted.items():
        if theirs != ours:
            problems.append(f"play_human {flag} plays {theirs!r}, a condition's default is {ours!r}")
    return problems


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
    """Lines of the `play_game` calls here that do not name `sheets` (IKA-123, IKA-460).

    `sheets` is required since the open game was removed, so such a call stops at run
    time; a tool nobody runs finds out late, so it is read here too.
    """
    return [
        line
        for line, kwargs in calls(path)
        if not kwargs & {"sheets", "**"}
    ]


#: The arguments a `play_game` call must name to play generation's agent: those whose default is
#: not what ships. Generation itself no longer calls `play_game` (its games and the board's are
#: the Rust workers', IKA-458 stage 3.7; `worker_drift` holds their configs), so the set is
#: written here: the leaf (None means hp-share), the narrowing order (False means damage), the
#: sheets (required), the width (8, where generation plays 12), the bench prior (None means the
#: uniform belief) and the fill (None means q-nocover by the process's Q). `depth`, `policy`,
#: `solve_sparsely` and `solve_restricted` default to what generation ships.
SHIPPING_ARGS = frozenset({
    "evaluate", "rank_by_leaf", "sheets", "search_limit", "bench_prior", "rank_fill",
})


def shipping_args() -> set[str]:
    """The arguments a `play_game` call must name whose default is not what ships."""
    return set(SHIPPING_ARGS) & set(AGENT_ARGS)


#: The keys of a Rust worker's config that decide which agent plays, and the one function in
#: `rustworker` each must come from (IKA-458 stage 3.7). Generation's config (`build_config`) and
#: the board's (`board_config`, per arm for the node) both take them from there, so an arm with
#: generation's leaf, Q and width plays generation's agent.
WORKER_AGENT = {
    "node": "agent_node",
    "exactBudget": "exact_budget",
    "maxTurns": "max_turns",
    "notes": "selection_notes",
}
#: The config builders held to `WORKER_AGENT`.
WORKER_BUILDERS = ("build_config", "board_config")
#: What builds an agent's parts: called only inside the `WORKER_AGENT` functions.
AGENT_PARTS = ("static_node", "static_request", "dump_budget")


def _callees(node: ast.AST) -> set[str]:
    """The names of the functions called in this expression."""
    out = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            out.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "?"))
    return out


def _agent_sources(function: ast.FunctionDef) -> dict[str, set[str]]:
    """For each `WORKER_AGENT` key a dict literal in this function sets: the functions its value
    calls (a name followed to what it was assigned; a value that calls none, as source text)."""
    assigned: dict[str, list[ast.AST]] = {}
    for sub in ast.walk(function):
        if isinstance(sub, ast.Assign):
            for target in sub.targets:
                if isinstance(target, ast.Name):
                    assigned.setdefault(target.id, []).append(sub.value)
    found: dict[str, set[str]] = {}
    for sub in ast.walk(function):
        if not isinstance(sub, ast.Dict):
            continue
        for key, value in zip(sub.keys, sub.values, strict=True):
            if not (isinstance(key, ast.Constant) and key.value in WORKER_AGENT):
                continue
            values = assigned.get(value.id, [value]) if isinstance(value, ast.Name) else [value]
            for each in values:
                found.setdefault(key.value, set()).update(_callees(each) or {ast.unparse(each)})
    return found


def _node_edits(builder: str, function: ast.FunctionDef) -> list[str]:
    """How this builder changes an agent's node other than dropping `BOARD_FILLS`."""
    problems = []
    for sub in ast.walk(function):
        targets = sub.targets if isinstance(sub, ast.Assign) else (
            [sub.target] if isinstance(sub, ast.AugAssign) else [])
        if any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
               and t.value.id == "node" for t in targets):
            problems.append(f"{builder} writes into an agent's node (line {sub.lineno})")
        if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                and isinstance(sub.func.value, ast.Name) and sub.func.value.id == "node"
                and sub.func.attr != "pop"):
            problems.append(f"{builder} changes an agent's node with {sub.func.attr} "
                            f"(line {sub.lineno})")
        if (isinstance(sub, ast.For) and any(
                isinstance(stmt, ast.Expr) and "node.pop(" in ast.unparse(stmt)
                for stmt in sub.body)
                and ast.unparse(sub.iter) != "BOARD_FILLS"):
            problems.append(f"{builder} drops {ast.unparse(sub.iter)} from a node, not BOARD_FILLS")

    def pops(tree: ast.AST) -> set[int]:
        return {id(c) for c in ast.walk(tree) if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute) and c.func.attr == "pop"
                and isinstance(c.func.value, ast.Name) and c.func.value.id == "node"}

    allowed: set[int] = set()
    for sub in ast.walk(function):
        if isinstance(sub, ast.For) and ast.unparse(sub.iter) == "BOARD_FILLS":
            allowed |= pops(sub)
    if pops(function) - allowed:
        problems.append(f"{builder} drops a key from an agent's node outside BOARD_FILLS")
    return problems


def worker_drift(path: Path | None = None) -> list[str]:
    """How the Rust workers' configs fail to build generation's agent (IKA-458 stage 3.7).

    Read from `rustworker.py` with `ast`: in `build_config` and `board_config`, every agent key
    of `WORKER_AGENT` is set from its one function; the agent's parts (`AGENT_PARTS`) are called
    inside those functions only; the board changes an arm's node only by dropping the keys it
    fills itself (`BOARD_FILLS`). And the board's default width is generation's. Empty when the
    two build the same agent.
    """
    path = path or ROOT / "src/pokeuraou/rustworker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    problems: list[str] = []
    for name, function in functions.items():
        if name in WORKER_AGENT.values():
            continue
        for part in sorted(_callees(function) & set(AGENT_PARTS)):
            problems.append(f"{name} calls {part}: an agent's part built outside "
                            f"{', '.join(sorted(set(WORKER_AGENT.values())))}")
    for builder in WORKER_BUILDERS:
        function = functions.get(builder)
        if function is None:
            problems.append(f"rustworker has no {builder}")
            continue
        found = _agent_sources(function)
        for key, source in WORKER_AGENT.items():
            if key not in found:
                problems.append(f"{builder} sets no {key}")
            elif found[key] != {source}:
                problems.append(f"{builder}'s {key} comes from {', '.join(sorted(found[key]))}, "
                                f"not {source}")
        problems += _node_edits(builder, function)
    generation = flag_defaults(ROOT / "tools/generate_queue.py").get("--limit")
    board = next((ast.unparse(n.value) for n in ast.parse(
        (ROOT / "tools/pool_match.py").read_text(encoding="utf-8")).body
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "GENERATION_LIMIT" for t in n.targets)), None)
    if generation != board:
        problems.append(f"the board's default width {board} is not generation's {generation}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="which tools build a different agent")
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero unless the drifted set is exactly KNOWN_DRIFT",
    )
    args = ap.parse_args(argv)

    reference = shipping_args()
    worker = worker_drift()
    if worker:
        print(f"  DRIFTED   rustworker.py: the board's worker builds another agent than "
              f"generation's: {'; '.join(worker)}")
    else:
        print("  ok        rustworker.py: generation's config and every board arm take the agent "
              f"({', '.join(WORKER_AGENT)}) from the same functions (IKA-458 stage 3.7)")
    print(f"  a play_game call must name: {', '.join(sorted(reference))}")
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
    if "resolve_rank_fill" not in (ROOT / "tools/selfplay.py").read_text(encoding="utf-8"):
        unresolved.append("tools/selfplay.py")
        print("  DEFAULT   selfplay.py                M-C generation's worker does not "
              "resolve its fill with resolve_rank_fill")
    play_drift = play_agent_drift()
    if play_drift:
        print(f"  DRIFTED   time_match.py (IKA-333) plays another agent than play_human: "
              f"{'; '.join(play_drift)}")
    else:
        print("  ok        time_match.py (IKA-333): play_human's agent, its defaults included")
    for note in PLAY_ONLY_DEFAULTS:
        print(f"  own way   play_human.py: {note}")
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
            print(f"  UNSTATED  {p.name}:{line} names no sheets")
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
            "  Pass the sixes as `sheets`."
        )
    if unresolved:
        print(
            f"\n  DEFAULT: {', '.join(unresolved)} resolve an unnamed fill or Q apart from "
            "generation.\n  Default the option to None and resolve it with "
            "search.resolve_rank_fill / qrank.tail_wants_default_q, or add the file to "
            "OWN_RESOLUTION with the reason."
        )
    if worker:
        print(
            f"\n  WORKER: {'; '.join(worker)}.\n  The board's arms have to play generation's "
            "agent: take each agent key from its function in rustworker (WORKER_AGENT)."
        )
    if play_drift:
        print(
            f"\n  PLAY: {'; '.join(play_drift)}.\n  The time match has to play the agent a "
            "person meets: build it as play_human does, with play_human's defaults."
        )
    return 1 if (appeared or fixed or silent or worker or unresolved or play_drift) else 0


if __name__ == "__main__":
    sys.exit(main())
