"""Which tools build a different agent than generation does.

Generation and the board are played by the Rust workers (IKA-458 stage 3.7); their agents are
the configs `rustworker.build_config` and `rustworker.board_config` write, and `worker_drift`
builds both from generation's flags and compares them by value. A tool that leaves its fill or
its Q unnamed must resolve it as generation does (`resolution_drift`), and the time match must
build the agent a person meets (`play_agent_drift`). A tool that drifts is not measuring the
agent that ships -- it is measuring a different one and labelling the difference as whatever
it set out to study.

That is not hypothetical. On 2026-09-19 `selection_check` was found never to pass
`evaluate`, so every board number it had ever produced came from games the hp-share
heuristic played while the value function's claim sat beside them; it voided G2's recorded
-21.8 point miss, a six-team calibration panel and a 0-for-30 that looked like a
value-function collapse. Until IKA-463 this file also read every tool's `play_game(...)` call
for the arguments that decide the agent; IKA-463 removed `play_game`, so no tool calls it.

    uv run python tools/agent_drift.py            # the report
    uv run python tools/agent_drift.py --check    # and fail on any drift

`--check` is what CI calls.
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


#: IKA-458 stage 3.7: generation's config (`rustworker.build_config`) and every board arm
#: (`rustworker.board_config`) built from generation's flags agree on every key, by value, except
#: those `pokeuraou.workeragent.ALLOWED_DIFFERENT` names with its reason.
WORKER_AGENT = "every key but pokeuraou.workeragent.ALLOWED_DIFFERENT"


def worker_drift(rustworker_source: Path | None = None) -> list[str]:
    """How a board arm of generation's leaf, Q and width plays another agent than generation.

    Builds both configs from the same flags on the stand-in inference server and compares
    them by value (`pokeuraou.workeragent.agent_differences`), checks the node keys the Rust workers
    write after reading a config (`rust_node_writes`), and holds the board's default width to
    generation's. Empty when the two build the same agent.
    """
    from pokeuraou import workeragent

    problems = workeragent.worker_drift(rustworker_source)
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
        help="exit non-zero on any drift",
    )
    args = ap.parse_args(argv)

    worker = worker_drift()
    if worker:
        print(f"  DRIFTED   rustworker.py: the board's worker builds another agent than "
              f"generation's: {'; '.join(worker)}")
    else:
        print("  ok        rustworker.py: generation's config and every board arm agree on "
              f"{WORKER_AGENT}, by value (IKA-458 stage 3.7)")
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
    if not args.check:
        return 0

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
            "agent: build each key the same way in rustworker, or name it in ALLOWED_DIFFERENT."
        )
    if play_drift:
        print(
            f"\n  PLAY: {'; '.join(play_drift)}.\n  The time match has to play the agent a "
            "person meets: build it as play_human does, with play_human's defaults."
        )
    return 1 if (worker or unresolved or play_drift) else 0


if __name__ == "__main__":
    sys.exit(main())
