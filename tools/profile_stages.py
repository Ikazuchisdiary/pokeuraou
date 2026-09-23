"""The time a real run spends, by stage, across every process it uses.

`tools/profile_generation.py` profiles whole games with `cProfile`, in one process, from a
call site of its own. That answers a narrower question than this one and carries two
problems the issue this was built for names directly. A profile attributes time to
*functions*, and three of the stages here are not functions: branch generation is
`resolve_turn` with the bridge off and a Rust child with it on, and the forward pass is in
this process, in a server, or nowhere visible. And a tool with its own `play_game` call is
a tool that can drift from the one generation makes -- `agent_drift.py` excuses the
profiler as "profiler", which excuses the *shape* of the call and says nothing about
whether the workload is representative. It is not: no `sheets` means no hidden bench, so
the matrix completion that hidden play pays for never runs, and no `rank_by_leaf` means
narrowing never touches the leaf. Both move the balance this is trying to report.

So nothing here plays a game. It runs the real driver, with the real flags, and reads what
the processes wrote about themselves::

    uv run --group learn python tools/profile_stages.py generation --games 200
    uv run --group learn python tools/profile_stages.py generation --games 200 --no-bridge
    uv run --group learn python tools/profile_stages.py match --games 60 --served
    uv run --group learn python tools/profile_stages.py analysis --limit 16

Generation takes `generate_queue.py`'s own options and passes on only the ones given, so
the shipping command line (tools/ika73_generate.sh) goes in as it is::

    uv run --group learn python tools/profile_stages.py generation --games 600 \\
        --seed 6601 --served --servers 2 --workers 24 --limit 12 \\
        --value data/models/value-gen11L.pt \\
        --selection-book data/selection/rizabanadohido-value-gen11L.jsonl.gz \\
        --hide-bench -- --rank-leaf

After the stage table it prints what IKA-98 added: the CPU of the tree by role, the
servers' CPU over the time they held a request, what one decision of each kind costs in
counts, the rest regressed on the decisions (with a warning when it is too large to read
the table as a breakdown), and what each worker's own report says it ran -- its argv and
the checkout its code came from. A worker that ran something else exits this with 3.

`pokeuraou.timing` is off unless `POKEURAOU_TIMING` is set, which this sets for the child
and everything below it. Two numbers come back for every stage:

    wall   summed over every process that reported. On eight workers this is about eight
           times the run's own clock, and that is the point -- a share of it is a share of
           the machine, which is what a speed-up would be buying
    cpu    `thread_time`, quantised at 15.625 ms on Windows. Read it to tell a stage that
           is burning CPU from one that is blocked on another process, not for its value

The CPU seconds of the process tree are polled here rather than taken from the reports,
because a report is written at exit and a child that has already been reaped reports
nothing. Polling also separates `python.exe` from `pokeuraou-damage.exe`, which is the
distinction that made "Rust 84% / Python 4%" read as a CPU split when it was a wall-clock
one.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.timing import BORROWED, PURPOSE_ROWS, PURPOSES, STAGES  # noqa: E402

#: Rows printed under a heading, so a table reads as a breakdown rather than a list.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("once a process", ("startup",)),
    ("Python", ("narrow", "branch", "encode", "forward", "lp", "belief")),
    ("bridge", ("rust.fill", "rust.ask", "rust.header", "rust.body", "rust.unpack",
                "rust.resolve", "rust.score")),
    ("served", ("serve.copy", "serve.wait")),
    ("child", ("rust.child.resolve", "rust.child.encode", "rust.child.parse",
               "rust.child.header")),
    ("server", ("server.held", "server.queue")),
    ("inclusive", ("refused",)),
    # `rust.fill` and the child's clocks again, cut by what the fill was for (IKA-98).
    ("by purpose", PURPOSE_ROWS),
)

#: The scripts a process of a run can be, by what its command line names. `tree_cpu` reads
#: them off each PID's command line, because by name every one of them is `python.exe`.
ROLE_SCRIPTS = (
    "inference_server.py", "selfplay.py", "generate_queue.py", "match_queue.py",
    "why_action.py", "profile_stages.py",
)

#: The Rust child, by the name its executable has.
DAMAGE_EXE = "pokeuraou-damage.exe"

#: Above this share of the workers' wall clock, the work that grows with the decisions and
#: has no row is too large for the table to be read as a breakdown (IKA-98's instrument 7).
UNNAMED_LIMIT = 0.05


def process_role(name: str, cmdline: list[str]) -> str:
    """What a process was in the run: the script its command line names, or its own name.

    The venv's `python.exe` is a launcher that starts the real interpreter with the same
    command line, so a script shows up as two PIDs; the launcher's CPU is next to nothing.
    """
    for item in cmdline:
        base = str(item).replace("\\", "/").rsplit("/", 1)[-1]
        if base in ROLE_SCRIPTS:
            return base
    if name.lower().startswith("pokeuraou-damage"):
        return DAMAGE_EXE
    return name

#: A stage added to `timing.STAGES` and not to a group above still prints, under the
#: extras at the bottom of the table -- but with no heading and in no particular order.
#: Saying so here is cheaper than noticing a row has gone quiet.
_UNGROUPED = sorted(set(STAGES) - {name for _heading, names in GROUPS for name in names})


def tree_cpu(
    pid: int, stop: threading.Event, every: float = 1.0, floor_gb: float = 2.0
) -> dict[str, Any]:
    """Poll a process tree's CPU seconds and RSS until `stop`, keeping the last reading.

    Per PID and not per name, because a worker that dies mid-run still spent what it
    spent: the kernel's counter is cumulative, so the largest reading a PID ever gave is
    its total, and summing those survives a process the poll missed the end of.

    It also stops the run when free memory falls below `floor_gb`. The analysis workload
    is the reason: a 24x24 depth-1 solve was measured at 15.3 GB of RSS and took the
    machine to 0.3 GB free, which is not a slow run -- it is every other process on the
    machine in trouble. A measurement is not worth that, and a run killed at the floor at
    least says so, where a machine that swapped to a halt says nothing.
    """
    import psutil

    totals: dict[int, tuple[str, float]] = {}
    roles: dict[int, str] = {}
    peak_rss = 0
    samples = 0
    low_water = float("inf")
    stopped_for_memory = False
    while not stop.is_set():
        rss = 0
        try:
            root = psutil.Process(pid)
            group = [root, *root.children(recursive=True)]
        except psutil.Error:
            group = []
        for process in group:
            try:
                times = process.cpu_times()
                spent = float(times.user + times.system)
                name = process.name()
                known = totals.get(process.pid)
                if known is None or spent > known[1]:
                    totals[process.pid] = (name, spent)
                if process.pid not in roles:
                    # Once a PID: a command line does not change, and reading it is not free.
                    try:
                        line = process.cmdline()
                    except psutil.Error:
                        line = []
                    roles[process.pid] = process_role(name, line)
                rss += int(process.memory_info().rss)
            except psutil.Error:
                continue
        peak_rss = max(peak_rss, rss)
        free = psutil.virtual_memory().available / 1e9
        low_water = min(low_water, free)
        if group and free < floor_gb:
            stopped_for_memory = True
            print(f"\n  free memory is {free:.1f} GB, under the {floor_gb:.1f} GB floor; "
                  f"stopping the run", file=sys.stderr, flush=True)
            for process in reversed(group):
                with contextlib.suppress(psutil.Error):
                    process.kill()
            break
        samples += 1
        stop.wait(every)

    by_name: dict[str, float] = {}
    for name, spent in totals.values():
        by_name[name] = by_name.get(name, 0.0) + spent
    # The same seconds by role, so the servers and the workers are not one `python.exe`.
    by_role: dict[str, float] = {}
    role_processes: dict[str, int] = {}
    for pid, (name, spent) in totals.items():
        role = roles.get(pid, name)
        by_role[role] = by_role.get(role, 0.0) + spent
        role_processes[role] = role_processes.get(role, 0) + 1
    return {
        "by_name": by_name,
        "by_role": by_role,
        "role_processes": role_processes,
        "total": sum(by_name.values()),
        "peak_rss": peak_rss,
        "free_low_water": 0.0 if low_water == float("inf") else low_water,
        "stopped_for_memory": stopped_for_memory,
        "processes": len(totals),
        "samples": samples,
    }


def collect(directory: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(directory.glob("timing-*.json")):
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return reports


#: The script an inference server runs. Its reports are kept out of the workers' table.
SERVER = "inference_server.py"


def _role(report: dict[str, Any]) -> str:
    """What a reporting process was, from the script it was started with."""
    argv = report.get("argv") or []
    for item in argv:
        name = Path(str(item)).name
        if name.endswith(".py") and name != "profile_stages.py":
            return name
    return "python"


def _split(reports: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Workers and servers apart, because their seconds cannot go in one column.

    A worker is one thread doing one thing, so its stages are a partition of its wall
    clock. A server answers every connection on its own thread, so the same wall second
    is inside as many `forward` rows as it had threads running -- 24 workers through two
    servers put the shares over 100% and the leftover row at *minus* 5.5%, which is the
    arithmetic saying the denominator is not the same quantity as the numerators.

    So the server gets its own block and its own denominator, and the note that its
    rows are thread-seconds rather than wall-seconds travels with them.
    """
    workers = [r for r in reports if _role(r) != SERVER]
    servers = [r for r in reports if _role(r) == SERVER]
    return workers, servers


def summarise(reports: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for report in reports:
        for name, row in (report.get("stages") or {}).items():
            into = stages.setdefault(name, {"wall": 0.0, "cpu": 0.0, "calls": 0.0})
            into["wall"] += float(row.get("wall", 0.0))
            into["cpu"] += float(row.get("cpu", 0.0))
            into["calls"] += float(row.get("calls", 0))
        for name, value in (report.get("counts") or {}).items():
            counts[name] = counts.get(name, 0) + int(value)
    own = sum(
        value["wall"] for name, value in stages.items() if name not in BORROWED
    )
    elapsed = sum(float(r.get("elapsed", 0.0)) for r in reports)
    roles: dict[str, int] = {}
    for report in reports:
        role = _role(report)
        roles[role] = roles.get(role, 0) + 1
    return {
        "stages": stages,
        "counts": counts,
        "accounted": own,
        "elapsed": elapsed,
        "processes": len(reports),
        "roles": roles,
    }


def _seconds(value: float) -> str:
    return f"{value:,.1f}"


def print_server(summary: dict[str, Any]) -> None:
    """The serving side's own block: its clock, its rows, and why they are not the same."""
    stages = summary["stages"]
    elapsed = summary["elapsed"]
    held = stages.get("server.held", {})
    queued = stages.get("server.queue", {})
    forward = stages.get("forward", {})
    print(f"\n  the serving side: {summary['processes']} server(s), "
          f"{_seconds(elapsed)} s of process wall clock between them")
    if held.get("calls"):
        print(f"   {'working (its own counter)':<28} {_seconds(held['wall']):>10} over "
              f"{held['calls']:,.0f} requests = {1000 * held['wall'] / held['calls']:.2f} ms each")
    if queued.get("calls"):
        print(f"   {'queueing for a thread':<28} {_seconds(queued['wall']):>10}")
    if forward.get("calls"):
        print(f"   {'forward, as staged here':<28} {_seconds(forward['wall']):>10} over "
              f"{forward['calls']:,.0f} calls")
    print("   these are thread-seconds, not wall-seconds: a server answers each connection\n"
          "   on its own thread, so concurrent threads put more seconds in a row than the\n"
          "   process lived. They are the worker's serve.wait seen from the other end.")


def print_table(summary: dict[str, Any], tree: dict[str, Any] | None, wall: float) -> None:
    stages = summary["stages"]
    elapsed = summary["elapsed"]
    accounted = summary["accounted"]
    share = (lambda v: f"{100.0 * v / elapsed:5.1f}%") if elapsed > 0 else (lambda _v: "   --")

    roles = ", ".join(f"{name}x{count}" for name, count in sorted(summary["roles"].items()))
    print(f"\n  run {wall:,.1f} s of wall clock; {summary['processes']} processes reported "
          f"({roles})")
    print(f"  their wall clock adds to {_seconds(elapsed)} s, which is what the shares "
          f"below are shares of\n")
    print(f"  {'stage':<20} {'wall s':>10} {'share':>7} {'cpu s':>10} {'calls':>12}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 7} {'-' * 10} {'-' * 12}")
    for heading, names in GROUPS:
        rows = [n for n in names if n in stages and stages[n]["calls"]]
        if not rows:
            continue
        print(f"  {heading}")
        for name in rows:
            row = stages[name]
            mark = "*" if name in BORROWED else " "
            print(f"  {mark}{name:<19} {_seconds(row['wall']):>10} {share(row['wall']):>7} "
                  f"{_seconds(row['cpu']):>10} {row['calls']:>12,.0f}")
    extra = sorted(set(stages) - {n for _h, names in GROUPS for n in names})
    if _UNGROUPED:
        print(f"  (not in any group: {', '.join(_UNGROUPED)})")
    for name in extra:
        row = stages[name]
        if row["calls"]:
            print(f"   {name:<19} {_seconds(row['wall']):>10} {share(row['wall']):>7} "
                  f"{_seconds(row['cpu']):>10} {row['calls']:>12,.0f}")
    print(f"  {'-' * 20} {'-' * 10} {'-' * 7} {'-' * 10} {'-' * 12}")
    print(f"   {'measured':<19} {_seconds(accounted):>10} {share(accounted):>7}")
    # A report from before IKA-98 has no startup row, and its rest still holds startup.
    held_apart = stages.get("startup", {}).get("calls", 0)
    what = "no timer: bridge, I/O, interpreter" + ("" if held_apart else ", startup")
    print(f"   {'the rest':<19} {_seconds(elapsed - accounted):>10} "
          f"{share(elapsed - accounted):>7}   {what}")
    print("\n  * time another row already holds, so never added into a total: the Rust\n"
          "    child's own clock inside rust.fill, the serving thread's inside forward,\n"
          "    and the refused-cell tail across branch, encode and forward at once.")

    counts = summary.get("counts") or {}
    if counts:
        # Leaves per forward pass first, because it is the one number that says whether
        # the batching this project paid 1.86x for is still being got.
        leaves = counts.get("leaves", 0)
        calls = stages.get("forward", {}).get("calls", 0)
        if leaves and calls:
            print(f"\n  {leaves:,} leaves over {calls:,.0f} forward passes = "
                  f"{leaves / calls:,.0f} rows a call")
        # What the bridge actually carried, and by which road. A body's size used to be
        # arrived at by multiplying a leaf count by a per-leaf figure out of another file,
        # and the share that figure was quoted beside was of a measurement.
        carried = counts.get("body.bytes", 0)
        through = counts.get("body.shm", 0)
        piped = counts.get("body.pipe", 0)
        if carried:
            nodes = through + piped
            road = f"{through:,} through the block, {piped:,} down the pipe"
            print(f"\n  {carried / 1e9:.2f} GB of encoded leaves over {nodes:,} nodes "
                  f"= {carried / max(nodes, 1) / 1e6:,.1f} MB a node ({road})")
        # And how much of a node the port's leaf sharing took off the crossing before it
        # was carried at all. Counted at `add_leaf` rather than inferred from how large
        # the node came out: the inference has been in the record twice and the count
        # never was.
        offered = counts.get("leaves.offered", 0)
        stored = counts.get("leaves.stored", 0)
        if offered and stored:
            print(f"\n  {offered:,} leaves offered, {stored:,} kept = "
                  f"{offered / stored:.2f}x, {100.0 * (offered - stored) / offered:.1f}% "
                  f"of them a position the node already had")
        refusals = {k: v for k, v in counts.items() if k.startswith("refused: ")}
        if refusals:
            total = sum(refusals.values())
            print(f"\n  why the port refused a cell ({total:,} cells, "
                  f"{len(refusals)} distinct reasons)")
            for reason, times in sorted(refusals.items(), key=lambda kv: -kv[1])[:12]:
                print(f"   {100.0 * times / total:5.1f}%  {times:>9,}  "
                      f"{reason[len('refused: '):]}")

    if tree:
        print(f"\n  CPU seconds of the process tree ({tree['processes']} processes, "
              f"{tree['samples']} samples)")
        for name, spent in sorted(tree["by_name"].items(), key=lambda kv: -kv[1]):
            portion = 100.0 * spent / tree["total"] if tree["total"] else 0.0
            print(f"   {name:<28} {_seconds(spent):>10} {portion:5.1f}%")
        print(f"   {'total':<28} {_seconds(tree['total']):>10}")
        by_role = tree.get("by_role") or {}
        if by_role:
            print("  the same seconds by role, from each process's command line")
            processes = tree.get("role_processes") or {}
            for role, spent in sorted(by_role.items(), key=lambda kv: -kv[1]):
                portion = 100.0 * spent / tree["total"] if tree["total"] else 0.0
                label = f"{role} x{processes.get(role, 0)}"
                print(f"   {label:<28} {_seconds(spent):>10} {portion:5.1f}%")
        print(f"   peak RSS of the tree: {tree['peak_rss'] / 1e9:.1f} GB; "
              f"free memory fell to {tree.get('free_low_water', 0.0):.1f} GB")
        if tree.get("stopped_for_memory"):
            print("   ** the run was killed at the memory floor: the table above is of "
                  "what it managed first **")
        if wall > 0:
            print(f"   busy cores: {tree['total'] / wall:.1f}")


def spin(
    tree: dict[str, Any] | None,
    servers: dict[str, Any] | None,
    reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The servers' CPU seconds over the seconds they held a request (instrument 4).

    Near 1 means a serving thread burns a core while it waits for the GPU (IKA-106); near
    0 means it sleeps. The tree's CPU is the whole process life, so each server's CPU at
    `timing.ready()` -- torch's import, the CUDA context, the model -- is taken out when
    its report has it. What is left still holds the serving work outside a request's hold
    (reading the control line, the accept loop, the five-second report), so the ratio is
    an upper bound on the waiting's share, not the waiting itself.
    """
    if not tree or not servers:
        return None
    cpu = (tree.get("by_role") or {}).get(SERVER)
    held = (servers.get("stages") or {}).get("server.held", {}).get("wall", 0.0)
    if cpu is None or not held:
        return None
    started = [
        float(r["startup_process_cpu"])
        for r in (reports or [])
        if _role(r) == SERVER and r.get("startup_process_cpu") is not None
    ]
    serving = cpu - sum(started)
    return {"cpu": cpu, "startup_cpu": sum(started), "servers_with_startup": len(started),
            "held": held, "ratio": serving / held}


def print_spin(found: dict[str, Any] | None) -> None:
    if found is None:
        return
    print(f"\n  server CPU over server.held: ({_seconds(found['cpu'])} s less "
          f"{_seconds(found['startup_cpu'])} s of start in {found['servers_with_startup']} "
          f"server(s)) / {_seconds(found['held'])} s = {found['ratio']:.2f}   "
          f"(near 1: waits by spinning)")


def _worker_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every process that played games: a generation's selfplay.py, a match's worker."""
    return _split(reports)[0]


def per_decision(reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every worker's decision stretches, summed by kind (from `timing.decided`)."""
    out: dict[str, dict[str, Any]] = {}
    for report in _worker_reports(reports):
        for kind, row in (report.get("decisions") or {}).items():
            into = out.setdefault(kind, {"n": 0, "wall": 0.0, "stages": {}, "counts": {}})
            into["n"] += int(row.get("n", 0))
            into["wall"] += float(row.get("wall", 0.0))
            for name, (wall, calls) in (row.get("stages") or {}).items():
                stage = into["stages"].setdefault(name, [0.0, 0])
                stage[0] += float(wall)
                stage[1] += int(calls)
            for name, value in (row.get("counts") or {}).items():
                into["counts"][name] = into["counts"].get(name, 0) + int(value)
    return out


#: Per-decision columns: (heading, where it comes from). A count, or a stage's calls.
DECISION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("passes", "count:forward.passes"),
    ("leaves", "count:leaves"),
    ("complet.", "count:completions"),
    *((f"fills@{p}", f"count:fills@{p}") for p in PURPOSES),
    ("cells", "count:fill.cells"),
    ("LPs", "calls:lp"),
)


def print_decisions(found: dict[str, dict[str, Any]]) -> None:
    """Table 3: what one decision of each kind costs, in counts and in wall clock."""
    if not found:
        print("\n  per decision: no worker reported any (a report from before IKA-98)")
        return
    heads = [h for h, _source in DECISION_COLUMNS]
    print("\n  per decision, over every worker (counts are the part a busy machine cannot move)")
    print(f"  {'kind':<22} {'n':>7} {'wall ms':>8} " + " ".join(f"{h:>10}" for h in heads))
    for kind, row in sorted(found.items(), key=lambda kv: -kv[1]["n"]):
        n = max(row["n"], 1)
        cells = []
        for _head, source in DECISION_COLUMNS:
            what, name = source.split(":", 1)
            counted = row["counts"].get(name, 0)
            value = counted if what == "count" else row["stages"].get(name, [0.0, 0])[1]
            cells.append(f"{value / n:>10.2f}")
        print(f"  {kind:<22} {row['n']:>7,} {1000 * row['wall'] / n:>8.1f} " + " ".join(cells))
    print("  `between` is a game's end, its record and the next game's set-up; a move's row\n"
          "  holds advancing the game to it as well as searching it.")


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Least squares y = a + b x; None when x does not vary. As scratchpad/rest_split.py."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return my - b * mx, b


def _games_decisions(report: dict[str, Any]) -> int | None:
    """Decisions recorded in the games file a worker's argv names, or None without one."""
    argv = [str(item) for item in report.get("argv") or []]
    if "--out" not in argv[:-1]:
        return None
    path = Path(argv[argv.index("--out") + 1])
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    decisions = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                decisions += len(json.loads(line).get("decisions") or [])
    return decisions


def rest_fit(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Instrument 7: each worker's rest against its decisions, and whether it is too large.

    The same regression as scratchpad/rest_split.py. Workers start together and the queue
    decides how many games each plays, so the intercept is the per-process fixed part and
    the slope the untimed work per decision. The decisions are `timing.decided`'s own
    count when the report has one -- it also sees the games a worker lost or discarded --
    and the games file's otherwise; both are kept, so a gap between them shows.
    """
    rows = []
    for report in _worker_reports(reports):
        stages = report.get("stages") or {}
        measured = sum(v["wall"] for k, v in stages.items() if k not in BORROWED)
        timed = sum(
            int(row.get("n", 0))
            for kind, row in (report.get("decisions") or {}).items()
            if not kind.startswith("between")
        )
        recorded = _games_decisions(report)
        x = timed if report.get("decisions") else recorded
        if x is None:
            continue
        rows.append({
            "elapsed": float(report.get("elapsed", 0.0)),
            "rest": float(report.get("elapsed", 0.0)) - measured,
            "decisions": x,
            "timed": timed if report.get("decisions") else None,
            "recorded": recorded,
        })
    if not rows:
        return None
    line = _fit([r["decisions"] for r in rows], [r["rest"] for r in rows])
    elapsed = sum(r["elapsed"] for r in rows)
    decisions = sum(r["decisions"] for r in rows)
    out: dict[str, Any] = {
        "workers": len(rows),
        "decisions": decisions,
        "recorded": sum(r["recorded"] or 0 for r in rows),
        "timed": sum(r["timed"] or 0 for r in rows) if rows[0]["timed"] is not None else None,
        "rest": sum(r["rest"] for r in rows),
        "elapsed": elapsed,
        "fit": None,
    }
    if line is not None and elapsed > 0:
        a, b = line
        out["fit"] = {"intercept": a, "slope": b, "steady": b * decisions / elapsed}
    return out


def print_rest_fit(found: dict[str, Any] | None) -> None:
    if found is None:
        return
    print(f"\n  the rest against decisions, {found['workers']} workers, "
          f"{found['decisions']:,} decisions")
    if found["timed"] is not None:
        print(f"   decisions timing counted {found['timed']:,}; the games files hold "
              f"{found['recorded']:,}")
    fit = found["fit"]
    if fit is None:
        print("   no fit: every worker played the same number of decisions")
        return
    print(f"   rest = {fit['intercept']:.2f} s + {1000 * fit['slope']:.2f} ms x decisions; "
          f"slope x decisions is {100 * fit['steady']:.1f}% of the workers' wall clock")
    if fit["steady"] > UNNAMED_LIMIT:
        print(f"   ** over {100 * UNNAMED_LIMIT:.0f}%: work that grows with the decisions has "
              f"no row. Do not read the table as a breakdown; name it first **")


def _flag(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


def _same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def delivery(
    reports: list[dict[str, Any]], args: argparse.Namespace | None
) -> dict[str, Any]:
    """Did the settings and the code this tool asked for reach the processes that ran?

    Read off each worker's report -- its own `argv` and the `source` of the `timing` it
    imported -- rather than off the command line written here, because a flag the driver
    dropped and a worktree run that imported the main checkout's src both run to the end
    and report as if they had not. `args` is None when only reading a directory.
    """
    here = str(ROOT / "src" / "pokeuraou")
    problems: list[str] = []
    sources: dict[str, int] = {}
    for report in reports:
        source = report.get("source")
        key = source or "(not recorded)"
        sources[key] = sources.get(key, 0) + 1
        if source and args is not None and not _same_path(source, here):
            problems.append(f"pid {report.get('pid')} ran {source}, not {here}")
    workers = [[str(item) for item in r.get("argv") or []] for r in _worker_reports(reports)]
    seen: dict[str, dict[str, int]] = {}
    for argv in workers:
        for flag in ("--limit", "--seed", "--roster", "--selection-book", "--force-lead",
                     "--inference-arm", "--value"):
            value = _flag(argv, flag)
            seen.setdefault(flag, {})
            seen[flag][str(value)] = seen[flag].get(str(value), 0) + 1
        for flag in ("--hide-bench", "--rank-leaf", "--no-bridge"):
            seen.setdefault(flag, {})
            present = "yes" if flag in argv else "no"
            seen[flag][present] = seen[flag].get(present, 0) + 1
    if args is not None and args.workload == "generation":
        wanted = {
            "--limit": args.limit, "--seed": args.seed, "--roster": args.roster,
            "--force-lead": args.force_lead,
        }
        for argv in workers:
            for flag, value in wanted.items():
                if value is not None and _flag(argv, flag) != str(value):
                    problems.append(f"a worker ran {flag} {_flag(argv, flag)}, asked {value}")
            if args.selection_book is not None:
                got = _flag(argv, "--selection-book")
                if got is None or not _same_path(got, str(args.selection_book)):
                    problems.append(f"a worker ran --selection-book {got}, "
                                    f"asked {args.selection_book}")
            if args.uniform_selection and "--selection-book" in argv:
                problems.append("a worker drew from a book under --uniform-selection")
            if args.hide_bench != ("--hide-bench" in argv):
                problems.append(f"a worker's --hide-bench is {'--hide-bench' in argv}, "
                                f"asked {args.hide_bench}")
    if args is not None:
        for argv in workers:
            missing = [item for item in args.rest if item not in argv]
            if missing:
                problems.append(f"a worker's argv lacks {missing} from the tail")
    return {"sources": sources, "workers": len(workers), "seen": seen,
            "problems": sorted(set(problems))}


def print_delivery(found: dict[str, Any]) -> None:
    print(f"\n  what the {found['workers']} workers ran, from their own reports")
    for source, n in sorted(found["sources"].items()):
        print(f"   source {source}  x{n}")
    for flag, values in found["seen"].items():
        shown = ", ".join(f"{value} x{n}" for value, n in sorted(values.items()))
        print(f"   {flag:<18} {shown}")
    if found["problems"]:
        print("  ** the run is not the one asked for **")
        for problem in found["problems"]:
            print(f"   {problem}")


def build(args: argparse.Namespace) -> list[str]:
    """The real driver's command line for the named workload.

    Whatever follows `--` is handed to the driver's own tail, unchanged. It is not a
    convenience: `match_queue.py` refuses to run without a `--selection-book` in that
    tail, because a uniform draw is worth about -141 Elo and a timing run played that way
    would be timing an agent nobody plays.
    """
    tail = list(args.rest)
    python = sys.executable

    def given(flag: str, value: Any) -> list[str]:  # noqa: ANN401
        return [flag, str(value)] if value is not None else []

    if args.workload == "generation":
        # Every option `generate_queue.py` takes, and none of them defaulted here: an
        # option left unset is left off, so the driver's own default is what runs. This
        # tool had its own defaults for two of them (width 24, 8 workers) while shipping
        # ran 12 and 24, and no way to pass a book or a seed (IKA-98).
        return [
            python, str(ROOT / "tools" / "generate_queue.py"),
            "--out", str(args.out),
            "--games", str(args.games),
            *given("--first-game", args.first_game),
            *given("--workers", args.workers),
            *given("--seed", args.seed),
            *given("--roster", args.roster),
            "--value", str(args.value),
            *given("--limit", args.limit),
            *given("--selection-book", args.selection_book),
            *(["--uniform-selection"] if args.uniform_selection else []),
            *given("--force-lead", args.force_lead),
            "--device", args.device,
            *(["--no-bridge"] if args.no_bridge else []),
            *(["--served", "--servers", str(args.servers)] if args.served else []),
            *(["--hide-bench"] if args.hide_bench else []),
            *(["--", *tail] if tail else []),
        ]
    if args.workload == "match":
        return [
            python, str(ROOT / "tools" / "match_queue.py"),
            "--out", str(args.out),
            "--games", str(args.games),
            "--workers", str(args.workers or (24 if args.served else 6)),
            "--value", str(args.value),
            "--baseline", str(args.baseline),
            "--device", args.device,
            *(["--no-bridge"] if args.no_bridge else []),
            *(["--served", "--servers", str(args.servers)] if args.served else []),
            *(["--hide-bench"] if args.hide_bench else []),
            *(["--", *tail] if tail else []),
        ]
    return [
        python, str(ROOT / "tools" / "why_action.py"),
        "--value", str(args.value),
        "--case", args.case,
        "--limit", str(args.limit if args.limit is not None else 24),
        *tail,
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workload", choices=("generation", "match", "analysis"))
    ap.add_argument("--games", type=int, default=200)
    # For generation, these are generate_queue.py's own options, passed only when given, so
    # an unset one runs the driver's default rather than one of this tool's.
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="analysis: 24 when unset")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--first-game", type=int, default=None)
    ap.add_argument("--roster", default=None)
    ap.add_argument("--selection-book", type=Path, default=None)
    ap.add_argument("--uniform-selection", action="store_true")
    ap.add_argument("--force-lead", default=None)
    ap.add_argument("--value", default="data/models/value-gen11L.pt")
    ap.add_argument("--baseline", default="data/models/value-gen10.pt")
    ap.add_argument("--case", default="sash-ko", help="analysis: a human_baseline case")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--served", action="store_true")
    ap.add_argument("--servers", type=int, default=2)
    ap.add_argument("--hide-bench", action="store_true")
    ap.add_argument("--no-bridge", action="store_true", help="POKEURAOU_RUST_NODE=0")
    ap.add_argument("--out", type=Path, default=None, help="where the run writes its games")
    ap.add_argument("--timing-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None,
                    help="skip the run and read the reports already in this directory")
    ap.add_argument("--json", type=Path, default=None, help="also write the summary here")
    ap.add_argument(
        "--memory-floor",
        type=float,
        default=2.0,
        help="GB of free memory below which the run is killed. The analysis workload has "
        "reached 15.3 GB of RSS and 0.3 GB free on this machine; a measurement is not "
        "worth taking the machine down for.",
    )
    # The tail is cut off `sys.argv` by hand rather than taken with `argparse.REMAINDER`.
    # REMAINDER after a positional grabs everything that follows it, including this
    # tool's own options: `match --games 2 --served -- --selection-book X` parsed as
    # games=200 (the default), served=False, and a tail holding the lot. It ran a
    # 400-seat-game match before anyone noticed, which is exactly the failure mode of a
    # flag that is silently not applied.
    argv = sys.argv[1:]
    tail: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, tail = argv[:cut], argv[cut + 1 :]
    args = ap.parse_args(argv)
    args.rest = tail
    generation_only = {
        "--seed": args.seed, "--first-game": args.first_game, "--roster": args.roster,
        "--selection-book": args.selection_book, "--force-lead": args.force_lead,
        "--uniform-selection": args.uniform_selection or None,
    }
    stray = [flag for flag, value in generation_only.items() if value is not None]
    if args.workload != "generation" and stray and args.report is None:
        # Accepted and then not passed on is how a run measures an agent nobody asked for.
        raise SystemExit(f"{', '.join(stray)} are generate_queue.py's; the {args.workload} "
                         f"workload takes them after --, if its driver has them")

    if args.report is not None:
        reports = collect(args.report)
        if not reports:
            raise SystemExit(f"no timing-*.json in {args.report}")
        workers, servers = _split(reports)
        summary = summarise(workers or reports)
        print_table(summary, None, max(float(r.get("elapsed", 0.0)) for r in reports))
        if servers:
            print_server(summarise(servers))
        print_decisions(per_decision(reports))
        print_rest_fit(rest_fit(reports))
        print_delivery(delivery(reports, None))
        return

    stamp = time.strftime("%m%d-%H%M%S")
    timing_dir = args.timing_dir or (ROOT / "data" / "timing" / f"{args.workload}-{stamp}")
    timing_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        args.out = ROOT / "data" / "timing" / f"{args.workload}-{stamp}-games"

    env = dict(os.environ)
    env["POKEURAOU_TIMING"] = str(timing_dir)
    env["PYTHONPATH"] = str(ROOT / "src")
    # The drivers set this for their workers; the analysis workload has no driver, so it
    # is set here and the drivers overwrite it with the same value.
    env["POKEURAOU_RUST_NODE"] = "0" if args.no_bridge else "1"

    command = build(args)
    print("  " + " ".join(command), file=sys.stderr)
    print(f"  timing -> {timing_dir}", file=sys.stderr, flush=True)

    started = time.perf_counter()
    process = subprocess.Popen(command, env=env, cwd=str(ROOT))  # noqa: S603
    stop = threading.Event()
    tree: dict[str, Any] = {}

    def poll() -> None:
        tree.update(tree_cpu(process.pid, stop, floor_gb=args.memory_floor))

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()
    code = process.wait()
    stop.set()
    watcher.join(timeout=10)
    wall = time.perf_counter() - started
    if code != 0:
        print(f"  the run exited {code}; the table below is of whatever it did first",
              file=sys.stderr)

    reports = collect(timing_dir)
    if not reports:
        raise SystemExit(
            f"the run wrote no timing reports into {timing_dir}. Either it died before "
            f"any process exited, or POKEURAOU_TIMING did not reach it."
        )
    workers, servers = _split(reports)
    summary = summarise(workers or reports)
    summary["command"] = command
    summary["wall"] = wall
    summary["tree"] = tree
    print_table(summary, tree, wall)
    if servers:
        server_summary = summarise(servers)
        summary["server"] = server_summary
        print_server(server_summary)
        summary["spin"] = spin(tree, server_summary, reports)
        print_spin(summary["spin"])
    summary["decisions"] = per_decision(reports)
    print_decisions(summary["decisions"])
    summary["rest_fit"] = rest_fit(reports)
    print_rest_fit(summary["rest_fit"])
    summary["delivery"] = delivery(reports, args)
    print_delivery(summary["delivery"])
    target = args.json or (timing_dir / "summary.json")
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {target}")
    if summary["delivery"]["problems"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
