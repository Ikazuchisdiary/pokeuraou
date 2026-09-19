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

from pokeuraou.timing import BORROWED, STAGES  # noqa: E402

#: Rows printed under a heading, so a table reads as a breakdown rather than a list.
GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Python", ("narrow", "branch", "encode", "forward", "lp")),
    ("bridge", ("rust.fill", "rust.ask", "rust.header", "rust.body", "rust.unpack",
                "rust.resolve", "rust.score")),
    ("served", ("serve.copy", "serve.wait")),
    ("child", ("rust.child.resolve", "rust.child.encode", "rust.child.parse",
               "rust.child.header")),
    ("server", ("server.held", "server.queue")),
    ("inclusive", ("refused",)),
)

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
    return {
        "by_name": by_name,
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
    print(f"   {'the rest':<19} {_seconds(elapsed - accounted):>10} "
          f"{share(elapsed - accounted):>7}   simulator bridge, I/O, startup, interpreter")
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
        print(f"   peak RSS of the tree: {tree['peak_rss'] / 1e9:.1f} GB; "
              f"free memory fell to {tree.get('free_low_water', 0.0):.1f} GB")
        if tree.get("stopped_for_memory"):
            print("   ** the run was killed at the memory floor: the table above is of "
                  "what it managed first **")
        if wall > 0:
            print(f"   busy cores: {tree['total'] / wall:.1f}")


def build(args: argparse.Namespace) -> list[str]:
    """The real driver's command line for the named workload.

    Whatever follows `--` is handed to the driver's own tail, unchanged. It is not a
    convenience: `match_queue.py` refuses to run without a `--selection-book` in that
    tail, because a uniform draw is worth about -141 Elo and a timing run played that way
    would be timing an agent nobody plays.
    """
    tail = list(args.rest)
    python = sys.executable
    if args.workload == "generation":
        return [
            python, str(ROOT / "tools" / "generate_queue.py"),
            "--out", str(args.out),
            "--games", str(args.games),
            "--workers", str(args.workers or 8),
            "--value", str(args.value),
            "--limit", str(args.limit),
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
        "--limit", str(args.limit),
        *tail,
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workload", choices=("generation", "match", "analysis"))
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=24)
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

    if args.report is not None:
        reports = collect(args.report)
        if not reports:
            raise SystemExit(f"no timing-*.json in {args.report}")
        workers, servers = _split(reports)
        summary = summarise(workers or reports)
        print_table(summary, None, max(float(r.get("elapsed", 0.0)) for r in reports))
        if servers:
            print_server(summarise(servers))
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
    target = args.json or (timing_dir / "summary.json")
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  {target}")


if __name__ == "__main__":
    main()
