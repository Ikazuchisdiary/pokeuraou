"""What is "the rest" in a profile_stages generation table: startup, or per-game work?

Asked on 2026-09-23 after IKA-97 quoted the IKA-54 run's "the rest 28.8%" as startup
without having measured it. `timing.elapsed` runs from the import of `pokeuraou.timing` to
exit, and "the rest" is elapsed minus every stage that has a timer -- so it holds startup
AND whatever per-game work has no timer. Those two scale differently: startup is paid once
a process, the other once a decision. A worker's game count is not chosen (the queue deals
them), so workers inside one run played different numbers of decisions from the same
start, and the slope of rest against decisions separates the two.

Reads only what the runs already wrote: each worker's report (`timing-*.json`) and the
games file its argv names. No timing is taken here.

    uv run python scratchpad/rest_split.py data/timing/generation-0920-044352 ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from pokeuraou.timing import BORROWED  # noqa: E402


def games_file(report: dict) -> Path | None:
    argv = report.get("argv") or []
    for flag, value in zip(argv, argv[1:], strict=False):
        if flag == "--out":
            return Path(value)
    return None


def load(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("timing-*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        argv = report.get("argv") or []
        if not any(str(item).endswith("selfplay.py") for item in argv):
            continue
        stages = report.get("stages") or {}
        measured = sum(v["wall"] for k, v in stages.items() if k not in BORROWED)
        staged_cpu = sum(v["cpu"] for k, v in stages.items() if k not in BORROWED)
        games = decisions = 0
        out = games_file(report)
        if out is not None and out.exists():
            with out.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        games += 1
                        decisions += len(json.loads(line).get("decisions") or [])
        process = report.get("process") or {}
        cpu = float(process.get("cpu_user", 0.0)) + float(process.get("cpu_system", 0.0))
        rows.append({
            "elapsed": float(report["elapsed"]),
            "measured": measured,
            "rest": float(report["elapsed"]) - measured,
            "cpu": cpu,
            "staged_cpu": staged_cpu,
            "games": games,
            "decisions": decisions,
            "hidden": "--hide-bench" in argv,
            "served": bool(report.get("served")),
        })
    return rows


def fit(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Least squares y = a + b x. None when x does not vary."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx
    return my - b * mx, b


for name in sys.argv[1:]:
    run = Path(name)
    rows = load(run)
    if not rows:
        print(f"{run.name}: no selfplay reports")
        continue
    total = sum(r["elapsed"] for r in rows)
    rest = sum(r["rest"] for r in rows)
    games = sum(r["games"] for r in rows)
    decisions = sum(r["decisions"] for r in rows)
    flags = ("hidden " if rows[0]["hidden"] else "open ") + ("served" if rows[0]["served"] else "direct")
    print(f"\n{run.name}  {len(rows)} workers  {games} games  {decisions} decisions  {flags}")
    print(f"  rest {rest:7.1f} s of {total:7.1f} s = {100 * rest / total:4.1f}%   "
          f"{rest / len(rows):5.2f} s a worker   {rest / max(decisions, 1) * 1000:6.1f} ms a decision")
    for r in sorted(rows, key=lambda r: r["decisions"]):
        print(f"    games {r['games']:4d}  decisions {r['decisions']:5d}  elapsed {r['elapsed']:7.2f}  "
              f"rest {r['rest']:6.2f}  cpu {r['cpu']:7.2f}  untimed cpu {r['cpu'] - r['staged_cpu']:6.2f}")
    line = fit([r["decisions"] for r in rows], [r["rest"] for r in rows])
    if line is not None:
        a, b = line
        print(f"  rest = {a:5.2f} s + {b * 1000:5.2f} ms x decisions   (within this run, {len(rows)} points)")
