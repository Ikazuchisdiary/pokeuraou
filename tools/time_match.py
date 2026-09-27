"""A match between two clocks: the human-play agent on both seats (IKA-333).

    uv run python tools/time_match.py --out data/time/100ms-1s \\
        --arm long:seconds=1 --arm short:seconds=0.1 --pairs 400 --sprt 0 10 --parallel 2
    uv run python tools/time_match.py --out data/time/g16 \\
        --arm g16:seconds=10,levels=16 --arm g8:seconds=10,levels=8 --pairs 200 --sprt 0 10

The game and the conditions are `pokeuraou.timematch`'s (its docstring says how the two
seats read, what is outside the clock and what a pair is). A condition is
``name:key=value,...`` with the keys ``seconds`` (required), ``threads``, ``cores``,
``clock`` (wall|count), ``oracle`` (sall|none|s<W>), ``levels`` (the deepening's guard; 0:
`deepen.MAX_LEVELS` unrecorded), ``width_only`` and ``child_q``. A key left out is the
human-play default -- what
``tools/play_human.py`` plays with no flag (IKA-343) -- and so are the leaf
(`play_human.DEFAULT_VALUE`), the menus (``q-nocover`` by `qrank.DEFAULT_Q`; a match stops
without the Q, as a board does, IKA-338) and the bench belief. **The first --arm is the
tested condition**: the Elo is its, and ``--sprt E0 E1`` tests it (`pokeuraou.sprt`: H1
is "it is worth E1 or more"). ``--pairs`` is the cap.

The pairs are handed out by `workqueue.run_workers` to ``--parallel`` processes, one game at
a time in each. Each process is held to its own share of the logical cores (``--cpu-sets``;
by default the machine split evenly, so ``--parallel 2`` gives each game four physical
cores, what a person's game uses), and the processes it starts inherit it. A process that
fails stops the run (``workers.json``, IKA-336), and so does a process whose deepening
workers died or whose memory watch stopped a move's deepening (``--allow-memory-stops``).
The SPRT reads the pairs in index order, once the queue has resolved them.

A run longer than one sitting is played in pieces: ``--resume`` with the next ``--start``
plays more pairs into the same ``--out`` (the settings must match; the SPRT goes on in pair
order; each piece's ``logs`` and ``workers.json`` are kept as ``logs.<n>``, ``workers.<n>.json``).

Output in ``--out``: ``settings.json`` (everything that decides the games, written before
the first; ``chunks`` lists the pieces), ``games-worker<k>.jsonl`` (`timematch.game_line`),
``logs/`` (each process echoes its settings at its start), ``sprt.json`` with ``--sprt``,
``summary.json`` at the end (the Elo with its interval, and per condition the seconds against the budget, the
width, the cells and the deepening's steps and depth).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from play_human import DEFAULT_VALUE, Q_FILL, leaf_name, memory_watch  # noqa: E402

from pokeuraou import analysis, humanplay, qrank, timematch  # noqa: E402
from pokeuraou.hidden import DEFAULT_BENCH_DROP, parse_bench_drop  # noqa: E402
from pokeuraou.selfplay import MAX_TURNS  # noqa: E402
from pokeuraou.sprt import Sprt  # noqa: E402
from pokeuraou.workqueue import WorkClient, run_workers  # noqa: E402

GAME_FILES = "games-worker*.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="a condition, name:key=value,... (twice; the first is tested)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, required=True, help="pairs to play at most (2 games each)")
    ap.add_argument("--start", type=int, default=0, help="the first pair's index")
    ap.add_argument("--seed", type=int, default=33300)
    ap.add_argument("--sprt", type=float, nargs=2, default=None, metavar=("ELO0", "ELO1"))
    ap.add_argument("--sprt-alpha", type=float, default=0.05)
    ap.add_argument("--sprt-beta", type=float, default=0.05)
    ap.add_argument("--parallel", type=int, default=2, help="games at once, one process each")
    ap.add_argument("--cpu-sets", default=None,
                    help="each process's logical cores, ';' between processes (e.g. 0-7;8-15), "
                    "or none. Default: the machine split evenly when --parallel > 1, else none")
    ap.add_argument("--pool", default="regmc-matchupweb")
    ap.add_argument("--value", type=Path, nargs="+", default=None,
                    help=f"the leaf. Default: {' '.join(DEFAULT_VALUE)}")
    ap.add_argument("--q-model", type=Path, default=None, help=f"the Q. Default: {qrank.DEFAULT_Q}")
    ap.add_argument("--bench-drop", default=DEFAULT_BENCH_DROP)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--leaf-graphs", default="on", choices=("on", "off"))
    ap.add_argument("--cuda-memory-gb", type=float, default=humanplay.PLAY_CUDA_MEMORY_GB)
    ap.add_argument("--max-rss-gb", type=float, default=analysis.Limits.rss_gb)
    ap.add_argument("--min-free-gb", type=float, default=analysis.Limits.free_gb)
    ap.add_argument("--max-gpu-gb", type=float, default=analysis.Limits.gpu_gb)
    ap.add_argument("--allow-memory-stops", action="store_true",
                    help="go on after the memory watch stopped a move's deepening (by default a "
                    "process whose pair had one fails, and the run stops: that move was not the "
                    "condition's)")
    ap.add_argument("--poll", type=float, default=15.0, help="seconds between the SPRT's looks")
    ap.add_argument("--resume", action="store_true",
                    help="play more pairs in a run's --out: the same settings (bar the pairs, the "
                    "processes and the checkout's head), the SPRT going on from the pairs in, in "
                    "order. Pairs already in are not played again")
    # A worker's own.
    ap.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--address", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cpus", default="", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def conditions(args: argparse.Namespace) -> tuple[timematch.Condition, timematch.Condition]:
    if len(args.arm) != 2:
        raise SystemExit(f"--arm twice: the tested condition, then the other ({len(args.arm)} given)")
    try:
        tested, other = (timematch.parse_condition(spec) for spec in args.arm)
    except ValueError as problem:
        raise SystemExit(f"--arm: {problem}") from problem
    if tested.name == other.name:
        raise SystemExit("the two conditions need different names")
    for c in (tested, other):
        priced = (humanplay.Agent.form, c.price_cores) in humanplay.COSTS
        if c.clock == "count" and not c.width_only and not priced:
            raise SystemExit(f"{c.name}: the count clock has prices for 1 core only (deepen.COSTS)")
    return tested, other


def files(args: argparse.Namespace) -> tuple[list[Path], Path]:
    from pokeuraou.regulation import repo_root

    values = args.value or [repo_root() / p for p in DEFAULT_VALUE]
    q_path = args.q_model or (repo_root() / qrank.DEFAULT_Q)
    missing = [str(p) for p in [*values, q_path] if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing: {', '.join(missing)} (--value / --q-model name them)")
    return [Path(v) for v in values], Path(q_path)


def cpu_sets(spec: str | None, parallel: int) -> list[list[int]]:
    if spec is None:
        if parallel <= 1:
            return [[] for _ in range(parallel)]
        logical = os.cpu_count() or 1
        share = logical // parallel
        if share < 1:
            raise SystemExit(f"{parallel} processes do not fit on {logical} logical cores")
        return [list(range(k * share, (k + 1) * share)) for k in range(parallel)]
    if spec == "none":
        return [[] for _ in range(parallel)]
    sets = []
    for part in spec.split(";"):
        cores: list[int] = []
        for item in part.split(","):
            lo, _, hi = item.partition("-")
            cores.extend(range(int(lo), int(hi or lo) + 1))
        sets.append(cores)
    if len(sets) != parallel:
        raise SystemExit(f"--cpu-sets names {len(sets)} sets for {parallel} processes")
    return sets


def settings(args: argparse.Namespace, tested, other, values, q_path) -> dict:  # noqa: ANN001
    import hashlib

    from pokeuraou.pool import find_pool

    try:
        head = subprocess.run(  # noqa: S603
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        head = ""
    pool_path = find_pool(args.pool)
    return {
        "tool": "time_match",
        "head": head,
        "tested": tested.to_json(),
        "other": other.to_json(),
        "leaf": [str(v) for v in values],
        "leafName": leaf_name(values),
        "rankFill": Q_FILL,
        "qModel": str(q_path),
        "benchDrop": args.bench_drop,
        "pool": {"id": args.pool, "sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest()},
        "seed": args.seed,
        "pairs": [args.start, args.start + args.pairs],
        "maxTurns": args.max_turns,
        "parallel": args.parallel,
        "cpuSets": cpu_sets(args.cpu_sets, args.parallel),
        "sprt": None if args.sprt is None else {
            "elo0": args.sprt[0], "elo1": args.sprt[1], "alpha": args.sprt_alpha,
            "beta": args.sprt_beta,
        },
        "cudaMemoryGb": args.cuda_memory_gb,
        "limits": {"rssGb": args.max_rss_gb, "freeGb": args.min_free_gb, "gpuGb": args.max_gpu_gb},
        "leafGraphs": args.leaf_graphs,
    }


# ----------------------------------------------------------------------------- the worker


def worker(args: argparse.Namespace) -> None:
    import numpy as np

    from pokeuraou import deepen
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.pool import draw_pair, load_pool

    if args.cpus:
        import psutil

        psutil.Process().cpu_affinity([int(c) for c in args.cpus.split(",")])
    tested, other = conditions(args)
    values, q_path = files(args)
    parse_bench_drop(args.bench_drop)
    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    humanplay.cap_cuda(args.cuda_memory_gb, args.device)
    evaluate, encoder, device = humanplay.load_leaf(
        reg, values, args.device, graphs=args.leaf_graphs == "on"
    )
    q = qrank.LocalQ(q_path, encoder, device=device)
    qrank.install(q)
    threads = max(tested.threads, other.threads)
    humanplay.use_threads(
        threads, reg,
        ([str(v) for v in values], str(device), args.leaf_graphs == "on", args.cuda_memory_gb),
    )
    started_workers = deepen.workers(reg)
    halt = threading.Event()
    watch = memory_watch(
        analysis.Limits(rss_gb=args.max_rss_gb, free_gb=args.min_free_gb, gpu_gb=args.max_gpu_gb),
        halt, lambda text: print(text, file=sys.stderr, flush=True),
    )
    match = timematch.Match(
        reg=reg, evaluate=evaluate, leaf_name=leaf_name(values), rank_fill=Q_FILL,
        bench_drop=args.bench_drop, tested=tested, other=other, seed=args.seed,
        max_turns=args.max_turns, halt=halt,
    )
    # The echo: what this process plays, as it resolved it (read this, not the command).
    print(
        f"worker {args.worker}: cpus {args.cpus or 'all'} "
        f"(affinity {len(__import__('psutil').Process().cpu_affinity())} logical)\n"
        f"  tested {tested.describe()}\n  other  {other.describe()}\n"
        f"  leaf {match.leaf_name} ({', '.join(str(v) for v in values)}) on {device}, "
        f"menus {Q_FILL} ({', '.join(q.describe())}), bench drop {args.bench_drop}\n"
        f"  deepening workers {started_workers}, threads {threads}, "
        f"CUDA cap {args.cuda_memory_gb} GB, seed {args.seed}, max turns {args.max_turns}",
        file=sys.stderr, flush=True,
    )
    out = args.out / f"games-worker{args.worker}.jsonl"
    with WorkClient(args.address) as client:
        while True:
            pair = client.take()
            if pair is None:
                break
            _k, a, b = draw_pair(np.random.default_rng([args.seed, pair]), pool.pairs)
            teams = (pool.teams[a], pool.teams[b])
            lines = timematch.play_pair(match, pair, teams)
            alive = deepen.workers_alive(reg)
            peak = watch.peak
            for line in lines:
                line["deepenWorkers"] = [started_workers, alive]
                line["worker"] = args.worker
                # The memory watch's worst so far in this process: its own and its workers'
                # resident memory, the host's least free, the card's most held (all
                # processes, so both games of a parallel run).
                line["peak"] = {"rssGb": round(peak.rss_gb, 2), "leastFreeGb": round(peak.free_gb, 2),
                                "gpuGb": round(peak.gpu_gb, 2)}
            with out.open("ab") as handle:
                for line in lines:
                    handle.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
            if alive < started_workers:
                raise SystemExit(
                    f"a deepening worker died during pair {pair} ({alive} of {started_workers} "
                    "alive): the reads after it were slower than the configured agent's"
                )
            braked = sum(line["memoryStops"] for line in lines)
            if braked and not args.allow_memory_stops:
                # A move the memory watch stopped did not read what its condition says: the
                # games are not the agents compared. 9/27: a card shared with other jobs
                # reached 11.1 GB and stopped 912 moves of a node-time run.
                raise SystemExit(
                    f"the memory watch stopped {braked} move(s) of pair {pair} (peak "
                    f"{lines[-1]['peak']}): those moves were not the configured agents'. Fewer "
                    "--parallel, or --allow-memory-stops to keep going"
                )
            client.finish(pair)
            print(
                f"pair {pair}: " + ", ".join(
                    f"game {ln['game']} {ln['conditions'][0]}/{ln['conditions'][1]} "
                    f"outcome {ln['outcome']} {ln['turns']} turns {ln['seconds']:.1f}s"
                    for ln in lines
                ),
                file=sys.stderr, flush=True,
            )
    watch.close()


# ----------------------------------------------------------------------------- the driver


def read_lines(out: Path) -> list[dict]:
    lines = []
    for path in sorted(out.glob(GAME_FILES)):
        for raw in path.read_bytes().splitlines():
            if raw.strip():
                lines.append(json.loads(raw))
    return lines


class Monitor:
    """Feeds the SPRT the pairs in index order, each once the queue has resolved it."""

    def __init__(self, out: Path, start: int, test: Sprt | None, before: set[int] | None = None) -> None:
        self.out = out
        self.next = start
        self.test = test
        #: Pairs written by earlier invocations of the run (``--resume``): resolved already.
        self.before = set(before or ())
        self.trail: list[float] = []
        self.skipped: dict[int, str] = {}
        self.stopped_at: int | None = None

    def __call__(self, queue) -> str | None:  # noqa: ANN001
        resolved = queue.resolved() | self.before
        scores, left = timematch.pair_scores(read_lines(self.out))
        while self.next in resolved:
            pair = self.next
            self.next += 1
            if pair in left or pair not in scores:
                self.skipped[pair] = left.get(pair, "not written")
                continue
            if self.test is None:
                continue
            self.test.add(scores[pair])
            self.trail.append(round(self.test.llr, 4))
            if self.stopped_at is None and self.test.decision is not None:
                self.stopped_at = pair
                self.save()
                return (
                    f"SPRT({self.test.elo0:+g}, {self.test.elo1:+g}) accepted "
                    f"{self.test.decision} after {self.test.pairs} pairs, LLR {self.test.llr:+.3f}"
                )
        done = [scores[p] for p in sorted(scores)]
        got = timematch.elo_interval(done)
        print(
            f"  {len(done)} pairs scored: tested {got.get('elo', '-')} Elo "
            f"[{got.get('low', '-')}, {got.get('high', '-')}]"
            + (f", LLR {self.test.llr:+.3f} over {self.test.pairs} in order" if self.test else ""),
            file=sys.stderr, flush=True,
        )
        if self.test is not None:
            self.save()
        return None

    def state(self) -> dict:
        if self.test is None:
            return {}
        lost, split, won = self.test.counts
        return {
            "registered": self.test.registration(),
            "decision": self.test.decision,
            "pairs": self.test.pairs,
            "counts": {"lost": lost, "split": split, "won": won},
            "llr": self.test.llr,
            "stoppedAtPair": self.stopped_at,
            "skippedPairs": {str(k): v for k, v in self.skipped.items()},
            "llrTrail": self.trail,
        }

    def save(self) -> None:
        if self.test is not None:
            (self.out / "sprt.json").write_bytes(
                (json.dumps(self.state(), indent=1) + "\n").encode("utf-8")
            )


def summary(out: Path, tested, other, monitor: Monitor, outcome: dict) -> dict:  # noqa: ANN001
    lines = read_lines(out)
    scores, left = timematch.pair_scores(lines)
    ordered = timematch.prefix(scores, left, min([ln["pair"] for ln in lines], default=0))
    return {
        "tested": tested.name,
        "other": other.name,
        "games": len(lines),
        "elo": timematch.elo_interval([scores[p] for p in sorted(scores)]),
        "eloInOrder": timematch.elo_interval(ordered),
        "noWinner": {str(k): v for k, v in left.items()},
        "sprt": monitor.state(),
        "conditions": {c.name: timematch.move_profile(lines, c.name) for c in (tested, other)},
        "fallbacks": {
            c.name: sum(ln["fallbacks"][s] for ln in lines for s in (0, 1)
                        if ln["conditions"][s] == c.name)
            for c in (tested, other)
        },
        "memoryStops": sum(ln["memoryStops"] for ln in lines),
        "peak": {
            "rssGb": max((ln["peak"]["rssGb"] for ln in lines), default=None),
            "leastFreeGb": min((ln["peak"]["leastFreeGb"] for ln in lines), default=None),
            "gpuGb": max((ln["peak"]["gpuGb"] for ln in lines), default=None),
        },
        "gameSeconds": timematch._quantiles([ln["seconds"] for ln in lines]),
        "turns": timematch._quantiles([ln["turns"] for ln in lines]),
        "workers": outcome,
    }


def driver(args: argparse.Namespace, argv: list[str]) -> int:
    tested, other = conditions(args)
    values, q_path = files(args)
    parse_bench_drop(args.bench_drop)
    out: Path = args.out
    sets = cpu_sets(args.cpu_sets, args.parallel)
    fixed = settings(args, tested, other, values, q_path)
    chunk = {k: fixed[k] for k in ("head", "pairs", "parallel", "cpuSets")}
    before: set[int] = set()
    first = args.start
    if args.resume:
        old = json.loads((out / "settings.json").read_bytes())
        differs = sorted(k for k in set(old) | set(fixed)
                         if k not in (*chunk, "chunks") and old.get(k) != fixed.get(k))
        if differs:
            raise SystemExit(f"--resume with other settings than {out}'s: {', '.join(differs)}")
        before = {int(ln["pair"]) for ln in read_lines(out)}
        overlap = sorted(before & set(range(args.start, args.start + args.pairs)))
        if overlap:
            raise SystemExit(f"pairs {overlap[0]}..{overlap[-1]} are in already: start past them")
        first = old["pairs"][0]
        n = len(old.get("chunks", []))
        # run_workers writes these afresh; the earlier invocations' are kept beside them.
        for name in ("logs", "workers.json"):
            if (out / name).exists():
                (out / name).rename(out / f"{name}.{n}" if name == "logs" else out / f"workers.{n}.json")
        old["chunks"] = [*old.get("chunks", []), chunk]
        fixed = old
    elif out.exists() and any(out.glob(GAME_FILES)):
        raise SystemExit(f"{out} already holds games: a run is one directory (--resume adds to it)")
    else:
        fixed["chunks"] = [chunk]
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.json").write_bytes((json.dumps(fixed, indent=1) + "\n").encode("utf-8"))
    print(
        f"time match -> {out}\n  tested {tested.describe()}\n  other  {other.describe()}\n"
        f"  leaf {fixed['leafName']}, menus {Q_FILL} by {q_path}, bench drop {args.bench_drop}, "
        f"pool {args.pool}\n  pairs {args.start}..{args.start + args.pairs - 1} (cap), seed "
        f"{args.seed}, {args.parallel} process(es) on "
        + ("; ".join(f"{s[0]}-{s[-1]}" if s else "all" for s in sets)),
        file=sys.stderr, flush=True,
    )
    test = None
    if args.sprt is not None:
        if (out / "sprt.json").exists() and not args.resume:
            raise SystemExit(f"{out / 'sprt.json'} already exists: a registered test belongs to one run")
        test = Sprt(args.sprt[0], args.sprt[1], alpha=args.sprt_alpha, beta=args.sprt_beta)
        if args.resume:
            registered = json.loads((out / "sprt.json").read_bytes()).get("registered")
            if registered != test.registration():
                raise SystemExit(f"--resume with another test than the one registered: {registered}")
        print(
            f"  registered SPRT({test.elo0:+g}, {test.elo1:+g}), alpha {test.alpha}, beta "
            f"{test.beta}: LLR bounds [{test.lower:+.3f}, {test.upper:+.3f}]",
            file=sys.stderr, flush=True,
        )
    monitor = Monitor(out, first, test, before)
    if before:
        # The pairs already in, in order, as the run left them.
        class _Nothing:
            @staticmethod
            def resolved() -> set[int]:
                return set()

        if monitor(_Nothing()) is not None or (test is not None and test.decision is not None):
            raise SystemExit(f"the SPRT of {out} has decided already: {monitor.state()['decision']}")
    monitor.save()

    def command(k: int, address: str) -> list[str]:
        return [sys.executable, str(Path(__file__).resolve()), *argv, "--worker", str(k),
                "--address", address, "--cpus", ",".join(str(c) for c in sets[k])]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    outcome: dict = {}
    code = run_workers(
        range(args.start, args.start + args.pairs), command, workers=args.parallel,
        out_dir=out, env=env, label="time match", monitor=monitor,
        poll=args.poll, max_failures=0, outcome=outcome,
    )
    monitor.save()
    result = summary(out, tested, other, monitor, outcome)
    (out / "summary.json").write_bytes((json.dumps(result, indent=1) + "\n").encode("utf-8"))
    e = result["elo"]
    print(
        f"\n{tested.name} over {other.name}: {e.get('elo', '-')} Elo "
        f"[{e.get('low', '-')}, {e.get('high', '-')}] over {e.get('pairs', 0)} pairs "
        f"(lost/split/won {e.get('counts')}); no winner {len(result['noWinner'])}, "
        f"fallbacks {result['fallbacks']}, memory stops {result['memoryStops']}, "
        f"peak {result['peak']}"
        + (f"\nSPRT: {monitor.state()['decision'] or 'no decision'}, LLR {monitor.state()['llr']:+.3f}"
           if test is not None else ""),
        file=sys.stderr,
    )
    for c in (tested, other):
        p = result["conditions"][c.name]
        print(
            f"  {c.name}: {p['moves']} moves, seconds/budget p50 {p['ratio'].get('p50')} "
            f"max {p['ratio'].get('max')}, width {p['width']}, steps p50 {p['steps'].get('p50')}, "
            f"depth {p['depth']}, node cells p50 {p['nodeCells'].get('p50')}",
            file=sys.stderr,
        )
    print(f"  -> {out / 'summary.json'}", file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.worker is not None:
        worker(args)
        return 0
    return driver(args, argv)


if __name__ == "__main__":
    sys.exit(main())
