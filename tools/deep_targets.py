"""Deeper search values as value targets: how wrong, how often, how dear (IKA-289, measure only).

IKA-289 asks whether relabelling recorded decisions with a deeper search's root value (the
cheap form of IKA-95, shogi's "zokin-shibori") would teach the leaf better than the game's
one outcome, above all on the positions just before the end. This measures, and trains
nothing:

  scan      Every decision of every game under the given directories (parallel over the
            worker files, `--jobs`): its kind, recorded menu product, standing bench per
            side, whether each side's four have all been shown, the recorded search value
            and the game's outcome. One compact line per game.
  reach     From a scan: what share of the training rows (every recorded decision is one)
            are target decisions (move, menu product <= 16 or <= 64, no bench), how many
            games hold one, and how many rows the IKA-95 relabelling would touch (a row is
            relabelled when a target decision comes at or after it in its game).
  read      Per target decision of a seeded sample of games (parallel, `--jobs`): the
            current master's search over the recorded menu, at depth 1, depth 2 (mixed
            and restricted) and best-first deepening at 100 and 200 cells (IKA-33), with
            the process CPU seconds and cells each took. The leaf is the given ensemble.
  endgame   The same readings on IKA-254's answer set (positions solved to the end in the
            search's own game model, `tools/endgame_exact.py`), with the recorded game's
            real outcome beside the exact value.
  analyse   Tables from `read` / `endgame` output: error against the exact value per
            menu-product bucket and against the recorded outcomes (Brier score, bias,
            calibration), the IKA-95 label shift on the sampled games, and the cost.
  mix       (IKA-296) From a `read` of every game and the same games encoded: each reading
            calibrated by isotonic regression on the target rows of the training games, and
            per-row training targets for `tools/train_value.py --target-file`, mixed into the
            target rows only (`<reading>-rows-<lam>`) or into those and every earlier row of
            the game (`<reading>-back-<lam>`, IKA-95's form).

Positions are read open (the recorded position is the true one), as IKA-254 and IKA-33 did.
A target's "no bench" is either no standing Pokemon off the field on either side
(`bench0`) or every standing one already shown to the other side (`shown`: IKA-33's
condition for deepening in a hidden-bench game, both determinizations exact).

    python tools/deep_targets.py scan --out scan.jsonl --jobs 4 data/selfplay-mc0
    python tools/deep_targets.py reach scan.jsonl
    python tools/deep_targets.py read --scan scan.jsonl --games 2000 --seed 1 --jobs 4 \\
        --out read.jsonl --value data/models/value-mc0.pt data/models/value-mc0-s1.pt
    python tools/deep_targets.py endgame --answers <ika254>/main-partial.jsonl \\
        --games-dir data/selfplay-mc0 --out endgame.jsonl --value ...
    python tools/deep_targets.py analyse --read read.jsonl --scan scan.jsonl
    python tools/deep_targets.py analyse --endgame endgame.jsonl

Each `read` worker appends one line per game to `<out>.part<N>` as it finishes it, so a
stopped run keeps what it did.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

#: The readings, in report order: label -> search keyword arguments.
READINGS: dict[str, dict[str, Any]] = {
    "d1": {},
    "d2m": {"depth": 2},
    "d2r": {"depth": 2, "solve_restricted": True},
    "m100": {"deepen": 100},
    "m200": {"deepen": 200},
}

#: Menu-product buckets (upper bounds, inclusive) a target can fall in.
BUCKETS = (16, 64)


# --------------------------------------------------------------------------------------
# scan


def _bench(position: dict[str, Any]) -> list[int]:
    out = []
    for side in position["sides"]:
        n = 0
        for mon in side.get("pokemon") or ():
            if mon.get("fainted") or mon.get("hp", 1) <= 0:
                continue
            if mon.get("activeIndex") is None and mon.get("active_index") is None:
                n += 1
        out.append(n)
    return out


def _summarise(record: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for d in record.get("decisions", ()):
        product = len(d.get("ownActions") or ()) * len(d.get("foeActions") or ())
        bench = _bench(d["position"])
        shown = d.get("shownIdentities") or [[], []]
        sizes = [len(s.get("pokemon") or ()) for s in d["position"]["sides"]]
        seen = [int(bench[s] == 0 or len(shown[s]) >= sizes[s]) for s in (0, 1)]
        rows.append([
            0 if d.get("kind") == "move" else 1, product, bench[0], bench[1],
            seen[0], seen[1], round(float(d.get("searchValue", float("nan"))), 5),
        ])
    return {"outcome": float(record["outcome"]), "decisions": rows}


def _scan_file(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("outcome") is None:
                continue
            out.append({"file": path, "line": line_no, **_summarise(record)})
    return out


def run_scan(args: argparse.Namespace) -> None:
    files = sorted(str(p) for d in args.dirs for p in Path(d).glob("games-worker*.jsonl"))
    with Pool(args.jobs) as pool:
        parts = pool.map(_scan_file, files)
    with open(args.out, "wb") as handle:
        for part in parts:
            for game in part:
                handle.write((json.dumps(game) + "\n").encode("utf-8"))
    print(f"{sum(map(len, parts)):,} games from {len(files)} files -> {args.out}")


def load_scan(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_bytes().splitlines() if x]


def is_target(row: Sequence[Any], max_product: int, bench: str) -> bool:
    kind, product, b0, b1, s0, s1, _ = row
    if kind != 0 or not 0 < product <= max_product:
        return False
    if bench == "bench0":
        return b0 == 0 and b1 == 0
    if bench == "shown":
        return bool(s0 and s1)
    return True  # "any"


# --------------------------------------------------------------------------------------
# reach


def reach_table(games: list[dict[str, Any]]) -> str:
    rows = sum(len(g["decisions"]) for g in games)
    moves = sum(1 for g in games for d in g["decisions"] if d[0] == 0)
    lines = [
        f"{len(games):,} games, {rows:,} training rows (decisions), {moves:,} move decisions",
        "",
        "  target              share of rows  share of moves  games with one  "
        "rows relabelled (IKA-95)  targets/game",
    ]
    for max_product in BUCKETS:
        for bench in ("bench0", "shown", "any"):
            targets = touched = with_one = 0
            for g in games:
                flags = [is_target(d, max_product, bench) for d in g["decisions"]]
                n = sum(flags)
                targets += n
                if n:
                    with_one += 1
                    last = max(k for k, f in enumerate(flags) if f)
                    touched += last + 1
            lines.append(
                f"  <={max_product:<3} {bench:<7}  {targets / rows:12.1%}  {targets / moves:14.1%}  "
                f"{with_one / len(games):14.1%}  {touched / rows:24.1%}  {targets / len(games):12.2f}"
            )
    return "\n".join(lines)


def run_reach(args: argparse.Namespace) -> None:
    print(reach_table(load_scan(args.scan)))


# --------------------------------------------------------------------------------------
# read


class Reader:
    """The current master's search over a recorded menu, every reading, with its cost."""

    def __init__(self, values: Sequence[str], device: str) -> None:
        import torch

        from pokeuraou import deepen as deepen_mod
        from pokeuraou import port as port_mod
        from pokeuraou import search as search_mod

        torch.set_num_threads(1)
        self.torch = torch
        self.values = values
        self.device = device
        self.search_mod = search_mod
        self.reg = None
        self.leaf = None
        self.count = {"cells": 0, "inside": 0}
        count = self.count
        real_turn = port_mod.turn

        def turn(*a: Any, **k: Any) -> Any:
            if not count["inside"]:
                count["cells"] += 1
            return real_turn(*a, **k)

        def counted(real: Any) -> Any:
            def bp(reg: Any, pos: Any, row: Any, col: Any, evaluate: Any, **k: Any) -> Any:
                count["cells"] += len(row) * len(col)
                count["inside"] += 1
                try:
                    return real(reg, pos, row, col, evaluate, **k)
                finally:
                    count["inside"] -= 1
            return bp

        port_mod.turn = turn
        search_mod.batched_payoff = counted(search_mod.batched_payoff)
        deepen_mod.batched_payoff = counted(deepen_mod.batched_payoff)

    def _setup(self, pos: Any) -> None:
        from pokeuraou.damage import register_mega_stones
        from pokeuraou.encode import Encoder
        from pokeuraou.regulation import load_regulation
        from pokeuraou.value import BatchedValue, load_ensemble

        self.reg = load_regulation(pos.format)
        register_mega_stones(self.reg)
        encoder = Encoder(self.reg)
        nets, _ = load_ensemble([Path(v) for v in self.values], encoder)
        device = self.torch.device(self.device)
        self.leaf = BatchedValue([m.to(device) for m in nets], encoder, device=device)

    def read(
        self,
        position: dict[str, Any],
        menu: Sequence[Sequence[str]],
        readings: Sequence[str] = tuple(READINGS),
    ) -> dict[str, Any]:
        import endgame_exact as ee

        from pokeuraou.budget import Budget
        from pokeuraou.position import Position

        pos = Position.from_json(position)
        if self.reg is None:
            self._setup(pos)
        sides = []
        for side in (0, 1):
            by = {a.to_choice(): a for a in ee.legal(self.reg, pos, side)}
            sides.append([by[c] for c in menu[side]])
        out: dict[str, Any] = {}
        for label in readings:
            kwargs = READINGS[label]
            self.count["cells"] = 0
            wall = time.perf_counter()
            cpu = time.process_time()
            result = self.search_mod.search(
                self.reg, pos, sides[0], sides[1], self.leaf, budget=Budget.matrix(), **kwargs
            )
            out[label] = {
                "value": float(result.equilibrium.value),
                "cpu": round(time.process_time() - cpu, 5),
                "wall": round(time.perf_counter() - wall, 5),
                "cells": self.count["cells"] - len(sides[0]) * len(sides[1]),
            }
        return out


def _line(path: str, line_no: int) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            if n == line_no:
                return json.loads(line)
    raise KeyError((path, line_no))


def sample_games(games: list[dict[str, Any]], n: int, seed: int) -> list[int]:
    order = list(range(len(games)))
    random.Random(seed).shuffle(order)
    return sorted(order[:n])


def _records(path: str, wanted: Sequence[int]) -> Any:
    """(line, record) for each wanted line of one file, in one pass over it."""
    want = set(wanted)
    last = max(want)
    with open(path, encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            if n in want:
                yield n, json.loads(line)
            if n >= last:
                return


def _share(games: list[dict[str, Any]], chosen: list[int], jobs: int, worker: int) -> list[int]:
    """This worker's games: whole files, round robin over the files the chosen games are in.

    A file is read once, front to back, by one worker (IKA-296: reading the whole pool game by
    game through `_line` re-read each ~200 MB file up to every game's line). Which worker reads a
    game does not change its readings.
    """
    files = sorted({games[g]["file"] for g in chosen})
    mine = set(files[worker :: max(jobs, 1)])
    return [g for g in chosen if games[g]["file"] in mine]


def run_read(args: argparse.Namespace) -> None:
    games = load_scan(args.scan)
    chosen = sample_games(games, args.games, args.seed)
    readings = tuple(args.readings.split(",")) if args.readings else tuple(READINGS)
    unknown = [r for r in readings if r not in READINGS]
    if unknown:
        raise SystemExit(f"unknown readings {unknown}; known: {sorted(READINGS)}")
    if args.worker is None and args.jobs > 1:
        base = [sys.executable, str(Path(__file__).resolve()), "read"]
        passed = ["--scan", str(args.scan), "--games", str(args.games), "--seed", str(args.seed),
                  "--jobs", str(args.jobs), "--out", str(args.out), "--device", args.device,
                  "--max-product", str(args.max_product), "--readings", ",".join(readings),
                  "--value", *args.value]
        procs = [subprocess.Popen([*base, *passed, "--worker", str(w)]) for w in range(args.jobs)]
        codes = [p.wait() for p in procs]
        if any(codes):
            raise SystemExit(f"a worker failed: {codes}")
        lines = []
        for w in range(args.jobs):
            part = Path(f"{args.out}.part{w}")
            lines.extend(part.read_bytes().splitlines())
            part.unlink()
        lines.sort(key=lambda x: json.loads(x)["game"])
        Path(args.out).write_bytes(b"".join(x + b"\n" for x in lines))
        return
    worker = args.worker or 0
    mine = _share(games, chosen, args.jobs, worker)
    reader = Reader(args.value, args.device)
    import pokeuraou

    print(f"[{worker}] pokeuraou from {pokeuraou.__file__}; readings {readings}", flush=True)
    part = Path(f"{args.out}.part{worker}") if args.jobs > 1 else Path(args.out)
    started = time.perf_counter()

    def where(g: int) -> list[int]:
        return [k for k, d in enumerate(games[g]["decisions"])
                if is_target(d, args.max_product, "shown") or is_target(d, args.max_product, "bench0")]

    by_file: dict[str, list[int]] = {}
    for g in mine:
        by_file.setdefault(games[g]["file"], []).append(g)
    n = 0
    with part.open("wb") as handle:
        for path, gs in by_file.items():
            at_line = {games[g]["line"]: g for g in gs}
            got: dict[int, dict[str, Any]] = {g: {} for g in gs}
            needed = [line for line, g in at_line.items() if where(g)]
            for line, record in (_records(path, needed) if needed else ()):
                g = at_line[line]
                for k in where(g):
                    d = record["decisions"][k]
                    got[g][str(k)] = reader.read(
                        d["position"], (d["ownActions"], d["foeActions"]), readings
                    )
            for g in gs:
                handle.write((json.dumps({"game": g, "readings": got[g]}) + "\n").encode("utf-8"))
            handle.flush()
            n += len(gs)
            print(f"[{worker}] {n}/{len(mine)} games, {time.perf_counter() - started:.0f}s", flush=True)


# --------------------------------------------------------------------------------------
# endgame


def run_endgame(args: argparse.Namespace) -> None:
    rows = [json.loads(x) for x in Path(args.answers).read_bytes().splitlines() if x]
    reader = Reader(args.value, args.device)
    import pokeuraou

    print(f"pokeuraou from {pokeuraou.__file__}", flush=True)
    with open(args.out, "wb") as handle:
        for row in rows:
            name, line_no, k = row["ref"]
            record = _line(str(Path(args.games_dir) / name), line_no)
            d = record["decisions"][k]
            exact = row["exact"]
            bounds = exact.get("bounds") or [exact["lo"], exact["hi"]]
            got = {
                "index": row["index"], "product": row["product"], "bench": row["bench"],
                "status": "ok" if exact["status"] == "ok" else "capped",
                "exact": exact.get("value"), "lo": bounds[0], "hi": bounds[1],
                "outcome": float(record["outcome"]),
                "recorded": float(d["searchValue"]),
                "readings": reader.read(d["position"], row["menu"]),
            }
            handle.write((json.dumps(got) + "\n").encode("utf-8"))
            handle.flush()


# --------------------------------------------------------------------------------------
# analyse


def _q(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if len(values) else float("nan")


def _boot(values: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> str:
    if values.size == 0:
        return "-"
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    return f"{values.mean():+.4f} [{np.quantile(means, 0.025):+.4f}, {np.quantile(means, 0.975):+.4f}]"


def endgame_tables(rows: list[dict[str, Any]]) -> str:
    rng = np.random.default_rng(1)
    out = []
    solved = [r for r in rows if r["status"] == "ok"]
    for name, lo, hi in (("<=16", 0, 16), ("17-64", 17, 64), ("<=64", 0, 64)):
        part = [r for r in solved if lo < r["product"] <= hi]
        if not part:
            continue
        v = np.array([r["exact"] for r in part])
        z = np.array([r["outcome"] for r in part])
        out.append(f"\n{name}: {len(part)} solved positions "
                   f"(exact within 0.01 of 0/1: {np.mean(np.minimum(v, 1 - v) <= 0.01):.1%})")
        out.append("  reading     MAE      RMSE     p90|e|   max|e|   bias(signed)  inward(toward 0.5)")
        for label in (*READINGS, "recorded"):
            est = np.array([r["recorded"] if label == "recorded" else r["readings"][label]["value"]
                            for r in part])
            e = est - v
            inward = np.where(v >= 0.5, -e, e)
            out.append(
                f"  {label:<9} {np.abs(e).mean():.4f}  {np.sqrt((e ** 2).mean()):.4f}  "
                f"{_q(np.abs(e), 0.9):.4f}  {np.abs(e).max():.4f}  {e.mean():+.4f}      "
                f"{inward.mean():+.4f}"
            )
        noise_rms = np.sqrt((v * (1 - v)).mean())
        out.append(
            f"  one outcome drawn from v: RMS sqrt(mean v(1-v)) {noise_rms:.4f}, "
            f"mean sqrt(v(1-v)) {np.sqrt(v * (1 - v)).mean():.4f}"
        )
        miss = np.abs(z - v)
        out.append(
            f"  the recorded game's outcome vs exact: |z-v| mean {miss.mean():.4f}, "
            f"RMS {np.sqrt((miss ** 2).mean()):.4f}, games against v (|z-v|>0.5) "
            f"{int((miss > 0.5).sum())}/{len(part)}"
        )
        for a, b in (("d2r", "d1"), ("m100", "d1"), ("m200", "d1"), ("m200", "d2r")):
            ea = np.abs(np.array([r["readings"][a]["value"] for r in part]) - v)
            eb = np.abs(np.array([r["readings"][b]["value"] for r in part]) - v)
            out.append(f"  |e| {a} - {b}: {_boot(ea - eb, rng)}")
    capped = [r for r in rows if r["status"] != "ok"]
    if capped:
        out.append(f"\nnot solved: {len(capped)} (bounds width median "
                   f"{_q([r['hi'] - r['lo'] for r in capped], 0.5):.3f})")
        out.append("  reading     inside [lo, hi]   distance outside (mean)")
        for label in READINGS:
            est = np.array([r["readings"][label]["value"] for r in capped])
            lo = np.array([r["lo"] for r in capped])
            hi = np.array([r["hi"] for r in capped])
            dist = np.maximum(lo - est, 0) + np.maximum(est - hi, 0)
            out.append(f"  {label:<9} {np.mean(dist == 0):14.1%}   {dist.mean():.4f}")
    return "\n".join(out)


def _brier_block(
    title: str, targets: list[tuple[dict[str, Any], float]], rng: np.random.Generator
) -> list[str]:
    out = [f"\n{title}: {len(targets)} target decisions"]
    if not targets:
        return out
    z = np.array([t[1] for t in targets])
    out.append("  reading    Brier     bias(mean v - z)  log loss   ECE(10 bins)")
    briers = {}
    for label in (*READINGS, "recorded"):
        v = np.array([t[0][label] for t in targets])
        vc = np.clip(v, 1e-4, 1 - 1e-4)
        bins = np.minimum((v * 10).astype(int), 9)
        ece = sum(abs(v[bins == b].mean() - z[bins == b].mean()) * (bins == b).mean()
                  for b in range(10) if (bins == b).any())
        briers[label] = (v - z) ** 2
        out.append(
            f"  {label:<9} {briers[label].mean():.4f}   {np.mean(v - z):+.4f}           "
            f"{-np.mean(z * np.log(vc) + (1 - z) * np.log(1 - vc)):.4f}    {ece:.4f}"
        )
    for a, b in (("d2r", "d1"), ("m100", "d1"), ("m200", "d1"), ("m200", "d2r")):
        out.append(f"  Brier {a} - {b}: {_boot(briers[a] - briers[b], rng)}")
    v = np.array([t[0]["m200"] for t in targets])
    out.append(
        f"  outcome noise if m200 were the truth: mean v(1-v) {np.mean(v * (1 - v)):.4f} "
        f"(RMS {np.sqrt(np.mean(v * (1 - v))):.4f})"
    )
    out.append("  calibration of m200 / d1 (bin: n, mean value, win rate)")
    for label in ("m200", "d1"):
        v = np.array([t[0][label] for t in targets])
        edges = (0, 0.02, 0.1, 0.3, 0.7, 0.9, 0.98, 1.0001)
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            m = (v >= lo) & (v < hi)
            if m.any():
                cells.append(
                    f"[{lo:.2f},{min(hi, 1):.2f}) {int(m.sum())}: "
                    f"{v[m].mean():.3f}/{z[m].mean():.3f}"
                )
        out.append(f"    {label}: " + "  ".join(cells))
    return out


def read_tables(reads: list[dict[str, Any]], games: list[dict[str, Any]]) -> str:
    rng = np.random.default_rng(1)
    out = []
    by_game = {r["game"]: r["readings"] for r in reads}
    # Targets against the recorded outcome.
    for max_product in BUCKETS:
        for bench in ("bench0", "shown"):
            targets = []
            for g, readings in by_game.items():
                game = games[g]
                for k, d in enumerate(game["decisions"]):
                    if is_target(d, max_product, bench) and str(k) in readings:
                        vals = {lab: readings[str(k)][lab]["value"] for lab in READINGS}
                        vals["recorded"] = d[6]
                        targets.append((vals, game["outcome"]))
            out += _brier_block(f"<={max_product} {bench}", targets, rng)
    # IKA-95 relabelling of every row before a target, per reading.
    out.append("\nIKA-95 relabelling on the sampled games (each row takes the value of the first "
               "target at or after it; rows after the last target keep the outcome)")
    out.append("  target        reading  rows   relabelled  mean|dz| (all rows)  mean|dz| (relabelled)  "
               "|dz|>0.1  |dz|>0.5")
    for max_product in BUCKETS:
        for bench in ("bench0", "shown"):
            for label in ("d1", "d2r", "m200"):
                rows = moved = 0
                deltas = []
                for g, readings in by_game.items():
                    game = games[g]
                    z = game["outcome"]
                    nxt = None
                    labels = []
                    for k in range(len(game["decisions"]) - 1, -1, -1):
                        d = game["decisions"][k]
                        if is_target(d, max_product, bench) and str(k) in readings:
                            nxt = readings[str(k)][label]["value"]
                        labels.append(z if nxt is None else nxt)
                    rows += len(labels)
                    moved += sum(1 for x in labels if x is not z)
                    deltas += [abs(x - z) for x in labels]
                dz = np.array(deltas)
                rel = dz[dz > 0] if (dz > 0).any() else np.zeros(1)
                out.append(
                    f"  <={max_product:<3} {bench:<7} {label:<7} {rows:6d}  {moved / rows:9.1%}  "
                    f"{dz.mean():18.4f}  {rel.mean():21.4f}  "
                    f"{np.mean(dz > 0.1):7.2%}  {np.mean(dz > 0.5):7.2%}"
                )
    # Cost per target decision.
    out.append("\ncost per target decision (process CPU seconds / wall seconds / cells), all read "
               "targets; cells count only what deepening spends (depth 2 refines are not counted)")
    cost: dict[str, list[tuple[float, float, int]]] = {lab: [] for lab in READINGS}
    for readings in by_game.values():
        for got in readings.values():
            for lab in READINGS:
                cost[lab].append((got[lab]["cpu"], got[lab]["wall"], got[lab]["cells"]))
    for lab, xs in cost.items():
        if not xs:
            continue
        cpu = [x[0] for x in xs]
        wall = [x[1] for x in xs]
        cells = [x[2] for x in xs]
        out.append(
            f"  {lab:<5} n {len(xs)}  cpu mean {np.mean(cpu):.4f} p50 {_q(cpu, 0.5):.4f} "
            f"p90 {_q(cpu, 0.9):.4f}"
            f"  wall mean {np.mean(wall):.4f}  cells mean {np.mean(cells):.0f}"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------------------
# mix (IKA-296: the training stage)


def isotonic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotonic regression of y on x (pool adjacent violators), as (block upper x, block mean y)."""
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    sums: list[float] = []
    counts: list[float] = []
    highs: list[float] = []
    for xv, yv in zip(xs, ys, strict=True):
        sums.append(yv)
        counts.append(1.0)
        highs.append(xv)
        while len(sums) > 1 and sums[-2] / counts[-2] > sums[-1] / counts[-1]:
            s, c, h = sums.pop(), counts.pop(), highs.pop()
            sums[-1] += s
            counts[-1] += c
            highs[-1] = h
    return np.array(highs), np.array(sums) / np.array(counts)


def calibrate(fit: tuple[np.ndarray, np.ndarray], v: np.ndarray) -> np.ndarray:
    """The fitted step function at v: the first block whose upper end is at or above v."""
    highs, means = fit
    idx = np.minimum(np.searchsorted(highs, v, side="left"), len(highs) - 1)
    return means[idx]


def relabel(
    game: np.ndarray,
    outcome: np.ndarray,
    target: np.ndarray,
    value: np.ndarray,
    lam: float,
    back: bool,
) -> np.ndarray:
    """Per-row training targets with a (calibrated) deep value mixed in.

    Rows are in record order: a game's rows are contiguous and in decision order. A target row
    becomes ``(1 - lam) * outcome + lam * value``. With ``back`` (IKA-95's form) every earlier row
    of the same game does too, with the value of the first target decision at or after it; rows
    after a game's last target keep the outcome.
    """
    out = outcome.astype(np.float64).copy()
    ahead = np.nan
    for i in range(len(out) - 1, -1, -1):
        if i == len(out) - 1 or game[i] != game[i + 1]:
            ahead = np.nan
        if target[i]:
            ahead = value[i]
        use = value[i] if target[i] else (ahead if back else np.nan)
        if not np.isnan(use):
            out[i] = (1.0 - lam) * outcome[i] + lam * use
    return out.astype(np.float32)


def _scores(v: np.ndarray, z: np.ndarray) -> str:
    from pokeuraou.value import auc

    c = np.clip(v, 1e-4, 1 - 1e-4)
    loss = -np.mean(z * np.log(c) + (1 - z) * np.log(1 - c))
    return f"Brier {np.mean((v - z) ** 2):.4f}  log loss {loss:.4f}  AUC {auc(v, z):.4f}"


def run_mix(args: argparse.Namespace) -> None:
    from pokeuraou.value import ValueConfig, load_dataset, split_for

    games = load_scan(args.scan)
    reads = {r["game"]: r["readings"] for r in load_scan(args.read)}
    data = load_dataset(args.data)
    game, outcome, kind = data.game, data.outcome.astype(np.float64), data.kind
    starts = np.flatnonzero(np.r_[True, game[1:] != game[:-1]])
    if len(starts) != len(games) or not np.array_equal(game[starts], np.arange(len(games))):
        raise SystemExit(f"{args.data} holds {len(starts)} games in order, the scan {len(games)}")
    labels = args.readings.split(",")
    target = np.zeros(len(game), bool)
    value = {lab: np.full(len(game), np.nan) for lab in labels}
    for g, (s, e) in enumerate(zip(starts, [*starts[1:], len(game)], strict=True)):
        rows = games[g]["decisions"]
        if len(rows) != e - s or outcome[s] != games[g]["outcome"]:
            raise SystemExit(f"game {g}: {e - s} rows / outcome {outcome[s]} in the data, "
                             f"{len(rows)} / {games[g]['outcome']} in the scan")
        for k, d in enumerate(rows):
            sv = data.search_value[s + k]
            if d[0] != kind[s + k] or not (abs(d[6] - sv) < 1e-4 or (np.isnan(d[6]) and np.isnan(sv))):
                raise SystemExit(f"game {g} decision {k}: scan and data disagree")
            if is_target(d, args.max_product, args.bench):
                if g not in reads or str(k) not in reads[g]:
                    raise SystemExit(f"game {g} decision {k} is a target without a reading")
                target[s + k] = True
                for lab in labels:
                    value[lab][s + k] = reads[g][str(k)][lab]["value"]
    config = ValueConfig(seed=args.split_seed, split_seed=args.split_seed)
    train_idx, val_idx = split_for(data, args.holdout, config)
    is_train = np.zeros(len(game), bool)
    is_train[train_idx] = True
    fit_rows = target & is_train
    val_rows = target & ~is_train
    print(f"{len(game):,} rows, {len(games):,} games; targets (move, product <= {args.max_product}, "
          f"{args.bench}) {int(target.sum()):,} = {target.mean():.1%} of rows; "
          f"calibration fitted on the {int(fit_rows.sum()):,} in the training games "
          f"(split seed {args.split_seed}, holdout {args.holdout}), {int(val_rows.sum()):,} held out")
    out: dict[str, np.ndarray] = {"target_rows": target}
    for lab in labels:
        fit = isotonic(value[lab][fit_rows], outcome[fit_rows])
        cal = np.where(target, calibrate(fit, np.nan_to_num(value[lab])), np.nan)
        out[f"{lab}-raw"] = value[lab].astype(np.float32)
        out[f"{lab}-cal"] = cal.astype(np.float32)
        for name, rows in (("training", fit_rows), ("held out", val_rows)):
            print(f"  {lab:<4} {name:<9} raw  {_scores(value[lab][rows], outcome[rows])}")
            print(f"  {lab:<4} {name:<9} cal  {_scores(cal[rows], outcome[rows])}")
        print(f"  {lab:<4} isotonic: {len(fit[0])} blocks")
        for lam in args.lam:
            for back in (False, True):
                name = f"{lab}-{'back' if back else 'rows'}-{lam:g}"
                mixed = relabel(game, outcome, target, cal, lam, back)
                shift = np.abs(mixed - outcome)
                out[name] = mixed
                print(f"  {name:<14} rows moved {np.mean(shift > 1e-9):6.1%}  mean |shift| "
                      f"(all rows) {shift.mean():.4f}  shift > 0.5 {np.mean(shift > 0.5):.2%}")
    for rows, name in ((val_rows, "held out"), (fit_rows, "training")):
        print(f"  recorded searchValue on the {name} targets: "
              f"{_scores(data.search_value[rows].astype(float), outcome[rows])}")
    np.savez_compressed(args.out, **out, meta_json=json.dumps({
        "scan": str(args.scan), "read": str(args.read), "data": str(args.data),
        "max_product": args.max_product, "bench": args.bench, "split_seed": args.split_seed,
        "holdout": args.holdout, "readings": labels, "lam": list(args.lam), "rows": len(game),
    }))
    print(f"-> {args.out}")


def run_analyse(args: argparse.Namespace) -> None:
    if args.endgame:
        print(endgame_tables(load_scan(args.endgame)))
    if args.read:
        print(read_tables(load_scan(args.read), load_scan(args.scan)))


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=run_scan)
    p = sub.add_parser("reach")
    p.add_argument("scan")
    p.set_defaults(func=run_reach)
    p = sub.add_parser("read")
    p.add_argument("--scan", required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-product", type=int, default=64)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--worker", type=int)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--value", nargs="+", required=True)
    p.add_argument("--readings", default="", help="comma-separated subset of " + ",".join(READINGS))
    p.set_defaults(func=run_read)
    p = sub.add_parser("mix")
    p.add_argument("--scan", required=True)
    p.add_argument("--read", required=True, help="a `read` of every game in the scan")
    p.add_argument("--data", required=True, help="the same games encoded (tools/encode_dataset.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--readings", default="d2r")
    p.add_argument("--max-product", type=int, default=64)
    p.add_argument("--bench", choices=("shown", "bench0"), default="shown")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--holdout", type=float, default=0.15)
    p.add_argument("--lam", type=float, nargs="+", default=[0.5, 1.0])
    p.set_defaults(func=run_mix)
    p = sub.add_parser("endgame")
    p.add_argument("--answers", required=True)
    p.add_argument("--games-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--value", nargs="+", required=True)
    p.set_defaults(func=run_endgame)
    p = sub.add_parser("analyse")
    p.add_argument("--read")
    p.add_argument("--scan")
    p.add_argument("--endgame")
    p.set_defaults(func=run_analyse)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
