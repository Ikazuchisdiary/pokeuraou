"""IKA-99 / IKA-100: do target-cpu, integer rolls or PGO change an answer, and what do they buy?

Every arm is a separately built `pokeuraou-damage` binary (`rust/experiments/ika99_build.sh`
builds base / v3 / native / introll; `rust/pgo.sh` builds llvm / pgo). This runs the same
workloads through each of them:

  damage   `pokeuraou-damage damage <reg> <cases>` -- agreement with the Python cases, and
           us/call over the repeats
  turns    `pokeuraou-damage turns <reg> <turns.json> <repeats>` -- exact/wrong/refused, the
           resolver's own counters, and us/turn
  node     width-W node fill on recorded positions through the warm `node` process, as the
           learned leaf asks for it (`fill_encoded`, `Budget.matrix()`): the child's own
           `resolveUs` / `encodeUs` from the header, and the wall time of the fill

and says whether the arms agree. Agreement is on the answers, not the clocks: every line
the binary prints except the speed line, and for a node the bytes of the encoded arrays
plus the spans, folds, exact mask and refusals. An arm that changes one of those is a
finding in itself and is printed as such.

    # identity only, one pass (what the arms were accepted on)
    python tools/bench_ika99.py check --arm base=out/pokeuraou-damage-base.exe \\
        --arm v3=out/pokeuraou-damage-v3.exe --cases out/cases/cases.json \\
        --turns rust/turns.json --games-dir data/ika73/w12 --positions 24

    # the timing table: arms interleaved, `--reps` passes, minimum of each
    python tools/bench_ika99.py bench --reps 5 --arm ... (same arguments)

    # PGO training input (`rust/pgo.sh` calls this with the instrumented binary)
    python tools/bench_ika99.py train --arm gen=out/pokeuraou-damage-pgogen.exe \\
        --games-dir data/ika73/w12 --train-positions 36

Positions are split once, after the seeded shuffle `budget_effect.load_positions` does:
the first `--train-positions` are the PGO training set and `check` / `bench` measure the
`--positions` that follow them, so a PGO arm is never timed on a node it was trained on.
The turns fixture is the same file for training and measurement; a PGO arm's turns row is
in-sample and is marked so.

Timing here is for an idle machine. Run it under the machine lock and nowhere else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from budget_effect import load_positions  # noqa: E402

from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.rustnode import RustNode  # noqa: E402

SPEED_TURNS = re.compile(r"= ([0-9.]+) us/turn")
SPEED_DAMAGE = re.compile(r"= ([0-9.]+) us/call")
#: Lines whose numbers are clocks. Everything else a binary prints is an answer.
CLOCK_LINE = re.compile(r"^(speed|encoded .* us each)")


def regulation_file(fixture: Path) -> Path:
    """The regulation a fixture was written for, from its own `format_id`."""
    with fixture.open(encoding="utf-8") as handle:
        head = handle.read(4096)
    match = re.search(r'"format_id":\s*"([a-z0-9]+)"', head)
    if not match:
        raise SystemExit(f"{fixture}: no format_id in the first 4 KB")
    return ROOT / "configs" / "regulations" / f"{match.group(1)}.json"


def run_binary(binary: Path, args: list[str]) -> str:
    """One run of a subcommand, in a scratch cwd: `turns` writes refused.json into it."""
    with tempfile.TemporaryDirectory(prefix="ika99-") as scratch:
        done = subprocess.run(
            [str(binary), *args],
            cwd=scratch,
            capture_output=True,
            text=True,
            check=False,
        )
    if done.returncode != 0:
        raise SystemExit(f"{binary.name} {args[0]} failed ({done.returncode}):\n{done.stderr}")
    return done.stdout


def answers(stdout: str) -> str:
    """The output with the clock lines taken out: what must be identical across arms."""
    kept = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if CLOCK_LINE.match(stripped):
            # The sink after the clock is an answer (a sum of results); keep it.
            sink = re.search(r"\[sink (-?\d+)\]", stripped)
            kept.append(f"<clock> sink={sink.group(1) if sink else '-'}")
            continue
        kept.append(line.rstrip())
    return "\n".join(kept)


def summary_line(stdout: str, prefix: str) -> str:
    for line in stdout.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return "(no summary line)"


def node_inputs(games_dir: Path, first: int, count: int, width: int, seed: int):  # noqa: ANN201
    """(reg, [(position, row actions, column actions)]) for positions[first:first+count]."""
    positions = load_positions(games_dir, 1, seed)
    chosen = positions[first : first + count]
    if len(chosen) < count:
        raise SystemExit(f"only {len(positions)} recorded positions under {games_dir}")
    reg = load_regulation(chosen[0].format)
    out = []
    for pos in chosen:
        row = narrow(reg, pos, 0, limit=width).actions
        col = narrow(reg, pos, 1, limit=width).actions
        if row and col:
            out.append((pos, row, col))
    return reg, out


def node_digest(filled) -> bytes:  # noqa: ANN001
    """Everything a node answers, as bytes: arrays, spans, folds, exact mask, refusals."""
    h = hashlib.sha256()
    encoded = filled.encoded
    for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
        h.update(np.ascontiguousarray(getattr(encoded, name)).tobytes())
    h.update(json.dumps(sorted(encoded.unknown_volatiles.items())).encode())
    for i, j, indices, weights in filled.spans:
        h.update(f"{i},{j},{indices}".encode())
        h.update(np.asarray(weights, dtype=np.float64).tobytes())
    h.update(json.dumps(filled.folded, sort_keys=True).encode())
    h.update(json.dumps(filled.exact).encode())
    h.update(json.dumps(filled.refused).encode())
    h.update(json.dumps(list(filled.unmodelled)).encode())
    return h.digest()


def node_pass(reg, binary: Path, nodes: list, budget: Budget) -> dict:  # noqa: ANN001
    """One pass over the nodes through a fresh warm process of this binary."""
    node = RustNode(reg, binary=binary)
    try:
        # One untimed node first: the first request pays for the shared block.
        pos, row, col = nodes[0]
        node.fill_encoded(pos, row, col, budget)
        digest = hashlib.sha256()
        resolve_us = encode_us = 0.0
        cells = leaves = 0
        wall = 0.0
        for pos, row, col in nodes:
            started = time.perf_counter()
            filled = node.fill_encoded(pos, row, col, budget)
            # Only the fill is timed: the digest below is this tool's own work.
            wall += time.perf_counter() - started
            digest.update(node_digest(filled))
            resolve_us += filled.resolve_us
            encode_us += filled.encode_us
            cells += len(row) * len(col)
            leaves += int(filled.encoded.mask.shape[0])
        wall_ms = wall * 1e3
    finally:
        node.close()
    return {
        "digest": digest.hexdigest()[:16],
        "resolve_ms": resolve_us / 1e3,
        "encode_ms": encode_us / 1e3,
        "wall_ms": wall_ms,
        "cells": cells,
        "leaves": leaves,
    }


def parse_arms(specs: list[str]) -> list[tuple[str, Path]]:
    arms = []
    for spec in specs:
        name, _, path = spec.partition("=")
        binary = Path(path)
        if not name or not binary.exists():
            raise SystemExit(f"--arm {spec!r}: want name=path to an existing binary")
        arms.append((name, binary))
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=("check", "bench", "train"))
    ap.add_argument("--arm", action="append", default=[], help="name=path; the first is the reference")
    ap.add_argument("--cases", action="append", default=[], type=Path, help="damage fixtures")
    ap.add_argument("--turns", type=Path, default=None, help="the turns fixture (rust/turns.json)")
    ap.add_argument("--games-dir", type=Path, default=None, help="recorded games for the node fill")
    ap.add_argument("--positions", type=int, default=24, help="nodes measured, after the training set")
    ap.add_argument("--train-positions", type=int, default=36, help="nodes reserved for PGO training")
    ap.add_argument("--width", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--turn-repeats", type=int, default=3, help="`turns` timing loop count")
    ap.add_argument("--damage-repeats", type=int, default=20, help="`damage` timing loop count")
    ap.add_argument("--in-sample", default="pgo", help="arms trained on the turns fixture")
    args = ap.parse_args()

    arms = parse_arms(args.arm)
    if not arms:
        raise SystemExit("no --arm given")
    budget = Budget.matrix()

    if args.mode == "train":
        if args.games_dir is None:
            raise SystemExit("train needs --games-dir")
        reg, nodes = node_inputs(args.games_dir, 0, args.train_positions, args.width, args.seed)
        for name, binary in arms:
            got = node_pass(reg, binary, nodes, budget)
            print(f"trained {name} on {len(nodes)} nodes, {got['cells']} cells, {got['leaves']} leaves")
        return

    reps = 1 if args.mode == "check" else args.reps
    nodes: list = []
    reg = None
    if args.games_dir is not None:
        reg, nodes = node_inputs(args.games_dir, args.train_positions, args.positions, args.width, args.seed)
        print(
            f"node fill: {len(nodes)} recorded positions from {args.games_dir.name} "
            f"(after the {args.train_positions} training ones), width {args.width}, Budget.matrix()"
        )

    workloads: list[tuple[str, list[str], re.Pattern | None]] = []
    for fixture in args.cases:
        workloads.append(
            (
                f"damage:{fixture.stem}",
                ["damage", str(regulation_file(fixture)), str(fixture), str(args.damage_repeats)],
                SPEED_DAMAGE,
            )
        )
    if args.turns is not None:
        workloads.append(
            (
                "turns",
                ["turns", str(regulation_file(args.turns)), str(args.turns), str(args.turn_repeats)],
                SPEED_TURNS,
            )
        )

    # best[arm][workload] = min over reps; seen[arm][workload] = the answer, once per rep
    best: dict[str, dict[str, float]] = {name: {} for name, _ in arms}
    seen: dict[str, dict[str, set[str]]] = {name: {} for name, _ in arms}
    summary: dict[str, dict[str, str]] = {name: {} for name, _ in arms}
    started = time.perf_counter()
    for rep in range(reps):
        # Rotate the order each pass so no arm always runs first (boost) or last.
        order = arms[rep % len(arms) :] + arms[: rep % len(arms)]
        for name, binary in order:
            for label, cmd, speed in workloads:
                out = run_binary(binary, cmd)
                answer = hashlib.sha256(answers(out).encode()).hexdigest()[:16]
                seen[name].setdefault(label, set()).add(answer)
                summary[name][label] = summary_line(
                    out, "agreement:" if label.startswith("damage") else "turns:"
                )
                match = speed.search(out) if speed else None
                if match:
                    value = float(match.group(1))
                    best[name][label] = min(best[name].get(label, value), value)
            if nodes:
                got = node_pass(reg, binary, nodes, budget)
                seen[name].setdefault("node", set()).add(got["digest"])
                summary[name]["node"] = f"{got['cells']} cells, {got['leaves']} leaves"
                for key in ("resolve_ms", "encode_ms", "wall_ms"):
                    label = f"node.{key}"
                    best[name][label] = min(best[name].get(label, got[key]), got[key])
        print(f"  pass {rep + 1}/{reps} done ({time.perf_counter() - started:.0f}s)", flush=True)

    reference = arms[0][0]
    labels = [label for label, _, _ in workloads] + (["node"] if nodes else [])
    print("\n== answers (identical to the reference arm?)")
    identical_all = True
    for name, _ in arms:
        for label in labels:
            ours = seen[name].get(label, set())
            ref = seen[reference].get(label, set())
            stable = len(ours) == 1
            same = stable and ours == ref
            identical_all &= same
            verdict = "identical" if same else ("UNSTABLE across passes" if not stable else "DIFFERS")
            print(f"  {name:<8} {label:<24} {verdict:<10} {summary[name].get(label, '')}")
    print(f"  all arms identical: {identical_all}")

    if args.mode == "check":
        return
    in_sample = {s for s in args.in_sample.split(",") if s}
    columns = [label for label, _, _ in workloads] + (
        ["node.resolve_ms", "node.encode_ms", "node.wall_ms"] if nodes else []
    )
    units = {
        c: ("us/call" if c.startswith("damage") else "us/turn" if c == "turns" else "ms") for c in columns
    }
    print(f"\n== minimum of {reps} interleaved passes (relative to {reference})")
    print(f"  {'arm':<8} " + " ".join(f"{c + ' ' + units[c]:>28}" for c in columns))
    for name, _ in arms:
        cells = []
        for c in columns:
            value = best[name].get(c)
            base = best[reference].get(c)
            if value is None:
                cells.append(f"{'-':>28}")
                continue
            rel = f"{(value / base - 1) * 100:+.1f}%" if base else ""
            mark = "*" if (c == "turns" and name in in_sample) else ""
            cells.append(f"{value:>16.3f} {rel:>8}{mark:>3}")
        print(f"  {name:<8} " + " ".join(cells))
    if in_sample & {name for name, _ in arms}:
        print("  * trained on this fixture (in-sample)")


if __name__ == "__main__":
    main()
