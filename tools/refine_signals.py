"""Which cells are worth a depth-2 value -- measured against every cell at depth 2 (IKA-281).

`search._cells_to_refine` deepens the cells the depth-1 equilibrium weights most, the top
`refine` actions per side by x_i and y_j. That looks only at sensitivity. It does not look
at how uncertain a cell's value is, nor at whether an action outside the support could
enter it once its cells move. IKA-254 (a best-first refine under a budget) needs to know
which of those is worth ranking by, and IKA-68 recorded that per cell `depth2 - depth1` is
-0.00076 +-0.00656 with sd 0.107 -- mostly movement without a direction. So the first
question is whether anything cheap predicts the SIZE of that movement.

Two steps, because the expensive one is the same for every question asked of it:

  collect   Per recorded position: both menus as the shipped agent builds them
            (`selfplay._menus`, leaf ranking, open information), the depth-1 matrix under
            the shipped leaf and under each of its two nets separately (one
            `port.batched_payoffs` call), and for EVERY cell every chance branch the port
            gives (`port.turn(full=True)`): its probability, its leaf value under the
            ensemble and each net, and its sub-game value (`search._subgame_value`, the
            shipped function, at `DEFAULT_SUB_LIMIT`). The shipped depth-2 value of a cell
            is the top `DEFAULT_SUB_BRANCHES` of those renormalised, which is recomputed
            from the stored numbers and checked against `search._refined_value` itself on
            two cells per position.
  analyse   Offline, from the stored numbers only: how well each signal predicts
            |depth2 - depth1| (Spearman, AUC for the top 10%), and, at equal budgets of
            refined cells, how close each way of choosing them brings the root's value and
            strategy to the all-cells-at-depth-2 answer. Plus the chance-branch choice
            against the all-branches value.

    uv run --group learn python tools/refine_signals.py collect --positions 120 --jobs 8 \\
        --games-dir data/selfplay-mc0 --out C:/tmp/ika281/w12.jsonl \\
        --value data/models/value-mc0.pt data/models/value-mc0-s1.pt
    uv run --group learn python tools/refine_signals.py analyse C:/tmp/ika281/w12.jsonl

Depth 2 has no road through `belief_solve` (IKA-111), so the positions are solved open:
the recorded position is the true one and both sides see it whole, as IKA-68 did.

Controls: `--jobs 1` and `--jobs N` write the same records (positions are chosen before
the split and each is computed alone); in `analyse` the oracle signal is a positive
control and a random score is the null (tests/test_refine_signals.py).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: Added to a regret before dividing by it, so an action on the support ranks by its
#: uncertainty rather than by an infinity. On the scale of a win probability.
GAP_FLOOR = 1e-3

#: Share of cells, by |depth2 - depth1|, that an AUC asks a signal to find.
TOP_SHARE = 0.10

BUDGETS = (16, 32, 64)
RECTANGLES = (4, 6, 8)


# --------------------------------------------------------------------------------------
# collect


def candidate_refs(
    games_dir: Path, games_per_file: int, min_turn: int, seed: int
) -> list[tuple[str, int, int]]:
    """(file, line, decision) of every move decision at or past `min_turn`, shuffled.

    The first `games_per_file` games of every worker file rather than all of one file, so
    that no single worker's stretch of the queue is the whole sample. Only references are
    kept; the positions are read again when they are measured.
    """
    refs: list[tuple[str, int, int]] = []
    for path in sorted(games_dir.glob("games-worker*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle):
                if line_no >= games_per_file:
                    break
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k, decision in enumerate(record.get("decisions", ())):
                    if decision.get("kind") == "move" and int(decision.get("turn", 0)) >= min_turn:
                        refs.append((path.name, line_no, k))
    random.Random(seed).shuffle(refs)
    return refs


def read_decision(games_dir: Path, ref: tuple[str, int, int]) -> dict[str, Any]:
    name, line_no, k = ref
    with (games_dir / name).open(encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            if n == line_no:
                return json.loads(line)["decisions"][k]
    raise KeyError(ref)


def collect_one(
    reg: Any,
    pos: Any,
    leaves: Sequence[Callable[[list[Any]], np.ndarray]],
    *,
    limit: int,
    sub_limit: int,
    check_cells: int,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Everything `analyse` needs about one position; None if a side has one action."""
    from pokeuraou import port
    from pokeuraou.budget import Budget
    from pokeuraou.search import DEFAULT_SUB_BRANCHES, _refined_value, _subgame_value
    from pokeuraou.selfplay import _menus

    ensemble = leaves[0]
    budget = Budget.matrix()
    ours, theirs = _menus(reg, pos, (limit, limit), ensemble, budget, True)
    if len(ours) < 2 or len(theirs) < 2:
        return None
    matrices, _notes, _exact = port.batched_payoffs(
        reg, pos, ours, theirs, list(leaves), budget=budget
    )
    cells: list[list[dict[str, Any] | None]] = []
    branch_positions: list[Any] = []
    owners: list[tuple[int, int, int]] = []
    for i in range(len(ours)):
        row: list[dict[str, Any] | None] = []
        for j in range(len(theirs)):
            result = port.turn(reg, pos, [ours[i], theirs[j]], budget, full=True)
            if result.suspended or not result.outcomes:
                row.append(None)
                continue
            # The same order `_refined_value` takes them in (a stable sort).
            branches = sorted(result.outcomes, key=lambda b: -b.probability)
            sub: list[float | None] = []
            solved: list[int] = []
            for b, branch in enumerate(branches):
                value, _n, did = _subgame_value(
                    reg, branch.position, ensemble, budget=budget, sub_limit=sub_limit
                )
                sub.append(value)
                solved.append(did)
                branch_positions.append(branch.position)
                owners.append((i, j, b))
            row.append({
                "p": [float(b.probability) for b in branches],
                "sub": sub,
                "solved": solved,
            })
        cells.append(row)
    # Every branch's leaf value, one call per net.
    for key, leaf in zip(("leaf", "leafa", "leafb"), leaves, strict=True):
        values = leaf(branch_positions) if branch_positions else np.zeros(0)
        for (i, j, _b), value in zip(owners, values, strict=True):
            cell = cells[i][j]
            assert cell is not None
            cell.setdefault(key, []).append(float(value))
    # The recomputed shipped value against the shipped function, on a few cells.
    live = [(i, j) for i in range(len(ours)) for j in range(len(theirs)) if cells[i][j]]
    checked = []
    for i, j in rng.sample(live, min(check_cells, len(live))):
        shipped, _n, _s = _refined_value(
            reg, pos, ours[i], theirs[j], ensemble,
            budget=budget, sub_limit=sub_limit, sub_branches=DEFAULT_SUB_BRANCHES,
        )
        mine = branch_estimate(cells[i][j], DEFAULT_SUB_BRANCHES, "prob", fill=False)
        checked.append([i, j, shipped, mine])
    return {
        "rows": [a.to_choice() for a in ours],
        "cols": [a.to_choice() for a in theirs],
        "d1": matrices[0].tolist(),
        "d1a": matrices[1].tolist(),
        "d1b": matrices[2].tolist(),
        "cells": cells,
        "checked": checked,
    }


def run_collect(args: argparse.Namespace) -> None:
    if args.jobs > 1 and args.worker is None:
        spawn(args)
        return
    import torch

    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.position import Position
    from pokeuraou.regulation import load_regulation
    from pokeuraou.value import BatchedValue, load_ensemble

    torch.set_num_threads(1)
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    refs = candidate_refs(args.games_dir, args.games_per_file, args.min_turn, args.seed)
    refs = refs[: args.positions]
    worker = args.worker or 0
    mine = [(k, ref) for k, ref in enumerate(refs) if k % args.jobs == worker]
    out = Path(args.out) if args.worker is None else Path(f"{args.out}.part{worker}")
    reg = None
    leaves: list[Any] = []
    with out.open("wb") as handle:
        for k, ref in mine:
            decision = read_decision(args.games_dir, ref)
            pos = Position.from_json(decision["position"])
            if reg is None:
                reg = load_regulation(pos.format)
                register_mega_stones(reg)
                encoder = Encoder(reg)
                nets, _ = load_ensemble(list(args.value), encoder)
                device = torch.device(args.device)
                nets = [n.to(device) for n in nets]
                leaves = [
                    BatchedValue(nets, encoder, device=device),
                    BatchedValue(nets[0], encoder, device=device),
                    BatchedValue(nets[1], encoder, device=device),
                ]
            started = time.perf_counter()
            got = collect_one(
                reg, pos, leaves, limit=args.limit, sub_limit=args.sub_limit,
                check_cells=args.check_cells, rng=random.Random(args.seed * 100003 + k),
            )
            row: dict[str, Any] = {"index": k, "ref": list(ref), "turn": decision["turn"]}
            if got is None:
                row["skipped"] = "a side with fewer than 2 actions"
            else:
                row.update(got)
            row["seconds"] = round(time.perf_counter() - started, 3)
            handle.write((json.dumps(row) + "\n").encode("utf-8"))
            handle.flush()
            print(f"  [{worker}] #{k} {row['seconds']:.1f}s", flush=True)


def spawn(args: argparse.Namespace) -> None:
    """One process per job, each measuring every `jobs`-th position; then one file."""
    base = [sys.executable, str(Path(__file__).resolve()), "collect"]
    passed = [
        "--positions", str(args.positions), "--limit", str(args.limit),
        "--sub-limit", str(args.sub_limit), "--games-dir", str(args.games_dir),
        "--games-per-file", str(args.games_per_file), "--min-turn", str(args.min_turn),
        "--seed", str(args.seed), "--device", args.device, "--jobs", str(args.jobs),
        "--check-cells", str(args.check_cells), "--out", str(args.out),
        "--value", *args.value,
    ]
    procs = [
        subprocess.Popen([*base, *passed, "--worker", str(w)]) for w in range(args.jobs)
    ]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a worker failed: {codes}")
    rows = []
    for w in range(args.jobs):
        part = Path(f"{args.out}.part{w}")
        rows.extend(part.read_bytes().splitlines())
        part.unlink()
    rows.sort(key=lambda line: json.loads(line)["index"])
    Path(args.out).write_bytes(b"".join(line + b"\n" for line in rows))


# --------------------------------------------------------------------------------------
# chance branches


def branch_estimate(
    cell: dict[str, Any], keep: int, how: str, *, fill: bool
) -> float | None:
    """A cell's depth-2 value from `keep` of its chance branches.

    ``how="prob"`` keeps the likeliest (the shipped choice); ``"spread"`` keeps the ones
    with the largest probability times distance of their leaf value from the cell's mean
    leaf value. ``fill=False`` renormalises over the kept ones (the shipped arithmetic);
    ``fill=True`` keeps every branch's weight and prices the dropped ones at their leaf.
    None when a kept branch had no sub-game value, which is when the shipped code gives up.
    """
    p = np.asarray(cell["p"], dtype=np.float64)
    n = p.size
    if how == "prob":
        order = list(range(n))
    elif how == "spread":
        leaf = np.asarray(cell["leaf"], dtype=np.float64)
        mean = float(p @ leaf / p.sum()) if p.sum() > 0 else 0.0
        score = p * np.abs(leaf - mean)
        order = sorted(range(n), key=lambda b: (-score[b], b))
    else:
        raise ValueError(how)
    kept = order[:keep]
    sub = cell["sub"]
    if any(sub[b] is None for b in kept):
        return None
    if not fill:
        weights = np.array([p[b] for b in sorted(kept)], dtype=np.float64)
        total = float(weights.sum())
        if total <= 0:
            return None
        weights /= total
        return float(np.array([sub[b] for b in sorted(kept)]) @ weights)
    leaf = np.asarray(cell["leaf"], dtype=np.float64)
    values = leaf.copy()
    for b in kept:
        values[b] = sub[b]
    return float(values @ p / p.sum())


def branch_cost(cell: dict[str, Any], keep: int, how: str) -> int:
    """Sub-games the choice solves (an ended branch is its leaf, and costs none)."""
    p = np.asarray(cell["p"], dtype=np.float64)
    if how == "prob":
        order = list(range(p.size))
    else:
        leaf = np.asarray(cell["leaf"], dtype=np.float64)
        mean = float(p @ leaf / p.sum())
        score = p * np.abs(leaf - mean)
        order = sorted(range(p.size), key=lambda b: (-score[b], b))
    return int(sum(cell["solved"][b] for b in order[:keep]))


# --------------------------------------------------------------------------------------
# analyse: one position


@dataclass
class Node:
    """One position's matrices, ready for offline selection."""

    d1: np.ndarray
    d1a: np.ndarray
    d1b: np.ndarray
    #: The shipped depth-2 value (top branches); d1 where a cell cannot be refined.
    d2: np.ndarray
    #: Every branch at depth 2; d1 where it cannot be.
    d2all: np.ndarray
    refinable: np.ndarray
    refinable_all: np.ndarray
    #: Probability-weighted sd of the leaf over a cell's chance branches.
    spread: np.ndarray
    branches: np.ndarray
    cells: list[list[dict[str, Any] | None]]
    index: int


def build_node(row: dict[str, Any], keep: int = 3) -> Node:
    d1 = np.asarray(row["d1"], dtype=np.float64)
    m, n = d1.shape
    d2 = d1.copy()
    d2all = d1.copy()
    refinable = np.zeros((m, n), dtype=bool)
    refinable_all = np.zeros((m, n), dtype=bool)
    spread = np.zeros((m, n))
    branches = np.zeros((m, n), dtype=np.int64)
    for i in range(m):
        for j in range(n):
            cell = row["cells"][i][j]
            if cell is None:
                continue
            p = np.asarray(cell["p"], dtype=np.float64)
            branches[i, j] = p.size
            leaf = np.asarray(cell["leaf"], dtype=np.float64)
            w = p / p.sum()
            spread[i, j] = float(np.sqrt(w @ (leaf - w @ leaf) ** 2))
            value = branch_estimate(cell, keep, "prob", fill=False)
            if value is not None:
                d2[i, j] = value
                refinable[i, j] = True
            everything = branch_estimate(cell, p.size, "prob", fill=False)
            if everything is not None:
                d2all[i, j] = everything
                refinable_all[i, j] = True
    return Node(
        d1=d1,
        d1a=np.asarray(row["d1a"], dtype=np.float64),
        d1b=np.asarray(row["d1b"], dtype=np.float64),
        d2=d2,
        d2all=d2all,
        refinable=refinable,
        refinable_all=refinable_all,
        spread=spread,
        branches=branches,
        cells=row["cells"],
        index=int(row["index"]),
    )


@dataclass
class Read:
    x: np.ndarray
    y: np.ndarray
    value: float


def read_mixed(d1: np.ndarray, d2: np.ndarray, mask: np.ndarray) -> Read:
    """The shipped reading: the whole matrix, refined cells at depth 2."""
    from pokeuraou.equilibrium import solve

    matrix = np.where(mask, d2, d1)
    eq = solve(matrix)
    return Read(eq.row_strategy, eq.col_strategy, float(eq.value))


def read_restricted(
    d1: np.ndarray, d2: np.ndarray, rows: Sequence[int], cols: Sequence[int]
) -> Read:
    """IKA-68's reading, one pass: the rectangle solved as its own game, zero elsewhere."""
    from pokeuraou.equilibrium import solve

    matrix = d2 if d2 is not None else d1
    eq = solve(matrix[np.ix_(list(rows), list(cols))])
    x = np.zeros(d1.shape[0])
    y = np.zeros(d1.shape[1])
    x[list(rows)] = eq.row_strategy
    y[list(cols)] = eq.col_strategy
    return Read(x, y, float(eq.value))


def measure(read: Read, reference: np.ndarray, ref: Read) -> dict[str, float]:
    """How far a reading is from the reference game's answer.

    `exploit` is what the reading's two strategies give up against best replies in the
    REFERENCE game, summed over both sides (NashConv). It is zero only for an equilibrium
    of the reference, and unlike TV it does not care which of several equilibria the LP
    returned.
    """
    row_loss = ref.value - float(np.min(read.x @ reference))
    col_loss = float(np.max(reference @ read.y)) - ref.value
    return {
        "value_err": abs(read.value - ref.value),
        "exploit": max(row_loss, 0.0) + max(col_loss, 0.0),
        "tv_x": 0.5 * float(np.abs(read.x - ref.x).sum()),
        "tv_y": 0.5 * float(np.abs(read.y - ref.y).sum()),
    }


_REGRETS: dict[bytes, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}


def regrets(d1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Depth-1 equilibrium and each action's regret against it (>= 0, 0 on the support).

    Cached by the matrix's bytes: every rule of a position asks for the same one.
    """
    key = d1.tobytes() + bytes(str(d1.shape), "ascii")
    if key not in _REGRETS:
        _REGRETS[key] = _regrets(d1)
    return _REGRETS[key]


def _regrets(d1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    from pokeuraou.equilibrium import solve

    eq = solve(d1)
    x, y, v = eq.row_strategy, eq.col_strategy, float(eq.value)
    r = np.clip(v - d1 @ y, 0.0, None)
    c = np.clip(x @ d1 - v, 0.0, None)
    return x, y, r, c, v


def cell_signals(node: Node) -> dict[str, np.ndarray]:
    """Every per-cell signal a selection might rank by; larger means refine first."""
    x, y, r, c, _v = regrets(node.d1)
    gap = r[:, None] + c[None, :]
    return {
        "sens": np.outer(x, y),
        "disagree": np.abs(node.d1a - node.d1b),
        "spread": node.spread,
        # A Bernoulli variance: a cell near 0 or 1 has little room to move.
        "bern": node.d1 * (1.0 - node.d1),
        "near_br": -gap,
        "branches": node.branches.astype(np.float64),
        "oracle": np.abs(node.d2 - node.d1),
        "gap": gap,
    }


#: Selection rules. Each is a list of keys, primary first; ties in all of them go to a
#: random order shared by every rule of the position (common random numbers). A key is
#: (kind, signal): "sens" is x_i*y_j, "sens*u" is x_i*y_j*u_ij, "u" is u_ij, "u/gap" is
#: u_ij / (regret_i + regret_j + GAP_FLOOR), "near" is -(regret_i + regret_j).
RULES: dict[str, list[tuple[str, str]]] = {
    "sens (now)": [("sens", "")],
    "sens>near-BR": [("sens", ""), ("near", "")],
    "sens*disagree": [("sens*u", "disagree"), ("u", "disagree")],
    "sens*spread": [("sens*u", "spread"), ("u", "spread")],
    "sens*bern": [("sens*u", "bern"), ("u", "bern")],
    "near-BR": [("near", "")],
    "disagree/gap": [("u/gap", "disagree")],
    "spread/gap": [("u/gap", "spread")],
    "bern/gap": [("u/gap", "bern")],
    "oracle*sens": [("sens*u", "oracle"), ("u", "oracle")],
    "oracle/gap": [("u/gap", "oracle")],
    "random (null)": [],
}


def cell_keys(signals: dict[str, np.ndarray], rule: str) -> list[np.ndarray]:
    out = []
    for kind, name in RULES[rule]:
        if kind == "sens":
            out.append(signals["sens"])
        elif kind == "sens*u":
            out.append(signals["sens"] * signals[name])
        elif kind == "u":
            out.append(signals[name])
        elif kind == "u/gap":
            out.append(signals[name] / (signals["gap"] + GAP_FLOOR))
        elif kind == "near":
            out.append(signals["near_br"])
        else:
            raise ValueError(kind)
    return out


def action_keys(
    signals: dict[str, np.ndarray], node: Node, rule: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The per-action form of a rule, for a rectangle: a row's u is its u against the
    opponent's depth-1 strategy, (u @ y)_i, and its gap is its own regret."""
    x, y, r, c, _v = regrets(node.d1)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for kind, name in RULES[rule]:
        u = signals.get(name)
        if kind == "sens":
            rows.append(x)
            cols.append(y)
        elif kind == "sens*u":
            rows.append(x * (u @ y))
            cols.append(y * (x @ u))
        elif kind == "u":
            rows.append(u @ y)
            cols.append(x @ u)
        elif kind == "u/gap":
            rows.append((u @ y) / (r + GAP_FLOOR))
            cols.append((x @ u) / (c + GAP_FLOOR))
        elif kind == "near":
            rows.append(-r)
            cols.append(-c)
        else:
            raise ValueError(kind)
    return rows, cols


def lex_order(keys: list[np.ndarray], tiebreak: np.ndarray) -> np.ndarray:
    """Flat indices, largest primary key first, then the next, then the tiebreak."""
    stack = [tiebreak.ravel()] + [-k.ravel() for k in reversed(keys)]
    return np.lexsort(stack)


def cell_order(keys: list[np.ndarray], tiebreak: np.ndarray) -> list[tuple[int, int]]:
    shape = tiebreak.shape
    return [
        tuple(int(v) for v in np.unravel_index(k, shape)) for k in lex_order(keys, tiebreak)
    ]


def select_cells(
    order: list[tuple[int, int]], refinable: np.ndarray, budget: int
) -> np.ndarray:
    """The first `budget` refinable cells of an order. An unrefinable cell costs a turn
    call and no sub-game, so it is passed over rather than counted."""
    mask = np.zeros(refinable.shape, dtype=bool)
    taken = 0
    for i, j in order:
        if taken >= budget:
            break
        if refinable[i, j]:
            mask[i, j] = True
            taken += 1
    return mask


def top_actions(keys: list[np.ndarray], tiebreak: np.ndarray, count: int) -> list[int]:
    return [int(k) for k in lex_order(keys, tiebreak)[:count]]


def shipped_search(node: Node, *, restricted: bool, passes: int, refine: int = 4) -> tuple[Read, int]:
    """`search.search` itself at depth 2, with the port and the sub-games replaced by the
    stored numbers -- so the shipped loop is what runs, not a copy of it."""
    from pokeuraou import search as search_mod
    from pokeuraou.budget import Budget

    m, n = node.d1.shape
    rows = [("row", i) for i in range(m)]
    cols = [("col", j) for j in range(n)]

    def fake_payoff(*_a: Any, **_k: Any) -> tuple[np.ndarray, set[str]]:
        return node.d1.copy(), set()

    def fake_refined(_reg: Any, _pos: Any, ours: Any, theirs: Any, *_a: Any, **_k: Any):  # noqa: ANN202
        i, j = ours[1], theirs[1]
        if not node.refinable[i, j]:
            return None, set(), 0
        return float(node.d2[i, j]), set(), 1

    saved = search_mod.batched_payoff, search_mod._refined_value
    search_mod.batched_payoff, search_mod._refined_value = fake_payoff, fake_refined
    try:
        got = search_mod.search(
            None, None, rows, cols, None, budget=Budget.matrix(), depth=2,
            refine=refine, passes=passes, solve_restricted=restricted,
        )
    finally:
        search_mod.batched_payoff, search_mod._refined_value = saved
    eq = got.equilibrium
    return Read(eq.row_strategy, eq.col_strategy, float(eq.value)), int(got.refined)


def classify(node: Node) -> str:
    x, y, *_ = regrets(node.d1)
    sx, sy = int((x > 1e-9).sum()), int((y > 1e-9).sum())
    if sx >= 2 and sy >= 2:
        return "mixed"
    if sx == 1 and sy == 1:
        return "pure"
    return "one-sided"


def analyse_node(node: Node, seed: int) -> dict[str, Any]:
    """Every selection's distance from the all-cells answer, for one position."""
    signals = cell_signals(node)
    rng = np.random.default_rng(seed * 7919 + node.index)
    tiebreak = rng.random(node.d1.shape)
    row_tie = rng.random(node.d1.shape[0])
    col_tie = rng.random(node.d1.shape[1])
    full = np.ones_like(node.refinable)
    ref = read_mixed(node.d1, node.d2, full)
    out: dict[str, Any] = {"class": classify(node), "index": node.index, "cells": {}, "rect": {}}
    out["depth1"] = measure(read_mixed(node.d1, node.d2, ~full), node.d2, ref)
    for name in RULES:
        order = cell_order(cell_keys(signals, name), tiebreak)
        for budget in BUDGETS:
            mask = select_cells(order, node.refinable, budget)
            out["cells"][(name, budget)] = measure(read_mixed(node.d1, node.d2, mask), node.d2, ref)
    for name in RULES:
        rs, cs = action_keys(signals, node, name)
        for side in RECTANGLES:
            rows = top_actions(rs, row_tie, side)
            cols = top_actions(cs, col_tie, side)
            mask = np.zeros_like(node.refinable)
            mask[np.ix_(rows, cols)] = True
            mask &= node.refinable
            out["rect"][(name, side, "mixed")] = measure(
                read_mixed(node.d1, node.d2, mask), node.d2, ref
            )
            prices = np.where(mask, node.d2, node.d1)
            out["rect"][(name, side, "restricted")] = measure(
                read_restricted(node.d1, prices, rows, cols), node.d2, ref
            )
    out["shipped"] = {}
    for restricted in (False, True):
        for passes in (1, 2):
            got, used = shipped_search(node, restricted=restricted, passes=passes)
            label = f"{'restricted' if restricted else 'mixed'} x{passes}"
            out["shipped"][label] = {**measure(got, node.d2, ref), "cells": used}
    return out


# --------------------------------------------------------------------------------------
# analyse: statistics


def rank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(a)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(a labelled cell outranks an unlabelled one), ties counted half."""
    pos = int(label.sum())
    neg = label.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rank(score)
    return float((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def top_label(target: np.ndarray, share: float = TOP_SHARE) -> np.ndarray:
    cut = np.quantile(target, 1.0 - share)
    return target >= cut if np.ptp(target) > 0 else np.zeros(target.size, dtype=bool)


def bootstrap(
    groups: list[Any], stat: Callable[[list[Any]], float], reps: int, seed: int
) -> tuple[float, float, float]:
    """The statistic and its 95% interval, resampling positions (not cells)."""
    rng = np.random.default_rng(seed)
    point = stat(groups)
    draws = []
    for _ in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        got = stat([groups[k] for k in pick])
        if np.isfinite(got):
            draws.append(got)
    if not draws:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def partial_spearman(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    """Rank correlation of a and b with z's rank correlation to both taken out."""
    ab, az, bz = spearman(a, b), spearman(a, z), spearman(b, z)
    if not all(np.isfinite(v) for v in (ab, az, bz)) or max(abs(az), abs(bz)) >= 1:
        return float("nan")
    return float((ab - az * bz) / np.sqrt((1 - az**2) * (1 - bz**2)))


#: The signals the table reports, in order.
SIGNALS = (
    "disagree", "spread", "bern", "sens", "near_br", "branches",
    "disagree/gap", "spread/gap", "bern/gap", "oracle",
)


def signal_table(
    nodes: list[Node], reps: int, seed: int
) -> list[tuple[str, str, str, str, str]]:
    """Per signal: pooled Spearman with |delta|, mean within-position Spearman, AUC for
    the top share, and the pooled Spearman with the Bernoulli variance partialled out."""
    per_node = []
    for node in nodes:
        sig = cell_signals(node)
        for name in ("disagree", "spread", "bern"):
            sig[f"{name}/gap"] = sig[name] / (sig["gap"] + GAP_FLOOR)
        live = node.refinable
        target = np.abs(node.d2 - node.d1)[live]
        per_node.append(({k: v[live] for k, v in sig.items()}, target))
    rows = []
    for name in SIGNALS:
        def pooled(group: list[Any], name: str = name) -> float:
            s = np.concatenate([g[0][name] for g in group])
            t = np.concatenate([g[1] for g in group])
            return spearman(s, t)

        def partial(group: list[Any], name: str = name) -> float:
            s = np.concatenate([g[0][name] for g in group])
            t = np.concatenate([g[1] for g in group])
            z = np.concatenate([g[0]["bern"] for g in group])
            return partial_spearman(s, t, z)

        # Each position's own correlation once; resampling positions resamples these.
        own = {id(g): spearman(g[0][name], g[1]) for g in per_node}

        def within(group: list[Any], own: dict[int, float] = own) -> float:
            vals = [own[id(g)] for g in group]
            vals = [v for v in vals if np.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        def area(group: list[Any], name: str = name) -> float:
            s = np.concatenate([g[0][name] for g in group])
            t = np.concatenate([g[1] for g in group])
            return auc(s, top_label(t))

        cells = []
        for stat in (pooled, within, area, partial):
            p, lo, hi = bootstrap(per_node, stat, reps, seed)
            cells.append(f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        rows.append((name, *cells))
    return rows


def mean_ci(values: list[float], reps: int, seed: int) -> str:
    if not values:
        return "n/a"
    p, lo, hi = bootstrap(values, lambda g: float(np.mean(g)), reps, seed)
    return f"{p:.4f} [{lo:.4f}, {hi:.4f}]"


def branch_table(nodes: list[Node]) -> tuple[list[str], dict[str, Any]]:
    """Per cell against the all-branches value, and at the root against its answer."""
    rules = [
        ("top3 by p (now)", 3, "prob", False),
        ("top3 by p, rest at leaf", 3, "prob", True),
        ("top3 by p*|dev|", 3, "spread", False),
        ("top3 by p*|dev|, rest at leaf", 3, "spread", True),
        ("top2 by p", 2, "prob", False),
        ("top1 by p", 1, "prob", False),
        ("leaf only (depth 1)", 0, "prob", True),
    ]
    errors: dict[str, list[float]] = {r[0]: [] for r in rules}
    many: dict[str, list[float]] = {r[0]: [] for r in rules}
    cost: dict[str, list[int]] = {r[0]: [] for r in rules}
    root: dict[str, list[dict[str, float]]] = {r[0]: [] for r in rules}
    for node in nodes:
        ref_mask = node.refinable_all
        ref = read_mixed(node.d1, node.d2all, np.ones_like(ref_mask))
        for name, keep, how, fill in rules:
            est = node.d1.copy()
            for i, j in zip(*np.nonzero(ref_mask), strict=True):
                cell = node.cells[i][j]
                value = branch_estimate(cell, keep, how, fill=fill) if keep else float(
                    np.asarray(cell["leaf"]) @ np.asarray(cell["p"]) / sum(cell["p"])
                )
                if value is None:
                    continue
                est[i, j] = value
                errors[name].append(value - node.d2all[i, j])
                if len(cell["p"]) > 3:
                    many[name].append(value - node.d2all[i, j])
                cost[name].append(branch_cost(cell, keep, how) if keep else 0)
            root[name].append(measure(read_mixed(node.d1, est, ref_mask), node.d2all, ref))
    lines = []
    for name, *_ in rules:
        e = np.asarray(errors[name])
        f = np.asarray(many[name]) if many[name] else np.zeros(1)
        r = root[name]
        lines.append(
            f"    {name:<30} {np.sqrt(np.mean(e ** 2)):.4f}  {np.mean(e):+.5f}  "
            f"{np.sqrt(np.mean(f ** 2)):.4f} (n={len(many[name])})  "
            f"{np.mean(cost[name]):5.2f}  "
            f"{np.mean([m['value_err'] for m in r]):.4f}  "
            f"{np.mean([m['exploit'] for m in r]):.4f}  "
            f"{np.mean([m['tv_x'] for m in r]):.3f}"
        )
    return lines, root


def run_analyse(args: argparse.Namespace) -> None:
    rows = []
    with Path(args.records).open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    measured = [r for r in rows if "skipped" not in r]
    print(f"  {len(rows)} positions read, {len(measured)} measured, "
          f"{len(rows) - len(measured)} skipped (a side with fewer than 2 actions)")
    checked = [c for r in measured for c in r["checked"]]
    bad = [c for c in checked if (c[2] is None) != (c[3] is None)
           or (c[2] is not None and c[2] != c[3])]
    print(f"  shipped `_refined_value` against the recomputed value: {len(checked) - len(bad)} "
          f"of {len(checked)} cells bit-identical")
    nodes = [build_node(r) for r in measured]
    seconds = [r["seconds"] for r in measured]
    cells = sum(n.d1.size for n in nodes)
    ref_cells = sum(int(n.refinable.sum()) for n in nodes)
    print(f"  {cells} cells, {ref_cells} refinable ({ref_cells / cells * 100:.1f}%), "
          f"branches per refinable cell {np.mean([b for n in nodes for b in n.branches[n.refinable]]):.2f}, "
          f"collect {np.mean(seconds):.1f}s a position")
    leaf_gap = []
    for n, r in zip(nodes, measured, strict=True):
        for i, j in zip(*np.nonzero(n.refinable), strict=True):
            cell = r["cells"][i][j]
            leaf_gap.append(abs(np.asarray(cell["leaf"]) @ np.asarray(cell["p"]) / sum(cell["p"])
                                - n.d1[i, j]))
    print(f"  |depth-1 cell - sum p*leaf over its branches|: max {max(leaf_gap):.2e} "
          f"(the branch spread is measured on the object whose mean is the cell)")
    classes = [classify(n) for n in nodes]
    print("  depth-1 classes: " + ", ".join(
        f"{c} {classes.count(c)}" for c in ("mixed", "one-sided", "pure")))
    deltas = np.concatenate([(n.d2 - n.d1)[n.refinable] for n in nodes])
    print(f"  depth2 - depth1 per refinable cell: mean {deltas.mean():+.5f}, sd {deltas.std():.4f}, "
          f"|.| mean {np.abs(deltas).mean():.4f}, median {np.median(np.abs(deltas)):.4f}")
    dis = np.concatenate([np.abs(n.d1a - n.d1b).ravel() for n in nodes])
    print(f"  |net a - net b| per cell: mean {dis.mean():.4f}, median {np.median(dis):.4f}")

    print(f"\n  signal -> |depth2 - depth1| (refinable cells; 95% by resampling positions, "
          f"{args.reps} draws)")
    print(f"    {'signal':<14} {'Spearman pooled':<26} {'Spearman within':<26} "
          f"{'AUC top ' + format(TOP_SHARE, '.0%'):<26} pooled, bern partialled out")
    for group, members in (("all", nodes), *(
        (c, [n for n, k in zip(nodes, classes, strict=True) if k == c]) for c in ("mixed", "pure")
    )):
        if len(members) < 3:
            continue
        print(f"   [{group}, {len(members)} positions]")
        for name, a, b, c, d in signal_table(members, args.reps, args.seed):
            print(f"    {name:<14} {a:<26} {b:<26} {c:<26} {d}")

    results = [analyse_node(n, args.seed) for n in nodes]
    for group in ("all", "mixed", "one-sided", "pure"):
        members = [r for r in results if group == "all" or r["class"] == group]
        if not members:
            continue
        print(f"\n  [{group}, {len(members)} positions] mean over positions; "
              "exploit = NashConv of the reading in the all-cells game")
        print("    depth 1 (no cell refined)       "
              + "  ".join(f"{k} {np.mean([r['depth1'][k] for r in members]):.4f}"
                          for k in ("value_err", "exploit", "tv_x", "tv_y")))
        for label in ("mixed x1", "mixed x2", "restricted x1", "restricted x2"):
            got = [r["shipped"][label] for r in members]
            print(f"    shipped search, {label:<15} "
                  + "  ".join(f"{k} {np.mean([g[k] for g in got]):.4f}"
                              for k in ("value_err", "exploit", "tv_x", "tv_y"))
                  + f"  cells {np.mean([g['cells'] for g in got]):.1f}")
        print(f"    cell budget, mixed reading:  {'rule':<16} "
              + " ".join(f"{'K=' + str(b) + ' err / exploit / TVx':<32}" for b in BUDGETS))
        for name in RULES:
            parts = []
            for budget in BUDGETS:
                got = [r["cells"][(name, budget)] for r in members]
                parts.append(
                    f"{np.mean([g['value_err'] for g in got]):.4f} / "
                    f"{np.mean([g['exploit'] for g in got]):.4f} / "
                    f"{np.mean([g['tv_x'] for g in got]):.3f}".ljust(32)
                )
            print(f"                                 {name:<16} " + " ".join(parts))
        for reading in ("mixed", "restricted"):
            print(f"    rectangle, {reading:<10} reading: {'rule':<16} "
                  + " ".join(f"{str(s) + 'x' + str(s) + ' err / exploit / TVx':<32}" for s in RECTANGLES))
            for name in RULES:
                parts = []
                for side in RECTANGLES:
                    got = [r["rect"][(name, side, reading)] for r in members]
                    parts.append(
                        f"{np.mean([g['value_err'] for g in got]):.4f} / "
                        f"{np.mean([g['exploit'] for g in got]):.4f} / "
                        f"{np.mean([g['tv_x'] for g in got]):.3f}".ljust(32)
                    )
                print(f"                                 {name:<16} " + " ".join(parts))

    print("\n  intervals for the exploit at each cell budget (all positions)")
    for name in RULES:
        print(f"    {name:<16} " + "  ".join(
            f"K={b} " + mean_ci([r["cells"][(name, b)]["exploit"] for r in results],
                                args.reps, args.seed) for b in BUDGETS))
    print("  paired against sens (now): exploit(rule) - exploit(sens), all positions")
    for name in list(RULES)[1:]:
        parts = []
        for b in BUDGETS:
            diffs = [r["cells"][(name, b)]["exploit"] - r["cells"][("sens (now)", b)]["exploit"]
                     for r in results]
            p, lo, hi = bootstrap(diffs, lambda g: float(np.mean(g)), args.reps, args.seed)
            parts.append(f"K={b} {p:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"    {name:<16} " + "  ".join(parts))

    print("\n  chance branches, against every branch at depth 2")
    print("    rule                           cell RMSE  bias      RMSE >3 branches     "
          "sub-games  root err  exploit  TVx")
    lines, _root = branch_table(nodes)
    for line in lines:
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--positions", type=int, default=120)
    c.add_argument("--limit", type=int, default=12, help="12 is generation's width")
    c.add_argument("--sub-limit", type=int, default=None)
    c.add_argument("--games-dir", type=Path, default=Path("data/selfplay-mc0"))
    c.add_argument("--games-per-file", type=int, default=20)
    c.add_argument("--min-turn", type=int, default=3)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--value", nargs=2, required=True, help="the two nets of the shipped leaf")
    c.add_argument("--device", default="cpu")
    c.add_argument("--jobs", type=int, default=1)
    c.add_argument("--worker", type=int, default=None)
    c.add_argument("--check-cells", type=int, default=2)
    c.add_argument("--out", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("records")
    a.add_argument("--reps", type=int, default=500)
    a.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if args.command == "collect":
        if args.sub_limit is None:
            from pokeuraou.search import DEFAULT_SUB_LIMIT

            args.sub_limit = DEFAULT_SUB_LIMIT
        run_collect(args)
    else:
        run_analyse(args)


if __name__ == "__main__":
    main()
