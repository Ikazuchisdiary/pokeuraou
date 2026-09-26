"""Teaching matrices for the candidate model Q(s, a, b) (IKA-274).

For recorded positions of a generation, the depth-1 matrix over EVERY legal action of
both sides (`qhead.legal_pool`: what `narrow` ranks, about 45 a side past turn 2), filled
by the port and the shipped leaf exactly as `search.search` fills a menu's matrix
(`port.batched_payoff` at `Budget.matrix()`). Not a width-12 search and not the ranking's
two replies: the model is to learn the cells the menu never sees.

Under a hidden bench each side ranked its menu from its own heaviest completion of the
other side's bench (`rankViews`); the matrix of side s is built on that completion and
the model's input is that same position (`qhead.decision_views`). One matrix serves both
sides when both see the whole board.

    index   scan the generation once: every move decision, its file offset, shuffled by a
            seed. A position's number is its place in this order, so a pilot is a prefix
            of the full set and a run is resumable by number.
    fill    matrices for positions [start, stop) in shards of `--shard`, by `--jobs`
            processes; a shard already written is skipped, so the same command resumes.
            Every `--check-every`-th position is also filled through `search.search`
            (the whole pool, and the width-12 menus `narrow` ranks by the same leaf), and
            the stored cells are compared bit for bit.
    menus   the views' positive control: with the leaf that played the games, `narrow` on
            each view reproduces the recorded menus (`ownActions` / `foeActions`).
    cost    per-view cells, leaves and seconds from the shards, and the projection.

    python tools/q_teach.py index --games-dir data/selfplay-mc0 --out <dir>/index.npz
    python tools/q_teach.py fill --index <dir>/index.npz --out <dir>/shards \\
        --start 0 --stop 3000 --jobs 8 --value data/models/value-mc0.pt data/models/value-mc0-s1.pt

A shard is an `.npz` (no pickles): per view its position number, side, turn, the two
action counts and offsets; the matrix cells (float32, side 0's win probability, rows =
side 0's pool, columns = side 1's); the view's `encode.Encoded` row; both sides' actions
by `qhead.encode_actions`; and the choice strings as JSON text.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ENCODED = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


# --------------------------------------------------------------------------------------
# index


def run_index(args: argparse.Namespace) -> None:
    files = sorted(Path(args.games_dir).glob("games-worker*.jsonl"))
    rows: list[tuple[int, int, int, int, int, int, int]] = []
    for f, path in enumerate(files):
        offset = 0
        with path.open("rb") as handle:
            for line_no, line in enumerate(handle):
                start, offset = offset, offset + len(line)
                if line_no < args.skip_lines:
                    continue
                try:
                    game = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k, decision in enumerate(game.get("decisions", ())):
                    if decision.get("kind") != "move":
                        continue
                    rows.append(
                        (
                            f,
                            start,
                            len(line),
                            line_no,
                            k,
                            int(decision.get("turn", 0)),
                            int(game.get("gameIndex", -1)),
                        )
                    )
        print(f"  {path.name}: {len(rows)} decisions so far", flush=True)
    table = np.array(rows, dtype=np.int64)
    order = np.random.default_rng(args.seed).permutation(len(table))
    table = table[order]
    buffer = io.BytesIO()
    np.savez(
        buffer,
        file_no=table[:, 0].astype(np.int32),
        offset=table[:, 1],
        length=table[:, 2],
        line=table[:, 3].astype(np.int32),
        decision=table[:, 4].astype(np.int32),
        turn=table[:, 5].astype(np.int32),
        game=table[:, 6].astype(np.int32),
        files=np.array(json.dumps([p.name for p in files])),
        games_dir=np.array(str(args.games_dir)),
        skip_lines=np.array(args.skip_lines),
        seed=np.array(args.seed),
    )
    Path(args.out).write_bytes(buffer.getvalue())
    print(f"{len(table)} move decisions from {len(files)} files -> {args.out}")


class Index:
    def __init__(self, path: Path, games_dir: Path | None = None) -> None:
        with np.load(path) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.files = json.loads(str(self.arrays["files"]))
        self.games_dir = Path(games_dir or str(self.arrays["games_dir"]))

    def __len__(self) -> int:
        return len(self.arrays["offset"])

    def game(self, k: int) -> dict[str, Any]:
        a = self.arrays
        path = self.games_dir / self.files[int(a["file_no"][k])]
        with path.open("rb") as handle:
            handle.seek(int(a["offset"][k]))
            return json.loads(handle.read(int(a["length"][k])))

    def ref(self, k: int) -> tuple[str, int, int]:
        a = self.arrays
        return self.files[int(a["file_no"][k])], int(a["line"][k]), int(a["decision"][k])


# --------------------------------------------------------------------------------------
# fill


def build_leaf(args: argparse.Namespace, reg: Any) -> tuple[Any, Any]:  # noqa: ANN401
    from pokeuraou.encode import Encoder

    encoder = Encoder(reg)
    if args.inference:
        from pokeuraou.inference import RemoteValue

        return RemoteValue(args.inference, "value", encoder), encoder
    import torch

    from pokeuraou.value import BatchedValue, load_ensemble

    torch.set_num_threads(1)
    nets, _metas = load_ensemble([Path(p) for p in args.value], encoder)
    device = torch.device(args.device)
    return BatchedValue([n.to(device) for n in nets], encoder, device=device), encoder


def _children_cpu() -> float:
    import psutil

    total = 0.0
    for child in psutil.Process().children(recursive=True):
        try:
            t = child.cpu_times()
            total += t.user + t.system
        except psutil.Error:
            pass
    return total


def fill_position(ctx: dict[str, Any], k: int, check: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Every view of position `k`: its matrix, encoding, actions, costs, and the control."""
    from pokeuraou import port, qhead
    from pokeuraou.budget import Budget

    reg, pool, leaf, encoder, index = (ctx["reg"], ctx["pool"], ctx["leaf"], ctx["encoder"], ctx["index"])
    budget = Budget.matrix()
    notes = {"views": 0, "skipped_one_action": 0, "refused": 0}
    game = index.game(k)
    decision = game["decisions"][int(index.arrays["decision"][k])]
    out: list[dict[str, Any]] = []
    notes["same_view"] = 0
    made: list[tuple[dict[str, Any], dict[str, Any]]] = []  # (position JSON, its row)
    for view in qhead.decision_views(reg, pool, game, decision):
        pos = view.position
        pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
        if len(pools[0]) < 2 or len(pools[1]) < 2:
            notes["skipped_one_action"] += 1
            continue
        # Both sides' heaviest completions are often the truth, and then the two views
        # are one position: its matrix is the same, so it is filled once and read by both.
        as_json = pos.to_json()
        twin = next((row for seen, row in made if seen == as_json), None)
        if twin is not None:
            twin["side"] = 2
            notes["same_view"] += 1
            continue
        wall0, cpu0, kids0, leaves0 = (
            time.perf_counter(),
            time.process_time(),
            _children_cpu(),
            leaf.evaluated,
        )
        spread = None
        try:
            if ctx.get("spread"):
                matrix, spread = filled_with_spread(reg, pos, pools, leaf, budget)
            else:
                matrix, _unmodelled = port.batched_payoff(reg, pos, pools[0], pools[1], leaf, budget=budget)
        except port.PortRefused:
            notes["refused"] += 1
            continue
        seconds = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
        kids = _children_cpu() - kids0
        leaves = leaf.evaluated - leaves0
        encoded = encoder.encode_positions([pos])
        row = {
            "k": k,
            "side": view.side,
            "completion": view.completion,
            "turn": int(decision["turn"]),
            "game": int(game.get("gameIndex", -1)),
            "matrix": matrix,
            "encoded": {name: getattr(encoded, name)[0] for name in ENCODED},
            "acts0": qhead.encode_actions(encoder.vocab, pos, 0, pools[0]),
            "acts1": qhead.encode_actions(encoder.vocab, pos, 1, pools[1]),
            "rows": [a.to_choice() for a in pools[0]],
            "cols": [a.to_choice() for a in pools[1]],
            "cost": [seconds, cpu, kids, float(leaves)],
            "check": control(ctx, pos, pools, matrix) if check else None,
            "spread": spread,
        }
        notes["views"] += 1
        out.append(row)
        made.append((as_json, row))
    return out, notes


#: Chance branches a cell keeps at depth 2 (`search.DEFAULT_SUB_BRANCHES`), for the
#: probability left outside them.
SPREAD_TOP = 3


def filled_with_spread(
    reg: Any, pos: Any, pools: tuple[list, list], leaf: Any, budget: Any
) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401
    """`port.batched_payoff`'s matrix, taken apart so each cell's chance branches are seen.

    The same three steps `port._encoded` takes (the port's encoded node, one scoring call,
    the fold), so the matrix is the one `batched_payoff` returns; beside it, per cell, the
    probability-weighted sd of its branches' leaf values, the probability outside its
    `SPREAD_TOP` likeliest branches, and how many branches it has (IKA-322's inputs). A
    cell paused mid-turn (a fold) has NaN, NaN, 0.
    """
    from pokeuraou import port

    plan = port.encoded_leaf_plan([leaf])
    if plan is None or plan[0][1] is None:
        raise ValueError("the branch spread needs a learned leaf on the encoded road")
    filled = port._fill_encoded(reg, pos, pools[0], pools[1], plan, [plan[0][1]], budget, None)  # noqa: SLF001
    values = np.asarray(plan[0][1](filled.encoded), dtype=np.float64)
    shape = (len(pools[0]), len(pools[1]))
    matrix = port._folded(filled, values, shape)  # noqa: SLF001
    spread = np.zeros((3, *shape), dtype=np.float32)
    spread[0:2] = np.nan
    for i, j, indices, weights in filled.spans:
        if not weights:
            continue
        w = np.asarray(weights, dtype=np.float64)
        v = values[indices]
        mean = float(v @ w) / float(w.sum())
        spread[0, i, j] = float(np.sqrt(w @ (v - mean) ** 2 / w.sum()))
        spread[1, i, j] = float(1.0 - np.sort(w)[::-1][:SPREAD_TOP].sum() / w.sum())
        spread[2, i, j] = len(w)
    return matrix, spread


def control(ctx: dict[str, Any], pos: Any, pools: tuple[list, list], matrix: np.ndarray) -> list[float]:  # noqa: ANN401
    """The stored cells against the search's own fill of the same position.

    [whole-pool cells equal, whole-pool max |diff|, width-12 cells equal, width-12 cells,
    width-12 max |diff|]. The whole pool through `search.search` is the same call on the
    same menus; the width-12 menus (`narrow` ranked by this leaf, as `_menus` ranks) are
    the matrix a generation node fills, and its cells must be the stored ones.
    """
    from pokeuraou.budget import Budget
    from pokeuraou.narrow import narrow
    from pokeuraou.search import leaf_ranking, search

    reg, leaf = ctx["reg"], ctx["leaf"]
    budget = Budget.matrix()
    whole = search(reg, pos, pools[0], pools[1], leaf, budget=budget).payoff
    menus = [
        narrow(
            reg,
            pos,
            side,
            limit=12,
            rank=leaf_ranking(reg, pos, side, leaf, budget=budget),
        ).actions
        for side in (0, 1)
    ]
    rows = [pools[0].index(a) for a in menus[0]]
    cols = [pools[1].index(a) for a in menus[1]]
    small = search(reg, pos, menus[0], menus[1], leaf, budget=budget).payoff
    part = matrix[np.ix_(rows, cols)]
    return [
        float(np.sum(whole == matrix)),
        float(np.max(np.abs(whole - matrix))),
        float(np.sum(small == part)),
        float(small.size),
        float(np.max(np.abs(small - part))),
        *port_cells(ctx, pos, pools, menus, rows, cols),
    ]


def port_cells(ctx: dict[str, Any], pos: Any, pools, menus, rows, cols) -> list[float]:  # noqa: ANN001, ANN401
    """Cell by cell, the whole-pool node against the width-12 node, below the leaf.

    The leaf net's float32 answer for a row moves with where the row sits in its batch
    (on the card, and here on the CPU too: 1-2 ulp), so the two matrices can differ in the
    last bits where nothing else does. This separates that: for every cell of the width-12
    node, are its leaves (every encoded array) and its branch weights the same as the
    whole-pool node's for that cell, and is its value the same when each node's rows for
    the cell are scored as a batch of their own.

    [cells compared, cells with the same leaves and weights, cells with the same value
    scored alone, folded (paused) cells not compared]
    """
    from pokeuraou import port
    from pokeuraou.budget import Budget

    reg, leaf = ctx["reg"], ctx["leaf"]
    plan = port.encoded_leaf_plan([leaf])
    if plan is None or plan[0][1] is None:
        return [np.nan] * 4
    budget = Budget.matrix()
    big = port._fill_encoded(reg, pos, pools[0], pools[1], plan, [plan[0][1]], budget, None)
    small = port._fill_encoded(reg, pos, menus[0], menus[1], plan, [plan[0][1]], budget, None)
    spans = {(i, j): (idx, w) for i, j, idx, w in big.spans}
    compared = same_port = same_alone = 0
    for i, j, idx, w in small.spans:
        found = spans.get((rows[i], cols[j]))
        if found is None or not idx:
            continue
        compared += 1
        b_idx, b_w = found
        same = list(b_w) == list(w) and all(
            np.array_equal(getattr(small.encoded, f)[list(idx)], getattr(big.encoded, f)[list(b_idx)])
            for f in ENCODED
        )
        same_port += same
        mine = float(leaf.from_encoded(small.encoded.slice(np.asarray(idx))) @ np.asarray(w))
        theirs = float(leaf.from_encoded(big.encoded.slice(np.asarray(b_idx))) @ np.asarray(b_w))
        same_alone += mine == theirs
    return [float(compared), float(same_port), float(same_alone), float(len(small.folded))]


def write_shard(path: Path, rows: list[dict[str, Any]], notes: dict[str, int], meta: dict[str, Any]) -> None:
    n0 = np.array([len(r["rows"]) for r in rows], dtype=np.int32)
    n1 = np.array([len(r["cols"]) for r in rows], dtype=np.int32)
    cells = n0.astype(np.int64) * n1
    arrays: dict[str, Any] = {
        "k": np.array([r["k"] for r in rows], dtype=np.int32),
        "side": np.array([r["side"] for r in rows], dtype=np.int8),
        "completion": np.array([r["completion"] for r in rows], dtype=np.int8),
        "turn": np.array([r["turn"] for r in rows], dtype=np.int16),
        "game": np.array([r["game"] for r in rows], dtype=np.int32),
        "n0": n0,
        "n1": n1,
        "cell_start": np.concatenate([[0], np.cumsum(cells)[:-1]]).astype(np.int64),
        "act0_start": np.concatenate([[0], np.cumsum(n0)[:-1]]).astype(np.int64),
        "act1_start": np.concatenate([[0], np.cumsum(n1)[:-1]]).astype(np.int64),
        "cells": (
            np.concatenate([r["matrix"].astype(np.float32).ravel() for r in rows])
            if rows
            else np.zeros(0, np.float32)
        ),
        "acts0": np.concatenate([r["acts0"] for r in rows]).astype(np.int16)
        if rows
        else np.zeros((0, 2, 7), np.int16),
        "acts1": np.concatenate([r["acts1"] for r in rows]).astype(np.int16)
        if rows
        else np.zeros((0, 2, 7), np.int16),
        "cost": np.array([r["cost"] for r in rows], dtype=np.float64).reshape(-1, 4),
        "check": np.array(
            [r["check"] if r["check"] is not None else [np.nan] * 9 for r in rows],
            dtype=np.float64,
        ).reshape(-1, 9),
        "choices": np.array(json.dumps([[r["rows"], r["cols"]] for r in rows])),
        "notes": np.array(json.dumps(notes)),
        "meta": np.array(json.dumps(meta)),
    }
    for name in ENCODED:
        arrays[f"enc_{name}"] = np.stack([r["encoded"][name] for r in rows]) if rows else np.zeros(0)
    if rows and rows[0].get("spread") is not None:
        # Per cell, in `cells` order: branch sd, probability outside the top branches, count.
        arrays["spread_sd"] = np.concatenate([r["spread"][0].ravel() for r in rows])
        arrays["spread_rest"] = np.concatenate([r["spread"][1].ravel() for r in rows])
        arrays["spread_n"] = np.concatenate([r["spread"][2].ravel() for r in rows]).astype(np.int16)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buffer.getvalue())
    os.replace(tmp, path)


def shard_path(out: Path, start: int) -> Path:
    return out / f"shard-{start:07d}.npz"


def run_worker(args: argparse.Namespace) -> None:
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.pool import load_pool

    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    index = Index(Path(args.index), args.games_dir)
    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    leaf, encoder = build_leaf(args, reg)
    ctx = {
        "reg": reg,
        "pool": pool,
        "leaf": leaf,
        "encoder": encoder,
        "index": index,
        "spread": args.branch_spread,
    }
    out = Path(args.out)
    meta = {
        "value": [Path(p).name for p in args.value] if not args.inference else "served",
        "device": args.device if not args.inference else "served",
        "pool": pool.id,
        "pool_sha256": pool.sha256,
        "worker": args.worker,
    }
    stop = min(args.stop, len(index))
    for start in range(args.start, stop, args.shard):
        path = shard_path(out, start)
        claim = path.with_suffix(".claim")
        if path.exists():
            continue
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(fd)
        rows: list[dict[str, Any]] = []
        notes: dict[str, int] = {}
        started = time.perf_counter()
        for k in range(start, min(start + args.shard, stop)):
            got, note = fill_position(ctx, k, args.check_every > 0 and k % args.check_every == 0)
            rows.extend(got)
            for key, value in note.items():
                notes[key] = notes.get(key, 0) + value
        notes["positions"] = min(start + args.shard, stop) - start
        notes["seconds"] = round(time.perf_counter() - started, 3)
        write_shard(path, rows, notes, meta)
        claim.unlink(missing_ok=True)
        print(f"  [{args.worker}] shard {start}: {len(rows)} views, {notes['seconds']:.1f}s", flush=True)


def run_fill(args: argparse.Namespace) -> None:
    if args.worker is not None:
        run_worker(args)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.claim"):  # a killed run's claims; one fill runs at a time
        stale.unlink()
    env = dict(os.environ)
    env["POKEURAOU_RUST_NODE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    servers: list[subprocess.Popen] = []
    addresses: list[str] = []
    for s in range(args.served):
        log = (out / f"inference{s}.log").open("w", encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                str(ROOT / "tools" / "inference_server.py"),
                "--device",
                "cuda",
                "--regulation",
                args.regulation,
                "--report-every",
                "0",
                "--arm",
                "value",
                *args.value,
            ],
            env=env,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
        )
        servers.append(process)
        assert process.stdout is not None
        address = process.stdout.readline().strip()
        if not address:
            raise SystemExit(f"inference server {s} named no address")
        addresses.append(address)
    started = time.perf_counter()
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "fill",
        "--index",
        str(args.index),
        "--out",
        str(out),
        "--start",
        str(args.start),
        "--stop",
        str(args.stop),
        "--shard",
        str(args.shard),
        "--pool",
        args.pool,
        "--device",
        args.device,
        "--check-every",
        str(args.check_every),
        *(["--branch-spread"] if args.branch_spread else []),
        "--value",
        *args.value,
    ]
    if args.games_dir:
        base += ["--games-dir", str(args.games_dir)]
    procs = []
    for w in range(args.jobs):
        extra = ["--inference", addresses[w % len(addresses)]] if addresses else []
        procs.append(subprocess.Popen([*base, "--worker", str(w), *extra], env=env))  # noqa: S603
    codes = [p.wait() for p in procs]
    for server in servers:
        server.terminate()
    print(
        f"fill [{args.start}, {args.stop}) with {args.jobs} jobs: {time.perf_counter() - started:.1f}s, "
        f"exit codes {codes}",
        flush=True,
    )
    if any(codes):
        raise SystemExit(1)


# --------------------------------------------------------------------------------------
# features: the port's per-candidate numbers, beside each shard


def feature_path(shard: Path) -> Path:
    return shard.with_name(shard.name.replace("shard-", "feat-"))


def run_features(args: argparse.Namespace) -> None:
    """`qhead.port_features` for every view of every shard, as `feat-*.npz` beside it.

    The views are rebuilt from the index as `fill` built them, and each stored view is
    matched by its side and both choice lists; a mismatch stops (the shard is not the
    position this rebuilds). Rows line up with the shard's `acts0` / `acts1`.
    """
    if args.worker is None and args.jobs > 1:
        base = [
            sys.executable,
            str(Path(__file__).resolve()),
            "features",
            "--index",
            str(args.index),
            "--out",
            str(args.out),
            "--pool",
            args.pool,
            "--jobs",
            str(args.jobs),
        ]
        if args.games_dir:
            base += ["--games-dir", str(args.games_dir)]
        procs = [subprocess.Popen([*base, "--worker", str(w)]) for w in range(args.jobs)]  # noqa: S603
        codes = [p.wait() for p in procs]
        print(f"features: exit codes {codes}", flush=True)
        if any(codes):
            raise SystemExit(1)
        return
    from pokeuraou import qhead
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.pool import load_pool

    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    index = Index(Path(args.index), args.games_dir)
    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    worker = args.worker or 0
    shards = sorted(Path(args.out).glob("shard-*.npz"))
    for n, shard in enumerate(shards):
        if n % args.jobs != worker or feature_path(shard).exists():
            continue
        with np.load(shard) as z:
            ks, sides = z["k"], z["side"]
            choices = json.loads(str(z["choices"]))
        f0: list[np.ndarray] = []
        f1: list[np.ndarray] = []
        seconds: list[float] = []
        refused = 0
        built: dict[int, list[tuple[Any, tuple[list, list]]]] = {}
        for v in range(len(ks)):
            k = int(ks[v])
            if k not in built:
                game = index.game(k)
                decision = game["decisions"][int(index.arrays["decision"][k])]
                built[k] = [
                    (view, (qhead.legal_pool(reg, view.position, 0), qhead.legal_pool(reg, view.position, 1)))
                    for view in qhead.decision_views(reg, pool, game, decision)
                ]
            want = choices[v]
            found = [
                (view, pools)
                for view, pools in built[k]
                if (view.side == int(sides[v]) or int(sides[v]) == 2)
                and [a.to_choice() for a in pools[0]] == want[0]
                and [a.to_choice() for a in pools[1]] == want[1]
            ]
            if not found:
                raise SystemExit(f"{shard.name} view {v} (position {k}) does not rebuild")
            view, pools = found[0]
            started = time.perf_counter()
            got = qhead.port_features(reg, view.position, pools)
            seconds.append(time.perf_counter() - started)
            if got is None:
                refused += 1
                got = (
                    np.zeros((len(pools[0]), qhead.FEATURE_WIDTH), np.float32),
                    np.zeros((len(pools[1]), qhead.FEATURE_WIDTH), np.float32),
                )
            f0.append(got[0])
            f1.append(got[1])
        buffer = io.BytesIO()
        np.savez(
            buffer,
            feats0=np.concatenate(f0) if f0 else np.zeros((0, qhead.FEATURE_WIDTH), np.float32),
            feats1=np.concatenate(f1) if f1 else np.zeros((0, qhead.FEATURE_WIDTH), np.float32),
            seconds=np.asarray(seconds),
            refused=np.array(refused),
        )
        tmp = feature_path(shard).with_suffix(".tmp")
        tmp.write_bytes(buffer.getvalue())
        os.replace(tmp, feature_path(shard))
        print(
            f"  [{worker}] {shard.name}: {len(ks)} views, {sum(seconds):.1f}s, refused {refused}", flush=True
        )


# --------------------------------------------------------------------------------------
# menus: the views reproduce the recorded menus


def run_menus(args: argparse.Namespace) -> None:
    from pokeuraou import qhead
    from pokeuraou.budget import Budget
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.narrow import narrow
    from pokeuraou.pool import load_pool
    from pokeuraou.search import leaf_ranking

    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    index = Index(Path(args.index), args.games_dir)
    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    leaf, _encoder = build_leaf(args, reg)
    budget = Budget.matrix()
    same = differ = hidden = 0
    rows = []
    for k in range(args.start, args.stop):
        game = index.game(k)
        decision = game["decisions"][int(index.arrays["decision"][k])]
        views = qhead.decision_views(reg, pool, game, decision)
        by_side = {v.side: v for v in views}
        for side, key in ((0, "ownActions"), (1, "foeActions")):
            view = by_side.get(side, by_side.get(2))
            hidden += int(view.side != 2)
            menu = narrow(
                reg,
                view.position,
                side,
                limit=int(game.get("searchLimit", 12)),
                rank=leaf_ranking(reg, view.position, side, leaf, budget=budget),
            ).actions
            got = [a.to_choice() for a in menu]
            ok = got == decision[key]
            same += ok
            differ += not ok
            rows.append({"k": k, "side": side, "view": view.side, "same": ok})
    print(json.dumps({"same": same, "differ": differ, "hidden_views": hidden}))
    if args.out:
        Path(args.out).write_bytes((json.dumps(rows) + "\n").encode("utf-8"))


# --------------------------------------------------------------------------------------
# cost


def load_shards(out: Path) -> list[dict[str, Any]]:
    got = []
    for path in sorted(out.glob("shard-*.npz")):
        with np.load(path) as z:
            got.append({k: z[k] for k in z.files if not k.startswith("enc_")} | {"path": path.name})
    return got


def run_cost(args: argparse.Namespace) -> None:
    shards = load_shards(Path(args.out))
    cost = np.concatenate([s["cost"] for s in shards])
    n0 = np.concatenate([s["n0"] for s in shards]).astype(np.float64)
    n1 = np.concatenate([s["n1"] for s in shards]).astype(np.float64)
    side = np.concatenate([s["side"] for s in shards])
    check = np.concatenate([s["check"] for s in shards])
    positions = sum(json.loads(str(s["notes"]))["positions"] for s in shards)
    shard_seconds = sum(json.loads(str(s["notes"]))["seconds"] for s in shards)
    notes: dict[str, int] = {}
    for s in shards:
        for key, value in json.loads(str(s["notes"])).items():
            if key != "seconds":
                notes[key] = notes.get(key, 0) + value
    views = len(cost)
    cells = n0 * n1
    summary = {
        "shards": len(shards),
        "positions": positions,
        "views": views,
        "views_per_position": views / max(positions, 1),
        "shared_views": int(np.sum(side == 2)),
        "notes": notes,
        "actions_mean": [float(n0.mean()), float(n1.mean())],
        "actions_median": [float(np.median(n0)), float(np.median(n1))],
        "cells_mean": float(cells.mean()),
        "cells_p90": float(np.percentile(cells, 90)),
        "cells_max": float(cells.max()),
        "leaves_mean": float(cost[:, 3].mean()),
        "leaves_per_cell": float(cost[:, 3].sum() / cells.sum()),
        "fill_wall_s_mean": float(cost[:, 0].mean()),
        "worker_cpu_s_mean": float(cost[:, 1].mean()),
        "port_cpu_s_mean": float(cost[:, 2].mean()),
        "shard_wall_s_per_position": shard_seconds / max(positions, 1),
        "bytes": sum((Path(args.out) / s["path"]).stat().st_size for s in shards),
    }
    checked = check[~np.isnan(check[:, 0])]
    if len(checked):
        whole_cells = (n0 * n1)[~np.isnan(check[:, 0])]
        summary["control"] = {
            "views": int(len(checked)),
            "whole_equal_cells": int(checked[:, 0].sum()),
            "whole_cells": int(whole_cells.sum()),
            "whole_views_all_equal": int(np.sum(checked[:, 0] == whole_cells)),
            "whole_max_diff": float(checked[:, 1].max()),
            "w12_equal_cells": int(checked[:, 2].sum()),
            "w12_cells": int(checked[:, 3].sum()),
            "w12_views_all_equal": int(np.sum(checked[:, 2] == checked[:, 3])),
            "w12_max_diff": float(checked[:, 4].max()),
            "port_cells_compared": int(np.nansum(checked[:, 5])),
            "port_cells_same_leaves_and_weights": int(np.nansum(checked[:, 6])),
            "port_cells_same_value_scored_alone": int(np.nansum(checked[:, 7])),
            "port_folded_cells_not_compared": int(np.nansum(checked[:, 8])),
        }
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index")
    index.add_argument("--games-dir", type=Path, required=True)
    index.add_argument("--out", type=Path, required=True)
    index.add_argument("--seed", type=int, default=274)
    index.add_argument(
        "--skip-lines",
        type=int,
        default=20,
        help="each worker file's first lines are IKA-281's yardstick games (IKA-310/311)",
    )

    for name in ("fill", "menus"):
        p = sub.add_parser(name)
        p.add_argument("--index", type=Path, required=True)
        p.add_argument("--games-dir", type=Path, default=None)
        p.add_argument("--out", type=Path, default=None)
        p.add_argument("--start", type=int, default=0)
        p.add_argument("--stop", type=int, required=True)
        p.add_argument("--pool", default="regmc-matchupweb")
        p.add_argument("--value", nargs="+", required=True)
        p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
        p.add_argument("--inference", default=None, help=argparse.SUPPRESS)
        if name == "fill":
            p.add_argument("--shard", type=int, default=100)
            p.add_argument("--jobs", type=int, default=1)
            p.add_argument(
                "--served",
                type=int,
                default=0,
                help="inference servers on cuda (0: each worker holds the leaf)",
            )
            p.add_argument("--regulation", default="gen9championsvgc2026regmc")
            p.add_argument("--check-every", type=int, default=0)
            p.add_argument(
                "--branch-spread",
                action="store_true",
                help="also record each cell's chance-branch spread (IKA-322; off by default)",
            )
            p.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)

    feats = sub.add_parser("features")
    feats.add_argument("--index", type=Path, required=True)
    feats.add_argument("--games-dir", type=Path, default=None)
    feats.add_argument("--out", type=Path, required=True, help="the shard directory")
    feats.add_argument("--pool", default="regmc-matchupweb")
    feats.add_argument("--jobs", type=int, default=1)
    feats.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)

    cost = sub.add_parser("cost")
    cost.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    {
        "index": run_index,
        "fill": run_fill,
        "menus": run_menus,
        "features": run_features,
        "cost": run_cost,
    }[args.command](args)


if __name__ == "__main__":
    main()
