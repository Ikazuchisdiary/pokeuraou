"""IKA-392 B0: does reading the selection deeper change it, and is the change an improvement?

For pairs drawn from the pool (the M-C pool by default): the leaf selection (`selection.
solve_selection`, what every path plays now), then the selection read deeper
(`selection_deep.solve_selection_deep`) with each way of pricing the cells not read
(``--variants``: by default the three ways of pricing the cells not read, none, const and
add; a variant is a reading spec added to ``--reading``, so it may also be another budget),
all sharing the cells read (a table per pair: a cell is read once, however many variants
want it). Written per pair to ``OUT/pairs/<n>.json`` (a pair done is not read again).

B0-a, *does it change*: per shift, the total variation of the row and the column strategy
from the leaf's, and the chance that the four a side draws differs when both are drawn from
the same uniform number (the coupling `humanplay.agent_pick` gives two arms with one seed:
1 - sum_k |[F_a(k-1), F_a(k)] & [F_b(k-1), F_b(k)]|). Control that the comparison can fail:
the same measure for the leaf matrix plus noise of the size the deep read moved the cells
(``noise``), and for the leaf against itself (0 by construction).

B0-b, *is it better*: for the first ``--ref-pairs`` pairs, the cells of a rectangle (the
supports of the leaf answer and of each shift's answer, filled up to ``--ref-rect`` by the
leaf answer's best replies) read by ``--ref-stage`` -- deeper than the selections' one stage
-- as the reference matrix; each strategy's loss in it is its value against the reference's
value: the row strategy's guarantee against the columns of the rectangle, the column's
the other way (`loss_row`, `loss_col`). The reference's cells are kept per pair
(``OUT/ref/<n>.json``): a run with more variants reads only the cells it adds.

The workers are the machine's inference server's clients: ``--served`` starts one.

    python tools/selection_sweep.py --out OUT --pairs 60 --ref-pairs 10 \\
        --workers 13 --served
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import humanplay  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402
from pokeuraou.selection import SpreadClass, solve_selection  # noqa: E402
from pokeuraou.selection_deep import (  # noqa: E402
    PoolReader,
    Reading,
    SerialReader,
    solve_selection_deep,
)

DEFAULT_VALUE = ("data/models/value-mc4st.pt", "data/models/value-mc4st-s1.pt")
DEFAULT_Q = "data/models/q-mc4.pt"


class TableReader:
    """A reader that keeps every cell it has read (per pair), so the readings that share a
    pair read a cell once."""

    def __init__(self, inner, reading_key: str) -> None:  # noqa: ANN001
        self.inner, self.key, self.table = inner, reading_key, {}
        self.asked = 0

    @property
    def workers(self) -> int:
        return self.inner.workers

    def read(self, reading, our, their, selections, cells, deadline=None):  # noqa: ANN001, ANN201
        need = [c for c in cells if c not in self.table]
        self.asked += len(cells)
        if need:
            self.table.update(self.inner.read(reading, our, their, selections, need, deadline))
        return {c: self.table[c] for c in cells if c in self.table}


def coupled_change(a: np.ndarray, b: np.ndarray) -> float:
    """The chance that one uniform number picks different actions from mixtures ``a`` and ``b``
    (drawn by the cumulative sums, as `humanplay.agent_pick`)."""
    ca, cb = np.cumsum(a) / np.sum(a), np.cumsum(b) / np.sum(b)
    la, lb = np.concatenate([[0.0], ca[:-1]]), np.concatenate([[0.0], cb[:-1]])
    overlap = np.maximum(0.0, np.minimum(ca, cb) - np.maximum(la, lb))
    return float(1.0 - overlap.sum())


def tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def strategy_loss(matrix: np.ndarray, x: np.ndarray, y: np.ndarray, rows: list[int],
                  cols: list[int]) -> dict:
    """The losses of ``x`` (row) and ``y`` (column) in the rectangle ``rows`` x ``cols`` of
    ``matrix``: the value the rectangle's game has minus what x guarantees against its
    columns (row), and what y allows against its rows minus the value (column)."""
    sub = matrix[np.ix_(rows, cols)]
    value = float(solve(sub).value)
    xr, yc = np.asarray(x)[rows], np.asarray(y)[cols]
    outside_row = float(np.asarray(x).sum() - xr.sum())
    outside_col = float(np.asarray(y).sum() - yc.sum())
    guarantee = float((xr / xr.sum() @ sub).min()) if xr.sum() > 0 else float("nan")
    allowed = float((sub @ (yc / yc.sum())).max()) if yc.sum() > 0 else float("nan")
    return {"value": value, "loss_row": value - guarantee, "loss_col": allowed - value,
            "outside_row": outside_row, "outside_col": outside_col}


def main() -> None:  # noqa: C901, PLR0912, PLR0915 - the sweep's steps
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pool", default="regmc-matchupweb")
    ap.add_argument("--pairs", type=int, default=60)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--seed", type=int, default=392)
    ap.add_argument("--ref-pairs", type=int, default=0)
    ap.add_argument("--ref-rect", type=int, default=12)
    ap.add_argument("--ref-stage", default="d2r4b3k8+d2r8b3k8+d2r12b3k8")
    ap.add_argument("--reading", default="stage=d2r4b3k8,rects=8-16,confirm=2")
    ap.add_argument("--variants", default="none=shift=none|const=shift=const|add=shift=add",
                    help="name=reading|name=reading: each is --reading with its keys added")
    ap.add_argument("--workers", type=int, default=1, help="reader processes (1: in this process)")
    ap.add_argument("--served", action="store_true")
    ap.add_argument("--inference", default=None)
    ap.add_argument("--value", type=Path, nargs="+", default=None)
    ap.add_argument("--q-model", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from pokeuraou.selection_deep import parse_reading

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    values = args.value or [ROOT / p for p in DEFAULT_VALUE]
    q_path = args.q_model or ROOT / DEFAULT_Q
    (args.out / "pairs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    server = None
    address = args.inference
    if args.served:
        from pokeuraou.inference import start_server

        server, address = start_server(
            {"value": values}, q_arms={"q": q_path}, regulation=reg.meta.format_id,
            log=args.out / "inference.log", env=env, device=args.device)
    if not address:
        raise SystemExit("the sweep reads through an inference server: --served or --inference")
    evaluate, _encoder = humanplay.served_leaf(reg, address, values, merge=False, q_path=q_path)
    spec = ("served", address, "value", False, str(q_path), "q")
    humanplay.use_threads(1, reg, None)
    base_reading = parse_reading(args.reading)
    reader = (PoolReader(reg, args.workers, spec) if args.workers > 1
              else SerialReader(reg, evaluate))
    variants = [(v.partition("=")[0], v.partition("=")[2]) for v in args.variants.split("|")]
    teams = pool.teams
    started = time.perf_counter()
    try:
        for n in range(args.first, args.first + args.pairs):
            path = args.out / "pairs" / f"{n}.json"
            if path.exists() and (n >= args.ref_pairs or "ref" in json.loads(path.read_bytes())):
                continue
            rng = np.random.default_rng([args.seed, n])
            a, b = (int(v) for v in rng.choice(len(teams), size=2, replace=False))
            row, col = teams[a], teams[b]
            t0 = time.perf_counter()
            leaf = solve_selection(reg, row.sets, [SpreadClass(1.0, tuple(col.sets), "sheet")], evaluate)
            leaf_seconds = time.perf_counter() - t0
            x0 = np.asarray(leaf.equilibrium.row_strategy)
            y0 = np.asarray(leaf.equilibrium.col_strategies[0])
            table = TableReader(reader, base_reading.label)
            out: dict = {"n": n, "teams": [row.id, col.id], "leafSeconds": round(leaf_seconds, 2),
                         "leafValue": float(leaf.value),
                         "leafSupport": [int((x0 > 1e-6).sum()), int((y0 > 1e-6).sum())],
                         "modes": {}}
            answers = {}
            for mode, extra in variants:
                reading = parse_reading(args.reading + "," + extra)
                began = time.perf_counter()
                deep, report = solve_selection_deep(reg, row.sets, col.sets, evaluate, table,
                                                    reading, base=leaf)
                x = np.asarray(deep.equilibrium.row_strategy)
                y = np.asarray(deep.equilibrium.col_strategies[0])
                answers[mode] = (x, y)
                out["modes"][mode] = {
                    "value": float(deep.value), "completed": report.completed,
                    "cells": report.cells, "seconds": round(time.perf_counter() - began, 1),
                    "tvRow": tv(x, x0), "tvCol": tv(y, y0),
                    "changeRow": coupled_change(x0, x), "changeCol": coupled_change(y0, y),
                    "argmaxRowChanged": bool(np.argmax(x) != np.argmax(x0)),
                    "support": [int((x > 1e-6).sum()), int((y > 1e-6).sum())],
                    "steps": report.steps, "x": x.round(6).tolist(), "y": y.round(6).tolist(),
                }
            read_cells = dict(table.table)
            leaf_matrix = np.asarray(leaf.matrices[0])
            residual = np.array([v - leaf_matrix[c] for c, v in read_cells.items()])
            out["cellsRead"] = len(read_cells)
            # The cells read, with the leaf's price of each: what the shift's held-out check
            # (records/IKA-392.md's held-out check) is fitted and scored on.
            out["readCells"] = [[int(i), int(j), round(float(v), 6), round(float(leaf_matrix[i, j]), 6)]
                                for (i, j), v in read_cells.items()]
            out["residual"] = {"mean": float(residual.mean()), "sd": float(residual.std())}
            # The control that the comparison can fail: the leaf matrix plus noise of the
            # deep read's size (sd of the residual around its mean), solved as the leaf is.
            noisy = np.clip(leaf_matrix + np.random.default_rng([args.seed, n, 7]).normal(
                0.0, float(residual.std()), leaf_matrix.shape), 0.0, 1.0)
            eq = solve(noisy)
            out["noise"] = {"tvRow": tv(eq.row_strategy, x0),
                            "changeRow": coupled_change(x0, eq.row_strategy),
                            "changeCol": coupled_change(y0, eq.col_strategy)}
            out["leafItself"] = {"changeRow": coupled_change(x0, x0)}
            out["x0"] = x0.round(6).tolist()
            out["y0"] = y0.round(6).tolist()
            if n - args.first < args.ref_pairs:
                sels = leaf.ours
                rows = sorted({int(i) for m in answers.values() for i in np.flatnonzero(m[0] > 1e-6)}
                              | {int(i) for i in np.flatnonzero(x0 > 1e-6)})
                cols = sorted({int(j) for m in answers.values() for j in np.flatnonzero(m[1] > 1e-6)}
                              | {int(j) for j in np.flatnonzero(y0 > 1e-6)})
                for k in np.argsort(-(leaf_matrix @ y0)):
                    if len(rows) >= args.ref_rect:
                        break
                    if int(k) not in rows:
                        rows.append(int(k))
                for k in np.argsort(x0 @ leaf_matrix):
                    if len(cols) >= args.ref_rect:
                        break
                    if int(k) not in cols:
                        cols.append(int(k))
                ref_reading = Reading(stage=args.ref_stage, width=base_reading.width,
                                   value=base_reading.value)
                began = time.perf_counter()
                cache_path = args.out / "ref" / f"{n}.json"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cached = ({tuple(int(v) for v in k.split(",")): val
                           for k, val in json.loads(cache_path.read_bytes()).items()}
                          if cache_path.exists() else {})
                need = [(i, j) for i in rows for j in cols if (i, j) not in cached]
                if need:
                    cached.update(reader.read(ref_reading, row.sets, col.sets, sels, need))
                    cache_path.write_bytes(json.dumps(
                        {f"{i},{j}": v for (i, j), v in cached.items()}).encode("utf-8"))
                got = cached
                out["refCellsRead"] = len(need)
                refm = np.full((len(rows), len(cols)), np.nan)
                for ii, i in enumerate(rows):
                    for jj, j in enumerate(cols):
                        refm[ii, jj] = got[(i, j)]
                full = np.zeros((len(sels), len(sels)))
                for ii, i in enumerate(rows):
                    for jj, j in enumerate(cols):
                        full[i, j] = refm[ii, jj]
                ref = {"rows": rows, "cols": cols, "seconds": round(time.perf_counter() - began, 1),
                       "leafMatrixInRect": float(np.mean([leaf_matrix[i, j] for i in rows for j in cols])),
                       "refMean": float(refm.mean()),
                       "leaf": strategy_loss(full, x0, y0, rows, cols)}
                for mode, (x, y) in answers.items():
                    ref[mode] = strategy_loss(full, x, y, rows, cols)
                out["ref"] = ref
            path.with_suffix(".tmp").write_bytes(json.dumps(out).encode("utf-8"))
            path.with_suffix(".tmp").replace(path)
            m = out["modes"]
            print(f"pair {n}: cells {out['cellsRead']}, "
                  + ", ".join(f"{k} changeRow {v['changeRow']:.3f} changeCol {v['changeCol']:.3f}"
                              for k, v in m.items())
                  + f", noise {out['noise']['changeRow']:.3f}"
                  + (f", ref loss leaf {out['ref']['leaf']['loss_row']:.4f} "
                     + " ".join(f"{k} {out['ref'][k]['loss_row']:.4f}" for k in m)
                     if "ref" in out else "")
                  + f"  [{time.perf_counter() - started:.0f}s]", file=sys.stderr, flush=True)
    finally:
        reader.close()
        if server is not None:
            server.terminate()


if __name__ == "__main__":
    main()
