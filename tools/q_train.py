"""Train the candidate model Q(s, a, b) on `q_teach.py`'s matrices (IKA-274).

The target is the matrix itself -- every cell, dense and smooth -- not an equilibrium's
weights (G1's policy head learned sparse equilibrium weights and lost). Soft binary cross
entropy per cell, each view weighted once whatever its size. Positions are split by game
(`--holdout` of the games), and each batch mirrors half its views (sides swapped, the
matrix transposed and complemented), which is the zero-sum symmetry the leaf has by
construction.

Held-out report: the cells' error, and each view's equilibrium -- the value error
|v(Q) - v(M)|, and what Q's equilibrium strategies give up in the true matrix (NashConv).

    python tools/q_train.py --shards <dir>/shards --out <dir>/q.pt --epochs 8 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ENCODED = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


class Views:
    """Every view of every shard, in memory, indexable by view number."""

    def __init__(
        self, shard_dir: Path, limit: int | None = None, features: bool = False, dedupe: bool = True
    ) -> None:
        import hashlib

        parts: dict[str, list[np.ndarray]] = {}
        cells: list[np.ndarray] = []
        acts0: list[np.ndarray] = []
        acts1: list[np.ndarray] = []
        feats0: list[np.ndarray] = []
        feats1: list[np.ndarray] = []
        keep: list[np.ndarray] = []
        seen: set[bytes] = set()
        self.duplicates = 0
        n_views = 0
        for path in sorted(shard_dir.glob("shard-*.npz")):
            with np.load(path) as z:
                if len(z["k"]) == 0:
                    continue
                if features:
                    with np.load(path.with_name(path.name.replace("shard-", "feat-"))) as f:
                        f0_all, f1_all = f["feats0"], f["feats1"]
                # A position met twice (the same opening of the same pair, or both
                # sides' guesses right) is one teaching matrix, not two.
                mine = np.ones(len(z["k"]), dtype=bool)
                if dedupe:
                    for v in range(len(z["k"])):
                        h = hashlib.sha1()
                        for name in ENCODED:
                            h.update(np.ascontiguousarray(z[f"enc_{name}"][v]).tobytes())
                        a, b = int(z["act0_start"][v]), int(z["act1_start"][v])
                        h.update(z["acts0"][a : a + int(z["n0"][v])].tobytes())
                        h.update(z["acts1"][b : b + int(z["n1"][v])].tobytes())
                        key = h.digest()
                        if key in seen:
                            mine[v] = False
                            self.duplicates += 1
                        seen.add(key)
                keep.append(mine)
                for key in ("k", "side", "turn", "game", "n0", "n1"):
                    parts.setdefault(f"meta_{key}", []).append(z[key][mine])
                for name in ENCODED:
                    parts.setdefault(f"enc_{name}", []).append(z[f"enc_{name}"][mine])
                n0 = z["n0"].astype(np.int64)
                n1 = z["n1"].astype(np.int64)
                flat = z["cells"]
                starts = z["cell_start"]
                a0, a1 = z["acts0"], z["acts1"]
                s0, s1 = z["act0_start"], z["act1_start"]
                for v in range(len(n0)):
                    if not mine[v]:
                        continue
                    cells.append(flat[starts[v] : starts[v] + n0[v] * n1[v]].reshape(n0[v], n1[v]))
                    acts0.append(a0[s0[v] : s0[v] + n0[v]])
                    acts1.append(a1[s1[v] : s1[v] + n1[v]])
                    if features:
                        feats0.append(f0_all[s0[v] : s0[v] + n0[v]])
                        feats1.append(f1_all[s1[v] : s1[v] + n1[v]])
                n_views += int(mine.sum())
            if limit and n_views >= limit:
                break
        self.meta = {k[5:]: np.concatenate(v) for k, v in parts.items() if k.startswith("meta_")}
        self.encoded = {k: np.concatenate(parts[f"enc_{k}"]) for k in ENCODED}
        self.cells = cells
        self.acts0 = acts0
        self.acts1 = acts1
        self.feats0 = feats0 if features else None
        self.feats1 = feats1 if features else None

    def __len__(self) -> int:
        return len(self.cells)

    def split(self, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        games = np.unique(self.meta["game"])
        rng = np.random.default_rng(seed)
        held = set(rng.choice(games, size=max(1, int(len(games) * holdout)), replace=False).tolist())
        is_held = np.array([g in held for g in self.meta["game"]])
        return np.flatnonzero(~is_held), np.flatnonzero(is_held)


def collate(
    views: Views, index: np.ndarray, mirror: np.ndarray, device: Any
) -> tuple[dict, Any, Any, Any, Any]:  # noqa: ANN401
    """Padded tensors for the views `index`; `mirror[i]` swaps view i's sides."""
    import torch

    b = len(index)
    batch: dict[str, Any] = {}
    for name in ENCODED:
        arr = views.encoded[name][index].copy()
        if name != "field":
            arr[mirror] = arr[mirror][:, ::-1]
        batch[name] = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
    rows = []
    cols = []
    mats = []
    frows = []
    fcols = []
    has_f = views.feats0 is not None
    for i, v in enumerate(index):
        if mirror[i]:
            # The mirrored position: side 1's actions are the rows, and side 0's win
            # probability is one minus the transposed matrix (the leaf is antisymmetric).
            rows.append(views.acts1[v])
            cols.append(views.acts0[v])
            mats.append(1.0 - views.cells[v].T)
            if has_f:
                frows.append(views.feats1[v])
                fcols.append(views.feats0[v])
        else:
            rows.append(views.acts0[v])
            cols.append(views.acts1[v])
            mats.append(views.cells[v])
            if has_f:
                frows.append(views.feats0[v])
                fcols.append(views.feats1[v])
    n0 = max(len(r) for r in rows)
    n1 = max(len(c) for c in cols)
    a0 = np.full((b, n0, 2, 7), -1, dtype=np.int64)
    a1 = np.full((b, n1, 2, 7), -1, dtype=np.int64)
    target = np.zeros((b, n0, n1), dtype=np.float32)
    mask = np.zeros((b, n0, n1), dtype=np.float32)
    for i in range(b):
        r, c = len(rows[i]), len(cols[i])
        a0[i, :r] = rows[i]
        a1[i, :c] = cols[i]
        target[i, :r, :c] = mats[i]
        mask[i, :r, :c] = 1.0
    # A padded action is a pass by nobody: kind 0, every row -1.
    a0[..., 0] = np.maximum(a0[..., 0], 0)
    a1[..., 0] = np.maximum(a1[..., 0], 0)
    a0[..., 2:5] = np.maximum(a0[..., 2:5], 0)
    a1[..., 2:5] = np.maximum(a1[..., 2:5], 0)
    f0 = f1 = None
    if has_f:
        width = frows[0].shape[1]
        g0 = np.zeros((b, n0, width), dtype=np.float32)
        g1 = np.zeros((b, n1, width), dtype=np.float32)
        for i in range(b):
            g0[i, : len(frows[i])] = frows[i]
            g1[i, : len(fcols[i])] = fcols[i]
        f0 = torch.from_numpy(g0).to(device)
        f1 = torch.from_numpy(g1).to(device)
    return (
        batch,
        torch.from_numpy(a0).to(device),
        torch.from_numpy(a1).to(device),
        torch.from_numpy(target).to(device),
        torch.from_numpy(mask).to(device),
        f0,
        f1,
    )


#: Padded cells in one batch at most: the pair head holds pair_dim floats per cell, so a
#: batch of 32 of the largest views (28,440 cells) would hold half a gigabyte per tensor.
CELL_BUDGET = 160_000


def by_cells(views: Views, index: np.ndarray, most: int, budget: int = CELL_BUDGET) -> list[np.ndarray]:
    """`index` (in its order) cut into batches of at most `most` views and `budget` padded cells."""
    out: list[np.ndarray] = []
    current: list[int] = []
    r = c = 0
    for v in index:
        n0, n1 = views.cells[v].shape
        r2, c2 = max(r, n0), max(c, n1)
        if current and (len(current) + 1 > most or (len(current) + 1) * r2 * c2 > budget):
            out.append(np.array(current))
            current, r2, c2 = [], n0, n1
        current.append(int(v))
        r, c = r2, c2
    if current:
        out.append(np.array(current))
    return out


def predict(net: Any, views: Views, index: np.ndarray, device: Any, batch_size: int = 64) -> list[np.ndarray]:  # noqa: ANN401
    """Q for each view in `index`, unmirrored, as side 0's win probability."""
    import torch

    net.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in by_cells(views, np.asarray(index), batch_size):
            batch, a0, a1, _t, _m, f0, f1 = collate(views, chunk, np.zeros(len(chunk), bool), device)
            q = torch.sigmoid(net(batch, a0, a1, f0, f1)).double().cpu().numpy()
            for i, v in enumerate(chunk):
                out.append(q[i, : len(views.acts0[v]), : len(views.acts1[v])])
    return out


def equilibrium_report(views: Views, index: np.ndarray, predicted: list[np.ndarray]) -> dict[str, float]:
    """Cells and equilibria of the held-out views: Q against the teaching matrix M."""
    from pokeuraou.equilibrium import solve

    abs_err: list[float] = []
    sq_err: list[float] = []
    base_abs: list[float] = []
    value_err: list[float] = []
    nashconv: list[float] = []
    nashconv_uniform: list[float] = []
    for v, q in zip(index, predicted, strict=True):
        m = views.cells[v].astype(np.float64)
        diff = q - m
        abs_err.append(float(np.abs(diff).mean()))
        sq_err.append(float((diff**2).mean()))
        base_abs.append(float(np.abs(m - m.mean()).mean()))
        true = solve(m)
        guess = solve(q)
        value_err.append(abs(float(guess.value) - float(true.value)))
        x, y = guess.row_strategy, guess.col_strategy
        nashconv.append(float((m @ y).max() - (x @ m).min()))
        ux = np.full(m.shape[0], 1.0 / m.shape[0])
        uy = np.full(m.shape[1], 1.0 / m.shape[1])
        nashconv_uniform.append(float((m @ uy).max() - (ux @ m).min()))
    return {
        "views": len(index),
        "cell_mae": float(np.mean(abs_err)),
        "cell_rmse": float(np.sqrt(np.mean(sq_err))),
        "cell_mae_view_mean_baseline": float(np.mean(base_abs)),
        "eq_value_abs_err": float(np.mean(value_err)),
        "eq_value_abs_err_median": float(np.median(value_err)),
        "nashconv_q_in_m": float(np.mean(nashconv)),
        "nashconv_q_in_m_median": float(np.median(nashconv)),
        "nashconv_uniform_in_m": float(np.mean(nashconv_uniform)),
    }


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="the seed of the held-out split (default: --seed). Two seeds of one model "
        "compare on one held-out set only when this is the same",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="views to load at most")
    ap.add_argument("--eval-limit", type=int, default=2000, help="held-out views in the report")
    ap.add_argument("--regulation", default="gen9championsvgc2026regmc")
    ap.add_argument(
        "--properties",
        action="store_true",
        help="moves by id plus the dex's properties and the port's candidate features "
        "(needs feat-*.npz beside the shards: q_teach.py features)",
    )
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument(
        "--init-trunk",
        type=Path,
        default=None,
        help="a leaf (value-*.pt) whose weights start the trunk; read only",
    )
    ap.add_argument("--max-hours", type=float, default=None, help="stop after this wall clock")
    args = ap.parse_args()

    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.qhead import QConfig, build_net, config_dict, load_trunk, move_table
    from pokeuraou.regulation import load_regulation

    torch.manual_seed(args.seed)
    started = time.perf_counter()
    views = Views(args.shards, args.limit, features=args.properties, dedupe=not args.keep_duplicates)
    train_ix, held_ix = views.split(
        args.holdout, args.seed if args.split_seed is None else args.split_seed
    )
    print(
        f"{len(views)} views ({len(train_ix)} train, {len(held_ix)} held out, "
        f"{views.duplicates} duplicates dropped, {2 * len(train_ix)} with mirrors), "
        f"loaded in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    reg = load_regulation(args.regulation)
    encoder = Encoder(reg)
    config = QConfig(properties=args.properties)
    table = move_table(reg, encoder.vocab) if args.properties else None
    device = torch.device(args.device)
    net = build_net(encoder, config, table)
    if args.init_trunk is not None:
        load_trunk(net, args.init_trunk, encoder)
        print(f"trunk initialised from {args.init_trunk}", flush=True)
    net = net.to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    per_epoch = len(
        by_cells(views, train_ix[np.argsort([views.cells[v].size for v in train_ix])], args.batch)
    )
    steps = int(args.epochs * per_epoch * 1.3) + 10  # shuffled blocks pad a little worse
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=steps, pct_start=0.1
    )
    rng = np.random.default_rng(args.seed)
    # Views of like size in a batch, so padding does not dominate: sort by cells within
    # shuffled blocks.
    sizes = np.array([c.size for c in views.cells])
    history = []
    for epoch in range(args.epochs):
        if args.max_hours is not None and time.perf_counter() - started > args.max_hours * 3600:
            print(f"stopping before epoch {epoch + 1}: past {args.max_hours} h", flush=True)
            break
        net.train()
        order = rng.permutation(train_ix)
        blocks = [order[i : i + args.batch * 50] for i in range(0, len(order), args.batch * 50)]
        batches = []
        for block in blocks:
            block = block[np.argsort(sizes[block], kind="stable")]
            batches += by_cells(views, block, args.batch)
        rng.shuffle(batches)
        total = 0.0
        t0 = time.perf_counter()
        for chunk in batches:
            mirror = np.zeros(len(chunk), dtype=bool)  # antisymmetric: a mirror is the same loss
            batch, a0, a1, target, mask, f0, f1 = collate(views, chunk, mirror, device)
            logits = net(batch, a0, a1, f0, f1)
            loss_cells = (
                torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask
            )
            per_view = loss_cells.sum((1, 2)) / mask.sum((1, 2))
            loss = per_view.mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            total += float(loss.detach()) * len(chunk)
        held = held_ix[: args.eval_limit]
        predicted = predict(net, views, held, device)
        report = equilibrium_report(views, held, predicted)
        report |= {
            "epoch": epoch + 1,
            "train_bce": total / len(train_ix),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        history.append(report)
        print(json.dumps(report), flush=True)
    blob = {
        "state": net.state_dict(),
        "config": config_dict(config),
        "vocab_sizes": encoder.vocab.sizes,
        "vocab_fingerprint": encoder.vocab.fingerprint(),
        "regulation": args.regulation,
        "move_table": table,
        "widths": encoder.widths,
        "history": history,
        "views": len(views),
        "train_views": len(train_ix),
        "held_views": len(held_ix),
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    torch.save(blob, args.out)
    print(f"saved {args.out} after {time.perf_counter() - started:.1f}s")


def load_q(path: Path, device: str = "cpu") -> Any:  # noqa: ANN401
    """`qhead.load_q` (moved there so the search can load a Q, IKA-274)."""
    from pokeuraou.qhead import load_q as load

    return load(path, device)


if __name__ == "__main__":
    main()
