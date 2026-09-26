"""Has the candidate model Q gone stale under a new leaf? The per-generation check (IKA-346/348).

Q ranks the candidates a width-12 menu keeps; it learned the depth-1 matrices of one leaf
(`q_teach.py fill`). A new leaf answers the same matrices differently, and a Q taught by the
old one ranks by answers the search no longer gives. The check fills the same positions
twice -- with the leaf Q was taught by and with the new one -- and asks how much Q's cell
error grows:

    r = MAE(Q, new leaf's cells) / MAE(Q, old leaf's cells) - 1

**r < 0.11 keeps the Q.** 11% is the size of an improvement that did not show on the board:
IKA-274 stage 3 cut Q's held-out cell error by 11% (10 -> 30 epochs) and the 4,000-pair board
said +4.1 [-2.4, +10.6], indistinguishable. A staleness smaller than that is below what the
board can see. At r >= 0.11 the Q is taught again with the new leaf (IKA-348).

Also reported, not part of the rule: the same with each matrix's mean difference taken out
(a level shift of a whole matrix moves no ranking), the rank correlation of the row and
column means (what orders each side's candidates), and the two leaves' own difference.

    python tools/q_teach.py index --games-dir data/selfplay-mcN --out <d>/index.npz --seed <s>
    python tools/q_teach.py fill --index <d>/index.npz --games-dir data/selfplay-mcN \\
        --out <d>/shards-old --start 0 --stop 3000 --jobs 16 --served 2 --value <old leaf files>
    python tools/q_teach.py fill ... --out <d>/shards-new ... --value <new leaf files>
    python tools/q_teach.py features --index <d>/index.npz --games-dir data/selfplay-mcN \\
        --out <d>/shards-old --jobs 16       (and copy feat-*.npz into shards-new)
    python tools/q_drift.py <d>/shards-old <d>/shards-new --q data/models/q-mcK.pt

gen-1 (IKA-346): q-mc0 on 3,000 gen-1 positions, value-mc0x2 against value-mc1x2:
r = +55.7% (level shift out: +29.4%), so Q was taught again.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

#: IKA-346 §0.4: the size of a Q improvement the board could not see (IKA-274 stage 3).
KEEP_BELOW = 0.11


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable")
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable")
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(a: list[np.ndarray], b: list[np.ndarray]) -> dict[str, float]:
    """Matrix by matrix, `a` against `b`: MAE, MAE with the mean difference out, rank
    correlations of the row means and the column means (each averaged over the matrices)."""
    rows = []
    for x, y in zip(a, b, strict=True):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        d = x - y
        rows.append((np.abs(d).mean(), np.abs(d - d.mean()).mean(),
                     _spearman(x.mean(1), y.mean(1)), _spearman(x.mean(0), y.mean(0))))
    r = np.array(rows)
    return {"mae": float(np.nanmean(r[:, 0])), "mae_demeaned": float(np.nanmean(r[:, 1])),
            "spearman_rows": float(np.nanmean(r[:, 2])), "spearman_cols": float(np.nanmean(r[:, 3]))}


def drift(q: list[np.ndarray], old: list[np.ndarray], new: list[np.ndarray]) -> dict[str, Any]:
    """The rule's numbers for one Q's predictions `q` against the old and the new leaf's cells."""
    vs_old, vs_new = compare(q, old), compare(q, new)
    r = vs_new["mae"] / vs_old["mae"] - 1
    return {
        "q_vs_old": vs_old,
        "q_vs_new": vs_new,
        "old_vs_new": compare(old, new),
        "r": r,
        "r_demeaned": vs_new["mae_demeaned"] / vs_old["mae_demeaned"] - 1,
        "keep": bool(r < KEEP_BELOW),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old", type=Path, help="shards filled with the leaf the Q was taught by")
    ap.add_argument("new", type=Path, help="the same positions filled with the new leaf")
    ap.add_argument("--q", type=Path, nargs="+", required=True, help="the Q(s) to check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="write the numbers here as JSON")
    args = ap.parse_args(argv)

    import torch
    from q_train import Views, predict

    from pokeuraou.qhead import load_q

    if args.device == "cpu":
        torch.set_num_threads(1)
    old = Views(args.old, features=True, dedupe=False)
    new = Views(args.new, features=True, dedupe=False)
    if len(old) != len(new):
        raise SystemExit(f"{len(old)} views against {len(new)}: not the same positions")
    for key in ("k", "side", "game"):
        if not np.array_equal(old.meta[key], new.meta[key]):
            raise SystemExit(f"the two fills differ in {key}: not the same positions")
    for v in range(len(old)):
        if not (np.array_equal(old.acts0[v], new.acts0[v]) and np.array_equal(old.acts1[v], new.acts1[v])):
            raise SystemExit(f"view {v}: the two fills offer different actions")
    out: dict[str, Any] = {"views": len(old), "games": int(len(np.unique(old.meta["game"]))),
                           "keep_below": KEEP_BELOW, "q": {}}
    index = np.arange(len(old))
    for path in args.q:
        net = load_q(path, args.device)
        predicted = predict(net, old, index, torch.device(args.device))
        report = drift(predicted, old.cells, new.cells)
        out["q"][str(path)] = report
        print(f"{path.name}: {len(old)} views; MAE against the old leaf {report['q_vs_old']['mae']:.4f}, "
              f"the new {report['q_vs_new']['mae']:.4f}; r {report['r']:+.1%} "
              f"(level shift out {report['r_demeaned']:+.1%}); the leaves differ by "
              f"{report['old_vs_new']['mae']:.4f} -> {'keep' if report['keep'] else 'teach again'} "
              f"(rule: r < {KEEP_BELOW:.0%})", flush=True)
    if args.out is not None:
        args.out.write_bytes((json.dumps(out, indent=1) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
