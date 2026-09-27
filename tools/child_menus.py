"""How much of a child game's equilibrium each way of building its menus keeps (IKA-362).

The deepening's children are read on small menus: `narrow`'s damage order with the cover
(every slot option somewhere on the menu) at ``sub_limit`` (8). This measures, on the child
positions a read of the verification position set meets, what each way of building those
menus keeps of the child's whole game:

* ``damage``: `narrow`'s damage order with the cover (what the deepening does);
* ``q``: the k best by the Q solved over both whole pools, no cover (`deepen._q_menus`,
  the ``c<k>`` labels);
* ``qcover``: the same order with the cover.

For each child (the likeliest branch of each cell of the position's depth-1 support, up to
``--rect`` a side), the whole game is every legal action of both sides (`qhead.legal_pool`)
at depth 1 with the leaf. Per menu: the recall (the whole game's equilibrium weight on the
menu, each side), the NashConv of the menu's equilibrium in the whole game, the share of the
menu's slots the cover took (`Narrowed.for_coverage`), and the joint weight of the whole
game's (row, column) pairs the menu's rectangle misses.

    python tools/child_menus.py --set SET --from 0 --to 15 --out OUT.jsonl
    python tools/child_menus.py --report OUT*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pokeuraou import openmp  # noqa: E402

openmp.quiet_wait()  # before anything loads torch (IKA-360)

import numpy as np  # noqa: E402

WIDTHS = (8, 16, 24)


def _menus(reg, pos, pools, q, width, cover):  # noqa: ANN001, ANN202
    """Both sides' menus by the Q's order (`deepen._q_menus`), with or without the cover."""
    from pokeuraou.equilibrium import EquilibriumError, solve
    from pokeuraou.narrow import narrow

    try:
        e = solve(q)
        scores = (np.asarray(e.row_ev), -np.asarray(e.col_ev))
    except (EquilibriumError, ValueError):
        scores = (q.mean(axis=1), -q.mean(axis=0))
    out = []
    for side in (0, 1):
        by = {a.to_choice(): float(s) for a, s in zip(pools[side], scores[side], strict=True)}

        def rank(pool, _scored=None, by=by):  # noqa: ANN001, ANN202
            return [by[a.to_choice()] for a in pool]

        out.append(narrow(reg, pos, side, limit=width, candidates=pools[side], rank=rank, cover=cover))
    return out


def measure(args: argparse.Namespace) -> None:
    from position_set import _Kit

    from pokeuraou import port, qhead, qrank
    from pokeuraou.budget import Budget
    from pokeuraou.equilibrium import solve
    from pokeuraou.narrow import narrow
    from pokeuraou.position import Position
    from pokeuraou.search import batched_payoff

    kit = _Kit(args)
    budget = Budget.matrix()
    out = Path(args.out)
    for n in range(args.start, min(args.stop, len(kit.positions))):
        ref = np.load(Path(args.set) / f"ref-{args.ref}" / f"{n}.npz")
        pos = Position.from_json(kit.positions[n]["position"])
        ours, theirs, _outside = kit.menus(pos)
        eq = solve(np.asarray(ref["d1"], dtype=np.float64))
        rows = np.argsort(-eq.row_strategy)[: args.rect]
        cols = np.argsort(-eq.col_strategy)[: args.rect]
        for i in rows:
            if eq.row_strategy[i] <= 1e-6:
                continue
            for j in cols:
                if eq.col_strategy[j] <= 1e-6:
                    continue
                result = port.turn(kit.reg, pos, [ours[i], theirs[j]], budget, full=True)
                if result.suspended or not result.outcomes:
                    continue
                child = max(result.outcomes, key=lambda b: b.probability).position
                if child.ended:
                    continue
                began = time.perf_counter()
                pools = (qhead.legal_pool(kit.reg, child, 0), qhead.legal_pool(kit.reg, child, 1))
                if not pools[0] or not pools[1]:
                    continue
                whole, _notes = batched_payoff(kit.reg, child, pools[0], pools[1], kit.leaf, budget=budget)
                whole = np.asarray(whole, dtype=np.float64)
                weq = solve(whole)
                index = [{a.to_choice(): k for k, a in enumerate(pools[s])} for s in (0, 1)]
                q_started = time.perf_counter()
                q = np.asarray(qrank.installed().matrix(kit.reg, child, pools), dtype=np.float64)
                q_seconds = time.perf_counter() - q_started
                rec = {"n": n, "cell": [int(i), int(j)], "pools": [len(pools[0]), len(pools[1])],
                       "wholeSupport": [int((weq.row_strategy > 1e-6).sum()),
                                        int((weq.col_strategy > 1e-6).sum())],
                       "qSeconds": round(q_seconds, 4)}
                for width in WIDTHS:
                    forms = {
                        "damage": [narrow(kit.reg, child, s, limit=width) for s in (0, 1)],
                        "q": _menus(kit.reg, child, pools, q, width, cover=False),
                        "qcover": _menus(kit.reg, child, pools, q, width, cover=True),
                    }
                    for form, got in forms.items():
                        sel = [[index[s][a.to_choice()] for a in got[s].actions if a.to_choice() in index[s]]
                               for s in (0, 1)]
                        recall = [float(weq.row_strategy[sel[0]].sum()),
                                  float(weq.col_strategy[sel[1]].sum())]
                        sub = whole[np.ix_(sel[0], sel[1])]
                        seq = solve(sub)
                        x = np.zeros(len(pools[0]))
                        x[sel[0]] = seq.row_strategy
                        y = np.zeros(len(pools[1]))
                        y[sel[1]] = seq.col_strategy
                        nashconv = float((whole @ y).max() - (x @ whole).min())
                        joint_missed = float(1.0 - recall[0] * recall[1])
                        cover_share = [got[s].for_coverage / max(1, len(got[s].kept)) for s in (0, 1)]
                        rec[f"{form}{width}"] = {
                            "recall": recall, "nashconv": nashconv, "jointMissed": joint_missed,
                            "coverShare": cover_share, "size": [len(sel[0]), len(sel[1])],
                        }
                rec["seconds"] = round(time.perf_counter() - began, 3)
                with out.open("ab") as handle:
                    handle.write((json.dumps(rec) + "\n").encode("utf-8"))
        print(f"position {n} done", file=sys.stderr, flush=True)


def report(paths: list[str]) -> None:
    rows = [json.loads(x) for p in paths for f in glob.glob(p)
            for x in Path(f).read_bytes().splitlines() if x.strip()]
    print(f"{len(rows)} children from {len({r['n'] for r in rows})} positions; pools mean "
          f"{np.mean([r['pools'] for r in rows], axis=0).round(1)}, whole support mean "
          f"{np.mean([r['wholeSupport'] for r in rows], axis=0).round(2)}, Q per child "
          f"{np.mean([r['qSeconds'] for r in rows]) * 1000:.1f} ms")
    for width in WIDTHS:
        for form in ("damage", "q", "qcover"):
            key = f"{form}{width}"
            rec = np.array([np.mean(r[key]["recall"]) for r in rows])
            nc = np.array([r[key]["nashconv"] for r in rows])
            jm = np.array([r[key]["jointMissed"] for r in rows])
            cs = np.array([np.mean(r[key]["coverShare"]) for r in rows])
            se = nc.std() / np.sqrt(len(nc))
            print(f"  {key:<9} recall {rec.mean():.3f}  NashConv {nc.mean():.4f} (se {se:.4f})"
                  f"  joint missed {jm.mean():.3f}  cover share {cs.mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", nargs="+", default=None)
    ap.add_argument("--set", type=Path)
    ap.add_argument("--ref", default="r24", help="the reference whose depth-1 matrix picks the cells")
    ap.add_argument("--rect", type=int, default=3)
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--to", dest="stop", type=int, default=10**9)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pool", default="regmc-matchupweb")
    ap.add_argument("--value", type=Path, nargs="+", default=None)
    ap.add_argument("--q-model", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cuda-memory-gb", type=float, default=0.7)
    args = ap.parse_args()
    if args.report:
        report(args.report)
    else:
        measure(args)


if __name__ == "__main__":
    main()
