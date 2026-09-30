"""IKA-401: how much of a better Q's menu reaches the answer's loss (depth-1 matrices, whole pools).

    q_loss_chain.py orders --shards DIR --out ORDERS.pkl --q q-mc0.pt q-mc1.pt ...
    q_loss_chain.py score  --shards DIR --orders ORDERS.pkl --out RESULT.pkl --jobs 6
    q_loss_chain.py report RESULT.pkl [--null-a NAME --null-b NAME]

Each view of `q_teach.py fill` is one leaf's depth-1 matrix M over both sides' whole legal
pools. A Q builds a width-W menu as the search's q-nocover does (IKA-361's recall.py: solve Q
over the whole pools, rank our rows by their value against the column mix, theirs by minus
their value against the row mix, keep the top W, no cover). The menu's equilibrium is scored
in the whole M:

  loss   one side's loss = half the NashConv of the WxW equilibrium in M (what the side that
         plays it gives up against a best reply; a win-probability unit)
  kept   the weight of M's full equilibrium inside the menu (mean of both sides)
  inside 1 when the menu holds every action of M's full equilibrium support (weight > 1e-3)

References per view: `leaf` (rank by M's own full-game values: the upper bound a ranking can
reach, the positive control) and `random` (W at random; the mean of 5 draws). Between two Qs
the answer's movement is the total-variation distance of the two menu equilibria (embedded
in the whole pool), and the views are split by whether the two menus differ in the support
actions they hold (`supdiff`).
"""

# ruff: noqa: E501 - report lines print long tables
from __future__ import annotations

import argparse
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

WIDTHS = (12, 16, 24, 32, 64)
SUP = 1e-3
DRAWS = 5


def _ranks(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pokeuraou.equilibrium import solve

    e = solve(a)
    r = np.asarray(e.row_ev, dtype=np.float64)
    c = -np.asarray(e.col_ev, dtype=np.float64)
    return np.argsort(-r, kind="stable"), np.argsort(-c, kind="stable")


def cmd_orders(args: argparse.Namespace) -> None:
    import torch
    from q_train import Views, predict

    from pokeuraou.qhead import load_q

    torch.set_num_threads(4)
    views = Views(Path(args.shards), features=True, dedupe=False)
    index = np.arange(len(views))
    out: dict[str, list] = {}
    for q in args.q:
        preds = predict(load_q(q, "cpu"), views, index, torch.device("cpu"))
        out[Path(q).stem] = [_ranks(np.asarray(p, dtype=np.float64)) for p in preds]
        print(f"orders {Path(q).stem}: {len(preds)} views", flush=True)
    Path(args.out).write_bytes(pickle.dumps(out))


def _score_menu(m, xfull, yfull, rows, cols, v, solve_on=None):  # noqa: ANN001, ANN201
    """The menu's equilibrium (of `solve_on`, default `m`) scored in `m`'s whole game."""
    from pokeuraou.equilibrium import solve

    sub = solve((m if solve_on is None else solve_on)[np.ix_(rows, cols)])
    xs = np.zeros(m.shape[0])
    ys = np.zeros(m.shape[1])
    xs[rows] = sub.row_strategy
    ys[cols] = sub.col_strategy
    row_loss = v - float((xs @ m).min())
    col_loss = float((m @ ys).max()) - v
    sx = np.flatnonzero(xfull > SUP)
    sy = np.flatnonzero(yfull > SUP)
    inside = bool(set(sx) <= set(rows) and set(sy) <= set(cols))
    kept = (float(xfull[rows].sum()) + float(yfull[cols].sum())) / 2
    return {
        "loss": (row_loss + col_loss) / 2,
        "rowloss": row_loss,
        "colloss": col_loss,
        "kept": kept,
        "inside": float(inside),
        "x": xs,
        "y": ys,
        "supin": (frozenset(set(sx) & set(rows)), frozenset(set(sy) & set(cols))),
    }


def _one(task):  # noqa: ANN001, ANN202
    view, m, orders, seed = task
    from pokeuraou.equilibrium import solve

    m = np.asarray(m, dtype=np.float64)
    n0, n1 = m.shape
    full = solve(m)
    xfull, yfull = np.asarray(full.row_strategy), np.asarray(full.col_strategy)
    v = float(full.value)
    rng = np.random.default_rng(seed)
    rec: dict[str, float] = {"view": view, "n0": n0, "n1": n1, "support0": int((xfull > SUP).sum()),
                             "support1": int((yfull > SUP).sum())}
    leaf_orders = _ranks(m)
    allorders = {"leaf": leaf_orders, **orders}
    for w in WIDTHS:
        got = {}
        for name, (o0, o1) in allorders.items():
            got[name] = _score_menu(m, xfull, yfull, sorted(o0[:w]), sorted(o1[:w]), v)
            for k in ("loss", "rowloss", "colloss", "kept", "inside"):
                rec[f"{name}@{w}:{k}"] = got[name][k]
        acc = dict.fromkeys(("loss", "rowloss", "colloss", "kept", "inside"), 0.0)
        draws = []
        for _ in range(DRAWS):
            r = _score_menu(m, xfull, yfull, sorted(rng.permutation(n0)[:w]), sorted(rng.permutation(n1)[:w]), v)
            draws.append(r)
            for k in acc:
                acc[k] += r[k] / DRAWS
        for k, val in acc.items():
            rec[f"random@{w}:{k}"] = val
        # a second, independent random draw set: the null (two random rankings compared)
        rec[f"random2@{w}:loss"] = float(np.mean([
            _score_menu(m, xfull, yfull, sorted(rng.permutation(n0)[:w]), sorted(rng.permutation(n1)[:w]), v)["loss"]
            for _ in range(DRAWS)]))
        # Head to head on this one matrix: A's menu answer plays B's (each side in turn). A's
        # advantage over an even game = half of (what A's row mix gets against B's column mix
        # minus what B's row mix gets against A's column mix): the game-value form of one
        # decision of a match between the two agents.
        every = list(allorders)
        for i, a in enumerate(every):
            for b in every[i + 1:]:
                ga, gb = got[a], got[b]
                rec[f"{a}>{b}@{w}:adv"] = 0.5 * (float(ga["x"] @ m @ gb["y"]) - float(gb["x"] @ m @ ga["y"]))
        names = [n for n in allorders if n not in ("leaf",)]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ga, gb = got[a], got[b]
                rec[f"{a}|{b}@{w}:tv"] = 0.5 * (np.abs(ga["x"] - gb["x"]).sum() + np.abs(ga["y"] - gb["y"]).sum()) / 2
                rec[f"{a}|{b}@{w}:supdiff"] = float(ga["supin"] != gb["supin"])
                rec[f"{a}|{b}@{w}:menudiff"] = float(
                    (set(allorders[a][0][:w]) != set(allorders[b][0][:w])) or (set(allorders[a][1][:w]) != set(allorders[b][1][:w])))
    return rec


def cmd_score(args: argparse.Namespace) -> None:
    from q_train import Views

    views = Views(Path(args.shards), features=False, dedupe=False)
    orders = pickle.loads(Path(args.orders).read_bytes())
    tasks = []
    for v in range(len(views)):
        m = views.cells[v]
        if min(m.shape) <= 12:
            continue  # every width-12 menu is the whole pool: nothing to compare (IKA-361)
        tasks.append((v, m, {k: orders[k][v] for k in orders}, 401 * 100000 + v))
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} views of {len(views)}", flush=True)
    with Pool(args.jobs) as pool:
        rows = []
        for i, r in enumerate(pool.imap(_one, tasks, chunksize=4)):
            rows.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(tasks)}", flush=True)
    Path(args.out).write_bytes(pickle.dumps({"rows": rows, "widths": WIDTHS, "names": list(orders)}))


def _paired(rows, ka, kb):  # noqa: ANN001, ANN202
    d = np.array([r[kb] - r[ka] for r in rows])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, d


def cmd_report(args: argparse.Namespace) -> None:
    data = pickle.loads(Path(args.result).read_bytes())
    rows, names = data["rows"], data["names"]
    print(f"views {len(rows)}; one-side loss in win-probability units (x100 = points)")
    print("\n| width | " + " | ".join(["leaf", *names, "random"]) + " |")
    print("|---|" + "---|" * (len(names) + 2))
    for w in WIDTHS:
        cells = []
        for n in ["leaf", *names, "random"]:
            vals = np.array([r[f"{n}@{w}:loss"] for r in rows])
            cells.append(f"{vals.mean():.4f} ({vals.std(ddof=1) / np.sqrt(len(vals)):.4f})")
        print(f"| {w} | " + " | ".join(cells) + " |")
    print("\nkept weight of the full equilibrium / share of views holding the whole support")
    print("| width | " + " | ".join(["leaf", *names, "random"]) + " |")
    print("|---|" + "---|" * (len(names) + 2))
    for w in WIDTHS:
        cells = []
        for n in ["leaf", *names, "random"]:
            k = np.mean([r[f"{n}@{w}:kept"] for r in rows])
            i = np.mean([r[f"{n}@{w}:inside"] for r in rows])
            cells.append(f"{k:.3f} / {i:.3f}")
        print(f"| {w} | " + " | ".join(cells) + " |")
    print("\npaired loss difference (second - first; negative = second is better), mean [95% CI], share of views that differ")
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    pairs += [("random", "random2")]
    for a, b in pairs:
        for w in WIDTHS:
            m, se, d = _paired(rows, f"{a}@{w}:loss", f"{b}@{w}:loss")
            print(f"  {b} - {a} @{w}: {m:+.4f} [{m - 1.96 * se:+.4f}, {m + 1.96 * se:+.4f}]  differ {np.mean(np.abs(d) > 1e-9):.3f}")
    print("\nhead to head, one decision: A's advantage over an even game (win-probability units; + = first listed wins), mean [95% CI]")
    every = ["leaf", *names]
    for i, a in enumerate(every):
        for b in every[i + 1:]:
            for w in WIDTHS:
                d = np.array([r[f"{a}>{b}@{w}:adv"] for r in rows])
                se = d.std(ddof=1) / np.sqrt(len(d))
                print(f"  {a} vs {b} @{w}: {d.mean():+.4f} [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]")
    print("\nanswer movement between two Qs (TV distance of the menu equilibria, mean) and views whose support differs")
    for a, b in [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]:
        for w in WIDTHS:
            tv = np.mean([r[f"{a}|{b}@{w}:tv"] for r in rows])
            sd = np.mean([r[f"{a}|{b}@{w}:supdiff"] for r in rows])
            md = np.mean([r[f"{a}|{b}@{w}:menudiff"] for r in rows])
            print(f"  {a}|{b} @{w}: tv {tv:.4f}  supdiff {sd:.3f}  menudiff {md:.3f}")
    print("\nloss difference split by whether the menus hold different support actions")
    for a, b in [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]:
        for w in (12, 24):
            for flag in (0.0, 1.0):
                sub = [r for r in rows if r[f"{a}|{b}@{w}:supdiff"] == flag]
                if len(sub) > 2:
                    m, se, _ = _paired(sub, f"{a}@{w}:loss", f"{b}@{w}:loss")
                    print(f"  {b} - {a} @{w} supdiff={int(flag)}: n={len(sub)} {m:+.4f} (se {se:.4f})")


R24_WIDTHS = (12, 16, 24)


def cmd_r24(args: argparse.Namespace) -> None:
    """Depth 2 on a position set's stored reference (IKA-394's ref-r24: d1 and d2 of each cell
    of the rows x cols menus, which the q-mc0 q-nocover ranking built at the set's width).

    Each Q ranks the position's whole pools (as the search does); the ranking restricted to
    the stored rows / cols gives its width-W menu. Scored in the d2 game of the stored menus:
      d2 answer  the menu's equilibrium of the d2 sub-matrix (a depth-2 reading of the menu)
      d1 answer  the menu's equilibrium of the d1 sub-matrix (a depth-1 reading), same scoring
    """
    import json

    import torch
    from q_menus import q_matrix
    from q_train import load_q

    from pokeuraou import qhead
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.equilibrium import solve
    from pokeuraou.pool import load_pool
    from pokeuraou.position import Position

    torch.set_num_threads(2)
    reg = load_pool(args.pool).reg
    register_mega_stones(reg)
    encoder = Encoder(reg)
    nets = {Path(q).stem: load_q(q, "cpu") for q in args.q}
    data = json.loads((Path(args.set) / "positions.json").read_bytes())
    rng = np.random.default_rng(4012)
    rows_out = []
    for n in range(args.limit or len(data["positions"])):
        path = Path(args.set) / f"ref-{args.ref}" / f"{n}.npz"
        if not path.exists():
            continue
        ref = np.load(path)
        if ref["d2"].ndim != 2:
            continue  # a hidden position's stack of worlds: not this command
        d1, d2 = ref["d1"].astype(np.float64), ref["d2"].astype(np.float64)
        pos = Position.from_json(data["positions"][n]["position"])
        pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
        names = [[a.to_choice() for a in p] for p in pools]
        at = [{c: i for i, c in enumerate(names[s])} for s in (0, 1)]
        store = [[str(c) for c in ref["rows"]], [str(c) for c in ref["cols"]]]
        if any(c not in at[s] for s in (0, 1) for c in store[s]):
            continue
        m0, m1 = d2.shape
        pos_in_pool = [np.array([at[s][c] for c in store[s]]) for s in (0, 1)]
        full2 = solve(d2)
        x2, y2 = np.asarray(full2.row_strategy), np.asarray(full2.col_strategy)
        v2 = float(full2.value)
        rec: dict = {"n": n, "m0": m0, "m1": m1}
        orders = {"oracle": _ranks(d2)}
        ctx_orders = {}
        for name, net in nets.items():
            ctx = {"reg": reg, "encoder": encoder, "net": net, "device": torch.device("cpu")}
            q = q_matrix(ctx, pos, pools)
            r, c = _ranks(q)  # best first, over the whole pools
            # the stored rows / cols in that order (a position in the pool -> its place in the store)
            rank_r = {int(i): k for k, i in enumerate(r)}
            rank_c = {int(i): k for k, i in enumerate(c)}
            ctx_orders[name] = (np.argsort([rank_r[int(i)] for i in pos_in_pool[0]], kind="stable"),
                                np.argsort([rank_c[int(i)] for i in pos_in_pool[1]], kind="stable"))
        orders.update(ctx_orders)
        for w in R24_WIDTHS:
            if w >= min(m0, m1):
                continue
            mine: dict[str, dict] = {}
            for name, (o0, o1) in orders.items():
                rows, cols = sorted(o0[:w]), sorted(o1[:w])
                a2 = _score_menu(d2, x2, y2, rows, cols, v2)
                a1 = _score_menu(d2, x2, y2, rows, cols, v2, solve_on=d1)
                mine[name] = a2
                rec[f"{name}@{w}:loss2"] = a2["loss"]
                rec[f"{name}@{w}:loss1"] = a1["loss"]
                rec[f"{name}@{w}:kept"] = a2["kept"]
            for i, a in enumerate(mine):
                for b in list(mine)[i + 1:]:
                    rec[f"{a}>{b}@{w}:adv"] = 0.5 * (float(mine[a]["x"] @ d2 @ mine[b]["y"])
                                                     - float(mine[b]["x"] @ d2 @ mine[a]["y"]))
            for tag in ("random", "random2"):
                acc = np.zeros(3)
                for _ in range(DRAWS):
                    rows = sorted(rng.permutation(m0)[:w])
                    cols = sorted(rng.permutation(m1)[:w])
                    a2 = _score_menu(d2, x2, y2, rows, cols, v2)
                    a1 = _score_menu(d2, x2, y2, rows, cols, v2, solve_on=d1)
                    acc += np.array([a2["loss"], a1["loss"], a2["kept"]]) / DRAWS
                rec[f"{tag}@{w}:loss2"], rec[f"{tag}@{w}:loss1"], rec[f"{tag}@{w}:kept"] = acc
        rows_out.append(rec)
    Path(args.out).write_bytes(pickle.dumps({"rows": rows_out, "names": list(nets), "widths": R24_WIDTHS}))
    print(f"{len(rows_out)} positions", flush=True)


def cmd_r24_report(args: argparse.Namespace) -> None:
    data = pickle.loads(Path(args.result).read_bytes())
    rows, names = data["rows"], data["names"]
    cols = ["oracle", *names, "random"]
    for key, label in (("loss2", "depth-2 answer (menu's d2 equilibrium)"), ("loss1", "depth-1 answer (menu's d1 equilibrium)")):
        print(f"\n{label}, scored in the stored d2 game; one-side loss, mean (se); positions with width < menu")
        print("| width | n | " + " | ".join(cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for w in data["widths"]:
            sub = [r for r in rows if f"oracle@{w}:{key}" in r]
            cells = [f"{np.mean([r[f'{c}@{w}:{key}'] for r in sub]):.4f} ({np.std([r[f'{c}@{w}:{key}'] for r in sub], ddof=1) / np.sqrt(len(sub)):.4f})"
                     for c in cols]
            print(f"| {w} | {len(sub)} | " + " | ".join(cells) + " |")
        for a, b in [(a, b) for i, a in enumerate(names) for b in names[i + 1:]] + [("random", "random2")]:
            for w in data["widths"]:
                sub = [r for r in rows if f"oracle@{w}:{key}" in r]
                m, se, d = _paired(sub, f"{a}@{w}:{key}", f"{b}@{w}:{key}")
                print(f"  {b} - {a} @{w}: {m:+.4f} [{m - 1.96 * se:+.4f}, {m + 1.96 * se:+.4f}]  differ {np.mean(np.abs(d) > 1e-9):.3f}")


def cmd_r24_adv(args: argparse.Namespace) -> None:
    data = pickle.loads(Path(args.result).read_bytes())
    rows, names = data["rows"], data["names"]
    every = ["oracle", *names]
    print("head to head at depth 2 (menu d2 equilibria in the stored d2 game), advantage over an even game, mean [95% CI] (n)")
    for i, a in enumerate(every):
        for b in every[i + 1:]:
            for w in data["widths"]:
                d = np.array([r[f"{a}>{b}@{w}:adv"] for r in rows if f"{a}>{b}@{w}:adv" in r])
                se = d.std(ddof=1) / np.sqrt(len(d))
                print(f"  {a} vs {b} @{w}: {d.mean():+.4f} [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}] ({len(d)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("orders")
    o.add_argument("--shards", required=True)
    o.add_argument("--out", required=True)
    o.add_argument("--q", nargs="+", required=True)
    s = sub.add_parser("score")
    s.add_argument("--shards", required=True)
    s.add_argument("--orders", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--jobs", type=int, default=6)
    s.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("report")
    r.add_argument("result")
    t = sub.add_parser("r24")
    t.add_argument("--set", required=True)
    t.add_argument("--ref", default="r24")
    t.add_argument("--out", required=True)
    t.add_argument("--q", nargs="+", required=True)
    t.add_argument("--pool", default="regmc-matchupweb")
    t.add_argument("--limit", type=int, default=0)
    tr = sub.add_parser("r24-report")
    tr.add_argument("result")
    ta = sub.add_parser("r24-adv")
    ta.add_argument("result")
    args = ap.parse_args()
    {"orders": cmd_orders, "score": cmd_score, "report": cmd_report, "r24": cmd_r24,
     "r24-report": cmd_r24_report, "r24-adv": cmd_r24_adv}[args.cmd](args)


if __name__ == "__main__":
    main()
