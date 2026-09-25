"""Width-12 menus from a learned Q, on IKA-310/311's yardstick (IKA-274).

The yardstick is IKA-281's width-24 file (235 M-C positions solved open: the width-24
menus of the shipped leaf ranking, refs2 order, and their depth-1 matrix d1). As in
IKA-310/311 every width-12 menu is chosen INSIDE the width-24 menu, because that is where
the true matrix exists, and is measured in the width-24 d1 game: the weight of the
width-24 equilibrium left outside it, and the NashConv of the 12x12 equilibrium.

Menus compared (both sides built the same way):

  refs2        the recorded order: greedy cover, then the ranking's order (today's menu)
  randfill     refs2's cover, the rest at random (IKA-310's null; mean of `--draws`)
  swap1        refs2 then one best-response swap per side (IKA-310 (c))
  coverq       solve cover x cover on d1, rank against its mix (IKA-311 form 3)
  oracle       rank against the width-24 d1 equilibrium's own mix (positive control)
  q-full       solve Q over both sides' whole legal pools, rank against the foe's mix
  q-w24        solve Q over the width-24 menus, rank against the foe's mix
  q-cover      solve Q over refs2's covers, rank against the foe's mix (coverq on Q)

each with the cover kept (`/cover`) and without (`/nocover`). Q is scored once per
position on the true position (the yardstick was solved open).

    python tools/q_menus.py --w24 <ika281>/w24.jsonl --games-dir data/selfplay-mc0 \\
        --model <dir>/q.pt --out <dir>/yardstick.json
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

SUP = 1e-6
TOL = 1e-6
WIDTH = 12


def keys_of(choices: list[str]) -> list[list[str]]:
    return [[f"{k}:{t.strip()}" for k, t in enumerate(c.split(","))] for c in choices]


def menu_ordered(choices, order, limit=WIDTH, cover=True, fixed_cover=None):  # noqa: ANN001, ANN201
    """`narrow`'s rule on `choices` ranked by `order` (best first): (kept, cover)."""
    n = len(choices)
    if n <= limit:
        return list(range(n)), set()
    if not cover:
        return sorted(order[:limit]), set()
    keys = keys_of(choices)
    nslots = len(keys[0])
    kept: list[int] = []
    taken: set[int] = set()
    if fixed_cover is not None:
        kept = list(fixed_cover)[:limit]
        taken = set(kept)
    else:
        uncovered = {k for ks in keys for k in ks}
        while uncovered and len(kept) < limit:
            best, best_gain = -1, 0
            for i in order:
                if i in taken:
                    continue
                gain = sum(1 for k in keys[i] if k in uncovered)
                if gain > best_gain:
                    best, best_gain = i, gain
                if best_gain == nslots:
                    break
            if best < 0:
                break
            kept.append(best)
            taken.add(best)
            uncovered -= set(keys[best])
    cov = set(kept)
    for i in order:
        if len(kept) >= limit:
            break
        if i not in taken:
            kept.append(i)
            taken.add(i)
    return sorted(kept), cov


def order_of(scores, choices):  # noqa: ANN001, ANN201
    return sorted(range(len(choices)), key=lambda i: (-scores[i], choices[i]))


def eq(a: np.ndarray, rows, cols):  # noqa: ANN001, ANN201
    from pokeuraou.equilibrium import solve

    e = solve(a[np.ix_(list(rows), list(cols))])
    x = np.zeros(a.shape[0])
    y = np.zeros(a.shape[1])
    x[list(rows)] = e.row_strategy
    y[list(cols)] = e.col_strategy
    return x, y, float(e.value)


def do_swap(a, rows, cols, k=1):  # noqa: ANN001, ANN201
    """IKA-310 (c): one round, k best-response swaps per side."""
    rows, cols = list(rows), list(cols)
    m, n = a.shape
    x, y, v = eq(a, rows, cols)
    out_r = [i for i in range(m) if i not in rows]
    out_c = [j for j in range(n) if j not in cols]
    g_r = sorted(((float(a[i] @ y) - v, i) for i in out_r), reverse=True)
    g_c = sorted(((v - float(x @ a[:, j]), j) for j in out_c), reverse=True)
    add_r = [i for g, i in g_r[:k] if g > TOL]
    add_c = [j for g, j in g_c[:k] if g > TOL]
    drop_r = sorted([i for i in rows if x[i] <= SUP], reverse=True)[: len(add_r)]
    drop_c = sorted([j for j in cols if y[j] <= SUP], reverse=True)[: len(add_c)]
    add_r, add_c = add_r[: len(drop_r)], add_c[: len(drop_c)]
    return sorted(set(rows) - set(drop_r) | set(add_r)), sorted(set(cols) - set(drop_c) | set(add_c))


def auc(scores, labels):  # noqa: ANN001, ANN201
    pos = [s for s, lab in zip(scores, labels, strict=True) if lab]
    neg = [s for s, lab in zip(scores, labels, strict=True) if not lab]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def q_matrix(ctx: dict[str, Any], pos: Any, pools: tuple[list, list]) -> np.ndarray:  # noqa: ANN401
    import torch

    from pokeuraou import qhead

    encoder, net, device = ctx["encoder"], ctx["net"], ctx["device"]
    enc = encoder.encode_positions([pos])
    batch = {
        name: torch.from_numpy(np.ascontiguousarray(getattr(enc, name))).to(device)
        for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    }
    a0 = qhead.encode_actions(encoder.vocab, pos, 0, pools[0]).astype(np.int64)
    a1 = qhead.encode_actions(encoder.vocab, pos, 1, pools[1]).astype(np.int64)
    feats: list[Any] = [None, None]
    if net.config.properties:
        got = qhead.port_features(ctx["reg"], pos, pools)
        if got is None:
            got = (
                np.zeros((len(pools[0]), qhead.FEATURE_WIDTH), np.float32),
                np.zeros((len(pools[1]), qhead.FEATURE_WIDTH), np.float32),
            )
        feats = [torch.from_numpy(f)[None].to(device) for f in got]
    with torch.no_grad():
        logits = net(
            batch,
            torch.from_numpy(a0)[None].to(device),
            torch.from_numpy(a1)[None].to(device),
            feats[0],
            feats[1],
        )
    return torch.sigmoid(logits[0]).double().cpu().numpy()


def one_position(
    ctx: dict[str, Any], row: dict[str, Any], rng: np.random.Generator, draws: int
) -> dict[str, Any]:  # noqa: ANN401, C901, PLR0915
    from pokeuraou import qhead
    from pokeuraou.position import Position

    name, line_no, k = row["ref"]
    with (ctx["games_dir"] / name).open("rb") as handle:
        for n, line in enumerate(handle):
            if n == line_no:
                game = json.loads(line)
                break
    pos = Position.from_json(game["decisions"][k]["position"])
    reg = ctx["reg"]
    pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
    ch = [row["rows"], row["cols"]]
    at = [[a.to_choice() for a in pools[s]].index for s in (0, 1)]
    menu_ix = [[at[s](c) for c in ch[s]] for s in (0, 1)]
    a = np.asarray(row["d1"], dtype=np.float64)
    m, n = a.shape
    qfull = q_matrix(ctx, pos, pools)
    qw = qfull[np.ix_(menu_ix[0], menu_ix[1])]
    x24, y24, v24 = eq(a, range(m), range(n))
    full = [x24, y24]
    # What each side actually played in the recorded game (its own width-12 menu, the
    # generating leaf, its hidden view), carried onto the width-24 menus by choice string
    # and renormalised over what lands there (IKA-312: value against the real opponent).
    decision = game["decisions"][k]
    played = []
    for key_a, key_p, size, menu in (
        ("ownActions", "ownPolicy", m, ch[0]),
        ("foeActions", "foePolicy", n, ch[1]),
    ):
        w = np.zeros(size)
        for choice, p in zip(decision[key_a], decision[key_p], strict=True):
            if choice in menu:
                w[menu.index(choice)] += p
        played.append(w)
    coverage = [float(played[0].sum()), float(played[1].sum())]
    x_act = played[0] / played[0].sum() if coverage[0] > 0 else full[0]
    y_act = played[1] / played[1].sum() if coverage[1] > 0 else full[1]

    def measure(rows, cols):  # noqa: ANN001, ANN202
        xa, ya, _ = eq(a, rows, cols)
        l0 = v24 - float((xa @ a).min())
        l1 = float((a @ ya).max()) - v24
        return {
            "nc": l0 + l1,
            # both sides' value against the other side's recorded play, summed
            "act": float(xa @ a @ y_act) + 1.0 - float(x_act @ a @ ya),
            "out": [
                float(sum(x24[i] for i in range(m) if i not in rows)),
                float(sum(y24[j] for j in range(n) if j not in cols)),
            ],
        }

    def scored(xmix_over_rows: np.ndarray, ymix_over_cols: np.ndarray, payoff: np.ndarray):  # noqa: ANN202
        """Scores of each width-24 action against the foe's mix, from `payoff` (m x n)."""
        return [payoff @ ymix_over_cols, -(xmix_over_rows @ payoff)]

    res: dict[str, dict[str, Any]] = {}

    def menus_from(form: str, scores, fixed=None):  # noqa: ANN001, ANN202
        orders = [order_of(scores[0], ch[0]), order_of(scores[1], ch[1])]
        for variant in ("cover", "nocover"):
            if fixed is not None and variant == "cover":
                r = menu_ordered(ch[0], orders[0], fixed_cover=fixed[0])[0]
                c = menu_ordered(ch[1], orders[1], fixed_cover=fixed[1])[0]
            else:
                r = menu_ordered(ch[0], orders[0], cover=variant == "cover")[0]
                c = menu_ordered(ch[1], orders[1], cover=variant == "cover")[0]
            got = measure(r, c)
            got["auc"] = [auc(list(scores[s]), [w > SUP for w in full[s]]) for s in (0, 1)]
            res[f"{form}/{variant}"] = got

    # today's menu, and its cover
    menus_from("refs2", [-np.arange(m, dtype=float), -np.arange(n, dtype=float)])
    cov = [menu_ordered(ch[s], list(range(len(ch[s]))))[1] for s in (0, 1)]
    covs = [sorted(cov[0]) or list(range(m)), sorted(cov[1]) or list(range(n))]
    r12 = menu_ordered(ch[0], list(range(m)))[0]
    c12 = menu_ordered(ch[1], list(range(n)))[0]
    rs, cs = do_swap(a, r12, c12, 1)
    res["swap1/cover"] = measure(rs, cs)
    # the null: refs2's cover, the rest at random
    ms = []
    for _ in range(draws):
        pick = []
        for s, size in ((0, m), (1, n)):
            if size <= WIDTH:
                pick.append(list(range(size)))
                continue
            rest = [i for i in range(size) if i not in cov[s]]
            fill = [int(i) for i in rng.choice(rest, size=WIDTH - len(cov[s]), replace=False)]
            pick.append(sorted(set(cov[s]) | set(fill)))
        ms.append(measure(pick[0], pick[1]))
    res["randfill/cover"] = {
        "nc": float(np.mean([x["nc"] for x in ms])),
        "act": float(np.mean([x["act"] for x in ms])),
        "out": [float(np.mean([x["out"][s] for x in ms])) for s in (0, 1)],
    }
    # coverq on the true d1 (IKA-311 form 3), and the oracle mix
    xc, yc, _ = eq(a, covs[0], covs[1])
    menus_from("coverq", scored(xc, yc, a), fixed=covs)
    menus_from("oracle", scored(x24, y24, a))
    # ranked against what the other side actually played (an upper bound for prediction)
    menus_from("played", scored(x_act, y_act, a))
    # Q forms
    from pokeuraou.equilibrium import solve

    e = solve(qfull)
    xq = np.asarray(e.row_strategy)[menu_ix[0]]
    yq = np.asarray(e.col_strategy)[menu_ix[1]]
    # score the width-24 actions against the whole-pool mix (the mix may sit outside the
    # width-24 menus, so read the full rows of Q)
    s_full = [
        (qfull @ np.asarray(e.col_strategy))[menu_ix[0]],
        -(np.asarray(e.row_strategy) @ qfull)[menu_ix[1]],
    ]
    menus_from("q-full", s_full)
    xw, yw, _ = eq(qw, range(m), range(n))
    menus_from("q-w24", scored(xw, yw, qw))
    xqc, yqc, _ = eq(qw, covs[0], covs[1])
    menus_from("q-cover", scored(xqc, yqc, qw), fixed=covs)
    # The opponent as expected rather than as the worst case: it plays the equilibrium of
    # today's 12 x 12 menus (as the opponent of IKA-312's S4 did), read off Q.
    xo, yo, _ = eq(qw, r12, c12)
    menus_from("q-opp12", scored(xo, yo, qw))
    return {
        "index": row["index"],
        "m": m,
        "n": n,
        "pools": [len(pools[0]), len(pools[1])],
        "q_mae_w24": float(np.abs(qw - a).mean()),
        "q_value_err_w24": abs(float(solve(qw).value) - v24),
        "q_mass_on_w24": [float(np.sum(xq)), float(np.sum(yq))],
        "played_coverage": coverage,
        "res": res,
    }


def summarise(per: list[dict[str, Any]], boots: int, seed: int) -> str:
    forms = list(per[0]["res"].keys())
    base = np.array([p["res"]["refs2/cover"]["nc"] for p in per])
    base_act = np.array([p["res"]["refs2/cover"]["act"] for p in per])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(boots, len(per)))
    lines = [
        f"positions {len(per)};  Q on the width-24 cells: MAE {np.mean([p['q_mae_w24'] for p in per]):.4f}, "
        f"|v(Q) - v(d1)| {np.mean([p['q_value_err_w24'] for p in per]):.4f};  "
        f"Q's whole-pool mix on the width-24 menus {np.mean([p['q_mass_on_w24'][0] for p in per]):.3f} / "
        f"{np.mean([p['q_mass_on_w24'][1] for p in per]):.3f};  recorded play on the width-24 menus "
        f"{np.mean([p['played_coverage'][0] for p in per]):.3f} / "
        f"{np.mean([p['played_coverage'][1] for p in per]):.3f}",
        "",
        f"  {'menu':<22}{'NashConv':>10}  {'- refs2 [95%]':<30}{'out 0 / 1':>16}  {'AUC 0 / 1':>14}"
        f"  {'vs played - refs2 [95%] (both sides, win prob.)':<40}",
    ]
    for form in forms:
        nc = np.array([p["res"][form]["nc"] for p in per])
        d = nc - base
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        g = np.array([p["res"][form]["act"] for p in per]) - base_act
        glo, ghi = np.percentile(g[idx].mean(1), [2.5, 97.5])
        out = np.array([p["res"][form]["out"] for p in per]).mean(0)
        aucs = [p["res"][form].get("auc") for p in per]
        if aucs[0] is not None:
            a0 = np.mean([x[0] for x in aucs if x and x[0] is not None])
            a1 = np.mean([x[1] for x in aucs if x and x[1] is not None])
            auc_txt = f"{a0:.3f} / {a1:.3f}"
        else:
            auc_txt = "-"
        lines.append(
            f"  {form:<22}{nc.mean():>10.4f}  {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]{'':<5}"
            f"{out[0]:>7.3f} / {out[1]:.3f}  {auc_txt:>14}"
            f"  {g.mean():+.4f} [{glo:+.4f}, {ghi:+.4f}]"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w24", type=Path, required=True)
    ap.add_argument("--games-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--pool", default="regmc-matchupweb")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--boots", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    from q_train import load_q

    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.pool import load_pool

    torch.set_num_threads(1)
    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    ctx = {
        "reg": reg,
        "encoder": Encoder(reg),
        "net": load_q(args.model, args.device),
        "device": torch.device(args.device),
        "games_dir": args.games_dir,
    }
    rng = np.random.default_rng(311)
    per = []
    with args.w24.open("rb") as handle:
        for line in handle:
            row = json.loads(line)
            if "skipped" in row:
                continue
            per.append(one_position(ctx, row, rng, args.draws))
            if args.limit and len(per) >= args.limit:
                break
    args.out.write_bytes(json.dumps(per).encode("utf-8"))
    text = summarise(per, args.boots, 274)
    args.out.with_suffix(".txt").write_bytes((text + "\n").encode("utf-8"))
    print(text)


if __name__ == "__main__":
    main()
