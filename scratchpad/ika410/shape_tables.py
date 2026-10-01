"""IKA-410: how often the leaf (value-mc2) is far from the recorded search value of the same
move node (r = V_seat - leaf_seat), by the position's shape; and how r compares with the
replays' d_node (D - L1) on the same nodes.

    python shape_tables.py <leafscan.jsonl> <out.md> [<stage2 out.jsonl>...]
"""

import json
import sys
from pathlib import Path

import numpy as np

rows = [json.loads(x) for x in Path(sys.argv[1]).read_bytes().splitlines() if x.strip()]
r = np.array([x["r"] for x in rows])
out: list[str] = []
P = out.append


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def line(label: str, m: np.ndarray) -> str:
    d = r[m]
    if not len(d):
        return f"| {label} | 0 | | | | | | |"
    return (f"| {label} | {len(d):,} | {pct(len(d) / len(r))} | {d.mean():+.3f} | {np.abs(d).mean():.3f} | "
            f"{pct((np.abs(d) > 0.1).mean())} | {pct((np.abs(d) > 0.2).mean())} | "
            f"{pct((d < -0.2).mean())} / {pct((d > 0.2).mean())} |")


HEAD = ["| 札 | 決定×席 | 全体に占める | r の平均 | 平均 \\|r\\| | \\|r\\|>0.1 | \\|r\\|>0.2 | r<−0.2 / r>+0.2 |",
        "|---|---|---|---|---|---|---|---|"]
games = {(x["file"], x["line"]) for x in rows}
P(f"行動の手番 × 席 {len(rows):,}（局 {len(games):,}）。r = その手番の席の探索値 − 本当の局面の葉（value-mc2）。\n")
P("### 全体と分位\n")
qs = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
P("| " + " | ".join(str(q) for q in qs) + " | 標準偏差 |")
P("|" + "---|" * (len(qs) + 1))
P("| " + " | ".join(f"{np.quantile(r, q):+.3f}" for q in qs) + f" | {r.std():.3f} |")
P("")
P("### 札ごと\n")
out += HEAD
P(line("すべて", np.ones(len(r), bool)))
f = lambda fn: np.array([fn(x) for x in rows])  # noqa: E731
turn = f(lambda x: x["turn"])
for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 10), (11, 99)):
    P(line(f"ターン {lo}" if lo == hi else f"ターン {lo}〜{hi}", (turn >= lo) & (turn <= hi)))
oa, fa = f(lambda x: x["own"]["alive"]), f(lambda x: x["foe"]["alive"])
tot = oa + fa
for t in range(8, 1, -1):
    P(line(f"残りの体の合計 {t}", tot == t))
for a, b in ((2, 2), (2, 1), (1, 2), (1, 1), (3, 3), (4, 4)):
    P(line(f"残り {a} 対 {b}（自分 対 相手）", (oa == a) & (fa == b)))
fs = f(lambda x: x["foe_shown"])
P(line("相手の 4 体が全部見えている", fs == 4))
P(line("相手の裏がまだ見えていない体がある", fs < 4))
ml_own, ml_foe = f(lambda x: x["own"]["mega_left"]), f(lambda x: x["foe"]["mega_left"])
P(line("どちらかがメガシンカをまだ残している", ml_own | ml_foe))
P(line("どちらもメガシンカを使った・持たない", ~(ml_own | ml_foe)))
tr = f(lambda x: x["trickroom"])
P(line("トリックルーム", tr))
weather = f(lambda x: x["weather"] or "")
for w in sorted(set(weather)):
    if w:
        P(line(f"天気 {w}", weather == w))
terrain = f(lambda x: x["terrain"] or "")
P(line("フィールドあり", terrain != ""))
tw = f(lambda x: ("tailwind" in x["own"]["conds"]) or ("tailwind" in x["foe"]["conds"]))
P(line("おいかぜ（どちらか）", tw))
scr = f(lambda x: any(c in x["own"]["conds"] + x["foe"]["conds"] for c in ("reflect", "lightscreen", "auroraveil")))
P(line("壁（どちらか）", scr))
st = f(lambda x: x["own"]["status"] + x["foe"]["status"])
P(line("場の体に状態異常", st > 0))
bo = f(lambda x: x["own"]["boosted"] + x["foe"]["boosted"])
P(line("場の体に能力上昇", bo > 0))
leaf = f(lambda x: x["leaf"])
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
    P(line(f"葉 {lo:.1f}〜{min(hi, 1):.1f}", (leaf >= lo) & (leaf < hi)))
P("")
# calibration of leaf vs search vs outcome where both disagree
oc = np.array([np.nan if x["outcome"] is None else x["outcome"] for x in rows])
v = f(lambda x: x["v"])
P("### 葉と探索値がずれた手番の、局の結果\n")
P("| 札 | 決定×席 | 葉の平均 | 探索値の平均 | 勝った割合 |")
P("|---|---|---|---|---|")
for label, m in (("r < −0.2（探索が葉より低い）", r < -0.2), ("r > +0.2", r > 0.2),
                 ("\\|r\\| ≤ 0.05", np.abs(r) <= 0.05)):
    m = m & ~np.isnan(oc)
    P(f"| {label} | {m.sum():,} | {leaf[m].mean():.3f} | {v[m].mean():.3f} | {oc[m].mean():.3f} |")
P("")
if len(sys.argv) > 3:
    s2 = []
    for p in sys.argv[3:]:
        s2 += [json.loads(x) for x in Path(p).read_bytes().splitlines() if x.strip()]
    s2 = [x for x in s2 if x.get("matched") and "error" not in x]
    key = {(x["file"], x["line"], x["d"], x["seat"]): x["r"] for x in rows}
    pairs = [(x["D"] - x["L1"], key[(x["file"], x["line"], x["ex"]["d1"], x["ex"]["seat"])]) for x in s2
             if (x["file"], x["line"], x["ex"]["d1"], x["ex"]["seat"]) in key]
    a = np.array(pairs)
    P(f"打ち直しの d_node（真の裏での探索値 − 葉）と、記録だけから出した r（信念での探索値 − 葉）: {len(a)} 組、"
      f"相関 {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.4f}、差の最大 {np.abs(a[:, 0] - a[:, 1]).max():.3f}、"
      f"差の平均 \\|・\\| {np.abs(a[:, 0] - a[:, 1]).mean():.4f}。\n")
Path(sys.argv[2]).write_bytes(("\n".join(out) + "\n").encode())
print("\n".join(out))
