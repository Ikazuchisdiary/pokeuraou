"""IKA-410 stage 2 tables from stage2 output rows.

    python stage2_tables.py <out.md> <stage2 out.jsonl>...
"""

import json
import sys
from pathlib import Path

import numpy as np

rows = []
for p in sys.argv[2:]:
    rows += [json.loads(x) for x in Path(p).read_bytes().splitlines() if x.strip()]
out: list[str] = []
P = out.append


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def terms(r: dict) -> dict:
    t = {
        "a": r["A"] - r["V0"], "a2": r["VT"] - r["A"], "b_foe": r["B1"] - r["VT"],
        "b_own": r["B2"] - r["B1"], "c_budget": r["CE"] - r["B2"], "e": r["V1"] - r["D"],
        "d_node": r["D"] - r["L1"], "cd": r["D"] - r["CE"],
        "total": r["V1"] - r["V0"],
    }
    if "D3" in r:
        t["d_node3"] = r["D3"] - r["L1_3"]
    if "C" in r:
        t["c"] = r["C"] - r["CE"]
        t["d"] = r["D"] - r["C"]
        if "C3" in r:
            t["d3"] = r["D3"] - r["C3"]
    return t


NAMES = [
    ("total", "合計 V1 − V0"),
    ("a", "(a) 裏が見えた（d1 までに見えた分で信念を条件づけ）"),
    ("a2", "(a') まだ見えていない裏の真の値 − 条件づけた信念"),
    ("b_foe", "(b') 相手の実際の手 − 自分が読んだ相手の混合"),
    ("b_own", "(b) 自分の混合から引いた手"),
    ("c_budget", "(c0) 行列の予算 → 厳密な升目"),
    ("c", "(c) 乱数（厳密な升目の期待 → 引いた枝の葉）"),
    ("d", "(d) 反省（引いた枝の葉 → d1 の探索値、真の裏）"),
    ("e", "(e) d1 の真の裏の値 → d1 の信念での値"),
    ("cd", "(c)+(d)（枝が見つからない例も含む）"),
    ("d_node", "(d*) d1 の局面の葉 → d1 の探索値（value-mc2）"),
    ("d_node3", "(d*) 同じ、value-mc3 で解き直し"),
    ("d3", "(d) 同じ、value-mc3（葉も探索も mc3）"),
]

groups = ("drop", "random")
P("## 打ち直しの一致\n")
P("| 組 | 例 | 打ち直しが d1 まで記録と一致 | 局の最後まで一致 | 分析の失敗 |")
P("|---|---|---|---|---|")
for g in groups:
    rs = [r for r in rows if r["ex"]["group"] == g]
    P(f"| {g} | {len(rs)} | {sum(bool(r.get('matched')) for r in rs)} | "
      f"{sum(r.get('first_diff', 0) is None for r in rs)} | {sum('error' in r for r in rs)} |")
good = [r for r in rows if r.get("matched") and "error" not in r]
P("")
P("## 対照\n")
chk = {
    "|Σ w v − 記録の d0 の値|": [abs(r["V0"] - r["V0_rec"]) for r in good],
    "|Σ w' v − 記録の d1 の値|": [abs(r["V1"] - r["V1_rec"]) for r in good],
    "d0 の信念を条件づけたもの − d1 の信念（最大差）": [r["belief_check"] for r in good],
    "d1 を value-mc2 で解き直した真の裏の値 − 取り出した値": [abs(r["D_resolve2"] - r["D"]) for r in good],
}
P("| 対照 | 例 | 最大 | 1e-4 を超えた数 |")
P("|---|---|---|---|")
for k, v in chk.items():
    v = np.array(v)
    P(f"| {k} | {len(v)} | {v.max():.2e} | {(v > 1e-4).sum()} |")
P(f"| d1 の解き直しで候補集合が取り出したものと同じ | {len(good)} | | 違う {sum(not r['menu_same2'] for r in good)} |")
P(f"| 引いた結果が厳密な枝の中に見つかった | {len(good)} | | 見つからない {sum(not r['drawn_found'] for r in good)} |")
P("")
for g in groups:
    rs = [r for r in good if r["ex"]["group"] == g]
    T = [terms(r) for r in rs]
    P(f"## {g}（{len(rs)} 例）\n")
    P("| 項 | 例 | 平均 | 中央 | 10% | 90% | 平均 \\|・\\| | \\|・\\|>0.1 | \\|・\\|>0.2 |")
    P("|---|---|---|---|---|---|---|---|---|")
    for key, name in NAMES:
        v = np.array([t[key] for t in T if key in t])
        if not len(v):
            continue
        P(f"| {name} | {len(v)} | {v.mean():+.3f} | {np.median(v):+.3f} | {np.quantile(v, 0.1):+.3f} | "
          f"{np.quantile(v, 0.9):+.3f} | {np.abs(v).mean():.3f} | {pct((np.abs(v) > 0.1).mean())} | {pct((np.abs(v) > 0.2).mean())} |")
    P("")
    # frequency with a binomial interval
    for key in ("d", "d_node"):
        v = np.array([t[key] for t in T if key in t])
        for th in (0.1, 0.2):
            k = int((np.abs(v) > th).sum())
            n = len(v)
            p = k / n
            se = np.sqrt(p * (1 - p) / n)
            P(f"- {key}: |・|>{th} は {k}/{n} = {pct(p)}（±{pct(1.96 * se)}）; "
              f"下向き（< −{th}）{pct((v < -th).mean())}、上向き {pct((v > th).mean())}")
    P("")
Path(sys.argv[1]).write_bytes(("\n".join(out) + "\n").encode())
print("\n".join(out))
