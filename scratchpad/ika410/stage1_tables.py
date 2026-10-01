"""IKA-410 stage 1 tables (markdown) from stage1.jsonl.

    python stage1_tables.py <stage1.jsonl> <out.md>
"""

import json
import sys
from pathlib import Path

import numpy as np

rows = [json.loads(x) for x in Path(sys.argv[1]).read_bytes().splitlines() if x.strip()]
dv = np.array([r["dv"] for r in rows])
T = (0.1, 0.2, 0.3, 0.4)
out: list[str] = []


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def line(label: str, mask: np.ndarray) -> str:
    d = dv[mask]
    n = len(d)
    cells = [label, f"{n:,}", pct(n / len(dv)), f"{d.mean():+.3f}" if n else "-"]
    for t in T:
        cells.append(f"{pct((np.abs(d) > t).mean())} ({pct((d < -t).mean())} / {pct((d > t).mean())})" if n else "-")
    return "| " + " | ".join(cells) + " |"


head = "| 札 | 組の数 | 全体に占める | ΔV の平均 | " + " | ".join(f"\\|ΔV\\|>{t}（下 / 上）" for t in T) + " |"
sep = "|" + "---|" * (4 + len(T))
games = {(r["file"], r["line"]) for r in rows}
out.append(f"組（同じ席の続く 2 つの行動の手番）{len(rows):,}、局 {len(games):,}。\n")
out.append("### 決定あたり\n")
out += [head, sep]
out.append(line("すべて", np.ones(len(rows), bool)))
f = lambda key: np.array([r[key] for r in rows])  # noqa: E731
own_f, foe_f, newly = f("own_faint"), f("foe_faint"), f("foe_newly_shown")
open0 = f("open0")
out.append(line("自分の体が倒れた", own_f > 0))
out.append(line("相手の体が倒れた", foe_f > 0))
out.append(line("どちらも倒れない", (own_f == 0) & (foe_f == 0)))
out.append(line("相手の裏が新しく見えた", newly > 0))
out.append(line("相手の裏は新しく見えない", newly == 0))
out.append(line("倒れない・裏も見えない", (own_f == 0) & (foe_f == 0) & (newly == 0)))
out.append(line("d0 で相手の 4 体が全部見えている", f("foe_shown0") == 4))
out.append("")
out.append("### ターン（d0 のターン）\n")
out += [head, sep]
turn = f("turn0")
for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 99)):
    out.append(line(f"T{lo}" if lo == hi else f"T{lo}〜{hi}", (turn >= lo) & (turn <= hi)))
out.append("")
out.append("### 残りの体（d0、自分 対 相手）\n")
out += [head, sep]
oa, fa = f("own_alive0"), f("foe_alive0")
for a in (4, 3, 2, 1):
    for b in (4, 3, 2, 1):
        m = (oa == a) & (fa == b)
        if m.sum() >= 50:
            out.append(line(f"{a} 対 {b}", m))
out.append("")
out.append("### d0 の値の帯\n")
out += [head, sep]
v0 = f("v0")
for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)):
    out.append(line(f"{lo:.1f}〜{min(hi, 1):.1f}", (v0 >= lo) & (v0 < hi)))
out.append("")
# per game
by_game: dict = {}
for r in rows:
    by_game.setdefault((r["file"], r["line"]), []).append(r["dv"])
out.append("### 局あたり（どちらかの席に 1 度でもあった局の割合）\n")
out.append("| 閾値 | \\|ΔV\\| | 下がった | 上がった | 1 局あたりの回数（\\|ΔV\\|） |")
out.append("|---|---|---|---|---|")
for t in T:
    g = [np.array(v) for v in by_game.values()]
    out.append(f"| {t} | {pct(np.mean([(np.abs(x) > t).any() for x in g]))} | "
               f"{pct(np.mean([(x < -t).any() for x in g]))} | {pct(np.mean([(x > t).any() for x in g]))} | "
               f"{np.mean([(np.abs(x) > t).sum() for x in g]):.2f} |")
out.append("")
out.append("### 分位（ΔV）\n")
qs = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
out.append("| " + " | ".join(f"{q}" for q in qs) + " | 標準偏差 |")
out.append("|" + "---|" * (len(qs) + 1))
out.append("| " + " | ".join(f"{np.quantile(dv, q):+.3f}" for q in qs) + f" | {dv.std():.3f} |")
Path(sys.argv[2]).write_bytes(("\n".join(out) + "\n").encode())
print("\n".join(out))
