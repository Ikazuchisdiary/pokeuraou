"""IKA-434: the record's tables from analysis.json, rust.bin and sample.npz (markdown on stdout)."""
import json
from pathlib import Path

import numpy as np

HERE = Path("C:/tmp/ika434")
r = json.loads((HERE / "analysis.json").read_text())
s = np.load(HERE / "sample.npz")
names = json.loads((HERE / "rust.json").read_text())["names"]
n = len(s["game"])
rust = np.fromfile(HERE / "rust.bin", dtype="<f4").reshape(n, 2, len(names))
py_names = list(s["py_names"])
p = np.load(HERE / "predict.npz")
side_names = list(p["side_names"])


def either(name: str) -> float:
    if name in names:
        v = rust[:, :, names.index(name)]
        if name == "absorb_best":
            return float((v >= 0).all(axis=1).mean())
        return float((v != 0).any(axis=1).mean())
    if name in py_names:
        return float((s["py"][:, :, py_names.index(name)] != 0).any(axis=1).mean())
    if name == "mega_available":
        return float((p["side"][:, :, side_names.index("mega_available")] != 0).any(axis=1).mean())
    return float("nan")


fs = json.loads((HERE / "foldslopes.json").read_text())
print("| 特徴 | どちらかの側にある | 差が 0 でない | 結果 − 予測の最大（bindaux） | 補正の利得 bindaux 1 次（se） | 区分ごと（se） | 手番の帯ごと（se） | 傾き 半分ずつ bindaux | st 1 次（se） | 傾き 半分ずつ st |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, f in r["features"].items():
    b, st = f["gain"]["bindaux"], f["gain"]["st"]
    big = [c for c in f["bins"] if c["share"] >= 0.01]
    worst = max(big, key=lambda c: abs(c["bindaux"]["res"]))
    sb = "・".join(f"{v:+.2f}" for v in fs[name]["bindaux"])
    ss = "・".join(f"{v:+.2f}" for v in fs[name]["st"])
    print(f"| {name} | {either(name):.1%} | {f['nonzero_diff']:.1%} | {worst['bindaux']['res']:+.2f}（{worst['bindaux']['se']:.2f}、値 {worst['bin']:+g}） | "
          f"{b['linear'] * 1e4:+.2f}（{b['linear_se'] * 1e4:.2f}） | {b['onehot'] * 1e4:+.2f}（{b['onehot_se'] * 1e4:.2f}） | "
          f"{b['banded'] * 1e4:+.2f}（{b['banded_se'] * 1e4:.2f}） | {sb} | {st['linear'] * 1e4:+.2f}（{st['linear_se'] * 1e4:.2f}） | {ss} |")
print()
for name in ["shard_bind_sure", "absorb_best", "good_switch", "two_hit_sure", "focus_ko_sure", "bench_bind_sure",
             "benchatk_ko_sure", "frac_noko", "species_shown", "weather_turns"]:
    f = r["features"][name]
    print(f"#### {name}")
    print("| 値 | 行 | 勝率 | bindaux 結果 − 予測（se） | st 結果 − 予測（se） |")
    print("|---|---|---|---|---|")
    for c in f["bins"]:
        if c["share"] < 0.005:
            continue
        print(f"| {c['bin']:+g} | {c['rows']:,} | {c['win']:.1%} | {c['bindaux']['res']:+.2f}（{c['bindaux']['se']:.2f}） | {c['st']['res']:+.2f}（{c['st']['se']:.2f}） |")
    print()
