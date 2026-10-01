"""IKA-412: held-out loss of value models under the four left/right arrangements.

    swap_effect.py <worktree> <out.json> name=seed0.pt,seed1.pt [name=...]

Holdout = the value-mc4 split (split_seed 0, holdout 0.15) of data/selfplay-mc01234-encoded.npz
(main checkout data). For each model (the mean of its seeds' logits, as value-mc4 x2) and each
arrangement (as recorded; side 0 exchanged; side 1 exchanged; both) the log loss per generation
group and the mean/percentiles of |p(arrangement) - p(as recorded)|, and the log loss of the
mean over the four arrangements (a test-time augmentation read).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

wt = Path(sys.argv[1])
sys.path.insert(0, str(wt / "src"))
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.slotswap import SwapSlots, position_reader_rows, swap_batch  # noqa: E402
from pokeuraou.value import ValueConfig, load_dataset, load_model, split_for  # noqa: E402

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
out_path = Path(sys.argv[2])
models = {a.split("=")[0]: a.split("=")[1].split(",") for a in sys.argv[3:]}
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = M / "data/selfplay-mc01234-encoded.npz"
meta = json.loads(str(np.load(path)["meta_json"]))
ds = load_dataset(path)
games = np.sort(np.unique(ds.game))
counts = [len(np.unique(load_dataset(M / f"data/selfplay-mc{k}-encoded.npz").game)) for k in range(5)]
assert sum(counts) == len(games)
_, val = split_for(ds, 0.15, ValueConfig(seed=0, split_seed=0))
g = ds.game[val]
bounds = [games[c - 1] for c in np.cumsum(counts)[:-1]]
grp = np.searchsorted(np.array(bounds), g, side="left")
enc = Encoder(load_regulation(meta["format_id"]))
swap = SwapSlots.of(enc.mon_names, enc.side_names)
imposter = enc.vocab.abilities.get("imposter", 0)
print("val rows", len(val), "imposter rows",
      int(position_reader_rows(ds.encoded.ability[val], ds.encoded.mask[val], imposter).sum()), flush=True)
ARR = {"as played": (False, False), "side 0": (True, False), "side 1": (False, True), "both": (True, True)}
y = ds.outcome[val].astype(np.float64)


@torch.no_grad()
def logits(net, flip):
    net.eval()
    out = np.empty(len(val))
    for s in range(0, len(val), 8192):
        idx = val[s : s + 8192]
        batch = ds.tensors(idx, dev)
        f = np.tile(np.array(flip), (len(idx), 1))
        batch = swap_batch(batch, f, swap)
        out[s : s + len(idx)] = net(batch).float().cpu().numpy()
    return out


def per(z):
    p = np.clip(1 / (1 + np.exp(-z)), 1e-7, 1 - 1e-7)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def se(v, mask):
    _, inv = np.unique(g[mask], return_inverse=True)
    pg = np.bincount(inv, weights=v[mask]) / np.bincount(inv)
    return float(pg.std(ddof=1) / np.sqrt(len(pg)))


groups = {"all": np.ones(len(val), bool), **{f"gen-{k}": grp == k for k in range(5)}}
result = {}
played = {}
for name, files in models.items():
    t = time.time()
    nets = [load_model(Path(f), enc)[0].to(dev) for f in files]
    z = {a: np.mean([logits(n, fl) for n in nets], axis=0) for a, fl in ARR.items()}
    played[name] = z['as played']
    p = {a: 1 / (1 + np.exp(-v)) for a, v in z.items()}
    result[name] = {}
    tta = np.mean([z[a] for a in ARR], axis=0)
    for gn, mask in groups.items():
        row = {}
        for a in ARR:
            l = per(z[a])
            row[a] = {"logloss": float(l[mask].mean()), "se": se(l, mask)}
            if a != "as played":
                d = np.abs(p[a] - p["as played"])[mask]
                row[a]["mean_abs_dp"] = float(d.mean())
                row[a]["pct"] = {str(q): float(np.percentile(d, q)) for q in (50, 90, 99, 99.9)}
                row[a]["max_abs_dp"] = float(d.max())
                row[a]["paired_vs_played"] = {
                    "mean": float((per(z[a]) - per(z["as played"]))[mask].mean()),
                    "se": se(per(z[a]) - per(z["as played"]), mask),
                }
        l = per(tta)
        row["tta4"] = {"logloss": float(l[mask].mean()), "se": se(l, mask)}
        d = per(tta) - per(z["as played"])
        row["tta4"]["paired_vs_played"] = {"mean": float(d[mask].mean()), "se": se(d, mask)}
        result[name][gn] = row
    print(name, f"{time.time() - t:.0f}s", flush=True)
    for gn in ("all",):
        for a in ARR:
            r = result[name][gn][a]
            print(f"  {gn:6s} {a:10s} logloss {r['logloss']:.4f}"
                  + (f"  mean|dp| {r['mean_abs_dp']:.4f} p50 {r['pct']['50']:.4f} p90 {r['pct']['90']:.4f} p99 {r['pct']['99']:.4f} max {r['max_abs_dp']:.3f}" if a != "as played" else ""), flush=True)
        print(f"  tta4 logloss {result[name][gn]['tta4']['logloss']:.4f}", flush=True)
names = list(models)
if len(names) == 2:
    d = per(played[names[0]]) - per(played[names[1]])
    result["paired_as_played_first_minus_second"] = {gn: {"mean": float(d[m].mean()), "se": se(d, m)} for gn, m in groups.items()}
    for gn, v in result["paired_as_played_first_minus_second"].items():
        print(f"paired {names[0]} - {names[1]} as played, {gn}: {v['mean']:+.5f} (se {v['se']:.5f})")
out_path.write_bytes(json.dumps(result, indent=1).encode())
