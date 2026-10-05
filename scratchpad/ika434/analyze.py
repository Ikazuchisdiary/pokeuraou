"""IKA-434 step 3: residual tables and the small correction per candidate feature.

    python analyze.py ARM [ARM...]        (the first ARM is the one ranked; the others are shown beside)

Reads sample.npz (game, turn, kind, outcome, Python's features), rust.bin (the port's
residual-features, 2 x 16 float32 per position, names in rust.json) and predict.npz (logits).
Every feature is antisymmetric: side 0's value minus side 1's. "result - model" is the mean
of (outcome - p) in points (100 x), IKA-429's sign: + means side 0 wins more than the model says.
se is over games (all of a game's positions are one cluster).

Correction: per feature, a logistic regression on the model's logit as an offset, with an
intercept and one weight per value bin, fitted on half of the games and scored on the other
half (both ways). "gain" is the scored log loss of offset+intercept minus offset+intercept+bins
(positive: the feature lowers the loss), with se over games. Writes analysis.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path("C:/tmp/ika434")
arms = sys.argv[1:]
s = np.load(HERE / "sample.npz")
p = np.load(HERE / "predict.npz")
rust_names = json.loads((HERE / "rust.json").read_text())["names"]
n = len(s["game"])
rust = np.fromfile(HERE / "rust.bin", dtype="<f4").reshape(n, 2, len(rust_names))
for k in ("game", "turn", "kind", "outcome"):
    a, b = s[k], p[k]
    assert len(a) == len(b) and np.array_equal(a.astype(np.float64), b.astype(np.float64)), k
y = s["outcome"].astype(np.float64)
game = s["game"]
turn = s["turn"]
side_names = list(p["side_names"])
side = p["side"].astype(np.float64)
field_names = list(p["field_names"])
field = p["field"].astype(np.float64)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


logit = {}
for arm in arms:
    members = sorted(k for k in p.files if k.startswith(f"logit_{arm}#"))
    logit[arm] = np.mean([p[k] for k in members], axis=0)
prob = {a: sigmoid(z) for a, z in logit.items()}

# The revision-4 sure bind count of each side, from the shard's own side columns (IKA-429's way).
total = side[:, :, side_names.index("foes_bound")] * 2
least = side[:, :, side_names.index("double_bind")]
hi, lo = total - least, least
shard_sure = (hi >= 1.0).astype(int) + (lo >= 1.0).astype(int)
port_sure = rust[:, :, rust_names.index("field_bind_sure")]
agree = float((shard_sure == port_sure).all(axis=1).mean())

feats: dict[str, np.ndarray] = {}
valid: dict[str, np.ndarray] = {}
for k, name in enumerate(rust_names):
    feats[name] = rust[:, 0, k] - rust[:, 1, k]
    valid[name] = np.ones(n, bool)
absorb = rust[:, :, rust_names.index("absorb_best")]
valid["absorb_best"] = (absorb >= 0).all(axis=1)
feats["absorb_best"] = np.where(valid["absorb_best"], absorb[:, 1] - absorb[:, 0], 0.0)  # + = side 0 absorbs better
for k, name in enumerate(list(s["py_names"])):
    feats[name] = s["py"][:, 0, k].astype(np.float64) - s["py"][:, 1, k].astype(np.float64)
    valid[name] = np.ones(n, bool)
feats["mega_available"] = side[:, 0, side_names.index("mega_available")] - side[:, 1, side_names.index("mega_available")]
valid["mega_available"] = np.ones(n, bool)
feats["shard_bind_sure"] = (shard_sure[:, 0] - shard_sure[:, 1]).astype(np.float64)
valid["shard_bind_sure"] = np.ones(n, bool)
# Reference, not a candidate: the generation's own one-turn search value (value-mc3's search,
# side 0's view) against the model, in logits. How much a one-turn look ahead still knows.
with np.load("C:/Users/Ikazuchi/repos/pokeuraou/data/selfplay-mc4-encoded.npz") as z:
    take = np.isin(z["game"], s["picked"])
    assert np.array_equal(z["game"][take], game)
    sv = np.clip(z["search_value"][take].astype(np.float64), 1e-4, 1 - 1e-4)
SV_LOGIT = np.log(sv / (1 - sv))
feats["sv_logit"] = np.clip(SV_LOGIT, -4, 4)
valid["sv_logit"] = np.ones(n, bool)
# Features whose value is already an input of the current net (a check that the table reads ~0).
INPUT = {"alive", "active_alive", "bench_alive", "mega_available", "field_ko_sure", "shard_bind_sure",
         "field_bind_sure", "pp_zero", "pp_quarter", "weather_turns", "terrain_turns"}

CONT = {"frac_noko": 0.25, "frac_all": 0.25, "absorb_best": 0.25, "outspeed": 1.0, "sv_logit": 1.0}


def bins_of(name: str) -> np.ndarray:
    d = feats[name]
    if name in CONT:
        w = CONT[name]
        lim = 1.5 if w < 1 else 3
        return np.clip(np.floor(d / w + 0.5) * w, -lim, lim)
    if name in ("weather_turns", "terrain_turns", "moves_shown", "pp_quarter"):
        return np.clip(np.floor(d / 2 + 0.5) * 2, -6, 6)
    return np.clip(np.floor(d + 0.5), -3, 3)


def ll(yy: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = np.clip(q, 1e-7, 1 - 1e-7)
    return -(yy * np.log(q) + (1 - yy) * np.log(1 - q))


def cluster_se(d: np.ndarray, g: np.ndarray) -> float:
    if len(d) < 2:
        return float("nan")
    _u, inv = np.unique(g, return_inverse=True)
    sums, counts = np.bincount(inv, weights=d), np.bincount(inv)
    mean = d.mean()
    return float(np.sqrt(((sums - mean * counts) ** 2).sum()) / counts.sum())


def fit(X: np.ndarray, off: np.ndarray, yy: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Newton on the log loss of sigmoid(off + X w); the ridge spares column 0 (the intercept)."""
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], ridge)
    pen[0] = 1e-6
    for _ in range(25):
        q = sigmoid(off + X @ w)
        g = X.T @ (q - yy) + pen * w
        h = (X * (q * (1 - q))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


uniq_games = np.unique(game)
fold_of_game = np.random.default_rng(434).permutation(len(uniq_games)) % 2
fold = fold_of_game[np.searchsorted(uniq_games, game)]


MIN_BIN = 1000  # a bin with fewer rows gets no weight of its own (it stays at the intercept)
BANDS = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 999)]


def onehot(name: str, mask: np.ndarray) -> np.ndarray:
    b = bins_of(name)
    vals, counts = np.unique(b[mask], return_counts=True)
    vals = vals[counts >= MIN_BIN]
    X = (b[:, None] == vals[None, :]).astype(np.float64)
    X[~mask] = 0.0
    return X


def linear(name: str, mask: np.ndarray) -> np.ndarray:
    d = np.clip(feats[name], -3, 3) if name not in CONT else feats[name]
    return np.where(mask, d, 0.0)[:, None]


def banded(name: str, mask: np.ndarray) -> np.ndarray:
    d = linear(name, mask)[:, 0]
    return np.stack([np.where((turn >= lo_t) & (turn <= hi_t), d, 0.0) for lo_t, hi_t in BANDS], axis=1)


def slope(arm: str, X: np.ndarray) -> float:
    """Points per unit of the feature near 50% (the fitted logit weight x 25), on all rows."""
    w = fit(np.hstack([np.ones((n, 1)), X]), logit[arm], y)
    return float(w[1] * 25)


def gain(arm: str, X: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Scored log-loss drop over all rows (rows outside mask keep the baseline)."""
    off = logit[arm]
    base_ll = np.zeros(n)
    feat_ll = np.zeros(n)
    ones = np.ones((n, 1))
    for f in (0, 1):
        tr, te = fold != f, fold == f
        wb = fit(ones[tr], off[tr], y[tr])
        Xi = np.hstack([ones, X])
        wf = fit(Xi[tr], off[tr], y[tr])
        base_ll[te] = ll(y[te], sigmoid(off[te] + ones[te] @ wb))
        feat_ll[te] = ll(y[te], sigmoid(off[te] + Xi[te] @ wf))
    d = base_ll - feat_ll
    return float(d.mean()), cluster_se(d, game), float(d[mask].mean()) if mask.any() else float("nan")


report: dict = {"rows": int(n), "games": int(len(uniq_games)), "bind_sure_agree": agree, "arms": {}, "features": {}}
for arm in arms:
    report["arms"][arm] = {"loss": float(ll(y, prob[arm]).mean()), "result_minus_model": float((y - prob[arm]).mean() * 100)}
order = ["shard_bind_sure"] + rust_names + [x for x in s["py_names"]] + ["mega_available", "sv_logit"]
for name in order:
    mask = valid[name]
    b = bins_of(name)
    rows = []
    for v in np.unique(b[mask]):
        m = mask & (b == v)
        cell = {"bin": float(v), "rows": int(m.sum()), "share": float(m.mean()), "win": float(y[m].mean())}
        for arm in arms:
            r = (y - prob[arm])[m]
            cell[arm] = {"res": float(r.mean() * 100), "se": cluster_se(r * 100, game[m]), "loss": float(ll(y[m], prob[arm][m]).mean())}
        rows.append(cell)
    entry = {"input": name in INPUT, "nonzero_diff": float((feats[name][mask] != 0).mean() * mask.mean()),
             "valid": float(mask.mean()), "bins": rows, "gain": {}}
    big = [c for c in rows if c["share"] >= 0.01]
    for arm in arms:
        overall = float((y - prob[arm]).mean() * 100)
        cells = {}
        for kind, X in (("onehot", onehot(name, mask)), ("linear", linear(name, mask)), ("banded", banded(name, mask))):
            g, se, _gm = gain(arm, X, mask)
            cells[kind] = g
            cells[kind + "_se"] = se
        cells["slope"] = slope(arm, linear(name, mask))
        cells["max_abs_res"] = max((abs(c[arm]["res"]) for c in big), default=float("nan"))
        # Relative to the arm's overall residual, and in standard errors.
        cells["max_abs_dev"] = max((abs(c[arm]["res"] - overall) for c in big), default=float("nan"))
        cells["max_z"] = max((abs(c[arm]["res"] - overall) / c[arm]["se"] for c in big), default=float("nan"))
        entry["gain"][arm] = cells
    report["features"][name] = entry
    print(name, json.dumps({a: {k: round(v, 5) for k, v in entry["gain"][a].items()} for a in arms}), flush=True)

# Everything at once (not the shard's own bind count, which the current net reads).
alls = [x for x in order if x not in ("shard_bind_sure", "sv_logit")]
report["all_features"] = {}
for arm in arms:
    cells = {}
    for kind, X in (("linear", np.hstack([linear(x, valid[x]) for x in alls])),
                    ("banded", np.hstack([banded(x, valid[x]) for x in alls])),
                    ("onehot", np.hstack([onehot(x, valid[x]) for x in alls]))):
        g, se, _ = gain(arm, X, np.ones(n, bool))
        cells[kind] = g
        cells[kind + "_se"] = se
        print("all", arm, kind, round(g, 5), round(se, 5), flush=True)
    report["all_features"][arm] = cells
# Turn bands, for the reader.
tb = []
for lo_t, hi_t in [(1, 1), (2, 3), (4, 6), (7, 10), (11, 99)]:
    m = (turn >= lo_t) & (turn <= hi_t)
    tb.append({"turns": f"{lo_t}-{hi_t}", "rows": int(m.sum()),
               **{a: float((y - prob[a])[m].mean() * 100) for a in arms}})
report["turn_bands"] = tb
(HERE / "analysis.json").write_bytes(json.dumps(report, indent=1).encode("utf-8"))
print(json.dumps({"rows": n, "agree": agree, "arms": report["arms"], "all": report["all_features"]}, indent=1))
