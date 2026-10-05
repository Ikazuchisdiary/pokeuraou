"""IKA-434 stage 1b, question 3: static candidates read off the reading of the top rows, on the
same correction (value-mc4bindaux x2's logit as the offset, cross-fitted on halves of games).
For each: its own gain, and how much of the look-ahead's gain (search - value-mc3 static, on top
of value-mc3 static) remains once it is in. Writes stage1b_q3.json."""
import json
import sys

sys.argv = [sys.argv[0], "bindaux", "st"]
src = open("C:/tmp/ika434/stage1b.py", encoding="utf-8").read().split("ALL = np.ones")[0]
exec(compile(src, "stage1b_head", "exec"))  # noqa: S102

r2 = np.fromfile(HERE / "rust2.bin", dtype="<f4").reshape(n, 2, 18).astype(np.float64)  # noqa: F821
names2 = json.loads((HERE / "rust2.json").read_text())["names"]  # noqa: F821
assert np.array_equal(r2[:, :, :16], rust.astype(np.float64))  # noqa: F821
per = np.load(HERE / "perish.npy").astype(np.float64)  # noqa: F821


def diff(a: np.ndarray) -> np.ndarray:
    return a[:, 0] - a[:, 1]


cands = {
    "outspeed_next": diff(r2[:, :, names2.index("outspeed_next")]),
    "order_change": diff(r2[:, :, names2.index("outspeed_next")]) - diff(r2[:, :, names2.index("outspeed")]),
    "hh_bind_sure": diff(r2[:, :, names2.index("hh_bind_sure")]),
    "perish_on": diff(per[:, :, 0]),
    "perish_doom": diff(per[:, :, 1]),
}
ALL = np.ones(n, bool)  # noqa: F821
g_base, _ = gain_in(col(MC3_1), ALL)  # noqa: F821
g_look, _ = gain_in(col(MC3_1, LOOK), ALL)  # noqa: F821
look_alone = g_look - g_base
out = {"look_alone": look_alone, "cands": {}}
for name, d in cands.items():
    X = col(np.clip(d, -3, 3))  # noqa: F821
    vals, counts = np.unique(np.clip(np.floor(d + 0.5), -3, 3), return_counts=True)
    keep = vals[counts >= 1000]
    H = (np.clip(np.floor(d + 0.5), -3, 3)[:, None] == keep[None, :]).astype(np.float64)
    g_lin, se_lin = gain_in(X, ALL)  # noqa: F821
    g_hot, se_hot = gain_in(H, ALL)  # noqa: F821
    g_st, se_st = gain_in(X, ALL, off=ST)  # noqa: F821
    g0, _ = gain_in(np.hstack([X, col(MC3_1)]), ALL)  # noqa: F821
    g1, _ = gain_in(np.hstack([X, col(MC3_1, LOOK)]), ALL)  # noqa: F821
    bins = []
    for v in keep:
        m = np.clip(np.floor(d + 0.5), -3, 3) == v
        r = (y - prob["bindaux"])[m] * 100  # noqa: F821
        bins.append({"bin": float(v), "rows": int(m.sum()), "res": float(r.mean()), "se": cluster_se(r, game[m])})  # noqa: F821
    out["cands"][name] = {"present": float((d != 0).mean()), "linear": [g_lin, se_lin], "onehot": [g_hot, se_hot],
                          "st_linear": [g_st, se_st], "look_after": g1 - g0, "bins": bins}
    print(f"{name:14s} nonzero {np.mean(d != 0):.3%} linear {g_lin * 1e4:+.2f} ({se_lin * 1e4:.2f}) onehot {g_hot * 1e4:+.2f} "
          f"({se_hot * 1e4:.2f}) st {g_st * 1e4:+.2f} | look alone {look_alone * 1e4:+.2f} after {(g1 - g0) * 1e4:+.2f}", flush=True)
    print("   ", " ".join(f"{b['bin']:+g}:{b['res']:+.2f}({b['se']:.2f},{b['rows']})" for b in bins), flush=True)
(HERE / "stage1b_q3.json").write_bytes(json.dumps(out, indent=1).encode("utf-8"))  # noqa: F821
