"""IKA-434: where does the generation's one-turn search value still know more than the model?
The sv gain (cross-fitted, linear on logit(sv) with the model's logit as offset) by turn band and
by the number of Pokemon left, and the correlation of each candidate feature with the gap
logit(sv) - logit(model). Prints a table; writes svdiag.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.argv = [sys.argv[0], "bindaux", "st"]
HERE = Path("C:/tmp/ika434")
src = (HERE / "analyze.py").read_text()
# Reuse analyze.py's loading and helpers, without its per-feature loop.
head = src.split("report: dict =")[0]
exec(compile(head, "analyze_head", "exec"))  # noqa: S102

out = {}
alive_total = s["py"][:, 0, list(s["py_names"]).index("alive")] + s["py"][:, 1, list(s["py_names"]).index("alive")]
for arm in ("bindaux", "st"):
    gap = SV_LOGIT - logit[arm]  # noqa: F821
    X = np.clip(SV_LOGIT, -6, 6)[:, None]  # noqa: F821
    rows = []
    groups = [(f"turn {a}-{b}", (turn >= a) & (turn <= b)) for a, b in BANDS]  # noqa: F821
    groups += [(f"left {a}-{b}", (alive_total >= a) & (alive_total <= b)) for a, b in [(7, 8), (5, 6), (3, 4), (2, 2)]]
    g_all, se_all, _ = gain(arm, X, np.ones(n, bool))  # noqa: F821
    for label, m in groups:
        Xm = np.where(m, X[:, 0], 0.0)[:, None]
        g, se, gm = gain(arm, Xm, m)  # noqa: F821
        rows.append({"group": label, "rows": int(m.sum()), "share": float(m.mean()),
                     "gain_all_rows": g, "se": se, "gain_in_group": gm,
                     "loss_model": float(ll(y[m], prob[arm][m]).mean()),  # noqa: F821
                     "loss_sv": float(ll(y[m], sv[m]).mean())})  # noqa: F821
    corr = {}
    for name in feats:  # noqa: F821
        if name in ("sv_logit",):
            continue
        f = feats[name]  # noqa: F821
        m = valid[name]  # noqa: F821
        if f[m].std() > 0:
            corr[name] = float(np.corrcoef(f[m], gap[m])[0, 1])
    out[arm] = {"gain": g_all, "se": se_all, "groups": rows, "corr_with_gap": corr,
                "loss_model": float(ll(y, prob[arm]).mean()), "loss_sv": float(ll(y, sv).mean())}  # noqa: F821
    print(arm, round(g_all, 5), round(se_all, 5), "loss model", round(out[arm]["loss_model"], 4), "sv", round(out[arm]["loss_sv"], 4))
    for r in rows:
        print("  %-10s %7d %.3f gain %+.5f (%.5f) in-group %+.5f | loss model %.4f sv %.4f" % (
            r["group"], r["rows"], r["share"], r["gain_all_rows"], r["se"], r["gain_in_group"], r["loss_model"], r["loss_sv"]))
    top = sorted(corr.items(), key=lambda kv: -abs(kv[1]))[:12]
    print("  corr with gap:", ", ".join(f"{k} {v:+.3f}" for k, v in top))
(HERE / "svdiag.json").write_bytes(json.dumps(out, indent=1).encode("utf-8"))
