"""IKA-434: the 1-parameter correction's slope fitted on each half of the games, per feature and
arm (points per unit near 50%: logit weight x 25). A stable effect has the same sign in both
halves. Writes foldslopes.json."""
import json
import sys

sys.argv = [sys.argv[0], "bindaux", "st"]
src = open("C:/tmp/ika434/analyze.py", encoding="utf-8").read().split("report: dict =")[0]
exec(compile(src, "analyze_head", "exec"))  # noqa: S102
names = ["shard_bind_sure"] + rust_names + list(s["py_names"]) + ["mega_available", "sv_logit"]  # noqa: F821
out = {}
for name in names:
    X = np.hstack([np.ones((n, 1)), linear(name, valid[name])])  # noqa: F821
    out[name] = {}
    for arm in ("bindaux", "st"):
        out[name][arm] = [float(fit(X[fold == f], logit[arm][fold == f], y[fold == f])[1] * 25) for f in (0, 1)]  # noqa: F821
    print(name, {a: [round(v, 3) for v in vs] for a, vs in out[name].items()}, flush=True)
open("C:/tmp/ika434/foldslopes.json", "wb").write(json.dumps(out, indent=1).encode("utf-8"))
