"""IKA-434 stage 1b: did the auxiliary targets (search:1, ahead2:5) take up the look-ahead?
The same search and look-ahead corrections with value-mc4bind x2 (binds, no auxiliary targets,
IKA-429) and value-mc4stx2 as the offset, beside value-mc4bindaux x2."""
import json
import sys

sys.argv = [sys.argv[0], "bindaux", "st"]
src = open("C:/tmp/ika434/stage1b.py", encoding="utf-8").read().split("ALL = np.ones")[0]
exec(compile(src, "stage1b_head", "exec"))  # noqa: S102
b = np.load(HERE / "predict-bind.npz")  # noqa: F821
assert np.array_equal(b["game"], game)  # noqa: F821
BIND = (b["logit_bind#0"] + b["logit_bind#1"]) / 2
COPY = (b["logit_bind#2"] + b["logit_bind#3"]) / 2
print("IKA-429 copy of bindaux equals the production pair:", float(np.abs(COPY - BA).max()))  # noqa: F821
ALL = np.ones(n, bool)  # noqa: F821
out = {}
for name, off in (("bindaux", BA), ("bind (no aux)", BIND), ("st", ST)):  # noqa: F821
    row = {"loss": float(ll(y, sigmoid(off)).mean())}  # noqa: F821
    for tname, X in (("search", col(SVL)), ("look", col(LOOK)), ("mc3 + look", col(MC3_1, LOOK)), ("mc3", col(MC3_1))):  # noqa: F821
        g, se = gain_in(X, ALL, off=off)  # noqa: F821
        row[tname] = [g, se]
    out[name] = row
    print(name, json.dumps({k: (round(v[0] * 1e4, 2), round(v[1] * 1e4, 2)) if isinstance(v, list) else round(v, 4)
                            for k, v in row.items()}), flush=True)
(HERE / "auxcheck.json").write_bytes(json.dumps(out, indent=1).encode("utf-8"))  # noqa: F821
