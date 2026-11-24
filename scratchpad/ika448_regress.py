import json, numpy as np
rows = json.load(open("C:/tmp/ika448/analysis.json"))
f = [x for x in rows if x["kind"] == "field"]; c = [x for x in rows if x["kind"] == "control"]
for name, sel in (("all", rows), ("field", f), ("control", c)):
    s = np.array([x["score"] for x in sel]); v = np.array([x["value"] for x in sel])
    print(name, "n", len(sel), "r(value, score) across teams %.2f" % np.corrcoef(v, s)[0, 1],
          "sd score %.3f sd value %.3f gap sd %.3f" % (s.std(ddof=1), v.std(ddof=1), (v - s).std(ddof=1)))
X = np.array([[1, x["unseen"], x["overlap"]] for x in f]); y = np.array([x["gap"] for x in f])
b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
e = y - X @ b; s2 = e @ e / (len(y) - 3); se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
print("gap ~ 1 + unseen + overlap:", [f"{a:+.4f} ({s:.4f})" for a, s in zip(b, se)])
X1 = X[:, :2]; b1 = np.linalg.lstsq(X1, y, rcond=None)[0]; e1 = y - X1 @ b1
se1 = np.sqrt(np.diag(e1 @ e1 / (len(y) - 2) * np.linalg.inv(X1.T @ X1)))
print("gap ~ 1 + unseen:", [f"{a:+.4f} ({s:.4f})" for a, s in zip(b1, se1)])
# noise floor: mean per-team gap se
print("mean per-team gap se %.3f" % np.mean([x["gap_se"] for x in rows]))
print("score mean field %.3f control %.3f" % (np.mean([x["score"] for x in f]), np.mean([x["score"] for x in c])))
