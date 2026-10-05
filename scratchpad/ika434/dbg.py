"""IKA-434: per-fold weights of the 1-parameter corrections (why a few gains are negative)."""
import sys

sys.argv = [sys.argv[0], "bindaux"]
src = open("C:/tmp/ika434/analyze.py", encoding="utf-8").read().split("report: dict =")[0]
exec(compile(src, "analyze_head", "exec"))  # noqa: S102
for name in ["species_shown", "absorb_best", "alive", "two_hit_sure"]:
    X = linear(name, valid[name])  # noqa: F821
    for f in (0, 1):
        tr = fold != f  # noqa: F821
        w = fit(np.hstack([np.ones((n, 1)), X])[tr], logit["bindaux"][tr], y[tr])  # noqa: F821
        print(name, "fit on fold", 1 - f, "intercept %+.4f slope %+.4f" % (w[0], w[1]))
    print(name, "gain", gain("bindaux", X, valid[name]))  # noqa: F821
for f in (0, 1):
    m = fold == f  # noqa: F821
    print("fold", f, "rows", int(m.sum()), "result-model %+.3f" % ((y[m] - prob["bindaux"][m]).mean() * 100))  # noqa: F821
