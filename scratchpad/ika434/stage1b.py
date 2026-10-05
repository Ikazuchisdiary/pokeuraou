"""IKA-434 stage 1b: what the generation's search value adds to value-mc4bindaux x2.

    python stage1b.py

Same 233,025 positions and the same cross-fitted correction as analyze.py (the model's logit as
the offset, an intercept and the named terms, fitted on half of the games and scored on the
other half). Writes stage1b.json and top.npz (the top-5% rows, for reading positions).
"""
import json
import sys
from types import SimpleNamespace

sys.argv = [sys.argv[0], "bindaux", "st"]
src = open("C:/tmp/ika434/analyze.py", encoding="utf-8").read().split("report: dict =")[0]
exec(compile(src, "analyze_head", "exec"))  # noqa: S102
sys.path.insert(0, "C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-aa55f8431dd4ced5d/src")
from pokeuraou import value as V  # noqa: E402

# ---------------------------------------------------------------- the nets' static values
q = np.load(HERE / "predict-mc3.npz")  # noqa: F821
assert np.array_equal(q["game"], game) and np.array_equal(q["turn"], turn)  # noqa: F821
MC3_1 = q["logit_mc3#0"].astype(np.float64)  # value-mc3.pt: the generation's own leaf (x1)
MC3_2 = (q["logit_mc3#0"] + q["logit_mc3#1"]).astype(np.float64) / 2
ST = logit["st"]  # noqa: F821
BA = logit["bindaux"]  # noqa: F821
SVL = np.clip(SV_LOGIT, -6, 6)  # noqa: F821
LOOK = np.clip(SV_LOGIT - MC3_1, -6, 6)  # noqa: F821  the look-ahead: depth 1 minus depth 0, same net
py_names = list(s["py_names"])  # noqa: F821
shown1 = s["py"][:, 1, py_names.index("species_shown")]  # noqa: F821
OPEN = shown1 >= 4  # side 0 has seen all four of side 1's: one completion, nothing hidden
alive = s["py"][:, :, py_names.index("alive")].astype(np.float64)  # noqa: F821
left = alive.sum(axis=1)


def col(*xs: np.ndarray) -> np.ndarray:
    return np.stack(xs, axis=1)


def gain_in(X: np.ndarray, m: np.ndarray, off: np.ndarray = BA) -> tuple[float, float]:
    """Cross-fitted log-loss drop on the rows of m only (fit and score inside m)."""
    idx = np.flatnonzero(m)
    base, feat = np.zeros(len(idx)), np.zeros(len(idx))
    ones = np.ones((len(idx), 1))
    Xi = np.hstack([ones, X[idx]])
    for f in (0, 1):
        tr = fold[idx] != f  # noqa: F821
        te = ~tr
        wb = fit(ones[tr], off[idx][tr], y[idx][tr])  # noqa: F821
        wf = fit(Xi[tr], off[idx][tr], y[idx][tr])  # noqa: F821
        base[te] = ll(y[idx][te], sigmoid(off[idx][te] + ones[te] @ wb))  # noqa: F821
        feat[te] = ll(y[idx][te], sigmoid(off[idx][te] + Xi[te] @ wf))  # noqa: F821
    d = base - feat
    return float(d.mean()), cluster_se(d, game[idx])  # noqa: F821


ALL = np.ones(n, bool)  # noqa: F821
groups = {"all": ALL, "open": OPEN, "hidden": ~OPEN}
out: dict = {"rows": int(n), "open_share": float(OPEN.mean())}  # noqa: F821
losses = {}
for gname, m in groups.items():
    losses[gname] = {k: float(ll(y[m], sigmoid(z[m])).mean()) for k, z in  # noqa: F821
                     (("bindaux", BA), ("st", ST), ("mc3x1", MC3_1), ("mc3x2", MC3_2), ("search", SV_LOGIT))}  # noqa: F821
out["losses"] = losses
print("losses", json.dumps(losses, indent=None), flush=True)

# ---------------------------------------------------------------- question 1
q1_terms = {
    "search": col(SVL),
    "mc3x1 static": col(MC3_1),
    "mc3x2 static": col(MC3_2),
    "stx2 static": col(ST),
    "search - mc3x1 static": col(LOOK),
    "mc3x1 static + (search - mc3x1)": col(MC3_1, LOOK),
    "mc3x2 + stx2 static": col(MC3_2, ST),
    "mc3x2 + stx2 + (search - mc3x1)": col(MC3_2, ST, LOOK),
}
out["q1"] = {}
for gname, m in groups.items():
    out["q1"][gname] = {}
    for tname, X in q1_terms.items():
        g, se = gain_in(X, m)
        out["q1"][gname][tname] = [g, se]
        print(f"q1 {gname:7s} {tname:35s} {g * 1e4:+7.2f} ({se * 1e4:.2f})", flush=True)

# ---------------------------------------------------------------- question 2
class _D:
    def __len__(self) -> int:
        return n  # noqa: F821


d = _D()
d.encoded = SimpleNamespace(side=side.astype(np.float32), mask=np.zeros((1, 2, 4), np.float32))  # noqa: F821
d.game, d.turn, d.outcome = game, turn, y  # noqa: F821
aux = V.aux_target_arrays(d, ["ko_next", "ahead1", "ahead2"])
ko = aux["ko_next"][:, :, 0].astype(np.float64)  # faints of each side before the next turn's decision
KO_DIFF = ko[:, 1] - ko[:, 0]  # + = more of side 1's faint (good for side 0)
A1 = aux["ahead1"].astype(np.float64)  # alive and HP differences (side 0 - side 1) one turn on
A2 = aux["ahead2"].astype(np.float64)
# The material now, so that the "ahead" terms read the change.
names = list(p["side_names"])  # noqa: F821
now = side[:, 0, [names.index("alive_fraction"), names.index("team_hp_fraction")]] - \
    side[:, 1, [names.index("alive_fraction"), names.index("team_hp_fraction")]]  # noqa: F821
q2_terms = {
    "ko this turn (diff)": col(KO_DIFF),
    "ahead1 change": A1 - now,
    "ahead2 change": A2 - now,
    "ko this turn + ahead2 change": np.hstack([col(KO_DIFF), A2 - now]),
}
out["q2_upper"] = {}
base_static = col(MC3_1)
for tname, X in q2_terms.items():
    gx, sx = gain_in(X, ALL)
    gxs, _ = gain_in(np.hstack([X, col(SVL)]), ALL)
    gsv, _ = gain_in(col(SVL), ALL)
    # the look-ahead part: on top of mc3x1 static, with and without X
    gl0, _ = gain_in(base_static, ALL)
    gl1, _ = gain_in(np.hstack([base_static, col(LOOK)]), ALL)
    gx0, _ = gain_in(np.hstack([X, base_static]), ALL)
    gx1, _ = gain_in(np.hstack([X, base_static, col(LOOK)]), ALL)
    out["q2_upper"][tname] = {"gain_x": [gx, sx], "search_alone": gsv, "search_after_x": gxs - gx,
                              "look_alone": gl1 - gl0, "look_after_x": gx1 - gx0}
    print(f"q2 {tname:30s} x {gx * 1e4:+8.2f} | search alone {gsv * 1e4:+.2f} after x {(gxs - gx) * 1e4:+.2f} | "
          f"look alone {(gl1 - gl0) * 1e4:+.2f} after x {(gx1 - gx0) * 1e4:+.2f}", flush=True)

# The top 5% of |search - model| and how they differ from all rows.
gap = SV_LOGIT - BA  # noqa: F821
cut = np.quantile(np.abs(gap), 0.95)
top = np.abs(gap) >= cut
bind = rust[:, :, rust_names.index("field_bind_sure")]  # noqa: F821
faint_sum = ko.sum(axis=1)


def share(mask: np.ndarray, cats: dict) -> dict:
    return {k: [float(c[mask].mean()), float(c.mean())] for k, c in cats.items()}


cats = {}
for lo_t, hi_t in BANDS:  # noqa: F821
    cats[f"turn {lo_t}-{hi_t}"] = (turn >= lo_t) & (turn <= hi_t)  # noqa: F821
for a, b in [(7, 8), (5, 6), (3, 4), (2, 2)]:
    cats[f"left {a}-{b}"] = (left >= a) & (left <= b)
cats["sure bind on some side"] = (bind > 0).any(axis=1)
cats["double bind on some side"] = (bind >= 2).any(axis=1)
for k in (0, 1, 2):
    cats[f"faints this turn {k}"] = faint_sum == k
cats["faints this turn 3+"] = faint_sum >= 3
cats["open"] = OPEN
p_search = sigmoid(SV_LOGIT)  # noqa: F821
p_model = prob["bindaux"]  # noqa: F821
p_mc3 = sigmoid(MC3_1)  # noqa: F821
out["top"] = {
    "cut_logit": float(cut), "rows": int(top.sum()),
    "loss_model": float(ll(y[top], p_model[top]).mean()), "loss_search": float(ll(y[top], p_search[top]).mean()),  # noqa: F821
    "loss_mc3x1": float(ll(y[top], p_mc3[top]).mean()),  # noqa: F821
    "search_on_right_side": float(((p_search[top] > 0.5) == (y[top] > 0.5)).mean()),  # noqa: F821
    "model_on_right_side": float(((p_model[top] > 0.5) == (y[top] > 0.5)).mean()),  # noqa: F821
    "mean_abs_look": float(np.abs(SV_LOGIT - MC3_1)[top].mean()),  # noqa: F821
    "mean_abs_net": float(np.abs(MC3_1 - BA)[top].mean()),
    "mean_abs_look_all": float(np.abs(SV_LOGIT - MC3_1).mean()),  # noqa: F821
    "mean_abs_net_all": float(np.abs(MC3_1 - BA).mean()),
    "shares": share(top, cats),
}
print(json.dumps(out["top"], indent=1), flush=True)
np.savez(HERE / "top.npz", row=np.flatnonzero(top), gap=gap[top], game=game[top], turn=turn[top],  # noqa: F821
         y=y[top], p_model=p_model[top], p_search=p_search[top], p_mc3=p_mc3[top], ko=ko[top],  # noqa: F821
         look=(SV_LOGIT - MC3_1)[top])  # noqa: F821
(HERE / "stage1b.json").write_bytes(json.dumps(out, indent=1).encode("utf-8"))  # noqa: F821
