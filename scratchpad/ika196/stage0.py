"""IKA-196 stage 0: each label on the captured generation-form games (captured.pkl)."""

import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-acde882fb885a4aa0/src")

from pokeuraou import eqselect  # noqa: E402
from pokeuraou.equilibrium import solve_bayesian  # noqa: E402

games = pickle.load(open("C:/tmp/ika196/captured.pkl", "rb"))
labels = ["unif", "ment0", "ment0.001", "qre0.005", "qre0.02", "qre0.2"]
shapes = [(g[0][0].shape[0], g[0][0].shape[1], len(g[0])) for g in games]
print(f"{len(games)} games; rows mean {np.mean([s[0] for s in shapes]):.1f}, cols mean "
      f"{np.mean([s[1] for s in shapes]):.1f}, completions mean {np.mean([s[2] for s in shapes]):.1f} "
      f"(max {max(s[2] for s in shapes)})")


def take(mats, w, x):  # noqa: ANN001, ANN201
    w = np.asarray(w) / np.sum(w)
    return float(sum(w[k] * (x @ m).mean() for k, m in enumerate(mats)))


lp_seconds = 0.0
solved = []
for mats, w in games:
    t0 = time.perf_counter()
    eq = solve_bayesian(mats, w)
    lp_seconds += time.perf_counter() - t0
    solved.append(eq)
mixed = sum(1 for eq in solved if (eq.row_strategy > 1e-6).sum() > 1)
print(f"LP pair: {1000 * lp_seconds / len(games):.2f} ms a game; LP strategy mixed in "
      f"{mixed}/{len(games)} ({100 * mixed / len(games):.1f}%)")
print("label | moved | moved % | TV mean (moved) | value loss max | uniform-reply gain mean (moved) | ms a game | fallbacks")
for label in labels:
    eqselect.STATS.clear()
    moved = 0
    tv = []
    loss = 0.0
    gain = []
    t0 = time.perf_counter()
    outs = []
    for (mats, w), eq in zip(games, solved, strict=True):
        x = eqselect.reselect_bayesian(mats, w, eq.value, eq.row_strategy, label)
        outs.append(x)
    seconds = time.perf_counter() - t0
    for (mats, w), eq, x in zip(games, solved, outs, strict=True):
        d = 0.5 * float(np.abs(x - eq.row_strategy).sum())
        loss = max(loss, eq.value - eqselect.guarantee(mats, w, x))
        if d > 1e-6:
            moved += 1
            tv.append(d)
            gain.append(take(mats, w, x) - take(mats, w, eq.row_strategy))
    st = eqselect.STATS.get(label, {})
    print(f"{label} | {moved} | {100 * moved / len(games):.1f} | {np.mean(tv) if tv else 0:.3f} | "
          f"{loss:.2e} | {np.mean(gain) if gain else 0:+.4f} | {1000 * seconds / len(games):.2f} | "
          f"{int(st.get('fallbacks', 0))}")
