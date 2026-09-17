"""How high could turn-1 AUC go, if the model were perfect?

AUC on a binary label is bounded by how spread out the true win probabilities are. If
every turn-1 position is genuinely a coin flip, a perfect predictor still scores near 0.5,
and no amount of training data moves it. If turn-1 probabilities are spread wide and the
model reads 0.681, the model is the problem and more data is the fix.

Measured on pool8, whose models this was asked about:

    turn        n   search spread   ceiling   the model at that turn
       1   60,225        0.178       0.704    0.672 at td-lambda 0, 0.694 at 0.5
       2   71,429        0.227       0.763
       5   81,602        0.350       0.903
      10                 0.427       0.971    0.963

The ceiling tracks the model the whole way down, and turn 1 already carries 60,225 rows --
more than turns 10 to 12 together, and only 18% fewer than turn 2, whose AUC is 0.077
higher. So the curve is not a data curve, and buying more turn-1 self-play would buy at
most the gap. What closed two thirds of that gap was `--td-lambda`, which changes the
target rather than the quantity (G5).

The true probabilities are not available, so this uses the recorded `search_value` as a
stand-in -- a biased but low-noise estimate -- and reports, per turn:

  spread        the standard deviation of the search value, which is what drives the bound
  Bayes AUC     drawing outcomes from the search value and scoring the search value
                against them: the AUC a PERFECT predictor of those probabilities gets
  model AUC     from the training log, for comparison

**The bias matters and is not removable here.** The search runs the same value net at its
leaves, so if the net is systematically unsure at turn 1 the search inherits it and this
understates the ceiling. The measurement that does not have this problem is replaying
identical turn-1 positions many times and using the empirical win rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based AUC, ties averaged."""
    if label.min() == label.max():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties
    sorted_scores = score[order]
    start = 0
    for i in range(1, len(score) + 1):
        if i == len(score) or sorted_scores[i] != sorted_scores[start]:
            ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    positives = label > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/selfplay-pool9-encoded.npz")
    data = np.load(path, allow_pickle=False)
    if "search_value" not in data:
        raise SystemExit(f"{path.name} carries no search_value")
    meta = json.loads(str(data["meta_json"]))
    leaves = meta.get("objectives") or {}
    wrong = sorted(k for k in leaves if not k.startswith("value:"))
    if wrong:
        raise SystemExit(f"search values are not win probabilities here: {wrong}")

    if "turn" not in data:
        raise SystemExit(f"{path.name} carries no turn column; keys: {sorted(data.keys())}")
    turn = data["turn"]
    search = data["search_value"].astype(np.float64)
    outcome = data["outcome"].astype(np.float64)

    rng = np.random.default_rng(0)
    print(f"  {path.name}   {len(search):,} decisions, leaves {sorted(leaves)}\n")
    print("   turn        n   search spread   Bayes AUC   actual AUC of search")
    for t in range(1, 13):
        keep = turn == t
        n = int(keep.sum())
        if n < 500:
            continue
        p = np.clip(search[keep], 1e-6, 1 - 1e-6)
        # Outcomes drawn from the search value, scored by the search value: the ceiling a
        # perfect predictor of these probabilities would hit. Averaged over draws.
        ceiling = np.mean([auc(p, (rng.random(n) < p).astype(np.float64)) for _ in range(9)])
        real = auc(p, outcome[keep])
        print(f"  {t:>5} {n:>8,}      {p.std():.3f}        {ceiling:.3f}        {real:.3f}")

    print("\n  If the ceiling column tracks the actual one, the limit is the game and not\n"
          "  the data. If the ceiling is far above, the model is leaving information on\n"
          "  the table and more turn-1 rows would be worth buying.")


if __name__ == "__main__":
    main()
