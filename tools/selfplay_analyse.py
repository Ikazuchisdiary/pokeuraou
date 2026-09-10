"""What is in the self-play data, and how much room a learned value function has.

Before writing a model it is worth measuring the thing the model is supposed to beat. The
search that generated these games scored its leaves with the `hp-share` proxy, and every
decision recorded that score next to the eventual result -- so the data answers, for free,
how well the proxy already predicts who wins.

That number decides what milestone 3 is for. If `hp-share` at turn 5 already separates
winners from losers cleanly, a learned function has little to add and the effort belongs
elsewhere. If it does not, the gap is the headroom, and it can be quoted rather than
assumed.

Two measures, both of which need only the outcome and the proxy:

- a **reliability curve**: bucket the proxy and show the actual win rate in each bucket,
  which is what "calibrated" means and is readable without a metric;
- **AUC**, the probability that a randomly chosen won position scores above a randomly
  chosen lost one. 0.5 is no information, 1.0 is perfect separation.

    uv run python tools/selfplay_analyse.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.selfplay import selfplay_dir


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a won position > score of a lost one), ties counting a half.

    Computed from ranks rather than by pairing, so it is exact and cheap on 100k points.
    """
    wins = labels > 0.5
    if not wins.any() or wins.all():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties so a constant score gives exactly 0.5.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    del unique
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    n_pos = int(wins.sum())
    n_neg = len(scores) - n_pos
    return float((ranks[wins].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--buckets", type=int, default=8)
    args = ap.parse_args()

    directory = args.dir or selfplay_dir()
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no games in {directory}")

    games = 0
    decisions = 0
    outcomes: list[float] = []
    by_archetype: dict[str, list[float]] = defaultdict(list)
    turn_counts: Counter[int] = Counter()
    kinds: Counter[str] = Counter()
    unmodelled: Counter[str] = Counter()
    branching: Counter[int] = Counter()

    # (proxy value, outcome) per decision, and the same split by how far into the game it is.
    scores: list[float] = []
    labels: list[float] = []
    by_turn: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a run still in flight can leave a partial last line
            outcome = record.get("outcome")
            if outcome is None:
                continue
            games += 1
            outcomes.append(float(outcome))
            by_archetype[record["foeArchetype"]].append(float(outcome))
            turn_counts[int(record["turns"])] += 1
            for name in record.get("unmodelled", ()):
                unmodelled[name] += 1
            for decision in record["decisions"]:
                decisions += 1
                kinds[decision["kind"]] += 1
                branching[len(decision["ownActions"])] += 1
                scores.append(float(decision["searchValue"]))
                labels.append(float(outcome))
                by_turn[int(decision["turn"])].append(
                    (float(decision["searchValue"]), float(outcome))
                )

    if not games:
        raise SystemExit("no finished games found")

    print(f"{len(files)} files, {games:,} finished games, {decisions:,} labelled decisions")
    print(f"  side-0 (our team) win rate {np.mean(outcomes) * 100:.1f}%")
    print(f"  decisions per game {decisions / games:.1f}  ({dict(kinds)})")
    turns = np.array(sorted(turn_counts.elements()), dtype=np.float64)
    print(
        f"  game length: median {np.median(turns):.0f} turns, "
        f"90th percentile {np.percentile(turns, 90):.0f}"
    )
    print(f"  actions offered per decision: {sorted(branching.items())[:6]}")

    print("\nby opponent archetype")
    for name, results in sorted(by_archetype.items(), key=lambda kv: np.mean(kv[1])):
        print(
            f"  {name:22} {len(results):>5} games  win {np.mean(results) * 100:5.1f}%"
        )

    score_array = np.array(scores, dtype=np.float64)
    label_array = np.array(labels, dtype=np.float64)
    print("\nhow well the hp-share proxy already predicts the winner")
    print(f"  AUC over all decisions: {auc(score_array, label_array):.3f}")
    edges = np.quantile(score_array, np.linspace(0, 1, args.buckets + 1))
    edges[-1] += 1e-9
    print(f"  {'proxy range':>22}  {'n':>7}  actual win rate")
    for low, high in zip(edges, edges[1:], strict=False):
        mask = (score_array >= low) & (score_array < high)
        if not mask.any():
            continue
        print(
            f"  {low:9.4f}..{high:<9.4f}  {int(mask.sum()):>7}  "
            f"{label_array[mask].mean() * 100:5.1f}%"
        )

    print("\n  the same, by how far into the game the decision was")
    print(f"  {'turn':>6}  {'n':>7}  {'AUC':>6}  mean proxy  win rate")
    for turn in sorted(by_turn):
        rows = by_turn[turn]
        if len(rows) < 200:
            continue
        s = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        print(
            f"  {turn:>6}  {len(rows):>7}  {auc(s, y):6.3f}  "
            f"{s.mean():10.4f}  {y.mean() * 100:5.1f}%"
        )

    if unmodelled:
        print("\neffects the resolver reported during these games (games affected)")
        for name, count in unmodelled.most_common(12):
            print(f"  {count:>6} ({count / games * 100:4.1f}%)  {name}")


if __name__ == "__main__":
    main()
