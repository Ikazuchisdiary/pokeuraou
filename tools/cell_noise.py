"""How far apart do two training seeds put the same position's win probability?

The selection book feeds a value per cell into an LP, and two seeds of one configuration
then print advice that costs each other 2.88 to 5.86 points (E6). That damage starts as a
disagreement about individual cell values, and measuring it there costs a forward pass
instead of the forty minutes a book solve takes -- which is the difference between
screening an idea and committing a morning to it.

Reported per turn, for every pair of the models given:

    spread      RMS difference in win probability between two models on the same position
    worst       the largest such difference, which is what an argmax over 90 cells sees
    |mean|      the difference of the models' means, so a systematic offset is not read
                as noise

Turn 1 is the row that matters: the book is built from turn-1 cells exclusively, and it is
where the value function is weakest -- though not, as it turns out, where it has most to
learn. The achievable AUC at turn 1 is about 0.698 against the 0.681 measured, so the
model is already at the ceiling and more turn-1 rows would buy 0.017 at most. What is left
to fix is not the ranking but the variance, which is what this measures.

**The rows are not held out.** Each model's own split lives in its training run and is not
recoverable from the file, so these are all the rows of the pool -- which the models were
mostly trained on. Two models fitted to the same rows agree on them more than on rows
neither has seen, so every number here is a **lower bound** on the disagreement a book's
cells would meet. The AUC column is in-sample for the same reason and is a sanity check,
not a score; the comparable figure is in the training log.

**A lower number here is necessary, not sufficient.** Two models can agree closely and
both be wrong -- training against the search value (`--td-lambda`) risks exactly that,
because it replaces a noisy unbiased label with a quiet biased one. The held-out AUC
against the real outcome is the guard, and the board is the judge.

Calibrated against the books those same three seeds produced, the cheap number orders the
expensive one exactly:

    pair            cell spread   the book's advice costs
    gen8 ~ s1          0.1078            5.86 points
    gen8 ~ s2          0.1016            3.09
    s1   ~ s2          0.0902            2.88

Against a model's own spread of 0.162 to 0.183 at turn 1, a disagreement of 0.09 to 0.11
is noise of about sixty per cent of the signal.

    uv run --group learn python tools/cell_noise.py data/models/value-gen8{,-s1,-s2}.pt
    uv run --group learn python tools/cell_noise.py --data data/selfplay-pool8-encoded.npz \
        data/models/value-gen8.pt data/models/value-gen8-s1.pt
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.value import auc, load_dataset, load_model, predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/selfplay-pool8-encoded.npz"))
    ap.add_argument("--rows", type=int, default=60_000, help="rows sampled per turn bucket")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = load_dataset(args.data)
    import json

    meta = json.loads(str(np.load(args.data)["meta_json"]))
    reg = load_regulation(meta["format_id"])
    encoder = Encoder(reg)
    device = torch.device(args.device)

    nets = []
    for path in args.models:
        net, net_meta = load_model(path, encoder)
        nets.append((path.stem, net.to(device), net_meta))
        print(f"  {path.stem}: AUC {net_meta.get('val_auc', float('nan')):.4f}  "
              f"td_lambda {net_meta.get('td_lambda', 0.0)}  "
              f"epochs {net_meta.get('epochs_run', '?')}")

    turn = dataset.turn if hasattr(dataset, "turn") else np.load(args.data)["turn"]
    outcome = np.asarray(dataset.outcome, dtype=np.float64)
    rng = np.random.default_rng(args.seed)

    header = "\n   turn        n"
    for (left, _, _), (right, _, _) in itertools.combinations(nets, 2):
        header += f"   {left.replace('value-', '')}~{right.replace('value-', '')}"
    print(header)

    for bucket in (1, 2, 5, 10):
        keep = np.flatnonzero(turn == bucket)
        if len(keep) < 1000:
            continue
        if len(keep) > args.rows:
            keep = rng.choice(keep, args.rows, replace=False)
        values = {}
        for name, net, _ in nets:
            values[name] = 1.0 / (1.0 + np.exp(-predict(net, dataset, keep, device=device)
                                               .astype(np.float64)))
        line = f"  {bucket:>5} {len(keep):>8,}"
        for (left, _, _), (right, _, _) in itertools.combinations(nets, 2):
            diff = values[left] - values[right]
            line += f"   {np.sqrt((diff ** 2).mean()):.4f}"
        print(line)

    # The same rows, said three ways, so a systematic offset is not read as noise.
    keep = np.flatnonzero(turn == 1)
    if len(keep) > args.rows:
        keep = rng.choice(keep, args.rows, replace=False)
    values = {
        name: 1.0 / (1.0 + np.exp(-predict(net, dataset, keep, device=device).astype(np.float64)))
        for name, net, _ in nets
    }
    print("\n  turn 1 in detail")
    print(f"    {'pair':<34} {'spread':>8} {'worst':>8} {'|mean|':>8}")
    for (left, _, _), (right, _, _) in itertools.combinations(nets, 2):
        diff = values[left] - values[right]
        print(f"    {left + ' ~ ' + right:<34} {np.sqrt((diff ** 2).mean()):>8.4f} "
              f"{np.abs(diff).max():>8.4f} {abs(diff.mean()):>8.4f}")
    print(f"\n    {'model':<34} {'own spread':>8} {'AUC in-sample':>14}")
    for name, _, _ in nets:
        print(f"    {name:<34} {values[name].std():>8.4f} "
              f"{auc(values[name], outcome[keep]):>11.4f}")
    print("\n  Read the spread against the model's own spread on the line below: a pair\n"
          "  that disagrees by a third of the range it spans is not ranking the same\n"
          "  positions the same way, whatever their AUCs say separately.")


if __name__ == "__main__":
    main()
