"""Learning an ordering from the equilibria already recorded.

The menu's job is to contain the support, and `tools/policy_ceiling.py` prices the gap: a
perfect ordering is free at width 6, the leaf ordering needs 48 to get there, and 36 cells
against 2,304 is what a good ordering is worth. This trains one on the recorded mixed
strategies and reports the only number that matters for that -- how much of the
equilibrium's weight the ordering puts in its top few.

A menu is one training example, not a row: the target is a distribution over the actions
offered, so the loss is cross-entropy against that distribution, and the model scores every
candidate in one pass the way `narrow` will have to.

The baseline to beat is the leaf ordering, which is what gen-7 was generated with, so the
recorded menu order *is* the baseline and needs no recomputation -- the `ranks` column is
where each action sat in it.

    uv run --group learn python tools/policy_train.py --data data/policy-gen7.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def top_weight(order: np.ndarray, target: np.ndarray, k: int) -> float:
    """Share of the equilibrium's weight sitting in the first `k` of an ordering."""
    keep = order[:k]
    return float(target[keep].sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="where to save the model")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256, help="menus per step")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import torch
    from torch import nn

    blob = np.load(args.data)
    features, targets = blob["features"], blob["targets"]
    ranks, lengths = blob["ranks"], blob["lengths"]
    starts = np.concatenate([[0], np.cumsum(lengths)])
    menus = len(lengths)
    print(f"{menus:,} menus, {len(features):,} rows, {features.shape[1]} features")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(menus)
    cut = int(menus * (1 - args.holdout))
    train_ids, test_ids = order[:cut], order[cut:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(features.shape[1], args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, args.hidden),
        nn.ReLU(),
        nn.Linear(args.hidden, 1),
    ).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    def batch_of(ids: np.ndarray):
        """Menus padded to the longest, with a mask, so one pass covers the batch."""
        widest = int(lengths[ids].max())
        x = np.zeros((len(ids), widest, features.shape[1]), dtype=np.float32)
        y = np.zeros((len(ids), widest), dtype=np.float32)
        mask = np.zeros((len(ids), widest), dtype=bool)
        for row, menu in enumerate(ids):
            start, stop = starts[menu], starts[menu] + lengths[menu]
            n = lengths[menu]
            x[row, :n] = features[start:stop]
            y[row, :n] = targets[start:stop]
            mask[row, :n] = True
        return (
            torch.from_numpy(x).to(device),
            torch.from_numpy(y).to(device),
            torch.from_numpy(mask).to(device),
        )

    def loss_for(x, y, mask):
        logits = model(x).squeeze(-1).masked_fill(~mask, -1e9)
        return -(y * torch.log_softmax(logits, dim=-1).clamp_min(-30)).sum(-1).mean()

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(train_ids)
        total, seen = 0.0, 0
        for start in range(0, len(shuffled), args.batch):
            ids = shuffled[start : start + args.batch]
            x, y, mask = batch_of(ids)
            loss = loss_for(x, y, mask)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss) * len(ids)
            seen += len(ids)
        model.eval()
        with torch.no_grad():
            x, y, mask = batch_of(test_ids[:2048])
            held = float(loss_for(x, y, mask))
        print(f"  epoch {epoch:>2}  train {total / seen:.4f}  holdout {held:.4f}",
              flush=True)

    # The verdict: where the equilibrium's weight lands under each ordering, on menus the
    # model has not seen. The leaf ordering is the recorded menu order itself.
    widths = [1, 2, 4, 6, 8, 12, 16, 24]
    got = {"leaf": {k: [] for k in widths}, "model": {k: [] for k in widths}}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(test_ids), 512):
            ids = test_ids[start : start + 512]
            x, _y, mask = batch_of(ids)
            scores = model(x).squeeze(-1).masked_fill(~mask, -1e9).cpu().numpy()
            for row, menu in enumerate(ids):
                begin, stop = starts[menu], starts[menu] + lengths[menu]
                target = targets[begin:stop]
                if target.sum() <= 0:
                    continue
                target = target / target.sum()
                by_leaf = np.argsort(ranks[begin:stop])
                by_model = np.argsort(-scores[row, : lengths[menu]])
                for k in widths:
                    got["leaf"][k].append(top_weight(by_leaf, target, k))
                    got["model"][k].append(top_weight(by_model, target, k))

    print(f"\nheld-out menus: {len(got['leaf'][1]):,}")
    print(f"  {'top':>4} {'leaf ordering':>14} {'learned':>10} {'gain':>8}")
    for k in widths:
        leaf = float(np.mean(got["leaf"][k]))
        learned = float(np.mean(got["model"][k]))
        print(f"  {k:>4} {leaf:>13.1%} {learned:>10.1%} {learned - leaf:>+8.1%}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "features": features.shape[1],
                    "hidden": args.hidden}, args.out)
        print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
