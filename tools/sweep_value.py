"""Sweeps value-function settings, because training is free and generation is not.

Measured on this dataset: generating the games cost 11.0 core-hours, training on them costs
11 seconds. A factor of 3,590. That asymmetry says two things, and the second one is the
reason this file exists:

- effort spent making the *model* faster is wasted; the bottleneck is entirely generation
  (and inside generation, 76% of a node is the resolver);
- effort spent *searching over training settings* is nearly free, so leaving the defaults
  unexamined is the one clearly avoidable mistake.

The axes are not a blind grid. The trained model overfits visibly -- validation loss bottoms
at epoch 4 and then climbs while training loss keeps falling -- so the configurations vary
what that symptom implicates: capacity, dropout, weight decay. Learning rate is included
because it interacts with early stopping.

Every configuration is judged on the same held-out *games* and by validation log loss, not
AUC: the tool has to print a probability, so being right about the level is the criterion.

    uv run --group learn python tools/sweep_value.py --data data/selfplay-worlds-encoded.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import Encoder
from pokeuraou.regulation import load_regulation
from pokeuraou.value import ValueConfig, auc, build, load_dataset, predict, train


def logloss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def variants(base: ValueConfig) -> list[tuple[str, ValueConfig]]:
    """Named configurations, one axis at a time so a result is attributable."""
    out: list[tuple[str, ValueConfig]] = [("default", base)]

    # Capacity. Overfitting at 406k parameters could mean too many; it could also mean too
    # few with the wrong regularisation, so both directions are tried.
    for name, scale in (("half", 0.5), ("double", 2.0)):
        out.append(
            (
                f"capacity {name}",
                replace(
                    base,
                    species_dim=int(base.species_dim * scale),
                    ability_dim=int(base.ability_dim * scale),
                    item_dim=int(base.item_dim * scale),
                    move_dim=int(base.move_dim * scale),
                    mon_dim=int(base.mon_dim * scale),
                    side_dim=int(base.side_dim * scale),
                    head_dim=int(base.head_dim * scale),
                ),
            )
        )

    # The two knobs the overfitting symptom points at directly.
    for dropout in (0.0, 0.2, 0.3, 0.4):
        out.append((f"dropout {dropout}", replace(base, dropout=dropout)))
    for decay in (1e-3, 3e-2, 1e-1):
        out.append((f"weight_decay {decay:g}", replace(base, weight_decay=decay)))
    for lr in (1e-3, 4e-3):
        out.append((f"lr {lr:g}", replace(base, lr=lr)))
    for batch in (512, 2048):
        out.append((f"batch {batch}", replace(base, batch_size=batch)))

    # Patience interacts with everything above: a run stopped at epoch 4 never sees whether
    # the climb was a bump.
    out.append(("patience 8", replace(base, patience=8, epochs=40)))

    # Combinations of the axes that survived three seeds. Kept separate from the
    # one-at-a-time block so a combined win cannot be mistaken for an attributable one.
    out.append(("dropout 0.5", replace(base, dropout=0.5)))
    out.append(("d0.4 + lr4e-3", replace(base, dropout=0.4, lr=4e-3)))
    out.append(("d0.5 + lr4e-3", replace(base, dropout=0.5, lr=4e-3)))
    out.append(
        (
            "d0.4 + half",
            replace(
                base,
                dropout=0.4,
                species_dim=24,
                ability_dim=12,
                item_dim=12,
                move_dim=16,
                mon_dim=80,
                side_dim=96,
                head_dim=128,
            ),
        )
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/selfplay-worlds-encoded.npz"))
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="repeat each config with this many seeds")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated substrings; run only the configurations whose name matches",
    )
    args = ap.parse_args()

    dataset = load_dataset(args.data)
    meta = json.loads(str(np.load(args.data)["meta_json"]))
    encoder = Encoder(load_regulation(meta["format_id"]))
    device = torch.device(args.device)

    # One split, shared by every configuration, so differences are the configuration.
    train_idx, val_idx = dataset.split_by_game(args.holdout, args.seed)
    labels = dataset.outcome[val_idx]
    proxy = dataset.proxy[val_idx]
    turn1 = dataset.turn[val_idx] <= 1

    base = ValueConfig(epochs=args.epochs, seed=args.seed)
    print(
        f"{len(dataset):,} decisions, {len(train_idx):,} train / {len(val_idx):,} "
        f"validation (split by game, shared across configurations)"
    )
    print(
        f"  proxy baseline: log loss {logloss(proxy, labels):.4f}, "
        f"AUC {auc(proxy, labels):.4f}, turn-1 AUC {auc(proxy[turn1], labels[turn1]):.4f}"
    )
    print(
        f"\n  {'configuration':>20}  {'params':>9}  {'log loss':>9}  {'AUC':>7}  "
        f"{'turn-1':>7}  {'epochs':>6}  {'s':>5}"
    )

    wanted = [w.strip() for w in args.only.split(",") if w.strip()]
    rows = []
    for name, config in variants(base):
        if wanted and not any(w in name for w in wanted):
            continue
        losses, aucs, t1s, epochs_run = [], [], [], []
        started = time.perf_counter()
        params = 0
        for offset in range(args.seeds):
            cfg = replace(config, seed=args.seed + offset)
            net = build(encoder, cfg).to(device)
            params = sum(p.numel() for p in net.parameters())
            history, best = train(
                net,
                dataset,
                cfg,
                device=device,
                log=None,
                train_index=train_idx,
                val_index=val_idx,
            )
            net.load_state_dict(best)
            p = 1.0 / (1.0 + np.exp(-predict(net, dataset, val_idx, device=device)))
            losses.append(logloss(p, labels))
            aucs.append(auc(p, labels))
            t1s.append(auc(p[turn1], labels[turn1]))
            epochs_run.append(len(history))
        elapsed = time.perf_counter() - started
        rows.append((float(np.mean(losses)), name, float(np.mean(aucs)), float(np.mean(t1s))))
        spread = f" +-{np.std(losses):.3f}" if args.seeds > 1 else ""
        print(
            f"  {name:>20}  {params:>9,}  {np.mean(losses):>9.4f}{spread}  "
            f"{np.mean(aucs):>7.4f}  {np.mean(t1s):>7.4f}  "
            f"{np.mean(epochs_run):>6.1f}  {elapsed:>5.0f}"
        )

    rows.sort()
    print("\nbest by validation log loss (the criterion, because the output is a probability)")
    for loss, name, score, t1 in rows[:5]:
        print(f"  {name:>20}  log loss {loss:.4f}  AUC {score:.4f}  turn-1 {t1:.4f}")
    print(
        "\nA difference smaller than the seed-to-seed spread is not a result; rerun with "
        "--seeds 3 before changing a default on the strength of one row."
    )


if __name__ == "__main__":
    main()
