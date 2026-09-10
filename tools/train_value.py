"""Trains the value function, and reports it against the proxy it has to beat.

The comparison is the point. `hp-share` already scores AUC 0.910 over all decisions, so a
learned function that reports 0.92 has done almost nothing -- the headroom is not spread
evenly. It sits at turn 1 (proxy AUC 0.769) and disappears by turn 13 (0.961), because
late in a game the remaining material *is* the answer. So every number here is broken out
by turn, and the proxy's number is printed beside the model's on the same rows and the
same validation games.

Calibration is reported next to discrimination. AUC only asks whether won positions score
above lost ones; this tool is supposed to print a win *probability*, so being right about
the level matters separately. The proxy is badly wrong there early on -- it says 0.51 at
turn 1 where the true rate is 72% -- and a model can fix that without ranking any better.

    uv run --group learn python tools/train_value.py --data data/selfplay-gen1-encoded.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import Encoder
from pokeuraou.regulation import load_regulation
from pokeuraou.value import (
    Dataset,
    ValueConfig,
    auc,
    build,
    load_dataset,
    predict,
    save_model,
    train,
)


def brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def logloss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def by_turn(
    dataset: Dataset,
    index: np.ndarray,
    model_p: np.ndarray,
    proxy: np.ndarray,
    minimum: int,
) -> None:
    turns = dataset.turn[index]
    labels = dataset.outcome[index]
    print(
        f"  {'turn':>5}  {'n':>7}  {'model AUC':>9}  {'proxy AUC':>9}  "
        f"{'model mean':>10}  {'proxy mean':>10}  {'actual':>7}"
    )
    for turn in sorted(set(turns.tolist())):
        pick = turns == turn
        if int(pick.sum()) < minimum:
            continue
        print(
            f"  {turn:>5}  {int(pick.sum()):>7}  {auc(model_p[pick], labels[pick]):>9.3f}  "
            f"{auc(proxy[pick], labels[pick]):>9.3f}  {model_p[pick].mean():>10.3f}  "
            f"{proxy[pick].mean():>10.3f}  {labels[pick].mean() * 100:>6.1f}%"
        )


def reliability(probabilities: np.ndarray, labels: np.ndarray, buckets: int) -> None:
    edges = np.quantile(probabilities, np.linspace(0, 1, buckets + 1))
    edges[-1] += 1e-9
    print(f"  {'predicted range':>22}  {'n':>7}  {'mean pred':>9}  actual")
    for low, high in zip(edges, edges[1:], strict=False):
        pick = (probabilities >= low) & (probabilities < high)
        if not pick.any():
            continue
        print(
            f"  {low:9.4f}..{high:<9.4f}  {int(pick.sum()):>7}  "
            f"{probabilities[pick].mean():>9.3f}  {labels[pick].mean() * 100:5.1f}%"
        )


def learning_curve(
    dataset: Dataset,
    encoder: Encoder,
    config: ValueConfig,
    device: torch.device,
    args,  # noqa: ANN001
    fractions: list[float],
) -> None:
    """Trains on a fraction of the *games*, always against the same validation games.

    Fractions are of games and not of decisions: dropping random decisions would leave
    most games partially present and each remaining decision would still be a near
    duplicate of one that was kept, so the curve would flatten for the wrong reason.
    """
    train_idx, val_idx = dataset.split_by_game(args.holdout, args.seed)
    train_games = np.unique(dataset.game[train_idx])
    rng = np.random.default_rng(args.seed)
    shuffled = train_games.copy()
    rng.shuffle(shuffled)
    labels = dataset.outcome[val_idx]
    proxy = dataset.proxy[val_idx]
    turn1 = dataset.turn[val_idx] <= 1

    print(
        f"\nlearning curve on {len(train_games):,} training games, "
        f"validated on the same {len(val_idx):,} held-out decisions throughout"
    )
    print(
        f"  {'games':>7}  {'decisions':>10}  {'AUC':>7}  {'log loss':>9}  "
        f"{'Brier':>7}  {'turn-1 AUC':>10}  {'epochs':>6}"
    )
    for fraction in fractions:
        keep = set(shuffled[: max(1, int(len(shuffled) * fraction))].tolist())
        subset = np.array([g in keep for g in dataset.game])
        # A Dataset view whose "training" games are the subset and whose validation games
        # are untouched: mark everything else as a game the splitter will not select.
        view = Dataset(
            encoded=dataset.encoded,
            outcome=dataset.outcome,
            game=np.where(subset | np.isin(np.arange(len(dataset)), val_idx), dataset.game, -1),
            turn=dataset.turn,
            proxy=dataset.proxy,
            kind=dataset.kind,
            foe=dataset.foe,
            foe_names=dataset.foe_names,
        )
        use = np.flatnonzero(subset)
        net = build(encoder, config).to(device)
        history, best = train(
            net,
            view,
            config,
            device=device,
            holdout=args.holdout,
            log=None,
            train_index=use,
            val_index=val_idx,
        )
        net.load_state_dict(best)
        p = 1.0 / (1.0 + np.exp(-predict(net, dataset, val_idx, device=device)))
        print(
            f"  {len(keep):>7}  {len(use):>10}  {auc(p, labels):>7.4f}  "
            f"{logloss(p, labels):>9.4f}  {brier(p, labels):>7.4f}  "
            f"{auc(p[turn1], labels[turn1]):>10.4f}  {len(history):>6}"
        )
    print(
        f"  {'proxy':>7}  {'--':>10}  {auc(proxy, labels):>7.4f}  "
        f"{logloss(proxy, labels):>9.4f}  {brier(proxy, labels):>7.4f}  "
        f"{auc(proxy[turn1], labels[turn1]):>10.4f}  {'--':>6}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/selfplay-gen1-encoded.npz"))
    ap.add_argument("--out", type=Path, default=Path("data/models/value-gen1.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-turn-rows", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument(
        "--curve",
        default="",
        help="comma-separated fractions of the training games, e.g. 0.125,0.25,0.5,1.0. "
        "Trains once per fraction against the same held-out games, which is what says "
        "whether more self-play would still help.",
    )
    args = ap.parse_args()

    dataset = load_dataset(args.data)
    reg = load_regulation(
        __import__("json").loads(str(np.load(args.data)["meta_json"]))["format_id"]
    )
    encoder = Encoder(reg)
    config = ValueConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed
    )
    device = torch.device(args.device)
    net = build(encoder, config).to(device)
    parameters = sum(p.numel() for p in net.parameters())

    train_idx, val_idx = dataset.split_by_game(args.holdout, args.seed)
    print(
        f"{len(dataset):,} decisions from {len(np.unique(dataset.game)):,} games "
        f"-> {len(train_idx):,} train / {len(val_idx):,} validation, split by game"
    )
    print(f"{parameters:,} parameters on {device}")

    # The identity the architecture is supposed to guarantee, checked before training so a
    # failure is a bug in the model rather than a training artefact. Eval mode matters:
    # dropout would otherwise apply a different mask to each of the two orderings and the
    # exact identity would look like a 0.3 error.
    check = dataset.tensors(val_idx[:256], device)
    net.eval()
    with torch.no_grad():
        forward = torch.sigmoid(net(check))
        flipped = {
            k: (v.flip(1) if v.dim() > 1 and v.shape[1] == 2 else v)
            for k, v in check.items()
        }
        mirrored = torch.sigmoid(net(flipped))
    worst = float((forward + mirrored - 1.0).abs().max())
    print(f"antisymmetry V(x) + V(mirror x) - 1: max |error| {worst:.2e} at init")

    if args.curve:
        learning_curve(
            dataset, encoder, config, device, args, [float(x) for x in args.curve.split(",")]
        )
        return

    def log(report) -> None:  # noqa: ANN001
        print(
            f"  epoch {report.epoch:>3}  train {report.train_loss:.4f}  "
            f"val {report.val_loss:.4f}  val AUC {report.val_auc:.4f}  "
            f"{report.seconds:.1f}s"
        )

    print("\ntraining")
    history, best = train(
        net, dataset, config, device=device, holdout=args.holdout, log=log
    )
    net.load_state_dict(best)

    logit = predict(net, dataset, val_idx, device=device)
    model_p = 1.0 / (1.0 + np.exp(-logit))
    proxy = dataset.proxy[val_idx]
    labels = dataset.outcome[val_idx]

    net.eval()
    with torch.no_grad():
        forward = torch.sigmoid(net(check))
        mirrored = torch.sigmoid(net(flipped))
    print(
        "\nantisymmetry after training: max |error| "
        f"{float((forward + mirrored - 1.0).abs().max()):.2e}"
    )

    print("\nheld-out games, model versus the proxy it has to beat")
    print(f"  {'':>18}  {'AUC':>7}  {'log loss':>9}  {'Brier':>7}")
    for name, score in (("learned value", model_p), ("hp-share proxy", proxy)):
        print(
            f"  {name:>18}  {auc(score, labels):>7.4f}  {logloss(score, labels):>9.4f}  "
            f"{brier(score, labels):>7.4f}"
        )
    base = float(labels.mean())
    print(
        f"  {'always ' + format(base, '.3f'):>18}  {'--':>7}  "
        f"{logloss(np.full_like(labels, base), labels):>9.4f}  "
        f"{brier(np.full_like(labels, base), labels):>7.4f}"
    )

    print("\nby turn (the aggregate hides this)")
    by_turn(dataset, val_idx, model_p, proxy, args.min_turn_rows)

    print("\nis the number a probability? learned value")
    reliability(model_p, labels, 8)
    print("\nthe same for the proxy")
    reliability(proxy, labels, 8)

    move = dataset.kind[val_idx] == 0
    for name, pick in (("move nodes", move), ("replacement nodes", ~move)):
        if pick.sum() > 50:
            print(
                f"\n{name}: n {int(pick.sum()):,}  model AUC "
                f"{auc(model_p[pick], labels[pick]):.4f}  proxy AUC "
                f"{auc(proxy[pick], labels[pick]):.4f}"
            )

    if not args.no_save:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_model(
            args.out,
            net,
            best,
            encoder.vocab,
            config,
            meta={
                "data": str(args.data),
                "decisions": len(dataset),
                "games": int(len(np.unique(dataset.game))),
                "val_auc": auc(model_p, labels),
                "val_logloss": logloss(model_p, labels),
                "proxy_auc": auc(proxy, labels),
                "proxy_logloss": logloss(proxy, labels),
                "epochs_run": len(history),
            },
        )
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
