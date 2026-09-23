"""Trains the value function, and reports it against the two baselines that matter.

The comparison is the point, and *which* comparison is the point is worth stating
carefully, because getting it wrong here is cheap and costly at once.

**hp-share** is the parameter-free objective. Beating it is what justifies having a
learned value function at all. On generation 2-4 data it scores AUC 0.828 over all
decisions and 0.544 at turn 1, where material is nearly uninformative -- the headroom is
not spread evenly, and it disappears by turn 10 (0.907), because late in a game the
remaining material *is* the answer.

**The generating search** is the value the search that produced the data reported: the
previous generation's model backed by a full 24x24 one-ply equilibrium. It scores 0.883.
A freshly trained *raw* evaluation sitting a little below that is the expected state of
affairs and not a defect -- the gap is what one ply of search is worth, which is the
whole reason the solver does any.

Those two were printed under one name for three generations. The array had been called
`proxy` when generation was driven by `hp-share` and the two really were the same number;
generation moved to a learned leaf and the name did not follow. Read as hp-share, the
number says a learned value function trained on 20,613 games cannot beat a material
heuristic, which is alarming and false.

Calibration is reported next to discrimination. AUC only asks whether won positions score
above lost ones; this tool is supposed to print a win *probability*, so being right about
the level matters separately.

    uv run --group learn python tools/train_value.py --data data/selfplay-gen1-encoded.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import ENCODING_REVISION, Encoder
from pokeuraou.regulation import load_regulation
from pokeuraou.value import (
    Dataset,
    ValueConfig,
    auc,
    build,
    load_dataset,
    predict,
    save_model,
    split_for,
    td_target,
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
    searched: np.ndarray,
    minimum: int,
) -> None:
    turns = dataset.turn[index]
    labels = dataset.outcome[index]
    print(
        f"  {'turn':>5}  {'n':>7}  {'model AUC':>9}  {'search AUC':>10}  "
        f"{'model mean':>10}  {'search mean':>11}  {'actual':>7}"
    )
    for turn in sorted(set(turns.tolist())):
        pick = turns == turn
        if int(pick.sum()) < minimum:
            continue
        print(
            f"  {turn:>5}  {int(pick.sum()):>7}  {auc(model_p[pick], labels[pick]):>9.3f}  "
            f"{auc(searched[pick], labels[pick]):>10.3f}  {model_p[pick].mean():>10.3f}  "
            f"{searched[pick].mean():>11.3f}  {labels[pick].mean() * 100:>6.1f}%"
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
    train_idx, val_idx = split_for(dataset, args.holdout, config)
    train_games = np.unique(dataset.game[train_idx])
    rng = np.random.default_rng(args.seed)
    shuffled = train_games.copy()
    rng.shuffle(shuffled)
    labels = dataset.outcome[val_idx]
    searched = dataset.search_value[val_idx]
    hp_share = (
        dataset.hp_share[val_idx] if dataset.hp_share.size else np.zeros(0, np.float32)
    )
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
            search_value=dataset.search_value,
            hp_share=dataset.hp_share,
            kind=dataset.kind,
            foe=dataset.foe,
            foe_names=dataset.foe_names,
            foe_search_value=dataset.foe_search_value,
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
        f"  {'gen search':>7}  {'--':>10}  {auc(searched, labels):>7.4f}  "
        f"{logloss(searched, labels):>9.4f}  {brier(searched, labels):>7.4f}  "
        f"{auc(searched[turn1], labels[turn1]):>10.4f}  {'--':>6}"
    )
    if hp_share.size:
        print(
            f"  {'hp-share':>7}  {'--':>10}  {auc(hp_share, labels):>7.4f}  "
            f"{logloss(hp_share, labels):>9.4f}  {brier(hp_share, labels):>7.4f}  "
            f"{auc(hp_share[turn1], labels[turn1]):>10.4f}  {'--':>6}"
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
    ap.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="which games are held out, separately from --seed. --seed moves three "
        "things at once -- initialisation, batch order and the held-out games -- so "
        "two runs that differ by it were also marked against different games, which "
        "changes what the validation number MEANS rather than adding noise to it. "
        "Left alone it is 0, so varying --seed now varies the fit and holds the "
        "marking fixed. That is a change only for a run that passes --seed, which "
        "before this default would have moved the split with it; no invocation of "
        "this tool in the record passes one. It is written into the model either way.",
    )
    ap.add_argument("--min-turn-rows", type=int, default=200)
    ap.add_argument(
        "--td-lambda",
        type=float,
        default=0.0,
        help="fit (1-L)*outcome + L*searchValue instead of the outcome alone. The outcome "
        "is the truth but a very noisy sample of it -- 13 decisions of a game share one "
        "label, so the independent information in a pool is its game count. The "
        "generating search's value is biased and not noisy, and on the same validation "
        "games it beats the raw network. Validation always stays against the real "
        "outcome, whatever this is set to, or the rows stop being comparable.",
    )
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

    # A model is trained on the arrays in the file and then searched with the encoder in
    # this tree. If a column changed meaning between the two, the model learns one feature
    # and is asked about another, and nothing downstream can see it (IKA-121: `can_mega`
    # stood on the wrong Pokemon in 24.9% of unspent-mega pairs until 9/23). A file from
    # before the revision was recorded is revision 1.
    revision = json.loads(str(np.load(args.data)["meta_json"])).get("encoding_revision", 1)
    if revision != ENCODING_REVISION:
        raise SystemExit(
            f"{args.data} was encoded at revision {revision} and this tree encodes at "
            f"{ENCODING_REVISION}; re-run tools/encode_dataset.py, which rebuilds the stale "
            "shards itself"
        )
    dataset = load_dataset(args.data)
    target = None
    if args.td_lambda:
        meta = __import__("json").loads(str(np.load(args.data)["meta_json"]))
        leaves = meta.get("objectives") or {}
        # `hp-share` and a learned leaf both land in [0, 1] and mean different things, so
        # a pool that mixes them cannot have its search values folded into one target.
        # Shards encoded before the leaf was recorded say nothing, and silence is not
        # permission.
        if not leaves:
            raise SystemExit(
                "this dataset does not record which leaf generated it; re-encode it "
                "before using --td-lambda"
            )
        wrong = sorted(k for k in leaves if not k.startswith("value:"))
        if wrong:
            raise SystemExit(
                f"--td-lambda needs searchValue to be a win probability, but {wrong} "
                "generated part of this pool"
            )
        target = td_target(dataset, args.td_lambda)
        print(
            f"TD target: {1 - args.td_lambda:.2f} x outcome + {args.td_lambda:.2f} x "
            f"searchValue (leaves {sorted(leaves)}). Validation stays on the outcome."
        )
    reg = load_regulation(
        __import__("json").loads(str(np.load(args.data)["meta_json"]))["format_id"]
    )
    encoder = Encoder(reg)
    config = ValueConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        split_seed=args.split_seed,
    )
    device = torch.device(args.device)
    net = build(encoder, config).to(device)
    parameters = sum(p.numel() for p in net.parameters())

    train_idx, val_idx = split_for(dataset, args.holdout, config)
    print(
        f"{len(dataset):,} decisions from {len(np.unique(dataset.game)):,} games "
        f"-> {len(train_idx):,} train / {len(val_idx):,} validation, split by game "
        f"at split-seed {args.split_seed} (fit seed {args.seed})"
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
        if args.td_lambda:
            # Rather than quietly train the curve on a different target than the banner
            # said. The curve answers "would more games help", which is a question about
            # the outcome label; mixing in the search value changes what "more games"
            # buys, so the two experiments do not belong in one run.
            raise SystemExit("--curve and --td-lambda answer different questions; run them apart")
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
        net, dataset, config, device=device, holdout=args.holdout, log=log, target=target
    )
    net.load_state_dict(best)

    logit = predict(net, dataset, val_idx, device=device)
    model_p = 1.0 / (1.0 + np.exp(-logit))
    searched = dataset.search_value[val_idx]
    hp_share = (
        dataset.hp_share[val_idx] if dataset.hp_share.size else np.zeros(0, np.float32)
    )
    labels = dataset.outcome[val_idx]

    net.eval()
    with torch.no_grad():
        forward = torch.sigmoid(net(check))
        mirrored = torch.sigmoid(net(flipped))
    print(
        "\nantisymmetry after training: max |error| "
        f"{float((forward + mirrored - 1.0).abs().max()):.2e}"
    )

    # Two baselines, because they answer different questions and one of them used to be
    # printed under the other's name. `hp-share` is the parameter-free objective: beating
    # it is what justifies having a learned value function at all. `generating search` is
    # the value the search that *produced* the data reported -- the previous model backed
    # by a full one-ply equilibrium -- so a freshly trained raw evaluation sitting
    # slightly below it is the expected state of affairs and not a defect. Reading the
    # second under the first's name once produced the conclusion that 20,613 games had
    # bought nothing.
    print("\nheld-out games, model versus the baselines")
    print(f"  {'':>18}  {'AUC':>7}  {'log loss':>9}  {'Brier':>7}")
    rows = [("learned value", model_p), ("generating search", searched)]
    if hp_share.size:
        rows.append(("hp-share", hp_share))
    for name, score in rows:
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
    by_turn(dataset, val_idx, model_p, searched, args.min_turn_rows)

    print("\nis the number a probability? learned value")
    reliability(model_p, labels, 8)
    print("\nthe same for the generating search")
    reliability(searched, labels, 8)

    move = dataset.kind[val_idx] == 0
    for name, pick in (("move nodes", move), ("replacement nodes", ~move)):
        if pick.sum() > 50:
            print(
                f"\n{name}: n {int(pick.sum()):,}  model AUC "
                f"{auc(model_p[pick], labels[pick]):.4f}  generating-search AUC "
                f"{auc(searched[pick], labels[pick]):.4f}"
            )

    if not args.no_save:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_model(
            args.out,
            net,
            best,
            encoder.vocab,
            config,
            widths=encoder.widths,
            meta={
                "data": str(args.data),
                "decisions": len(dataset),
                "games": int(len(np.unique(dataset.game))),
                "val_auc": auc(model_p, labels),
                "val_logloss": logloss(model_p, labels),
                "search_auc": auc(searched, labels),
                "search_logloss": logloss(searched, labels),
                **(
                    {"hp_share_auc": auc(hp_share, labels)} if hp_share.size else {}
                ),
                "epochs_run": len(history),
                "td_lambda": args.td_lambda,
                "seed": args.seed,
                # Separately, because --seed moves three things and only this one
                # decides what the row above was marked against. A model whose record
                # gives one number for both cannot say whether a rival's better AUC came
                # from a better fit or from easier games.
                "split_seed": args.split_seed,
                "holdout": args.holdout,
                "encoding_revision": ENCODING_REVISION,
                # What the games that taught this could see. A value trained on omniscient
                # play predicts omniscient play, and the book prints that prediction into
                # a condition where the bench is hidden: on place 109 the leaf claimed
                # 62.8%, open play returned 59.2% and hidden play 50.0%. Carried from the
                # dataset so the claim can say which agent it is about.
                "information": json.loads(
                    str(np.load(args.data)["meta_json"])
                ).get("information", {}),
            },
        )
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
