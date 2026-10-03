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
from pokeuraou.slotswap import SwapSlots, position_reader_rows
from pokeuraou.value import (
    Dataset,
    ValueConfig,
    auc,
    build,
    final_material_targets,
    game_weights,
    load_dataset,
    load_model,
    predict,
    save_model,
    split_for,
    td_target,
    train,
    widen_net,
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


def warm_start(
    path: Path, encoder: Encoder, run: dict
) -> tuple[torch.nn.Module, dict, ValueConfig]:
    """The model at `path`, read for `encoder`, and the config to go on training it with.

    Through `load_model` and nothing else, so the vocabulary is grown -- zero rows for the
    ids the encoder appended, an M-B model onto M-C -- by the one path whose null control
    is on record (IKA-82: bit-identical on 4,000 M-B positions), and a vocabulary that is
    not an extension is refused the same way. The architecture and regularisation come
    from the model (its sizes must match its weights); what this run sets -- epochs,
    learning rate, seeds, schedule -- replaces the rest.
    """
    from dataclasses import replace

    net, meta = load_model(path, encoder)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    config = replace(ValueConfig(**blob["config"]), **run)
    if config.attention and not net.config.attention:
        # IKA-90: the attention layer's output projection starts at zero, so before the
        # first step this net answers exactly as the loaded one.
        torch.manual_seed(config.seed)
        grown = build(encoder, config)
        grown._active_feature = net._active_feature
        missing, unexpected = grown.load_state_dict(net.state_dict(), strict=False)
        if unexpected or any(not k.startswith("mon_attention.") for k in missing):
            raise SystemExit(f"warm start into attention: missing {missing}, unexpected {unexpected}")
        net = grown
    if config.move_properties and not net.config.move_properties:
        # IKA-318: an id-only model gains the move-property branch. Its input columns start
        # at zero, so before the first step this net answers exactly as the loaded one; the
        # branch's own layer is initialised from --seed, not from whatever ran before.
        torch.manual_seed(config.seed)
        grown = build(encoder, config)
        grown._active_feature = net._active_feature
        missing, unexpected = grown.load_state_dict(net.state_dict(), strict=False)
        new = {"move_props", *(k for k in grown.state_dict() if k.startswith("move_prop_"))}
        if unexpected or set(missing) != new:
            raise SystemExit(f"warm start into move properties: missing {missing}, unexpected {unexpected}")
        net = grown
    if config.aux_weight > 0 and not hasattr(net, "aux"):
        # IKA-425: the auxiliary regression head is new. The win logit does not read it, so
        # before the first step this net answers exactly as the loaded one; the head's own
        # initialisation is drawn from --seed.
        torch.manual_seed(config.seed)
        grown = build(encoder, config)
        grown._active_feature = net._active_feature
        missing, unexpected = grown.load_state_dict(net.state_dict(), strict=False)
        if unexpected or any(not k.startswith("aux.") for k in missing):
            raise SystemExit(f"warm start into the aux head: missing {missing}, unexpected {unexpected}")
        net = grown
    if (config.dropout, config.encoder_dropout) != (net.config.dropout, net.config.encoder_dropout):
        # IKA-425: the dropout of the model read is replaced by this run's. The head's
        # Dropout modules are built from the config, so set them as well as the record.
        net.config = replace(
            net.config, dropout=config.dropout, encoder_dropout=config.encoder_dropout
        )
        for module in net.head:
            if isinstance(module, torch.nn.Dropout):
                module.p = config.dropout
    record = {
        "path": str(path),
        "format_id": blob["format_id"],
        "vocab_fingerprint": blob["vocab_fingerprint"],
        "vocab_grown_from": meta.get("vocab_grown_from"),
        "vocab_extended_from": meta.get("vocab_extended_from"),
        "data": meta.get("data"),
        "epochs_run": meta.get("epochs_run"),
    }
    return net, record, config


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
    ap.add_argument(
        "--td-from-game",
        type=int,
        default=None,
        help="with --td-lambda: rows of games numbered below this keep the outcome alone, "
        "the mix applies from this game on (IKA-403). In a pool of several generations "
        "the first shard's teacher (value-gen11L) is worse than the student; the later "
        "ones are better.",
    )
    ap.add_argument(
        "--target-file",
        type=Path,
        default=None,
        help="fit the per-row targets in this .npz (key --target-key) instead of the outcome, "
        "e.g. `tools/deep_targets.py mix` (IKA-296). One value in [0, 1] per row of --data, in "
        "its order. Validation stays against the real outcome.",
    )
    ap.add_argument("--target-key", default=None)
    ap.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="start from this model's weights instead of a fresh initialisation (IKA-194). "
        "Read through load_model, so an M-B model starts an M-C run with zero rows for "
        "the ids M-C appended, and the architecture (sizes, dropout, weight decay) is the "
        "model's. --epochs 0 saves it unchanged: the null control.",
    )
    ap.add_argument("--keep", choices=("best", "last"), default=None,
                    help="default: ValueConfig's. best = early stopping at the best epoch; "
                    "last = the whole schedule, final or averaged weights (IKA-86)")
    ap.add_argument("--average", choices=("none", "ema", "swa"), default=None)
    ap.add_argument("--ema-decay", type=float, default=None)
    ap.add_argument("--swa-from", type=float, default=None)
    ap.add_argument("--pct-start", type=float, default=None)
    ap.add_argument(
        "--width-scale",
        type=float,
        default=1.0,
        help="IKA-398: multiply every embedding and hidden width by this (species 48, "
        "ability 24, item 24, move 32, mon 160, side 192, head 256). Fresh initialisation "
        "only: a warm start needs the source model's shapes.",
    )
    ap.add_argument(
        "--widen",
        type=int,
        default=1,
        help="IKA-405: with --init-from, grow the hidden layers (mon 160, side 192, head 256) "
        "to this many times their width before training, in the form that leaves the model's "
        "answers unchanged (value.widen_net). 1 (default) does not touch the shapes.",
    )
    ap.add_argument(
        "--move-properties",
        action="store_true",
        help="IKA-318: read each move's dex properties (qhead.move_table) beside its id "
        "embedding. With --init-from an id-only model, the new branch's input columns start "
        "at zero, so the first step starts from that model's own answers.",
    )
    ap.add_argument(
        "--attention",
        action="store_true",
        help="IKA-90: one residual attention layer across the Pokemon tokens of both sides. "
        "With --init-from a model without it, its output projection starts at zero, so the "
        "first step starts from that model's own answers.",
    )
    ap.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="IKA-425: dropout of the head (a warm start otherwise keeps the model's, 0.4)",
    )
    ap.add_argument(
        "--encoder-dropout",
        type=float,
        default=None,
        help="IKA-425: dropout on the per-Pokemon and per-side vectors, training only "
        "(default 0: none)",
    )
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="IKA-425: AdamW weight decay (a warm start otherwise keeps the model's, 0.01)",
    )
    ap.add_argument(
        "--aux-weight",
        type=float,
        default=0.0,
        help="IKA-425: add this x the mean squared error of an auxiliary head that regresses "
        "the end-of-game material difference (value.final_material_targets: alive fraction "
        "and team HP fraction of the game's last recorded position, side 0 minus side 1). "
        "The saved model drops the head.",
    )
    ap.add_argument(
        "--game-weights",
        default="",
        help='IKA-425: per-row weights of the cross entropy by game number, "FROM:W,FROM:W" '
        '(e.g. "0:0.5,79997:1,679990:2" weights games from 0 by 0.5, from 79997 by 1, from '
        "679990 by 2; mc01234's games start at 0, 39998, 79997, 279997, 679990). Off: every "
        "row counts the same, as in every run before.",
    )
    ap.add_argument(
        "--drop-train-moves",
        default="",
        help="comma-separated move ids: leave every training decision whose position holds "
        "one of them out of the fit, and keep the validation games as they are -- a move "
        "the net has never been taught, on the same marking (IKA-318)",
    )
    ap.add_argument(
        "--swap-slots",
        action="store_true",
        help="show every training row in a random arrangement of each side's left and right "
        "Pokemon, drawn afresh each epoch (IKA-412). Rows holding a Pokemon whose ability "
        "reads its position (Imposter) are never exchanged. Validation is never exchanged.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument(
        "--unpacked",
        action="store_true",
        help="read the dataset into plain int64/float32 arrays (29 GB for mc0123) instead of "
        "the default lossless packed form (about a third of that, same batches bit for "
        "bit; pokeuraou.packed). For A/B of the packing only.",
    )
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
    dataset = load_dataset(args.data, packed=not args.unpacked)
    target = None
    if args.td_from_game is not None and not args.td_lambda:
        raise SystemExit("--td-from-game needs --td-lambda")
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
        target = td_target(dataset, args.td_lambda, from_game=args.td_from_game)
        moved = target != dataset.outcome
        print(
            f"TD target: {1 - args.td_lambda:.2f} x outcome + {args.td_lambda:.2f} x "
            f"searchValue (leaves {sorted(leaves)}). Validation stays on the outcome."
        )
        print(
            f"  from game {args.td_from_game}: {int(moved.sum()):,} of {len(target):,} rows "
            f"have a target other than the outcome, mean |target - outcome| "
            f"{float(np.abs(target - dataset.outcome).mean()):.4f}"
        )
    if args.target_file is not None:
        if args.td_lambda or args.target_key is None:
            raise SystemExit("--target-file needs --target-key and excludes --td-lambda")
        target = np.load(args.target_file)[args.target_key].astype(np.float32)
        if target.shape != dataset.outcome.shape or not (
            np.isfinite(target).all() and (target >= 0).all() and (target <= 1).all()
        ):
            raise SystemExit(
                f"{args.target_file}[{args.target_key}] is not one value in [0, 1] per row "
                f"of {args.data} ({target.shape} against {dataset.outcome.shape})"
            )
        moved = float(np.mean(np.abs(target - dataset.outcome) > 1e-9))
        print(
            f"target: {args.target_file}[{args.target_key}], {moved:.1%} of rows differ from "
            "the outcome. Validation stays on the outcome."
        )
    reg = load_regulation(
        __import__("json").loads(str(np.load(args.data)["meta_json"]))["format_id"]
    )
    encoder = Encoder(reg)
    schedule = {
        name: value
        for name, value in (
            ("keep", args.keep),
            ("average", args.average),
            ("ema_decay", args.ema_decay),
            ("swa_from", args.swa_from),
            ("pct_start", args.pct_start),
            ("move_properties", True if args.move_properties else None),
            ("attention", True if args.attention else None),
            ("dropout", args.dropout),
            ("encoder_dropout", args.encoder_dropout),
            ("weight_decay", args.weight_decay),
            ("aux_weight", args.aux_weight if args.aux_weight else None),
        )
        if value is not None
    }
    if args.width_scale != 1.0:
        if args.init_from is not None:
            raise SystemExit("--width-scale changes the shapes; it cannot warm-start")
        base = ValueConfig()
        for name in (
            "species_dim", "ability_dim", "item_dim", "move_dim", "mon_dim", "side_dim",
            "head_dim",
        ):
            schedule[name] = int(round(getattr(base, name) * args.width_scale))
    run = dict(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        split_seed=args.split_seed,
        **schedule,
    )
    device = torch.device(args.device)
    init_meta: dict = {}
    if args.init_from is not None:
        net, init_meta, config = warm_start(args.init_from, encoder, run)
        if args.widen != 1:
            from dataclasses import replace

            net = widen_net(net, encoder, args.widen, seed=args.seed)
            config = replace(
                config,
                mon_dim=net.config.mon_dim,
                side_dim=net.config.side_dim,
                head_dim=net.config.head_dim,
                width_groups=net.config.width_groups,
            )
            init_meta["widened"] = args.widen
        net = net.to(device)
        print(
            f"warm start from {args.init_from} ({init_meta.get('format_id')}, grown "
            f"{init_meta.get('vocab_grown_from') or 'none'})"
        )
    else:
        config = ValueConfig(**run)
        net = build(encoder, config).to(device)
    parameters = sum(p.numel() for p in net.parameters())

    train_idx, val_idx = split_for(dataset, args.holdout, config)
    print(
        f"{len(dataset):,} decisions from {len(np.unique(dataset.game)):,} games "
        f"-> {len(train_idx):,} train / {len(val_idx):,} validation, split by game "
        f"at split-seed {args.split_seed} (fit seed {args.seed})"
    )
    dropped_moves = [m for m in args.drop_train_moves.split(",") if m]
    if dropped_moves:
        ids = [encoder.vocab.moves[m] for m in dropped_moves]
        flat = dataset.encoded.moves[train_idx].reshape(len(train_idx), -1)
        train_idx = train_idx[~np.isin(flat, ids).any(axis=1)]
        print(
            f"--drop-train-moves: {len(train_idx):,} training decisions left without "
            f"{', '.join(dropped_moves)} (validation unchanged)"
        )
    print(f"{parameters:,} parameters on {device}")
    swap = None
    if args.swap_slots:
        reader = np.zeros(len(dataset), dtype=bool)
        imposter = encoder.vocab.abilities.get("imposter", 0)
        for start in range(0, len(dataset), 1_000_000):
            part = np.arange(start, min(start + 1_000_000, len(dataset)))
            reader[part] = position_reader_rows(
                dataset.encoded.ability[part], dataset.encoded.mask[part], imposter
            )
        swap = SwapSlots.of(encoder.mon_names, encoder.side_names, exclude=reader)
        print(
            f"--swap-slots: a random left/right arrangement per training row and epoch; "
            f"{int(reader.sum()):,} rows hold an Imposter and stay as played"
        )

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
        if args.td_lambda or args.target_file is not None:
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

    if init_meta:
        # Before one step: what the warm start knows from its own pool alone, on these
        # held-out games. The point a warm-versus-scratch comparison starts from.
        p0 = 1.0 / (1.0 + np.exp(-predict(net, dataset, val_idx, device=device)))
        init_meta["val_auc"] = auc(p0, dataset.outcome[val_idx])
        init_meta["val_logloss"] = logloss(p0, dataset.outcome[val_idx])
        print(
            f"before training: val log loss {init_meta['val_logloss']:.4f}  "
            f"AUC {init_meta['val_auc']:.4f}"
        )

    print(
        f"\ntraining: {config.epochs} epochs OneCycle to lr {config.lr:g} "
        f"(warm-up {config.pct_start:g}), keep {config.keep}, average {config.average}"
    )
    extra: dict = {}
    if config.aux_weight > 0:
        aux_target = final_material_targets(dataset)
        extra["aux_target"] = aux_target
        print(
            f"auxiliary head: {config.aux_weight:g} x MSE on the final material difference "
            f"(alive, HP; mean {aux_target.mean(axis=0).round(4).tolist()}, "
            f"std {aux_target.std(axis=0).round(4).tolist()})"
        )
    if args.game_weights:
        weights = game_weights(dataset.game, args.game_weights)
        extra["row_weight"] = weights
        print(
            f"--game-weights {args.game_weights}: mean row weight over training rows "
            f"{float(weights[train_idx].mean()):.4f}"
        )
    history, best = train(
        net,
        dataset,
        config,
        device=device,
        holdout=args.holdout,
        log=log,
        target=target,
        **extra,
        # Only when moves were dropped: otherwise `train` resolves the same split itself,
        # as every run before IKA-318 did.
        **({"train_index": train_idx, "val_index": val_idx} if dropped_moves else {}),
        **({"swap_slots": swap} if swap is not None else {}),
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
        if config.aux_weight > 0:
            # The head only taught the trunk; a saved model is the plain one.
            from dataclasses import replace

            best = {k: v for k, v in best.items() if not k.startswith("aux.")}
            config = replace(config, aux_weight=0.0)
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
                # The model these weights started from, and what it scored here before
                # training (IKA-194). None for a fresh initialisation.
                "init_from": init_meta or None,
                "td_lambda": args.td_lambda,
                "td_from_game": args.td_from_game,
                "target_file": None if args.target_file is None else str(args.target_file),
                "target_key": args.target_key,
                "seed": args.seed,
                # Separately, because --seed moves three things and only this one
                # decides what the row above was marked against. A model whose record
                # gives one number for both cannot say whether a rival's better AUC came
                # from a better fit or from easier games.
                "split_seed": args.split_seed,
                "swap_slots": bool(args.swap_slots),
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
