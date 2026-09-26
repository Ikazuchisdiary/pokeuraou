"""Validation loss of value models, split by how rare the moves in a position are (IKA-318).

A move embedding is learned from the decisions it appears in, so a move the pool rarely
uses is a move the net barely knows. This marks trained leaves on the held-out games of
`tools/train_value.py` (the same `--holdout` and `--split-seed`), and splits those games'
decisions by the training count of the rarest move in the position: how many training
decisions hold that move anywhere on the board, either side.

Each ``--arm NAME=a.pt,b.pt,...`` is a group of runs of one recipe at different seeds. For
each it prints every run's log loss per bin, their mean and spread (sample sd), and the
log loss of the group's logit mean (the served ensemble's shape).

``--moves a,b,c`` adds one more subset: the held-out decisions that hold any of those
moves (the ones a ``--drop-train-moves`` run never saw), and the rest. Their training
count here is the full split's, not the dropped run's.

    uv run --group learn python tools/value_move_rarity.py --data data/selfplay-mc0-encoded.npz \
        --arm id=m/id-s0.pt,m/id-s1.pt --arm props=m/props-s0.pt,m/props-s1.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.value import ValueConfig, load_dataset, load_model, predict, split_for  # noqa: E402


def logloss(logit: np.ndarray, labels: np.ndarray) -> float:
    """Binary cross-entropy from logits, as `value.train` marks validation."""
    return float(np.mean(np.logaddexp(0.0, logit) - labels * logit))


def presence(moves: np.ndarray, rows: int) -> np.ndarray:
    """(N, move rows) bool: which moves a decision's position holds, anywhere."""
    flat = moves.reshape(len(moves), -1)
    out = np.zeros((len(moves), rows), dtype=bool)
    out[np.repeat(np.arange(len(moves)), flat.shape[1]), flat.ravel()] = True
    out[:, 0] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--arm", action="append", default=[], help="NAME=path,path,...")
    ap.add_argument(
        "--edges",
        default="10000,20000,40000",
        help="training-count edges of the rarest-move bins",
    )
    ap.add_argument("--moves", default="", help="comma-separated move ids for a subset")
    ap.add_argument("--json", type=Path, default=None, help="also write the table here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dataset = load_dataset(args.data)
    meta = json.loads(str(np.load(args.data)["meta_json"]))
    encoder = Encoder(load_regulation(meta["format_id"]))
    rows = encoder.vocab.sizes["move"]
    train_idx, val_idx = split_for(
        dataset, args.holdout, ValueConfig(seed=args.split_seed, split_seed=args.split_seed)
    )
    held = presence(dataset.encoded.moves[val_idx], rows)
    count = presence(dataset.encoded.moves[train_idx], rows).sum(axis=0)
    # The training count of the rarest move in each held-out position.
    rarest = np.where(held, count[None, :], np.iinfo(np.int64).max).min(axis=1)
    labels = dataset.outcome[val_idx].astype(np.float64)

    edges = [int(e) for e in args.edges.split(",") if e]
    subsets: list[tuple[str, np.ndarray]] = [("all", np.ones(len(val_idx), dtype=bool))]
    low = 0
    for edge in [*edges, None]:
        pick = rarest >= low if edge is None else (rarest >= low) & (rarest < edge)
        name = f"rarest >= {low:,}" if edge is None else f"rarest {low:,}-{edge - 1:,}"
        subsets.append((name, pick))
        low = edge or low
    if args.moves:
        ids = [encoder.vocab.moves[m] for m in args.moves.split(",")]
        has = held[:, ids].any(axis=1)
        subsets += [("with --moves", has), ("without --moves", ~has)]

    used = count[1:] > 0
    print(
        f"{len(val_idx):,} held-out decisions (split-seed {args.split_seed}); "
        f"{int(used.sum())} moves appear in training, rarest {int(count[1:][used].min()):,} rows"
    )
    for name, pick in subsets:
        print(f"  {name:>20}  n {int(pick.sum()):>7,}  win rate {labels[pick].mean():.3f}")

    device = torch.device(args.device)
    table: dict[str, dict[str, object]] = {}
    for spec in args.arm:
        name, paths = spec.split("=", 1)
        logits = []
        for path in paths.split(","):
            net, _meta = load_model(path, encoder)
            logits.append(predict(net.to(device), dataset, val_idx, device=device).astype(np.float64))
        stack = np.stack(logits)
        mean_logit = stack.mean(axis=0)
        table[name] = {}
        for subset, pick in subsets:
            each = [logloss(lg[pick], labels[pick]) for lg in stack]
            table[name][subset] = {
                "runs": each,
                "mean": float(np.mean(each)),
                "sd": float(np.std(each, ddof=1)) if len(each) > 1 else 0.0,
                "ensemble": logloss(mean_logit[pick], labels[pick]),
            }

    header = f"  {'subset':>20}  {'n':>7}" + "".join(
        f"  {name + ' mean +-sd':>22}  {'ens':>6}" for name in table
    )
    print("\nvalidation log loss")
    print(header)
    for subset, pick in subsets:
        line = f"  {subset:>20}  {int(pick.sum()):>7,}"
        for name in table:
            cell = table[name][subset]
            line += f"  {cell['mean']:>13.4f} +-{cell['sd']:.4f}  {cell['ensemble']:>6.4f}"
        print(line)
    if len(table) >= 2:
        first, *others = list(table)
        for other in others:
            print(f"\n{other} - {first} (mean of runs; per-seed pairs by position in --arm)")
            for subset, _pick in subsets:
                a, b = table[first][subset]["runs"], table[other][subset]["runs"]
                pairs = [y - x for x, y in zip(a, b, strict=False)]
                print(
                    f"  {subset:>20}  {np.mean(pairs):+.4f}  pairs "
                    + " ".join(f"{d:+.4f}" for d in pairs)
                )
    if args.json is not None:
        args.json.write_bytes(json.dumps(table, indent=1).encode())


if __name__ == "__main__":
    main()
