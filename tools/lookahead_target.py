"""IKA-434: the `lookahead` auxiliary target, one value per row of an encoded dataset.

    tools/lookahead_target.py --data D.npz --models "0:value-gen11L.pt,39998:value-mc0.pt,..."
        --out lookahead.npy [--static-out static.npy] [--device cuda] [--clip 6]

A row's lookahead is the generating search's value minus the generating evaluation model's
static value, both as logits of side 0 winning:

    clip(logit(clip(search_value, 1e-4, 1 - 1e-4)) - static_logit, -CLIP, CLIP)

search_value is the depth-1 search over the generating model as the leaf, so the difference
is what one turn of look ahead adds, without the leaf's own static error. `--models` is
"FROM:path,...": a game numbered at least FROM (and below the next FROM) was generated with
that model, as `--game-weights` reads its list. A path of `-` marks games whose leaf was not a
learned model: their rows get NaN (no target; the loss leaves them out). The dataset's
recorded `objectives` (meta_json: which leaf generated how many games) is checked against the
list, so a list that names the wrong generating model stops here.

The static logit is one forward pass of the model on the row's encoded arrays, read a chunk of
rows at a time (the dense arrays are never in memory whole). It is the same pass the search's
leaf makes, so for a model the search used (one net, not the mean of two) it is the value the
search read. `--static-out` keeps the raw static logits; the run can be resumed
(`<static-out>.progress`) after it stops.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_models(spec: str) -> list[tuple[int, str]]:
    """``"FROM:path,..."`` -> [(FROM, path)], FROM starting at 0 and increasing."""
    entries = []
    for part in spec.split(","):
        head, _, path = part.partition(":")
        entries.append((int(head), path))
    froms = [a for a, _ in entries]
    if froms[0] != 0 or froms != sorted(set(froms)):
        raise ValueError(f"{spec!r}: FROM must start at 0 and increase")
    return entries


def lookahead_from(
    search_value: np.ndarray, static_logit: np.ndarray, clip: float = 6.0, eps: float = 1e-4
) -> np.ndarray:
    """The target from a search value (probability) and a static logit, float32; NaN stays NaN."""
    sv = np.clip(np.asarray(search_value, np.float64), eps, 1.0 - eps)
    return np.clip(np.log(sv / (1.0 - sv)) - np.asarray(static_logit, np.float64), -clip, clip).astype(
        np.float32
    )


def check_objectives(entries: list[tuple[int, str]], game: np.ndarray, objectives: dict) -> None:
    """Each model's games against the dataset's record of which leaf generated how many."""
    froms = [a for a, _ in entries]
    counts = np.bincount(np.searchsorted(froms, np.unique(game), side="right") - 1, minlength=len(froms))
    for (_from, path), n in zip(entries, counts, strict=True):
        if path == "-":
            continue
        key = "value:" + Path(path).stem
        if objectives and objectives.get(key) != int(n):
            raise SystemExit(
                f"{path}: this list gives it {int(n)} games, the data records {objectives.get(key)} "
                f"for {key!r} ({objectives})"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--static-out", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--clip", type=float, default=6.0)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop (resumable) after this long")
    args = ap.parse_args()

    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.packed import NpzMember
    from pokeuraou.regulation import load_regulation
    from pokeuraou.value import load_model

    started = time.time()
    entries = parse_models(args.models)
    with np.load(args.data) as z:
        game, search_value = z["game"], z["search_value"]
        meta = json.loads(str(z["meta_json"]))
    check_objectives(entries, game, meta.get("objectives") or {})
    rows = len(game)
    which = np.searchsorted([a for a, _ in entries], game, side="right") - 1
    encoder = Encoder(load_regulation(meta["format_id"]))
    device = torch.device(args.device)
    nets = {}
    for k, (_from, path) in enumerate(entries):
        if path != "-":
            nets[k] = load_model(path, encoder)[0].eval().to(device)

    static_path = Path(args.static_out or str(args.out) + ".static.tmp.npy")
    progress = Path(str(static_path) + ".progress")
    static = np.full(rows, np.nan, np.float32)
    done_rows = 0
    if progress.exists() and static_path.exists():
        saved = np.load(static_path)
        if saved.shape == static.shape:
            static, done_rows = saved, int(progress.read_text())
            print(f"resuming at row {done_rows:,}", flush=True)
    keys = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
    streams = [NpzMember(args.data, k).chunks() for k in keys]
    row = 0
    since_save = 0
    with torch.no_grad():
        for parts in zip(*streams, strict=True):
            n = len(parts[0])
            lo, hi = row, row + n
            row = hi
            if hi <= done_rows:
                continue
            use = np.flatnonzero(np.isin(which[lo:hi], list(nets)))
            for k, net in nets.items():
                pick = use[which[lo:hi][use] == k]
                for s in range(0, len(pick), args.batch):
                    sel = pick[s : s + args.batch]
                    batch = {}
                    for key, part in zip(keys, parts, strict=True):
                        a = np.ascontiguousarray(part[sel])
                        a = a.astype(np.int64) if a.dtype.kind in "iu" else a
                        batch[key] = torch.from_numpy(a).to(device)
                    static[lo + sel] = net(batch).float().cpu().numpy()
            done_rows = hi
            since_save += 1
            if since_save >= 25 or (args.max_seconds and time.time() - started > args.max_seconds):
                np.save(static_path, static)
                progress.write_bytes(str(done_rows).encode())
                since_save = 0
                print(f"  {done_rows:,}/{rows:,} rows, {time.time() - started:.0f}s", flush=True)
                if args.max_seconds and time.time() - started > args.max_seconds:
                    print("stopped for the time limit; run again to resume", flush=True)
                    sys.exit(3)
    assert row == rows
    np.save(static_path, static)
    progress.write_bytes(str(rows).encode())
    target = lookahead_from(search_value, static, args.clip)
    target[~np.isin(which, list(nets))] = np.nan
    np.save(args.out, target)
    scored = target[~np.isnan(target)].astype(np.float64)
    print(
        f"rows {rows:,}, with a target {len(scored):,}; mean {scored.mean():+.4f} std {scored.std():.4f} "
        f"at the clip {args.clip:g}: {float((np.abs(scored) >= args.clip).mean()):.4%}; "
        f"{time.time() - started:.0f}s",
        flush=True,
    )
    if not args.static_out:
        static_path.unlink()
        progress.unlink()


if __name__ == "__main__":
    main()
