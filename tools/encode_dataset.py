"""Encodes self-play games into arrays once, so training does not re-parse 1.2 GB of JSON.

Also the place where the dataset's own properties get printed before anything is trained
on it: how many games, how the outcomes are balanced, how many decisions offered a real
choice, and which volatiles the encoder did not recognise. A volatile in the `other`
bucket is a feature the network cannot see, so the count is worth reading rather than
discovering later as unexplained error.

    uv run --group learn python tools/encode_dataset.py --dir data/selfplay-gen1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import Encoded, Encoder
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import Position
from pokeuraou.provenance import SELF_PLAY
from pokeuraou.regulation import load_regulation
from pokeuraou.value import Dataset, save_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("data/selfplay-gen1"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--regulation", default=None, help="default: read it from the games")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many games")
    ap.add_argument("--chunk", type=int, default=4096, help="positions per encode call")
    ap.add_argument(
        "--kinds",
        nargs="*",
        default=None,
        help="keep only games with these provenance kinds (selfplay, generation-match, "
        "width-match, ...). Default keeps everything and prints the breakdown, because "
        "the question of whether match games help is one to measure, not to assume.",
    )
    args = ap.parse_args()

    files = sorted(args.dir.glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no games in {args.dir}")

    encoder: Encoder | None = None
    pending: list[Position] = []
    chunks: list[Encoded] = []
    outcomes: list[float] = []
    games: list[int] = []
    turns: list[int] = []
    #: The value the generating search reported. Named `proxies` when that was hp-share;
    #: it has been the previous model's searched value since generation 2.
    proxies: list[float] = []
    #: The parameter-free hp-share of the position, which is what "is a learned value
    #: function worth having" is measured against.
    hp_shares: list[float] = []
    kinds: list[int] = []
    foes: list[int] = []
    foe_names: list[str] = []
    foe_index: dict[str, int] = {}
    branching: Counter[int] = Counter()
    search_limits: Counter[str] = Counter()
    provenances: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    game_id = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal pending
        if not pending or encoder is None:
            return
        piece = encoder.encode_positions(pending)
        unknown.update(piece.unknown_volatiles)
        chunks.append(piece)
        pending = []

    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a run still in flight can leave a partial last line
                if record.get("outcome") is None:
                    continue
                if args.limit and game_id >= args.limit:
                    break
                if encoder is None:
                    format_id = args.regulation or record["decisions"][0]["position"]["format"]
                    encoder = Encoder(load_regulation(format_id))
                    print(
                        f"encoding for {encoder.vocab.format_id} "
                        f"(vocab {encoder.vocab.fingerprint()}), widths {encoder.widths}"
                    )
                # Where the game came from. Self-play is the default and the only thing
                # that existed before the head-to-head tools started recording; a match
                # game has real outcomes but a *mismatched* pair of agents, so a dataset
                # that silently mixes them would blur what the value is conditional on.
                kind = (record.get("provenance") or {}).get("kind", SELF_PLAY)
                if args.kinds and kind not in args.kinds:
                    continue
                provenances[kind] += 1
                search_limits[str(record.get("searchLimit"))] += 1
                label = record.get("foeArchetype", "?")
                if label not in foe_index:
                    foe_index[label] = len(foe_names)
                    foe_names.append(label)
                for decision in record["decisions"]:
                    # Converted once and used twice: the encoder took JSON and converted
                    # it internally, so this moves the conversion rather than adding one,
                    # and hp-share needs the same object.
                    position = Position.from_json(decision["position"])
                    pending.append(position)
                    outcomes.append(float(record["outcome"]))
                    games.append(game_id)
                    turns.append(int(decision["turn"]))
                    proxies.append(float(decision["searchValue"]))
                    hp_shares.append(float(HP_SHARE(position)))
                    kinds.append(0 if decision["kind"] == "move" else 1)
                    foes.append(foe_index[label])
                    branching[len(decision["ownActions"])] += 1
                    if len(pending) >= args.chunk:
                        flush()
                game_id += 1
        if args.limit and game_id >= args.limit:
            break
    flush()

    if encoder is None or not chunks:
        raise SystemExit("no finished games found")

    encoded = Encoded(
        species=np.concatenate([c.species for c in chunks]),
        ability=np.concatenate([c.ability for c in chunks]),
        item=np.concatenate([c.item for c in chunks]),
        moves=np.concatenate([c.moves for c in chunks]),
        mon=np.concatenate([c.mon for c in chunks]),
        mask=np.concatenate([c.mask for c in chunks]),
        side=np.concatenate([c.side for c in chunks]),
        field=np.concatenate([c.field for c in chunks]),
        unknown_volatiles=dict(unknown),
    )
    dataset = Dataset(
        encoded=encoded,
        outcome=np.array(outcomes, dtype=np.float32),
        game=np.array(games, dtype=np.int32),
        turn=np.array(turns, dtype=np.int16),
        search_value=np.array(proxies, dtype=np.float32),
        hp_share=np.array(hp_shares, dtype=np.float32),
        kind=np.array(kinds, dtype=np.int8),
        foe=np.array(foes, dtype=np.int32),
        foe_names=tuple(foe_names),
    )

    print(f"\n{game_id:,} finished games, {len(dataset):,} decisions")
    print(f"  side-0 win rate {dataset.outcome.mean() * 100:.1f}%")
    print(f"  search budget per game: {dict(search_limits)}")
    print(f"  games by provenance: {dict(provenances)}")
    single = branching[1]
    total = sum(branching.values())
    print(
        f"  decisions offering only one action: {single:,} of {total:,} "
        f"({single / total * 100:.1f}%) -- no policy to learn there, value only"
    )
    print(f"  opponent pools: {dict(Counter(foe_names[i] for i in foes).most_common(6))}")
    if unknown:
        print("  volatiles the encoder does not name (they land in one 'other' bit):")
        for name, count in unknown.most_common(12):
            print(f"    {count:>7}  {name}")
    else:
        print("  every volatile encountered is in the vocabulary")

    out = args.out or (args.dir.parent / f"{args.dir.name}-encoded.npz")
    save_dataset(
        out,
        dataset,
        meta={
            "format_id": encoder.vocab.format_id,
            "vocab_fingerprint": encoder.vocab.fingerprint(),
            "source_dir": str(args.dir),
            "games": game_id,
            "search_limits": dict(search_limits),
            "provenances": dict(provenances),
        },
    )
    size = out.stat().st_size / 1e6
    print(f"\n-> {out} ({size:,.0f} MB) in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
