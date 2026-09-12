"""Encodes self-play games into arrays once, so training does not re-parse 1.2 GB of JSON.

Encoding is per directory and cached. A generation is written once and never changes, so
re-encoding the whole pool every time a new one lands is work that grows with the square
of the number of generations: 46,604 games took 169 seconds, and at the 800,000 games this
is heading for it would be well over an hour on every single training run. Each directory
gets its own `-encoded.npz`, and the pool is the concatenation.

A cached shard is reused only when the directory's files still have exactly the byte sizes
they had when it was written, and when the vocabulary and the filtering arguments match.
Self-play files are append-only, so a worker that wrote more games after the shard was
built shows up as a size change and the shard is rebuilt. `--force` rebuilds regardless.

Also the place where the dataset's own properties get printed before anything is trained
on it: how many games, how the outcomes are balanced, how many decisions offered a real
choice, and which volatiles the encoder did not recognise. A volatile in the `other`
bucket is a feature the network cannot see, so the count is worth reading rather than
discovering later as unexplained error.

    uv run --group learn python tools/encode_dataset.py --dir data/selfplay-gen1
    uv run --group learn python tools/encode_dataset.py \
        --dir data/selfplay-gen5 data/selfplay-gen6 data/selfplay-gen7 \
        --out data/selfplay-fresh-encoded.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.encode import Encoded, Encoder
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import Position
from pokeuraou.provenance import SELF_PLAY
from pokeuraou.regulation import load_regulation
from pokeuraou.value import Dataset, concat_datasets, load_dataset, save_dataset


def shard_path(directory: Path) -> Path:
    return directory.parent / f"{directory.name}-encoded.npz"


def sources_of(directory: Path) -> list[list[Any]]:
    """What the directory holds right now: each file's name and size, sorted.

    Size rather than mtime. A checkout, a copy or a touch moves mtime without changing a
    byte, and would throw away an hour of encoding for nothing; a generation worker that
    appends more games does change the size. The one thing this misses -- a file rewritten
    to exactly its old length -- does not happen to append-only JSONL.
    """
    return [[p.name, p.stat().st_size] for p in sorted(directory.glob("*.jsonl"))]


def encode_dir(directory: Path, args: argparse.Namespace) -> tuple[Dataset, dict[str, Any]]:
    """Encodes one directory of games, or reads back the cache if it is still valid."""
    cache = shard_path(directory)
    sources = sources_of(directory)
    if not sources:
        raise SystemExit(f"no games in {directory}")
    want = {
        "sources": sources,
        "kinds_filter": sorted(args.kinds) if args.kinds else None,
        "limit": args.limit,
    }
    # A filtered run does not get a cache, in either direction. `--limit 50` is a
    # debugging flag, and letting it write `data/selfplay-gen7-encoded.npz` would replace
    # a full generation with fifty games under a name that says otherwise -- a trap that
    # would be sprung an hour later by a training run that read the file and believed it.
    cacheable = not args.limit and not args.kinds
    if cacheable and cache.exists() and not args.force:
        try:
            have = json.loads(str(np.load(cache, allow_pickle=False)["meta_json"]))
        except (KeyError, ValueError, OSError):
            have = {}
        if all(have.get(k) == v for k, v in want.items()):
            dataset = load_dataset(cache)
            print(f"  {directory.name}: {len(dataset):,} decisions from cache")
            return dataset, have
        print(f"  {directory.name}: cache is stale, re-encoding")

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
    engines: Counter[str] = Counter()
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

    for name, _size in sources:
        with (directory / name).open(encoding="utf-8") as handle:
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
                engine = record.get("engine") or {}
                engines[str(engine.get("sources", "unrecorded"))] += 1
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
        raise SystemExit(f"no finished games found in {directory}")

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
    meta = {
        **want,
        "format_id": encoder.vocab.format_id,
        "vocab_fingerprint": encoder.vocab.fingerprint(),
        "source_dir": str(directory),
        "games": game_id,
        "search_limits": dict(search_limits),
        "provenances": dict(provenances),
        # Which build of the engine played these games. Recorded per shard because a
        # dataset that spans an engine fix needs to be able to say where the seam is.
        "engines": dict(engines),
        # How many actions each decision offered. Kept in the shard because it is a
        # property of the games, not of the arrays, and re-deriving it would mean
        # re-reading the JSON that the shard exists to avoid re-reading.
        "branching": {str(k): v for k, v in sorted(branching.items())},
    }
    if cacheable:
        save_dataset(cache, dataset, meta=meta)
    print(
        f"  {directory.name}: {game_id:,} games, {len(dataset):,} decisions "
        f"in {time.perf_counter() - started:.0f}s"
        + (f" -> {cache.name}" if cacheable else " (filtered run, not cached)")
    )
    return dataset, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, nargs="+", default=[Path("data/selfplay-gen1")])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--regulation", default=None, help="default: read it from the games")
    ap.add_argument(
        "--limit", type=int, default=0, help="stop after this many games, per directory"
    )
    ap.add_argument("--chunk", type=int, default=4096, help="positions per encode call")
    ap.add_argument(
        "--force", action="store_true", help="re-encode even when a shard looks current"
    )
    ap.add_argument(
        "--kinds",
        nargs="*",
        default=None,
        help="keep only games with these provenance kinds (selfplay, generation-match, "
        "width-match, ...). Default keeps everything and prints the breakdown, because "
        "the question of whether match games help is one to measure, not to assume.",
    )
    args = ap.parse_args()

    started = time.perf_counter()
    parts: list[Dataset] = []
    metas: list[dict[str, Any]] = []
    for directory in args.dir:
        dataset, meta = encode_dir(directory, args)
        parts.append(dataset)
        metas.append(meta)

    fingerprints = {m.get("vocab_fingerprint") for m in metas}
    if len(fingerprints) > 1:
        raise SystemExit(
            f"shards were encoded with different vocabularies ({fingerprints}); "
            "delete the stale ones and re-encode, because the embedding indices in one "
            "do not mean the same species in another"
        )
    dataset = concat_datasets(parts)

    def merged(key: str) -> dict[str, int]:
        out: Counter[str] = Counter()
        for meta in metas:
            out.update(meta.get(key) or {})
        return dict(out)

    total_games = sum(int(m.get("games", 0)) for m in metas)
    search_limits = merged("search_limits")
    provenances = merged("provenances")
    engines = merged("engines")
    unknown = Counter(dataset.encoded.unknown_volatiles)

    print(f"\n{total_games:,} finished games, {len(dataset):,} decisions")
    print(f"  side-0 win rate {dataset.outcome.mean() * 100:.1f}%")
    print(f"  search budget per game: {search_limits}")
    print(f"  games by provenance: {provenances}")
    print(f"  games by engine build: {engines}")
    branching = merged("branching")
    total = sum(branching.values())
    if total:
        single = branching.get("1", 0)
        print(
            f"  decisions offering only one action: {single:,} of {total:,} "
            f"({single / total * 100:.1f}%) -- no policy to learn there, value only"
        )
    print(
        "  opponent pools: "
        f"{dict(Counter(dataset.foe_names[i] for i in dataset.foe).most_common(6))}"
    )
    if unknown:
        print("  volatiles the encoder does not name (they land in one 'other' bit):")
        for name, count in unknown.most_common(12):
            print(f"    {count:>7}  {name}")
    else:
        print("  every volatile encountered is in the vocabulary")

    out = args.out or shard_path(args.dir[0])
    if out != shard_path(args.dir[0]) or len(args.dir) > 1:
        save_dataset(
            out,
            dataset,
            meta={
                "format_id": metas[0]["format_id"],
                "vocab_fingerprint": metas[0]["vocab_fingerprint"],
                "source_dir": [str(d) for d in args.dir],
                "games": total_games,
                "search_limits": search_limits,
                "provenances": provenances,
                "engines": engines,
            },
        )
    size = out.stat().st_size / 1e6
    print(f"\n-> {out} ({size:,.0f} MB) in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
