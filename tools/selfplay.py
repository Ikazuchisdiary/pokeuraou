"""Writes the config of M-C generation's worker, `pokeuraou-damage selfplay-worker` (IKA-458).

`--pool regmc-matchupweb` has no own side: both seats are drawn from the pool's 65 teams
as a uniform pair of the 2,145 (mirrors included) with a coin for the seats, and each
game's selection is solved with the leaf where it starts. The bench is hidden in every game.

The games are played by the Rust worker (`rust/src/worker.rs`); `tools/generate_queue.py`
runs this once per run to turn the flags into the JSON every worker reads
(`pokeuraou.rustworker.build_config`). The Python loop that played them here was removed in
IKA-458 stage 3.7, with the flags only it played (a leaf loaded here, the uniform selection,
the deepening, depth 2, the bench drop, the recorded rankings).

    uv run python tools/selfplay.py --pool regmc-matchupweb --inference HOST:PORT --q-arm q \\
        --rank-leaf --rust-config worker-config.json
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

# IKA-431: OpenBLAS commits a buffer per thread it may run when it loads. Set before numpy loads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import qrank, rustworker
from pokeuraou.benchflags import add_bench_flags
from pokeuraou.damage import register_mega_stones
from pokeuraou.regulation import Regulation
from pokeuraou.search import (
    SHIPPED_RANK_FILL,
    parse_rank_fill,
    resolve_rank_fill,
)
from pokeuraou.selection_book import (
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
)
from pokeuraou.selfplay import MAX_TURNS, SEARCH_LIMIT, selfplay_dir


def add_pool_flags(ap: argparse.ArgumentParser) -> None:
    """The pool-against-pool path's own flags (IKA-81)."""
    ap.add_argument(
        "--pool",
        required=True,
        help="draw BOTH seats from this pool (a data/pool/<id>.json, e.g. regmc-matchupweb). "
        "The regulation is the pool's.",
    )
    ap.add_argument(
        "--selection-store",
        type=Path,
        default=None,
        help="the directory the pair solves are shared through, so a pair is solved once for "
        "the whole run rather than once per worker. Default: selection-solved/ beside --out, "
        "which is every worker's directory under tools/generate_queue.py.",
    )
    ap.add_argument(
        "--rank-fill",
        default=None,
        help="with --rank-leaf: how the leaf ranking fills its cells. The Rust worker plays "
        f"a q-nocover fill only. Default {SHIPPED_RANK_FILL} (IKA-338), by {qrank.DEFAULT_Q} "
        "unless --q-arm / --q-model names a Q.",
    )
    ap.add_argument(
        "--pairs-with",
        nargs="+",
        default=None,
        metavar="SPECIES",
        help="draw only the pairs with a team holding one of these species "
        "(IKA-444; as tools/pool_match.py). Default: every pair.",
    )
    ap.add_argument(
        "--pairs-no-mirror",
        action="store_true",
        help="leave out the pairs of a team with itself (IKA-444)",
    )


def pool_pairs(
    pool: Any, species: list[str] | None, no_mirror: bool, error: Any
) -> list[tuple[int, int]] | None:
    """The pairs `--pairs-with` / `--pairs-no-mirror` leave, or None for every pair."""
    if not species and not no_mirror:
        return None
    pairs = list(pool.pairs)
    if species:
        unknown = [s for s in species if s not in pool.reg.species]
        if unknown:
            error(f"--pairs-with: not species ids of {pool.reg.meta.format_id}: {unknown}")
        holding = {t for t, team in enumerate(pool.teams)
                   if any(s.species in species for s in team.sets)}
        pairs = [(a, b) for a, b in pairs if a in holding or b in holding]
    if no_mirror:
        pairs = [(a, b) for a, b in pairs if a != b]
    if not pairs:
        error(f"--pairs-with {species}{' --pairs-no-mirror' if no_mirror else ''}: no pair left")
    return pairs


def _install_q(
    args: argparse.Namespace, reg: Regulation, ap: argparse.ArgumentParser
) -> dict[str, list[str]]:
    """The Q a q rank fill ranks by (IKA-274), installed and echoed; nothing otherwise.
    Returns what each record names it by (`qrank.record_fields`, IKA-340), empty without."""
    if qrank.is_q(args.rank_fill) and not args.rank_leaf:
        ap.error(f"--rank-fill {args.rank_fill} ranks the leaf-ranked menu: it needs --rank-leaf")
    from pokeuraou.encode import Encoder

    wanted = qrank.is_q(args.rank_fill)
    encoder = Encoder(reg) if wanted else None
    models = qrank.install_from_args(args, encoder, (args.rank_fill,), ap.error)
    if models:
        print(f"  Q: {', '.join(qrank.describe_installed(models))} "
              + (f"(the {args.q_arm} arm on {args.inference})" if args.q_arm else "(loaded here)"),
              file=sys.stderr)
    return qrank.record_fields(models)


def build_leaf(args: argparse.Namespace, reg: Regulation) -> tuple[Any, str]:
    """The inference server's leaf, and the label every record carries for it."""
    from pokeuraou.encode import Encoder
    from pokeuraou.inference import RemoteValue

    evaluate = RemoteValue(args.inference, args.inference_arm, Encoder(reg))
    # Asked of the server, not taken from this command line. A worker is told which arm to
    # use and never what that arm holds, and this label is stamped into every game it
    # records -- it is how a dataset says which model made it, months later.
    files = evaluate.describe()
    stem = re.sub(r"-s\d+$", "", Path(files[0]).stem) if files else args.inference_arm
    leaf_label = f"value:{stem}" if len(files) <= 1 else f"value:{stem}x{len(files)}"
    print(f"leaf = the {args.inference_arm} arm on {args.inference} = {leaf_label}",
          file=sys.stderr)
    return evaluate, leaf_label


def write_config(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """The Rust worker's config for these flags, written once."""
    from pokeuraou.pool import load_pool

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    refused = rustworker.unsupported(args)
    if refused:
        ap.error(f"the Rust worker does not play {', '.join(refused)}")
    evaluate, leaf_label = build_leaf(args, reg)
    q_record = _install_q(args, reg, ap)
    pairs = pool_pairs(pool, args.pairs_with, args.pairs_no_mirror, ap.error)
    print(pool.summary(), file=sys.stderr)
    if pairs is not None:
        print(f"  pairs: {len(pairs)} of {len(pool.pairs)} (--pairs-with {args.pairs_with}"
              f"{', no mirror' if args.pairs_no_mirror else ''})", file=sys.stderr)
    print(
        f"pool vs pool / {reg.meta.format_id} / search {args.limit}x{args.limit} / "
        f"leaf {leaf_label} / selection solved "
        f"(eps={args.explore_epsilon}, T={args.explore_temperature}) / bench hidden / "
        f"leaf ranking, fill {args.rank_fill}",
        file=sys.stderr,
    )
    out = args.out or selfplay_dir() / f"pool-{pool.id}-seed{args.seed}.jsonl"
    store = args.selection_store or out.parent / "selection-solved"
    rustworker.write_config(
        rustworker.build_config(args, pool, evaluate, leaf_label, q_record, pairs, store),
        args.rust_config,
    )
    print(f"  rust worker config -> {args.rust_config}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=SEARCH_LIMIT)
    ap.add_argument(
        "--rank-leaf",
        action="store_true",
        help="order candidates by the leaf instead of by expected damage; the Rust worker "
        "plays the leaf ranking only, so this is required",
    )
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument(
        "--inference",
        default=None,
        metavar="HOST:PORT",
        help="the inference server that scores the leaves (required)",
    )
    ap.add_argument(
        "--inference-arm",
        default="value",
        help="which named arm on the server to score with",
    )
    qrank.add_q_flags(ap)
    add_bench_flags(ap)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where the games go; only places the default --selection-store",
    )
    ap.add_argument(
        "--explore-epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help="share of games drawn off the equilibrium, for coverage. 0 plays the "
        "equilibrium exactly and visits one selection pair per opponent, which starves "
        "the next solve of cells.",
    )
    ap.add_argument(
        "--explore-temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="in win probability: how far down the EV-loss list the exploration share "
        "reaches. inf spreads it uniformly.",
    )
    add_pool_flags(ap)
    ap.add_argument(
        "--rust-config",
        type=Path,
        required=True,
        help="write the config `pokeuraou-damage selfplay-worker` reads for these flags "
        "(IKA-458 stage 3.7; tools/generate_queue.py does this once a run)",
    )
    args = ap.parse_args()
    # Named, or the pool path's shipped fill for a leaf-ranked menu (IKA-338).
    args.rank_fill = resolve_rank_fill(args.rank_fill, args.rank_leaf)
    try:
        parse_rank_fill(args.rank_fill)
    except ValueError as problem:
        ap.error(str(problem))
    write_config(args, ap)


if __name__ == "__main__":
    main()
