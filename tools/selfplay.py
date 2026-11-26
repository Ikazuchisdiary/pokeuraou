"""Generates M-C self-play games and writes them as JSONL (IKA-81).

`--pool regmc-matchupweb` has no own side: both seats are drawn from the pool's 65 teams
as a uniform pair of the 2,145 (mirrors included) with a coin for the seats, and each
game's selection is solved with the leaf where it starts (`pokeuraou.poolplay`). The bench
is hidden in every game. M-B's roster path and the open game were removed (IKA-460).

    uv run --group learn python tools/selfplay.py --pool regmc-matchupweb --games 2 \\
        --value <an M-C model>

plays what M-C ships. With `--queue host:port` the games come from
`tools/generate_queue.py` one at a time instead of being dealt in advance, which is how a
run stops ending when its unluckiest worker does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# IKA-431: OpenBLAS commits a buffer per thread it may run when it loads -- numpy's copy
# and scipy's, about 0.5 GB each on this 16-thread machine, a gigabyte in a worker that runs
# on one core. Set before numpy loads (a launcher sets it already); no answer moves
# (records/IKA-431.md).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import qrank, rustnode
from pokeuraou.benchflags import add_bench_flags
from pokeuraou.damage import register_mega_stones
from pokeuraou.deepen import DEFAULT_DEEPEN, deepen_spec, parse_deepen
from pokeuraou.hidden import DEFAULT_BENCH_DROP, parse_bench_drop
from pokeuraou.payoff import OBJECTIVES
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
        default=None,
        help="draw BOTH seats from this pool (a data/pool/<id>.json, e.g. "
        "regmc-matchupweb) instead of fixing --roster at seat 0. The regulation is the "
        "pool's. Hidden bench unless --open-bench is given.",
    )
    ap.add_argument(
        "--uniform-selection",
        action="store_true",
        help="with --pool: draw both fours uniformly (a reference) instead of solving the "
        "pair's selection game with the leaf at the start of each game",
    )
    ap.add_argument(
        "--no-selection-memo",
        action="store_true",
        help="with --pool: solve the selection game at every game instead of once per "
        "pair. Changes no game; it exists to show that.",
    )
    ap.add_argument(
        "--selection-store",
        type=Path,
        default=None,
        help="with --pool: the directory the pair solves are shared through, so a pair is "
        "solved once for the whole run rather than once per worker. Default: "
        "selection-solved/ beside --out, which is every worker's directory under "
        "tools/generate_queue.py.",
    )
    ap.add_argument(
        "--rank-fill",
        default=None,
        help="with --pool and --rank-leaf: how the leaf ranking fills its cells -- "
        "refs<N> replies at the matrix budget, refs<N>-fast at Budget.fast (IKA-268), "
        "q / q-nocover rank by a Q (IKA-274). "
        f"Default {SHIPPED_RANK_FILL} (IKA-338), by {qrank.DEFAULT_Q} unless --q-arm / "
        f"--q-model names a Q; refs2 is played when named.",
    )
    ap.add_argument(
        "--bench-drop",
        default=DEFAULT_BENCH_DROP,
        help="with --pool and a hidden bench: which completions of the opponent's unseen "
        "slots the belief leaves out -- w<P> those under P%% of the weight, m<P> all but "
        "the heaviest carrying P%%; the heaviest always stays and the rest are "
        f"renormalised (IKA-283). Default {DEFAULT_BENCH_DROP}.",
    )
    ap.add_argument(
        "--deepen",
        default=DEFAULT_DEEPEN,
        help="with --pool: how each move decision deepens its answer best first after the "
        "depth-1 solve, where no bench is hidden -- m<N> spends N cells and reads the root "
        "whole, r<N> reads it as the restricted game (IKA-33); m<N>o<W> / m<N>oall "
        "also widen the root by a double oracle over the rest of the width-W menu / "
        "every legal action, s<W> / sall swapping a weightless action out for each, "
        "b<N>o<W> / b<N>s<W> the oracle without deepening (IKA-293). "
        f"Default {DEFAULT_DEEPEN}, off.",
    )
    ap.add_argument(
        "--record-rank-scores",
        action="store_true",
        help="with --pool, --rank-leaf and a refs --rank-fill (a q fill has no cells to "
        "record; IKA-341): also write each game's leaf rankings -- every "
        "candidate's leaf values against the replies and the score narrow ordered by -- "
        "to rank-<name>.jsonl.gz beside --out (IKA-278, teacher data for IKA-274). Changes "
        "no game and no byte of --out.",
    )
    ap.add_argument(
        "--pairs-with",
        nargs="+",
        default=None,
        metavar="SPECIES",
        help="with --pool: draw only the pairs with a team holding one of these species "
        "(IKA-444; as tools/pool_match.py). Default: every pair.",
    )
    ap.add_argument(
        "--pairs-no-mirror",
        action="store_true",
        help="with --pool: leave out the pairs of a team with itself (IKA-444)",
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


def rank_scores_path(
    args: argparse.Namespace, ap: argparse.ArgumentParser, out: Path
) -> Path | None:
    """Where `--record-rank-scores` writes, or None; without a refs leaf ranking it stops."""
    if not args.record_rank_scores:
        return None
    if not args.rank_leaf:
        ap.error("--record-rank-scores records the leaf ranking; pass --rank-leaf")
    if qrank.is_q(args.rank_fill):
        # IKA-341: a q fill ranks by the Q and fills no cell; the file came out with no
        # ranking in any game's line (the shipped fill since IKA-338).
        ap.error(f"--record-rank-scores records the refs leaf ranking, and {args.rank_fill} "
                 "ranks by a Q with no cell to record: name --rank-fill refs2 (a refs fill)")
    from pokeuraou.rank_scores import path_for

    return path_for(out)


def _install_q(
    args: argparse.Namespace, reg: Regulation, ap: argparse.ArgumentParser
) -> dict[str, list[str]]:
    """The Q a q rank fill ranks by (IKA-274), installed and echoed; nothing otherwise.
    Returns what each record names it by (`qrank.record_fields`, IKA-340), empty without."""
    if qrank.is_q(args.rank_fill) and not args.rank_leaf:
        ap.error(f"--rank-fill {args.rank_fill} ranks the leaf-ranked menu: it needs --rank-leaf")
    from pokeuraou.encode import Encoder

    # A deepen label whose oracle probes by a Q (IKA-322's q<k>) wants the default Q; it
    # is named to `install_from_args` as a q fill would be.
    # So does one whose children's menus a Q ranks (IKA-307's c<k>).
    spec = deepen_spec(args.deepen)
    probing = ["q"] if spec.q_probe is not None or spec.child_q is not None else []
    wanted = qrank.is_q(args.rank_fill) or bool(probing)
    encoder = Encoder(reg) if wanted else None
    models = qrank.install_from_args(args, encoder, (args.rank_fill, *probing), ap.error)
    if models:
        print(f"  Q: {', '.join(qrank.describe_installed(models))} "
              + (f"(the {args.q_arm} arm on {args.inference})" if args.q_arm else "(loaded here)"),
              file=sys.stderr)
    return qrank.record_fields(models)


def run_pool(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """`--pool`: both seats from the pool, the selection solved per game (IKA-81)."""
    from pokeuraou.pool import load_pool
    from pokeuraou.poolplay import SOLVED, generate_pool

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    evaluate, leaf_label = build_leaf(args, reg)
    q_record = _install_q(args, reg, ap)
    selection = "uniform" if args.uniform_selection else SOLVED
    pairs = pool_pairs(pool, args.pairs_with, args.pairs_no_mirror, ap.error)
    print(pool.summary(), file=sys.stderr)
    if pairs is not None:
        print(f"  pairs: {len(pairs)} of {len(pool.pairs)} (--pairs-with {args.pairs_with}"
              f"{', no mirror' if args.pairs_no_mirror else ''})", file=sys.stderr)
    if pool.character:
        print(f"  {pool.character}", file=sys.stderr)
    print(
        f"pool vs pool / {reg.meta.format_id} / search {args.limit}x{args.limit} / "
        f"leaf {leaf_label} / selection {selection}"
        f"{'' if selection == 'uniform' else f' (eps={args.explore_epsilon}, T={args.explore_temperature})'}"
        " / bench hidden"
        f" / {'leaf ranking, fill ' + args.rank_fill if args.rank_leaf else 'damage ranking'}"
        f" / bench drop {args.bench_drop}"
        f" / deepen {args.deepen}"
        + (f" / depth {args.depth}"
           + (" restricted" if args.solve_restricted else "") if args.depth != 1 else ""),
        file=sys.stderr,
    )
    client = None
    drawn: Iterator[int] | None = None
    if args.queue:
        from pokeuraou.workqueue import WorkClient

        client = WorkClient(args.queue)

        def from_queue(source: WorkClient) -> Iterator[int]:
            while (index := source.take()) is not None:
                yield index

        drawn = from_queue(client)
    out = args.out or selfplay_dir() / f"pool-{pool.id}-seed{args.seed}.jsonl"
    store = args.selection_store or out.parent / "selection-solved"
    ranks_out = rank_scores_path(args, ap, out)
    if ranks_out is not None:
        # The worker log's own echo, which is what says the run recorded (IKA-278).
        print(f"  rank scores: recording every leaf ranking to {ranks_out}", file=sys.stderr)
    # A generation worker edits no position it has sent: each decision's positions are
    # written, and a repeated `score` asked, once (IKA-264).
    rustnode.hold_positions()
    started = time.perf_counter()
    stats = generate_pool(
        reg,
        pool,
        games=args.games,
        seed=args.seed,
        out=out,
        selection=selection,
        evaluate=evaluate,
        memo=not args.no_selection_memo,
        store=None if args.no_selection_memo else store,
        objective=OBJECTIVES[args.objective],
        search_limit=args.limit,
        max_turns=args.max_turns,
        leaf=leaf_label,
        explore_epsilon=args.explore_epsilon,
        explore_temperature=args.explore_temperature,
        rank_by_leaf=args.rank_leaf,
        rank_fill=args.rank_fill,
        bench_drop=args.bench_drop,
        deepen=args.deepen,
        depth=args.depth,
        solve_restricted=args.solve_restricted,
        indices=drawn,
        on_finish=client.finish if client is not None else None,
        rank_scores_out=ranks_out,
        q_record=q_record,
        pairs=pairs,
    )
    if client is not None:
        client.close()
    elapsed = time.perf_counter() - started
    finished = stats["finished"] or 1
    # IKA-455: the worker's own echo of whether its LPs went to the port, and what went.
    from pokeuraou import portlp

    print(f"  {portlp.echo()}", file=sys.stderr)
    if ranks_out is not None:
        written = stats.get("rank_scores_bytes", 0)
        print(
            f"  rank scores: {written} bytes ({written / finished:.0f} a game) -> {ranks_out}",
            file=sys.stderr,
        )
    print(
        f"{stats['games']} games in {elapsed:.1f}s "
        f"({elapsed / max(stats['games'], 1):.2f}s each)\n"
        f"  finished {stats['finished']}, discarded unfinished "
        f"{stats['discarded_unfinished']}\n"
        f"  seat-0 win rate {stats['wins'] / finished * 100:.1f}%\n"
        f"  decisions {stats['decisions']} "
        f"({stats['decisions'] / finished:.1f} per game), "
        f"turns {stats['turns'] / finished:.1f} per game\n"
        f"  mirrors {stats['mirror_games']} (wins {stats['mirror_wins']}, "
        f"draws {stats['mirror_draws']})\n"
        f"  -> {stats['path']}"
    )
    if "solves" in stats:
        solves = stats["solves"] or 1
        print(
            f"  selection: {stats['solves']} solves ({stats['solves_reused']} reused, "
            f"{stats['solves_loaded']} read from {store}), "
            f"{stats['solve_seconds']:.1f}s = {stats['solve_seconds'] / solves:.2f}s a "
            f"solve, {stats['solve_positions']} positions, worst antisymmetry "
            f"{stats['solve_worst_antisymmetry']:.2e}"
        )
    if args.report:
        report(Path(stats["path"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument(
        "--queue",
        default=None,
        help="host:port of a work queue to draw game numbers from, instead of playing "
        "--games in a row. Each game is then seeded from its own number rather than from "
        "the games before it, so it is the same game whichever worker draws it.",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=SEARCH_LIMIT)
    ap.add_argument(
        "--rank-leaf",
        action="store_true",
        help="order candidates by the leaf instead of by expected damage. The two "
        "disagree, and the disagreement costs games -- see tools/narrow_regret.py and "
        "tools/regret_playout.py. Costs about 1.4x per decision.",
    )
    ap.add_argument(
        "--solve-restricted",
        action="store_true",
        help="at depth 2, read the strategy off the refined rectangle solved as its own "
        "game rather than off the full matrix with the refined cells written into it. "
        "Does nothing at depth 1, which is the default and what generation runs.",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=1,
        help="1 is the one-ply search this project has always used. 2 refines the cells "
        "the equilibrium actually weights with the next turn's own equilibrium, which "
        "costs several times as much per decision -- worth it or not is a question for "
        "tools/depth_match.py, not for a default.",
    )
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--objective", default="hp-share", choices=sorted(OBJECTIVES))
    ap.add_argument(
        "--value",
        type=Path,
        default=None,
        help="a trained value function to score the search's leaves. Generation 2: the "
        "leaf becomes a win probability instead of an HP share, so the equilibrium value "
        "at every node is a win probability too. Needs the optional learn group.",
    )
    ap.add_argument(
        "--inference",
        default=None,
        metavar="HOST:PORT",
        help="score leaves on a shared inference server instead of loading the model "
        "here. The worker then imports no torch and holds no CUDA context -- 4.0 GB of "
        "commit and 1.5 GB of VRAM it does not need. Answers are unchanged: the server "
        "runs each request as it arrives and never merges one worker's with another's.",
    )
    ap.add_argument(
        "--inference-arm",
        default="value",
        help="which named arm on the server to score with",
    )
    qrank.add_q_flags(ap)
    ap.add_argument("--device", default=None, help="cuda or cpu; default is cuda if present")
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="threads per worker. 1 is right for a parallel run: N processes each spawning "
        "a pool on the same cores is slower than one process. Raise it for a single run.",
    )
    add_bench_flags(ap)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", action="store_true", help="print a summary of the output file")
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
    args = ap.parse_args()
    if args.pool is None:
        ap.error("--pool is required: M-B's roster path was removed (IKA-460)")
    # Named, or the pool path's shipped fill for a leaf-ranked menu (IKA-338).
    args.rank_fill = resolve_rank_fill(args.rank_fill, args.rank_leaf)
    try:
        parse_rank_fill(args.rank_fill)
        parse_bench_drop(args.bench_drop)
        parse_deepen(args.deepen)
    except ValueError as problem:
        ap.error(str(problem))
    run_pool(args, ap)


def build_leaf(args: argparse.Namespace, reg: Regulation) -> tuple[Any, str]:
    """The leaf the flags name, and the label every record carries for it."""
    evaluate = None
    leaf_label = args.objective
    if args.inference is not None:
        # No torch here at all: the encoder is numpy, and the arrays go to the server.
        from pokeuraou.encode import Encoder
        from pokeuraou.inference import RemoteValue

        evaluate = RemoteValue(args.inference, args.inference_arm, Encoder(reg))
        # Asked of the server, not taken from this command line. A worker is told which
        # arm to use and never what that arm holds, and this label is stamped into every
        # game it records -- it is how a dataset says which model made it, months later.
        # Without asking it read `value:value`, which is the arm's name and nothing about
        # the model. The policy's position representation was the same mistake.
        files = evaluate.describe()
        stem = re.sub(r"-s\d+$", "", Path(files[0]).stem) if files else args.inference_arm
        leaf_label = (
            f"value:{stem}" if len(files) <= 1 else f"value:{stem}x{len(files)}"
        )
        print(
            f"leaf = the {args.inference_arm} arm on {args.inference} = {leaf_label} "
            f"(this worker holds no model)",
            file=sys.stderr,
        )
    elif args.value is not None:
        # Imported here so a run without --value never loads torch: the resolver, the
        # differential tests and the M1 path stay installable without a CUDA wheel.
        import torch

        torch.set_num_threads(args.torch_threads)

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        if not args.value.exists():
            raise SystemExit(f"no model at {args.value}; train one with tools/train_value.py")
        encoder = Encoder(reg)
        net, meta = load_model(args.value, encoder)
        device = torch.device(
            args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        evaluate = BatchedValue(net.to(device), encoder, device=device)
        leaf_label = f"value:{args.value.stem}"
        print(
            f"leaf = {args.value.name} on {device}  "
            f"(trained on {meta.get('games', '?')} games, "
            f"val AUC {meta.get('val_auc', float('nan')):.4f})",
            file=sys.stderr,
        )
        print(
            "  そして葉が勝率になったので、各ノードの均衡値も勝率です",
            file=sys.stderr,
        )
    return evaluate, leaf_label


def report(path: Path) -> None:
    """What is actually in the file, so the training step is not the first to look."""
    outcomes: list[float] = []
    decisions = 0
    unmodelled: dict[str, int] = {}
    archetypes: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        outcomes.append(float(record["outcome"]))
        decisions += len(record["decisions"])
        archetypes.setdefault(record["foeArchetype"], []).append(float(record["outcome"]))
        for name in record["unmodelled"]:
            unmodelled[name] = unmodelled.get(name, 0) + 1

    if not outcomes:
        print("\nno finished games in the file yet")
        return
    print(f"\n{path.name}: {len(outcomes)} games, {decisions} labelled decisions")
    print(f"  side-0 win rate {sum(outcomes) / len(outcomes) * 100:.1f}%")
    print("  by opponent archetype:")
    for name, results in sorted(archetypes.items(), key=lambda kv: -len(kv[1])):
        print(f"    {name:22} {len(results):>4} games  win {sum(results) / len(results) * 100:5.1f}%")
    if unmodelled:
        print("  effects the resolver reported during these games:")
        for name, count in sorted(unmodelled.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {count:>4} games  {name}")


if __name__ == "__main__":
    main()
