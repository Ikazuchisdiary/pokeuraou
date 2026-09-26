"""Generates self-play games and writes them as JSONL.

Our six are fixed and the opponent is drawn from the cited WCS2026 archetypes, with the
spreads sampled from usage rather than pinned -- the same composition is played many ways,
and a value function trained against one published spread would learn that the opponent's
investment is knowable.

    uv run python tools/selfplay.py --games 200 --seed 1 --hide-bench
    uv run python tools/selfplay.py --games 20 --hide-bench --report   # a short run

`--hide-bench` or `--open-bench` is required (IKA-123): leaving both off used to
generate open teacher data without a word.

With `--queue host:port` the games come from `tools/generate_queue.py` one at a time
instead of being dealt in advance, which is how a run stops ending when its unluckiest
worker does.

`--pool regmc-matchupweb` is M-C generation (IKA-81): no own side, both seats drawn from
the pool's 65 teams as a uniform pair of the 2,145 (mirrors included) with a coin for the
seats, and each game's selection solved with the leaf where it starts (`pokeuraou.poolplay`).
The bench is hidden unless `--open-bench` names the reference (IKA-128), so

    uv run --group learn python tools/selfplay.py --pool regmc-matchupweb --games 2 \\
        --value <an M-C model>

plays what M-C ships.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou import qrank, rustnode
from pokeuraou.benchflags import add_bench_flags, require_bench
from pokeuraou.damage import register_mega_stones
from pokeuraou.deepen import DEFAULT_DEEPEN, deepen_spec, parse_deepen
from pokeuraou.hidden import DEFAULT_BENCH_DROP, parse_bench_drop
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.priors import build_cooccurrence, find_cached_chaos, load_chaos
from pokeuraou.regulation import Regulation
from pokeuraou.search import DEFAULT_RANK_FILL, parse_rank_fill
from pokeuraou.selection_book import (
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    SelectionBook,
)
from pokeuraou.selfplay import MAX_TURNS, SEARCH_LIMIT, generate, selfplay_dir
from pokeuraou.standings import find_cached_standings, load_standings
from pokeuraou.teams import load_archetypes, load_roster, usable_archetypes

DEFAULT_ROSTER = "rizabanadohido"


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
        default=DEFAULT_RANK_FILL,
        help="with --pool and --rank-leaf: how the leaf ranking fills its cells -- "
        "refs<N> replies at the matrix budget, refs<N>-fast at Budget.fast (IKA-268). "
        f"Default {DEFAULT_RANK_FILL}.",
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
        help="with --pool and --rank-leaf: also write each game's leaf rankings -- every "
        "candidate's leaf values against the replies and the score narrow ordered by -- "
        "to rank-<name>.jsonl.gz beside --out (IKA-278, teacher data for IKA-274). Changes "
        "no game and no byte of --out.",
    )


#: Flags of the roster path that mean nothing without an own side, with their defaults.
_ROSTER_ONLY = {
    "roster": None,
    "selection_book": None,
    "mirror_share": 0.0,
    "force_lead": None,
    "opponents": "worlds",
    # `depth` and `solve_restricted` left this list with IKA-111: the pool path hides the
    # bench, and `belief_solve` now takes depth 2 in the restricted reading.
    "solve_sparsely": False,
}


def rank_scores_path(
    args: argparse.Namespace, ap: argparse.ArgumentParser, out: Path
) -> Path | None:
    """Where `--record-rank-scores` writes, or None; without a leaf ranking it stops."""
    if not args.record_rank_scores:
        return None
    if not args.rank_leaf:
        ap.error("--record-rank-scores records the leaf ranking; pass --rank-leaf")
    from pokeuraou.rank_scores import path_for

    return path_for(out)


def _install_q(args: argparse.Namespace, reg: Regulation, ap: argparse.ArgumentParser) -> None:
    """The Q a q rank fill ranks by (IKA-274), installed and echoed; nothing otherwise."""
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


def run_pool(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """`--pool`: both seats from the pool, the selection solved per game (IKA-81)."""
    from pokeuraou.pool import load_pool
    from pokeuraou.poolplay import SOLVED, generate_pool

    given = [
        "--" + name.replace("_", "-")
        for name, default in _ROSTER_ONLY.items()
        if getattr(args, name) != default
    ]
    if given:
        ap.error(f"{', '.join(given)} belong to the roster path; --pool has no own side")
    # IKA-128: M-C starts hidden. The pool path is new, so no recorded command without a
    # flag ever meant open here, and the reason `require_bench` stops does not apply.
    hide_bench = True if args.hide_bench is None else bool(args.hide_bench)

    pool = load_pool(args.pool)
    reg = pool.reg
    register_mega_stones(reg)
    evaluate, leaf_label = build_leaf(args, reg)
    _install_q(args, reg, ap)
    selection = "uniform" if args.uniform_selection else SOLVED
    print(pool.summary(), file=sys.stderr)
    if pool.character:
        print(f"  {pool.character}", file=sys.stderr)
    print(
        f"pool vs pool / {reg.meta.format_id} / search {args.limit}x{args.limit} / "
        f"leaf {leaf_label} / selection {selection}"
        f"{'' if selection == 'uniform' else f' (eps={args.explore_epsilon}, T={args.explore_temperature})'}"
        f" / bench {'hidden' if hide_bench else 'OPEN (reference)'}"
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
        hide_bench=hide_bench,
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
    )
    if client is not None:
        client.close()
    elapsed = time.perf_counter() - started
    finished = stats["finished"] or 1
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
    ap.add_argument(
        "--roster",
        default=None,
        help=f"our six at seat 0 (default {DEFAULT_ROSTER}). Not with --pool, which has "
        "no own side.",
    )
    ap.add_argument("--archetypes", default="wcs2026-regmb")
    ap.add_argument("--limit", type=int, default=SEARCH_LIMIT)
    ap.add_argument(
        "--rank-leaf",
        action="store_true",
        help="order candidates by the leaf instead of by expected damage. The two "
        "disagree, and the disagreement costs games -- see tools/narrow_regret.py and "
        "tools/regret_playout.py. Costs about 1.4x per decision.",
    )
    ap.add_argument(
        "--solve-sparsely",
        action="store_true",
        help="prove the equilibrium from the fifth of the matrix that settles it rather "
        "than filling all of it. Not an approximation -- the round ends only when neither "
        "side has a better reply over every action, which is a proof about the cells never "
        "solved. It does land on a different vertex of a degenerate optimum, so the games "
        "differ from a full-matrix run; measured at 2,544 games that is worth -0.2 points "
        "[-2.1, +1.8] against 1.94x the speed.",
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
    add_bench_flags(
        ap,
        hidden_help="neither side's search is shown the other's unplayed bench. Each "
        "solves over every four the opponent's sheet still allows, as the Bayesian game it "
        "is. The open game hands the search the opponent's whole four, which 47.4%% of "
        "decisions and every opening one had no right to; on 40 openings that was worth "
        "0.96 points and moved the advice in 90%% of them. Costs more per game than the "
        "open game, because each side solves its own game with up to six completions in "
        "it: 3.8x when first measured, 2.9x later, and the ratio moves with the width. "
        "One of this or --open-bench is required (IKA-123).",
        open_help="the open game: the search is handed the opponent's four. A reference, "
        "and what omitting both flags meant before IKA-123.",
    )
    ap.add_argument(
        "--force-lead",
        default=None,
        help="comma-separated species that must occupy our first two slots. Everything "
        "else is drawn as usual, so the games differ in one thing only. This is how a "
        "plan gets into the teacher data: forcing a *move* is undone on the next turn by "
        "a search that does not value the plan, while a Pokemon on the field stays on it. "
        "Toxapex leads 2.5%% of the book's mass and led 1.44%% of generation 9 against 47%% "
        "of the human repertoire, so those positions are the ones that do not exist.",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", action="store_true", help="print a summary of the output file")
    ap.add_argument(
        "--opponents",
        default="worlds",
        choices=("worlds", "metagame", "archetypes", "mix"),
        help="worlds (default): a real entry from the cached tournament standings, which "
        "is the field itself rather than a summary of it. metagame: six drawn from the "
        "measured pairwise co-occurrence -- statistically faithful to pairs but unable to "
        "represent a mixture of archetypes. archetypes: the cited compositions only (28 "
        "of 220 species). mix: half metagame, half archetypes.",
    )
    ap.add_argument("--season", default="2026")
    ap.add_argument(
        "--regulation-check",
        action="store_true",
        help="print a note when the event's stated format does not obviously match the "
        "regulation loaded. The event field is free text ('Regulation M-B') and ours is an "
        "id, so this cannot be an assertion.",
    )
    ap.add_argument("--event", default="worlds")
    ap.add_argument(
        "--standings-pool",
        default="all",
        choices=("all", "phase2", "cut"),
        help="all (default) is the whole field, which is what a player faces. cut is the "
        "13 teams that won, which is a different distribution and not a metagame.",
    )
    ap.add_argument(
        "--selection-book",
        type=Path,
        default=None,
        help="draw both sides' four-of-six from cached selection equilibria instead of "
        "uniformly (tools/solve_selection_book.py). Looked up by the opponent's team "
        "sheet, so our draw never sees which four they brought.",
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
    ap.add_argument(
        "--mirror-share",
        type=float,
        default=0.0,
        help="fraction of games played against our own six, spreads included. The mirror "
        "is where the only external knowledge about this team lives, and self-play had "
        "never played one -- so every mirror judgement was extrapolation. A true mirror is "
        "antisymmetric, so its win rate must come out at 50%%: a free calibration check on "
        "search, resolver and evaluator together.",
    )
    add_pool_flags(ap)
    args = ap.parse_args()
    try:
        parse_rank_fill(args.rank_fill)
        parse_bench_drop(args.bench_drop)
        parse_deepen(args.deepen)
    except ValueError as problem:
        ap.error(str(problem))
    if args.pool is not None:
        run_pool(args, ap)
        return
    if args.rank_fill != DEFAULT_RANK_FILL:
        # The roster path's `generate` takes no fill, and a flag it dropped would be
        # recorded nowhere and played by nobody.
        ap.error("--rank-fill is the pool path's (IKA-268); the roster path ranks at "
                 f"{DEFAULT_RANK_FILL}")
    if args.bench_drop != DEFAULT_BENCH_DROP:
        # The same for the belief: the roster path's `generate` takes no drop.
        ap.error("--bench-drop is the pool path's (IKA-283); the roster path believes "
                 "every completion")
    if args.deepen != DEFAULT_DEEPEN:
        # The same for the deepening: the roster path's `generate` takes no budget.
        ap.error("--deepen is the pool path's (IKA-33); the roster path searches at depth 1")
    if args.record_rank_scores:
        ap.error("--record-rank-scores is the pool path's (IKA-278)")
    if args.roster is None:
        args.roster = DEFAULT_ROSTER
    require_bench(args)

    roster = load_roster(args.roster)
    reg, archetypes = load_archetypes(args.archetypes)
    if reg.meta.format_id != roster.reg.meta.format_id:
        raise SystemExit(
            f"roster is {roster.reg.meta.format_id} but the archetypes are "
            f"{reg.meta.format_id}; training on a mixture of regulations would produce a "
            "value function valid for neither"
        )
    register_mega_stones(reg)

    evaluate, leaf_label = build_leaf(args, reg)

    chaos = find_cached_chaos(reg.meta.format_id)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    if prior.format_id != reg.meta.format_id:
        print(
            f"warning: usage data is {prior.format_id}, teams are {reg.meta.format_id}",
            file=sys.stderr,
        )
    standings = None
    if args.opponents == "worlds":
        cached = find_cached_standings(args.season, args.event)
        if cached is None:
            raise SystemExit(
                f"no cached standings for {args.season}/{args.event}; run "
                f"tools/fetch_standings.py {args.season} {args.event}"
            )
        standings = load_standings(cached, reg)
        print(standings.summary(), file=sys.stderr)
        if args.regulation_check and reg.meta.format_id not in standings.event_format:
            print(
                f"note: the event was played under {standings.event_format} and the "
                f"regulation loaded is {reg.meta.format_id}; check they are the same rules",
                file=sys.stderr,
            )

    book = None
    if args.selection_book is not None:
        if standings is None:
            raise SystemExit(
                "--selection-book is keyed on tournament team sheets, so it only applies "
                "to --opponents worlds"
            )
        book = SelectionBook.read(args.selection_book)
        book.require_roster(args.roster)
        hit, total = book.covered(standings.pool(args.standings_pool))
        print(
            f"選出解: {len(book)} チーム分（{book.model}）"
            f" プールの {hit}/{total} を被覆、"
            f"ε={args.explore_epsilon}, T={args.explore_temperature}",
            file=sys.stderr,
        )
        if hit == 0:
            raise SystemExit(
                f"{args.selection_book} covers none of the {total} teams in this pool; "
                "it was solved for another field or another roster"
            )

    cooc = build_cooccurrence(prior) if args.opponents in ("metagame", "mix") else None
    if cooc is not None:
        print(cooc.summary(), file=sys.stderr)
    archetype_share = {"worlds": 0.0, "metagame": 0.0, "archetypes": 1.0, "mix": 0.5}[
        args.opponents
    ]

    usable, blocked = usable_archetypes(prior, archetypes)
    for archetype, missing in blocked:
        print(
            f"excluding {archetype.id}: no usage data for {', '.join(missing)}",
            file=sys.stderr,
        )
    if not usable and archetype_share > 0:
        raise SystemExit("no archetype can be built from this usage data")

    pool = {
        "worlds": (
            f"{len(standings.pool(args.standings_pool)) if standings else 0} real teams "
            f"from {standings.event if standings else '?'}"
        ),
        "metagame": "the measured metagame",
        "archetypes": f"{len(usable)} cited archetypes",
        "mix": f"half metagame, half of {len(usable)} cited archetypes",
    }[args.opponents]
    print(
        f"{roster.name} vs {pool} / {reg.meta.format_id} / "
        f"search {args.limit}x{args.limit} / leaf {leaf_label}",
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

    rustnode.hold_positions()  # as the pool path above (IKA-264)
    started = time.perf_counter()
    stats = generate(
        reg,
        prior,
        roster,
        usable,
        games=args.games,
        seed=args.seed,
        indices=drawn,
        on_finish=client.finish if client is not None else None,
        out=args.out,
        objective=OBJECTIVES[args.objective],
        search_limit=args.limit,
        depth=args.depth,
        rank_by_leaf=args.rank_leaf,
        solve_sparsely=args.solve_sparsely,
        solve_restricted=args.solve_restricted,
        hide_bench=args.hide_bench,
        force_lead=tuple(
            x.strip() for x in args.force_lead.split(",") if x.strip()
        ) if args.force_lead else None,
        max_turns=args.max_turns,
        cooc=cooc,
        archetype_share=archetype_share,
        standings=standings,
        standings_pool=args.standings_pool,
        evaluate=evaluate,
        leaf=leaf_label,
        book=book,
        explore_epsilon=args.explore_epsilon,
        explore_temperature=args.explore_temperature,
        mirror_share=args.mirror_share,
    )
    if client is not None:
        client.close()
    elapsed = time.perf_counter() - started
    finished = stats["finished"] or 1
    print(
        f"{stats['games']} games in {elapsed:.1f}s "
        f"({elapsed / max(stats['games'], 1):.2f}s each)\n"
        f"  finished {stats['finished']}, discarded unfinished "
        f"{stats['discarded_unfinished']}\n"
        f"  side-0 win rate {stats['wins'] / finished * 100:.1f}%\n"
        f"  decisions {stats['decisions']} "
        f"({stats['decisions'] / finished:.1f} per game), "
        f"turns {stats['turns'] / finished:.1f} per game\n"
        f"  -> {stats['path']}"
    )
    if stats["mirror_games"]:
        rate = stats["mirror_wins"] / stats["mirror_games"]
        half = 1.96 * (0.25 / stats["mirror_games"]) ** 0.5
        verdict = "要件どおり" if abs(rate - 0.5) <= half else "ずれている（座席バイアス）"
        print(
            f"  ミラー {stats['mirror_games']} 戦の勝率 {rate * 100:.1f}% "
            f"±{half * 100:.1f}（50%% が要件 → {verdict}）"
        )
    if book is not None:
        print(
            f"  選出は選出解から {stats['book_hits']} 件、一様に戻したのが "
            f"{stats['book_misses']} 件\n"
            "  注: 生成は均衡＋探索の混合から引いているので、上の勝率は均衡値ではない"
        )

    if args.report:
        report(Path(stats["path"]))


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
