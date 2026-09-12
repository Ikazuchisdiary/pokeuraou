"""Generates self-play games and writes them as JSONL.

Our six are fixed and the opponent is drawn from the cited WCS2026 archetypes, with the
spreads sampled from usage rather than pinned -- the same composition is played many ways,
and a value function trained against one published spread would learn that the opponent's
investment is knowable.

    uv run python tools/selfplay.py --games 200 --seed 1
    uv run python tools/selfplay.py --games 20 --report   # what a short run looks like
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.priors import build_cooccurrence, find_cached_chaos, load_chaos
from pokeuraou.selection_book import (
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    SelectionBook,
)
from pokeuraou.selfplay import MAX_TURNS, SEARCH_LIMIT, generate
from pokeuraou.standings import find_cached_standings, load_standings
from pokeuraou.teams import load_archetypes, load_roster, usable_archetypes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--roster", default="rizabanadohido")
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
    ap.add_argument("--device", default=None, help="cuda or cpu; default is cuda if present")
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="threads per worker. 1 is right for a parallel run: N processes each spawning "
        "a pool on the same cores is slower than one process. Raise it for a single run.",
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
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg, archetypes = load_archetypes(args.archetypes)
    if reg.meta.format_id != roster.reg.meta.format_id:
        raise SystemExit(
            f"roster is {roster.reg.meta.format_id} but the archetypes are "
            f"{reg.meta.format_id}; training on a mixture of regulations would produce a "
            "value function valid for neither"
        )
    register_mega_stones(reg)

    evaluate = None
    leaf_label = args.objective
    if args.value is not None:
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
    started = time.perf_counter()
    stats = generate(
        reg,
        prior,
        roster,
        usable,
        games=args.games,
        seed=args.seed,
        out=args.out,
        objective=OBJECTIVES[args.objective],
        search_limit=args.limit,
        depth=args.depth,
        rank_by_leaf=args.rank_leaf,
        solve_sparsely=args.solve_sparsely,
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
            f"±{half * 100:.1f}（50% が要件 → {verdict}）"
        )
    if book is not None:
        print(
            f"  選出は選出解から {stats['book_hits']} 件、一様に戻したのが "
            f"{stats['book_misses']} 件\n"
            "  注: 生成は均衡＋探索の混合から引いているので、上の勝率は均衡値ではない"
        )

    if args.report:
        report(Path(stats["path"]))


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
