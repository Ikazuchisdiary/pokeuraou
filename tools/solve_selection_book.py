"""Solves the 6->4 selection once per opponent in the field and caches the answers.

    bash tools/solve_book_parallel.sh                       # the whole field, 14 workers
    uv run --group learn python tools/solve_selection_book.py --limit 5 --classes 2
    uv run --group learn python tools/solve_selection_book.py --merge   # parts -> one book

One entry per team sheet in the tournament standings: our equilibrium mixture over the 90
ordered selections, the opponent's mixture per spread class, the class weights and the
equilibrium value. Self-play then draws both sides' selections from the entry instead of
uniformly, which costs a lookup per game rather than a 12-second solve.

The cache is keyed on the opponent's *six* -- their public team sheet -- and never on the
four they bring. See :mod:`pokeuraou.selection_book` for that and the three other places
where a cache like this could turn into 後出しジャンケン.

**Where the time goes**, measured per class of 8,100 cells: building the positions 2.23s,
encoding them 0.49s, the forward pass 0.77s on CPU. So this is a CPU-bound, single-threaded
job that leaves a GPU idle, and the fix is processes rather than a faster kernel:
``--shard i --shards n`` solves ``teams[i::n]`` into its own part file and ``--merge``
combines them. The spread classes are seeded per *team*, not per worker, so a book solved
by fourteen workers is identical to one solved by one.

Resumable at the entry: a part file is valid gzip after every append, and a restarted shard
skips the teams already in it.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.names import localiser
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.selection import SpreadClass, book_entry, solve_selection
from pokeuraou.selection_book import (
    BookEntry,
    SelectionBook,
    append_entry,
    explore_mixture,
    key_for_team,
    perplexity,
    selection_dir,
    start_book,
)
from pokeuraou.standings import (
    TournamentTeam,
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster
from pokeuraou.value import BatchedValue, load_ensemble

#: The exploration settings the coverage table sweeps. The pair generation will use is
#: chosen from this table rather than by taste: a setting that leaves one of the 90
#: selections with no games recreates a failure this project already had, where the value
#: function had seen 6 of 15 lead pairs and the other 9 were extrapolation.
EPSILONS = (0.0, 0.25, 0.5)
TEMPERATURES = (0.05, 0.15, 0.5, float("inf"))
#: Games in a generation, for turning a probability into "how many games would that be".
GENERATION_GAMES = 7000


def book_stem(models: Sequence[Path]) -> str:
    """The default file stem for a book solved on these nets.

    Carries the member count, because an ensemble book exists to be compared against the
    single-net book it came from and must not be able to overwrite it by default.
    """
    return models[0].stem if len(models) == 1 else f"{models[0].stem}-ens{len(models)}"


def model_label(models: Sequence[Path]) -> str:
    """What the book records as its leaf. Every member, so a book can say what made it."""
    return "+".join(m.name for m in models)


def part_path(out: Path, shard: int) -> Path:
    stem = out.name.removesuffix(".jsonl.gz")
    return out.parent / f"{stem}.part{shard}.jsonl.gz"


def part_files(out: Path) -> list[Path]:
    """The shard files belonging to ``out``, which live beside it and nowhere else.

    ``--merge`` used to glob :func:`selection_dir` while :func:`part_path` wrote beside
    ``out``, so any run given an ``--out`` outside the default directory solved its eight
    shards and then refused to merge them. The two halves now read the same directory.
    """
    stem = out.name.removesuffix(".jsonl.gz")
    return sorted(out.parent.glob(f"{stem}.part*.jsonl.gz"))


def solve_one(
    reg: object,
    roster: object,
    prior: object,
    team: TournamentTeam,
    value: object,
    *,
    classes: int,
    seed: int,
    index: int,
    model: str,
) -> tuple[BookEntry, float, int]:
    """One team's entry, plus its solve time and the size of its equilibrium support.

    The spread classes are drawn from ``[seed, index]`` -- the team's position in the pool
    -- so which worker happened to take the team cannot change the answer. A book solved in
    parallel is then byte-comparable with one solved serially, which is the only way the
    sharding can be checked at all.
    """
    rng = np.random.default_rng([seed, index])
    spread_classes = [
        SpreadClass(
            weight=1.0 / classes,
            sets=tuple(sample_standings_team(rng, reg, prior, team)),
            label=f"class {k + 1}",
        )
        for k in range(classes)
    ]
    analysis = solve_selection(reg, roster.sets, spread_classes, value)
    entry = book_entry(
        analysis,
        key=key_for_team(team),
        player=team.player,
        place=team.place,
        model=model,
    )
    return entry, analysis.seconds, int(len(analysis.equilibrium.row_support()))


def coverage_table(book: SelectionBook, selections: list[tuple[int, ...]]) -> str:
    """How much of our 90 the field's equilibria actually play, per exploration setting.

    Three numbers per setting, and the third is the one that decides:

    - *effective width*: ``exp(entropy)`` of the field-pooled distribution over our 90;
    - *rarest selection*: games out of a generation that the least-played of the 90 gets;
    - *rarest lead pair*: the same for the 15 lead pairs, which is the quantity a previous
      generation actually failed on.

    A setting whose rarest lead pair rounds to zero games is disqualified: the next solve
    would ask the value function about a turn-1 position it has never seen, and the answer
    would be an extrapolation dressed as a win probability.
    """
    leads = sorted({s[:2] for s in selections})
    lead_index = [leads.index(s[:2]) for s in selections]
    lines = [
        "■ 探索設定ごとの被覆（自陣の 90 通り、全チームでプールした分布）",
        f"  {'ε':>4} {'T':>5} | {'実効幅':>7} {'最薄の選出':>10} {'最薄の先発ペア':>14} "
        f"{'均衡どおり':>9}",
    ]
    for epsilon in EPSILONS:
        for temperature in TEMPERATURES:
            pooled = np.zeros(len(selections), dtype=np.float64)
            for entry in book.entries.values():
                pooled += explore_mixture(
                    entry.our_strategy,
                    entry.our_ev_loss,
                    epsilon=epsilon,
                    temperature=temperature,
                )
            pooled /= pooled.sum()
            lead = np.zeros(len(leads), dtype=np.float64)
            for i, p in enumerate(pooled):
                lead[lead_index[i]] += p
            lines.append(
                f"  {epsilon:>4} {temperature:>5} | {perplexity(pooled):>7.1f} "
                f"{pooled.min() * GENERATION_GAMES:>10.1f} "
                f"{lead.min() * GENERATION_GAMES:>14.1f} "
                f"{(1 - epsilon) * 100:>8.0f}%"
            )
            if epsilon == 0.0:
                break  # the equilibrium alone does not depend on the temperature
    lines.append(
        f"  「最薄の…」は {GENERATION_GAMES:,} ゲーム換算の割り当て。"
        "0 になる設定は次の解が外挿になるので不可"
    )
    return "\n".join(lines)


def report(
    book: SelectionBook,
    reg: object,
    seconds: float | None = None,
    namer: object = None,
) -> str:
    """The whole report as text, so it can be written as UTF-8 rather than only printed.

    Python's stderr on Windows encodes as cp932 and the terminal mangles it, which for a
    report nobody can read is the same as not producing one -- the same reason
    ``show_game.py`` writes to a file.
    """
    if not book.entries:
        return "空の本です"
    values = np.array([e.value for e in book.entries.values()])
    gaps = np.array([e.duality_gap for e in book.entries.values()])
    anti = np.array([e.antisymmetry_error for e in book.entries.values()])
    supports = np.array(
        [int((e.our_strategy > 1e-6).sum()) for e in book.entries.values()]
    )
    selections = all_selections(reg.meta.team_size, reg.meta.picked_team_size)

    out: list[str] = []
    head = f"■ {len(book)} チーム"
    if seconds is not None:
        head += f"、{seconds / 60:.1f} 分"
    out.append(head)
    out.append(
        f"  均衡値 平均 {values.mean() * 100:.1f}%  中央 {np.median(values) * 100:.1f}%  "
        f"最小 {values.min() * 100:.1f}%  最大 {values.max() * 100:.1f}%"
    )
    out.append(
        f"  LP の双対ギャップ 最大 {gaps.max():.2e}、"
        f"反対称性の検査 最大 {anti.max():.2e}（どちらも 0 が要件）"
    )
    out.append(
        f"  均衡の支持 平均 {supports.mean():.2f} 通り / 90"
        f"（1 は純戦略。{int((supports == 1).sum())} チームがそう）"
    )
    # Which of our selections the field's equilibria name, and how often. Printed because
    # "the solver says bring these four against everyone" is a claim about the metagame
    # and should be readable as one.
    pooled = np.zeros(len(selections), dtype=np.float64)
    for entry in book.entries.values():
        pooled += entry.our_strategy
    pooled /= pooled.sum()
    out.append("  自陣の均衡選出の分布（相手を通じて平均）:")
    for index in np.argsort(-pooled)[:6]:
        if pooled[index] <= 1e-6:
            break
        selection = selections[index]

        def name(i: int) -> str:
            return namer(i) if namer is not None else str(i)

        out.append(
            f"    {pooled[index] * 100:5.1f}%  #{int(index):>2} "
            f"{'+'.join(name(i) for i in selection[:2])} / "
            f"{'+'.join(name(i) for i in selection[2:])}"
        )
    out.append("")
    out.append(coverage_table(book, selections))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--model",
        type=Path,
        nargs="+",
        default=[Path("data/models/value-gen2.pt")],
        help="one net, or several to average as one leaf. Several because a book solved\n"
        "from one training seed is largely that seed: three seeds of one configuration "
        "cost each other 2.88 to 5.86 points of the game's own value, on a scale where "
        "having no book at all costs about 15, while two disjoint two-net ensembles cost "
        "each other 1.14. Members cost 1.13x-1.19x one net, not Nx.",
    )
    ap.add_argument("--out", type=Path, default=None, help="default data/selection/<name>.jsonl.gz")
    ap.add_argument("--pool", default="all", help="all | phase2 | cut")
    ap.add_argument(
        "--classes",
        type=int,
        default=8,
        help="spread classes per opponent: independent draws of their six spreads from "
        "the belief the sheet leaves open. The cost is linear in this and the solve is "
        "sharded, so eight rather than four -- the class weights are 1/K and K draws is "
        "how finely their private investment is represented.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="solve only this many teams")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument(
        "--shards",
        type=int,
        default=1,
        help="split the field across n processes. Shard i takes teams[i::n] and writes "
        "its own part file; --merge then combines them.",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="combine the part files into one book and report on it. Solves nothing.",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="report on an existing book without solving or merging anything.",
    )
    ap.add_argument(
        "--text",
        type=Path,
        default=None,
        help="also write the report here as UTF-8. The Windows console mangles Japanese, "
        "and a report nobody can read is not a report.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="threads per process. 1 is right for a sharded run: torch otherwise opens a "
        "machine-sized pool per process and fourteen of those on eight cores is slower "
        "than one process. Raise it only for a single unsharded solve.",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    models = list(args.model)
    stem, model_name = book_stem(models), model_label(models)
    out = args.out or (selection_dir() / f"{args.roster}-{stem}.jsonl.gz")
    loc = localiser(reg, "ja")

    def namer(index: int) -> str:
        """Party index -> the Japanese name of that member of our six.

        Our roster is fixed, so a selection is a set of indices into it and printing the
        indices makes the report unreadable for the person who has to act on it.
        """
        species_id = roster.sets[index].species
        return loc.species(species_id) if loc else reg.species[species_id].name

    def emit(text: str) -> None:
        """Prints the report and, with --text, writes it as UTF-8 as well."""
        print(text, file=sys.stderr)
        if args.text is not None:
            args.text.parent.mkdir(parents=True, exist_ok=True)
            args.text.write_text(text + "\n", encoding="utf-8")
            print(f"（読める形: {args.text}）", file=sys.stderr)

    if args.report:
        emit(report(SelectionBook.read(out), reg, namer=namer))
        return

    if args.merge:
        merged = SelectionBook(
            roster=args.roster, model=model_name, format_id=reg.meta.format_id
        )
        parts = part_files(out)
        if not parts:
            raise SystemExit(f"no part files next to {out}")
        for part in parts:
            book = SelectionBook.read(part)
            book.require_roster(args.roster)
            # The header is rewritten from this process's own --model, which defaults to
            # value-gen2. A merge that forgot the flag would relabel the book with a leaf
            # that never touched it, and the file would still look right.
            if book.model and book.model != model_name:
                raise SystemExit(
                    f"{part.name} was solved with {book.model!r} but this merge would "
                    f"label it {model_name!r}; pass the same --model the shards used"
                )
            for entry in book.entries.values():
                merged.add(entry)
            print(f"  {part.name}: {len(book)} チーム", file=sys.stderr)
        merged.write(out)
        print(f"→ {out}（{len(merged)} チーム）", file=sys.stderr)
        emit(report(merged, reg, namer=namer))
        return

    chaos = find_cached_chaos(reg.meta.format_id)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    cached = find_cached_standings()
    if cached is None:
        raise SystemExit("no cached standings; run tools/fetch_standings.py 2026 worlds")
    standings = load_standings(cached, reg)
    teams = standings.pool(args.pool)

    missing = [m for m in models if not m.exists()]
    if missing:
        raise SystemExit(
            f"no model at {', '.join(str(m) for m in missing)}; the cells of the "
            "selection game are win probabilities and only a trained value function "
            "produces those."
        )
    encoder = Encoder(reg)
    nets, _metas = load_ensemble(models, encoder)
    device = torch.device(args.device)
    value = BatchedValue([n.to(device) for n in nets], encoder, device=device)

    target = out if args.shards == 1 else part_path(out, args.shard)
    done: set[str] = set()
    if target.exists():
        existing = SelectionBook.read(target)
        existing.require_roster(args.roster)
        done = set(existing.entries)
        print(f"再開: {len(done)} チーム分が既にある", file=sys.stderr)
    start_book(
        target, roster=args.roster, model=model_name, format_id=reg.meta.format_id
    )

    # Indexed before sharding, because the index is the seed: team 37 gets the same eight
    # spread classes whether it is solved by one process or by the eighth of fourteen.
    numbered = list(enumerate(teams))
    if args.limit is not None:
        numbered = numbered[: args.limit]
    mine = numbered[args.shard :: args.shards]
    pending = [(i, t) for i, t in mine if key_for_team(t) not in done]
    print(
        f"shard {args.shard}/{args.shards}: {len(pending)} チームを解く"
        f"（{args.pool} プール {len(teams)} 中、{args.classes} 配分クラス、{args.device}）",
        file=sys.stderr,
    )

    started = time.perf_counter()
    for count, (index, team) in enumerate(pending, start=1):
        entry, seconds, support = solve_one(
            reg,
            roster,
            prior,
            team,
            value,
            classes=args.classes,
            seed=args.seed,
            index=index,
            model=model_name,
        )
        append_entry(target, entry)
        elapsed = time.perf_counter() - started
        eta = elapsed / count * (len(pending) - count)
        print(
            f"[{count}/{len(pending)}] {team.place:>3}位 {team.player[:22]:<22} "
            f"勝率 {entry.value * 100:5.1f}%  支持 {support:>2}/90  "
            f"{seconds:4.1f}s  残り {eta / 60:.0f}分",
            file=sys.stderr,
        )

    if args.shards == 1:
        emit(
            report(
                SelectionBook.read(target),
                reg,
                time.perf_counter() - started,
                namer=namer,
            )
        )
    else:
        print(
            f"shard {args.shard} 完了: {len(pending)} チーム、"
            f"{(time.perf_counter() - started) / 60:.1f} 分 → {target.name}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
