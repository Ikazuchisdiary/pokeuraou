"""Solves the 6->4 selection once per opponent in the field and caches the answers.

    uv run --group learn python tools/solve_selection_book.py --model data/models/value-gen2.pt
    uv run --group learn python tools/solve_selection_book.py --limit 5 --classes 2   # smoke

One entry per team sheet in the tournament standings: our equilibrium mixture over the 90
ordered selections, the opponent's mixture per spread class, the class weights and the
equilibrium value. Self-play then draws both sides' selections from the entry instead of
uniformly, which costs a lookup per game rather than a 12-second solve.

The cache is keyed on the opponent's *six* -- their public team sheet -- and never on the
four they bring. See :mod:`pokeuraou.selection_book` for that and the three other places
where a cache like this could turn into 後出しジャンケン.

Resumable: run it again and it skips the teams already in the book, so a stopped run
costs only the team it was solving.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.selection import SpreadClass, book_entry, solve_selection
from pokeuraou.selection_book import (
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    SelectionBook,
    append_entry,
    key_for_team,
    perplexity,
    selection_dir,
    start_book,
)
from pokeuraou.standings import (
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import load_roster
from pokeuraou.value import BatchedValue, load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--model", type=Path, default=Path("data/models/value-gen2.pt"))
    ap.add_argument("--out", type=Path, default=None, help="default data/selection/<name>.jsonl.gz")
    ap.add_argument("--pool", default="all", help="all | phase2 | cut")
    ap.add_argument(
        "--classes",
        type=int,
        default=4,
        help="spread classes per opponent. Each is a full 90x90 matrix, so the cost is "
        "linear in this. Four is what the single-team tool defaults to.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="solve only this many teams")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)

    chaos = find_cached_chaos(reg.meta.format_id)
    if chaos is None:
        raise SystemExit("no cached usage stats; run tools/fetch_priors.py first")
    prior = load_chaos(chaos, reg)
    cached = find_cached_standings()
    if cached is None:
        raise SystemExit("no cached standings; run tools/fetch_standings.py 2026 worlds")
    standings = load_standings(cached, reg)
    teams = standings.pool(args.pool)

    if not args.model.exists():
        raise SystemExit(
            f"no model at {args.model}; the cells of the selection game are win "
            "probabilities and only a trained value function produces those."
        )
    encoder = Encoder(reg)
    net, meta = load_model(args.model, encoder)
    value = BatchedValue(net.to(torch.device(args.device)), encoder, device=torch.device(args.device))

    out = args.out or (selection_dir() / f"{args.roster}-{args.model.stem}.jsonl.gz")
    done: set[str] = set()
    if out.exists():
        existing = SelectionBook.read(out)
        if existing.roster and existing.roster != args.roster:
            raise SystemExit(
                f"{out} was solved for roster {existing.roster!r}, not {args.roster!r}. "
                "The 90 selection indices refer to a party order, so the two are not "
                "interchangeable -- write to a different --out."
            )
        done = set(existing.entries)
        print(f"再開: {len(done)} チーム分が既にある", file=sys.stderr)
    start_book(out, roster=args.roster, model=args.model.name, format_id=reg.meta.format_id)

    pending = [t for t in teams if key_for_team(t) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"{len(pending)} チームを解く（{args.pool} プール {len(teams)} 中、"
        f"{args.classes} 配分クラス、{args.device}）",
        file=sys.stderr,
    )

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    values: list[float] = []
    supports: list[int] = []
    coverage: list[float] = []
    # Pooled over teams: what share of the 90 our side actually plays across the whole
    # field. Per-team width is not the number that matters -- if every team's equilibrium
    # names the same selection, generation visits one of our 90 no matter how many teams
    # it faces, and the next solve has 89 columns of cells nobody ever saw.
    pooled_equilibrium = np.zeros(90, dtype=np.float64)
    pooled_mixture = np.zeros(90, dtype=np.float64)
    for done_count, team in enumerate(pending, start=1):
        classes = [
            SpreadClass(
                weight=1.0 / args.classes,
                sets=tuple(sample_standings_team(rng, reg, prior, team)),
                label=f"class {k + 1}",
            )
            for k in range(args.classes)
        ]
        analysis = solve_selection(reg, roster.sets, classes, value)
        entry = book_entry(
            analysis,
            key=key_for_team(team),
            player=team.player,
            place=team.place,
            model=args.model.name,
        )
        append_entry(out, entry)
        values.append(entry.value)
        supports.append(int(len(analysis.equilibrium.row_support())))
        mixture = entry.our_mixture(
            epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE
        )
        coverage.append(perplexity(mixture))
        pooled_equilibrium += entry.our_strategy
        pooled_mixture += mixture
        elapsed = time.perf_counter() - started
        eta = elapsed / done_count * (len(pending) - done_count)
        print(
            f"[{done_count}/{len(pending)}] {team.place:>3}位 {team.player[:24]:<24} "
            f"勝率 {entry.value * 100:5.1f}%  支持 {supports[-1]:>2}/90  "
            f"生成の実効幅 {coverage[-1]:4.1f}  {analysis.seconds:4.1f}s  "
            f"残り {eta / 60:.0f}分",
            file=sys.stderr,
        )

    if not values:
        print("解くものがない（すべてキャッシュ済み）", file=sys.stderr)
        return
    arr = np.array(values)
    print("", file=sys.stderr)
    print(f"■ {len(values)} チーム、{(time.perf_counter() - started) / 60:.1f} 分", file=sys.stderr)
    print(
        f"  均衡値 平均 {arr.mean() * 100:.1f}%  中央 {np.median(arr) * 100:.1f}%  "
        f"最小 {arr.min() * 100:.1f}%  最大 {arr.max() * 100:.1f}%",
        file=sys.stderr,
    )
    print(
        f"  均衡の支持 平均 {np.mean(supports):.2f} 通り"
        f"（1 なら純戦略。生成が一様なら 90）",
        file=sys.stderr,
    )
    print(
        f"  生成の実効幅 平均 {np.mean(coverage):.1f} 通り"
        f"（ε={DEFAULT_EPSILON}, T={DEFAULT_TEMPERATURE} の混合。"
        "次世代の解に必要なセルをどれだけ踏むかの目安）",
        file=sys.stderr,
    )
    print(
        f"  自陣の選出の広さ（全チームで平均した分布）: 均衡 "
        f"{perplexity(pooled_equilibrium):.1f} 通り、生成 "
        f"{perplexity(pooled_mixture):.1f} 通り / 90",
        file=sys.stderr,
    )
    top = np.argsort(-pooled_mixture)[:5]
    share = pooled_mixture / pooled_mixture.sum()
    print(
        "  最も多い選出: "
        + "、".join(f"#{int(i)} {share[i] * 100:.1f}%" for i in top),
        file=sys.stderr,
    )
    print(f"  → {out}", file=sys.stderr)
    print(
        "  注: 生成はこの混合から引くので、生成の勝率は均衡値そのものではない。"
        "別の量として読むこと。",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
