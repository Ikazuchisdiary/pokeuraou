"""Does the cached selection equilibrium win more games than a uniform draw -- over the
whole field, not against one opponent?

``tools/selection_check.py`` asks this of a single team sheet. That is the right question
to ask first and the wrong one to answer with, because the answer against the 1st-place
team is one draw from a distribution: the solver's advice can be worth 10 points against
one composition and negative against another, and generation faces all 394. This runs the
same three arms against the field the book was solved for, plus a fourth that plays exactly
what generation will play.

    bash tools/book_check_parallel.sh                    # 14 shards
    uv run --group learn python tools/book_check.py --merge --out data/matches/book-check.jsonl

Four arms, and the reason each exists:

- ``uniform/uniform`` -- the baseline generation used until now;
- ``book/uniform`` -- our side takes the solver's advice, the opponent does not. The
  difference from the baseline is *the value of the advice*, and it is the number that says
  whether the book should be used at all. Sensitive to the optimiser's curse by
  construction: solving takes something close to an argmax over 8,100 estimated cells, the
  operation most sensitive to their error, so a value function that merely *evaluates* well
  can still lose here;
- ``book/book`` -- both sides play the equilibrium. This is the arm with a *prediction*
  attached: the LP claims a value, and the measured rate either matches it or does not. A
  large positive gap for us means the solver underestimates the opponent's best reply,
  which is the direction a value function trained against weaker play gets wrong;
- ``gen/gen`` -- both sides draw from equilibrium-plus-exploration, which is what
  generation actually does. Its rate is the number a generation run should be compared
  against; comparing a generation's win rate to the equilibrium value compares two
  different quantities.

Games are **paired across arms**: game *g* draws the same opponent and the same spread
class in every arm, so the arms differ only in the selection rule and the difference can be
measured per game rather than between two independent averages.

The anti-後出しジャンケン rule of :mod:`pokeuraou.selection_book` holds here too, and one
extra care is needed because this harness mixes rules: our selection is drawn from a
stream that never sees the opponent's spread class, and it is drawn before theirs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# tools/ is sys.path[0] for a script run as `python tools/book_check.py`, and the Wilson
# interval is eight lines that should not exist twice.
from pool_matches import wilson

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.selection_book import (
    ARMS,
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    SelectionBook,
    draw_arm,
    selection_dir,
)
from pokeuraou.selfplay import play_game
from pokeuraou.standings import find_cached_standings, load_standings
from pokeuraou.teams import all_selections, load_roster
from pokeuraou.value import BatchedValue, load_ensemble


def merge(out: Path) -> None:
    rows: list[dict] = []
    parts = sorted(out.parent.glob(f"{out.stem}.part*.jsonl"))
    for part in parts:
        rows.extend(
            json.loads(line) for line in part.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if not rows:
        raise SystemExit(f"no rows in {out.parent}/{out.stem}.part*.jsonl")
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    by_arm: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
    claimed: dict[int, float] = {}
    unfinished = 0
    for row in rows:
        if row["outcome"] is None:
            unfinished += 1
            continue
        by_arm.setdefault(row["arm"], {})[row["game"]] = float(row["outcome"])
        claimed[row["game"]] = float(row["value"])

    print(f"■ {len(rows)} 対戦（打ち切り {unfinished}）、{len(parts)} シャード")
    print(f"  {'arm':>16}  {'games':>6}  {'win':>7}  {'95% Wilson':>16}")
    for arm in ARMS:
        outcomes = by_arm.get(arm, {})
        if not outcomes:
            continue
        wins = int(sum(outcomes.values()))
        rate, low, high = wilson(wins, len(outcomes))
        print(
            f"  {arm:>16}  {len(outcomes):>6}  {rate * 100:6.1f}%  "
            f"[{low * 100:5.1f}, {high * 100:5.1f}]"
        )

    def paired(a: str, b: str) -> tuple[float, float, int] | None:
        shared = sorted(set(by_arm.get(a, {})) & set(by_arm.get(b, {})))
        if len(shared) < 2:
            return None
        diff = np.array([by_arm[a][g] - by_arm[b][g] for g in shared])
        return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff))), len(diff)

    print("")
    advice = paired("book/uniform", "uniform/uniform")
    if advice is not None:
        mean, se, n = advice
        print(
            f"  助言の価値: {mean * 100:+.1f} ポイント "
            f"[{(mean - 1.96 * se) * 100:+.1f}, {(mean + 1.96 * se) * 100:+.1f}]"
            f"（同じ相手・同じ配分で対にした {n} 組、相手は一様のまま）"
        )
        if mean - 1.96 * se <= 0.0:
            print(
                "  → 0 を含む。均衡選出が一様に勝つとまだ言えない。"
                "セルの推定誤差を最適化が拾っている可能性（optimiser's curse）が残る"
            )
        else:
            print("  → 0 を上回る。助言として出せる")

    both = by_arm.get("book/book", {})
    if both:
        measured = sum(both.values()) / len(both)
        expected = float(np.mean([claimed[g] for g in both]))
        print(
            f"  均衡値の校正: LP の主張 {expected * 100:.1f}% に対し実測 "
            f"{measured * 100:.1f}%（差 {(measured - expected) * 100:+.1f} ポイント）"
        )
        print(
            "  実測が大きく上なら、ソルバは相手の最善を過小評価している"
            "（弱い方策で学習した価値関数が誤る向き）"
        )
    gen = by_arm.get("gen/gen", {})
    if gen:
        rate = sum(gen.values()) / len(gen)
        print(
            f"  生成が実際に出す勝率の見込み: {rate * 100:.1f}%"
            "（均衡値ではない。生成のログはこの数字と比べる）"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--model",
        type=Path,
        nargs="+",
        default=[Path("data/models/value-gen2.pt")],
        help="the leaf that plays. Several are averaged as one, and must be the same "
        "several that solved the book -- this compares a book's claim with the board, "
        "and a book played by a different leaf compares two agents instead.",
    )
    ap.add_argument("--book", type=Path, default=None)
    ap.add_argument("--games", type=int, default=280, help="games per arm, across shards")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument(
        "--rank-by-leaf",
        action="store_true",
        help="order the narrowing by the leaf rather than by expected damage. Generation "
        "moved to this ordering and the measuring tools did not, so every board number "
        "taken here was played by an agent narrowing worse than the one being shipped. "
        "The record prices the difference at 61.3%% of the equilibrium's mass kept against "
        "86.4%%, and 1.78 points given up against 0.55.",
    )
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data/matches/book-check.jsonl"))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
    args = ap.parse_args()

    if args.merge:
        merge(args.out)
        return

    torch.set_num_threads(args.torch_threads)
    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    models = list(args.model)
    stem = (
        models[0].stem if len(models) == 1 else f"{models[0].stem}-ens{len(models)}"
    )
    book_path = args.book or (selection_dir() / f"{args.roster}-{stem}.jsonl.gz")
    book = SelectionBook.read(book_path)
    book.require_roster(args.roster)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    covered = [team for team in pool if book.get(team) is not None]
    if not covered:
        raise SystemExit(f"{book_path} covers none of the {len(pool)} teams in the pool")

    encoder = Encoder(reg)
    nets, _metas = load_ensemble(models, encoder)
    device = torch.device(args.device)
    evaluate = BatchedValue([n.to(device) for n in nets], encoder, device=device)
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    target = (
        args.out
        if args.shards == 1
        else args.out.parent / f"{args.out.stem}.part{args.shard}.jsonl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    mine = list(range(args.games))[args.shard :: args.shards]
    print(
        f"shard {args.shard}/{args.shards}: {len(mine)} ゲーム x {len(ARMS)} arm、"
        f"{len(covered)}/{len(pool)} チームを被覆、ε={args.epsilon}, T={args.temperature}",
        file=sys.stderr,
    )

    with target.open("w", encoding="utf-8") as handle:
        for done, game in enumerate(mine, start=1):
            # Shared across arms: the opponent and their investment. Seeded by the game
            # index alone, so arm order and shard layout cannot change who is faced.
            shared = np.random.default_rng([args.seed, game])
            team = covered[int(shared.integers(len(covered)))]
            entry = book.get(team)
            assert entry is not None
            class_index = int(
                np.searchsorted(
                    np.cumsum(entry.class_weights / entry.class_weights.sum()),
                    shared.random(),
                    side="right",
                )
            )
            class_index = min(class_index, len(entry.class_sets) - 1)
            foe_six = entry.class_sets[class_index]

            for arm_index, arm in enumerate(ARMS):
                # A separate stream per arm, so an arm that consumes more draws cannot
                # shift another's, and one that never looks at the class cannot be
                # influenced by its value.
                pick_rng = np.random.default_rng([args.seed, game, arm_index])
                ours, theirs = draw_arm(
                    arm,
                    entry,
                    class_index,
                    pick_rng,
                    epsilon=args.epsilon,
                    temperature=args.temperature,
                )
                own_pick = selections[ours]
                foe_pick = selections[theirs]
                record = play_game(
                    reg,
                    np.random.default_rng([args.seed, game, arm_index, 7]),
                    [roster.sets[i] for i in own_pick],
                    [foe_six[j] for j in foe_pick],
                    team.player,
                    objective=OBJECTIVES["hp-share"],
                    search_limit=args.limit,
                    max_turns=args.max_turns,
                    evaluate=evaluate,
                    rank_by_leaf=args.rank_by_leaf,
                )
                handle.write(
                    json.dumps(
                        {
                            "arm": arm,
                            "game": game,
                            "player": team.player,
                            "place": team.place,
                            "classIndex": class_index,
                            "ownPick": list(own_pick),
                            "foePick": list(foe_pick),
                            "outcome": record.outcome,
                            "turns": record.turns,
                            "value": entry.value,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
            if done % 5 == 0:
                print(f"  {done}/{len(mine)}", file=sys.stderr)


if __name__ == "__main__":
    main()
