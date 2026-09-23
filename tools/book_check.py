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

Games are also **played in both seats**. Every matchup runs once with our four at side 0
and once at side 1, and our result is read from our own side in each, so ``--games`` counts
matchups per arm and the run plays twice that. The paired differences above never needed
it -- both arms carried the same seat term and it subtracted out -- but the calibration
line does: it compares a measured win rate against an LP value with no seat term in it at
all. Our four sat at side 0 in every game this tool played before 2026-09-19, so whatever
the seat was worth went into that difference as calibration error. ``tools/seats.py`` has
the arithmetic and the reason the seat gap is printed rather than merely cancelled.

The anti-後出しジャンケン rule of :mod:`pokeuraou.selection_book` holds here too, and one
extra care is needed because this harness mixes rules: our selection is drawn from a
stream that never sees the opponent's spread class, and it is drawn before theirs.

**Open game only: a reference** (IKA-123). This tool has no hidden-bench path, so its
search is shown the opponent's four -- not the game that ships. It says so on stderr
when it runs, and `tools/agent_drift.py` lists it under KNOWN_DRIFT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# tools/ is sys.path[0] for a script run as `python tools/book_check.py`, and the Wilson
# interval is eight lines that should not exist twice. Nor is the seat arithmetic, which
# is how this tool and `selection_check` came to have none of it while `generation_match`
# had all of it.
from pool_matches import wilson
from seats import SEATS, SeatTally, play_paired, seat_label

from pokeuraou.benchflags import say_open_reference
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.selection_book import (
    ARMS,
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    BookEntry,
    SelectionBook,
    draw_arm,
    selection_dir,
)
from pokeuraou.selfplay import play_game
from pokeuraou.standings import TournamentTeam, find_cached_standings, load_standings
from pokeuraou.teams import all_selections, load_roster


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

    # Keyed by (game, SEAT) and holding OUR win, not side 0's. Two rows now share a game
    # index -- the two seats of one matchup -- so a key that named only the game would
    # call a perfectly good pair an overlap, and one that kept side 0's outcome would
    # average our result in one seat with the opponent's in the other.
    by_arm: dict[str, dict[tuple[int, int], float]] = {arm: {} for arm in ARMS}
    tallies: dict[str, SeatTally] = {arm: SeatTally(arm) for arm in ARMS}
    claimed: dict[int, float] = {}
    unfinished = 0
    # An overlap is two runs, not one. `{stem}.part*.jsonl` matches whatever is beside the
    # output, a shard's stride comes from the shard COUNT, and `by_arm[arm][key] = x`
    # makes a collision a silent overwrite decided by lexicographic filename order --
    # `part10` sorts before `part2`. Six leftover parts of an aborted 14-shard run sat
    # beside an 8-shard one and ten game indices were in both: the `gen/gen` arm merged to
    # 155/280 = 55.36% where the eight real shards alone give 154/280 = 55.00%, and
    # GENERATIONS.md recorded 55.4%. `tools/selection_check.py` has had this guard since
    # it was written; this copy did not.
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        if "seat" not in row:
            # Rows written before the swap are one seat's games -- our four at side 0 --
            # with nothing saying so. Pooling them with swapped rows would leave the seat
            # term in half the total and call the result seat-free.
            raise SystemExit(
                f"a row in {out.parent}/{out.stem}.part*.jsonl has no `seat`: it was "
                "written before the seat swap and every one of its games sat at side 0. "
                "Re-run those shards rather than merging them with swapped ones."
            )
        seat = int(row["seat"])
        key = (str(row["arm"]), int(row["game"]), seat)
        if key in seen:
            raise SystemExit(
                f"{key} appears in two of {len(parts)} part files beside {out}. That is "
                "two runs merged, not one run's shards -- delete the stale parts (check "
                "their mtimes and sizes) and merge again."
            )
        seen.add(key)
        tally = tallies.setdefault(str(row["arm"]), SeatTally(str(row["arm"])))
        won = tally.add(seat, row["outcome"])
        if won is None:
            unfinished += 1
            continue
        by_arm.setdefault(row["arm"], {})[(int(row["game"]), seat)] = float(won)
        claimed[int(row["game"])] = float(row["value"])

    print(
        f"■ {len(rows)} 座席ゲーム（打ち切り {unfinished}）、{len(parts)} シャード、"
        f"{len(SEATS)} 席"
    )
    print(f"  {'arm':>16}  {'seat-games':>6}  {'win':>7}  {'95% Wilson':>16}")
    for arm in ARMS:
        outcomes = by_arm.get(arm, {})
        if not outcomes:
            continue
        # Our wins in both seats, which is what `by_arm` holds -- the flip happened once,
        # in `seats.our_win`, when the row was read. `tools/match_result.py` states the
        # same rule from the other end: both seats report the named arm's wins, so the
        # total needs no second flip and putting one here is how a sign gets lost.
        wins = int(sum(outcomes.values()))
        rate, low, high = wilson(wins, len(outcomes))
        print(
            f"  {arm:>16}  {len(outcomes):>6}  {rate * 100:6.1f}%  "
            f"[{low * 100:5.1f}, {high * 100:5.1f}]"
        )
        # The gap needs no mirror and no matching pair of rules to be interpretable: the
        # two seats are the same matchups reseated, so a machine with no seat in it wins
        # the same games in both. `SeatTally.gap` has the algebra.
        for line in tallies[arm].seat_lines():
            print(line)

    def paired(a: str, b: str) -> tuple[float, float, int] | None:
        # Paired on (game, seat): the two arms played the same matchup from the same side,
        # so the difference has neither the opponent nor the seat in it.
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
        expected = float(np.mean([claimed[game] for game, _seat in both]))
        # The line the swap was for. An LP value has no seat term in it, so comparing it
        # against a rate measured with our four at side 0 in every game charged whatever
        # the seat is worth to the solver's calibration. `measured` is both seats now.
        print(
            f"  均衡値の校正: LP の主張 {expected * 100:.1f}% に対し実測 "
            f"{measured * 100:.1f}%（差 {(measured - expected) * 100:+.1f} ポイント、"
            "両席あわせた値なので席の項は入っていない）"
        )
        print(
            "  実測が大きく上なら、ソルバは相手の最善を過小評価している"
            "（弱い方策で学習した価値関数が誤る向き）"
        )
        seat_gap = tallies["book/book"].gap
        if not np.isnan(seat_gap):
            print(
                f"  そのうち席の項は {seat_gap * 100:+.1f} ポイント"
                "（side 0 のときの勝率 - side 1 のとき。上の差からは既に落ちている）"
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
    ap.add_argument(
        "--games",
        type=int,
        default=280,
        help="matchups per arm, across shards. Each is played in BOTH seats, so an arm "
        "plays twice this many games -- the counting `tools/generation_match.py` uses.",
    )
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
    say_open_reference()

    if args.merge:
        merge(args.out)
        return

    # Below the merge branch, so that reading back a finished run -- which is arithmetic
    # over a JSONL file and nothing else -- does not want a 3 GB CUDA wheel. Deferring the
    # leaf like this is what a dozen tools here already do, and it is what lets the seat
    # aggregation be tested at all: everything above this line imports without torch.
    import torch

    from pokeuraou.value import BatchedValue, load_ensemble

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
        f"shard {args.shard}/{args.shards}: {len(mine)} 対戦 x {len(ARMS)} arm x "
        f"{len(SEATS)} 席、{len(covered)}/{len(pool)} チームを被覆、"
        f"ε={args.epsilon}, T={args.temperature}",
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

                def one_seat(
                    seat: int,
                    side0: list,
                    side1: list,
                    *,
                    arm: str = arm,
                    arm_index: int = arm_index,
                    game: int = game,
                    team: TournamentTeam = team,
                    entry: BookEntry = entry,
                    class_index: int = class_index,
                    own_pick: tuple[int, ...] = own_pick,
                    foe_pick: tuple[int, ...] = foe_pick,
                ) -> float | None:
                    # One stream per (game, arm), NOT per seat: the two seats are supposed
                    # to be the same matchup mirrored and to differ in nothing else, which
                    # is `tools/cycle_match.py`'s rule ("same seed for both seats") for the
                    # same reason.
                    record = play_game(
                        reg,
                        np.random.default_rng([args.seed, game, arm_index, 7]),
                        side0,
                        side1,
                        team.player,
                        objective=OBJECTIVES["hp-share"],
                        search_limit=args.limit,
                        max_turns=args.max_turns,
                        evaluate=evaluate,
                        rank_by_leaf=args.rank_by_leaf,
                        open_information=True,
                    )
                    handle.write(
                        json.dumps(
                            {
                                "arm": arm,
                                "game": game,
                                # Which side OUR four sat on. `outcome` stays SIDE 0's, as
                                # it is everywhere else here, and the flip lives once in
                                # `seats.our_win` where the row is read back.
                                "seat": seat,
                                "player": team.player,
                                "place": team.place,
                                "classIndex": class_index,
                                "ownPick": list(own_pick),
                                "foePick": list(foe_pick),
                                "outcome": record.outcome,
                                "turns": record.turns,
                                "value": entry.value,
                                # What actually played. generation_match records this and
                                # stayed correct for a year; the two tools that did not
                                # both drifted away from the generation path without
                                # anyone noticing -- this one narrowed by damage after
                                # generation moved to the leaf, and selection_check played
                                # hp-share while comparing itself against the value
                                # function's claim. A tool that writes down its agent is a
                                # tool whose agent gets checked.
                                "provenance": {
                                    "kind": "book-check",
                                    "leaf": "+".join(m.name for m in models),
                                    "limit": args.limit,
                                    "ranking": (
                                        "leaf" if args.rank_by_leaf else "damage"
                                    ),
                                    "book": str(book_path.name),
                                    # Which seat, in the spelling `tools/match_result.py`
                                    # parses: it reads the trailing "side 0"/"side 1" and
                                    # prints nothing rather than guess.
                                    "seat": seat_label(arm, seat),
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    handle.flush()
                    return record.outcome

                play_paired(
                    [roster.sets[i] for i in own_pick],
                    [foe_six[j] for j in foe_pick],
                    one_seat,
                )
            if done % 5 == 0:
                print(f"  {done}/{len(mine)}", file=sys.stderr)


if __name__ == "__main__":
    main()
