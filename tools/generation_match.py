"""Plays one generation's search against another's, which is the only fair comparison.

Validation AUC cannot compare generations. Each one plays differently, so each generates a
different distribution of positions, and a higher AUC may only mean "this generation
created positions that are easier to judge". Winning games against the previous generation
is not confounded that way.

Both sides get the same teams, the same selections and the same seed; only the leaf
evaluation differs. Sides are swapped halfway, because a residual seat advantage would
otherwise be credited to whichever generation sat in the better seat -- the speed-tie bug
was exactly that, and it was worth 9 points in a mirror.

    uv run --group learn python tools/generation_match.py --value data/models/value-worlds.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.provenance import open_games, provenance, write_game
from pokeuraou.selfplay import play_game
from pokeuraou.standings import find_cached_standings, load_standings, sample_standings_team
from pokeuraou.teams import all_selections, load_roster
from pokeuraou.value import BatchedValue, load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--value", type=Path, default=Path("data/models/value-worlds.pt"))
    ap.add_argument("--games", type=int, default=200, help="games per seat, so twice this in total")
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="the older generation's *model*. Given, the match is model against model, "
        "which is what every generation after the second needs -- the first comparison "
        "was a value function against hp-share and that is all --objective can express. "
        "`play_game` already takes a leaf per side.",
    )
    ap.add_argument("--objective", default="hp-share", help="the leaf the older generation used")
    ap.add_argument(
        "--depth",
        type=int,
        default=1,
        help="search depth for the arm under test. With --baseline pointing at the same "
        "model, --depth 2 --baseline-depth 1 makes this a pure depth comparison: same "
        "leaf, same width, same teams, same seed, one side looking a ply further.",
    )
    ap.add_argument("--baseline-depth", type=int, default=1, help="depth for the other arm")
    ap.add_argument(
        "--baseline-limit",
        type=int,
        default=None,
        help="candidate width for the other arm; defaults to --limit. With the same model "
        "on both sides this makes the match a pure width comparison, which is how "
        "'does a wider menu actually win' gets answered without building anything.",
    )
    ap.add_argument(
        "--rank-leaf",
        action="store_true",
        help="the arm under test ranks candidates with the leaf instead of the damage "
        "score. With the same model and width on both sides this is a pure ranking "
        "comparison.",
    )
    ap.add_argument("--baseline-rank-leaf", action="store_true", help="same for the other arm")
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="append one JSON line per seat as it completes, so a stopped run is not lost "
        "and several runs can be pooled by reading the files.",
    )
    ap.add_argument(
        "--report-every",
        type=int,
        default=25,
        help="print a progress line every N games. Redirected stdout is fully buffered, so "
        "these are flushed explicitly or they are invisible until the process exits.",
    )
    ap.add_argument(
        "--games-out",
        type=Path,
        default=None,
        help="also append every game played, as self-play-shaped JSONL with a provenance "
        "block. These carry real outcomes and are training data that has already been "
        "paid for; the block records the leaf and width *per side*, because the two are "
        "deliberately mismatched here and a dataset must be able to say so.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="threads per worker. 1 is right for a parallel run: N processes each spawning "
        "a pool on the same cores is slower than one process. Raise it for a single run.",
    )
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    encoder = Encoder(reg)
    net, meta = load_model(args.value, encoder)
    device = torch.device(args.device)
    value = BatchedValue(net.to(device), encoder, device=device)
    objective = OBJECTIVES[args.objective]
    baseline = None
    if args.baseline is not None:
        if not args.baseline.exists():
            raise SystemExit(f"no baseline model at {args.baseline}")
        base_net, base_meta = load_model(args.baseline, encoder)
        baseline = BatchedValue(base_net.to(device), encoder, device=device)

    print(
        f"new leaf: {args.value.name} (trained on {meta.get('games', '?')} games, "
        f"val AUC {meta.get('val_auc', float('nan')):.4f})",
        file=sys.stderr,
    )
    if baseline is not None:
        print(
            f"old leaf: {args.baseline.name} (trained on "
            f"{base_meta.get('games', '?')} games, val AUC "
            f"{base_meta.get('val_auc', float('nan')):.4f})",
            file=sys.stderr,
        )
        print(
            "  同じ探索・同じチーム・同じ選出・同じ乱数で、葉だけが違います。"
            "検証 AUC は世代間で比較できないので、勝率だけが判定です",
            file=sys.stderr,
        )
    else:
        print(f"old leaf: {args.objective}（評価軸）", file=sys.stderr)

    # Seat A: the value function is side 0. Seat B: it is side 1. Both seats use our six
    # against the field, so a seat advantage cancels when the two are combined.
    print(f"\n  {'seat':>30}  {'games':>6}  {'new win':>8}  {'95%':>6}  {'s/game':>7}")
    wins = played = 0
    games_file = open_games(args.games_out)
    new_name = args.value.name
    old_name = args.baseline.name if args.baseline else args.objective
    # The seat label names the model, not "gen2". It is stamped into every recorded
    # game's provenance, and a reader of that dataset a month from now has no way to know
    # which generation "gen2" meant on the day the match ran -- the `leaves` pair is
    # authoritative, but a label that contradicts it is worse than no label.
    other_limit = args.limit if args.baseline_limit is None else args.baseline_limit
    tags = ""
    if args.depth != args.baseline_depth:
        tags += f"@d{args.depth}"
    if other_limit != args.limit:
        tags += f"@w{args.limit}"
    if args.rank_leaf != args.baseline_rank_leaf:
        tags += "@leafrank" if args.rank_leaf else "@damagerank"
    arm = f"{new_name}{tags}" if tags else new_name
    for seat, leaves, depths, limits, ranks in (
        (f"{arm} = side 0", (value, baseline), (args.depth, args.baseline_depth),
         (args.limit, other_limit), (args.rank_leaf, args.baseline_rank_leaf)),
        (f"{arm} = side 1", (baseline, value), (args.baseline_depth, args.depth),
         (other_limit, args.limit), (args.baseline_rank_leaf, args.rank_leaf)),
    ):
        side_leaves = (
            (new_name, old_name) if leaves[0] is value else (old_name, new_name)
        )
        rng = np.random.default_rng(args.seed)
        seat_wins = seat_played = unfinished = 0
        started = time.perf_counter()
        for played_so_far in range(args.games):
            if played_so_far and played_so_far % args.report_every == 0:
                rate = seat_wins / seat_played * 100 if seat_played else float("nan")
                print(
                    f"    [{seat}] {played_so_far}/{args.games} played, "
                    f"new {rate:.1f}%, {(time.perf_counter() - started) / played_so_far:.2f} s/game",
                    flush=True,
                )
            team = pool[int(rng.integers(len(pool)))]
            foe_six = sample_standings_team(rng, reg, prior, team)
            own_pick = selections[int(rng.integers(len(selections)))]
            foe_pick = selections[int(rng.integers(len(selections)))]
            record = play_game(
                reg,
                rng,
                [roster.sets[i] for i in own_pick],
                [foe_six[j] for j in foe_pick],
                seat,
                objective=objective,
                search_limit=limits,
                max_turns=args.max_turns,
                evaluate=leaves,
                depth=depths,
                rank_by_leaf=ranks,
            )
            if record.outcome is None:
                unfinished += 1
                continue
            write_game(
                games_file,
                record,
                objective=f"value:{args.value.stem}",
                search_limit=args.limit,
                source=provenance(
                    "generation-match",
                    seat=seat,
                    leaves=side_leaves,
                    limits=limits,
                    depths=depths,
                    rankings=tuple("leaf" if r else "damage" for r in ranks),
                    note=(
                        f"search depth {depths[0]} vs {depths[1]} by side"
                        if depths[0] != depths[1]
                        else ""
                    ),
                ),
            )
            seat_played += 1
            # `outcome` is side 0's result, so flip it when the value function sits at 1.
            new_won = record.outcome > 0.5 if leaves[0] is value else record.outcome < 0.5
            seat_wins += int(new_won)
        elapsed = time.perf_counter() - started
        rate = seat_wins / seat_played if seat_played else float("nan")
        half = (
            1.96 * (rate * (1 - rate) / seat_played) ** 0.5
            if seat_played
            else float("nan")
        )
        print(
            f"  {seat:>30}  {seat_played:>6}  {rate * 100:>7.1f}%  +-{half * 100:.1f}  "
            f"{elapsed / args.games:>7.2f}   (打ち切り {unfinished})",
            flush=True,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "seat": seat,
                            "seed": args.seed,
                            "model": args.value.name,
                            "baseline": (
                                args.baseline.name if args.baseline else args.objective
                            ),
                            "objective": args.objective,
                            "limit": args.limit,
                            "depth": args.depth,
                            "baselineDepth": args.baseline_depth,
                            "played": seat_played,
                            "gen2_wins": seat_wins,
                            "unfinished": unfinished,
                            "seconds": elapsed,
                        }
                    )
                    + "\n"
                )
        wins += seat_wins
        played += seat_played

    rate = wins / played if played else float("nan")
    half = 1.96 * (rate * (1 - rate) / played) ** 0.5 if played else float("nan")
    print(
        f"\n  両席あわせて {played} ゲーム: {arm} の勝率 {rate * 100:.1f}% +-{half * 100:.1f}",
        flush=True,
    )
    if rate - half > 0.5:
        print(f"  → {arm} が有意に強い。次世代のデータ生成に使える。")
    elif rate + half < 0.5:
        print(
            f"  → {arm} が有意に弱い。この設定は採用しない。"
            "生成する前に原因を出すべき。"
        )
    else:
        print(
            "  → 差は有意でない。区間の幅が "
            f"{half * 200:.1f} ポイントなので、判定するにはゲーム数を増やす必要がある。"
        )


if __name__ == "__main__":
    main()
