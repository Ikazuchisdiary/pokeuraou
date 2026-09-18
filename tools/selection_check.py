"""Does the selection solver's advice actually win more games?

Worth asking of any solver, and worth asking here for a reason that turned out to be
instructive.

The equilibrium never brings Charizard, and prices the team's own signature Charizard +
Venusaur lead at -24 points. That looked like it contradicted the self-play data, which
says leaving Charizard behind is the *worst* thing our side can do: 55.1%% against 78.5%% for
leaving Toxapex behind. It does not contradict it. The 55.1%% is a **marginal** statistic --
uniform selection averaged over all 394 opponents -- and the solver answers a
**conditional** question: what to bring against one specific team. A member can be the most
valuable on average and the wrong pick in a particular matchup, and confusing the two is a
straightforward error of reasoning rather than a defect in either number.

Independently: not bringing Charizard in this composition's mirror is known metagame
knowledge, which the value function reproduced from outcomes alone. That makes this tool a
validation rather than a suspicion -- and the mirror is the arm with an answer that does not
come from the model.

The reason to keep measuring anyway is the optimiser's curse. The cells come from a value
function whose turn-1 AUC is 0.74, so it ranks turn-1 positions only moderately well, and
solving a game takes something close to the argmax over 8,100 of those estimates -- the
operation most sensitive to their error. A function good enough to *evaluate* a position is
not automatically good enough to *optimise over* a matrix of them, and only play settles it.

Our side draws its selection from the equilibrium in one arm and uniformly in the other,
against the same opponent drawing uniformly in both, with the same seed.

    uv run --group learn python tools/selection_check.py --mirror --games 300
    uv run --group learn python tools/selection_check.py --place 1 --games 300
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
from pokeuraou.names import localiser
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.regulation import to_id
from pokeuraou.selection import SpreadClass, solve_selection
from pokeuraou.selfplay import play_game
from pokeuraou.standings import find_cached_standings, load_standings, sample_standings_team
from pokeuraou.teams import all_selections, load_roster
from pokeuraou.value import BatchedValue, load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--model", type=Path, default=Path("data/models/value-worlds.pt"))
    ap.add_argument("--place", type=int, default=1)
    ap.add_argument("--games", type=int, default=300, help="games per arm")
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--lang", default="ja")
    ap.add_argument(
        "--mirror",
        action="store_true",
        help="play our six against itself, spreads included. This is the case with an "
        "independent answer: the mirror of this composition is known not to want Charizard, "
        "so the equilibrium arm reproducing that is a check against human knowledge rather "
        "than against the model's own estimates.",
    )
    ap.add_argument(
        "--force-lead",
        default=None,
        help="two species, comma separated, to put in the lead -- 'charizard,incineroar'. "
        "Adds an arm that plays the best selection *with that lead* against the same "
        "equilibrium opponent as the equilibrium arm, which is how a claim about human "
        "practice gets tested against the solver's advice rather than argued with. The "
        "book brings Incineroar in 71.5%% of its mass and leads it in 0.0%%, and the human "
        "report is that leading it is the common line; Intimidate and Fake Out are both "
        "lead effects, so a value function that under-prices them would produce exactly "
        "the bring-but-bench distribution observed.",
    )
    ap.add_argument(
        "--force-selection",
        action="append",
        default=None,
        help="a whole selection to test, four species comma separated, lead first: "
        "'charizard,incineroar,toxapex,venusaur'. Repeatable, one arm each, all against "
        "the same equilibrium opponent as the equilibrium arm. --force-lead fixes only "
        "the front two and lets the solver pick the back two by equilibrium mass, which "
        "is meaningless when the lead carries no mass; naming all four tests a line as "
        "played rather than a lead with an arbitrary bench.",
    )
    ap.add_argument(
        "--hide-bench",
        action="store_true",
        help="play the condition that actually exists: the opponent's six is public "
        "and which four they brought is not. Without it the search is handed side 1's "
        "whole four from turn 1, which is the assumption G2 names as the reason the "
        "solver is optimistic about its own side -- and the assumption the recorded "
        "-21.8 point miss on place 109 was measured under.",
    )
    ap.add_argument(
        "--only-arm",
        default=None,
        help="play only the arms whose name contains this, e.g. '均衡 vs 均衡'. The "
        "calibration question needs that one arm, and the other two cost two thirds of "
        "the run.",
    )
    ap.add_argument(
        "--rank-by-leaf",
        action="store_true",
        help="order the narrowing by the leaf instead of by expected damage. Measured on "
        "the turn-1 node of place 305: the damage ordering keeps none of the equilibrium's "
        "43.7%% mass on moving a slot out of a 4x Blizzard, and solves to 0.2779; the leaf "
        "ordering keeps 61.6%% of it and solves to 0.3260, against 0.3256 for the full "
        "110x136 legal set. Generation moved to this ordering; this tool had not.",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument(
        "--shards",
        type=int,
        default=1,
        help="split the games across n processes. Shard i plays game indices i::n, "
        "and because each game seeds from [seed, index] the split changes nothing "
        "about which games are played or how the arms line up.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write one row per game here, so shards can be merged and so a paired "
        "interval can be taken later. Shard i writes <out>.part<i>.",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="read the part files beside --out and report, playing nothing.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg

    if args.merge:
        if args.out is None:
            raise SystemExit("--merge needs --out to know which part files to read")
        parts = sorted(args.out.parent.glob(f"{args.out.stem}.part*{args.out.suffix}"))
        if not parts:
            raise SystemExit(f"no part files beside {args.out}")
        claimed = None
        seen_headers: set[str] = set()
        tally: dict[str, list[int]] = {}
        seen: set[tuple[str, int]] = set()
        for part in parts:
            for line in part.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if "header" in row:
                    header = row["header"]
                    claimed = header["claimed"]
                    # Printed, not merely stored. A provenance nobody reads is a
                    # provenance that does not stop the next drift.
                    seen_headers.add(
                        f"place {header.get('place', '?')}"
                        f"  leaf {header.get('leaf', '?')}"
                        f"  width {header.get('limit', '?')}"
                        f"  narrowing {header.get('ranking', '?')}"
                        f"  {header.get('information', '?')}"
                        f"  classes {header.get('classes', '?')}"
                        f"  seed {header.get('seed', '?')}"
                        + ("  MIRROR" if header.get("mirror") else "")
                    )
                    continue
                key = (row["arm"], int(row["game"]))
                if key in seen:
                    raise SystemExit(f"{key} appears in two shards; the split overlaps")
                seen.add(key)
                got = tally.setdefault(row["arm"], [0, 0])
                got[0] += int(float(row["outcome"]) > 0.5)
                got[1] += 1
        for line in sorted(seen_headers):
            print(f"  played by: {line}")
        if len(seen_headers) > 1:
            raise SystemExit("these parts were played by different agents")
        print(f"  {len(parts)} part files, {len(seen)} games")
        results = {}
        for arm, (wins, n) in tally.items():
            rate = wins / n if n else float("nan")
            half = 1.96 * (rate * (1 - rate) / n) ** 0.5 if n else float("nan")
            results[arm] = rate
            print(f"  {arm:>20}  {n:>6}  {rate * 100:6.1f}%  +-{half * 100:.1f}")
        if claimed is None:
            raise SystemExit("no header line in any part file; re-run the shards")

        class _Claim:
            value = claimed

        report(results, _Claim(), args)
        return
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    team = next(t for t in standings.teams if t.place == args.place)

    encoder = Encoder(reg)
    net, _meta = load_model(args.model, encoder)
    device = torch.device(args.device)
    value = BatchedValue(net.to(device), encoder, device=device)

    loc = localiser(reg, args.lang)

    def namer(species_id: str) -> str:
        return loc.species(species_id) if loc else reg.species[species_id].name

    rng = np.random.default_rng(args.seed)
    if args.mirror:
        classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]
    else:
        classes = [
            SpreadClass(
                weight=1.0 / args.classes,
                sets=tuple(sample_standings_team(rng, reg, prior, team)),
                label=f"class {k + 1}",
            )
            for k in range(args.classes)
        ]
    analysis = solve_selection(reg, roster.sets, classes, value)
    strategy = np.asarray(analysis.equilibrium.row_strategy, dtype=np.float64)
    strategy = strategy / strategy.sum()
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    print(f"相手: {team.player}（{team.place} 位）", file=sys.stderr)
    print("  " + " / ".join(namer(s) for s in team.species), file=sys.stderr)
    print(
        f"  ソルバの均衡値 {analysis.value * 100:.1f}%、採用 "
        f"{len(analysis.equilibrium.row_support())} 通り",
        file=sys.stderr,
    )
    for selection, frequency, _loss in analysis.our_frequencies()[:4]:
        if frequency <= 1e-6:
            break
        print(
            f"    {frequency * 100:5.1f}%  {analysis.label(0, selection, namer)}",
            file=sys.stderr,
        )

    # The opponent's equilibrium is one strategy per spread class, because they know their
    # own investment and we do not. Drawing from it means drawing a class first.
    col = [np.asarray(y, dtype=np.float64) for y in analysis.equilibrium.col_strategies]
    col = [y / y.sum() if y.sum() > 0 else np.full(len(y), 1.0 / len(y)) for y in col]
    class_weights = np.asarray(analysis.equilibrium.weights, dtype=np.float64)
    class_weights = class_weights / class_weights.sum()

    print(f"\n  {'自陣 vs 相手':>20}  {'games':>6}  {'win':>7}  {'95%':>6}")
    results: dict[str, float] = {}
    forced: list[int] = []
    forced_arm = ""
    if args.force_lead:
        want = {to_id(x) for x in args.force_lead.split(",") if x.strip()}
        species = [s.species for s in roster.sets]
        forced = [
            i
            for i, sel in enumerate(selections)
            if {species[sel[0]], species[sel[1]]} == want
        ]
        if not forced:
            raise SystemExit(f"no selection leads with {sorted(want)}")
        # The best of them by the solver's own lights, so the comparison is against the
        # advice rather than against a random member of the restricted set.
        forced.sort(key=lambda i: -strategy[i])
        forced_arm = f"{args.force_lead} 先発 vs 均衡"
        print(
            f"  強制先発: {args.force_lead} に合う選出 {len(forced)} 通り、"
            f"うち均衡質量 {sum(strategy[i] for i in forced) * 100:.1f}%"
        )
    named: dict[str, int] = {}
    for spec in args.force_selection or []:
        want = [to_id(x) for x in spec.split(",") if x.strip()]
        if len(want) != len(roster.sets) - 2:
            raise SystemExit(f"--force-selection wants 4 species, got {len(want)}: {spec}")
        species = [s.species for s in roster.sets]
        index = next(
            (
                i
                for i, sel in enumerate(selections)
                # Lead order within the pair is not a choice the game exposes, so the
                # front two are compared as a set; the split between front and back is.
                if {species[sel[0]], species[sel[1]]} == set(want[:2])
                and {species[sel[2]], species[sel[3]]} == set(want[2:])
            ),
            None,
        )
        if index is None:
            raise SystemExit(f"no selection matches {spec}")
        label = "+".join(want[:2]) + " / " + "+".join(want[2:])
        named[label] = index
        print(f"  指定選出 {label}: 均衡質量 {strategy[index] * 100:.1f}%")
    arms = ["均衡 vs 一様", "一様 vs 一様", "均衡 vs 均衡"]
    if forced_arm:
        arms.append(forced_arm)
    arms.extend(named)
    if args.only_arm:
        # Calibration needs one arm: the claim is about equilibrium play on both sides, and
        # the other two answer a different question at two thirds of the cost.
        wanted = [a for a in arms if args.only_arm in a]
        if not wanted:
            raise SystemExit(f"--only-arm {args.only_arm!r} matches none of {arms}")
        arms = wanted
    rows: list[dict] = []
    for arm_no, arm in enumerate(arms, start=1):
        wins = finished = unfinished = 0
        # A shard that prints nothing until an arm ends is a shard whose remaining time
        # cannot be estimated, and this run's was guessed wrong four times. The line goes
        # to stderr and is flushed, because stdout is block-buffered into a log file.
        todo = len(range(args.shard, args.games, max(args.shards, 1)))
        started_at = time.monotonic()
        print(f"  [{arm_no}/{len(arms)}] {arm}: {todo} games", file=sys.stderr, flush=True)
        for done, game_index in enumerate(
            range(args.shard, args.games, max(args.shards, 1)), start=1
        ):
            if done > 1 and (done - 1) % 5 == 0:
                rate = (time.monotonic() - started_at) / (done - 1)
                left = rate * (todo - done + 1) / 60
                print(
                    f"    {done - 1}/{todo}  {rate:.0f}s a game, {left:.0f} min left",
                    file=sys.stderr,
                    flush=True,
                )
            # Seeded from the game's INDEX, not from a position in a shared stream: game g
            # is then the same game in every arm and in every shard. Lining arms up by
            # consuming one generator in step holds only while every arm draws the same
            # number of values, and this project has already lost a paired comparison that
            # way.
            game_rng = np.random.default_rng([args.seed + 1, game_index])
            foe_six = (
                list(roster.sets)
                if args.mirror
                else sample_standings_team(game_rng, reg, prior, team)
            )
            if arm == "均衡 vs 均衡" or arm == forced_arm or arm in named:
                which = int(game_rng.choice(len(col), p=class_weights))
                foe_pick = selections[int(game_rng.choice(len(col[which]), p=col[which]))]
            else:
                foe_pick = selections[int(game_rng.integers(len(selections)))]
            if arm in named:
                index = named[arm]
            elif arm == forced_arm:
                index = forced[0]
            elif arm.startswith("均衡"):
                index = int(game_rng.choice(len(strategy), p=strategy))
            else:
                index = int(game_rng.integers(len(selections)))
            own_pick = selections[index]
            record = play_game(
                reg,
                game_rng,
                [roster.sets[i] for i in own_pick],
                [foe_six[j] for j in foe_pick],
                f"place{args.place}",
                objective=OBJECTIVES["hp-share"],
                search_limit=args.limit,
                max_turns=args.max_turns,
                # The leaf that solved the selection has to be the leaf that plays it.
                # Without this the games were driven by the hp-share heuristic while the
                # claim came from the value function, so the -21.8 point miss G2 recorded
                # and everything measured with this tool since compared two agents.
                evaluate=value,
                rank_by_leaf=args.rank_by_leaf,
                sheets=(
                    (list(roster.sets), list(foe_six)) if args.hide_bench else None
                ),
            )
            if record.outcome is None:
                unfinished += 1
                continue
            finished += 1
            wins += int(record.outcome > 0.5)
            rows.append(
                {
                    "arm": arm,
                    "game": game_index,
                    "outcome": float(record.outcome),
                    "ownPick": list(own_pick),
                    "foePick": list(foe_pick),
                }
            )
        rate = wins / finished if finished else float("nan")
        half = 1.96 * (rate * (1 - rate) / finished) ** 0.5 if finished else float("nan")
        results[arm] = rate
        print(
            f"  {arm:>20}  {finished:>6}  {rate * 100:6.1f}%  +-{half * 100:.1f}"
            f"   (打ち切り {unfinished})"
        )
        # A mirror asserts 50.0%, exactly. Our six against our six with one leaf and one
        # width on both sides is antisymmetric, so any deviation is the seat -- the
        # engine, the resolver and the evaluator together -- and not a fact about the
        # agents. This tool never swaps the seats (our roster is always side 0), so the
        # mirror arm is the only seat diagnostic it has, and it was printed as an
        # ordinary win rate with an interval and nothing else. A speed-tie bug worth nine
        # points in a mirror is in this project's history.
        if args.mirror and finished and abs(rate - 0.5) > half:
            print(
                f"  ! a mirror is antisymmetric, so this arm asserts 50.0% and reads "
                f"{rate * 100:.1f}%.\n"
                "    The gap is the seat, not the agents: this tool keeps our roster at "
                "side 0\n    and never swaps, so nothing else here can separate them."
            )

    if args.out is not None:
        target = (
            args.out
            if args.shards <= 1
            else args.out.with_suffix(f".part{args.shard}{args.out.suffix}")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            # The claim belongs in the file: a merge that had to re-solve 8,100 cells per
            # class to print one number would cost more than the games it is summarising.
            handle.write(
                json.dumps(
                    {
                        "header": {
                            "claimed": float(analysis.value),
                            "place": args.place,
                            "games": args.games,
                            # What actually played. This tool wrote none of it and spent a
                            # year comparing the value function's claim against games the
                            # hp-share heuristic played, narrowed by damage after
                            # generation had moved to the leaf. generation_match records
                            # its agent and stayed correct; the tools that did not, drifted.
                            "leaf": str(args.model),
                            "limit": args.limit,
                            "ranking": "leaf" if args.rank_by_leaf else "damage",
                            "information": "hidden-bench" if args.hide_bench else "open",
                            # The rest of what decides the experiment. `claimed` is the
                            # LP value of THIS solve, and a solve depends on how many
                            # spread classes it drew and from which seed -- so two runs
                            # that differ in either are two experiments whose games must
                            # not be averaged against one claim. `mirror` replaces the
                            # opponent with our own six, which is a different question
                            # entirely and used to fingerprint identically to a field run.
                            "classes": args.classes,
                            "seed": args.seed,
                            "mirror": bool(args.mirror),
                        }
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n  → {target}（{len(rows)} 行）")
        if args.shards > 1:
            print("  残りのシャードと合わせて --merge で集計する")
            return

    report(results, analysis, args)


def report(results: dict, analysis, args) -> None:  # noqa: ANN001
    """The lines the measurement exists for, from whatever arms are in hand.

    `--only-arm` can leave the advice-value pair out, and a report that assumed all three
    arms would raise a KeyError on the run that skipped two thirds of the cost.
    """
    if {"均衡 vs 一様", "一様 vs 一様"} <= set(results):
        gap = (results["均衡 vs 一様"] - results["一様 vs 一様"]) * 100
        print(f"\n  助言の価値: {gap:+.1f} ポイント（相手は一様のまま、自陣の選出だけ変えた差）")
        if gap <= 0:
            print(
                "  → 均衡選出が一様に勝てていない。セルの推定誤差を最適化が拾っている"
                "（optimiser's curse）ので、この助言はまだ出せない。"
            )
        else:
            print("  → 均衡選出のほうが勝っている。助言として出せる。")

    if "均衡 vs 均衡" not in results:
        print("\n  （均衡 vs 均衡 の腕が無いので、主張値との突き合わせはできない）")
        return

    claimed = analysis.value * 100
    measured = results["均衡 vs 均衡"] * 100
    print(
        f"\n  主張した均衡値 {claimed:.1f}% に対し、両者が均衡を指した実測 {measured:.1f}%"
        f"（差 {measured - claimed:+.1f}）"
    )
    print(
        "  この2つが近ければ、印字している勝率は校正されている。"
        "自陣が大きく上回るなら、ソルバは相手の最善を過小評価している"
        "（弱い方策で学習した価値関数が誤る向き）。"
    )


if __name__ == "__main__":
    main()
