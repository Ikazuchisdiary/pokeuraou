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
import sys
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
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
    for arm in arms:
        # Same seed per arm, so the opponent's spreads, selections and every roll inside
        # the games line up and only our selection rule differs.
        game_rng = np.random.default_rng(args.seed + 1)
        wins = finished = unfinished = 0
        for _ in range(args.games):
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
            )
            if record.outcome is None:
                unfinished += 1
                continue
            finished += 1
            wins += int(record.outcome > 0.5)
        rate = wins / finished if finished else float("nan")
        half = 1.96 * (rate * (1 - rate) / finished) ** 0.5 if finished else float("nan")
        results[arm] = rate
        print(
            f"  {arm:>20}  {finished:>6}  {rate * 100:6.1f}%  +-{half * 100:.1f}"
            f"   (打ち切り {unfinished})"
        )

    gap = (results["均衡 vs 一様"] - results["一様 vs 一様"]) * 100
    print(f"\n  助言の価値: {gap:+.1f} ポイント（相手は一様のまま、自陣の選出だけ変えた差）")
    if gap <= 0:
        print(
            "  → 均衡選出が一様に勝てていない。セルの推定誤差を最適化が拾っている"
            "（optimiser's curse）ので、この助言はまだ出せない。"
        )
    else:
        print("  → 均衡選出のほうが勝っている。助言として出せる。")

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
