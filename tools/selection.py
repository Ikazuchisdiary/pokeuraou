"""Solves the 6->4 selection against one opponent's team sheet.

    uv run --group learn python tools/selection.py --place 1
    uv run --group learn python tools/selection.py --player "Antonio Sanchez" --classes 8

The opponent is named by placement or by player, from the cached tournament standings, so
the question being answered is "what do I bring against *this* team" rather than against an
average of the field. Their SP spreads are the hidden part and are drawn from usage
conditioned on the nature their sheet reveals -- one spread class per draw, solved as the
Bayesian game where they know their own investment and we do not.
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
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.selection import SpreadClass, render, solve_selection
from pokeuraou.standings import find_cached_standings, load_standings, sample_standings_team
from pokeuraou.teams import load_roster
from pokeuraou.value import BatchedValue, load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--model", type=Path, default=Path("data/models/value-worlds.pt"))
    ap.add_argument("--place", type=int, default=None, help="opponent by placement")
    ap.add_argument("--player", default=None, help="opponent by player name (substring)")
    ap.add_argument(
        "--classes",
        type=int,
        default=4,
        help="spread classes for the opponent's six. Each is a full 90x90 matrix, so the "
        "cost is linear in this and the covered probability mass is not: the classes are "
        "draws from the belief, not a partition of it.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--lang", default="ja")
    ap.add_argument(
        "--mirror",
        action="store_true",
        help="solve our six against itself, spreads included. The value must come out at "
        "exactly 0.500: with identical sides the matrix satisfies M[i][j] = 1 - M[j][i] by "
        "the value function's antisymmetry, so the game is symmetric and its value is "
        "forced. Any other answer is a wiring bug, not a modelling choice.",
    )
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

    if args.player:
        matches = [t for t in standings.teams if args.player.lower() in t.player.lower()]
        if not matches:
            raise SystemExit(f"no player matching {args.player!r}")
        team = matches[0]
    else:
        place = args.place if args.place is not None else 1
        matches = [t for t in standings.teams if t.place == place]
        if not matches:
            raise SystemExit(f"no entry placed {place}")
        team = matches[0]

    if not args.model.exists():
        raise SystemExit(
            f"no model at {args.model}; train one with tools/train_value.py. "
            "Selection cannot be solved without a value function -- the cells are win "
            "probabilities, and hp-share is not one."
        )
    encoder = Encoder(reg)
    net, meta = load_model(args.model, encoder)
    device = torch.device(args.device)
    value = BatchedValue(net.to(device), encoder, device=device)

    loc = localiser(reg, args.lang)

    def namer(species_id: str) -> str:
        return loc.species(species_id) if loc else reg.species[species_id].name

    print(
        f"相手: {team.player}（{team.place} 位, {team.country}, "
        f"{team.wins}-{team.losses}）",
        file=sys.stderr,
    )
    print("  " + " / ".join(namer(s) for s in team.species), file=sys.stderr)
    print(
        f"モデル: {args.model.name}  検証 AUC {meta.get('val_auc', float('nan')):.4f}, "
        f"log loss {meta.get('val_logloss', float('nan')):.4f} "
        f"({meta.get('games', '?')} ゲームで学習)",
        file=sys.stderr,
    )

    # Each class is one draw of the opponent's six spreads from the belief. Independent
    # draws rather than a partition: the belief has no joint structure across their six,
    # and pretending otherwise would invent a correlation the usage data does not contain.
    rng = np.random.default_rng(args.seed)
    if args.mirror:
        classes = [SpreadClass(weight=1.0, sets=tuple(roster.sets), label="mirror")]
        analysis = solve_selection(reg, roster.sets, classes, value)
        print(render(analysis, namer, top=args.top))
        error = abs(analysis.value - 0.5)
        print("")
        print(f"  ミラー検査: 均衡値 {analysis.value:.6f}, 誤差 {error:.2e}（0 が要件）")
        raise SystemExit(0 if error < 1e-6 else 1)

    classes = [
        SpreadClass(
            weight=1.0 / args.classes,
            sets=tuple(sample_standings_team(rng, reg, prior, team)),
            label=f"class {k + 1}",
        )
        for k in range(args.classes)
    ]

    analysis = solve_selection(reg, roster.sets, classes, value)
    print(render(analysis, namer, top=args.top))


if __name__ == "__main__":
    main()
