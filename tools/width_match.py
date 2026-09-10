"""Plays a wide candidate set against a narrow one, with everything else identical.

`tools/narrow_effect.py` found that the production width (16) cannot place 22.9% of the
equilibrium's weight on average -- 70% in the worst position it saw -- while the game's
*value* barely moves (0.003). Two readings fit that, and they point at different work:

  the excluded actions are near-substitutes, so narrowing costs the reported mixture its
  accuracy but costs a player almost nothing; or
  the value shift understates the cost, the way validation AUC understated the value
  function's -- a net that loses on AUC (0.8926 against 0.9132) and on log loss beat the
  hp-share leaf by +5.8 points in play.

Having just been wrong about that once, the tie is broken by games rather than by another
scalar. Both sides get the same teams, the same selections, the same seed and the same leaf;
only the number of candidate actions differs. `play_game` already takes `search_limit` as a
pair for exactly this diagnostic.

Seats are swapped, because both seats put our six against the field and the roster's edge
would otherwise be credited to whichever width sat in the better seat. The R/a decomposition
below separates them and R doubles as the setup's self-check: it has been measured
independently at 60.9-62.1%.

    uv run --group learn python tools/width_match.py --wide 48 --narrow 16 --games 40
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
from pokeuraou.selfplay import play_game
from pokeuraou.standings import find_cached_standings, load_standings, sample_standings_team
from pokeuraou.teams import all_selections, load_roster
from pokeuraou.value import BatchedValue, load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--value", type=Path, default=Path("data/models/value-gen2.pt"))
    ap.add_argument("--wide", type=int, default=48)
    ap.add_argument("--narrow", type=int, default=16)
    ap.add_argument("--games", type=int, default=40, help="games per seat")
    ap.add_argument("--objective", default="hp-share", choices=sorted(OBJECTIVES))
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--out", type=Path, default=None, help="append one JSON line per seat")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch-threads", type=int, default=1)
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
    net, _meta = load_model(args.value, encoder)
    device = torch.device(args.device)
    value = BatchedValue(net.to(device), encoder, device=device)
    objective = OBJECTIVES[args.objective]

    print(
        f"same leaf on both sides ({args.value.name}); only the candidate width differs: "
        f"{args.wide} against {args.narrow}",
        file=sys.stderr,
    )

    for seat, limits in (
        ("wide = side 0", (args.wide, args.narrow)),
        ("wide = side 1", (args.narrow, args.wide)),
    ):
        rng = np.random.default_rng(args.seed)
        seat_wins = seat_played = unfinished = 0
        started = time.perf_counter()
        for _ in range(args.games):
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
                evaluate=(value, value),
            )
            if record.outcome is None:
                unfinished += 1
                continue
            seat_played += 1
            # `outcome` is side 0's result, so flip it when the wide side sits at 1.
            wide_won = record.outcome > 0.5 if limits[0] == args.wide else record.outcome < 0.5
            seat_wins += int(wide_won)
        elapsed = time.perf_counter() - started
        rate = seat_wins / seat_played if seat_played else float("nan")
        print(
            f"  {seat:>16}  {seat_played:>4} games  wide {rate * 100:>5.1f}%  "
            f"{elapsed / max(args.games, 1):>6.2f} s/game  (unfinished {unfinished})",
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
                            "wide": args.wide,
                            "narrow": args.narrow,
                            "model": args.value.name,
                            "played": seat_played,
                            "wide_wins": seat_wins,
                            "unfinished": unfinished,
                            "seconds": elapsed,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
