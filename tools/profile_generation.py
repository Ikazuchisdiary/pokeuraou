"""Where a generated game's time actually goes, with the learned leaf in place.

`tools/profile_resolve.py` profiles the resolver alone, against the hp-share leaf. That
answered the question it was built for and is now the wrong workload: generation runs a
learned value net, at candidate width 24 rather than 16, and the two changes move the
balance. The launcher still repeats a measurement taken at width 16 -- "the forward pass is
3% of the per-leaf cost" -- as its reason for staying on CPU, and a claim that old should
be re-measured before anything is optimised on the strength of it.

So this profiles exactly what `tools/selfplay.py` runs: whole games, same leaf, same
width, same one-thread torch. Reading the table, the question to ask of each row is not
"is it slow" but "would generation notice if it were free":

    uv run --group learn python tools/profile_generation.py \
        --value data/models/value-gen23.pt --limit 24 --games 2

Absolute times are only meaningful on an idle machine. Run it while fourteen workers are
generating and the shares stay honest while every microsecond figure inflates.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.selfplay import play_game
from pokeuraou.standings import find_cached_standings, load_standings, sample_standings_team
from pokeuraou.teams import all_selections, load_roster


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument("--value", type=Path, default=None, help="omit for the hp-share leaf")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    evaluate = None
    if args.value is not None:
        import torch

        torch.set_num_threads(args.torch_threads)
        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        encoder = Encoder(load_roster(args.roster).reg)
        net, _meta = load_model(args.value, encoder)
        device = torch.device(args.device)
        evaluate = BatchedValue(net.to(device), encoder, device=device)

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    def work(seed: int) -> tuple[int, int]:
        rng = np.random.default_rng(seed)
        turns = decisions = 0
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
                "profile",
                objective=OBJECTIVES["hp-share"],
                search_limit=args.limit,
                max_turns=40,
                evaluate=evaluate,
            )
            turns += record.turns
            decisions += len(record.decisions)
        return turns, decisions

    work(args.seed)  # warm up: first-call imports and caches are not the measurement
    started = time.perf_counter()
    turns, decisions = work(args.seed + 1000)
    elapsed = time.perf_counter() - started
    leaves = evaluate.evaluated if evaluate is not None else 0
    print(
        f"{args.games} games in {elapsed:.1f} s = {elapsed / args.games:.2f} s/game, "
        f"{turns / args.games:.1f} turns, {decisions / args.games:.1f} decisions, "
        f"{leaves / max(args.games, 1):,.0f} leaves/game"
    )

    profiler = cProfile.Profile()
    profiler.enable()
    work(args.seed + 2000)
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs()
    stats.sort_stats("tottime").print_stats(args.top)
    text = stream.getvalue()
    marker = "ncalls"
    index = text.find(marker)
    print(text[index:] if index >= 0 else text)


if __name__ == "__main__":
    main()
