"""How fast can this machine generate games, and with how many processes?

Generation is the whole bottleneck -- 11 core-hours against 11 seconds of training -- so the
process count is worth measuring rather than guessing. It was guessed at 10 on a machine
with 8 physical cores and 16 threads, which is already past the point where more processes
help much.

Generation 2 adds a second constraint the first did not have. Each worker holds a CUDA
context of roughly 1.1 GB, so a 12 GB card caps the run at eight or ten workers regardless
of the CPU. But the forward pass is only 3% of the per-leaf cost (to_json 0.228 ms, encode
0.117 ms, forward 0.012 ms), so CPU inference might cost almost nothing and remove the cap
entirely -- which is what this measures.

`torch.set_num_threads(1)` matters more than it looks: torch defaults to a thread pool per
process, and sixteen processes each spawning eight threads on eight cores is how a parallel
run ends up slower than a serial one.

    uv run --group learn python tools/bench_generation.py --device cpu --games 6
    uv run --group learn python tools/bench_generation.py --device cuda --games 6
"""

from __future__ import annotations

import argparse
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
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--depth", type=int, default=1, help="1 = one ply, 2 = refined cells")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--solve-sparsely",
        action="store_true",
        help="solve each node by resolving the fifth of it that proves the equilibrium "
        "rather than filling the matrix. Same value, not necessarily the same vertex.",
    )
    ap.add_argument(
        "--rank-leaf",
        action="store_true",
        help="rank the menu by the leaf, as a real generation run does. It is 75% on top "
        "at width 24 and almost nothing at width 48, so a benchmark that leaves it out is "
        "not measuring the run it is meant to stand for.",
    )
    ap.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="1 is right for a parallel run: N processes each spawning a thread pool on the "
        "same cores is slower than one process. Raise it only for a single-process run.",
    )
    args = ap.parse_args()

    evaluate = None
    if args.value is not None:
        import torch

        torch.set_num_threads(args.torch_threads)
        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        roster_reg = load_roster(args.roster).reg
        encoder = Encoder(roster_reg)
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
    rng = np.random.default_rng(args.seed)

    if evaluate is not None:
        # One forward pass before the clock starts. The first one in a process pays for
        # the CUDA context and for whatever kernels the shapes need built, and that lands
        # inside the timed loop otherwise: the same command measured 9.17 s/game and
        # 1.35 s/game on the same model, which is one 8-second warm-up spread over four
        # games or over forty.
        from pokeuraou.selfplay import position_from_sets

        picked = [roster.sets[i] for i in selections[0]]
        evaluate([position_from_sets(reg, picked, picked)] * 64)

    turns = decisions = finished = 0
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
            "bench",
            objective=OBJECTIVES["hp-share"],
            search_limit=args.limit,
            rank_by_leaf=args.rank_leaf,
            solve_sparsely=args.solve_sparsely,
            depth=args.depth,
            max_turns=40,
            evaluate=evaluate,
        )
        turns += record.turns
        decisions += len(record.decisions)
        finished += int(record.outcome is not None)
    elapsed = time.perf_counter() - started

    leaf = args.value.stem if args.value else "hp-share"
    print(
        f"{leaf} on {args.device} width {args.limit} depth {args.depth} "
        f"(rank-leaf {args.rank_leaf}, sparse {args.solve_sparsely}, "
        f"torch threads {args.torch_threads}): "
        f"{elapsed / args.games:.3f} s/game, {args.games / elapsed * 60:.1f} games/min "
        f"per process"
    )
    print(
        f"  {decisions / max(args.games, 1):.1f} decisions/game, "
        f"{turns / max(args.games, 1):.1f} turns/game, "
        f"finished {finished}/{args.games}"
    )
    if evaluate is not None:
        print(f"  positions evaluated: {evaluate.evaluated:,}")


if __name__ == "__main__":
    main()
