"""Does solving a node without filling it reach the same answer?

`node_solver.solve_node` resolves the cells that prove an equilibrium and no others. What has to
be checked is not that it is fast but that it is *right*: the value must equal the full
matrix's to the last place, and every action outside its support must really be no better
than the ones in it -- which is a statement about cells it never asked for.

    POKEURAOU_RUST_NODE=1 uv run --group learn python tools/diff_solve_node.py \
        --nodes 8 --limit 48 --value data/models/value-gen2345.pt

The strategies are reported rather than asserted. Where a game has several equilibria of
equal value, this reaches one and the full LP reaches another, and the count of nodes where
they differ is the cost of the method -- the thing to weigh, not a bug to fix.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.node_solver import solve_node  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--nodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=404)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--value", default=None, help="a trained value function, or hp-share")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeded", type=int, default=1, help="top-k candidates to start from")
    args = ap.parse_args()

    roster = load_roster("rizabanadohido")
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    if args.value:
        import torch

        from pokeuraou.encode import Encoder
        from pokeuraou.value import BatchedValue, load_model

        encoder = Encoder(reg)
        net, _meta = load_model(Path(args.value), encoder)
        device = torch.device(args.device)
        leaf = BatchedValue(net.to(device), encoder, device=device)
        scorer, playing = leaf, leaf.objective("win")
    else:
        scorer, playing = OBJECTIVES["hp-share"].batch, OBJECTIVES["hp-share"]

    # Nodes a real game visits, collected with the bridge off so the hook sees them.
    import pokeuraou.resolve as resolve_mod
    import pokeuraou.search as search_mod

    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    real = resolve_mod.resolve_turn
    seen: list = []
    keys: set[str] = set()
    calls = [0]

    def recording(reg_, pos, actions, *, budget):  # noqa: ANN001
        # Every cell of a node resolves the same position, so distinct ones or nothing.
        calls[0] += 1
        if calls[0] % 89 == 0 and len(seen) < args.nodes:
            key = str(pos.to_json())
            if key not in keys:
                keys.add(key)
                seen.append(pos.copy())
        return real(reg_, pos, actions, budget=budget)

    resolve_mod.resolve_turn = recording
    search_mod.resolve_turn = recording
    rng = np.random.default_rng(args.seed)
    for _ in range(args.games):
        team = pool[int(rng.integers(len(pool)))]
        foe_six = sample_standings_team(rng, reg, prior, team)
        own = selections[int(rng.integers(len(selections)))]
        foe = selections[int(rng.integers(len(selections)))]
        play_game(
            reg, rng,
            [roster.sets[i] for i in own],
            [foe_six[j] for j in foe],
            "diff-oracle",
            objective=playing,
            search_limit=args.limit,
            max_turns=40,
        )
        if len(seen) >= args.nodes:
            break
    resolve_mod.resolve_turn = real
    search_mod.resolve_turn = real
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()

    build = rustnode.require_current_binary()
    print(f"binary {build['sha256']} built {build['built']}")
    print(
        f"{'node':>10}  {'resolved':>8}  {'share':>6}  {'rounds':>6}  {'value gap':>10}  "
        f"{'exploitable':>11}  {'same play':>9}  {'full s':>7}  {'oracle s':>8}"
    )

    total_cells = total_resolved = differing = 0
    worst_value = worst_exploit = 0.0
    full_seconds = oracle_seconds = 0.0
    for pos in seen:
        row = narrow(reg, pos, 0, limit=args.limit).actions
        col = narrow(reg, pos, 1, limit=args.limit).actions
        if len(row) < 2 or len(col) < 2:
            continue

        started = time.perf_counter()
        filled, _notes, _exact = batched_payoffs(
            reg, pos, row, col, [scorer], budget=Budget.matrix()
        )
        whole = solve(np.asarray(filled[0]))
        full_seconds += time.perf_counter() - started

        started = time.perf_counter()
        solved = solve_node(
            reg, pos, row, col, scorer, budget=Budget.matrix(), seeded=args.seeded
        )
        oracle_seconds += time.perf_counter() - started

        # The claim is about cells it never asked for: measured against the full matrix,
        # neither side may have a reply that beats the value it settled on.
        matrix = np.asarray(filled[0])
        against_y = matrix @ solved.equilibrium.col_strategy
        against_x = solved.equilibrium.row_strategy @ matrix
        exploitable = float(
            max(against_y.max() - solved.equilibrium.value,
                solved.equilibrium.value - against_x.min(), 0.0)
        )
        gap = abs(solved.equilibrium.value - whole.value)
        moved = max(
            float(np.abs(solved.equilibrium.row_strategy - whole.row_strategy).max()),
            float(np.abs(solved.equilibrium.col_strategy - whole.col_strategy).max()),
        )
        same = moved < 1e-9
        differing += int(not same)
        total_cells += solved.cells
        total_resolved += solved.resolved
        worst_value = max(worst_value, gap)
        worst_exploit = max(worst_exploit, exploitable)
        print(
            f"{len(row):>4}x{len(col):<5} {solved.resolved:>8}  "
            f"{solved.resolved / solved.cells:>5.0%}  {solved.rounds:>6}  {gap:>10.2e}  "
            f"{exploitable:>11.2e}  {'yes' if same else 'no':>9}  "
            f"{full_seconds:>7.2f}  {oracle_seconds:>8.2f}"
        )

    rustnode.reset()
    if not total_cells:
        raise SystemExit("no node was wide enough to compare")
    print(
        f"\n{total_resolved}/{total_cells} cells = {total_resolved / total_cells:.0%}"
    )
    print(f"  worst value difference   {worst_value:.3e}")
    print(f"  worst exploitability     {worst_exploit:.3e}  (0 means it really is one)")
    print(f"  nodes whose play differs {differing}")
    print(f"  full {full_seconds:.2f} s   oracle {oracle_seconds:.2f} s")
    if oracle_seconds > 0:
        print(f"  solving a node is {full_seconds / oracle_seconds:.1f}x faster")


if __name__ == "__main__":
    main()
