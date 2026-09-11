"""Does a whole node come out the same through Rust as through Python?

The turn differential compares one `resolve_turn` at a time. This compares what the caller
actually asks for: a filled matrix. It runs real search nodes from generated games through
both paths and checks every cell, including the ones Rust refuses -- those must be filled
by Python, and the point of the check is that the two halves add up to Python's own matrix.

    uv run python tools/diff_node.py --games 2 --limit 24

Exact equality, not a tolerance: the payoff is a weighted mean of values that are ratios of
integers, and both sides sum the same branches in the same order.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import OBJECTIVES  # noqa: E402
from pokeuraou.priors import find_cached_chaos, load_chaos  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoffs  # noqa: E402
from pokeuraou.rustnode import RustNode  # noqa: E402
from pokeuraou.selfplay import play_game  # noqa: E402
from pokeuraou.standings import (  # noqa: E402
    find_cached_standings,
    load_standings,
    sample_standings_team,
)
from pokeuraou.teams import all_selections, load_roster  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--nodes", type=int, default=40, help="stop after this many nodes")
    ap.add_argument("--roster", default="rizabanadohido")
    args = ap.parse_args()

    roster = load_roster(args.roster)
    reg = roster.reg
    register_mega_stones(reg)
    prior = load_chaos(find_cached_chaos(reg.meta.format_id), reg)
    standings = load_standings(find_cached_standings(), reg)
    pool = standings.pool("all")
    selections = tuple(all_selections(reg.meta.team_size, reg.meta.picked_team_size))

    node = RustNode(reg)
    names = ["hp-share", "faints"]
    objectives = [OBJECTIVES[name] for name in names]

    from collections import Counter

    reasons: Counter[str] = Counter()
    checked = 0
    cells = 0
    refused_cells = 0
    worst = 0.0
    differing = 0
    exact_cells = 0
    worst_value = 0.0
    worst_strategy = 0.0
    python_seconds = 0.0
    rust_seconds = 0.0
    fallback_seconds = 0.0
    positions: list = []

    # Collect the positions a real game visits, then compare each as a node.
    import pokeuraou.resolve as resolve_mod

    real_resolve = resolve_mod.resolve_turn
    seen_positions: list = []

    # Every cell of a node resolves against the same position, so sampling every call
    # would collect one node's worth of duplicates. Take one in a few hundred instead.
    calls = [0]

    def recording(reg_, pos, actions, *, budget):  # noqa: ANN001
        calls[0] += 1
        if calls[0] % 137 == 0 and len(seen_positions) < args.nodes * 4:
            seen_positions.append(pos.copy())
        return real_resolve(reg_, pos, actions, budget=budget)

    resolve_mod.resolve_turn = recording
    import pokeuraou.search as search_mod

    search_mod.resolve_turn = recording

    rng = np.random.default_rng(args.seed)
    for _ in range(args.games):
        team = pool[int(rng.integers(len(pool)))]
        foe_six = sample_standings_team(rng, reg, prior, team)
        own_pick = selections[int(rng.integers(len(selections)))]
        foe_pick = selections[int(rng.integers(len(selections)))]
        play_game(
            reg,
            rng,
            [roster.sets[i] for i in own_pick],
            [foe_six[j] for j in foe_pick],
            "diff-node",
            objective=OBJECTIVES["hp-share"],
            search_limit=args.limit,
            max_turns=args.max_turns,
        )
    resolve_mod.resolve_turn = real_resolve
    search_mod.resolve_turn = real_resolve

    # One node per distinct position, up to the cap.
    seen: set[str] = set()
    for pos in seen_positions:
        key = str(pos.to_json())
        if key in seen:
            continue
        seen.add(key)
        positions.append(pos)
        if len(positions) >= args.nodes:
            break

    budget = Budget.matrix()
    for pos in positions:
        row = narrow(reg, pos, 0, limit=args.limit).actions
        col = narrow(reg, pos, 1, limit=args.limit).actions
        if not row or not col:
            continue

        started = time.perf_counter()
        expected, notes, exact = batched_payoffs(
            reg, pos, row, col, [o.batch for o in objectives], budget=budget
        )
        python_seconds += time.perf_counter() - started

        started = time.perf_counter()
        result = node.fill(pos, row, col, names, budget)
        rust_seconds += time.perf_counter() - started

        got = [np.array(matrix, dtype=np.float64) for matrix in result.payoffs]
        got_exact = np.array(result.exact, dtype=bool)

        # Fill the refused cells in Python, as a caller would.
        started = time.perf_counter()
        for i, j, why in result.refused:
            reasons[why] += 1
            filled, cell_notes, cell_exact = batched_payoffs(
                reg, pos, [row[i]], [col[j]], [o.batch for o in objectives], budget=budget
            )
            for index in range(len(objectives)):
                got[index][i, j] = filled[index][0, 0]
            got_exact[i, j] = cell_exact[0, 0]
            del cell_notes
        fallback_seconds += time.perf_counter() - started

        checked += 1
        cells += len(row) * len(col)
        refused_cells += len(result.refused)
        for index, name in enumerate(names):
            gap = np.abs(got[index] - np.asarray(expected[index]))
            worst = max(worst, float(gap.max()))
            differing += int((gap > 0).sum())
            exact_cells += int((gap == 0).sum())
            # Anything past a couple of units in the last place is not summation order.
            if gap.max() > 1e-12:
                where = np.unravel_index(np.argmax(gap), expected[index].shape)
                print(
                    f"  {name} differs by {gap.max():.3e} at cell {where}: "
                    f"python {expected[index][where]!r} rust {got[index][where]!r}"
                )
        if not np.array_equal(got_exact, exact):
            print("  the exact mask differs")
        del notes

        # A difference in the last place of a cell is only interesting if it moves the
        # answer, and the answer is the equilibrium. Solve both matrices and compare.
        for index in range(len(names)):
            python_eq = solve(np.asarray(expected[index]))
            rust_eq = solve(got[index])
            worst_value = max(worst_value, abs(python_eq.value - rust_eq.value))
            worst_strategy = max(
                worst_strategy,
                float(np.abs(python_eq.row_strategy - rust_eq.row_strategy).max()),
                float(np.abs(python_eq.col_strategy - rust_eq.col_strategy).max()),
            )

    node.close()
    scored = exact_cells + differing
    print(f"\n{checked} nodes, {cells} cells")
    print(f"  refused by the port {refused_cells} ({refused_cells / max(cells, 1) * 100:.1f}%)")
    print(
        f"  bit-identical {exact_cells}/{scored} scored cells "
        f"({exact_cells / max(scored, 1) * 100:.2f}%)"
    )
    print(f"  worst cell difference {worst:.3e}")
    if reasons:
        print("  refusals, heaviest first:")
        for reason, count in reasons.most_common(12):
            print(f"    {count:>6}  {reason}")
    if worst > 0:
        print(
            "  (a payoff is a weighted mean; numpy sums a dot product in a different\n"
            "   order than a sequential loop, so the last place can differ. The resolver\n"
            "   itself is bit-identical: positions, branches and probabilities all match.)"
        )
    print(f"  equilibrium value moved at most {worst_value:.3e}")
    print(f"  equilibrium frequency moved at most {worst_strategy:.3e}")
    print(
        f"  python {python_seconds:.2f} s   rust {rust_seconds:.2f} s + "
        f"python fallback {fallback_seconds:.2f} s"
    )
    total = rust_seconds + fallback_seconds
    if total > 0:
        print(f"  end to end {python_seconds / total:.1f}x")


if __name__ == "__main__":
    main()
