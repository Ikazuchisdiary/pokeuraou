"""Does the hidden-bench fast path fill the cells the port refuses?

Noticed on 2026-09-23 while counting calls for IKA-104. `belief_payoffs` asks the port for
the whole node once (`fill_encoded(position, ..., cells=None)`) and never reads the
answer's `refused` list: a refused cell has no span and no fold, so unless `reaches_bench`
or a fold marked it dirty its payoff stays at the 0.0 the matrix was made with. Every other
caller of the port (`resolve._rust_encoded_payoffs`) fills refused cells in Python.

`tests/test_beliefnode.py` compares the fast path with a matrix per completion on one
turn-1 position, which has no refused cell, so it cannot see this. Here the same comparison
runs on recorded mid-game decisions -- menus, sixes, bench priors and `shown` rebuilt as
`scratchpad/both_exact_calls.py` rebuilds them -- with the same stand-in leaf.

    POKEURAOU_RUST_NODE=1 python scratchpad/fastpath_refused.py data/ika73/w12 --games 10

What it found (9/23 11:52, games 0-9): no refused cell at all in 53 fast-path decisions, so
not the defect it was written for -- but 44 of 444 completion matrices were not equal to
the bit to the one resolved from scratch, 1,429 cells, none of them refused. With the leaf
made smooth (no `% 7.0`) 1,429 cells differ again, in 44 matrices (worst 3e-06 against
0.004) -- the count does not move with the leaf, which points at the leaves' inputs rather
than the stand-in's arithmetic (that the cells are the same ones was not checked). Not
IKA-104's change: the path is the same before and after it. Left for its own issue.

IKA-139 (9/23): on master c1b7e4c this prints 0 of 444. The 44 / 1,429 were IKA-121 and
IKA-119: the same run on 250a1db (before both) gives 44 / 1,429 again, on adb1e7b (IKA-121
only) 29 / 722, and after IKA-119 none. This leaf does not read `side`, so what it saw was
`can_mega` alone; `scratchpad/ika139_patched_vs_scratch.py` compares the inputs.
And the defect this was written for is real, only not in games 0-9: games 10-209 hold 19
decisions with a shared refused cell (Feint), 3,886 cells left at 0.0. Fixed in IKA-139.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from both_exact_calls import Leaf, menu  # noqa: E402
from replay_match import recorded  # noqa: E402

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.beliefnode import belief_payoffs, reaches_bench  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.hidden import completions, seen_slots  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoff  # noqa: E402
from pokeuraou.selection_book import (  # noqa: E402
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    BenchPrior,
    SelectionBook,
)
from pokeuraou.selfplay import GameRecord, _bench_weights  # noqa: E402
from pokeuraou.standings import find_cached_standings, load_standings  # noqa: E402
from pokeuraou.teams import load_archetypes, load_roster  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=6601)
    ap.add_argument(
        "--selection-book", type=Path,
        default=HERE.parent / "data" / "selection" / "rizabanadohido-value-gen11L.jsonl.gz",
    )
    args = ap.parse_args()
    if not rustnode.available():
        raise SystemExit("the port is not available; there is no fast path to check")

    roster = load_roster("rizabanadohido")
    reg, _archetypes = load_archetypes("wcs2026-regmb")
    register_mega_stones(reg)
    pool = load_standings(find_cached_standings("2026", "worlds"), reg).pool("all")
    book = SelectionBook.read(args.selection_book)
    leaf = Leaf(Encoder(reg))
    budget = Budget.matrix()
    port = rustnode.node_for(reg)

    decisions = with_refused = matrices = wrong_matrices = 0
    wrong_cells = refused_wrong = refused_zero = other_wrong = refused_total = 0
    refused_dirty = 0
    worst = 0.0
    for index, game in sorted(recorded(args.pool, set(range(args.games))).items()):
        rng = np.random.default_rng([args.seed, index])
        entry = book.get(pool[int(rng.integers(len(pool)))])
        drawn = entry.draw(rng, epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE)
        foe_six = list(drawn.foe_six)
        assert [s.species for s in foe_six] == game["foeSix"]
        sheets = (list(roster.sets), foe_six)
        prior = (
            BenchPrior.of(entry, 0, [s.species for s in roster.sets],
                          epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE),
            BenchPrior.of(entry, 1, [s.species for s in foe_six],
                          epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE),
        )
        record = GameRecord(own_team=[], foe_team=[], foe_archetype="replay")
        shown = [frozenset(), frozenset()]
        for decision in game["decisions"]:
            if decision["kind"] not in ("move", "replacement"):
                continue
            pos = Position.from_json(decision["position"])
            shown = [seen_slots(pos, i, shown[i]) for i in (0, 1)]
            if decision["kind"] != "move":
                continue
            spreads = {
                side: completions(
                    reg, pos, side, sheets[side], seen=shown[side],
                    weights=_bench_weights(prior, side, pos, shown[side], record),
                )
                for side in (0, 1)
            }
            if all(spreads[side][0].exact for side in (0, 1)):
                continue  # `_per_completion`, not the fast path
            decisions += 1
            ours = menu(reg, pos, 0, decision["ownActions"])
            theirs = menu(reg, pos, 1, decision["foeActions"])
            refused = {(i, j) for i, j, _why in port.fill_encoded(pos, ours, theirs, budget).refused}
            hidden = {
                side: (items[0].slots if items and not items[0].exact else ())
                for side, items in spreads.items()
            }
            dirty = reaches_bench(reg, ours, theirs, hidden)
            refused_total += len(refused)
            refused_dirty += sum(1 for i, j in refused if dirty[i, j])
            with_refused += bool(refused)
            node = belief_payoffs(
                reg, pos, ours, theirs, leaf, budget=budget, spreads=spreads
            )
            for side, items in spreads.items():
                for k, item in enumerate(items):
                    expected, _notes = batched_payoff(
                        reg, item.position, ours, theirs, leaf, budget=budget
                    )
                    got = node.matrices[1 - side][k]
                    matrices += 1
                    off = np.argwhere(got != expected)
                    if len(off):
                        wrong_matrices += 1
                    for i, j in off:
                        wrong_cells += 1
                        worst = max(worst, abs(float(got[i, j] - expected[i, j])))
                        if (int(i), int(j)) in refused:
                            refused_wrong += 1
                            refused_zero += float(got[i, j]) == 0.0
                        else:
                            other_wrong += 1

    print(f"{decisions} move decisions on the fast path (games 0..{args.games - 1})")
    print(f"  with a refused cell in the node: {with_refused}")
    print(f"  refused cells {refused_total}, of which dirty (re-resolved) {refused_dirty}")
    print(f"  completion matrices compared: {matrices}, not equal to the bit: {wrong_matrices}")
    print(f"  cells that differ: {wrong_cells}  (worst |diff| {worst:.3g})")
    print(f"    refused by the port: {refused_wrong}, of which exactly 0.0: {refused_zero}")
    print(f"    any other cell: {other_wrong}")


if __name__ == "__main__":
    main()
