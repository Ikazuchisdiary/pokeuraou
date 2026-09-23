"""What does a move decision cost in fills and forward passes, before and after IKA-104?

Asked on 2026-09-23. IKA-104 makes `_per_completion` resolve a position both sides' lists
hold once instead of once per side, which is every move decision with nothing hidden on
either side. The claim is "2 -> 1 there, unchanged everywhere else, and not one bit of any
answer moves"; this counts it on recorded decisions rather than on a turn-1 position.

    git show <the commit before IKA-104>:src/pokeuraou/beliefnode.py > before.py
    POKEURAOU_RUST_NODE=1 python scratchpad/both_exact_calls.py data/ika73/w12 before.py \
        --games 40

Each recorded move decision goes through `belief_solve` the way `selfplay.play_game` calls
it: the recorded menus, both sides' sixes and bench priors rebuilt from the game's own seed
(`[seed, gameIndex]`, drawn in the order `selfplay.generate` draws them, and checked
against the recorded sixes), and `shown` carried from decision to decision as the game
carried it -- updated at move and replacement nodes, not at a self-switch. It is solved
twice, once with the tree's `beliefnode` and once with the file given, with one leaf object
for both sides as in generation.

The leaf is a deterministic stand-in scored from the encoding (the arithmetic of
`tests/test_beliefnode.py`'s `_Leaf`), because what is counted is structural: how many
times the node is filled, scored and folded. It is not the shipped net, so the strategies
here are not the recorded ones; old and new are compared with each other, to the bit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from replay_match import recorded  # noqa: E402

from pokeuraou import beliefnode, resolve, rustnode  # noqa: E402
from pokeuraou.actions import side_actions  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError  # noqa: E402
from pokeuraou.hidden import completions, seen_slots  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.resolve import Budget  # noqa: E402
from pokeuraou.search import belief_solve  # noqa: E402
from pokeuraou.selection_book import (  # noqa: E402
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    BenchPrior,
    SelectionBook,
)
from pokeuraou.selfplay import GameRecord, _bench_weights  # noqa: E402
from pokeuraou.standings import find_cached_standings, load_standings  # noqa: E402
from pokeuraou.teams import load_archetypes, load_roster  # noqa: E402


class Leaf:
    """Scored from the encoding, so the port's encoded crossing is the one taken."""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.passes = 0
        self.rows = 0

    def from_encoded(self, encoded) -> np.ndarray:  # noqa: ANN001
        self.passes += 1
        self.rows += len(encoded)
        parts = (
            encoded.species.reshape(len(encoded), -1).sum(axis=1) * 0.7,
            encoded.ability.reshape(len(encoded), -1).sum(axis=1) * 0.3,
            encoded.item.reshape(len(encoded), -1).sum(axis=1) * 0.11,
            encoded.moves.reshape(len(encoded), -1).sum(axis=1) * 0.013,
            encoded.mon.reshape(len(encoded), -1).sum(axis=1).astype(np.float64) * 0.017,
            encoded.mask.reshape(len(encoded), -1).sum(axis=1).astype(np.float64) * 0.5,
        )
        return 1.0 / (1.0 + np.exp(-(sum(parts) % 7.0) + 3.0))

    def __call__(self, positions: list) -> np.ndarray:
        return self.from_encoded(self.encoder.encode_positions(positions))


COUNTED = ("batched_payoff", "batched_payoffs (dirty cells)", "fill_encoded", "resolve_turn (Python)")


def load_before(path: Path):  # noqa: ANN201
    spec = importlib.util.spec_from_file_location("pokeuraou._beliefnode_before", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def menu(reg, pos, side, names):  # noqa: ANN001, ANN201
    by_choice = {a.to_choice(): a for a in side_actions(reg, pos, side)}
    return [by_choice[name] for name in names]


def same_answers(a: dict, b: dict) -> bool:
    for side in (0, 1):
        x, y = a[side], b[side]
        if not np.array_equal(x.strategy, y.strategy) or x.value != y.value:
            return False
        if len(x.replies) != len(y.replies):
            return False
        if any(not np.array_equal(p, q) for p, q in zip(x.replies, y.replies, strict=True)):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("before", type=Path, help="beliefnode.py as it was before the change")
    ap.add_argument("--games", type=int, default=40, help="game indices 0..N-1")
    ap.add_argument("--seed", type=int, default=6601)
    ap.add_argument("--roster", default="rizabanadohido")
    ap.add_argument(
        "--selection-book", type=Path,
        default=HERE.parent / "data" / "selection" / "rizabanadohido-value-gen11L.jsonl.gz",
    )
    ap.add_argument("--out", type=Path, default=None, help="per-decision rows as JSONL")
    args = ap.parse_args()
    if not rustnode.available():
        raise SystemExit("the port is not available; the counts would be of the Python path")

    roster = load_roster(args.roster)
    reg, _archetypes = load_archetypes("wcs2026-regmb")
    register_mega_stones(reg)
    pool = load_standings(find_cached_standings("2026", "worlds"), reg).pool("all")
    book = SelectionBook.read(args.selection_book)
    before = load_before(args.before)
    leaf = Leaf(Encoder(reg))
    budget = Budget.matrix()

    counts: dict[str, int] = defaultdict(int)

    def counting(name, function):  # noqa: ANN001, ANN202
        def inner(*a, **k):  # noqa: ANN002, ANN003, ANN202
            counts[name] += 1
            return function(*a, **k)

        return inner

    for module in (beliefnode, before):
        module.batched_payoff = counting("batched_payoff", module.batched_payoff)
        module.batched_payoffs = counting("batched_payoffs (dirty cells)", module.batched_payoffs)
    rustnode.RustNode.fill_encoded = counting("fill_encoded", rustnode.RustNode.fill_encoded)
    resolve.resolve_turn = counting("resolve_turn (Python)", resolve.resolve_turn)
    tree_payoffs = beliefnode.belief_payoffs

    def solve(which: str, pos, ours, theirs, spreads):  # noqa: ANN001, ANN202
        beliefnode.belief_payoffs = before.belief_payoffs if which == "before" else tree_payoffs
        counts.clear()
        leaf.passes = leaf.rows = 0
        try:
            answers = belief_solve(
                reg, pos, ours, theirs, spreads, {0: leaf, 1: leaf}, budget=budget
            )
        except EquilibriumError:
            answers = None
        finally:
            beliefnode.belief_payoffs = tree_payoffs
        return answers, {
            **{name: counts[name] for name in COUNTED},
            "forward passes": leaf.passes,
            "rows scored": leaf.rows,
        }

    games = recorded(args.pool, set(range(args.games)))
    rows: list[dict] = []
    for index in sorted(games):
        game = games[index]
        rng = np.random.default_rng([args.seed, index])
        team = pool[int(rng.integers(len(pool)))]
        entry = book.get(team)
        if entry is None:
            raise SystemExit(f"game {index}: the book has no entry for its opponent")
        drawn = entry.draw(rng, epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE)
        foe_six = list(drawn.foe_six)
        if [s.species for s in foe_six] != game["foeSix"] or list(drawn.our_pick) != game["ownPick"]:
            raise SystemExit(f"game {index}: the rebuilt draw is not the recorded one")
        sheets = (list(roster.sets), foe_six)
        prior = (
            BenchPrior.of(entry, 0, [s.species for s in roster.sets],
                          epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE),
            BenchPrior.of(entry, 1, [s.species for s in foe_six],
                          epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE),
        )
        record = GameRecord(own_team=[], foe_team=[], foe_archetype="replay")
        shown = [frozenset(), frozenset()]
        for step, decision in enumerate(game["decisions"]):
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
            exact = [spreads[side][0].exact for side in (0, 1)]
            kind = (
                "both exact" if all(exact)
                else "both hidden" if not any(exact)
                else "one side hidden"
            )
            ours = menu(reg, pos, 0, decision["ownActions"])
            theirs = menu(reg, pos, 1, decision["foeActions"])
            old, old_counts = solve("before", pos, ours, theirs, spreads)
            new, new_counts = solve("tree", pos, ours, theirs, spreads)
            identical = (old is None and new is None) or (
                old is not None and new is not None and same_answers(old, new)
            )
            rows.append({
                "game": index, "decision": step, "kind": kind,
                "cells": len(ours) * len(theirs),
                "completions": [len(spreads[0]), len(spreads[1])],
                "before": old_counts, "after": new_counts, "identical": identical,
            })

    if args.out is not None:
        args.out.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n"
        )
    print(f"{len(games)} games (indices 0..{args.games - 1}), {len(rows)} move decisions")
    for kind in ("both exact", "one side hidden", "both hidden"):
        group = [row for row in rows if row["kind"] == kind]
        if not group:
            continue
        same = sum(row["identical"] for row in group)
        print(
            f"\n  {kind}: {len(group)} decisions ({100.0 * len(group) / len(rows):.1f}%), "
            f"answers identical before/after {same}/{len(group)}"
        )
        for name in (*COUNTED, "forward passes", "rows scored"):
            old = sum(row["before"][name] for row in group)
            new = sum(row["after"][name] for row in group)
            print(
                f"    {name:<30} {old / len(group):>10.2f} -> {new / len(group):>10.2f}"
                f"   per decision  (total {old} -> {new})"
            )
    print(f"\nanswers identical before/after: {sum(r['identical'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
