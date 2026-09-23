"""IKA-139: one recorded decision where the fast path and the definition disagree, taken apart.

    PYTHONPATH=<tree>/src POKEURAOU_RUST_NODE=1 \\
        python scratchpad/ika139_one_decision.py data/ika73/w12 --game 16 --decision 0
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

from both_exact_calls import menu  # noqa: E402
from ika139_patched_vs_scratch import LinearLeaf  # noqa: E402
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


def load(pool: Path, game_index: int, decision_index: int, seed: int, book_path: Path):  # noqa: ANN201
    roster = load_roster("rizabanadohido")
    reg, _archetypes = load_archetypes("wcs2026-regmb")
    register_mega_stones(reg)
    standings = load_standings(find_cached_standings("2026", "worlds"), reg).pool("all")
    book = SelectionBook.read(book_path)
    game = recorded(pool, {game_index})[game_index]
    rng = np.random.default_rng([seed, game_index])
    entry = book.get(standings[int(rng.integers(len(standings)))])
    drawn = entry.draw(rng, epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE)
    foe_six = list(drawn.foe_six)
    sheets = (list(roster.sets), foe_six)
    prior = (
        BenchPrior.of(entry, 0, [s.species for s in roster.sets],
                      epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE),
        BenchPrior.of(entry, 1, [s.species for s in foe_six],
                      epsilon=DEFAULT_EPSILON, temperature=DEFAULT_TEMPERATURE),
    )
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="replay")
    shown = [frozenset(), frozenset()]
    for number, decision in enumerate(game["decisions"]):
        if decision["kind"] not in ("move", "replacement"):
            continue
        pos = Position.from_json(decision["position"])
        shown = [seen_slots(pos, i, shown[i]) for i in (0, 1)]
        if number != decision_index:
            continue
        spreads = {
            side: completions(
                reg, pos, side, sheets[side], seen=shown[side],
                weights=_bench_weights(prior, side, pos, shown[side], record),
            )
            for side in (0, 1)
        }
        ours = menu(reg, pos, 0, decision["ownActions"])
        theirs = menu(reg, pos, 1, decision["foeActions"])
        return reg, pos, ours, theirs, spreads
    raise SystemExit("no such decision")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("--game", type=int, required=True)
    ap.add_argument("--decision", type=int, required=True)
    ap.add_argument("--seed", type=int, default=6601)
    ap.add_argument(
        "--selection-book", type=Path,
        default=HERE.parent / "data" / "selection" / "rizabanadohido-value-gen11L.jsonl.gz",
    )
    args = ap.parse_args()
    reg, pos, ours, theirs, spreads = load(
        args.pool, args.game, args.decision, args.seed, args.selection_book
    )
    leaf = LinearLeaf(Encoder(reg))
    budget = Budget.matrix()
    port = rustnode.node_for(reg)
    print(f"turn {pos.turn}, menus {len(ours)}x{len(theirs)}, completions "
          f"{[len(spreads[s]) for s in (0, 1)]}, exact {[spreads[s][0].exact for s in (0, 1)]}")
    node = belief_payoffs(reg, pos, ours, theirs, leaf, budget=budget, spreads=spreads)
    hidden = {s: (items[0].slots if not items[0].exact else ()) for s, items in spreads.items()}
    dirty = reaches_bench(reg, ours, theirs, hidden)
    ref = port.fill_encoded(pos, ours, theirs, budget)
    for i, j, _r in ref.folded:
        dirty[i, j] = True
    print(f"refused by the port on the true position: {len(ref.refused)} cells, reasons "
          f"{sorted({why for _a, _b, why in ref.refused})}, of them dirty "
          f"{sum(bool(dirty[a, b]) for a, b, _w in ref.refused)}")
    for side, items in spreads.items():
        for k, item in enumerate(items):
            expected, _n = batched_payoff(reg, item.position, ours, theirs, leaf, budget=budget)
            off = np.argwhere(node.matrices[1 - side][k] != expected)
            if not len(off):
                continue
            print(f"side {side} completion {k} exact={item.exact}: {len(off)} cells, e.g. "
                  f"{[tuple(map(int, c)) for c in off[:6]]}")
            i, j = map(int, off[0])
            print(f"  cell ({i},{j}) {ours[i].to_choice()!r} vs {theirs[j].to_choice()!r}: "
                  f"dirty {bool(dirty[i, j])}, fast {node.matrices[1 - side][k][i, j]!r}, "
                  f"definition {expected[i, j]!r}")
            spans = {(a, b): (ix, w) for a, b, ix, w in ref.spans}
            own = port.fill_encoded(item.position, ours, theirs, budget)
            own_spans = {(a, b): (ix, w) for a, b, ix, w in own.spans}
            again = port.fill_encoded(pos, ours, theirs, budget)
            again_spans = {(a, b): (ix, w) for a, b, ix, w in again.spans}
            for label, table in (("true", spans), ("true again", again_spans),
                                 ("completion", own_spans)):
                ix, w = table.get((i, j), ([], []))
                print(f"  {label:10s} leaves {len(ix)} weights {list(w)[:8]}")
            print(f"  folded here? true {any((a, b) == (i, j) for a, b, _ in ref.folded)} "
                  f"completion {any((a, b) == (i, j) for a, b, _ in own.folded)}; refused "
                  f"true {[(a, b) for a, b, _ in ref.refused if (a, b) == (i, j)]} "
                  f"completion {[(a, b) for a, b, _ in own.refused if (a, b) == (i, j)]}")
            return


if __name__ == "__main__":
    main()
