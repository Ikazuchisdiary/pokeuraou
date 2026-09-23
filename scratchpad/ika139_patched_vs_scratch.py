"""IKA-139: the fast path's patched leaves against the completion's own leaves, feature by feature.

`scratchpad/fastpath_refused.py` found (9/23, before IKA-119) 44 of 444 completion matrices
on `data/ika73/w12` games 0-9 that the hidden-bench fast path did not get to the bit. It
compared payoffs through a stand-in leaf that sums every array -- a sum that cannot see two
bench rows trade places and does not read `side` or `field` at all. This one compares the
inputs instead, and scores with a leaf that can see both:

1. For every fast-path decision, the port's leaves of the true position (`reference`) are
   patched per completion with `beliefnode._patched`, and the port is asked for the same
   node on the completion itself. For every cell the fast path shares (not `dirty`), the
   two cells' leaves are matched in order and every array is compared element by element;
   a difference is booked by (array, side, slot, feature name).
2. `belief_payoffs` against `batched_payoff` per completion, bit for bit, with a leaf that
   is a fixed random linear map of every element of every array (so a slot swap, a side
   feature or a field feature each move it).

    PYTHONPATH=<tree>/src POKEURAOU_RUST_NODE=1 \\
        python scratchpad/ika139_patched_vs_scratch.py <pool> --games 10 [--old-patch]

`--old-patch` puts back `_patched` as it was before IKA-119 (the positive control: the
measurement has to find the difference that fix removed).

What it found (9/23, 1 core each):

* games 0-9 on master c1b7e4c: inputs 0 differing leaves of 105,093; payoffs 0 of 444.
  On 250a1db (before IKA-121 and IKA-119): 20,349 cells, `team_hp_fraction`,
  `mega_available` and `can_mega` (also on active rows: the slot-numbered `can_mega` read
  the completion's `mega_capable_slots`); the summing leaf's 1,429 cells are all among the
  1,652 whose difference is outside `side` / `field`. On adb1e7b (IKA-121 only): 20,134
  cells, `side` plus the bench's `can_mega`.
* games 10-209 on master: inputs still 0 of 2,372,950 leaves, but payoffs differ in 179 of
  10,158 matrices, 3,886 cells, the same cells through both leaves -- all of them cells
  the port REFUSED (`breaksProtect move`), which part 1 skips because a refused cell has
  no leaves to compare. `belief_payoffs` left them at 0.0. Fixed in IKA-139 (a refused
  cell is dirty); after the fix, games 0-209: 0 of 10,602 matrices, both leaves.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from both_exact_calls import Leaf, menu  # noqa: E402
from replay_match import recorded  # noqa: E402

from pokeuraou import beliefnode, rustnode  # noqa: E402
from pokeuraou.beliefnode import belief_payoffs, reaches_bench  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoded, Encoder  # noqa: E402
from pokeuraou.hidden import completions, seen_slots  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoff, batched_payoffs  # noqa: E402
from pokeuraou.selection_book import (  # noqa: E402
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    BenchPrior,
    SelectionBook,
)
from pokeuraou.selfplay import GameRecord, _bench_weights  # noqa: E402
from pokeuraou.standings import find_cached_standings, load_standings  # noqa: E402
from pokeuraou.teams import load_archetypes, load_roster  # noqa: E402

ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def _patched_before_ika119(reference, side, slots, item, reg, position):  # noqa: ANN001, ANN202
    """`beliefnode._patched` before IKA-119: side vector and bench `can_mega` not rebuilt."""
    encoder = beliefnode._encoder_for(reg)
    source = encoder.encode_positions([item.position])
    out = Encoded(
        species=reference.species.copy(), ability=reference.ability.copy(),
        item=reference.item.copy(), moves=reference.moves.copy(), mon=reference.mon.copy(),
        mask=reference.mask.copy(), side=reference.side, field=reference.field,
        unknown_volatiles=dict(reference.unknown_volatiles),
    )
    for slot in slots:
        for name in ("species", "ability", "item", "moves", "mon", "mask"):
            getattr(out, name)[:, side, slot] = getattr(source, name)[0, side, slot]
    return out


class LinearLeaf:
    """A fixed random linear map of every element of every array, squashed.

    Unlike a sum, it tells two bench rows apart and reads `side` and `field`.
    """

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.weights: dict[str, np.ndarray] = {}

    def _w(self, name: str, size: int) -> np.ndarray:
        found = self.weights.get(name)
        if found is None or found.size != size:
            found = np.random.default_rng([139, len(name), size]).normal(size=size) * 0.01
            self.weights[name] = found
        return found

    def from_encoded(self, encoded) -> np.ndarray:  # noqa: ANN001
        n = len(encoded)
        raw = np.zeros(n, dtype=np.float64)
        for name in ARRAYS:
            flat = getattr(encoded, name).reshape(n, -1).astype(np.float64)
            # Not `flat @ w`: a matrix product's summation order can depend on the batch,
            # and the two paths score different batches. A row-wise reduction cannot.
            raw += (flat * self._w(name, flat.shape[1])).sum(axis=1)
        return 1.0 / (1.0 + np.exp(-raw))

    def __call__(self, positions: list) -> np.ndarray:
        return self.from_encoded(self.encoder.encode_positions(positions))


def feature_name(encoder: Encoder, name: str, index: tuple[int, ...]) -> str:
    # index is (side, slot, feature) for mon, (side, slot) / (side, slot, k) for ids,
    # (side, feature) for side, (feature,) for field.
    if name == "mon":
        s, p, f = index
        return f"mon[{s},{p}].{encoder.mon_names[f]}"
    if name == "side":
        s, f = index
        return f"side[{s}].{encoder.side_names[f]}"
    if name == "field":
        return f"field.{encoder.field_names[index[0]]}"
    return f"{name}[{','.join(map(str, index))}]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--seed", type=int, default=6601)
    ap.add_argument("--old-patch", action="store_true")
    ap.add_argument(
        "--selection-book", type=Path,
        default=HERE.parent / "data" / "selection" / "rizabanadohido-value-gen11L.jsonl.gz",
    )
    args = ap.parse_args()
    if not rustnode.available():
        raise SystemExit("the port is not available; there is no fast path to check")
    if args.old_patch:
        beliefnode._patched = _patched_before_ika119
    patch = beliefnode._patched

    roster = load_roster("rizabanadohido")
    reg, _archetypes = load_archetypes("wcs2026-regmb")
    register_mega_stones(reg)
    pool = load_standings(find_cached_standings("2026", "worlds"), reg).pool("all")
    book = SelectionBook.read(args.selection_book)
    encoder = Encoder(reg)
    leaf = LinearLeaf(encoder)
    sum_leaf = Leaf(encoder)  # the stand-in `fastpath_refused.py` scored with
    sum_cells: set[tuple] = set()
    sum_matrices = 0
    budget = Budget.matrix()
    port = rustnode.node_for(reg)

    decisions = patched_count = cells_compared = cells_off = shape_off = 0
    leaves_compared = 0
    features: Counter[str] = Counter()
    per_array: Counter[str] = Counter()
    off_cells: set[tuple] = set()
    off_cells_mon: set[tuple] = set()
    matrices = wrong_matrices = wrong_cells = 0
    payoff_cells: set[tuple] = set()
    worst = 0.0
    turns: Counter[int] = Counter()
    classes: Counter[tuple] = Counter()
    places: set[tuple] = set()
    restricted_off = 0
    wanted = set(range(args.first, args.first + args.games))
    for index, game in sorted(recorded(args.pool, wanted).items()):
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
        for number, decision in enumerate(game["decisions"]):
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
                continue
            decisions += 1
            turns[pos.turn] += 1
            ours = menu(reg, pos, 0, decision["ownActions"])
            theirs = menu(reg, pos, 1, decision["foeActions"])
            hidden = {
                side: (items[0].slots if items and not items[0].exact else ())
                for side, items in spreads.items()
            }
            ref = port.fill_encoded(pos, ours, theirs, budget)
            dirty = reaches_bench(reg, ours, theirs, hidden)
            for i, j, _root in ref.folded:
                dirty[i, j] = True
            ref_spans = {(i, j): (list(ix), list(w)) for i, j, ix, w in ref.spans}

            # 1. inputs, array by array
            for side, items in spreads.items():
                slots = hidden.get(side, ())
                for k, item in enumerate(items):
                    if item.exact or not slots:
                        continue
                    patched_count += 1
                    got = patch(ref.encoded, side, slots, item, reg, pos)
                    comp = port.fill_encoded(item.position, ours, theirs, budget)
                    comp_spans = {(i, j): (list(ix), list(w)) for i, j, ix, w in comp.spans}
                    for (i, j), (rix, rw) in ref_spans.items():
                        if dirty[i, j] or not rw:
                            continue
                        cells_compared += 1
                        cix, cw = comp_spans.get((i, j), ([], []))
                        if len(cix) != len(rix) or cw != rw:
                            shape_off += 1
                            off_cells.add((index, number, side, k, i, j))
                            continue
                        bad = False
                        bad_mon = False
                        for a, b in zip(rix, cix, strict=True):
                            leaves_compared += 1
                            for name in ARRAYS:
                                x = getattr(got, name)[a]
                                y = getattr(comp.encoded, name)[b]
                                if name in ("species", "ability", "item", "moves"):
                                    x, y = x.astype(np.int64), y.astype(np.int64)
                                diff = np.argwhere(x != y)
                                if len(diff):
                                    bad = True
                                    bad_mon |= name != "side" and name != "field"
                                    per_array[name] += 1
                                    for where in diff:
                                        idx = tuple(int(v) for v in where)
                                        if name == "moves":
                                            idx = idx[:2]
                                        features[feature_name(encoder, name, idx)] += 1
                        if bad_mon:
                            off_cells_mon.add((index, number, side, k, i, j))
                        if bad:
                            cells_off += 1
                            off_cells.add((index, number, side, k, i, j))

            # 2. payoffs through the linear leaf
            node = belief_payoffs(reg, pos, ours, theirs, leaf, budget=budget, spreads=spreads)
            for side, items in spreads.items():
                for k, item in enumerate(items):
                    expected, _notes = batched_payoff(
                        reg, item.position, ours, theirs, leaf, budget=budget
                    )
                    got_m = node.matrices[1 - side][k]
                    matrices += 1
                    off = np.argwhere(got_m != expected)
                    wrong_matrices += bool(len(off))
                    for i, j in off:
                        wrong_cells += 1
                        payoff_cells.add((index, number, side, k, int(i), int(j)))
                        classes[(
                            "dirty" if dirty[i, j] else "shared",
                            "exact" if item.exact else "patched",
                        )] += 1
                        places.add((index, number))
                    if len(off):
                        wanted_cells = [
                            (int(i), int(j)) for i, j in np.argwhere(dirty)
                        ]
                        if wanted_cells:
                            part, _n, _e = batched_payoffs(
                                reg, item.position, ours, theirs, [leaf], budget=budget,
                                cells=wanted_cells,
                            )
                            restricted_off += sum(
                                part[0][i, j] != expected[i, j] for i, j in wanted_cells
                            )
                        worst = max(worst, abs(float(got_m[i, j] - expected[i, j])))
            # 3. and through the summing leaf, to ask whether it saw the same cells
            node = belief_payoffs(
                reg, pos, ours, theirs, sum_leaf, budget=budget, spreads=spreads
            )
            for side, items in spreads.items():
                for k, item in enumerate(items):
                    expected, _notes = batched_payoff(
                        reg, item.position, ours, theirs, sum_leaf, budget=budget
                    )
                    off = np.argwhere(node.matrices[1 - side][k] != expected)
                    sum_matrices += bool(len(off))
                    for i, j in off:
                        sum_cells.add((index, number, side, k, int(i), int(j)))

    label = "pre-IKA-119 _patched" if args.old_patch else "current _patched"
    print(f"[{label}] games {args.first}..{args.first + args.games - 1}: "
          f"{decisions} fast-path move decisions, turns {dict(sorted(turns.items()))}")
    print(f"inputs: {patched_count} patched completions, {cells_compared} shared cells, "
          f"{leaves_compared} leaves compared; cells with a differing leaf {cells_off}, "
          f"cells whose leaf layout differs {shape_off}")
    print(f"  leaves differing, by array: {dict(per_array)}")
    for name, count in features.most_common(30):
        print(f"    {count:7d}  {name}")
    print(f"payoffs (linear leaf): {matrices} matrices, {wrong_matrices} not equal to the bit, "
          f"{wrong_cells} cells (worst {worst:.3g})")
    print(f"  cell sets: input-differing {len(off_cells)}, payoff-differing {len(payoff_cells)}, "
          f"payoff but not input {len(payoff_cells - off_cells)}, "
          f"input but not payoff {len(off_cells - payoff_cells)}")
    print(f"  cells with a differing leaf outside side/field (all a sum leaf can see): "
          f"{len(off_cells_mon)}")
    print(f"  summing leaf: {sum_matrices} matrices, {len(sum_cells)} cells differ; "
          f"of them in the input-differing set {len(sum_cells & off_cells)}, "
          f"in the outside-side/field set {len(sum_cells & off_cells_mon)}")
    print(f"  payoff-differing cells by (cell, completion): {dict(classes)}")
    print(f"  dirty cells where cells=-restricted batched_payoffs != full batched_payoff "
          f"(in matrices that differ): {restricted_off}")
    print(f"  decisions (game, decision#) with a differing cell: {sorted(places)}")


if __name__ == "__main__":
    main()
