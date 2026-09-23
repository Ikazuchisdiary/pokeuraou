"""IKA-121: does the encoder's `can_mega` sit on the stone holder, on recorded positions?

Provenance: copied on 9/23 from `mega_slots_stale.py`, the script the IKA-116 inventory
session wrote to find this (it compared the stored `side.mega_capable_slots` with the
stone holders), and extended to read the encoder's own output. Reading the encoder is what
lets one script answer both halves: "is the feature wrong" on the tree before the fix and
"is it right" on the tree after it.

Per recorded decision and side with the side's mega unspent, three sets of party
positions:

    truth    Pokemon whose species and held item are a mega pairing (`reg.mega_target`),
             the rule `actions.py` offers the mega move by
    stale    `side.mega_capable_slots`, slot numbers written at game start and never moved
    encoded  rows where the encoder in this tree wrote can_mega = 1

Both rules are also rebuilt here as arrays (the encoder's output with the can_mega column
and the side's mega_available overwritten), and the encoder has to equal exactly one of
them -- the stale one before the fix, the identity one after. That is the check on the
check: the other rule's count is the positive control.

`--value` scores a sample of the decisions whose encoding the fix changes, under both
rules, with a trained model on one CPU thread -- no search, only the leaf.

    uv run python scratchpad/ika121_can_mega_stale.py data/ika73/w12 3000
    uv run python scratchpad/ika121_can_mega_stale.py data/ika73/w12 3000 \\
        --value data/models/value-gen11L.pt --sample 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoded, Encoder  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def with_rule(encoded: Encoded, positions: list[Position], rule: str, enc: Encoder) -> Encoded:
    """The encoder's arrays with can_mega and mega_available rewritten by one rule."""
    mon = encoded.mon.copy()
    side = encoded.side.copy()
    k = enc.mon_names.index("can_mega")
    a = enc.side_names.index("mega_available")
    reg = enc.reg
    for b, pos in enumerate(positions):
        for s, one in enumerate(pos.sides):
            holders = {
                m.slot for m in one.pokemon if reg.mega_target(m.species, m.item) is not None
            }
            named = set(one.mega_capable_slots) if rule == "stale" else holders
            side[b, s, a] = 1.0 if not one.mega_used and named else 0.0
            for p, m in enumerate(one.pokemon[: enc.mons_per_side]):
                mon[b, s, p, k] = (
                    1.0 if m.slot in named and not one.mega_used and not m.is_mega else 0.0
                )
    out = Encoded(**{name: getattr(encoded, name) for name in ARRAYS},
                  unknown_volatiles=dict(encoded.unknown_volatiles))
    out.mon = mon
    out.side = side
    return out


def same(x: Encoded, y: Encoded, b: int) -> bool:
    return all(np.array_equal(getattr(x, n)[b], getattr(y, n)[b]) for n in ARRAYS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("limit", type=int, nargs="?", default=3000)
    ap.add_argument("--value", type=Path, default=None, help="a model to score both rules with")
    ap.add_argument("--sample", type=int, default=400, help="changed decisions to score")
    ap.add_argument("--seed", type=int, default=121)
    args = ap.parse_args()

    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    enc = Encoder(reg)
    k = enc.mon_names.index("can_mega")

    games = decisions = 0
    checked = stale_pairs = stale_games = 0
    encoded_vs_truth = encoded_vs_stale = 0
    available_rules_differ = 0
    changed_decisions = 0
    changed_games = 0
    matches = {"stale": 0, "identity": 0}
    changed: list[tuple[Position, int]] = []  # (position, affected side or -1 for both)
    example = None

    for path in sorted(args.pool.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if games >= args.limit:
                    break
                game = json.loads(line)
                games += 1
                positions = [Position.from_json(d["position"]) for d in game["decisions"]]
                decisions += len(positions)
                encoded = enc.encode_positions(positions)
                rules = {r: with_rule(encoded, positions, r, enc) for r in ("stale", "identity")}
                hit = moved = False
                for b, pos in enumerate(positions):
                    for rule, arrays in rules.items():
                        matches[rule] += same(encoded, arrays, b)
                    if not same(rules["stale"], rules["identity"], b):
                        changed_decisions += 1
                        moved = True
                        sides = []
                    else:
                        sides = None
                    for s, one in enumerate(pos.sides):
                        holders = {
                            m.slot for m in one.pokemon
                            if not m.is_mega and reg.mega_target(m.species, m.item) is not None
                        }
                        available_rules_differ += (
                            float(rules["stale"].side[b, s, enc.side_names.index("mega_available")])
                            != float(rules["identity"].side[b, s,
                                                            enc.side_names.index("mega_available")])
                        )
                        if one.mega_used:
                            continue
                        if not holders and not one.mega_capable_slots:
                            continue
                        checked += 1
                        wrote = {
                            m.slot for p, m in enumerate(one.pokemon[: enc.mons_per_side])
                            if encoded.mon[b, s, p, k] == 1.0
                        }
                        if holders != set(one.mega_capable_slots):
                            stale_pairs += 1
                            hit = True
                            if sides is not None:
                                sides.append(s)
                            if example is None:
                                example = (path.name, game.get("gameIndex"), pos.turn, s,
                                           [(m.slot, m.species, m.item) for m in one.pokemon],
                                           list(one.mega_capable_slots), sorted(wrote))
                        encoded_vs_truth += wrote != holders
                        encoded_vs_stale += wrote != set(one.mega_capable_slots)
                    if sides is not None:
                        changed.append((pos, sides[0] if len(sides) == 1 else -1))
                stale_games += hit
                changed_games += moved
        if games >= args.limit:
            break

    print(f"{games} games, {decisions} decisions; encoder tree: {enc.reg.meta.format_id}")
    print(f"(decision, side) pairs with the mega unspent and a holder or a slot named: {checked}")
    print(f"  mega_capable_slots != holders: {stale_pairs} ({stale_pairs / max(checked, 1):.1%}); "
          f"games with one: {stale_games} ({stale_games / max(games, 1):.1%})")
    print(f"  encoder's can_mega != holders:            {encoded_vs_truth}")
    print(f"  encoder's can_mega != mega_capable_slots: {encoded_vs_stale}")
    print(f"decisions whose arrays equal the stale rule's: {matches['stale']} of {decisions}; "
          f"the identity rule's: {matches['identity']} of {decisions}")
    print(f"decisions the fix changes (any array): {changed_decisions} "
          f"({changed_decisions / max(decisions, 1):.1%}); games: {changed_games} "
          f"({changed_games / max(games, 1):.1%})")
    print(f"(decision, side) pairs where the two rules' mega_available differ: "
          f"{available_rules_differ}")
    print("example:", example)

    if args.value is None:
        return
    import torch

    from pokeuraou.value import load_model

    torch.set_num_threads(1)
    net, _meta = load_model(args.value, enc)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(changed), size=min(args.sample, len(changed)), replace=False)
    chosen = [changed[i] for i in sorted(pick)]
    positions = [pos for pos, _s in chosen]
    encoded = enc.encode_positions(positions)
    values = {}
    for rule in ("stale", "identity"):
        arrays = with_rule(encoded, positions, rule, enc)
        batch = {n: torch.from_numpy(np.ascontiguousarray(getattr(arrays, n))) for n in ARRAYS}
        with torch.no_grad():
            values[rule] = torch.sigmoid(net(batch)).double().numpy()
    delta = values["identity"] - values["stale"]
    # Signed from the affected side's own view, where only one side's rows moved.
    signed = np.array(
        [d if s == 0 else -d for d, (_p, s) in zip(delta, chosen, strict=True) if s >= 0]
    )
    size = np.abs(delta)
    print(f"\nvalue {args.value.name}, {len(chosen)} changed decisions (seed {args.seed}), "
          f"P(side 0 wins) identity rule - stale rule:")
    print(f"  |d| mean {size.mean():.4f}  median {np.median(size):.4f}  "
          f"p90 {np.quantile(size, 0.9):.4f}  max {size.max():.4f}")
    print(f"  |d| > 0.01: {(size > 0.01).mean():.1%}   > 0.05: {(size > 0.05).mean():.1%}")
    if len(signed):
        print(f"  signed, from the side whose can_mega moved ({len(signed)} one-side decisions): "
              f"mean {signed.mean():+.4f}  ({(signed > 0).mean():.1%} up)")


if __name__ == "__main__":
    main()
