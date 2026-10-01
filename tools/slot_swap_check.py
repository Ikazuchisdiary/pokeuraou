"""Does exchanging the left and right Pokemon of a side change the turn? (IKA-412)

For positions sampled from recorded games (one decision each, uniformly over decisions), the
port fills the one-turn hp-share matrix of the position as recorded, and again of the
position with one side's two positions exchanged (`slotswap.swap_positions`), the menus
carried over (`slotswap.swap_action`). A cell that differs is a rule that reads a position.

    python tools/slot_swap_check.py --dir DATA/selfplay-mc4 --count 3000 --out OUT.jsonl
    python tools/slot_swap_check.py ... --control actions   # a swap that forgets the menus
    python tools/slot_swap_check.py ... --control sources   # ... that forgets source slots

Variants: `own` (side 0's positions), `foe` (side 1's), `both`. `--control` breaks the swap on
purpose, to show the comparison can fail.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from pokeuraou import slotswap
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.port import batched_payoff
from pokeuraou.position import Position
from pokeuraou.regulation import load_regulation

VARIANTS = {"own": (True, False), "foe": (False, True), "both": (True, True)}


def sample_positions(directory: Path, count: int, seed: int) -> list[dict]:
    """`count` recorded move decisions, a file, a line and a decision drawn uniformly."""
    rng = random.Random(seed)
    files = sorted(directory.glob("games-*.jsonl"))
    sizes = [f.stat().st_size for f in files]
    out: list[dict] = []
    while len(out) < count:
        f = rng.choices(files, weights=sizes)[0]
        size = f.stat().st_size
        with f.open("rb") as fh:
            fh.seek(rng.randrange(size))
            fh.readline()
            line = fh.readline()
            if not line.strip():
                continue
        game = json.loads(line)
        decisions = [d for d in game["decisions"] if d.get("kind") == "move"]
        if not decisions:
            continue
        d = rng.choice(decisions)
        out.append({"file": f.name, "gameIndex": game.get("gameIndex"), "turn": d["turn"],
                    "position": d["position"]})
    return out


def swapped_pair(pos: Position, ours, theirs, which: tuple[bool, bool], control: str | None):
    moved = slotswap.swap_positions(pos, which)
    if control == "sources":
        # The same swap, but every Effect keeps its source_slot as it was.
        keep = slotswap._swap_position_of_source
        slotswap._swap_position_of_source = lambda source, _sides: source
        try:
            moved = slotswap.swap_positions(pos, which)
        finally:
            slotswap._swap_position_of_source = keep
    if control == "actions":
        return moved, list(ours), list(theirs)
    return (
        moved,
        [slotswap.swap_action(a, which[0], which[1]) for a in ours],
        [slotswap.swap_action(a, which[1], which[0]) for a in theirs],
    )


def describe(pos: Position) -> dict:
    """What the position holds that a rule could read, for sorting the mismatches."""
    out: dict[str, list[str]] = {"ability": [], "volatile": [], "slotcond": [], "sidecond": [],
                                 "field": [], "item": [], "status": []}
    for side in pos.sides:
        for mon in side.active_pokemon():
            if mon is None:
                continue
            out["ability"].append(mon.ability)
            out["item"].append(mon.item or "")
            out["status"].append(mon.status or "")
            out["volatile"].extend(v.id for v in mon.volatiles)
        for group in side.slot_conditions:
            out["slotcond"].extend(c.id for c in group)
        out["sidecond"].extend(c.id for c in side.side_conditions)
    out["field"] = [x for x in (pos.field.weather, pos.field.terrain) if x]
    out["field"] += [e.id for e in pos.field.pseudo_weather]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--control", choices=["actions", "sources"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = sample_positions(Path(args.dir), args.count, args.seed)
    reg = load_regulation(Position.from_json(samples[0]["position"]).format)
    register_mega_stones(reg)
    budget = Budget.matrix()
    started = time.time()
    tally: dict[str, Counter] = {v: Counter() for v in VARIANTS}
    sizes: dict[str, list[float]] = {v: [] for v in VARIANTS}
    all_features: Counter = Counter()
    bad_features: dict[str, Counter] = {v: Counter() for v in VARIANTS}
    refused = Counter()
    n_done = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for k, sample in enumerate(samples):
            pos = Position.from_json(sample["position"])
            if pos.ended or pos.request_state != "move":
                continue
            try:
                ours = narrow(reg, pos, 0, limit=args.limit).actions
                theirs = narrow(reg, pos, 1, limit=args.limit).actions
                if not ours or not theirs:
                    continue
                base, _ = batched_payoff(reg, pos, ours, theirs, HP_SHARE.batch, budget=budget)
            except Exception as exc:  # noqa: BLE001
                refused[type(exc).__name__ + ": " + str(exc)[:80]] += 1
                continue
            n_done += 1
            feats = describe(pos)
            keys = {f"{kind}:{x}" for kind, xs in feats.items() for x in xs}
            all_features.update(keys)
            for name, which in VARIANTS.items():
                moved, mo, mt = swapped_pair(pos, ours, theirs, which, args.control)
                try:
                    mat, _ = batched_payoff(reg, moved, mo, mt, HP_SHARE.batch, budget=budget)
                except Exception as exc:  # noqa: BLE001
                    refused[f"{name} " + type(exc).__name__ + ": " + str(exc)[:80]] += 1
                    continue
                diff = np.abs(mat - base)
                worst = float(diff.max())
                tally[name]["positions"] += 1
                tally[name]["cells"] += diff.size
                tally[name]["cells>1e-6"] += int((diff > 1e-6).sum())
                tally[name]["cells>2e-3"] += int((diff > 2e-3).sum())
                tally[name]["pos>1e-6"] += int(worst > 1e-6)
                tally[name]["pos>2e-3"] += int(worst > 2e-3)
                sizes[name].append(worst)
                if worst > 1e-6:
                    bad_features[name].update(keys)
                    out.write(json.dumps({
                        "variant": name, "file": sample["file"], "game": sample["gameIndex"],
                        "turn": sample["turn"], "worst": worst,
                        "cells": [[int(i), int(j), float(base[i, j]), float(mat[i, j]),
                                   ours[i].to_choice(), theirs[j].to_choice()]
                                  for i, j in zip(*np.nonzero(diff > 1e-6), strict=True)][:6],
                        "features": feats, "position": sample["position"]},
                        ensure_ascii=False) + "\n")
            if (k + 1) % 500 == 0:
                print(f"{k + 1}/{len(samples)} {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    print(f"positions solved {n_done} of {len(samples)}; refused/failed {sum(refused.values())}")
    for why, c in refused.most_common(8):
        print(f"  {c:5d}  {why}")
    for name in VARIANTS:
        t = tally[name]
        s = np.array(sizes[name]) if sizes[name] else np.zeros(1)
        print(f"{name:5s} positions {t['positions']}  differing positions (>1e-6) {t['pos>1e-6']}"
              f"  (>2e-3) {t['pos>2e-3']}  cells {t['cells>1e-6']}/{t['cells']}"
              f"  worst cell mean {s.mean():.2e} max {s.max():.3f}")
    print("features over-represented among differing positions (variant: feature count_bad/count_all):")
    for name in VARIANTS:
        rows = [(f, c, all_features[f]) for f, c in bad_features[name].most_common(25)]
        print(f"  {name}: " + "; ".join(f"{f} {c}/{a}" for f, c, a in rows))


if __name__ == "__main__":
    main()
