"""IKA-413: how much of the production's one-turn game does the spread-move fix change?

Positions are sampled from recorded games (one decision each, uniformly over decisions, as in
tools/slot_swap_check.py). With the NEW exe the candidates are narrowed (width 6, both sides) and
the hp-share matrix is filled; with the OLD exe the same candidates are filled again. A cell that
differs is a turn a spread move's second target meets another answer.

    python spread_impact.py --dir DATA/selfplay-mc4 --count 3000 --old OLD.exe --new NEW.exe --out OUT.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/tmp/ika413/wt/src")
sys.path.insert(0, "C:/tmp/ika413/wt/tools")
from slot_swap_check import sample_positions  # noqa: E402

from pokeuraou import rustnode  # noqa: E402
from pokeuraou.budget import Budget  # noqa: E402
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.payoff import HP_SHARE  # noqa: E402
from pokeuraou.port import batched_payoff  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

HOLDERS = {"friendguard", "cloudnine", "airlock", "fairyaura", "darkaura", "aurabreak"}
SPREAD_NOTE = {"allAdjacentFoes", "allAdjacent"}


def use(exe: str) -> None:
    os.environ[rustnode.ENV_ENABLE] = "1"
    os.environ[rustnode.ENV_BINARY] = exe
    rustnode.reset()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    samples = sample_positions(Path(args.dir), args.count, args.seed)
    reg = load_regulation(Position.from_json(samples[0]["position"]).format)
    register_mega_stones(reg)
    budget = Budget.matrix()
    t0 = time.time()
    use(args.new)
    kept = []
    for k, sample in enumerate(samples):
        pos = Position.from_json(sample["position"])
        if pos.ended or pos.request_state != "move":
            continue
        try:
            ours = narrow(reg, pos, 0, limit=args.limit).actions
            theirs = narrow(reg, pos, 1, limit=args.limit).actions
            if not ours or not theirs:
                continue
            mat, _ = batched_payoff(reg, pos, ours, theirs, HP_SHARE.batch, budget=budget)
        except Exception:  # noqa: BLE001
            continue
        kept.append((sample, pos, ours, theirs, mat))
        if (k + 1) % 500 == 0:
            print(f"new {k + 1}/{len(samples)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    use(args.old)
    tally = Counter()
    worst_all = []
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        for k, (sample, pos, ours, theirs, new_mat) in enumerate(kept):
            try:
                old_mat, _ = batched_payoff(reg, pos, ours, theirs, HP_SHARE.batch, budget=budget)
            except Exception:  # noqa: BLE001
                tally["old_failed"] += 1
                continue
            diff = np.abs(new_mat - old_mat)
            worst = float(diff.max())
            abil = [m.ability for side in pos.sides for m in side.active_pokemon() if m is not None]
            holder = any(a in HOLDERS for a in abil)
            tally["positions"] += 1
            tally["with_holder_active"] += holder
            tally["cells"] += diff.size
            tally["cells>1e-6"] += int((diff > 1e-6).sum())
            tally["cells>2e-3"] += int((diff > 2e-3).sum())
            tally["pos>1e-6"] += int(worst > 1e-6)
            tally["pos>2e-3"] += int(worst > 2e-3)
            tally["pos>1e-6 with holder"] += int(worst > 1e-6 and holder)
            worst_all.append(worst)
            if worst > 1e-6:
                out.write(json.dumps({
                    "file": sample["file"], "game": sample["gameIndex"], "turn": sample["turn"],
                    "worst": worst, "abilities": abil,
                    "cells": [[int(i), int(j), float(old_mat[i, j]), float(new_mat[i, j]),
                               ours[i].to_choice(), theirs[j].to_choice()]
                              for i, j in zip(*np.nonzero(diff > 1e-6), strict=True)][:6],
                    "position": sample["position"]}, ensure_ascii=False) + "\n")
    w = np.array(worst_all) if worst_all else np.zeros(1)
    print(dict(tally))
    print(f"worst-cell per position: mean {w.mean():.2e}  max {w.max():.3f}  seconds {time.time() - t0:.0f}")


if __name__ == "__main__":
    main()
