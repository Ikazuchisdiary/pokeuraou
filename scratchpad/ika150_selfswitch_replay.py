"""IKA-150: where a hidden-bench self-switch decision's time goes, and fast == definition.

Replays recorded `selfswitch` decisions whose opponent had an unseen slot at the pause
(rebuilt as IKA-120's `ika120_selfswitch_fire_rate.py` rebuilds them, whose functions this
imports) through `_do_self_switch_node` with `definition=True` (every completion resolved
from scratch) and with the shared path, in alternating order per decision:

    POKEURAOU_TIMING=<scratch dir> PYTHONPATH=<tree>/src python \\
        scratchpad/ika150_selfswitch_replay.py <pool dir> <model.pt> --positions 50

Timing must be on (the variable is read at import): the per-stage split is the change in
each stage's clock across one call. Per decision it prints nothing; at the end:

* per stage, mean ms a decision for each path, and the counts (options, completions,
  leaves, resumes, shared options)
* with the value net: chosen switch-in and `search_value` compared, bit for bit
* with a test leaf that weights every element of every encoded array (`tests/
  test_beliefnode.py`'s `_SlotLeaf`): the leaf positions handed to it compared in JSON,
  and `search_value` bit for bit

Rough CPU numbers: one core, a shared machine, one torch thread.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scratchpad"))

import ika120_selfswitch_fire_rate as base  # noqa: E402

ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


class SlotLeaf:
    """Every element of every encoded array with a weight of its own, reduced per row."""

    def __init__(self, encoder) -> None:  # noqa: ANN001
        self.encoder = encoder
        self._weights: dict[tuple[str, int], np.ndarray] = {}
        self.seen: list[list[str]] = []

    def _w(self, name: str, size: int) -> np.ndarray:
        key = (name, size)
        if key not in self._weights:
            self._weights[key] = (
                np.random.default_rng([150, len(name), size]).normal(size=size) * 0.01
            )
        return self._weights[key]

    def __call__(self, positions: list) -> np.ndarray:  # noqa: ANN001
        self.seen.append([json.dumps(p.to_json(), sort_keys=True) for p in positions])
        encoded = self.encoder.encode_positions(positions)
        n = len(encoded)
        raw = np.zeros(n, dtype=np.float64)
        for name in ARRAYS:
            flat = getattr(encoded, name).reshape(n, -1).astype(np.float64)
            raw += (flat * self._w(name, flat.shape[1])).sum(axis=1)
        return 1.0 / (1.0 + np.exp(-raw))


def _clocks() -> tuple[dict[str, float], dict[str, int]]:
    from pokeuraou import timing

    return (
        {name: s.wall for name, s in timing._STAGES.items()},
        dict(timing._COUNTS),
    )


def _call(pause, hidden, leaf, definition: bool):  # noqa: ANN001, ANN202
    from pokeuraou.selfplay import GameRecord, _do_self_switch_node

    record = GameRecord(own_team=[], foe_team=[], foe_archetype="replay")
    walls, counts = _clocks()
    started = time.perf_counter()
    _do_self_switch_node(
        base.STATE["reg"], pause, record, (leaf, leaf), None, hidden=hidden,
        definition=definition,
    )
    elapsed = time.perf_counter() - started
    walls2, counts2 = _clocks()
    made = record.decisions[-1]
    return {
        "elapsed": elapsed,
        "stages": {k: walls2[k] - walls.get(k, 0.0) for k in walls2 if walls2[k] != walls.get(k, 0.0)},
        "counts": {k: counts2[k] - counts.get(k, 0) for k in counts2 if counts2[k] != counts.get(k, 0)},
        "chosen": made.own_chosen if made.own_actions != ["pass"] else made.foe_chosen,
        "value": made.search_value,
        "unmodelled": list(record.unmodelled),
    }


def run_one(case: dict, flip: bool, test_leaf: SlotLeaf) -> dict:
    from pokeuraou.selfplay import _HiddenBench

    reg, leaf = base.STATE["reg"], base.STATE["leaf"]
    sheets = base.sheets_for(case["game"])
    if sheets is None:
        return {"skipped": "sheet"}
    pause, why = base._pause_for(reg, case)
    if pause is None:
        return {"skipped": why}
    hidden = _HiddenBench(
        sheets=sheets,
        seen=[frozenset(s) for s in case["seen"]],
        bench_prior=None,
        leads=[frozenset(x) if x is not None else None for x in case["leads"]],
    )
    order = (False, True) if flip else (True, False)
    out: dict = {"turn": case["decision"]["turn"]}
    for definition in order:
        out["def" if definition else "fast"] = _call(pause, hidden, leaf, definition)
    for definition in order:
        test_leaf.seen = []
        got = _call(pause, hidden, test_leaf, definition)
        got["positions"] = test_leaf.seen
        out["test_def" if definition else "test_fast"] = got
    return out


def main() -> None:
    from pokeuraou import timing

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("--positions", type=int, default=50)
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=150)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not timing.ON:
        raise SystemExit(f"set {timing.ENV_DIR}: the split is read off the stage clocks")
    started = time.time()
    cases, counts = base.collect(args.pool, args.games)
    hidden_cases = [c for c in cases if c["unseen"] > 0]
    print(f"{counts['selfswitch']} selfswitch decisions in {counts['games']} games, "
          f"{len(hidden_cases)} replayable with an unseen opponent slot", flush=True)
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(hidden_cases), size=min(args.positions, len(hidden_cases)),
                             replace=False))
    base._init(str(args.model))
    from pokeuraou.encode import Encoder

    test_leaf = SlotLeaf(Encoder(base.STATE["reg"]))
    results = []
    timing.decided("between")  # ends `startup` here, not inside the first decision
    for n, i in enumerate(pick):
        results.append(run_one(hidden_cases[i], bool(n % 2), test_leaf))
    done = [r for r in results if "skipped" not in r]
    print(f"sampled {len(pick)}, replayed {len(done)}, "
          f"skipped {[r['skipped'] for r in results if 'skipped' in r]}, "
          f"wall {time.time() - started:.0f}s")

    same_choice = sum(r["def"]["chosen"] == r["fast"]["chosen"] for r in done)
    same_value = sum(r["def"]["value"] == r["fast"]["value"] for r in done)
    same_notes = sum(r["def"]["unmodelled"] == r["fast"]["unmodelled"] for r in done)
    print(f"value net: chosen identical {same_choice}/{len(done)}, search_value bit-identical "
          f"{same_value}/{len(done)}, unmodelled identical {same_notes}/{len(done)}")
    t_pos = sum(r["test_def"]["positions"] == r["test_fast"]["positions"] for r in done)
    t_val = sum(r["test_def"]["value"] == r["test_fast"]["value"] for r in done)
    t_ch = sum(r["test_def"]["chosen"] == r["test_fast"]["chosen"] for r in done)
    leaves = sum(len(r["test_def"]["positions"][0]) for r in done)
    print(f"test leaf: leaf positions identical in JSON {t_pos}/{len(done)} ({leaves} leaves), "
          f"search_value bit-identical {t_val}/{len(done)}, chosen identical {t_ch}/{len(done)}")

    for path in ("def", "fast"):
        rows = [r[path] for r in done]
        stages: dict[str, float] = {}
        tallies: dict[str, int] = {}
        for row in rows:
            for k, v in row["stages"].items():
                stages[k] = stages.get(k, 0.0) + v
            for k, v in row["counts"].items():
                tallies[k] = tallies.get(k, 0) + v
        n = len(rows)
        total = sum(row["elapsed"] for row in rows)
        staged = sum(stages.values())
        print(f"\n[{path}] {n} decisions, {1000 * total / n:.1f} ms a decision "
              f"(median {1000 * float(np.median([row['elapsed'] for row in rows])):.1f}, "
              f"max {1000 * max(row['elapsed'] for row in rows):.1f})")
        for k, v in sorted(stages.items(), key=lambda kv: -kv[1]):
            print(f"  {k:22s} {1000 * v / n:8.1f} ms  {v / total:6.1%}")
        print(f"  {'(unstaged)':22s} {1000 * (total - staged) / n:8.1f} ms  "
              f"{(total - staged) / total:6.1%}")
        for k, v in sorted(tallies.items()):
            print(f"  count {k:28s} {v / n:8.2f} a decision")
    if args.out is not None:
        for r in results:
            for key in ("test_def", "test_fast"):
                if key in r:
                    r[key]["positions"] = len(r[key]["positions"][0])
        args.out.write_bytes(json.dumps({"counts": counts, "results": results}).encode())


if __name__ == "__main__":
    main()
