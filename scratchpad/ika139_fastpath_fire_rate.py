"""IKA-139: with the shipping net, does the fast path ever answer differently from the definition?

The same 200 recorded positions, menus, sheets, completions and leaf as
`scratchpad/ika119_patched_side_fire_rate.py` (imported from it, so the two cannot drift),
solved by `belief_solve` twice with two of these arms:

* `fast`       -- `belief_payoffs` as it ships (the shared path, `_patched` as on master);
* `definition` -- `beliefnode._per_completion`: a matrix per completion, resolved from
                  scratch, which is what the fast path is defined to equal;
* `old-patch`  -- the shared path with `_patched` as it was before IKA-119;
* `unfixed`    -- `belief_payoffs` from the file given as `--before` (master c1b7e4c, i.e.
                  before IKA-139: a cell the port refuses is left at 0.0). Its `_patched`
                  calls are not counted.

    PYTHONPATH=<tree>/src POKEURAOU_RUST_NODE=1 python scratchpad/ika139_fastpath_fire_rate.py \\
        <pool dir> <model.pt> --arms fast definition

`--arms unfixed fast` is how often the IKA-139 fix moves the answer. `--arms fast fast` is
the null control (every number must be 0); `--arms old-patch
definition` is the positive control (it must move, by about what IKA-119 measured).
Runs in one process on one core: no pool, one torch thread.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import ika119_patched_side_fire_rate as base  # noqa: E402

BEFORE = None


def _definition(reg, position, ours, theirs, evaluate, *, budget, spreads):  # noqa: ANN001, ANN202
    from pokeuraou import beliefnode

    return beliefnode._per_completion(reg, list(ours), list(theirs), evaluate, budget, spreads)


def solve(arm: str, case: dict, pos, menus, spreads):  # noqa: ANN001, ANN201
    from pokeuraou import beliefnode
    from pokeuraou.resolve import Budget
    from pokeuraou.search import belief_solve

    shipped_payoffs, shipped_patch = beliefnode.belief_payoffs, beliefnode._patched
    calls = {"patch": 0}

    def counted(*a, **k):  # noqa: ANN002, ANN003, ANN202
        calls["patch"] += 1
        return (base._patched_before_ika119 if arm == "old-patch" else shipped_patch)(*a, **k)

    try:
        beliefnode._patched = counted
        if arm == "definition":
            beliefnode.belief_payoffs = _definition
        elif arm == "unfixed":
            beliefnode.belief_payoffs = BEFORE.belief_payoffs
        leaf = base.STATE["leaf"]
        answer = belief_solve(
            base.STATE["reg"], pos, menus[0], menus[1], spreads, {0: leaf, 1: leaf},
            budget=Budget.matrix(),
        )
    finally:
        beliefnode.belief_payoffs, beliefnode._patched = shipped_payoffs, shipped_patch
    return answer, calls["patch"]


def run_one(case: dict, arms: tuple[str, str]) -> dict:
    from pokeuraou.actions import side_actions
    from pokeuraou.equilibrium import EquilibriumError
    from pokeuraou.hidden import completions
    from pokeuraou.position import Position

    reg = base.STATE["reg"]
    sheets = base.sheets_for(case)
    if sheets is None:
        return {"skipped": "sheet"}
    pos = Position.from_json(case["position"])
    menus = []
    for s, recorded in ((0, case["own"]), (1, case["foe"])):
        legal = {a.to_choice(): a for a in side_actions(reg, pos, s)}
        menu = [legal.get(c) for c in recorded]
        if not menu or any(a is None for a in menu):
            return {"skipped": "menu"}
        menus.append(menu)
    try:
        spreads = {
            s: completions(reg, pos, s, sheets[s], seen=frozenset(case["shown"][s]))
            for s in (0, 1)
        }
    except ValueError:
        return {"skipped": "completions"}
    started = time.perf_counter()
    try:
        (old, old_calls), (new, new_calls) = (
            solve(arm, case, pos, menus, spreads) for arm in arms
        )
    except EquilibriumError:
        return {"skipped": "equilibrium"}
    out = {"seconds": time.perf_counter() - started, "patch_calls": [old_calls, new_calls]}
    for s in (0, 1):
        a, b = old[s], new[s]
        out[f"dv{s}"] = b.value - a.value
        out[f"tv{s}"] = 0.5 * float(np.abs(b.strategy - a.strategy).sum())
        out[f"argmax{s}"] = int(np.argmax(b.strategy) != np.argmax(a.strategy))
        out[f"crn{s}"] = base._crn_change(a.strategy, b.strategy)
        out[f"bits{s}"] = int(
            not np.array_equal(a.strategy, b.strategy) or a.value != b.value
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", type=Path)
    ap.add_argument("model", type=Path)
    ap.add_argument("--arms", nargs=2, default=["fast", "definition"],
                    choices=["fast", "definition", "old-patch", "unfixed"])
    ap.add_argument("--before", type=Path, help="beliefnode.py for the `unfixed` arm")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=119)
    args = ap.parse_args()
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    if "unfixed" in args.arms:
        from both_exact_calls import load_before

        global BEFORE
        BEFORE = load_before(args.before)
    started = time.time()
    cases, counts = base.collect(args.pool, args.games, args.positions, args.seed)
    print(f"{counts['games']} games, {counts['moves']} move decisions, "
          f"{counts['hidden_moves']} with a hidden slot; sampled {len(cases)}; "
          f"arms {args.arms[0]} -> {args.arms[1]}", flush=True)
    base._init(str(args.model), False)
    results = [run_one(case, tuple(args.arms)) for case in cases]
    done = [r for r in results if "skipped" not in r]
    skipped: dict[str, int] = {}
    for r in results:
        if "skipped" in r:
            skipped[r["skipped"]] = skipped.get(r["skipped"], 0) + 1
    print(f"solved {len(done)}, skipped {skipped}, wall {time.time() - started:.0f}s; "
          f"shared path ran (patch calls > 0) in arm 1 {sum(r['patch_calls'][0] > 0 for r in done)}, "
          f"arm 2 {sum(r['patch_calls'][1] > 0 for r in done)} of {len(done)}")
    for s in (0, 1):
        dv = np.abs([r[f"dv{s}"] for r in done])
        tv = np.array([r[f"tv{s}"] for r in done])
        crn = np.array([r[f"crn{s}"] for r in done])
        am = np.array([r[f"argmax{s}"] for r in done])
        bits = np.array([r[f"bits{s}"] for r in done])
        print(f"side {s}: not bit-identical (value or strategy) {bits.mean():.1%}  "
              f"|dv| mean {dv.mean():.4f} max {dv.max():.4g}  TV mean {tv.mean():.4f} "
              f"max {tv.max():.4f}; argmax changes {am.mean():.1%}; "
              f"played move changes (shared draw) mean {crn.mean():.1%}")
    either = np.array([max(r["crn0"], r["crn1"]) for r in done])
    print(f"either side's played move changes (max of the two), mean {either.mean():.1%}")


if __name__ == "__main__":
    main()
