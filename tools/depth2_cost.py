"""What a *true* depth-2 decision costs, and what collapsing the chance axis takes off it.

Not the double-oracle refinement in `search.py`, which is what `depth=2` actually runs --
this is the honest thing the docstring there calls three orders of magnitude: every cell
of the root matrix gets the equilibrium value of the position it leads to.

Cost is linear in cells, so a sample measures it: time N cells and multiply by the matrix.
That extrapolation is the one assumption here and it is stated rather than hidden -- cells
differ in how many chance branches they produce, which is why the branch count is
reported beside the time.

Two arms, because the question this was written for is what rounding the chance away buys:

    all branches   every chance branch of the cell gets its own sub-game, weighted
    modal only     the likeliest branch stands for all of them

`--sub-limit` is the other lever: the sub-game's width. The shipped refinement uses 8,
and the difference between 8 and 24 here is the same breadth-squared that makes the root
matrix expensive.

One thing this cannot be read as: a claim about the damage rolls. `Budget.matrix()` pins
roll 8, so the branches counted here are accuracy, status checks, secondaries and Speed
ties -- the damage axis is already one branch before anything is rounded.

    uv run --group learn python tools/depth2_cost.py --cells 24 --sub-limit 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.equilibrium import EquilibriumError, solve  # noqa: E402
from pokeuraou.narrow import narrow  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.resolve import Budget, batched_payoff, resolve_turn  # noqa: E402

#: `search.DEFAULT_SUB_BRANCHES`: the chance branches a refined cell keeps.
SHIPPED_SUB_BRANCHES = 3


def load_positions(games_dir: Path, min_turn: int, seed: int) -> list[Position]:
    out: list[Position] = []
    for path in sorted(games_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for decision in record.get("decisions", ()):
                    if decision.get("kind") == "move" and int(decision.get("turn", 0)) >= min_turn:
                        out.append(Position.from_json(decision["position"]))
                if len(out) > 200:
                    break
        break
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", type=int, default=24, help="cells to time, then extrapolate")
    ap.add_argument("--limit", type=int, default=24, help="root width")
    ap.add_argument("--sub-limit", type=int, default=8, help="sub-game width")
    ap.add_argument("--games-dir", type=Path, default=Path("data/selfplay-gen11L"))
    ap.add_argument("--min-turn", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--value", type=Path, default=Path("data/models/value-gen11L.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--skip-node-pass",
        action="store_true",
        help="leave out the whole-node pass that counts distinct successors and how "
        "often the shipped sub-branch cap binds",
    )
    args = ap.parse_args()

    # The port is the production path and this is a cost measurement, so it stays on.
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")

    positions = load_positions(args.games_dir, args.min_turn, args.seed)
    if not positions:
        raise SystemExit(f"no recorded positions found under {args.games_dir}")
    reg = load_regulation(positions[0].format)
    register_mega_stones(reg)

    import torch

    from pokeuraou.encode import Encoder
    from pokeuraou.value import BatchedValue, load_model

    torch.set_num_threads(1)
    encoder = Encoder(reg)
    net, _meta = load_model(args.value, encoder)
    device = torch.device(args.device)
    leaf = BatchedValue(net.to(device), encoder, device=device)
    budget = Budget.matrix()

    position = next(
        p
        for p in positions
        if len(narrow(reg, p, 0, limit=args.limit).actions) >= 8
        and len(narrow(reg, p, 1, limit=args.limit).actions) >= 8
    )
    ours = list(narrow(reg, position, 0, limit=args.limit).actions)
    theirs = list(narrow(reg, position, 1, limit=args.limit).actions)
    cells_total = len(ours) * len(theirs)
    print(f"  root {len(ours)}x{len(theirs)} = {cells_total} cells, sub-game width {args.sub_limit}")

    started = time.perf_counter()
    payoff, _notes = batched_payoff(reg, position, ours, theirs, leaf, budget=budget)
    solve(payoff)
    root_seconds = time.perf_counter() - started
    print(f"  depth-1 decision on this node:  {root_seconds:.3f} s\n")

    rng = random.Random(7)
    sample = [(rng.randrange(len(ours)), rng.randrange(len(theirs))) for _ in range(args.cells)]

    for label, modal_only in (("all branches", False), ("modal only", True)):
        branches_seen = sub_games = 0
        started = time.perf_counter()
        for i, j in sample:
            result = resolve_turn(reg, position, [ours[i], theirs[j]], budget=budget)
            if result.suspended or not result.branches:
                continue
            picked = sorted(result.branches, key=lambda b: -b.probability)
            if modal_only:
                picked = picked[:1]
            branches_seen += len(picked)
            for branch in picked:
                sub_ours = list(narrow(reg, branch.position, 0, limit=args.sub_limit).actions)
                sub_theirs = list(narrow(reg, branch.position, 1, limit=args.sub_limit).actions)
                if not sub_ours or not sub_theirs:
                    continue
                sub_payoff, _n = batched_payoff(
                    reg, branch.position, sub_ours, sub_theirs, leaf, budget=budget
                )
                sub_games += 1
                # A subgame the LP refuses still cost what it cost; the timing is the
                # subject here, so it is counted and the refusal ignored.
                with contextlib.suppress(EquilibriumError):
                    solve(sub_payoff)
        seconds = time.perf_counter() - started
        per_cell = seconds / len(sample)
        whole = per_cell * cells_total
        print(
            f"  {label:12s}  {branches_seen / len(sample):4.2f} branches/cell  "
            f"{sub_games:4d} sub-games  {per_cell * 1000:6.1f} ms/cell  "
            f"-> whole node {whole:7.1f} s = {whole / root_seconds:5.0f}x depth 1"
        )

    print(
        "\n  the shipped depth=2 is not this: it refines the support only "
        f"(`DEFAULT_REFINE`=4 per side = 16 cells of {cells_total}), which is what makes it "
        "affordable.\n  Multiply the ms/cell above by 16 to get that, and by the matrix to "
        "get the honest two-ply search."
    )

    if args.skip_node_pass:
        return

    # ------------------------------------------------------------------
    # Two things the multiplication above gets wrong, both measured on the
    # whole node rather than on the sample.
    # ------------------------------------------------------------------
    print("\n  the whole node, resolved once (no sub-games):\n")
    for name, sub_budget in (
        ("merged", budget),
        ("unmerged", replace(budget, merge_duplicates=False)),
    ):
        seen: Counter[str] = Counter()
        branches = suspended = 0
        over_cap = 0
        kept_mass: list[float] = []
        for a in ours:
            for b in theirs:
                result = resolve_turn(reg, position, [a, b], budget=sub_budget)
                suspended += len(result.suspended)
                if not result.branches:
                    continue
                branches += len(result.branches)
                for branch in result.branches:
                    seen[json.dumps(branch.position.to_json(), sort_keys=True)] += 1
                weights = sorted((br.probability for br in result.branches), reverse=True)
                total = sum(weights)
                over_cap += int(len(weights) > SHIPPED_SUB_BRANCHES)
                if total > 0:
                    kept_mass.append(sum(weights[:SHIPPED_SUB_BRANCHES]) / total)
        distinct = len(seen)
        mass = np.array(kept_mass) if kept_mass else np.array([1.0])
        print(
            f"  {name:9s} {branches:6d} branches -> {distinct:5d} distinct "
            f"({1 - distinct / max(branches, 1):5.1%} of them are a successor another cell "
            f"already reaches, {branches / max(distinct, 1):.2f}x reuse)"
        )
        print(
            f"            the shipped `sub_branches={SHIPPED_SUB_BRANCHES}` cuts "
            f"{over_cap / cells_total:5.1%} of cells; the kept branches carry "
            f"{mass.mean():.4f} of the mass on average, {mass.min():.4f} at worst"
        )
    print(
        "\n  reuse is the ceiling on caching sub-games by position -- depth 1 already takes\n"
        "  it, in the port's leaf sharing. It is a factor, not an order of magnitude."
    )


if __name__ == "__main__":
    main()
