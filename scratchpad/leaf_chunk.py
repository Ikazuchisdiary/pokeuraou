"""What one node's leaves cost when they are all alive at once, and when they are not.

IKA-27 is about the list in `batched_payoffs`: with the Rust bridge off, every leaf of
every cell is materialised before anything is scored, and a 24x24 depth-1 node took the
machine to 0.3 GB free. `tools/profile_stages.py analysis --no-bridge` measures the whole
analysis (two menus ranked by the leaf, a solve, a second objective); this measures the
one call the issue names, so the peak can be attributed rather than inferred.

One setting per process, because RSS does not come back down: Python's allocator keeps the
arenas a freed chunk leaves behind, so a second setting measured in the same process would
be reading the first one's high-water mark. The payoff matrix is saved, so the two runs can
be held against each other afterwards:

The case reads a recorded game under `data/`, which is gitignored and therefore absent
from a fresh worktree; run it from the main checkout, or link `data` in and take the
link out again afterwards -- a junction left inside a worktree is something `git clean
-fdx` would follow into the real models and games.

    $env:POKEURAOU_RUST_NODE=0
    uv run python scratchpad/leaf_chunk.py --limit 24 --chunk 0 --out data/analysis/lc-0.npz
    uv run python scratchpad/leaf_chunk.py --limit 24 --chunk 32768 --out data/analysis/lc-32k.npz
    uv run python scratchpad/leaf_chunk.py --compare data/analysis/lc-0.npz data/analysis/lc-32k.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def sample_memory(stop: threading.Event, every: float = 0.05) -> dict[str, float]:
    """Peak RSS of this process and the low-water mark of the machine's free memory.

    Both, because they are different claims. RSS is what this call is holding; free memory
    is what the rest of the machine has left, which is the number that made the issue
    urgent -- 0.3 GB of 31.1.
    """
    import psutil

    me = psutil.Process()
    peak = 0
    free_low = float("inf")
    while not stop.is_set():
        peak = max(peak, int(me.memory_info().rss))
        free_low = min(free_low, float(psutil.virtual_memory().available))
        stop.wait(every)
    return {"peak_rss": float(peak), "free_low": free_low}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="sash-ko")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=0, help="resolve.LEAF_CHUNK for this run")
    ap.add_argument("--value", type=Path, nargs="+",
                    default=[Path("data/models/value-gen11L.pt")])
    ap.add_argument("--device", default=None, help="cuda or cpu; default is what torch has")
    ap.add_argument("--format", default="gen9championsvgc2026regmb")
    ap.add_argument("--out", type=Path, default=None, help="write the payoff matrix here")
    ap.add_argument("--compare", type=Path, nargs=2, default=None,
                    help="read two saved matrices and report the difference, running nothing")
    args = ap.parse_args()

    if args.compare is not None:
        first, second = (np.load(p, allow_pickle=False) for p in args.compare)
        a = np.asarray(first["payoff"], dtype=np.float64)
        b = np.asarray(second["payoff"], dtype=np.float64)
        same = int(np.count_nonzero(a == b))
        print(f"  {args.compare[0].name}  vs  {args.compare[1].name}")
        print(f"  {same:,} of {a.size:,} cells identical to the bit")
        print(f"  largest difference {np.abs(a - b).max():.3e}")
        print(f"  leaves {int(first['leaves']):,} vs {int(second['leaves']):,}")
        # The matrix is not the answer; the equilibrium is. A difference in the last
        # places of a payoff is only worth reporting through the thing it is read for.
        from pokeuraou.equilibrium import solve

        one, two = solve(a), solve(b)
        print(f"  game value {one.value:.12f} vs {two.value:.12f} "
              f"(difference {abs(one.value - two.value):.3e})")
        for name, x, y in (
            ("row", one.row_strategy, two.row_strategy),
            ("column", one.col_strategy, two.col_strategy),
        ):
            moved = np.abs(np.asarray(x) - np.asarray(y)).max()
            print(f"  largest change in the {name} mixture {moved:.3e}")
        return

    from human_baseline import CASES, load_leaf, position_of

    from pokeuraou import resolve as resolve_module
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.narrow import narrow
    from pokeuraou.regulation import load_regulation
    from pokeuraou.resolve import Budget, batched_payoffs
    from pokeuraou.search import leaf_ranking

    resolve_module.LEAF_CHUNK = args.chunk

    case = next((c for c in CASES if c.name == args.case), None)
    if case is None:
        raise SystemExit(f"no case named {args.case!r}")
    reg = load_regulation(args.format)
    register_mega_stones(reg)
    pos, _decision = position_of(case)
    budget = Budget()
    encoder = Encoder(reg)
    leaf = load_leaf(list(args.value), encoder)
    if args.device is not None:
        import torch

        leaf.device = torch.device(args.device)
        leaf.nets = [n.to(leaf.device) for n in leaf.nets]
        leaf.net = leaf.nets[0]
        if len(leaf.nets) > 1:
            leaf._stack()  # noqa: SLF001 - the stacked path is the only one it has

    # The same menus the analyser fills, which is the node the issue measured. Ranking
    # them is itself resolver work and is deliberately outside the timed section below.
    ours = narrow(reg, pos, 0, limit=args.limit,
                  rank=leaf_ranking(reg, pos, 0, leaf, budget=budget)).actions
    theirs = narrow(reg, pos, 1, limit=args.limit,
                    rank=leaf_ranking(reg, pos, 1, leaf, budget=budget)).actions

    # Rows per call and the largest call, so "the leaves were released" is a measurement
    # rather than a claim about the shape of the code.
    calls: list[int] = []

    def watched(positions: list) -> np.ndarray:  # noqa: ANN001
        calls.append(len(positions))
        return leaf(positions)

    stop = threading.Event()
    result: dict[str, float] = {}
    watcher = threading.Thread(
        target=lambda: result.update(sample_memory(stop)), daemon=True
    )
    # What the process is already holding before a single leaf exists: torch, the CUDA
    # context, the regulation. Subtracted below, because the question is what the leaves
    # cost and this run pays about 1.3 GB for the rest of itself either way.
    import psutil

    base_rss = float(psutil.Process().memory_info().rss)
    bridge = os.environ.get("POKEURAOU_RUST_NODE", "unset")
    print(f"  case {case.name}, {len(ours)}x{len(theirs)}, chunk {args.chunk or 'off'}, "
          f"bridge {bridge}, device {leaf.device}")
    watcher.start()
    started = time.perf_counter()
    payoffs, _unmodelled, _exact = batched_payoffs(
        reg, pos, ours, theirs, [watched], budget=budget
    )
    elapsed = time.perf_counter() - started
    stop.set()
    watcher.join()

    leaves = sum(calls)
    held = result["peak_rss"] - base_rss
    print(f"  {elapsed:,.1f} s   peak RSS {result['peak_rss'] / 1e9:.2f} GB "
          f"({base_rss / 1e9:.2f} GB before the fill, {held / 1e9:+.2f} GB for it)   "
          f"free fell to {result['free_low'] / 1e9:.1f} GB")
    print(f"  {leaves:,} leaves over {len(calls)} evaluator calls; "
          f"largest {max(calls) if calls else 0:,}")
    print(f"  game value of the fill: {payoffs[0].mean():.6f} (mean cell)")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, payoff=payoffs[0], leaves=leaves, chunk=args.chunk,
                 calls=np.asarray(calls), seconds=elapsed,
                 peak_rss=result["peak_rss"])
        print(f"  saved to {args.out}")
    print(json.dumps({
        "chunk": args.chunk,
        "limit": args.limit,
        "seconds": round(elapsed, 2),
        "peak_rss_gb": round(result["peak_rss"] / 1e9, 3),
        "base_rss_gb": round(base_rss / 1e9, 3),
        # Against the largest call and not the total: what is alive at the peak is one
        # chunk, so dividing the peak by every leaf the node ever had would price the
        # chunked run's leaves at a sixth of the unchunked run's for holding fewer.
        "bytes_a_leaf": round(held / max(calls)) if calls else 0,
        "free_low_gb": round(result["free_low"] / 1e9, 2),
        "leaves": leaves,
        "calls": len(calls),
        "largest_call": max(calls) if calls else 0,
    }))


if __name__ == "__main__":
    main()
