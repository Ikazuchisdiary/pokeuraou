"""IKA-139: does value-gen11L give a leaf the same bits whatever batch it is scored in?

With the fix the fast path and `_per_completion` agree to the bit through two row-wise stand-
in leaves on 10,602 matrices, but through the shipping net they differ in the 1e-9 place on
8-15% of positions. The two paths score the same rows in different batches (the true
position's leaves patched, against each completion's own), so this scores one node's leaves
once whole and once in slices and counts rows whose value is not the same float.

    PYTHONPATH=<tree>/src POKEURAOU_RUST_NODE=1 \\
        python scratchpad/ika139_net_batch_bits.py <model.pt>
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


def main() -> None:
    import torch

    from pokeuraou import rustnode
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.narrow import narrow
    from pokeuraou.regulation import load_regulation
    from pokeuraou.resolve import Budget
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import load_roster
    from pokeuraou.value import BatchedValue, load_model

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    encoder = Encoder(reg)
    net, _meta = load_model(str(args.model), encoder)
    leaf = BatchedValue(net, encoder, device=torch.device("cpu"))
    sheet = list(load_roster("rizabanadohido").sets)[:6]
    pos = position_from_sets(reg, sheet[:4], sheet[2:6])
    ours = narrow(reg, pos, 0, limit=10).actions
    theirs = narrow(reg, pos, 1, limit=10).actions
    filled = rustnode.node_for(reg).fill_encoded(pos, ours, theirs, Budget.matrix())
    encoded = filled.encoded
    whole = np.asarray(leaf.from_encoded(encoded), dtype=np.float64)
    n = len(encoded)
    for size in (1, 7, 64, n // 3):
        parts = [
            np.asarray(leaf.from_encoded(encoded.slice(np.arange(a, min(a + size, n)))))
            for a in range(0, n, size)
        ]
        sliced = np.concatenate(parts).astype(np.float64)
        off = whole != sliced
        print(f"{n} leaves, batches of {size}: {int(off.sum())} rows differ, "
              f"max {float(np.abs(whole - sliced).max()):.3g}")


if __name__ == "__main__":
    main()
