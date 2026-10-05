"""IKA-434 step 2: the picked games' rows of the gen-4 revision-4 shard, read a chunk at a
time, through each arm's nets (logit per member). CPU.

    python predict.py THREADS NAME=a.pt[,b.pt] [NAME=...]

Writes predict.npz: game, turn, kind, outcome of the rows (to check against sample.npz), the
shard's side and field columns of those rows, and logit_<NAME>#<k>.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-aa55f8431dd4ced5d/src")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.packed import NpzMember  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402
from pokeuraou.value import load_model  # noqa: E402

OUT = Path("C:/tmp/ika434")
SHARD = Path("C:/Users/Ikazuchi/repos/pokeuraou/data/selfplay-mc4-encoded.npz")
torch.set_num_threads(int(sys.argv[1]))
arms = {name: paths.split(",") for name, paths in (s.split("=", 1) for s in sys.argv[2:])}
started = time.time()

picked = np.load(OUT / "picked.npy")
with np.load(SHARD) as z:
    game, turn, kind, outcome = z["game"], z["turn"], z["kind"], z["outcome"]
take = np.isin(game, picked)
print(f"rows {int(take.sum()):,} of {len(game):,}", flush=True)

enc = Encoder(load_regulation("gen9championsvgc2026regmc"))
nets = {name: [load_model(p, enc)[0].eval() for p in paths] for name, paths in arms.items()}
logits = {f"{n}#{k}": [] for n, ps in arms.items() for k in range(len(ps))}
sides, fields = [], []
keys = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")
streams = [NpzMember(SHARD, k).chunks() for k in keys]
row = 0
with torch.no_grad():
    for parts in zip(*streams, strict=True):
        n = len(parts[0])
        pick = take[row: row + n]
        row += n
        if not pick.any():
            continue
        arr = {k: np.ascontiguousarray(p[pick]) for k, p in zip(keys, parts, strict=True)}
        sides.append(arr["side"].astype(np.float32))
        fields.append(arr["field"].astype(np.float32))
        batch = {k: torch.from_numpy(v.astype(np.int64) if v.dtype.kind in "iu" else v) for k, v in arr.items()}
        for name, members in nets.items():
            for k, net in enumerate(members):
                out = [net({kk: vv[s: s + 8192] for kk, vv in batch.items()}).float().numpy()
                       for s in range(0, len(arr["mon"]), 8192)]
                logits[f"{name}#{k}"].append(np.concatenate(out))
        print(f"  {row:,} rows read, {time.time() - started:.0f}s", flush=True)
assert row == len(game)
np.savez(OUT / "predict.npz", game=game[take], turn=turn[take], kind=kind[take], outcome=outcome[take],
         side=np.concatenate(sides), field=np.concatenate(fields),
         side_names=np.array(enc.side_names), field_names=np.array(enc.field_names),
         **{f"logit_{k}": np.concatenate(v).astype(np.float64) for k, v in logits.items()})
print(f"{time.time() - started:.0f}s", flush=True)
