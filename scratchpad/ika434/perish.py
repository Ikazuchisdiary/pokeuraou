"""IKA-434 stage 1b: perish features per side from positions.jsonl (same rows as sample.npz).
perish_on: live actives under Perish Song; perish_doom: of those, more than the live bench
can replace (they cannot all be switched out). Writes perish.npy (rows, 2, 2)."""
import json

import numpy as np

rows = []
with open("C:/tmp/ika434/positions.jsonl", "rb") as fh:
    for line in fh:
        pos = json.loads(line)
        per_side = []
        for side in pos["sides"]:
            mons = side["pokemon"]
            active = [i for i in side["active"] if i is not None]
            live_active = [mons[i] for i in active if not mons[i]["fainted"] and mons[i]["hp"] > 0]
            bench = [m for k, m in enumerate(mons[:4]) if k not in active and not m["fainted"] and m["hp"] > 0]
            on = sum(1 for m in live_active if any(v["id"] == "perishsong" for v in m.get("volatiles", [])))
            per_side.append([on, max(0, on - len(bench))])
        rows.append(per_side)
arr = np.array(rows, dtype=np.float32)
np.save("C:/tmp/ika434/perish.npy", arr)
print(arr.shape, (arr[:, :, 0] > 0).any(axis=1).mean(), (arr[:, :, 1] > 0).any(axis=1).mean())
