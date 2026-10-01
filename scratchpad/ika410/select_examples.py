"""IKA-410 stage 2 selection: 200 rows with dv < -0.3 and 200 rows uniformly from stage 1.

    python select_examples.py <stage1.jsonl> <out examples.jsonl> <out games.jsonl>

Writes the chosen rows (with "group": "drop" or "random") and every chosen game's
original record line (one line per distinct game, keyed by file and line).
"""

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path("C:/Users/Ikazuchi/repos/pokeuraou/data/selfplay-mc3")
rows = [json.loads(x) for x in Path(sys.argv[1]).read_bytes().splitlines() if x.strip()]
rng = np.random.default_rng(410)
drops = [r for r in rows if r["dv"] < -0.3]
pick_drop = rng.choice(len(drops), size=200, replace=False)
pick_rand = rng.choice(len(rows), size=200, replace=False)
chosen = [dict(drops[k], group="drop") for k in sorted(pick_drop)]
chosen += [dict(rows[k], group="random") for k in sorted(pick_rand)]
print(f"rows {len(rows)} drops {len(drops)} chosen {len(chosen)}", file=sys.stderr)

need: dict[str, set[int]] = {}
for r in chosen:
    need.setdefault(r["file"], set()).add(r["line"])
with Path(sys.argv[3]).open("wb") as f:
    for name, lines in sorted(need.items()):
        all_lines = [ln for ln in (SRC / name).read_bytes().split(b"\n") if ln.strip()]
        for k in sorted(lines):
            f.write(json.dumps({"file": name, "line": k}).encode() + b"\t" + all_lines[k] + b"\n")
Path(sys.argv[2]).write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in chosen))
print(f"games {sum(len(v) for v in need.values())}", file=sys.stderr)
