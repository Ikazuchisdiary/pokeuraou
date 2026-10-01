"""IKA-410: self-play from a recorded d1 position (both sides' four seen: the open game), with the
generation's leaf (value-mc2), width and Q ranking, as C:/tmp/spreads/rollout.py.

    python rollout2.py <picks.jsonl> <games.jsonl> <games per position> <shard> <shards> <out.jsonl>

Each pick names (file, line, d1, seat). Writes the seat's win rate there.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from pokeuraou import qrank, rustnode
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.pool import load_pool
from pokeuraou.position import Position
from pokeuraou.selfplay import play_game
from pokeuraou.value import BatchedValue, load_model

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
picks = [json.loads(x) for x in Path(sys.argv[1]).read_bytes().splitlines() if x.strip()]
raw = {}
for ln in Path(sys.argv[2]).read_bytes().splitlines():
    if ln.strip():
        key, body = ln.split(b"\t", 1)
        k = json.loads(key)
        raw[(k["file"], k["line"])] = body
games, shard, shards, out = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6])
torch.set_num_threads(1)
pool = load_pool(str(M / "data/pool/regmc-matchupweb.json"))
reg = pool.reg
register_mega_stones(reg)
encoder = Encoder(reg)
net, _ = load_model(M / "data/models/value-mc2.pt", encoder)
leaf = BatchedValue(net.to("cpu"), encoder, device=torch.device("cpu"))
qrank.install(qrank.LocalQ(M / "data/models/q-mc2.pt", encoder, device="cpu"), "")
rustnode.hold_positions()
with out.open("ab") as f:
    for n, pk in enumerate(picks):
        if n % shards != shard:
            continue
        g = json.loads(raw[(pk["file"], pk["line"])])
        start = Position.from_json(g["decisions"][pk["d1"]]["position"])
        lv = float(leaf([start])[0])
        wins = []
        for i in range(games):
            rec = play_game(reg, np.random.default_rng([410, n, i]), [], [], "rollout", evaluate=leaf,
                            search_limit=12, start=start, open_information=True, rank_by_leaf=True,
                            rank_fill="q-nocover")
            if i == 0:
                first = float(rec.decisions[0].search_value)
                first_seat = first if pk["seat"] == 0 else 1.0 - first
            if rec.outcome is not None:
                wins.append(rec.outcome if pk["seat"] == 0 else 1.0 - rec.outcome)
        w = np.array(wins)
        res = {**pk, "games": len(w), "win": float(w.mean()), "ci95": float(1.96 * w.std(ddof=1) / np.sqrt(len(w))),
               "leaf_start": lv if pk["seat"] == 0 else 1 - lv, "first_search": first_seat}
        f.write(json.dumps(res).encode() + b"\n")
        f.flush()
        print(res, file=sys.stderr, flush=True)
