"""IKA-410: why replays differ from the record. Replays the named games and prints the first
decision that differs, field by field.

    python diag.py <examples.jsonl> <games.jsonl>
"""

import json
import sys
import tempfile
from pathlib import Path

import torch

from pokeuraou import poolplay, qrank, rustnode
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.pool import load_pool
from pokeuraou.value import BatchedValue, load_model

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
examples = [json.loads(x) for x in Path(sys.argv[1]).read_bytes().splitlines() if x.strip()]
raw = {}
for ln in Path(sys.argv[2]).read_bytes().splitlines():
    if ln.strip():
        key, body = ln.split(b"\t", 1)
        k = json.loads(key)
        raw[(k["file"], k["line"])] = body
torch.set_num_threads(1)
pool = load_pool(str(M / "data/pool/regmc-matchupweb.json"))
reg = pool.reg
register_mega_stones(reg)
encoder = Encoder(reg)
net, _ = load_model(M / "data/models/value-mc2.pt", encoder)
leaf = BatchedValue(net.to("cpu"), encoder, device=torch.device("cpu"))
qmodel = qrank.LocalQ(M / "data/models/q-mc2.pt", encoder, device="cpu")
qrank.install(qmodel, "")
q_record = qrank.record_fields({"": qmodel})
rustnode.hold_positions()
for ex in examples:
    orig = json.loads(raw[(ex["file"], ex["line"])])
    with tempfile.TemporaryDirectory(dir="C:/tmp/ika410/tmp") as tmp:
        p = Path(tmp) / "g.jsonl"
        poolplay.generate_pool(
            reg, pool, games=1, hide_bench=True, seed=40002, out=p, selection=poolplay.SOLVED,
            evaluate=leaf, memo=True, store=Path("C:/tmp/ika410/store-0"), leaf="value:value-mc2",
            search_limit=12, rank_by_leaf=True, rank_fill="q-nocover", indices=[orig["gameIndex"]],
            q_record=q_record,
        )
        rep = json.loads(p.read_bytes().splitlines()[0])
    print("game", orig["gameIndex"])
    for key in ("ownSix", "foeSix", "ownPick", "foePick", "leads", "turns", "outcome"):
        if orig.get(key) != rep.get(key):
            print("  top-level differs:", key, orig.get(key), rep.get(key))
    for n, (a, b) in enumerate(zip(orig["decisions"], rep["decisions"])):
        diffs = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
        if diffs:
            print("  decision", n, a["kind"], a["turn"])
            for k in sorted(diffs):
                print("   ", k, str(a.get(k))[:300], "|", str(b.get(k))[:300])
            break
