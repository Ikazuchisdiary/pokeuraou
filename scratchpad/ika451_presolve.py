"""IKA-451 (pool from POOL env); was IKA-427 (copy of IKA-425 presolve.py; ROOT from IKA438_ROOT): solve every pair's selection with a leaf into one store before the boards (IKA-292's presolve).

    presolve.py <leaf name> <store> <workers> <model> [<model> ...]            (launcher: 2 servers)
    presolve.py <leaf name> <store> <workers> --worker <k> <addr>             (worker k)

The store's tag is the one pool_match.py writes (`<pool sha>|value:<leaf name>`).
"""

import os
import subprocess
import sys
from pathlib import Path

W = Path(__file__).resolve().parent
ROOT = Path(os.environ["IKA438_ROOT"])
sys.path.insert(0, str(ROOT / "src"))

leaf, store, workers = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])

if sys.argv[4] != "--worker":
    import json

    from pokeuraou.pool import find_pool

    models = sys.argv[4:]
    store.mkdir(parents=True, exist_ok=True)
    reg = json.loads(find_pool(os.environ["POOL"]).read_bytes().decode("utf-8"))["regulation"]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), POKEURAOU_RUST_NODE="1")
    servers, addrs = [], []
    for i in range(2):
        p = subprocess.Popen(
            [sys.executable, str(ROOT / "tools/inference_server.py"), "--device", "cuda",
             "--regulation", reg, "--arm", "value", *models],
            env=env, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=open(W / f"presolve-{leaf}-server{i}.log", "w"), text=True)
        servers.append(p)
        addrs.append(p.stdout.readline().strip())
    print("servers", addrs, flush=True)
    kids = [subprocess.Popen([sys.executable, __file__, leaf, str(store), str(workers), "--worker",
                              str(k), addrs[k % 2]], env=env, cwd=str(ROOT)) for k in range(workers)]
    rc = max(k.wait() for k in kids)
    for p in servers:
        p.terminate()
    print("files", len(list(store.glob("*.json"))), "rc", rc, flush=True)
    sys.exit(rc)

k, addr = int(sys.argv[5]), sys.argv[6]
from pokeuraou.damage import register_mega_stones  # noqa: E402
from pokeuraou.encode import Encoder  # noqa: E402
from pokeuraou.inference import RemoteValue  # noqa: E402
from pokeuraou.pool import load_pool  # noqa: E402
from pokeuraou.poolplay import SolvedSelections  # noqa: E402

pool = load_pool(os.environ["POOL"])
reg = pool.reg
register_mega_stones(reg)
enc = Encoder(reg)
value = RemoteValue(addr, "value", enc)
solver = SolvedSelections(reg, pool.teams, value, model=f"value:{leaf}", store=store,
                          tag=f"{pool.sha256}|value:{leaf}")
for i, (a, b) in enumerate(pool.pairs):
    if i % workers == k:
        solver.canonical(a, b)
print(f"worker {k}: solved {solver.solves}, loaded {solver.loaded}", flush=True)
