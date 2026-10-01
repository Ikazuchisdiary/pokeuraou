"""IKA-413 matches: the spread-move fix (new port) against the port before it, through
tools/match_queue.py --pool.

    board.py <name> [<name> ...]          run the named matches in order

Every match: pool regmc-matchupweb, pairs with a holder (floetteeternal, maushold, altaria:
714 of 2,145 pairs), served 2 servers / 24 workers, hidden bench, width 12, rank-leaf,
q-nocover with q-mc4, both arms value-mc4 x2, selection read from IKA-409's presolved store.
The game itself is resolved by the process's own port (the new exe); the "old" arm's move-node
search is read through the exe before the fix (--baseline-rust-binary).
Records: C:/tmp/ika413/matches/<name>/.

  smoke      4 games per seat, 1 server / 4 workers
  aa         both arms plain (new port)                      -- A/A, and the change share's reference
  ab         tested plain (new) vs baseline through the old exe, same seed as aa -- change share
  sprt       the same, SPRT(-10, 0), other seed
  fixed      the same, fixed count, other seed
"""

import os
import subprocess
import sys
import time
from pathlib import Path

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
HERE = Path("C:/tmp/ika413")
TREE = HERE / "wt2"
PY = str(M / ".venv/Scripts/python.exe")
POOL = str(M / "data/pool/regmc-matchupweb.json")
Q = str(M / "data/models/q-mc4.pt")
FILES = [str(M / "data/models/value-mc4.pt"), str(M / "data/models/value-mc4-s1.pt")]
STORE = "C:/tmp/ika409/store-mc4x2"
NEW_EXE = str(TREE / "rust/target/release/pokeuraou-damage.exe")
OLD_EXE = str(HERE / "old.exe")
HOLDERS = ["--pairs-with", "floetteeternal", "maushold", "altaria"]
OLD = ["--baseline-rust-binary", OLD_EXE]

# name -> (seed, games per seat, sprt, extra worker flags, servers, workers)
MATCHES = {
    "smoke": (41300, 4, None, OLD, 1, 4),
    "aa": (41301, 200, None, [], 2, 24),
    "ab": (41301, 200, None, OLD, 2, 24),
    "sprt": (41302, 2500, ("-10", "0"), OLD, 2, 24),
    "fixed": (41303, 2000, None, OLD, 2, 24),
}


def run(name: str) -> int:
    seed, games, sprt, extra, servers, workers = MATCHES[name]
    out = HERE / "matches" / name
    if out.exists():
        print(f"{out} exists", flush=True)
        return 1
    cmd = [PY, str(TREE / "tools/match_queue.py"), "--pool", POOL, "--out", str(out),
           "--games", str(games), "--seed", str(seed), "--served", "--servers", str(servers),
           "--workers", str(workers), "--hide-bench", *(["--sprt", *sprt] if sprt else []),
           "--q-model", Q, "--value", *FILES, "--baseline", *FILES,
           "--", "--limit", "12", "--rank-leaf", "--rank-fill", "q-nocover", "--selection-store", STORE,
           "--baseline-limit", "12", "--baseline-rank-leaf", "--baseline-rank-fill", "q-nocover",
           *HOLDERS, *extra]
    log = HERE / "matches" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(TREE / "src"), "POKEURAOU_RUST_NODE_BIN": NEW_EXE}
    started = time.time()
    with log.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"== {name} {time.strftime('%F %T')}\n{' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(TREE), stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        fh.write(f"== end {time.strftime('%F %T')} rc={rc} after {time.time() - started:.0f}s\n")
    print(name, "rc", rc, f"{time.time() - started:.0f}s", flush=True)
    return rc


for name in sys.argv[1:]:
    if run(name) != 0:
        sys.exit(1)
