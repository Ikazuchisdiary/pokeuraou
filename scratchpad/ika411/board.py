"""IKA-411 matches: the fix against the pre-fix belief, through tools/match_queue.py --pool.

    board.py <name> [<name> ...]          run the named matches in order (one exclusive grant)

Every match: pool regmc-matchupweb, served 2 servers / 24 workers, hidden bench, width 12,
rank-leaf, q-nocover with q-mc3, both arms value-mc3 x2 (the files are IKA-405's
value-mc3e6-s0/s1, byte-identical to data/models/value-mc3.pt / -s1, so the leaf is named
value-mc3e6x2 and IKA-405's presolved store, copied to C:/tmp/ika411/store-mc3e6x2, is read).
Records: C:/tmp/ika411/matches/<name>/.

  id-old      origin/master (before the fix), both arms plain, all pairs     -- identity, old side
  id-newdex   this branch, both arms --dex-base-belief, all pairs            -- identity, new side
  chg-newdex  this branch, both arms --dex-base-belief, forme pairs          -- change share, before
  chg-new     this branch, both arms plain (the fix), forme pairs            -- change share, after; A/A
  main-sprt   fix vs pre-fix belief, forme pairs, SPRT(-10, 0)
  main-fixed  fix vs pre-fix belief, forme pairs, fixed count
"""

import os
import subprocess
import sys
import time
from pathlib import Path

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
HERE = Path("C:/tmp/ika411")
NEW, OLD = HERE / "wt", HERE / "old"
PY = str(M / ".venv/Scripts/python.exe")
POOL = str(M / "data/pool/regmc-matchupweb.json")
Q = str(M / "data/models/q-mc3.pt")
FILES = [str(HERE.parent / "ika405/models/value-mc3e6-s0.pt"), str(HERE.parent / "ika405/models/value-mc3e6-s1.pt")]
STORE = str(HERE / "store-mc3e6x2")
EXE = str(HERE.parent / "ika406/wt/rust/target/release/pokeuraou-damage.exe")
FORMES = ["--pairs-with", "floetteeternal", "meowsticf"]
DEX = ["--dex-base-belief"]
BDEX = ["--baseline-dex-base-belief"]

# name -> (checkout, seed, games per seat, sprt, extra worker flags)
MATCHES = {
    "id-old": (OLD, 41110, 200, None, []),
    "id-newdex": (NEW, 41110, 200, None, DEX + BDEX),
    "chg-newdex": (NEW, 41111, 200, None, FORMES + DEX + BDEX),
    "chg-new": (NEW, 41111, 200, None, FORMES),
    "main-sprt": (NEW, 41112, 4000, ("-10", "0"), FORMES + BDEX),
    "main-fixed": (NEW, 41113, 1500, None, FORMES + BDEX),
}


def run(name: str) -> int:
    tree, seed, games, sprt, extra = MATCHES[name]
    out = HERE / "matches" / name
    if out.exists():
        print(f"{out} exists", flush=True)
        return 1
    cmd = [PY, str(tree / "tools/match_queue.py"), "--pool", POOL, "--out", str(out),
           "--games", str(games), "--seed", str(seed), "--served", "--servers", "2", "--workers", "24",
           "--hide-bench", *(["--sprt", *sprt] if sprt else []),
           "--q-model", Q, "--value", *FILES, "--baseline", *FILES,
           "--", "--limit", "12", "--rank-leaf", "--rank-fill", "q-nocover", "--selection-store", STORE,
           "--baseline-limit", "12", "--baseline-rank-leaf", "--baseline-rank-fill", "q-nocover",
           *extra]
    log = HERE / "matches" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "POKEURAOU_RUST_NODE_BIN": EXE}
    started = time.time()
    with log.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"== {name} {time.strftime('%F %T')} tree={tree}\n{' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(tree), stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        fh.write(f"== end {time.strftime('%F %T')} rc={rc} after {time.time() - started:.0f}s\n")
    print(name, "rc", rc, f"{time.time() - started:.0f}s", flush=True)
    return rc


for name in sys.argv[1:]:
    if run(name) != 0:
        sys.exit(1)
