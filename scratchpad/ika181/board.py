"""IKA-181 matches through tools/match_queue.py --pool (the generation form).

    board.py <name> [<name> ...]          run the named matches in order

Every match: pool regmc-matchupweb, served 2 servers / 24 workers, hidden bench, width 12,
rank-leaf, q-nocover with q-mc4, both arms value-mc4 x2 (IKA-409's files, byte-identical to
data/models/value-mc4.pt / -s1), IKA-409's presolved store copied to C:/tmp/ika181/store-mc4x2.
Records: C:/tmp/ika181/matches/<name>/.

  id-old     master's code (C:/tmp/ika181/master) + the port before the fix, both arms off,
             all pairs
  id-new     this branch + the fixed port, both arms off, all pairs  -- identity with id-old
  chg-ben    this branch + the fixed port, both arms benefit, all pairs -- what the mode changes
  main-sprt  benefit vs off, SPRT(0, 10), pairs --pairs-ally-benefit (either team has a
             member gaining from its ally's move)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
HERE = Path("C:/tmp/ika181")
NEW = M / ".claude/worktrees/agent-af06b00bddc87aa09"
OLD = HERE / "master"
PY = str(M / ".venv/Scripts/python.exe")
POOL = str(M / "data/pool/regmc-matchupweb.json")
Q = str(M / "data/models/q-mc4.pt")
FILES = ["C:/tmp/ika409/models/value-mc4-s0.pt", "C:/tmp/ika409/models/value-mc4-s1.pt"]
STORE = str(HERE / "store-mc4x2")
EXE_OLD = str(HERE / "bin/pokeuraou-damage.exe")
EXE_NEW = str(HERE / "bin-fixed/pokeuraou-damage.exe")
PAIRS = ["--pairs-ally-benefit"]

# name -> (tree, exe, seed, games per seat, sprt, extra worker flags)
MATCHES = {
    "id-old": (OLD, EXE_OLD, 18100, 100, None, []),
    "id-new": (NEW, EXE_NEW, 18100, 100, None, []),
    "chg-ben": (NEW, EXE_NEW, 18100, 100, None,
                ["--ally-targets", "benefit", "--baseline-ally-targets", "benefit"]),
    "main-fixed": (NEW, EXE_NEW, 18102, 1500, None, PAIRS + ["--ally-targets", "benefit"]),
    "smoke": (NEW, EXE_NEW, 18199, 4, None, PAIRS + ["--ally-targets", "benefit"]),
    "main-sprt": (NEW, EXE_NEW, 18101, 4000, ("0", "10"),
                  PAIRS + ["--ally-targets", "benefit"]),
}


def run(name: str) -> int:
    tree, exe, seed, games, sprt, extra = MATCHES[name]
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
           "--baseline-selection-store", STORE, *extra]
    log = HERE / "matches" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(tree / "src"), "POKEURAOU_RUST_NODE_BIN": exe}
    started = time.time()
    with log.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"== {name} {time.strftime('%F %T')} tree={tree} exe={exe}\n{' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=str(tree), stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        fh.write(f"== end {time.strftime('%F %T')} rc={rc} after {time.time() - started:.0f}s\n")
    print(name, "rc", rc, f"{time.time() - started:.0f}s", flush=True)
    return rc


for name in sys.argv[1:]:
    if run(name) != 0:
        sys.exit(1)
