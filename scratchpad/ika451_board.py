"""IKA-451 (from IKA-438 board.py): value-mc7w x2 against value-mc6 x2.

    board.py <name> <workers> <servers>

Pool regmc-matchupweb, served, hidden bench, width 12, rank-leaf, q-nocover with q-mc6 for both
arms, each arm's own presolved selection store, --baseline named. No heavy.py (user 11/26).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = M / ".claude/worktrees/pokeuraou-priority-tasks-27bb3d"
PY = str(M / ".venv/Scripts/python.exe")
HERE = Path("C:/tmp/ika451")
OUT = HERE / "matches"
EXE = "C:/tmp/ika444/exe/pokeuraou-damage.exe"
Q = M / "data/models/q-mc6.pt"
ARMS = {
    "new": ([HERE / "models/value-mc7w.pt", HERE / "models/value-mc7w-s1.pt"], HERE / "store-mc7wx2"),
    "old": ([M / "data/models/value-mc6.pt", M / "data/models/value-mc6-s1.pt"],
            M / "data/selection-solved/value-mc6x2"),
}
MATCHES = {
    "null": ("new", "new", 45150, 40, None),
    "mc7w-vs-mc6": ("new", "old", 45151, 4000, ("0", "10")),
}
name, workers, servers = sys.argv[1], sys.argv[2], sys.argv[3]
tested, other, seed, games, sprt = MATCHES[name]
out = OUT / name
if out.exists():
    sys.exit(f"{out} exists")
OUT.mkdir(parents=True, exist_ok=True)
tf, ts = ARMS[tested]
of, os_ = ARMS[other]
cmd = [PY, str(WT / "tools/match_queue.py"), "--pool", "regmc-matchupweb", "--out", str(out),
       "--games", str(games), "--seed", str(seed), "--served", "--servers", servers, "--workers", workers,
       "--hide-bench", *(["--sprt", *sprt] if sprt else []),
       "--q-model", str(Q), "--value", *map(str, tf), "--baseline", *map(str, of),
       "--", "--limit", "12", "--rank-leaf", "--rank-fill", "q-nocover", "--selection-store", str(ts),
       "--baseline-limit", "12", "--baseline-rank-leaf", "--baseline-rank-fill", "q-nocover",
       "--baseline-selection-store", str(os_)]
with (HERE / f"board-{name}.log").open("w", encoding="utf-8", newline="\n") as fh:
    fh.write(f"== {name} {time.strftime('%F %T')} tested={tested} other={other}\n" + " ".join(cmd) + "\n")
    fh.flush()
    t = time.time()
    rc = subprocess.run(cmd, cwd=str(WT), stdout=fh, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONPATH": str(WT / "src"), "POKEURAOU_RUST_NODE_BIN": EXE}).returncode
    fh.write(f"== end {time.strftime('%F %T')} rc={rc} after {time.time() - t:.0f}s\n")
print(name, "rc", rc, f"{time.time() - t:.0f}s", flush=True)
sys.exit(rc)
