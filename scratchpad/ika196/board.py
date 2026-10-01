"""IKA-196 board: equilibrium selection, tested arm --eq-select X against lp.

    board.py <name>

Generation form (IKA-409's board.py): pool regmc-matchupweb, served 2 servers / 24 workers,
hidden bench, width 12, rank-leaf, q-nocover with q-mc4, value-mc4x2 on both arms, each
arm its own copy of the presolved selection store. Records: C:/tmp/ika196/matches/<name>.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = Path("C:/Users/Ikazuchi/repos/pokeuraou/.claude/worktrees/agent-acde882fb885a4aa0")
PY = str(M / ".venv/Scripts/python.exe")
HERE = Path("C:/tmp/ika196")
OUT = HERE / "matches"
POOL = str(M / "data/pool/regmc-matchupweb.json")
VALUE = [str(M / "data/models/value-mc4.pt"), str(M / "data/models/value-mc4-s1.pt")]
Q = str(M / "data/models/q-mc4.pt")

# name -> (tested label, seed, pairs, sprt)
MATCHES = {
    "aa": ("lp", 19600, 100, None),
    "qre0.2": ("qre0.2", 19601, 4000, ("0", "10")),
    "unif": ("unif", 19602, 4000, ("0", "10")),
    "ment0": ("ment0", 19603, 4000, ("0", "10")),
    "qre0.005": ("qre0.005", 19604, 4000, ("0", "10")),
    "qre0.02": ("qre0.02", 19605, 4000, ("0", "10")),
    # Fixed counts: the size of the exact selections' effect, after their SPRTs said H0.
    "unif-fix": ("unif", 19606, 4000, None),
    "ment0-fix": ("ment0", 19607, 4000, None),
    # Reruns: the first two stopped for failed workers (exit 0xC000070A).
    "unif-fix2": ("unif", 19606, 4000, None),
    "ment0-fix2": ("ment0", 19607, 4000, None),
}

name = sys.argv[1]
label, seed, games, sprt = MATCHES[name]
out = OUT / name
if out.exists():
    sys.exit(f"{out} exists")
cmd = [PY, str(WT / "tools/match_queue.py"), "--pool", POOL, "--out", str(out),
       "--games", str(games), "--seed", str(seed), "--served", "--servers", "2", "--workers", "24",
       "--hide-bench", *(["--sprt", *sprt] if sprt else []),
       "--q-model", Q, "--value", *VALUE, "--baseline", *VALUE,
       "--", "--limit", "12", "--rank-leaf", "--rank-fill", "q-nocover",
       "--selection-store", str(HERE / "store-mc4x2"),
       "--baseline-limit", "12", "--baseline-rank-leaf", "--baseline-rank-fill", "q-nocover",
       "--baseline-selection-store", str(HERE / "store-mc4x2-b"),
       "--eq-select", label]
q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, check=True).stdout.strip().split(",")
if int(q[1]) - int(q[0]) < 3 * 1024:
    sys.exit(f"GPU free {int(q[1]) - int(q[0])} MB < 3 GB: not starting")
OUT.mkdir(parents=True, exist_ok=True)
log = HERE / f"board-{name}.log"
gpu = (HERE / f"board-{name}-gpu.csv").open("w", encoding="utf-8", newline="\n")
with log.open("w", encoding="utf-8", newline="\n") as fh:
    fh.write(f"== {name} {time.strftime('%F %T')} label={label} gpu_before={q}\n")
    fh.write(" ".join(cmd) + "\n")
    fh.flush()
    started = time.time()
    proc = subprocess.Popen(cmd, cwd=str(WT), stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONPATH": str(WT / "src")})
    peak = 0
    while proc.poll() is None:
        used = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, check=False).stdout.strip() or 0)
        peak = max(peak, used)
        gpu.write(f"{time.time() - started:.0f},{used}\n")
        gpu.flush()
        if used > 11 * 1024:
            import psutil
            for k in psutil.Process(proc.pid).children(recursive=True):
                k.kill()
            proc.kill()
            fh.write(f"== STOPPED: GPU {used} MB > 11 GB\n")
            break
        time.sleep(10)
    rc = proc.wait()
    fh.write(f"== end {time.strftime('%F %T')} rc={rc} after {time.time() - started:.0f}s gpu_peak={peak}\n")
print(name, "rc", rc, f"{time.time() - started:.0f}s", "gpu peak", peak)
sys.exit(rc)
