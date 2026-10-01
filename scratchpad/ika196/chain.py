"""Run IKA-196's boards one after another, each through heavy.py (GPU, 14 cores)."""

import subprocess
import sys
import time

PY = "C:/Users/Ikazuchi/repos/pokeuraou/.venv/Scripts/python.exe"
HEAVY = "C:/tmp/pokeuraou-machine/heavy.py"

for name in sys.argv[1:]:
    started = time.strftime("%F %T")
    rc = subprocess.run(
        [PY, HEAVY, "--agent", "ika-196", "--gpu", "--cores", "14", "--est-min", "15",
         "--why", f"IKA-196 board {name}: eq-select vs lp, SPRT(0,10), generation form",
         "--", PY, "C:/tmp/ika196/board.py", name],
        stdout=open(f"C:/tmp/ika196/chain-{name}.out", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT, check=False,
    ).returncode
    print(f"{started} -> {time.strftime('%F %T')} {name} rc={rc}", flush=True)
