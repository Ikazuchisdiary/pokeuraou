"""IKA-412: value-mc4's recipe (IKA-409) with --swap-slots, from value-mc3 seeds 0 and 1.

    train.py [seeds]

Seed k starts from value-mc1 seed k (0 <- value-mc2.pt, 1 <- value-mc2-s1.pt).
Models: C:/tmp/ika409/models/value-mc3-s<seed>.pt. Samples nvidia-smi memory every 2 s beside it.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path("C:/tmp/ika412")
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = HERE / "wt"
PY = str(M / ".venv/Scripts/python.exe")
DATA = M / "data/selfplay-mc01234-encoded.npz"
INIT = {0: M / "data/models/value-mc3.pt", 1: M / "data/models/value-mc3-s1.pt"}
env = dict(os.environ, PYTHONPATH=str(WT / "src"))

seeds = [int(x) for x in sys.argv[1].split(",")]
EPOCHS = sys.argv[2]  # 2 or 4
TAG = "value-mc4swap"
(HERE / "models").mkdir(exist_ok=True)


def used() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=False).stdout.strip()
    return int(out or 0)


for seed in seeds:
    out = HERE / "models" / f"{TAG}-s{seed}.pt"
    if out.exists():
        sys.exit(f"{out} exists")
    cmd = [PY, str(WT / "tools/train_value.py"), "--data", str(DATA),
           "--init-from", str(INIT[seed]), "--epochs", EPOCHS, "--lr", "5e-4",
           "--keep", "last", "--swap-slots", "--split-seed", "0", "--holdout", "0.15", "--seed", str(seed), "--out", str(out)]
    before = used()
    peak = [before]
    stop = threading.Event()

    def sample() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], used())
            stop.wait(2)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t = time.time()
    with (HERE / "models" / f"{TAG}-s{seed}.log").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        subprocess.run(cmd, check=True, cwd=str(WT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    stop.set()
    th.join()
    print(f"seed {seed}: {time.time() - t:.0f}s, GPU (whole card) before {before} MB, peak {peak[0]} MB", flush=True)
