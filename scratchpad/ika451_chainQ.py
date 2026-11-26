"""IKA-451 production chain (from IKA-438's): presolve value-mc7x2 (boards) and value-mc7 x1 (generation) on
regmc-matchupweb, gen-7w Q index, teaching fill (value-mc7 x2, positions 0-100,000), features, Q training
(30 epochs, one call), null (100 pairs), SPRT(-10,0) against q-mc6. No heavy.py (user 11/26); the training
still takes the shared GPU lock. A step whose marker exists is skipped. Log: C:/tmp/ika451/chainQ.log
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

H = Path("C:/tmp/ika451")
Q = H / "q"
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = M / ".claude/worktrees/pokeuraou-priority-tasks-27bb3d"
PY = str(M / ".venv/Scripts/python.exe")
LOCK = Path("C:/tmp/pokeuraou-machine/train.lock")
env = dict(os.environ, IKA438_ROOT=str(WT), PYTHONPATH=str(WT / "src"), POOL="regmc-matchupweb",
           HEAVY_CORES="14", POKEURAOU_RUST_NODE_BIN="C:/tmp/ika444/exe/pokeuraou-damage.exe")
Q.mkdir(exist_ok=True)


def say(t):
    with open(H / "chainQ.log", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {t}\n")


def run(marker, why, cmd, lock=False):
    if (Q / marker).exists():
        return
    while lock:
        try:
            LOCK.mkdir()
            (LOCK / "pid").write_bytes(f"{os.getpid()}\n".encode())
            break
        except FileExistsError:
            time.sleep(60)
    t = time.time()
    try:
        with open(Q / f"step-{marker}.out", "w", encoding="utf-8", newline="\n") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    finally:
        if lock:
            shutil.rmtree(LOCK, ignore_errors=True)
    say(f"{why} rc={rc} {time.time() - t:.0f}s")
    if rc:
        sys.exit(rc)
    (Q / marker).write_bytes(b"")


V0, V1 = str(M / "data/models/value-mc7.pt"), str(M / "data/models/value-mc7-s1.pt")
run("presolve-x2.done", "presolve value-mc7x2 selections",
    [PY, str(H / "presolve.py"), "value-mc7x2", str(M / "data/selection-solved/value-mc7x2"), "14", V0, V1])
run("presolve-x1.done", "presolve value-mc7 x1 selections (generation store)",
    [PY, str(H / "presolve.py"), "value-mc7", str(M / "data/selection-solved/value-mc7"), "14", V0])
run("index.done", "Q index on gen-7w", [PY, str(H / "qrun.py"), "index"])
run("teach-smoke.done", "Q teach smoke [0, 100)", [PY, str(H / "qrun.py"), "teach", "0", "100"])
for a, b in ((100, 25000), (25000, 50000), (50000, 75000), (75000, 100000)):
    run(f"teach-{a}-{b}.done", f"Q teach [{a}, {b}) with value-mc7 x2", [PY, str(H / "qrun.py"), "teach", str(a), str(b)])
run("features.done", "Q features", [PY, str(H / "qrun.py"), "features"])
run("train.done", "Q train 30 epochs", [PY, str(H / "qrun.py"), "train", "30"], lock=True)
if not (Q / "q-mc7.pt").exists():
    say("no q-mc7.pt after training")
    sys.exit(1)
run("null.done", "Q board null (100 pairs)", [PY, str(H / "qrun.py"), "board", "null"])
run("board.done", "Q board SPRT(-10,0)", [PY, str(H / "qrun.py"), "board", "qnew-vs-qold"])
say("chainQ end")
