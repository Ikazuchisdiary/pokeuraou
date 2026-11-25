"""IKA-451: presolve value-mc7wx2 selections on regmc-matchupweb, null (40 pairs), SPRT(0,10) vs value-mc6x2.
No heavy.py (user 11/26). Log: chainB.log"""
import os
import subprocess
import sys
import time
from pathlib import Path

H = Path("C:/tmp/ika451")
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = M / ".claude/worktrees/pokeuraou-priority-tasks-27bb3d"
PY = str(M / ".venv/Scripts/python.exe")
env = dict(os.environ, IKA438_ROOT=str(WT), PYTHONPATH=str(WT / "src"), POOL="regmc-matchupweb",
           POKEURAOU_RUST_NODE_BIN="C:/tmp/ika444/exe/pokeuraou-damage.exe")


def say(t):
    with open(H / "chainB.log", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {t}\n")


def run(cmd, log):
    t = time.time()
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t


store = H / "store-mc7wx2"
if len(list(store.glob("*.json"))) < 2145:
    rc, s = run([PY, str(H / "presolve.py"), "value-mc7wx2", str(store), "14",
                 str(H / "models/value-mc7w.pt"), str(H / "models/value-mc7w-s1.pt")], H / "presolve-mc7wx2.out")
    say(f"presolve rc={rc} {s:.0f}s files={len(list(store.glob('*.json')))}")
    if rc:
        sys.exit(rc)
for m in (sys.argv[1:] or ["null", "mc7w-vs-mc6"]):
    if (H / "matches" / m).exists():
        say(f"{m} exists")
        continue
    rc, s = run([PY, str(H / "board.py"), m, "14", "1"], H / f"chain-{m}.out")
    say(f"{m} rc={rc} {s:.0f}s")
    if rc:
        sys.exit(rc)
say("chainB end")
