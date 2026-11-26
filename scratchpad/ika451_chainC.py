"""IKA-451: presolve both x2 leaves on pool120 (7,260 pairs each), null (40 pairs), SPRT(0,10) on pool120. No heavy.py."""
import os
import subprocess
import sys
import time
from pathlib import Path

H = Path("C:/tmp/ika451")
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = M / ".claude/worktrees/pokeuraou-priority-tasks-27bb3d"
PY = str(M / ".venv/Scripts/python.exe")
env = dict(os.environ, IKA438_ROOT=str(WT), PYTHONPATH=str(WT / "src"), POOL=str(H / "pool120.json"),
           POKEURAOU_RUST_NODE_BIN="C:/tmp/ika444/exe/pokeuraou-damage.exe")


def say(t):
    with open(H / "chainC.log", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {t}\n")


def run(cmd, log):
    t = time.time()
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - t


LEAVES = {
    "value-mc7wx2": [H / "models/value-mc7w.pt", H / "models/value-mc7w-s1.pt"],
    "value-mc6x2": [M / "data/models/value-mc6.pt", M / "data/models/value-mc6-s1.pt"],
}
for leaf, models in LEAVES.items():
    store = H / f"store120-{leaf.removeprefix('value-')}"
    if len(list(store.glob("*.json"))) < 7260:
        rc, s = run([PY, str(H / "presolve.py"), leaf, str(store), "14", *map(str, models)], H / f"presolve120-{leaf}.out")
        say(f"presolve {leaf} rc={rc} {s:.0f}s files={len(list(store.glob('*.json')))}")
        if rc:
            sys.exit(rc)
for m in ["null120", "mc7w-vs-mc6-120"]:
    if (H / "matches" / m).exists():
        say(f"{m} exists")
        continue
    rc, s = run([PY, str(H / "board120.py"), m, "14", "1"], H / f"chain-{m}.out")
    say(f"{m} rc={rc} {s:.0f}s")
    if rc:
        sys.exit(rc)
say("chainC end")
