"""IKA-451: encode gen-7w (revision 4, 16 parts), streamed join of gen-0..6 + 7w, then train two seeds
from nothing (CLAUDE.md's next-generation command), one at a time under the shared GPU lock."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

H = Path("C:/tmp/ika451")
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = M / ".claude/worktrees/pokeuraou-priority-tasks-27bb3d"
PY = str(M / ".venv/Scripts/python.exe")
HEAVY = "C:/tmp/pokeuraou-machine/heavy.py"
DATA = M / "data"
JOIN = "selfplay-mc01234567w-encoded.npz"
LOCK = Path("C:/tmp/pokeuraou-machine/train.lock")
env = dict(os.environ, PYTHONPATH=str(WT / "src"), POKEURAOU_RUST_NODE_BIN="C:/tmp/ika444/exe/pokeuraou-damage.exe")


def say(t):
    with open(H / "pipe.log", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {t}\n")


def gpu_used() -> int:
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=False).stdout.strip()
    return int(o or 0)


def run(why, heavy_args, cmd, log, cwd):
    while psutil.virtual_memory().available < 9 * 2**30:
        time.sleep(60)
    t = time.time()
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        rc = subprocess.run([PY, HEAVY, "--agent", "ika-451", *heavy_args, "--why", why, "--",
                             PY, str(H / "memrun.py"), "--", *cmd],
                            cwd=str(cwd), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    say(f"{why}: rc={rc} {time.time() - t:.0f}s")
    if rc:
        sys.exit(rc)


if not (DATA / "selfplay-mc7w-encoded.npz").exists():
    run("IKA-451 encode gen-7w at revision 4 (16 parts)", ["--cores", "4", "--est-min", "12"],
        [PY, str(WT / "tools/encode_dataset.py"), "--dir", "selfplay-mc7w", "--jobs", "4", "--parts", "16"],
        H / "encode-mc7w.log", DATA)
if not (DATA / JOIN).exists():
    run("IKA-451 streamed join of gen-0..6 and 7w", ["--cores", "1", "--est-min", "14"],
        [PY, str(WT / "tools/encode_dataset.py"), "--dir", *[f"selfplay-mc{k}" for k in range(7)], "selfplay-mc7w",
         "--out", JOIN], H / "encode-join.log", DATA)
for k in (0, 1):
    out = H / "models" / ("value-mc7w.pt" if k == 0 else f"value-mc7w-s{k}.pt")
    if out.exists():
        say(f"{out.name} exists, skipped")
        continue
    while True:
        if psutil.virtual_memory().available >= 9 * 2**30 and gpu_used() < 7 * 1024:
            try:
                LOCK.mkdir()
                (LOCK / "pid").write_bytes(f"{os.getpid()}\n".encode())
                break
            except FileExistsError:
                pass
        time.sleep(60)
    say(f"start {out.name}")
    try:
        run(f"IKA-451 train value-mc7w seed {10 + k}", ["--gpu", "--cores", "3", "--est-min", "90", "--wait-max", "720"],
            [PY, str(WT / "tools/train_value.py"), "--data", str(DATA / JOIN), "--state-inputs", "--bind-inputs",
             "--aux-weight", "5", "--aux-targets", "search:1,ahead2:5", "--epochs", "30", "--lr", "2e-3",
             "--keep", "best", "--split-seed", "0", "--holdout", "0.15", "--seed", str(10 + k), "--out", str(out)],
            H / "models" / f"{out.stem}.log", WT)
    finally:
        shutil.rmtree(LOCK, ignore_errors=True)
say("pipe end")
