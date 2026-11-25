"""IKA-451 chain: 20 blocks of 20,000 games (gen-7 on the 120-team pool), each through heavy.py (GPU, elastic cores).

A file C:/tmp/ika451/STOP (created by hand) stops the chain before the next block.
A finished block's games-workerK.jsonl move to data/selfplay-mc7w/games-bNN-workerK.jsonl.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

H = Path("C:/tmp/ika451")
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
PY = str(M / ".venv/Scripts/python.exe")
HEAVY = "C:/tmp/pokeuraou-machine/heavy.py"
DEST = M / "data/selfplay-mc7w"


def heavy(why, est, cmd, log):
    full = [PY, HEAVY, "--agent", "ika-451", "--gpu", "--cores", "14", "--cores-min", "10", "--est-min", str(est),
            "--why", why, "--", *cmd]
    with open(log, "w", encoding="utf-8", newline="\n") as fh:
        return subprocess.run(full, stdout=fh, stderr=subprocess.STDOUT).returncode


def say(text):
    with open(H / "chain.log", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {text}\n")


DEST.mkdir(parents=True, exist_ok=True)
first_block = int(sys.argv[1]) if len(sys.argv) > 1 else 0
for b in range(first_block, 20):
    if (H / "STOP").exists():
        say(f"STOP file seen before block {b}")
        break
    name = f"B{b:02d}"
    if (DEST / f"games-b{b:02d}-worker0.jsonl").exists():
        say(f"{name} exists, skipped")
        continue
    rc = heavy(f"IKA-451 gen-7w block {name} games {b * 20000}..{b * 20000 + 19999}", 18,
               [PY, str(H / "gen.py"), name, str(b * 20000), "20000", "45"], H / f"chain-{name}.out")
    say(f"{name} rc={rc}")
    if rc:
        break
    for p in sorted((H / "pools" / name).glob("games-worker*.jsonl")):
        shutil.move(str(p), str(DEST / f"games-b{b:02d}-{p.name[len('games-'):]}"))
    say(f"{name} moved")
say("chain end")
