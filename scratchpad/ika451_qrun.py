"""IKA-451 (from IKA-438's qrun.py): one watched Q step (GPU memory logged every 10 s, the tree
stopped above 11 GB).

    qrun.py index                  q_teach index on gen-7w's games, seed 451
    qrun.py teach <start> <stop>   q_teach fill with value-mc7 x2 into C:/tmp/ika451/qteach/shards
    qrun.py teach-old <start> <stop>   the drift record's old side: value-mc6 x2 into shards-old
    qrun.py features               q_teach features for every shard
    qrun.py train <stop-after>     q_train, 30 epochs, state+bind trunk from value-mc7, resumable
    qrun.py board <name>           a board on value-mc7 x2 (see BOARDS)
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path("C:/tmp/ika451/q")
TEACH = Path("C:/tmp/ika451/qteach")
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = M / ".claude/worktrees/pokeuraou-priority-tasks-27bb3d"
PY = str(M / ".venv/Scripts/python.exe")
EXE = "C:/tmp/ika444/exe/pokeuraou-damage.exe"
LEAF = [str(M / "data/models/value-mc7.pt"), str(M / "data/models/value-mc7-s1.pt")]
OLDLEAF = [str(M / "data/models/value-mc6.pt"), str(M / "data/models/value-mc6-s1.pt")]
INDEX = TEACH / "index.npz"  # gen-7w's positions, seed 451
GAMES = str(M / "data/selfplay-mc7w")
STORE = str(M / "data/selection-solved/value-mc7x2")
QNEW = HERE / "q-mc7.pt"
QOLD = M / "data/models/q-mc6.pt"
POOL120 = "C:/tmp/ika451/pool120.json"  # gen-7w was played on it

# name -> (seed, pairs, sprt, the tested arm's named Q file, servers)
BOARDS = {
    "null": (45170, 100, False, QOLD, 1),
    "qnew-vs-qold": (45171, 4000, True, QNEW, 1),
}


def gpu() -> tuple[int, int]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=False).stdout.strip().split(",")
    return int(q[0]), int(q[1])


def command(kind: str, rest: list[str]) -> tuple[list[str], str]:
    jobs = os.environ.get("HEAVY_CORES", "14")
    if kind == "index":
        return [PY, str(WT / "tools/q_teach.py"), "index", "--games-dir", GAMES, "--out", str(INDEX),
                "--seed", "451"], "index"
    if kind == "teach":
        start, stop = rest
        return [PY, str(WT / "tools/q_teach.py"), "fill", "--index", str(INDEX),
                "--games-dir", GAMES, "--out", str(TEACH / "shards"), "--start", start, "--stop", stop,
                "--jobs", jobs, "--served", "2", "--pool", POOL120, "--value", *LEAF], f"teach-{start}-{stop}"
    if kind == "teach-old":
        start, stop = rest
        return [PY, str(WT / "tools/q_teach.py"), "fill", "--index", str(INDEX),
                "--games-dir", GAMES, "--out", str(TEACH / "shards-old"), "--start", start, "--stop", stop,
                "--jobs", jobs, "--served", "2", "--value", *OLDLEAF], f"teach-old-{start}-{stop}"
    if kind == "features":
        return [PY, str(WT / "tools/q_teach.py"), "features", "--index", str(INDEX),
                "--games-dir", GAMES, "--out", str(TEACH / "shards"), "--jobs", jobs, "--pool", POOL120], "features"
    if kind == "train":
        (stop_after,) = rest
        return [PY, str(WT / "tools/q_train.py"), "--shards", str(TEACH / "shards"), "--out", str(QNEW),
                "--epochs", "30", "--batch", "32", "--lr", "0.001", "--weight-decay", "0.01",
                "--holdout", "0.1", "--seed", "0", "--split-seed", "0", "--eval-limit", "3000",
                "--properties", "--state-inputs", "--bind-inputs", "--init-trunk", LEAF[0], "--device", "cuda",
                "--checkpoint", str(HERE / "q-mc7.ckpt"), "--stop-after", stop_after], f"train-{int(time.time())}"
    if kind == "board":
        (name,) = rest
        seed, pairs, sprt, qfile, servers = BOARDS[name]
        out = HERE / "matches" / name
        if out.exists():
            sys.exit(f"{out} exists")
        return [PY, str(WT / "tools/match_queue.py"), "--pool", "regmc-matchupweb", "--out", str(out),
                "--games", str(pairs), "--seed", str(seed), "--served", "--servers", str(servers), "--workers", jobs,
                "--hide-bench", *(["--sprt", "-10", "0"] if sprt else []),
                "--q-model", str(QOLD), "--q-model-named", "new", str(qfile),
                "--value", *LEAF, "--baseline", *LEAF,
                "--", "--baseline-inference-arm", "value", "--rank-leaf", "--baseline-rank-leaf",
                "--limit", "12", "--baseline-limit", "12",
                "--rank-fill", "q-nocover.new", "--baseline-rank-fill", "q-nocover",
                "--selection-store", STORE, "--baseline-selection-store", STORE], f"board-{name}"
    sys.exit(f"unknown {kind}")


cmd, name = command(sys.argv[1], sys.argv[2:])
used, total = gpu()
if total - used < 3 * 1024:
    sys.exit(f"GPU free {total - used} MB < 3 GB: not starting")
HERE.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONPATH=str(WT / "src"), POKEURAOU_RUST_NODE_BIN=EXE)
log = (HERE / f"{name}.log").open("w", encoding="utf-8", newline="\n")
csv = (HERE / f"{name}-gpu.csv").open("w", encoding="utf-8", newline="\n")
log.write(f"== {name} {time.strftime('%F %T')} gpu_before={used}\n{' '.join(cmd)}\n")
log.flush()
t0 = time.time()
proc = subprocess.Popen(cmd, cwd=str(WT), env=env, stdout=log, stderr=subprocess.STDOUT)
peak, stopped = used, ""
while proc.poll() is None:
    used, _ = gpu()
    peak = max(peak, used)
    csv.write(f"{time.time() - t0:.0f},{used}\n")
    csv.flush()
    if used > 11 * 1024:
        stopped = f"GPU {used} MB > 11 GB"
        for kid in psutil.Process(proc.pid).children(recursive=True):
            kid.kill()
        proc.kill()
        break
    time.sleep(10)
rc = proc.wait()
log.write(f"== end {time.strftime('%F %T')} rc={rc} after {time.time() - t0:.0f}s gpu_peak={peak} {stopped}\n")
log.close()
print(name, "rc", rc, f"{time.time() - t0:.0f}s", "gpu peak", peak, stopped, flush=True)
sys.exit(rc if not stopped else 3)
