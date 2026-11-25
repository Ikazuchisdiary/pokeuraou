"""IKA-451 (from IKA-438 gen.py; 120-team pool): one generation chunk, watched.

    gen.py <name> <first game> <games> <limit minutes> [<seed>]

Runs tools/generate_queue.py (this worktree's) for games [first, first+games) into
C:/tmp/ika438/pools/<name>, logs nvidia-smi every 10 s to <name>-gpu.csv, takes the
machine's busy CPU seconds before and after, and kills the whole tree when the GPU
uses more than 11 GB or the chunk runs past <limit minutes>. Writes <name>-summary.json.
Workers = 1.5 x HEAVY_CORES (24 at 16 cores, as in IKA-409). The exe is the worktree's build,
copied to C:/tmp/ika438/exe and named through POKEURAOU_RUST_NODE_BIN.
"""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

HERE = Path("C:/tmp/ika451")
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = M / ".claude/worktrees/pokeuraou-priority-tasks-27bb3d"
PY = str(M / ".venv/Scripts/python.exe")
EXE = "C:/tmp/ika444/exe/pokeuraou-damage.exe"
STORE = HERE / "store-gen-mc6"
VALUE = M / "data/models/value-mc6.pt"
QFILE = M / "data/models/q-mc6.pt"
GPU_STOP_MB = 11 * 1024

name, first, games, limit_min = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
seed = sys.argv[5] if len(sys.argv) > 5 else "45107"
cores = int(os.environ.get("HEAVY_CORES", "16"))
workers = max(8, round(cores * 1.5))
out = HERE / "pools" / name
if out.exists() and any(out.glob("games-worker*.jsonl")):
    sys.exit(f"{out} already has games")
out.mkdir(parents=True, exist_ok=True)

cmd = [PY, str(WT / "tools/generate_queue.py"), "--out", str(out), "--games", str(games),
       "--first-game", str(first), "--seed", seed, "--served", "--servers", "2", "--workers", str(workers),
       "--limit", "12", "--value", str(VALUE),
       "--pool", str(HERE / "pool120.json"), "--hide-bench", "--q-model", str(QFILE),
       "--", "--rank-leaf", "--selection-store", str(STORE), "--rank-fill", "q-nocover"]


def gpu_used() -> tuple[int, int]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, check=False).stdout.strip().split(",")
    return int(q[0]), int(q[1])


def busy() -> float:
    t = psutil.cpu_times()
    return sum(t) - t.idle


used, total = gpu_used()
if total - used < 3 * 1024:
    sys.exit(f"GPU free {total - used} MB < 3 GB: not starting")

stop_reason: list[str] = []
peak = [used]
done = threading.Event()


def kill_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
        kids = parent.children(recursive=True)
        for k in kids:
            k.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


log = (out.parent / f"{name}.log").open("w", encoding="utf-8", newline="\n")
log.write(" ".join(cmd) + f"\ncores={cores} workers={workers}\n")
log.flush()
env = dict(os.environ, PYTHONPATH=str(WT / "src"), POKEURAOU_RUST_NODE_BIN=EXE)
b0 = busy()
t0 = time.time()
proc = subprocess.Popen(cmd, cwd=str(WT), env=env, stdout=log, stderr=subprocess.STDOUT)


def watch() -> None:
    psutil.cpu_percent(None, percpu=True)
    with (out.parent / f"{name}-gpu.csv").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("t,used_mb,cpu_pct,gpu_util_pct,per_core_pct\n")
        while not done.is_set():
            u, _ = gpu_used()
            peak[0] = max(peak[0], u)
            gu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=False).stdout.strip()
            cores_pct = psutil.cpu_percent(None, percpu=True)
            fh.write(f"{time.time() - t0:.0f},{u},{sum(cores_pct) / len(cores_pct):.0f},{gu},"
                     f"{' '.join(f'{c:.0f}' for c in cores_pct)}\n")
            fh.flush()
            if u > GPU_STOP_MB:
                stop_reason.append(f"GPU {u} MB > 11 GB")
                kill_tree(proc)
                return
            if time.time() - t0 > limit_min * 60:
                stop_reason.append(f"past {limit_min} min")
                kill_tree(proc)
                return
            done.wait(10)


th = threading.Thread(target=watch, daemon=True)
th.start()
rc = proc.wait()
wall = time.time() - t0
cpu = busy() - b0
done.set()
th.join()
written = sum(1 for p in out.glob("games-worker*.jsonl") for line in p.open(encoding="utf-8") if line.strip())
workers_info = {}
wj = out / "workers.json"
if wj.exists():
    workers_info = json.loads(wj.read_bytes().decode("utf-8"))
summary = {"name": name, "first": first, "games": games, "seed": seed, "cores": cores, "workers": workers,
           "rc": rc, "wall_s": wall,
           "busy_cpu_s": cpu, "written": written, "cpu_s_per_game": cpu / max(written, 1),
           "games_per_min": written / wall * 60, "gpu_peak_mb": peak[0], "stopped": stop_reason,
           "workers_failed": workers_info.get("failed") if isinstance(workers_info, dict) else None,
           "workers_oom": workers_info.get("outOfMemory") if isinstance(workers_info, dict) else None}
(out.parent / f"{name}-summary.json").write_bytes((json.dumps(summary, indent=1) + "\n").encode("utf-8"))
log.write(json.dumps(summary) + "\n")
log.close()
print(json.dumps(summary), flush=True)
sys.exit(rc or (3 if stop_reason else 0))
