"""IKA-450 stage 2: the top 10 candidates and the IKA-443 winner against the 65 pool teams.

    stage2.py <name> <games> <seed>

match_queue on pool-stage2.json (65 + 11), pairs of a focus team against an original pool
team, no mirrors. Both arms value-mc6 x2 (the shipped leaf), q-nocover with q-mc6, width 12,
rank-leaf, hidden bench, one selection store, selection epsilon 0 (a board).
"""
import json, os, subprocess, sys, time
from pathlib import Path
M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
WT = M / ".claude/worktrees/pokeuraou-priority-tasks-27bb3d"
PY = str(M / ".venv/Scripts/python.exe")
HERE = Path("C:/tmp/ika450")
EXE = "C:/tmp/ika444/exe/pokeuraou-damage.exe"
V = [str(M / "data/models/value-mc6.pt"), str(M / "data/models/value-mc6-s1.pt")]
name, games, seed = sys.argv[1:4]
teams = json.load(open(HERE / "stage2-teams.json"))
pool = json.load(open(HERE / "pool-stage2.json", encoding="utf-8"))
field = [t["id"] for t in pool["teams"][:65]]
out = HERE / "matches" / name
if out.exists():
    sys.exit(f"{out} exists")
store = HERE / "store-stage2"
cmd = [PY, str(WT / "tools/match_queue.py"), "--pool", str(HERE / "pool-stage2.json"), "--out", str(out),
       "--games", games, "--seed", seed, "--served", "--servers", "2", "--workers", os.environ.get("HEAVY_CORES", "14"),
       "--hide-bench", "--q-model", str(M / "data/models/q-mc6.pt"), "--value", *V, "--baseline", *V,
       "--", "--limit", "12", "--rank-leaf", "--rank-fill", "q-nocover", "--selection-store", str(store),
       "--baseline-limit", "12", "--baseline-rank-leaf", "--baseline-rank-fill", "q-nocover",
       "--baseline-selection-store", str(store),
       "--pairs-focus", *(os.environ["FOCUS"].split(",") if os.environ.get("FOCUS") else teams["candidates"]), "--pairs-field", *field, "--pairs-no-mirror",
       "--explore-epsilon", "0", "--explore-temperature", "0.5"]
log = HERE / f"stage2-{name}.log"
with log.open("w", encoding="utf-8", newline="\n") as fh:
    fh.write(" ".join(cmd) + "\n"); fh.flush()
    t = time.time()
    rc = subprocess.run(cmd, cwd=str(WT), stdout=fh, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONPATH": str(WT / "src"), "POKEURAOU_RUST_NODE_BIN": EXE}).returncode
    fh.write(f"== end rc={rc} after {time.time() - t:.0f}s\n")
print(name, "rc", rc); sys.exit(rc)
