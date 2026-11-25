import glob
import json
import re

rows = []
tot_w = tot_g = tot_c = 0
for p in sorted(glob.glob("C:/tmp/ika451/pools/B*-summary.json")):
    s = json.loads(open(p, encoding="utf-8").read())
    n = s["name"]
    gp = open(f"C:/tmp/ika451/pools/{n}-gpu.csv", encoding="utf-8").read().strip().splitlines()[1:]
    cpu = [int(r.split(",")[2]) for r in gp]
    gu = [int(r.split(",")[3]) for r in gp if r.split(",")[3].strip().isdigit()]
    solves = 0
    for lp in glob.glob(f"C:/tmp/ika451/pools/{n}/logs/worker*.log"):
        for m in re.finditer(r"selection: (\d+) solves", open(lp, encoding="utf-8", errors="replace").read()):
            solves += int(m.group(1))
    print(f"{n} {s['wall_s']:7.0f} {s['games_per_min']:7.0f} {s['cpu_s_per_game']:.3f} {s['written']:6d} gpu_max {s['gpu_peak_mb']} "
          f"cpu% {sum(cpu)/len(cpu):.0f} gpu% {sum(gu)/len(gu):.0f} solves {solves} cores {s['cores']} fail {s['workers_failed']} stopped {s['stopped']}")
    tot_w += s["wall_s"]
    tot_g += s["written"]
    tot_c += s["busy_cpu_s"]
print(f"total wall {tot_w:.0f}s ({tot_w/60:.1f} min) games {tot_g} mean {tot_g/tot_w*60:.0f}/min cpu/game {tot_c/tot_g:.3f}")
