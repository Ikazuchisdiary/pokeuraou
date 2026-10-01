"""IKA-196: per match, the SPRT outcome, pairs, and the selection's own counts and seconds."""

import json
import re
import sys
from pathlib import Path

ROOT = Path("C:/tmp/ika196/matches")
pat = re.compile(r"equilibrium (\S+): ([\d,]+) solves, ([\d,]+) moved off the LP's vertex, "
                 r"([\d,]+) fell back to it, ([\d.]+) s")
seat = re.compile(r"= side (\d): (\d+) games, won ([\d.]+)% \(([\d.]+) s/game")
for name in sys.argv[1:]:
    d = ROOT / name
    sprt = json.loads((d / "sprt.json").read_text()) if (d / "sprt.json").exists() else {}
    calls = moved = fall = 0
    secs = 0.0
    games = 0
    gsecs = 0.0
    for log in sorted((d / "logs").glob("worker*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        for m in pat.finditer(text):
            if m.group(1) == "lp":
                continue
            calls += int(m.group(2).replace(",", ""))
            moved += int(m.group(3).replace(",", ""))
            fall += int(m.group(4).replace(",", ""))
            secs += float(m.group(5))
        for m in seat.finditer(text):
            games += int(m.group(2))
            gsecs += int(m.group(2)) * float(m.group(4))
    counts = sprt.get("counts", {})
    print(f"{name}: SPRT {sprt.get('decision')} after {sprt.get('pairs')} pairs, LLR "
          f"{sprt.get('llr', float('nan')):+.3f}, counts {counts}; selection solves {calls:,}, "
          f"moved {moved:,} ({100 * moved / max(calls, 1):.1f}%), fallbacks {fall}, "
          f"{secs:.1f} s = {1000 * secs / max(calls, 1):.2f} ms a solve; games {games:,}, "
          f"{gsecs / max(games, 1):.2f} s a game (both arms), selection "
          f"{100 * secs / max(gsecs, 1e-9):.2f}% of game time")
