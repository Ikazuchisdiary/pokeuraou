"""IKA-410 stage 1: the change of a seat's search value between its consecutive move decisions.

    python stage1.py <games-per-file> <jobs> <out.jsonl>

Every file of data/selfplay-mc3 (gen-3 generation, value-mc2), a fixed random sample of lines
per file (seed 410). One output row per (game, seat, consecutive move-decision pair):
the seat's own win probability before and after (seat 0: searchValue, seat 1: 1 - foeSearchValue,
both fields are in side 0's units; where foeSearchValue is absent seat 1 uses searchValue),
and labels taken from the two recorded positions.
"""

import json
import zlib
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
SRC = M / "data/selfplay-mc3"


def alive(pos: dict, side: int) -> int:
    return sum(1 for p in pos["sides"][side]["pokemon"] if not p.get("fainted") and p["hp"] > 0)


def features(pos: dict, side: int) -> dict:
    mons = pos["sides"][side]["pokemon"]
    return {
        "mega": any(p.get("isMega") for p in mons),
    }


def seat_value(d: dict, seat: int) -> float | None:
    v = d.get("searchValue")
    if v is None:
        return None
    if seat == 0:
        return float(v)
    f = d.get("foeSearchValue")
    return 1.0 - float(v if f is None else f)


def rows_of(line: bytes, file: str, lineno: int) -> list[dict]:
    g = json.loads(line)
    moves = [(i, d) for i, d in enumerate(g["decisions"]) if d["kind"] == "move"]
    out = []
    for seat in (0, 1):
        foe = 1 - seat
        for (i0, d0), (i1, d1) in zip(moves, moves[1:]):
            v0, v1 = seat_value(d0, seat), seat_value(d1, seat)
            if v0 is None or v1 is None:
                continue
            p0, p1 = d0["position"], d1["position"]
            s0 = d0.get("shownIdentities") or [[], []]
            s1 = d1.get("shownIdentities") or [[], []]
            out.append({
                "file": file, "line": lineno, "game": g["gameIndex"], "seat": seat,
                "d0": i0, "d1": i1, "turn0": d0["turn"], "turn1": d1["turn"],
                "v0": v0, "v1": v1, "dv": v1 - v0,
                "own_alive0": alive(p0, seat), "foe_alive0": alive(p0, foe),
                "own_faint": alive(p0, seat) - alive(p1, seat),
                "foe_faint": alive(p0, foe) - alive(p1, foe),
                "foe_shown0": len(s0[foe]), "foe_newly_shown": len(set(s1[foe]) - set(s0[foe])),
                "own_newly_shown": len(set(s1[seat]) - set(s0[seat])),
                "open0": "foeSearchValue" not in d0,
                "outcome": None if g["outcome"] is None else (g["outcome"] if seat == 0 else 1 - g["outcome"]),
                "own_mega0": features(p0, seat)["mega"], "foe_mega0": features(p0, foe)["mega"],
                "weather0": p0["field"].get("weather"), "terrain0": p0["field"].get("terrain"),
                "trickroom0": "trickroom" in [x if isinstance(x, str) else x.get("id")
                                               for x in p0["field"].get("pseudoWeather", [])],
                "game_turns": g["turns"],
            })
    return out


def work(args):
    path, per = args
    lines = Path(path).read_bytes().split(b"\n")
    lines = [ln for ln in lines if ln.strip()]
    rng = np.random.default_rng([410, zlib.crc32(Path(path).name.encode())])
    pick = sorted(rng.choice(len(lines), size=min(per, len(lines)), replace=False))
    rows = []
    for k in pick:
        rows.extend(rows_of(lines[k], Path(path).name, int(k)))
    return Path(path).name, len(lines), len(pick), rows


if __name__ == "__main__":
    per, jobs, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    files = sorted(str(p) for p in SRC.glob("games-*.jsonl"))
    total_lines = total_pick = 0
    with Pool(jobs) as pool, out.open("wb") as f:
        for name, n, k, rows in pool.imap_unordered(work, [(p, per) for p in files]):
            total_lines += n
            total_pick += k
            for r in rows:
                f.write(json.dumps(r).encode() + b"\n")
    print(f"files {len(files)} games {total_lines} sampled {total_pick}", file=sys.stderr)
