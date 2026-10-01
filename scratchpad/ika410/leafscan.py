"""IKA-410: at every move decision of the stage-1 sample, the seat's recorded search value against
the leaf (value-mc2) of the recorded (true) position: r = V_seat - leaf_seat. One row per
(decision, seat), with the position's features.

    python leafscan.py <stage1.jsonl> <jobs> <out.jsonl>
"""

import json
import sys
from multiprocessing import Pool
from pathlib import Path

M = Path("C:/Users/Ikazuchi/repos/pokeuraou")
SRC = M / "data/selfplay-mc3"
_state: dict = {}


def init() -> None:
    import torch

    from pokeuraou.damage import register_mega_stones
    from pokeuraou.encode import Encoder
    from pokeuraou.pool import load_pool
    from pokeuraou.value import BatchedValue, load_model

    torch.set_num_threads(1)
    pool = load_pool(str(M / "data/pool/regmc-matchupweb.json"))
    reg = pool.reg
    register_mega_stones(reg)
    encoder = Encoder(reg)
    net, _ = load_model(M / "data/models/value-mc2.pt", encoder)
    _state["leaf"] = BatchedValue(net.to("cpu"), encoder, device=torch.device("cpu"))


def alive(pos: dict, side: int) -> int:
    return sum(1 for p in pos["sides"][side]["pokemon"] if not p.get("fainted") and p["hp"] > 0)


def side_features(pos: dict, side: int) -> dict:
    s = pos["sides"][side]
    mons = s["pokemon"]
    act = [mons[i] for i in s.get("active", []) if i is not None and i < len(mons)]
    act = [m for m in act if not m.get("fainted") and m["hp"] > 0]
    conds = s.get("sideConditions") or s.get("conditions") or {}
    names = list(conds) if isinstance(conds, dict) else [c if isinstance(c, str) else c.get("id") for c in conds]
    return {
        "alive": alive(pos, side),
        "hp": sum(m["hp"] / m["maxhp"] for m in mons if not m.get("fainted")),
        "mega_used": bool(s.get("megaUsed")) or any(m.get("isMega") for m in mons),
        "mega_left": bool(s.get("megaCapableSlots")) and not s.get("megaUsed"),
        "status": sum(1 for m in act if m.get("status")),
        "boosted": sum(1 for m in act if any(v > 0 for v in (m.get("boosts") or {}).values())),
        "conds": sorted(n for n in names if n),
    }


def work(args):
    from pokeuraou.position import Position

    path, lines = args
    raw = [ln for ln in Path(path).read_bytes().split(b"\n") if ln.strip()]
    out = []
    for k in lines:
        g = json.loads(raw[k])
        moves = [(i, d) for i, d in enumerate(g["decisions"]) if d["kind"] == "move"]
        if not moves:
            continue
        leaves = _state["leaf"]([Position.from_json(d["position"]) for _i, d in moves])
        for (i, d), lv in zip(moves, leaves):
            pos = d["position"]
            f0, f1 = side_features(pos, 0), side_features(pos, 1)
            pw = [x if isinstance(x, str) else x.get("id") for x in pos["field"].get("pseudoWeather", [])]
            sh = d.get("shownIdentities") or [[], []]
            for seat in (0, 1):
                v = float(d["searchValue"]) if seat == 0 else 1.0 - float(d.get("foeSearchValue", d["searchValue"]))
                leaf = float(lv) if seat == 0 else 1.0 - float(lv)
                own, foe = (f0, f1) if seat == 0 else (f1, f0)
                out.append({
                    "file": Path(path).name, "line": k, "game": g["gameIndex"], "d": i, "seat": seat,
                    "turn": d["turn"], "v": v, "leaf": leaf, "r": v - leaf,
                    "own": own, "foe": foe, "foe_shown": len(sh[1 - seat]), "own_shown": len(sh[seat]),
                    "weather": pos["field"].get("weather"), "terrain": pos["field"].get("terrain"),
                    "trickroom": "trickroom" in pw,
                    "outcome": None if g["outcome"] is None else (g["outcome"] if seat == 0 else 1 - g["outcome"]),
                })
    return out


if __name__ == "__main__":
    rows = [json.loads(x) for x in Path(sys.argv[1]).read_bytes().splitlines() if x.strip()]
    need: dict[str, set[int]] = {}
    for r in rows:
        need.setdefault(r["file"], set()).add(r["line"])
    tasks = [(str(SRC / f), sorted(v)) for f, v in sorted(need.items())]
    with Pool(int(sys.argv[2]), initializer=init) as pool, Path(sys.argv[3]).open("wb") as f:
        n = 0
        for got in pool.imap_unordered(work, tasks):
            for r in got:
                f.write(json.dumps(r).encode() + b"\n")
            n += 1
            if n % 40 == 0:
                print("files", n, file=sys.stderr, flush=True)
