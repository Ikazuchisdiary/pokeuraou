"""IKA-181: two match directories game by game.

    python compare.py <dir A> <dir B>

Pairs records by (gameIndex, seatIndex). A game is the same when its decisions agree in kind,
turn, both chosen actions, both menus, both policies and both search values (|gap| <= 1e-9),
and the outcome agrees. Also counts, per directory, the games whose records say a menu held
an ally target (allyMenus) and the games where one was played (allyPlayed).
"""

import json
import sys
from collections import Counter
from pathlib import Path

TOL = 1e-9


def load(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("games-*.jsonl")):
        for line in f.read_bytes().splitlines():
            if line.strip():
                g = json.loads(line)
                out[(g["gameIndex"], g["seatIndex"])] = g
    return out


def first_diff(a: dict, b: dict) -> int | None:
    da, db = a["decisions"], b["decisions"]
    for n in range(max(len(da), len(db))):
        if n >= len(da) or n >= len(db):
            return n
        x, y = da[n], db[n]
        for key in ("kind", "turn", "ownChosen", "foeChosen", "ownActions", "foeActions"):
            if x.get(key) != y.get(key):
                return n
        for key in ("ownPolicy", "foePolicy"):
            p, q = x.get(key) or [], y.get(key) or []
            if len(p) != len(q) or any(abs(u - v) > TOL for u, v in zip(p, q)):
                return n
        for key in ("searchValue", "foeSearchValue"):
            p, q = x.get(key), y.get(key)
            if (p is None) != (q is None) or (p is not None and abs(p - q) > TOL):
                return n
    if a.get("outcome") != b.get("outcome"):
        return len(da)
    return None


A, B = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
common = sorted(set(A) & set(B))
c = Counter()
for key in common:
    a, b = A[key], B[key]
    c["games"] += 1
    flags = {}
    for name, g in (("A", a), ("B", b)):
        menus = sum(g.get("allyMenus") or [0])
        played = sum(g.get("allyPlayed") or [0])
        c[f"{name}_games_ally_menu"] += int(menus > 0)
        c[f"{name}_games_ally_played"] += int(played > 0)
        flags[name] = menus > 0
    n = first_diff(a, b)
    if n is None:
        c["same"] += 1
        c["same_with_ally_menu"] += int(flags["A"] or flags["B"])
    else:
        c["differ"] += 1
        c["differ_with_ally_menu"] += int(flags["A"] or flags["B"])
print(f"A {len(A)} games, B {len(B)} games, common {len(common)}")
print(json.dumps(dict(c), indent=1))
