"""IKA-413: two match directories game by game.

    python compare.py <dir A> <dir B>

Pairs records by (gameIndex, seatIndex). A game is the same when its decisions agree in kind,
turn, both chosen actions, both menus, both policies and both search values (|gap| <= 1e-9),
and the outcome agrees. Positions are not compared (their baseSpecies spelling is what the fix
changes). Also counts the games whose record names a port executable for a side (rustBinary, the
positive control that the arm was read through another build).
"""

import json
import sys
from collections import Counter
from pathlib import Path

TOL = 1e-9
FORMES = ("floettemega", "meowsticfmega")


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
diffs = []
for key in common:
    a, b = A[key], B[key]
    c["games"] += 1
    for name, g in (("A", a), ("B", b)):
        if any(g.get("rustBinary") or []):
            c[f"{name}_games_with_a_port_named"] += 1
    n = first_diff(a, b)
    if n is None:
        c["same"] += 1
    else:
        c["differ"] += 1
        if len(diffs) < 10:
            diffs.append((key, n))
print(f"A {len(A)} records, B {len(B)} records, common {len(common)}")
for k in sorted(c):
    print(f"  {k}: {c[k]}")
print("first differences (game, seat), decision:", diffs)
