"""IKA-411: two match directories game by game.

    python compare.py <dir A> <dir B>

Pairs records by (gameIndex, seatIndex). A game is the same when its decisions agree in kind,
turn, both chosen actions, both menus, both policies and both search values (|gap| <= 1e-9),
and the outcome agrees. Positions are not compared (their baseSpecies spelling is what the fix
changes). Also counts, per side, the games where a record says the pre-fix belief changed a
belief (dexBaseRewrites, the positive control that the rule fired), and the games where a
Floette-Eternal / Meowstic-F Mega stood on the board.
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


def forme_mega(g: dict) -> bool:
    text = json.dumps(g["decisions"])
    return any(f'"species": "{f}"' in text for f in FORMES)


A, B = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
common = sorted(set(A) & set(B))
c = Counter()
diffs = []
for key in common:
    a, b = A[key], B[key]
    mega = forme_mega(a) or forme_mega(b)
    c["games"] += 1
    c["forme_mega"] += mega
    for name, g in (("A", a), ("B", b)):
        rw = g.get("dexBaseRewrites")
        if rw and sum(rw):
            c[f"{name}_games_with_rewrites"] += 1
            c[f"{name}_rewrites"] += sum(rw)
    n = first_diff(a, b)
    if n is None:
        c["same"] += 1
        c["same_forme_mega"] += mega
    else:
        c["differ"] += 1
        c["differ_forme_mega"] += mega
        if len(diffs) < 10:
            diffs.append((key, n, mega))
print(f"A {len(A)} records, B {len(B)} records, common {len(common)}")
for k in sorted(c):
    print(f"  {k}: {c[k]}")
print("first differences (game, seat), decision, forme mega:", diffs)
