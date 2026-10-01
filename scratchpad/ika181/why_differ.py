"""IKA-181: the games of chg-ben that differ from id-new with no ally target in a final menu --
did the first differing decision's legal pool (what the Q ranks) grow under benefit?"""
import sys

sys.path.insert(0, "C:/tmp/ika181")
from pathlib import Path  # noqa: E402

from pokeuraou import actions  # noqa: E402
from pokeuraou.position import Position  # noqa: E402
from pokeuraou.qhead import legal_pool  # noqa: E402
from pokeuraou.regulation import load_regulation  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("cmp", "C:/tmp/ika181/compare_lib.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)

reg = load_regulation("gen9championsvgc2026regmc")
A, B = cmp.load(Path(sys.argv[1])), cmp.load(Path(sys.argv[2]))
grew = same = 0
for key in sorted(set(A) & set(B)):
    a, b = A[key], B[key]
    if sum(b.get("allyMenus") or [0]):
        continue
    n = cmp.first_diff(a, b)
    if n is None:
        continue
    d = a["decisions"][n]
    pos = Position.from_json(d["position"])
    sizes = {}
    for m in ("off", "benefit"):
        with actions.ally_targets(m):
            sizes[m] = (len(legal_pool(reg, pos, 0)), len(legal_pool(reg, pos, 1)))
    if sizes["off"] != sizes["benefit"]:
        grew += 1
    else:
        same += 1
        print("no growth at", key, "decision", n, d["kind"], sizes)
print(f"differing games with no ally target in a menu: pool grew at the first difference {grew}, did not {same}")
