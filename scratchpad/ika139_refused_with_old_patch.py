"""IKA-139: `fastpath_refused.py` rerun with `_patched` as it was before IKA-119.

`fastpath_refused.py` found 44 matrices / 1,429 cells on `data/ika73/w12` games 0-9 before
IKA-119 and finds 0 after it. This puts the old `_patched` back into the current tree
(everything else as on master) to see whether that alone reproduces the 44 / 1,429.

    PYTHONPATH=<tree>/src POKEURAOU_RUST_NODE=1 \\
        python scratchpad/ika139_refused_with_old_patch.py data/ika73/w12 --games 10
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import fastpath_refused  # noqa: E402
from ika139_patched_vs_scratch import _patched_before_ika119  # noqa: E402

from pokeuraou import beliefnode  # noqa: E402

if __name__ == "__main__":
    beliefnode._patched = _patched_before_ika119
    fastpath_refused.main()
