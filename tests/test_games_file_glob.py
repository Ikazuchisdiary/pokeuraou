"""Game files are read under both names: games-workerK (old) and games-bNN-workerK (gen-2).

The readers globbed ``games-worker*.jsonl`` only, so gen-2's ``games-b00-worker3.jsonl`` was
invisible (IKA-400 hard-linked around it). Summary json and rank-worker files stay unread.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _game(index: int) -> bytes:
    return (json.dumps({"gameIndex": index, "decisions": [{"kind": "move", "turn": 1}]}) + "\n").encode()


def test_q_teach_index_reads_both_names(tmp_path) -> None:  # noqa: ANN001
    (tmp_path / "games-worker0.jsonl").write_bytes(_game(0) + _game(1))
    (tmp_path / "games-b00-worker1.jsonl").write_bytes(_game(2) + _game(3))
    # Must not be picked up.
    (tmp_path / "rank-worker0.jsonl.gz").write_bytes(b"x")
    (tmp_path / "workers-Q1.json").write_bytes(b"{}")
    (tmp_path / "summary.jsonl").write_bytes(_game(9))
    out = tmp_path / "index.npz"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "q_teach.py"), "index", "--games-dir", str(tmp_path),
         "--out", str(out), "--skip-lines", "0"],
        check=True, capture_output=True,
    )
    with np.load(out) as z:
        assert json.loads(str(z["files"])) == ["games-b00-worker1.jsonl", "games-worker0.jsonl"]
        assert sorted(z["game"].tolist()) == [0, 1, 2, 3]
