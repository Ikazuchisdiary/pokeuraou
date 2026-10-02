"""tools/selection_diag.py (IKA-419): the game reader and the statistics, on small inputs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "selection_diag.py"
_SPEC = importlib.util.spec_from_file_location("selection_diag", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
sd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sd)


def _line(pool: dict, outcome: float) -> bytes:
    # A big nested part before and after, as in a real record, with a decoy "teams" inside.
    record = {"ownTeam": [{"species": "a"}], "outcome": outcome, "decisions": [{"teams": ["zz", "yy"]}],
              "pool": pool}
    return json.dumps(record).encode("utf-8") + b"\n"


def test_scan_reads_both_key_orders(tmp_path: Path) -> None:
    generation = {"id": "p", "sha256": "ab", "teams": ["aa11", "bb22"], "names": ["x", "y"], "pair": 3}
    board = {"id": "p", "sha256": "ab", "pair": 7, "teams": ["bb22", "aa11"], "mirror": False,
             "picks": [[0, 1, 2, 3], [0, 1, 2, 3]]}
    path = tmp_path / "games-1.jsonl"
    path.write_bytes(_line(generation, 1.0) + _line(generation, 0.0) + _line(board, 1.0))
    part = sd._scan_file((str(path), 10))
    assert part["lines"] == 3 and part["skipped"] == 0
    assert part["counts"]["aa11|bb22"] == [2, 1.0]
    assert part["counts"]["bb22|aa11"] == [1, 1.0]


def test_scan_stops_when_the_regex_and_the_parser_disagree(tmp_path: Path) -> None:
    # The decoy "outcome" before the real one makes the regex read the wrong number; the check
    # against json.loads must stop the run (the control that the check can fail).
    pool = {"id": "p", "sha256": "ab", "teams": ["aa11", "bb22"]}
    record = {"x": {"outcome": 0.0}, "outcome": 1.0, "pool": pool}
    path = tmp_path / "games-1.jsonl"
    path.write_bytes(json.dumps(record).encode("utf-8") + b"\n")
    with pytest.raises(SystemExit):
        sd._scan_file((str(path), 5))


def test_wilson_and_spearman() -> None:
    lo, hi = sd.wilson(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21
    x = np.array([0.1, 0.2, 0.3, 0.4])
    assert sd.spearman(x, x**3) == pytest.approx(1.0)
    assert sd.spearman(x, -x) == pytest.approx(-1.0)
