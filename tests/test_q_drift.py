"""`tools/q_drift.py`'s rule reads the right numbers (IKA-348): r is the growth of Q's cell
error from the old leaf's cells to the new leaf's, and a level shift of a whole matrix counts
in r but not in the demeaned figure."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("q_drift", ROOT / "tools" / "q_drift.py")
q_drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q_drift)


def _matrices(seed: int, count: int = 20) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = [(int(rng.integers(3, 9)), int(rng.integers(3, 9))) for _ in range(count)]
    return [rng.uniform(0.1, 0.9, size=shape) for shape in shapes]


def test_the_same_leaf_twice_is_no_drift():
    old = _matrices(1)
    q = [m + np.random.default_rng(2).normal(0, 0.03, m.shape) for m in old]
    report = q_drift.drift(q, old, [m.copy() for m in old])
    assert report["r"] == 0.0 and report["keep"]
    assert report["old_vs_new"]["mae"] == 0.0
    assert report["old_vs_new"]["spearman_rows"] > 0.999


def test_a_level_shift_counts_in_r_and_not_in_the_demeaned_figure():
    old = _matrices(3)
    q = [m + np.random.default_rng(4).normal(0, 0.02, m.shape) for m in old]
    new = [m + 0.05 for m in old]
    report = q_drift.drift(q, old, new)
    assert report["r"] > q_drift.KEEP_BELOW and not report["keep"]
    assert abs(report["r_demeaned"]) < 1e-9
    assert report["q_vs_new"]["spearman_rows"] == report["q_vs_old"]["spearman_rows"]


def test_a_new_leaf_that_reorders_the_candidates_is_drift_in_both():
    old = _matrices(5)
    q = [m + np.random.default_rng(6).normal(0, 0.02, m.shape) for m in old]
    new = [m[::-1, ::-1].copy() for m in old]
    report = q_drift.drift(q, old, new)
    assert report["r"] > q_drift.KEEP_BELOW and report["r_demeaned"] > q_drift.KEEP_BELOW
    assert report["q_vs_new"]["spearman_rows"] < report["q_vs_old"]["spearman_rows"]
