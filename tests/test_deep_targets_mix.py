"""tools/deep_targets.py mix (IKA-296): the isotonic fit and the two ways of mixing a deep value in."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deep_targets", ROOT / "tools" / "deep_targets.py")
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)


def test_isotonic_pools_violators_and_is_monotone() -> None:
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
    fit = dt.isotonic(x, y)
    # 0.2 and 0.3 violate and pool to 0.5; the rest stay.
    assert np.allclose(dt.calibrate(fit, x), [0.0, 0.5, 0.5, 1.0, 1.0])
    grid = dt.calibrate(fit, np.linspace(0, 1, 50))
    assert np.all(np.diff(grid) >= 0)


def test_relabel_rows_and_back() -> None:
    # Two games: rows 0-3 and rows 4-6. Targets at rows 1 and 3 of game 0, none in game 1.
    game = np.array([0, 0, 0, 0, 1, 1, 1])
    z = np.array([1, 1, 1, 1, 0, 0, 0], float)
    target = np.array([0, 1, 0, 1, 0, 0, 0], bool)
    value = np.array([np.nan, 0.6, np.nan, 0.8, np.nan, np.nan, np.nan])

    rows = dt.relabel(game, z, target, value, 1.0, back=False)
    assert np.allclose(rows, [1, 0.6, 1, 0.8, 0, 0, 0])

    back = dt.relabel(game, z, target, value, 1.0, back=True)
    # Each row takes the first target at or after it in its own game; game 1 is untouched.
    assert np.allclose(back, [0.6, 0.6, 0.8, 0.8, 0, 0, 0])

    half = dt.relabel(game, z, target, value, 0.5, back=True)
    assert np.allclose(half, [0.8, 0.8, 0.9, 0.9, 0, 0, 0])


def test_relabel_does_not_carry_a_value_across_games() -> None:
    # A target in the first row of game 1 must not reach the last rows of game 0.
    game = np.array([0, 0, 1, 1])
    z = np.array([0, 0, 1, 1], float)
    target = np.array([0, 0, 1, 0], bool)
    value = np.array([np.nan, np.nan, 0.3, np.nan])
    assert np.allclose(dt.relabel(game, z, target, value, 1.0, back=True), [0, 0, 0.3, 1])
