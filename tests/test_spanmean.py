"""`spanmean.SpanTable` gives the per-cell `values[indices] @ weights` to the bit (IKA-265).

The per-cell loop is the definition, so every test compares against it bit for bit. The
positive control is the obvious vectorisation -- `(x * w).sum()` -- which these spans do
tell apart from `@`, so agreement here is not a comparison too blunt to see a last bit.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from pokeuraou import spanmean
from pokeuraou.spanmean import SpanTable, _fma, _split


def _spans(rng: np.random.Generator, count: int, leaves: int) -> list:
    lengths = rng.choice([0, 1, 1, 2, 2, 3, 4, 6, 8, 9, 12, 15, 16, 17, 24, 31], size=count)
    spans = []
    for k, length in enumerate(lengths):
        indices = rng.integers(0, leaves, length).tolist()
        weights = rng.random(length)
        weights = (weights / weights.sum()).tolist() if length else []
        spans.append((k // 12, k % 12, indices, weights))
    return spans


def _values(rng: np.random.Generator, size: int) -> np.ndarray:
    # A leaf's value is a float32 win probability, read as a double.
    return rng.random(size).astype(np.float32).astype(np.float64)


def _loop(spans: list, values: np.ndarray, start: int = 0) -> list[float]:
    return [
        float(values[start : start + 10**9][list(indices)] @ np.asarray(weights))
        for _i, _j, indices, weights in spans
        if weights
    ]


def test_the_probe_finds_this_blas_is_the_fused_loop() -> None:
    # Not a requirement of the code (it falls back), but on the machine the records were
    # taken on it is, and the tests below would then only exercise the fallback.
    assert spanmean.exact()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_means_are_the_loops_doubles(seed: int) -> None:
    rng = np.random.default_rng(seed)
    leaves = 3000
    spans = _spans(rng, 1500, leaves)
    values = _values(rng, leaves * 3)
    table = SpanTable(spans)
    starts = [0, leaves, 2 * leaves]
    got = table.means(values, starts)
    for row, start in enumerate(starts):
        want = np.array(_loop(spans, values, start))
        assert got[row].tobytes() == want.tobytes()


def test_the_obvious_vectorisation_is_told_apart() -> None:
    """Positive control: a rounded product per element and numpy's sum is another double."""
    rng = np.random.default_rng(0)
    spans = _spans(rng, 1500, 3000)
    values = _values(rng, 3000)
    naive = [
        float((values[list(indices)] * np.asarray(weights)).sum())
        for _i, _j, indices, weights in spans
        if weights
    ]
    assert sum(a != b for a, b in zip(naive, _loop(spans, values), strict=True)) > 50


def test_the_fallback_is_the_loop_too(monkeypatch) -> None:  # noqa: ANN001
    rng = np.random.default_rng(4)
    spans = _spans(rng, 600, 1000)
    values = _values(rng, 1000)
    monkeypatch.setattr(spanmean, "_EXACT", False)
    got = SpanTable(spans).means(values)[0]
    assert got.tobytes() == np.array(_loop(spans, values)).tobytes()


def test_groups_read_their_own_rows_and_write_their_own_cells() -> None:
    rng = np.random.default_rng(5)
    a, b = _spans(rng, 144, 400), _spans(rng, 144, 300)
    values = _values(rng, 1000)
    table = SpanTable(groups=[(a, 100), (b, 600)])
    means = table.means(values)[0]
    for group, (spans, offset) in enumerate([(a, 100), (b, 600)]):
        target = np.zeros((12, 12))
        table.write(target, means, group)
        want = np.zeros((12, 12))
        for i, j, indices, weights in spans:
            if weights:
                want[i, j] = float(values[offset:][list(indices)] @ np.asarray(weights))
        assert target.tobytes() == want.tobytes()


def test_a_cell_named_twice_keeps_the_later_span() -> None:
    values = np.array([0.25, 0.5, 0.75])
    spans = [(0, 0, [0], [1.0]), (0, 1, [1], [1.0]), (0, 0, [2], [1.0])]
    target = np.zeros((1, 2))
    table = SpanTable(spans)
    table.write(target, table.means(values)[0])
    assert target.tolist() == [[0.75, 0.5]]


def test_values_the_emulation_cannot_promise_go_to_the_loop() -> None:
    values = np.array([1e-300, 0.5, np.inf, 0.25])
    spans = [(0, 0, [0, 1], [0.5, 0.5]), (0, 1, [1, 3], [0.25, 0.75])]
    with np.errstate(invalid="ignore"):
        got = SpanTable(spans).means(values)[0]
    assert got.tobytes() == np.array(_loop(spans, values)).tobytes()


def test_the_emulated_fused_multiply_add_rounds_once() -> None:
    rng = np.random.default_rng(9)
    n = 20000
    a = rng.random(n) * np.exp2(rng.integers(-40, 4, n))
    b = rng.random(n) * np.exp2(rng.integers(-40, 4, n))
    c = (rng.random(n) - 0.5) * np.exp2(rng.integers(-40, 4, n))
    c[: n // 4] = -(a[: n // 4] * b[: n // 4]) * (1 + (rng.random(n // 4) - 0.5) * 1e-15)
    high, low = _split(b)
    got = _fma(a, b, high, low, c)
    for k in range(0, n, 5):
        assert got[k] == float(Fraction(a[k]) * Fraction(b[k]) + Fraction(c[k])), k
