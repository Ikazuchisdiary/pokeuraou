"""A stand-in leaf and Q whose numbers are the same bits on every machine (IKA-464).

`test_humanplay._noisy` and `standin.QStub` read their inputs through ``sin``, ``tanh``,
``exp``, float sums and BLAS products, whose last bits move with the platform's maths library
and the CPU's vector kernels. `QStub` is also built to tie its games up to the last bits (what
`tests/test_port_menus.py` checks), and adds a term that moves with the request's batch. A
deepening steered by such numbers takes other steps on another machine: CI on Linux took 27
steps where Windows took 29 on the same budget (records/IKA-464.md section 11), and nudging the
Q's answer down by one unit in the last place gave the same 27 here.

These two are the same shapes -- a smooth function of the same arrays, cyclic so the games mix
-- built from operations IEEE rounds the same way everywhere: the arrays' values are scaled to
integers (``rint``) and summed as integers, and what follows is ``+ - * /`` and ``floor`` on
single numbers, in a fixed order (no ``sin``, no ``exp``, no BLAS). Pure numpy, no `pokeuraou`
import, so the reference of an old commit (`tests/fixtures/ika464-python-deepening.json`) is
made with them too.
"""

from __future__ import annotations

import numpy as np

#: The integer grid the arrays' floats are read on (they are short binary fractions).
_SCALE = 1 << 20


def _sums(array: np.ndarray) -> np.ndarray:
    """Each row's (first axis) values summed exactly, as a float: integers as they are,
    floats on the `_SCALE` grid, the sum in integers."""
    a = np.asarray(array)
    rows = a.reshape(len(a), -1)
    if a.dtype.kind in "iub":
        return rows.astype(np.int64).sum(axis=1).astype(np.float64)
    ints = np.rint(rows.astype(np.float64) * _SCALE).astype(np.int64)
    return ints.sum(axis=1).astype(np.float64) / _SCALE


def _wave(x: np.ndarray) -> np.ndarray:
    """A periodic stand-in for ``sin``: a triangle wave of period 2 in [-1, 1]."""
    t = x - 2.0 * np.floor(x / 2.0)  # [0, 2)
    return np.where(t < 1.0, 2.0 * t - 1.0, 3.0 - 2.0 * t)


def _squash(z: np.ndarray) -> np.ndarray:
    """A stand-in for ``tanh``: z / (1 + |z|)."""
    return z / (1.0 + np.abs(z))


def exact_leaf(arrays: dict[str, np.ndarray], rows: int) -> np.ndarray:
    """A served leaf, `_noisy`'s shape: a wave of the species', the Pokemon's and the sides'
    arrays, each position's value in [0.05, 0.95]."""
    del rows
    x = _sums(arrays["species"]) * 0.37 + _sums(arrays["mon"]) * 1.3 + _sums(arrays["side"]) * 0.71
    return 0.5 + 0.45 * _wave(x)


class ExactQStub:
    """A Q arm of the stand-in server, `QStub`'s shape without its batch term: each pair's
    score a squashed cyclic function of the difference of the two actions' scores, which read
    the actions' codes and features and the position's Pokemon."""

    files = ["q-exact-stub"]
    digests = ["0" * 64]
    properties = True

    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint

    def _one(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        base = float(_sums(arrays["mon"][None])[0]) * 1e-4
        w = np.arange(1, 15, dtype=np.int64)

        def vec(side: int) -> np.ndarray:
            acts = arrays[f"acts{side}"]
            codes = acts.reshape(len(acts), -1).astype(np.int64)
            width = min(codes.shape[1], len(w))
            dot = (codes[:, :width] * w[:width]).sum(axis=1).astype(np.float64)
            return _squash(dot * 0.01 + _sums(arrays[f"feats{side}"]) * 0.05 + base)

        u, v = vec(0), vec(1)
        d = u[:, None] - v[None, :]
        logit = _wave(9.0 * d / 3.0) + 0.2 * d
        return 0.5 + 0.5 * _squash(logit)

    def __call__(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        return self._one(arrays)

    def batch(self, requests: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
        return [self._one(r) for r in requests]
