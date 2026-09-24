"""A node's span means in one pass, to the bit of the per-cell `values[indices] @ weights`.

A port fill describes each cell whose turn did not stop as a weighted mean of leaf values:
``(i, j, indices, weights)``. The readers (`beliefnode.belief_payoffs`, `port._encoded`)
wrote each one as ``float(values[indices] @ np.asarray(weights))`` -- a Python loop with a
fancy index, an array from a list and a BLAS call per cell, 16% of the generation worker's
CPU with the loops of `belief_payoffs` (IKA-258). This computes every span of a fill at
once (IKA-265).

It has to be the *same* double, or the games change. `@` on two float64 vectors is
OpenBLAS's `ddot`, and below 16 elements that kernel is a sequential loop of fused
multiply-adds, ``dot = fma(x[k], y[k], dot)`` from ``dot = 0`` (the compiler contracts the
tail loop; measured on the recorded spans, `records/IKA-265.md`). A plain ``(x * y).sum()``
rounds every product and adds in another order, and differs in the last bit on about a
third of spans. So the short spans here run that same loop, a column at a time across all
spans, with the fused multiply-add emulated exactly in plain double arithmetic (Boldo and
Melquiond's algorithm: an exact product and an exact sum, their tails rounded to odd, one
final rounding). Spans of 16 or more go to the kernel's blocked path, which this does not
reproduce, so they are still one `@` each -- 2.6% of spans.

Whether the loop *is* the kernel is a property of the BLAS build and the CPU it dispatched
to, not of this code. `exact()` checks it once per process against `@` itself on fixed
probes, and when it disagrees every span goes back to its own `@`: slower, never different.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import chain

import numpy as np

#: The kernel's tail loop runs below this length; at it the blocked path starts.
SHORT = 16
#: Veltkamp's constant for splitting a double into two 26-bit halves.
_SPLIT = 134217729.0  # 2**27 + 1
#: Below this magnitude a product's rounding error could be subnormal, and the exact
#: product would not be exact. Leaf values and chance weights are nowhere near it.
_TINY = 2.0**-900
_BIG = 2.0**900

_EXACT: bool | None = None


def _two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = _SPLIT * a
    high = t - (t - a)
    return high, a - high


def _fma(
    a: np.ndarray, b: np.ndarray, b_high: np.ndarray, b_low: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """``a * b + c`` rounded once, for finite inputs well inside the exponent range."""
    product = a * b
    a_high, a_low = _split(a)
    product_err = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low
    high, low = _two_sum(c, product)
    # low + product_err, rounded to odd: round to nearest, and when that was inexact and
    # landed on an even significand, step to the neighbour on the other side of the truth.
    tail, err = _two_sum(low, product_err)
    even = (tail.view(np.int64) & 1) == 0
    step = (err != 0) & even
    if step.any():
        tail = np.where(step, np.nextafter(tail, np.where(err > 0, np.inf, -np.inf)), tail)
    return high + tail


class SpanTable:
    """One fill's spans, laid out once, read against any number of value rows.

    Spans without weights are dropped, as every reader skips them. `i` and `j` are each
    kept span's cell, in the fill's order.

    `groups` stacks several fills' spans in one table, each with the offset of its own rows
    in the values it will be read against: ``[(spans, offset), ...]``. `bounds[g]` is where
    group `g`'s kept spans sit in the table.
    """

    def __init__(
        self,
        spans: Sequence[tuple[int, int, Sequence[int], Sequence[float]]] = (),
        *,
        groups: Sequence[tuple[Sequence, int]] | None = None,
    ) -> None:
        if groups is None:
            groups = [(spans, 0)]
        kept: list = []
        kept_offsets: list[int] = []
        self.bounds: list[tuple[int, int]] = []
        for group, offset in groups:
            lo = len(kept)
            kept.extend(span for span in group if span[3])
            kept_offsets.extend([int(offset)] * (len(kept) - lo))
            self.bounds.append((lo, len(kept)))
        self._kept = kept
        count = len(kept)
        self.size = count
        self._offsets = np.asarray(kept_offsets, dtype=np.intp)
        self.i = np.fromiter((span[0] for span in kept), dtype=np.intp, count=count)
        self.j = np.fromiter((span[1] for span in kept), dtype=np.intp, count=count)
        lengths = np.fromiter((len(span[3]) for span in kept), dtype=np.intp, count=count)
        total = int(lengths.sum())
        flat_index = np.fromiter(
            chain.from_iterable(span[2] for span in kept), dtype=np.intp, count=total
        ) + np.repeat(self._offsets, lengths)
        flat_weight = np.fromiter(
            chain.from_iterable(span[3] for span in kept), dtype=np.float64, count=total
        )
        long = lengths >= SHORT
        self._long = [self._span(k) for k in np.flatnonzero(long)]
        self._each_made: list | None = None
        # The short spans, longest first, so the spans still running at column k are a prefix.
        short = np.flatnonzero(~long)
        order = short[np.argsort(-lengths[short], kind="stable")]
        self._order = order
        widest = int(lengths[order[0]]) if len(order) else 0
        self._active = [int((lengths[order] > k).sum()) for k in range(widest)]
        index = np.zeros((widest, len(order)), dtype=np.intp)
        weight = np.zeros((widest, len(order)), dtype=np.float64)
        # Each element of a short span to (its place in the span, its span's place in order).
        rank = np.empty(count, dtype=np.intp)
        rank[order] = np.arange(len(order))
        span_of = np.repeat(np.arange(count), lengths)
        place = np.arange(total) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        mine = ~long[span_of]
        index[place[mine], rank[span_of[mine]]] = flat_index[mine]
        weight[place[mine], rank[span_of[mine]]] = flat_weight[mine]
        self._index = index
        self._weight = weight
        self._weight_split = _split(weight)
        self._weight_ok = bool(
            np.isfinite(weight).all() and (np.abs(weight) < _BIG).all()
        )

    def means(self, values: np.ndarray, starts: Sequence[int] = (0,)) -> np.ndarray:
        """``(len(starts), size)``: each span's mean over ``values[start + indices]``.

        Bit for bit ``float(values[start + indices] @ np.asarray(weights))`` span by span.
        """
        values = np.asarray(values, dtype=np.float64)
        out = np.zeros((len(starts), self.size), dtype=np.float64)
        if not self.size:
            return out
        base = np.asarray(starts, dtype=np.intp)
        if self._fast(values):
            order = self._order
            acc = np.zeros((len(starts), len(order)), dtype=np.float64)
            w_high, w_low = self._weight_split
            for k, active in enumerate(self._active):
                x = values[base[:, None] + self._index[k, :active]]
                w = self._weight[k, :active]
                if k == 0:
                    # fma(x, w, 0) is the rounded product.
                    acc[:, :active] = x * w
                else:
                    acc[:, :active] = _fma(
                        x, w, w_high[k, :active], w_low[k, :active], acc[:, :active]
                    )
            # numpy adds the kernel's answer to a zero it starts from.
            out[:, order] = acc + 0.0
            slow = self._long
        else:
            if self._each_made is None:
                self._each_made = [self._span(k) for k in range(self.size)]
            slow = self._each_made
        for k, indices, weights in slow:
            for row, start in enumerate(base):
                out[row, k] = float(values[start + indices] @ weights)
        return out

    def _span(self, k: int) -> tuple[int, np.ndarray, np.ndarray]:
        """Span `k` as its own `@` takes it: the long ones always, every one when the loop
        is not this BLAS's."""
        _i, _j, indices, weights = self._kept[int(k)]
        return (
            int(k),
            np.asarray(indices, dtype=np.intp) + self._offsets[int(k)],
            np.asarray(weights, dtype=np.float64),
        )

    def write(self, target: np.ndarray, means: np.ndarray, group: int | None = None) -> None:
        """``target[i, j] = mean`` for every kept span (of one group), in the fill's order.

        A cell two spans name keeps the later one's, as the loop this replaces did.
        """
        lo, hi = (0, self.size) if group is None else self.bounds[group]
        i, j = self.i[lo:hi], self.j[lo:hi]
        if self._unique(lo, hi):
            target[i, j] = means[lo:hi]
            return
        for k in range(lo, hi):
            target[self.i[k], self.j[k]] = means[k]

    def _unique(self, lo: int, hi: int) -> bool:
        if hi - lo < 2:
            return True
        key = self.i[lo:hi] * (int(self.j[lo:hi].max()) + 1) + self.j[lo:hi]
        return len(np.unique(key)) == hi - lo

    def _fast(self, values: np.ndarray) -> bool:
        if not exact() or not self._weight_ok:
            return False
        magnitude = np.abs(values)
        if not np.isfinite(values).all() or (magnitude >= _BIG).any():
            return False
        # A product that is tiny but not zero could lose its exact tail to underflow.
        small = (magnitude < _TINY) & (magnitude != 0)
        return not small.any() and not (
            (np.abs(self._weight) < _TINY) & (self._weight != 0)
        ).any()


def exact() -> bool:
    """Is `@` on short float64 vectors the fused sequential loop here? Checked once."""
    global _EXACT  # noqa: PLW0603 - a per-process answer about the BLAS in this process
    if _EXACT is None:
        _EXACT = _probe()
    return _EXACT


def _probe() -> bool:
    rng = np.random.default_rng(265)
    spans = []
    values = []
    at = 0
    for length in range(1, SHORT):
        for trial in range(64):
            x = rng.random(length)
            if trial % 2:
                x = x.astype(np.float32).astype(np.float64)
            w = rng.random(length)
            w = w / w.sum()
            values.append(x)
            spans.append((0, 0, list(range(at, at + length)), w.tolist()))
            at += length
    flat = np.concatenate(values)
    table = SpanTable(spans)
    global _EXACT  # noqa: PLW0603
    _EXACT = True  # let `means` take the loop it is being checked on
    try:
        got = table.means(flat)[0]
    finally:
        _EXACT = None
    want = np.array(
        [float(flat[list(ix)] @ np.asarray(w)) for _i, _j, ix, w in spans], dtype=np.float64
    )
    return got.tobytes() == want.tobytes()


__all__ = ["SHORT", "SpanTable", "exact"]
