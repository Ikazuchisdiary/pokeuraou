"""A node's span means in one pass, to the bit of the per-cell `values[indices] @ weights`.

A port fill describes each cell whose turn did not stop as a weighted mean of leaf values:
``(i, j, indices, weights)``. The readers (`beliefnode.belief_payoffs`, `port._encoded`)
wrote each one as ``float(values[indices] @ np.asarray(weights))`` -- a Python loop with a
fancy index, an array from a list and a BLAS call per cell. This computes the spans of a
fill together (IKA-265).

It has to be the *same* double, or the games change. `@` on two float64 vectors is
OpenBLAS's `ddot`, and below 16 elements that kernel is a sequential loop of fused
multiply-adds, ``dot = fma(x[k], y[k], dot)`` from ``dot = 0`` (the compiler contracts the
tail loop; measured, `records/IKA-265.md`). A plain ``(x * y).sum()`` rounds every product
and adds in another order, and differs in the last bit on about a third of spans. So the
short spans here run that same loop, a column at a time across all spans, with the fused
multiply-add emulated exactly in plain double arithmetic (Boldo and Melquiond's algorithm:
an exact product and an exact sum, their tails rounded to odd, one final rounding).

A column costs a fixed ~20 numpy calls whatever the number of spans, and a span left to its
own `@` costs about four, so the widest spans are cheaper one by one: each read picks the
number of columns that makes the whole cheapest, and the spans longer than that -- and all
of 16 or more, where the kernel's blocked path starts, which this does not reproduce -- are
still one `@` each. Either road is the same double; the choice is only speed.

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
#: The cost of one column across every span, and of one span read on its own, in the same
#: (rough) unit: numpy calls. Only the speed depends on them.
_COLUMN_COST = 20.0
_SPAN_COST = 4.0

_EXACT: bool | None = None


def _two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _split(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = _SPLIT * a
    high = t - (t - a)
    return high, a - high


def _exact_product(
    a: np.ndarray, b: np.ndarray, b_high: np.ndarray, b_low: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(p, e)`` with ``p = a * b`` rounded and ``p + e`` exactly ``a * b`` (Dekker)."""
    product = a * b
    a_high, a_low = _split(a)
    err = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low
    return product, err


def _fma_step(c: np.ndarray, product: np.ndarray, product_err: np.ndarray) -> np.ndarray:
    """``fma(a, b, c)`` given the exact product of ``a * b`` as ``product + product_err``."""
    high, low = _two_sum(c, product)
    # low + product_err, rounded to odd: round to nearest, and when that was inexact and
    # landed on an even significand, step to the neighbour on the other side of the truth.
    tail, err = _two_sum(low, product_err)
    step = (err != 0) & ((tail.view(np.int64) & 1) == 0)
    if step.any():
        tail = np.where(step, np.nextafter(tail, np.where(err > 0, np.inf, -np.inf)), tail)
    return high + tail


def _fma(
    a: np.ndarray, b: np.ndarray, b_high: np.ndarray, b_low: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """``a * b + c`` rounded once, for finite inputs well inside the exponent range."""
    return _fma_step(c, *_exact_product(a, b, b_high, b_low))


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
        self._built: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        count = len(kept)
        self.size = count
        self._offsets = np.asarray(kept_offsets, dtype=np.intp)
        rows, cols, indices, weights = zip(*kept, strict=True) if kept else ((), (), (), ())
        self.i = np.array(rows, dtype=np.intp)
        self.j = np.array(cols, dtype=np.intp)
        lengths = np.fromiter(map(len, weights), dtype=np.intp, count=count)
        total = int(lengths.sum())
        flat_index = np.fromiter(chain.from_iterable(indices), dtype=np.intp, count=total)
        flat_index += np.repeat(self._offsets, lengths)
        flat_weight = np.fromiter(chain.from_iterable(weights), dtype=np.float64, count=total)
        self._unique_cells = self._cells_unique(len(groups))
        long = lengths >= SHORT
        self._long = [int(k) for k in np.flatnonzero(long)]
        # The short spans, longest first: the spans longer than any cut are a prefix, and
        # the spans still running at column k of those after it are a prefix of the rest.
        short = np.flatnonzero(~long)
        order = short[np.argsort(-lengths[short], kind="stable")]
        self._order = order
        widest = int(lengths[order[0]]) if len(order) else 0
        ordered = lengths[order]
        #: longer[k]: how many short spans are longer than k (a prefix of `order`).
        self._longer = (
            len(order) - np.cumsum(np.bincount(ordered, minlength=widest + 1))
        ).tolist()
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
        magnitude = np.abs(flat_weight)
        self._weight_ok = bool(((magnitude < _BIG) & ((magnitude >= _TINY) | (magnitude == 0))).all())

    def _cells_unique(self, groups: int) -> bool:
        """No cell named twice within a group: the cells can then be written at once."""
        if self.size < 2:
            return True
        width = int(self.j.max()) + 1
        key = self.i * width + self.j
        if groups > 1:
            group = np.repeat(np.arange(groups), [hi - lo for lo, hi in self.bounds])
            key += group * (int(self.i.max()) + 1) * width
        return int(np.bincount(key).max()) == 1

    def means(
        self, values: np.ndarray, starts: Sequence[int] = (0,), *, cut: int | None = None
    ) -> np.ndarray:
        """``(len(starts), size)``: each span's mean over ``values[start + indices]``.

        Bit for bit ``float(values[start + indices] @ np.asarray(weights))`` span by span.
        `cut` forces the number of columns run together (the probe's; otherwise the cheapest).
        """
        values = np.asarray(values, dtype=np.float64)
        reads = len(starts)
        out = np.zeros((reads, self.size), dtype=np.float64)
        if not self.size:
            return out
        base = np.asarray(starts, dtype=np.intp)
        alone: list[int]
        if self._fast(values):
            longer = self._longer
            if cut is None:
                cut = min(
                    range(len(longer)),
                    key=lambda k: _COLUMN_COST * k + _SPAN_COST * reads * longer[k],
                )
            cut = min(cut, len(longer) - 1)
            order = self._order
            first = longer[cut]  # spans of `order` before this are longer than the cut
            if cut and first < len(order):
                lanes = order[first:]
                x = values[base[:, None, None] + self._index[None, :cut, first:]]
                w_high, w_low = self._weight_split
                product, err = _exact_product(
                    x, self._weight[:cut, first:], w_high[:cut, first:], w_low[:cut, first:]
                )
                # fma(x, w, 0) is the rounded product.
                acc = product[:, 0, :].copy()
                for k in range(1, cut):
                    active = longer[k] - first
                    if active <= 0:
                        break
                    acc[:, :active] = _fma_step(
                        acc[:, :active], product[:, k, :active], err[:, k, :active]
                    )
                # numpy adds the kernel's answer to a zero it starts from.
                out[:, lanes] = acc + 0.0
            alone = self._long + [int(k) for k in order[:first]]
        else:
            alone = list(range(self.size))
        for k in alone:
            indices, weights = self._span(k)
            for row, start in enumerate(base):
                out[row, k] = float(values[start + indices] @ weights)
        return out

    def _span(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Span `k` as its own `@` takes it."""
        built = self._built.get(k)
        if built is None:
            _i, _j, indices, weights = self._kept[k]
            built = (
                np.asarray(indices, dtype=np.intp) + self._offsets[k],
                np.asarray(weights, dtype=np.float64),
            )
            self._built[k] = built
        return built

    def write(self, target: np.ndarray, means: np.ndarray, group: int | None = None) -> None:
        """``target[i, j] = mean`` for every kept span (of one group), in the fill's order.

        A cell two spans name keeps the later one's, as the loop this replaces did.
        """
        lo, hi = (0, self.size) if group is None else self.bounds[group]
        if self._unique_cells:
            target[self.i[lo:hi], self.j[lo:hi]] = means[lo:hi]
            return
        for k in range(lo, hi):
            target[self.i[k], self.j[k]] = means[k]

    def _fast(self, values: np.ndarray) -> bool:
        if not self._weight_ok or not exact():
            return False
        magnitude = np.abs(values)
        # NaN fails every comparison, so this is also the finiteness check; and a product
        # that is tiny but not zero could lose its exact tail to underflow.
        return bool(((magnitude < _BIG) & ((magnitude >= _TINY) | (magnitude == 0))).all())


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
    _EXACT = True  # let `means` take the loop it is being checked on, every column of it
    try:
        got = table.means(flat, cut=SHORT - 1)[0]
    finally:
        _EXACT = None
    want = np.array(
        [float(flat[list(ix)] @ np.asarray(w)) for _i, _j, ix, w in spans], dtype=np.float64
    )
    return got.tobytes() == want.tobytes()


__all__ = ["SHORT", "SpanTable", "exact"]
