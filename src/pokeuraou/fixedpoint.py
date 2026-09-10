"""Showdown's fixed-point modifier arithmetic, reproduced exactly.

Damage in Showdown is not computed with floats and rounded at the end. Every modifier is
a 12-bit fixed-point number (4096 = 1.0), modifiers within one event are *chained* with a
specific rounding rule, and the accumulated modifier is applied to the value once. Getting
this wrong shifts damage by 1 point, which is enough to change a KO threshold, so it is
reproduced here rather than approximated.

From ``sim/battle.ts``::

    chainModify(num, den = 1) {
        const previousMod = trunc(this.event.modifier * 4096);
        const nextMod = trunc(num * 4096 / den);
        this.event.modifier = ((previousMod * nextMod + 2048) >> 12) / 4096;
    }

    modify(value, num, den = 1) {
        const modifier = trunc(num * 4096 / den);
        return trunc((trunc(value * modifier) + 2048 - 1) / 4096);
    }

and ``sim/dex.ts``::

    trunc(num, bits = 0) {
        if (bits) return (num >>> 0) % (2 ** bits);
        return num >>> 0;
    }

``num >>> 0`` is an unsigned 32-bit cast, so for the non-negative values that occur in a
damage calculation it is a floor.
"""

from __future__ import annotations

import numpy as np

#: 1.0 in Showdown's modifier fixed point.
ONE = 4096

_UINT32 = 1 << 32
_UINT16 = 1 << 16


def trunc(x: np.ndarray | float, bits: int = 0) -> np.ndarray:
    """``Dex#trunc``: floor to an integer, optionally wrapped to ``bits`` bits."""
    v = np.trunc(np.asarray(x, dtype=np.float64)).astype(np.int64) % _UINT32
    if bits:
        return v % (1 << bits)
    return v


def trunc16(x: np.ndarray) -> np.ndarray:
    """The final truncation in ``modifyDamage``, which can wrap a huge value to 0."""
    return np.asarray(x, dtype=np.int64) % _UINT16


def to_fp(numerator: float, denominator: float = 1.0) -> int:
    """``trunc(num * 4096 / den)`` -- a ratio as a fixed-point modifier."""
    return int(numerator * ONE / denominator)


class Chain:
    """One event's modifier accumulator.

    Modifiers are chained in the order Showdown's handlers run (by descending handler
    priority). The chaining rule rounds at each step, so the order is not free: chaining
    the same set of modifiers in a different order can differ by 1 in the final damage.
    """

    __slots__ = ("modifier", "_applied")

    def __init__(self) -> None:
        self.modifier = ONE
        self._applied: list[tuple[str, int]] = []

    def add(self, numerator: float, denominator: float = 1.0, *, label: str = "") -> Chain:
        next_mod = to_fp(numerator, denominator)
        self.modifier = (self.modifier * next_mod + 2048) >> 12
        self._applied.append((label, next_mod))
        return self

    def add_fp(self, next_mod: int, *, label: str = "") -> Chain:
        """Chains a modifier already expressed in 4096 units (e.g. ``[5325, 4096]``)."""
        self.modifier = (self.modifier * next_mod + 2048) >> 12
        self._applied.append((label, next_mod))
        return self

    def apply(self, value: np.ndarray) -> np.ndarray:
        """``modify(value, accumulatedModifier)``."""
        return apply_fp(value, self.modifier)

    @property
    def is_identity(self) -> bool:
        return self.modifier == ONE

    @property
    def applied(self) -> tuple[tuple[str, int], ...]:
        """(label, modifier) for each chained modifier, for divergence reporting."""
        return tuple(self._applied)

    def __repr__(self) -> str:
        parts = ", ".join(f"{label}={mod / ONE:.4g}" for label, mod in self._applied) or "identity"
        return f"Chain({self.modifier / ONE:.6g}: {parts})"


def apply_fp(value: np.ndarray, modifier: int) -> np.ndarray:
    """``modify(value, modifier / 4096)`` for an integer-valued array."""
    v = np.asarray(value, dtype=np.int64)
    if modifier == ONE:
        return v
    return (v * modifier + 2047) >> 12


def modify(value: np.ndarray, numerator: float, denominator: float = 1.0) -> np.ndarray:
    """``Battle#modify``: a single modifier applied directly, with no chaining."""
    return apply_fp(value, to_fp(numerator, denominator))
