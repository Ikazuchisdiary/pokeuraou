"""What the opponent's HP bar actually tells us.

The opponent's exact HP is never shown -- only a percentage -- so "the foe is on 62%" is
not a number, it is a *set* of numbers. Treating it as a point estimate quietly moves
every damage comparison by a point or two, which is exactly the margin a KO turns on.

Champions has its own display rule, and Showdown special-cases it::

    // sim/pokemon.ts getHealth()
    const percentage = Math.floor(100 * this.hp / this.maxhp) || 1;
    shared = `${percentage}/100`;
    if (percentage === 20) shared += this.hp * 5 > this.maxhp ? 'y' : 'r';
    else if (percentage === 50) shared += this.hp * 2 > this.maxhp ? 'g' : 'y';

Three things follow, and all three matter:

- It **floors**, where every other modern generation rounds up. A displayed 62% means
  ``floor(100*hp/maxhp) == 62``, so the consistent HP values sit at the *bottom* of the
  band, not around its middle.
- A Pokemon that is alive but rounds to 0% is shown as **1%**, so 1% is the widest band
  of all: anything from 1 HP up.
- At exactly **20% and 50%** the bar colour carries an extra bit -- whether HP is strictly
  above the fifth or the half. That is free information the game hands over, and using it
  narrows the band.

The width of the band is about ``maxhp / 100``, so one to three HP for the stats this
format produces. Small, and load-bearing: whether a hit is lethal can sit inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regulation import Regulation

#: The colour flags Showdown appends at the two thresholds, and what each one asserts.
#: 'r' and 'y' at 20% mean at-or-below and above a fifth; 'y' and 'g' at 50% mean
#: at-or-below and above a half.
COLOUR_AT_20 = {"r": False, "y": True}
COLOUR_AT_50 = {"y": False, "g": True}


class HpDisplayError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Band:
    """The exact HP values consistent with one displayed percentage."""

    percent: int
    maxhp: int
    low: int
    high: int
    colour: str | None = None

    @property
    def values(self) -> range:
        return range(self.low, self.high + 1)

    @property
    def width(self) -> int:
        return self.high - self.low + 1

    def __str__(self) -> str:
        if self.width == 1:
            return f"{self.percent}% = {self.low}"
        return f"{self.percent}% = {self.low}..{self.high}（{self.width}通り）"


def uses_floor_display(reg: Regulation) -> bool:
    """Whether this regulation's mod is the one that floors.

    Read from the dumped mod id rather than hardcoded per format, so a regulation on a
    different mod is not silently given Champions' rule.
    """
    return reg.meta.mod.startswith("champions")


def displayed_percent(hp: int, maxhp: int, *, floor_rule: bool = True) -> int:
    """The percentage the opponent sees. 0 only when the Pokemon has fainted."""
    if maxhp <= 0:
        raise HpDisplayError("maxhp must be positive")
    if hp <= 0:
        return 0
    if floor_rule:
        return (100 * hp // maxhp) or 1
    percent = -(-100 * hp // maxhp)  # ceil
    return 99 if percent == 100 and hp < maxhp else percent


def displayed_colour(hp: int, maxhp: int) -> str | None:
    """The bar colour flag, for the two percentages that carry one."""
    percent = displayed_percent(hp, maxhp)
    if percent == 20:
        return "y" if hp * 5 > maxhp else "r"
    if percent == 50:
        return "g" if hp * 2 > maxhp else "y"
    return None


def band(
    percent: int, maxhp: int, *, colour: str | None = None, floor_rule: bool = True
) -> Band:
    """Every exact HP consistent with a displayed percentage.

    Derived by inverting :func:`displayed_percent` by enumeration rather than by algebra.
    The band is at most a few HP wide, so enumeration is cheap, and it cannot disagree
    with the forward function the way a hand-derived inequality can.
    """
    if maxhp <= 0:
        raise HpDisplayError("maxhp must be positive")
    if percent <= 0:
        return Band(percent=0, maxhp=maxhp, low=0, high=0, colour=None)
    if not 1 <= percent <= 100:
        raise HpDisplayError(f"{percent}% is not a percentage")

    consistent = [
        hp
        for hp in range(1, maxhp + 1)
        if displayed_percent(hp, maxhp, floor_rule=floor_rule) == percent
    ]
    if colour:
        table = (
            COLOUR_AT_20 if percent == 20 else COLOUR_AT_50 if percent == 50 else {}
        )
        flag = table.get(colour)
        if flag is None:
            raise HpDisplayError(
                f"colour {colour!r} means nothing at {percent}%; Showdown only appends one "
                "at 20 and 50"
            )
        threshold = 5 if percent == 20 else 2
        consistent = [hp for hp in consistent if (hp * threshold > maxhp) is flag]

    if not consistent:
        raise HpDisplayError(
            f"no HP out of {maxhp} displays as {percent}%"
            + (f" with colour {colour!r}" if colour else "")
            + "; the observation and the spread are inconsistent, which is information "
            "rather than an error -- the belief should drop this particle"
        )
    return Band(
        percent=percent,
        maxhp=maxhp,
        low=consistent[0],
        high=consistent[-1],
        colour=colour,
    )


def band_bounds(
    percent: int,
    maxhp: np.ndarray,
    *,
    colour: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`band` over a whole particle axis at once, as (low, high) arrays.

    The belief update runs this per observation over thousands of particles, so it is
    solved rather than enumerated: ``floor(100h/M) == p`` is ``p*M/100 <= h < (p+1)*M/100``,
    which for integers is ``ceil(p*M/100) <= h <= ceil((p+1)*M/100) - 1``. The two ends
    the source special-cases -- 1% swallowing everything that floors to zero, 100% meaning
    exactly full -- are applied afterwards.

    :func:`band` stays as the reference implementation and a test asserts the two agree
    for every (hp, maxhp) in range; an algebraic inversion that disagrees with the forward
    function is the kind of bug that silently reweights a belief.
    """
    m = np.asarray(maxhp, dtype=np.int64)
    if (m <= 0).any():
        raise HpDisplayError("maxhp must be positive")
    if percent <= 0:
        zeros = np.zeros_like(m)
        return zeros, zeros
    if not 1 <= percent <= 100:
        raise HpDisplayError(f"{percent}% is not a percentage")

    low = -(-percent * m // 100)
    high = -(-(percent + 1) * m // 100) - 1
    if percent == 1:
        # `|| 1`: everything that would floor to 0% is displayed as 1%.
        low = np.ones_like(m)
    if percent == 100:
        low = m.copy()
        high = m.copy()
    low = np.maximum(low, 1)
    high = np.minimum(high, m)

    if colour:
        table = COLOUR_AT_20 if percent == 20 else COLOUR_AT_50 if percent == 50 else {}
        flag = table.get(colour)
        if flag is None:
            raise HpDisplayError(
                f"colour {colour!r} means nothing at {percent}%; Showdown only appends one "
                "at 20 and 50"
            )
        threshold = 5 if percent == 20 else 2
        # hp * threshold > maxhp is the boundary the colour reports, so it either cuts the
        # band's bottom value off or keeps only that value.
        edge = m // threshold
        if flag:
            low = np.maximum(low, edge + 1)
        else:
            high = np.minimum(high, edge)

    # An empty band (low > high) means this particle's max HP cannot produce the
    # observation. That is evidence, not an error: the caller drops those particles.
    return low, high


__all__ = [
    "COLOUR_AT_20",
    "COLOUR_AT_50",
    "Band",
    "HpDisplayError",
    "band",
    "band_bounds",
    "displayed_colour",
    "displayed_percent",
    "uses_floor_display",
]
