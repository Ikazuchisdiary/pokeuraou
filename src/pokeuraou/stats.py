"""SP -> final stat computation.

Champions replaces EVs with Stat Points: at most 32 per stat and 66 in total. Showdown's
champions mod stores SP directly in ``set.evs`` and computes stats with, at level 50:

    HP     = base + SP + 75
    others = trunc(trunc((base + SP + 20) * natureNumerator, 16) / 100)

which is the closed form of the game's

    HP     = (base*2 + 31 + SP*2) * 50 / 100 + 60
    others = ((base*2 + 31 + SP*2) * 50 / 100 + 5) * natureMultiplier

There is no SP-to-EV conversion anywhere in this project: the boundary to Showdown passes
SP through unchanged. See ``tests/test_stats.py``, which checks every team-legal species
against Showdown for all natures and a set of boundary spreads.

Everything here is vectorised over a leading "particle" axis, because the belief layer
evaluates thousands of candidate spreads for one Pokemon at a time.
"""

from __future__ import annotations

import numpy as np

from .regulation import STAT_IDS, Regulation

_UINT16 = 1 << 16


def _trunc16(x: np.ndarray) -> np.ndarray:
    """Showdown's ``trunc(x, 16)``: truncate to an integer, then to 16 bits.

    Champions stats never reach 65536 (the largest possible intermediate is
    (255 + 32 + 20) * 110 = 33770), but the mask is applied anyway so the implementation
    matches the reference rather than merely agreeing with it on today's data.
    """
    return np.trunc(x).astype(np.int64) % _UINT16


def nature_multipliers(reg: Regulation, natures: np.ndarray | list[str]) -> np.ndarray:
    """(N,) nature names -> (N, 6) numerator per stat: 110 boosted, 90 hindered, else 100.

    HP is never affected, so column 0 is always 100.
    """
    names = np.asarray(natures, dtype=object).reshape(-1)
    out = np.full((names.shape[0], len(STAT_IDS)), 100, dtype=np.int64)
    for i, name in enumerate(names):
        nat = reg.natures.get(str(name))
        if nat is None:
            raise KeyError(f"Unknown nature: {name!r}")
        if nat.plus:
            out[i, STAT_IDS.index(nat.plus)] = 110
        if nat.minus:
            out[i, STAT_IDS.index(nat.minus)] = 90
    return out


def stats_from_sp(
    reg: Regulation,
    base_stats: np.ndarray,
    sp: np.ndarray,
    nature_numerators: np.ndarray,
    level: int | None = None,
) -> np.ndarray:
    """Final stats for a batch of spreads.

    Args:
        base_stats: (6,) or (N, 6) species base stats, in ``STAT_IDS`` order.
        sp: (N, 6) stat points, 0..32 per stat.
        nature_numerators: (N, 6) or (6,) from :func:`nature_multipliers`.
        level: defaults to the regulation's level.

    Returns:
        (N, 6) int64 final stats.
    """
    lvl = reg.meta.level if level is None else level
    sp = np.asarray(sp, dtype=np.int64)
    if sp.ndim == 1:
        sp = sp[None, :]
    base = np.asarray(base_stats, dtype=np.int64)
    if base.ndim == 1:
        base = np.broadcast_to(base, sp.shape)
    num = np.asarray(nature_numerators, dtype=np.int64)
    if num.ndim == 1:
        num = np.broadcast_to(num, sp.shape)

    if reg.meta.uses_level_clause_mod:
        # General level-dependent form. `max(2*SP - 1, 0)` is how the mod maps stat points
        # onto the mainline formula's EV slot: the first point is worth 1, the rest 2.
        core = np.trunc((2 * base + reg.meta.fixed_iv + np.maximum(2 * sp - 1, 0)) * lvl / 100)
        core = core.astype(np.int64)
        hp = core[:, 0] + lvl + 10
        others = core[:, 1:] + 5
    else:
        if lvl != 50:
            raise ValueError(
                f"{reg.meta.format_id} has no Level Clause Mod, so Showdown uses its level-50 "
                f"closed form; level={lvl} is not representable."
            )
        hp = base[:, 0] + sp[:, 0] + 75
        others = base[:, 1:] + sp[:, 1:] + 20

    scaled = _trunc16(others * num[:, 1:])
    scaled = np.trunc(scaled / 100).astype(np.int64)
    return np.concatenate([hp[:, None], scaled], axis=1)


def stats_for(
    reg: Regulation,
    species_id: str,
    nature: str,
    sp: dict[str, int] | np.ndarray,
    level: int | None = None,
) -> dict[str, int]:
    """Scalar convenience wrapper. Prefer :func:`stats_from_sp` in hot paths."""
    species = reg.species.get(species_id)
    if species is None:
        raise KeyError(f"Unknown species: {species_id!r}")
    if isinstance(sp, dict):
        vec = np.array([[sp.get(s, 0) for s in STAT_IDS]], dtype=np.int64)
    else:
        vec = np.asarray(sp, dtype=np.int64).reshape(1, len(STAT_IDS))
    num = nature_multipliers(reg, [nature])
    out = stats_from_sp(reg, np.array(species.base_stats, dtype=np.int64), vec, num, level)[0]
    return dict(zip(STAT_IDS, (int(v) for v in out), strict=True))


# -- SP spread legality ------------------------------------------------------


def sp_is_legal(reg: Regulation, sp: dict[str, int] | np.ndarray) -> bool:
    vec = (
        np.array([sp.get(s, 0) for s in STAT_IDS])
        if isinstance(sp, dict)
        else np.asarray(sp).reshape(-1)
    )
    if vec.shape[0] != len(STAT_IDS):
        return False
    if (vec < 0).any() or (vec > reg.meta.sp_per_stat_max).any():
        return False
    return int(vec.sum()) <= reg.meta.sp_limit


def sp_legal_mask(reg: Regulation, sp: np.ndarray) -> np.ndarray:
    """(N, 6) -> (N,) bool. Both the per-stat cap and the total are enforced."""
    sp = np.asarray(sp, dtype=np.int64)
    within = (sp >= 0).all(axis=1) & (sp <= reg.meta.sp_per_stat_max).all(axis=1)
    return within & (sp.sum(axis=1) <= reg.meta.sp_limit)


# Showdown uses boostTable = [1, 1.5, 2, 2.5, 3, 3.5, 4] with
#   boost >= 0: floor(stat * table[boost])
#   boost <  0: floor(stat / table[-boost])
# Expressed as exact integer ratios indexed by stage + 6, so there is no float rounding
# to reason about. (The float form is in fact exact for these operands, but the integer
# form removes the question.)
BOOST_NUMERATOR = np.array([2, 2, 2, 2, 2, 2, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
BOOST_DENOMINATOR = np.array([8, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2], dtype=np.int64)


def apply_boost(stat: np.ndarray, stage: np.ndarray) -> np.ndarray:
    """Applies a -6..+6 stat stage the way Showdown does, as an exact integer ratio."""
    stat = np.asarray(stat, dtype=np.int64)
    idx = np.clip(np.asarray(stage, dtype=np.int64), -6, 6) + 6
    return stat * BOOST_NUMERATOR[idx] // BOOST_DENOMINATOR[idx]
