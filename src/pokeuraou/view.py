"""Turning a :class:`~pokeuraou.position.Position` into calculator inputs.

A ``Position`` is the wire format: JSON-shaped, one entry per Pokemon, SP possibly hidden.
A :class:`~pokeuraou.damage.Battler` is what the calculator wants: final stats as arrays
over the belief particles, everything else scalar.

This module is the only place that bridges the two, so the rule about which fields are
public (species, ability, item, nature, boosts, status) and which are uncertain (the SP
spread, and therefore every final stat and max HP) is stated once.
"""

from __future__ import annotations

import numpy as np

from .battler import Battler, FieldState
from .position import Pokemon, Position, Side
from .regulation import STAT_IDS, Regulation
from .stats import nature_multipliers, stats_from_sp

#: Memo for single-spread stat vectors, keyed by everything they depend on.
#:
#: A turn asks for the same Pokemon's stats once per hit, once per speed comparison and
#: once per residual, and the answer cannot change unless the key does: Mega Evolution
#: changes the species, Transform takes the override path and is not cached at all.
#: Entries are read-only arrays -- a shared mutable array would make an aliasing bug look
#: like a damage-calculation bug.
_STATS_MEMO: dict[tuple[object, ...], np.ndarray] = {}

#: Above this the memo is cleared wholesale. A single analysis touches a few dozen keys;
#: a long self-play run would otherwise grow it without bound.
STATS_MEMO_LIMIT = 4096


def clear_stats_memo() -> None:
    _STATS_MEMO.clear()


def _cached_stats(
    reg: Regulation, species_id: str, nature: str, sp: dict[str, int], level: int
) -> np.ndarray:
    key = (
        reg.meta.format_id,
        species_id,
        nature,
        level,
        tuple(sp.get(stat, 0) for stat in STAT_IDS),
    )
    hit = _STATS_MEMO.get(key)
    if hit is not None:
        return hit

    species = reg.species[species_id]
    values = np.array([[sp.get(stat, 0) for stat in STAT_IDS]], dtype=np.int64)
    stats = stats_from_sp(
        reg,
        np.array(species.base_stats, dtype=np.int64),
        values,
        nature_multipliers(reg, [nature]),
        level=level,
    )
    stats.flags.writeable = False
    if len(_STATS_MEMO) >= STATS_MEMO_LIMIT:
        _STATS_MEMO.clear()
    _STATS_MEMO[key] = stats
    return stats


def stats_array(
    reg: Regulation, mon: Pokemon, sp: np.ndarray | None = None
) -> np.ndarray:
    """(N, 6) final stats for a Pokemon.

    ``sp`` supplies candidate spreads for a Pokemon whose spread is hidden; when the
    spread is known (our own side, or a fully-informed test) the single known spread is
    used and N is 1.
    """
    species = reg.species.get(mon.species)
    if species is None:
        raise KeyError(f"Unknown species in position: {mon.species!r}")
    if sp is None:
        # Transform copies the target's stats except HP, so a transformed Pokemon's stats
        # do not follow from its own spread and the simulator's numbers are the only
        # correct ones.
        if mon.stats_override is not None and (mon.transformed or mon.sp is None):
            return np.array(
                [[mon.stats_override[k] for k in STAT_IDS]], dtype=np.int64
            )
        if mon.sp is None:
            raise ValueError(
                f"{mon.species} has a hidden SP spread; supply candidate spreads from the "
                "belief layer"
            )
        return _cached_stats(reg, species.id, mon.nature, mon.sp, mon.level)
    sp = np.asarray(sp, dtype=np.int64).reshape(-1, len(STAT_IDS))
    num = nature_multipliers(reg, [mon.nature] * sp.shape[0])
    return stats_from_sp(
        reg, np.array(species.base_stats, dtype=np.int64), sp, num, level=mon.level
    )


def battler(
    reg: Regulation,
    mon: Pokemon,
    *,
    sp: np.ndarray | None = None,
    hp_fraction: np.ndarray | float | None = None,
    hp_values: np.ndarray | None = None,
) -> Battler:
    """Builds a Battler for one Pokemon.

    ``hp_values`` is the exact per-particle HP the belief layer derived from the displayed
    percentage, and is the right input for a hidden spread: the display is a *band* of
    consistent HP values, so a single fraction cannot represent it. ``hp_fraction`` is the
    cruder version kept for callers that only have a ratio; it rounds, and the rounded
    value need not be one the game could have displayed. When the spread is known the
    position's absolute HP is used directly.
    """
    species = reg.species[mon.species]
    stats = stats_array(reg, mon, sp)
    maxhp = stats[:, 0].copy()
    if hp_values is not None:
        hp = np.asarray(hp_values, dtype=np.int64).reshape(-1)
        if hp.shape[0] != stats.shape[0]:
            raise ValueError(
                f"hp_values has {hp.shape[0]} entries for {stats.shape[0]} particles"
            )
    elif hp_fraction is None:
        hp = np.full(stats.shape[0], mon.hp, dtype=np.int64)
        if stats.shape[0] == 1:
            # Transform leaves HP alone, so the position's own maxhp is right even when the
            # other stats were copied.
            maxhp = np.array([mon.maxhp], dtype=np.int64)
    else:
        frac = np.asarray(hp_fraction, dtype=np.float64).reshape(-1)
        # Showdown reports the foe's HP as a rounded percentage, so the exact value is a
        # small interval; the nearest integer is the best point estimate and the belief
        # layer keeps the interval.
        hp = np.maximum(np.rint(maxhp * frac).astype(np.int64), 1 if mon.hp > 0 else 0)
    return Battler(
        species=species.id,
        # The live types when the producer recorded them; the species' types otherwise.
        types=mon.types or species.types,
        ability=mon.ability,
        item=mon.item,
        level=mon.level,
        stats=stats,
        hp=hp,
        maxhp=maxhp,
        boosts=dict(mon.boosts),
        status=mon.status,
        volatiles=frozenset(v.id for v in mon.volatiles),
        gender=mon.gender,
        ability_state=dict(mon.ability_state),
    )


def field_state(pos: Position, reg: Regulation) -> FieldState:
    del reg
    conditions = tuple(frozenset(c.id for c in side.side_conditions) for side in pos.sides)
    abilities = tuple(
        tuple(mon.ability for mon in side.active_pokemon() if mon is not None and not mon.fainted)
        for side in pos.sides
    )
    return FieldState(
        weather=pos.field.weather,
        terrain=pos.field.terrain,
        pseudo_weather=frozenset(p.id for p in pos.field.pseudo_weather),
        side_conditions=(conditions[0], conditions[1]),
        active_abilities=(abilities[0], abilities[1]),
        active_per_half=len(pos.sides[0].active),
    )


def active_battlers(reg: Regulation, side: Side) -> list[Battler | None]:
    """Battlers for a side's active slots, with None for empty or fainted slots."""
    out: list[Battler | None] = []
    for mon in side.active_pokemon():
        if mon is None or mon.fainted:
            out.append(None)
        else:
            out.append(battler(reg, mon))
    return out


def move_hits_multiple(reg: Regulation, move_id: str, n_live_foes: int) -> bool:
    """Whether Showdown would set ``move.spreadHit`` and apply the 0.75 modifier.

    A spread move that ends up hitting a single target does not take the modifier, so this
    depends on how many targets are actually there, not just on the move.
    """
    target = reg.moves[move_id].target
    if target == "allAdjacentFoes":
        return n_live_foes > 1
    if target == "allAdjacent":
        # Hits both foes and the ally, so two or more targets is the norm in doubles.
        return n_live_foes >= 1
    return False
