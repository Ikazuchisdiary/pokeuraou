"""Battle-time views of a Pokemon and the field.

Split out from :mod:`pokeuraou.damage` so that :mod:`pokeuraou.moveinfo` -- which needs
these types to compute variable base power -- does not import the calculator that in turn
needs it.

The batching rule lives here: ``stats``, ``hp`` and ``maxhp`` carry a leading particle
axis because they follow from the SP spread, which is the hidden parameter. Species,
ability, item, nature, boosts, status and volatiles are public information under Champions
Open Team Sheets, so they are scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .regulation import STAT_INDEX
from .stats import apply_boost

#: Number of damage rolls Showdown uses: 100%, 99%, ..., 85%.
N_ROLLS = 16

@dataclass(slots=True)
class Battler:
    """One Pokemon on the field.

    ``stats``, ``hp`` and ``maxhp`` are batched over a leading particle axis because they
    depend on the SP spread, which is the hidden parameter. Everything else -- species,
    ability, item, status, boosts, volatiles -- is public information under Champions Open
    Team Sheets and therefore scalar.
    """

    species: str
    types: tuple[str, ...]
    ability: str
    item: str | None
    level: int
    #: (N, 6) final stats in STAT_IDS order.
    stats: np.ndarray
    #: (N,) current HP and (N,) max HP.
    hp: np.ndarray
    maxhp: np.ndarray
    boosts: dict[str, int] = field(default_factory=dict)
    status: str | None = None
    volatiles: frozenset[str] = frozenset()
    gender: str = "N"
    #: The ability's own state. Read for the abilities that record "already fired" there
    #: rather than in a volatile (Protean, Libero).
    ability_state: dict[str, object] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.stats.shape[0])

    def stat(self, key: str, *, ignore_boost: bool = False) -> np.ndarray:
        """A boosted stat, using Showdown's exact integer boost ratios."""
        raw = self.stats[:, STAT_INDEX[key]]
        if ignore_boost:
            return raw
        stage = self.boosts.get(key, 0)
        if stage == 0:
            return raw
        return apply_boost(raw, np.array(stage))

    @property
    def at_full_hp(self) -> np.ndarray:
        return self.hp >= self.maxhp


@dataclass(slots=True)
class FieldState:
    weather: str | None = None
    terrain: str | None = None
    pseudo_weather: frozenset[str] = frozenset()
    #: Side conditions per side index.
    side_conditions: tuple[frozenset[str], frozenset[str]] = (frozenset(), frozenset())
    #: Abilities of the active Pokemon on each side. Needed for effects that reach across
    #: slots: Friend Guard reduces damage to its ally, and the Aura abilities boost a type
    #: for every Pokemon on the field regardless of side.
    active_abilities: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    active_per_half: int = 2

    @property
    def trick_room(self) -> bool:
        return "trickroom" in self.pseudo_weather


@dataclass(slots=True)
class DamageResult:
    """Damage for every roll and every particle, plus how it was derived."""

    #: (N, 16) int64 damage; column 0 is the 100% roll, column 15 the 85% roll.
    rolls: np.ndarray
    #: Type effectiveness multiplier (0 means immune).
    effectiveness: float
    #: Showdown's clamped log2 effectiveness.
    type_mod: int
    #: True when the move cannot damage the target at all.
    immune: bool
    #: Effects that contributed, by event slot, for explanation and divergence reporting.
    applied: dict[str, tuple[tuple[str, int], ...]] = field(default_factory=dict)
    #: Named effects present on either Pokemon that this module does not model.
    unmodelled: tuple[str, ...] = ()

    @property
    def min(self) -> np.ndarray:
        return self.rolls[:, -1]

    @property
    def max(self) -> np.ndarray:
        return self.rolls[:, 0]

    def ko_probability(self, target_hp: np.ndarray) -> np.ndarray:
        """(N,) fraction of the 16 rolls that reach the target's HP.

        Damage rolls are uniform over the 16 values, so this is exact for a single hit
        rather than an estimate.
        """
        return (self.rolls >= np.asarray(target_hp, dtype=np.int64)[:, None]).mean(axis=1)


