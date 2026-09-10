"""Turning a position into numbers, and the two properties that has to preserve.

This is the load-bearing half of the value function. A model can be retrained; an encoding
that quietly drops Speed or leaks the outcome produces a plausible-looking win probability
that means nothing, and nothing downstream would notice. So the module is written to make
two properties checkable rather than hoped for.

**The encoding is antisymmetric.** The game is zero-sum, so a position and the same
position viewed from the other side must give win probabilities summing to one. Rather than
hope a network learns that from data, :func:`mirror` swaps the side axis of every array,
and the model scores ``g(x) - g(mirror(x))``. The property then holds exactly, at
initialisation and after any amount of training, and :func:`Encoded.mirror` is what makes
it testable.

**The encoding is regulation-pinned.** Every categorical index is an offset into a
vocabulary built from the regulation dump, and the vocabulary's fingerprint is stored with
the weights. A model trained on M-B loaded against M-C is a silent disaster -- the same
integer means a different Pokemon -- so :class:`Vocabulary` refuses instead.

What is deliberately *not* here: anything the player could not see at decision time. The
recorded position carries both sides' spreads because self-play generated them, and that
is correct for a value function (the belief layer integrates our uncertainty outside it),
but nothing may come from later in the game. The features are read from one position object
and there is no access to the record that contains it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .regulation import STAT_IDS, Regulation
from .stats import nature_multipliers, stats_from_sp

#: Volatiles the resolver can set, as an ordered vocabulary. Derived from the volatile ids
#: actually present across the generated games plus the ones ``resolve.py`` adds by name;
#: anything else lands in the ``other`` bucket, and :attr:`Encoded.unknown_volatiles`
#: counts it so the omission shows up as a number instead of as silence.
VOLATILES = (
    "stall",
    "partiallytrapped",
    "choicelock",
    "mustrecharge",
    "twoturnmove",
    "roost",
    "encore",
    "yawn",
    "perishsong",
    "imprison",
    "leechseed",
    "curse",
    "substitute",
    "stockpile",
    "taunt",
    "endure",
    "healblock",
    "disable",
    "confusion",
    "flashfire",
    "helpinghand",
    "followme",
    "protect",
    "pendingselfswitch",
    "pendingforceswitch",
    # Found by the `other`-bucket counter once the opponent pool became real tournament
    # teams. Neither appeared in the archetype-pool data at all, because that pool had no
    # Sneasler -- and Sneasler is in 26% of tournament teams. Unburden is not decoration:
    # it doubles Speed while the item is gone, so a value function that cannot see it is
    # guessing turn order for a quarter of the field.
    "unburden",
    "lockedmove",
)

#: Status conditions, in the order Showdown names them. Index 0 is "no status".
STATUSES = ("brn", "par", "slp", "frz", "psn", "tox")

BOOST_IDS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

WEATHERS = (
    "sunnyday",
    "raindance",
    "sandstorm",
    "snowscape",
    "hail",
    "desolateland",
    "primordialsea",
    "deltastream",
)
TERRAINS = ("electricterrain", "grassyterrain", "mistyterrain", "psychicterrain")
PSEUDO_WEATHERS = ("trickroom", "gravity", "magicroom", "wonderroom", "mudsport", "watersport")

SIDE_CONDITIONS = (
    "tailwind",
    "reflect",
    "lightscreen",
    "auroraveil",
    "safeguard",
    "mist",
    "luckychant",
    "stealthrock",
    "spikes",
    "toxicspikes",
    "stickyweb",
)
SLOT_CONDITIONS = ("wideguard", "quickguard", "craftyshield", "matblock", "healingwish", "lunardance")

#: Stats are divided by this before they reach the network. Level 50 with 32 SP tops out
#: near 200 for a non-HP stat, so the scaled features sit in roughly [0.2, 1.4] -- the
#: point is only that the scale is fixed and stated, not fitted to a particular dataset.
STAT_SCALE = 200.0
HP_SCALE = 250.0

#: Turn number is clipped here before scaling. Games past this are rare and the network
#: has no reason to extrapolate; the clip is reported through the feature name list.
TURN_CLIP = 40.0


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """The integer meaning of every categorical feature, pinned to one regulation.

    Index 0 is reserved for "absent or unknown" in every table, so an item slot with no
    item and a species the dump does not know both encode as 0 rather than as some
    arbitrary neighbour.
    """

    format_id: str
    species: dict[str, int]
    abilities: dict[str, int]
    items: dict[str, int]
    moves: dict[str, int]
    types: tuple[str, ...]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "species": len(self.species) + 1,
            "ability": len(self.abilities) + 1,
            "item": len(self.items) + 1,
            "move": len(self.moves) + 1,
        }

    def fingerprint(self) -> str:
        """A digest of every vocabulary, stored with the weights.

        Two regulations that happen to have the same species *count* still map different
        ids to the same integer, so the check has to be over the mapping and not its size.
        """
        digest = hashlib.sha256()
        digest.update(self.format_id.encode())
        for name, table in (
            ("species", self.species),
            ("abilities", self.abilities),
            ("items", self.items),
            ("moves", self.moves),
        ):
            digest.update(name.encode())
            for key, index in sorted(table.items()):
                digest.update(f"{key}:{index}".encode())
        digest.update(",".join(self.types).encode())
        digest.update(",".join(VOLATILES).encode())
        return digest.hexdigest()[:16]


def build_vocabulary(reg: Regulation) -> Vocabulary:
    """Vocabularies straight from the regulation dump, in a stable order.

    Sorted by id, not by usage or by dump order, so rebuilding it from the same dump gives
    the same integers -- a model's weights are meaningless the moment that stops being
    true.
    """
    species = {sid: i + 1 for i, sid in enumerate(sorted(reg.species))}
    abilities = {aid: i + 1 for i, aid in enumerate(sorted(reg.abilities))}
    items = {iid: i + 1 for i, iid in enumerate(sorted(reg.items))}
    moves = {mid: i + 1 for i, mid in enumerate(sorted(reg.moves))}
    types = tuple(sorted({t for sp in reg.species.values() for t in sp.types}))
    return Vocabulary(
        format_id=reg.meta.format_id,
        species=species,
        abilities=abilities,
        items=items,
        moves=moves,
        types=types,
    )


@dataclass(slots=True)
class Encoded:
    """One batch of positions as arrays, with the side axis kept explicit.

    Every per-Pokemon array has shape ``(B, 2, M, ...)`` where axis 1 is the side and M is
    the number of Pokemon a side brings to the field. Keeping the side axis rather than
    concatenating the two sides is what makes :meth:`mirror` a single flip, and therefore
    what makes the model's antisymmetry free.
    """

    species: np.ndarray  # (B, 2, M) int64
    ability: np.ndarray  # (B, 2, M) int64
    item: np.ndarray  # (B, 2, M) int64
    moves: np.ndarray  # (B, 2, M, 4) int64
    mon: np.ndarray  # (B, 2, M, F) float32
    mask: np.ndarray  # (B, 2, M) float32, 1 where a Pokemon is present
    side: np.ndarray  # (B, 2, S) float32
    field: np.ndarray  # (B, G) float32, side-independent only
    #: Volatiles encountered that :data:`VOLATILES` does not name, by id.
    unknown_volatiles: dict[str, int]

    def __len__(self) -> int:
        return int(self.species.shape[0])

    def mirror(self) -> Encoded:
        """The same positions with the two sides exchanged.

        ``field`` is untouched by construction: anything that belongs to one side lives in
        ``side``, so a mirrored position is a flip of axis 1 and nothing else. That is the
        whole reason the side axis is not folded into the feature vector.
        """
        return Encoded(
            species=self.species[:, ::-1],
            ability=self.ability[:, ::-1],
            item=self.item[:, ::-1],
            moves=self.moves[:, ::-1],
            mon=self.mon[:, ::-1],
            mask=self.mask[:, ::-1],
            side=self.side[:, ::-1],
            field=self.field,
            unknown_volatiles=dict(self.unknown_volatiles),
        )

    def slice(self, index: np.ndarray) -> Encoded:
        return Encoded(
            species=self.species[index],
            ability=self.ability[index],
            item=self.item[index],
            moves=self.moves[index],
            mon=self.mon[index],
            mask=self.mask[index],
            side=self.side[index],
            field=self.field[index],
            unknown_volatiles=dict(self.unknown_volatiles),
        )


def mon_feature_names(vocab: Vocabulary) -> tuple[str, ...]:
    """Every per-Pokemon numeric feature, in order.

    Exists so the width is derived from one list instead of from a hand-counted constant,
    and so a feature can be found by name when a trained model behaves oddly.
    """
    names = ["hp_fraction", "maxhp_scaled"]
    names += [f"stat_{s}" for s in STAT_IDS]
    names += [f"sp_{s}" for s in STAT_IDS]
    names += [f"boost_{b}" for b in BOOST_IDS]
    names += ["status_none"] + [f"status_{s}" for s in STATUSES]
    names += [
        "is_active",
        "active_slot_0",
        "active_slot_1",
        "fainted",
        "is_mega",
        "can_mega",
        "trapped",
        "newly_switched",
    ]
    names += [f"type_{t}" for t in vocab.types]
    names += [f"pp_fraction_{i}" for i in range(4)]
    names += [f"move_disabled_{i}" for i in range(4)]
    names += [f"volatile_{v}" for v in VOLATILES] + ["volatile_other"]
    return tuple(names)


def side_feature_names() -> tuple[str, ...]:
    names = [f"side_{c}" for c in SIDE_CONDITIONS]
    names += ["mega_used", "mega_available", "alive_fraction", "team_hp_fraction"]
    for slot in range(2):
        names += [f"slot{slot}_{c}" for c in SLOT_CONDITIONS]
    return tuple(names)


def field_feature_names() -> tuple[str, ...]:
    names = ["weather_none"] + [f"weather_{w}" for w in WEATHERS] + ["weather_duration"]
    names += ["terrain_none"] + [f"terrain_{t}" for t in TERRAINS] + ["terrain_duration"]
    names += [f"pseudo_{p}" for p in PSEUDO_WEATHERS]
    names += ["turn_scaled", "turn_is_first"]
    return tuple(names)


class _StatCache:
    """Live stats from SP, memoised.

    The network is given the actual stat line rather than left to infer it from the species
    embedding and the SP features. Speed decides turn order, which decides most of what
    happens next, and there is no reason to make a model rediscover an exact formula that
    is already tested against the game itself.
    """

    def __init__(self, reg: Regulation) -> None:
        self._reg = reg
        self._cache: dict[tuple[str, str, tuple[int, ...]], np.ndarray] = {}

    def get(self, species_id: str, nature: str, sp: tuple[int, ...]) -> np.ndarray:
        key = (species_id, nature, sp)
        found = self._cache.get(key)
        if found is None:
            species = self._reg.species.get(species_id)
            if species is None:
                found = np.zeros(6, dtype=np.float64)
            else:
                found = stats_from_sp(
                    self._reg,
                    np.array(species.base_stats, dtype=np.int64),
                    np.array([sp], dtype=np.int64),
                    nature_multipliers(self._reg, [nature]),
                    level=self._reg.meta.level,
                )[0].astype(np.float64)
            self._cache[key] = found
        return found


class Encoder:
    """Encodes positions, in the JSON form the self-play records carry."""

    def __init__(self, reg: Regulation, vocab: Vocabulary | None = None) -> None:
        if vocab is not None and vocab.format_id != reg.meta.format_id:
            raise ValueError(
                f"vocabulary is for {vocab.format_id} but the regulation is "
                f"{reg.meta.format_id}; the same integer would mean a different Pokemon"
            )
        self.reg = reg
        self.vocab = vocab or build_vocabulary(reg)
        self.mons_per_side = reg.meta.picked_team_size
        self.mon_names = mon_feature_names(self.vocab)
        self.side_names = side_feature_names()
        self.field_names = field_feature_names()
        self._type_index = {t: i for i, t in enumerate(self.vocab.types)}
        self._volatile_index = {v: i for i, v in enumerate(VOLATILES)}
        self._stats = _StatCache(reg)

    @property
    def widths(self) -> dict[str, int]:
        return {
            "mon": len(self.mon_names),
            "side": len(self.side_names),
            "field": len(self.field_names),
        }

    def encode(self, positions: list[dict[str, Any]]) -> Encoded:
        n = len(positions)
        m = self.mons_per_side
        species = np.zeros((n, 2, m), dtype=np.int64)
        ability = np.zeros((n, 2, m), dtype=np.int64)
        item = np.zeros((n, 2, m), dtype=np.int64)
        moves = np.zeros((n, 2, m, 4), dtype=np.int64)
        mon = np.zeros((n, 2, m, len(self.mon_names)), dtype=np.float32)
        mask = np.zeros((n, 2, m), dtype=np.float32)
        side = np.zeros((n, 2, len(self.side_names)), dtype=np.float32)
        field = np.zeros((n, len(self.field_names)), dtype=np.float32)
        unknown: dict[str, int] = {}

        for b, position in enumerate(positions):
            self._encode_field(field[b], position)
            for s, side_json in enumerate(position["sides"]):
                self._encode_side(side[b, s], side_json)
                for p, mon_json in enumerate(side_json["pokemon"][:m]):
                    mask[b, s, p] = 1.0
                    species[b, s, p] = self.vocab.species.get(mon_json["species"], 0)
                    ability[b, s, p] = self.vocab.abilities.get(mon_json.get("ability") or "", 0)
                    item[b, s, p] = self.vocab.items.get(mon_json.get("item") or "", 0)
                    for k, slot in enumerate(mon_json.get("moves", [])[:4]):
                        moves[b, s, p, k] = self.vocab.moves.get(slot["id"], 0)
                    self._encode_mon(mon[b, s, p], mon_json, side_json, unknown)
        return Encoded(
            species=species,
            ability=ability,
            item=item,
            moves=moves,
            mon=mon,
            mask=mask,
            side=side,
            field=field,
            unknown_volatiles=unknown,
        )

    # ------------------------------------------------------------------ pieces --

    def _encode_field(self, out: np.ndarray, position: dict[str, Any]) -> None:
        names = self.field_names
        base = 0
        weather = position["field"].get("weather")
        wid = (weather or {}).get("id") if isinstance(weather, dict) else weather
        out[base] = 1.0 if not wid else 0.0
        for i, name in enumerate(WEATHERS):
            out[base + 1 + i] = 1.0 if wid == name else 0.0
        duration = (weather or {}).get("duration") if isinstance(weather, dict) else None
        out[base + 1 + len(WEATHERS)] = float(duration or 0) / 8.0
        base += 2 + len(WEATHERS)

        terrain = position["field"].get("terrain")
        tid = (terrain or {}).get("id") if isinstance(terrain, dict) else terrain
        out[base] = 1.0 if not tid else 0.0
        for i, name in enumerate(TERRAINS):
            out[base + 1 + i] = 1.0 if tid == name else 0.0
        duration = (terrain or {}).get("duration") if isinstance(terrain, dict) else None
        out[base + 1 + len(TERRAINS)] = float(duration or 0) / 8.0
        base += 2 + len(TERRAINS)

        present = {
            (p.get("id") if isinstance(p, dict) else p)
            for p in position["field"].get("pseudoWeather", [])
        }
        for i, name in enumerate(PSEUDO_WEATHERS):
            out[base + i] = 1.0 if name in present else 0.0
        base += len(PSEUDO_WEATHERS)

        turn = float(position.get("turn", 1))
        out[base] = min(turn, TURN_CLIP) / TURN_CLIP
        out[base + 1] = 1.0 if turn <= 1 else 0.0
        assert base + 2 == len(names)

    def _encode_side(self, out: np.ndarray, side_json: dict[str, Any]) -> None:
        conditions = {c["id"] for c in side_json.get("sideConditions", [])}
        base = 0
        for i, name in enumerate(SIDE_CONDITIONS):
            out[base + i] = 1.0 if name in conditions else 0.0
        base += len(SIDE_CONDITIONS)

        mons = side_json["pokemon"]
        alive = [p for p in mons if not p.get("fainted")]
        out[base] = 1.0 if side_json.get("megaUsed") else 0.0
        # Mega is a once-per-battle side resource, so "still has it" is a real feature of
        # the side and not of any one Pokemon.
        out[base + 1] = (
            1.0
            if not side_json.get("megaUsed") and side_json.get("megaCapableSlots")
            else 0.0
        )
        out[base + 2] = len(alive) / max(len(mons), 1)
        total = sum(p["maxhp"] for p in mons) or 1
        out[base + 3] = sum(p["hp"] for p in mons) / total
        base += 4

        slots = side_json.get("slotConditions") or []
        for slot in range(2):
            ids = {c["id"] for c in slots[slot]} if slot < len(slots) else set()
            for i, name in enumerate(SLOT_CONDITIONS):
                out[base + i] = 1.0 if name in ids else 0.0
            base += len(SLOT_CONDITIONS)
        assert base == len(self.side_names)

    def _encode_mon(
        self,
        out: np.ndarray,
        mon: dict[str, Any],
        side_json: dict[str, Any],
        unknown: dict[str, int],
    ) -> None:
        maxhp = float(mon["maxhp"]) or 1.0
        base = 0
        out[base] = float(mon["hp"]) / maxhp
        out[base + 1] = maxhp / HP_SCALE
        base += 2

        sp = mon.get("sp") or {}
        sp_tuple = tuple(int(sp.get(s, 0)) for s in STAT_IDS)
        stats = self._stats.get(mon["species"], mon.get("nature", "Serious"), sp_tuple)
        out[base : base + 6] = stats / STAT_SCALE
        base += 6
        out[base : base + 6] = np.array(sp_tuple, dtype=np.float32) / 32.0
        base += 6

        boosts = mon.get("boosts") or {}
        for i, boost in enumerate(BOOST_IDS):
            out[base + i] = float(boosts.get(boost, 0)) / 6.0
        base += len(BOOST_IDS)

        status = mon.get("status")
        out[base] = 1.0 if not status else 0.0
        for i, name in enumerate(STATUSES):
            out[base + 1 + i] = 1.0 if status == name else 0.0
        base += 1 + len(STATUSES)

        active_index = mon.get("activeIndex")
        capable = set(side_json.get("megaCapableSlots") or [])
        out[base + 0] = 1.0 if active_index is not None else 0.0
        out[base + 1] = 1.0 if active_index == 0 else 0.0
        out[base + 2] = 1.0 if active_index == 1 else 0.0
        out[base + 3] = 1.0 if mon.get("fainted") else 0.0
        out[base + 4] = 1.0 if mon.get("isMega") else 0.0
        out[base + 5] = (
            1.0
            if mon["slot"] in capable and not side_json.get("megaUsed") and not mon.get("isMega")
            else 0.0
        )
        out[base + 6] = 1.0 if mon.get("trapped") else 0.0
        out[base + 7] = 1.0 if mon.get("newlySwitched") else 0.0
        base += 8

        for kind in mon.get("types", []):
            index = self._type_index.get(kind)
            if index is not None:
                out[base + index] = 1.0
        base += len(self.vocab.types)

        move_slots = mon.get("moves", [])
        for i in range(4):
            if i < len(move_slots):
                slot = move_slots[i]
                out[base + i] = float(slot["pp"]) / max(float(slot["maxpp"]), 1.0)
                out[base + 4 + i] = 1.0 if slot.get("disabled") else 0.0
        base += 8

        for volatile in mon.get("volatiles", []):
            vid = volatile["id"] if isinstance(volatile, dict) else volatile
            index = self._volatile_index.get(vid)
            if index is None:
                out[base + len(VOLATILES)] = 1.0
                unknown[vid] = unknown.get(vid, 0) + 1
            else:
                out[base + index] = 1.0
        for vid in mon.get("unmodelledVolatiles", []):
            out[base + len(VOLATILES)] = 1.0
            unknown[vid] = unknown.get(vid, 0) + 1
        base += len(VOLATILES) + 1
        assert base == len(self.mon_names)


__all__ = [
    "BOOST_IDS",
    "SIDE_CONDITIONS",
    "STATUSES",
    "VOLATILES",
    "Encoded",
    "Encoder",
    "Vocabulary",
    "build_vocabulary",
    "field_feature_names",
    "mon_feature_names",
    "side_feature_names",
]
