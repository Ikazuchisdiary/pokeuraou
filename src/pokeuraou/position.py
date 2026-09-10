"""Python mirror of the canonical position representation.

The shape is defined in ``packages/sim-bridge/src/position.ts``; this module reads and
writes exactly that JSON. Field names stay camelCase on the wire and become snake_case
here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .regulation import BOOST_IDS, STAT_IDS


@dataclass(slots=True)
class Effect:
    id: str
    duration: int | None = None
    layers: int | None = None
    counter: int | None = None
    source_slot: str | None = None
    move: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_json(d: dict[str, Any]) -> Effect:
        return Effect(
            id=d["id"],
            duration=d.get("duration"),
            layers=d.get("layers"),
            counter=d.get("counter"),
            source_slot=d.get("sourceSlot"),
            move=d.get("move"),
            extra=dict(d.get("extra", {})),
        )

    def copy(self) -> Effect:
        return Effect(
            id=self.id,
            duration=self.duration,
            layers=self.layers,
            counter=self.counter,
            source_slot=self.source_slot,
            move=self.move,
            extra=dict(self.extra) if self.extra else {},
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id}
        for key, value in (
            ("duration", self.duration),
            ("layers", self.layers),
            ("counter", self.counter),
            ("sourceSlot", self.source_slot),
            ("move", self.move),
        ):
            if value is not None:
                out[key] = value
        if self.extra:
            out["extra"] = self.extra
        return out


@dataclass(slots=True)
class MoveSlot:
    id: str
    pp: int
    maxpp: int
    disabled: bool = False
    #: Used since coming in. Last Resort reads it, and it resets on switch-in, so spent PP
    #: is not a substitute.
    used: bool = False

    @staticmethod
    def from_json(d: dict[str, Any]) -> MoveSlot:
        return MoveSlot(
            id=d["id"],
            pp=d["pp"],
            maxpp=d["maxpp"],
            disabled=bool(d.get("disabled")),
            used=bool(d.get("used")),
        )

    def copy(self) -> MoveSlot:
        return MoveSlot(
            id=self.id, pp=self.pp, maxpp=self.maxpp, disabled=self.disabled, used=self.used
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "pp": self.pp,
            "maxpp": self.maxpp,
            "disabled": self.disabled,
            "used": self.used,
        }

    @property
    def usable(self) -> bool:
        return self.pp > 0 and not self.disabled


@dataclass(slots=True)
class Pokemon:
    slot: int
    species: str
    base_species: str
    #: Types as they are now. Empty means "use the species' types"; they differ once
    #: Protean, Libero, Mimicry, Soak or Reflect Type has fired.
    types: tuple[str, ...]
    ability: str
    nature: str
    moves: list[MoveSlot]
    hp: int
    maxhp: int
    level: int = 50
    gender: str = "N"
    item: str | None = None
    base_item: str | None = None
    #: SP spread, or None when hidden (the opponent's, before the belief layer fills it).
    sp: dict[str, int] | None = None
    fainted: bool = False
    status: str | None = None
    status_duration: int | None = None
    status_counter: int | None = None
    boosts: dict[str, int] = field(default_factory=dict)
    volatiles: list[Effect] = field(default_factory=list)
    unmodelled_volatiles: list[str] = field(default_factory=list)
    is_mega: bool = False
    active_index: int | None = None
    trapped: bool = False
    newly_switched: bool = False
    last_move: str | None = None
    locked_move: str | None = None
    #: Whether last turn's move failed. Stomping Tantrum and Temper Flare read it, and it
    #: refers to a turn that is already over, so it cannot be derived.
    move_last_turn_failed: bool = False
    #: Times this Pokemon has been hit, which is Rage Fist's base power.
    times_attacked: int = 0
    #: Move actions taken since coming in -- Showdown's `activeMoveActions`, reset by a
    #: switch and incremented per move. Fake Out and its two relatives refuse to work once
    #: this is above one, which is the whole of "only on your first turn out".
    active_move_actions: int = 0
    #: The ability's own state. Protean and Libero record here that they have already
    #: retyped their user this switch-in, so it is not derivable from the volatiles.
    ability_state: dict[str, Any] = field(default_factory=dict)
    #: Final stats as the simulator holds them, when the producer knows them. Authoritative
    #: for a transformed Pokemon, whose stats do not follow from its own SP spread.
    stats_override: dict[str, int] | None = None
    transformed: bool = False

    @property
    def is_active(self) -> bool:
        return self.active_index is not None

    @property
    def hp_fraction(self) -> float:
        return 0.0 if self.maxhp <= 0 else self.hp / self.maxhp

    def volatile(self, vid: str) -> Effect | None:
        for v in self.volatiles:
            if v.id == vid:
                return v
        return None

    def has_volatile(self, vid: str) -> bool:
        return self.volatile(vid) is not None

    def move_slot(self, move_id: str) -> MoveSlot | None:
        for m in self.moves:
            if m.id == move_id:
                return m
        return None

    def boost(self, stat: str) -> int:
        return self.boosts.get(stat, 0)

    def copy(self) -> Pokemon:
        """A deep copy, by hand.

        ``copy.deepcopy`` costs about 390 microseconds on a full position, and the resolver
        makes one per branch per action -- which put a single turn at 21 seconds. Copying
        the known fields explicitly is roughly twenty times faster, and the shape is fixed
        so there is nothing to miss.
        """
        return Pokemon(
            slot=self.slot,
            species=self.species,
            base_species=self.base_species,
            types=self.types,
            ability=self.ability,
            nature=self.nature,
            moves=[m.copy() for m in self.moves],
            hp=self.hp,
            maxhp=self.maxhp,
            level=self.level,
            gender=self.gender,
            item=self.item,
            base_item=self.base_item,
            sp=dict(self.sp) if self.sp is not None else None,
            fainted=self.fainted,
            status=self.status,
            status_duration=self.status_duration,
            status_counter=self.status_counter,
            boosts=dict(self.boosts),
            volatiles=[v.copy() for v in self.volatiles],
            unmodelled_volatiles=list(self.unmodelled_volatiles),
            is_mega=self.is_mega,
            active_index=self.active_index,
            trapped=self.trapped,
            newly_switched=self.newly_switched,
            last_move=self.last_move,
            locked_move=self.locked_move,
            move_last_turn_failed=self.move_last_turn_failed,
            times_attacked=self.times_attacked,
            active_move_actions=self.active_move_actions,
            ability_state=dict(self.ability_state),
            stats_override=dict(self.stats_override) if self.stats_override is not None else None,
            transformed=self.transformed,
        )

    @staticmethod
    def from_json(d: dict[str, Any]) -> Pokemon:
        return Pokemon(
            slot=d["slot"],
            species=d["species"],
            base_species=d.get("baseSpecies", d["species"]),
            types=tuple(d.get("types", ())),
            ability=d["ability"],
            nature=d.get("nature", "Serious"),
            moves=[MoveSlot.from_json(m) for m in d["moves"]],
            hp=d["hp"],
            maxhp=d["maxhp"],
            level=d.get("level", 50),
            gender=d.get("gender", "N"),
            item=d.get("item"),
            base_item=d.get("baseItem"),
            sp=dict(d["sp"]) if d.get("sp") is not None else None,
            fainted=bool(d.get("fainted")),
            status=d.get("status"),
            status_duration=d.get("statusDuration"),
            status_counter=d.get("statusCounter"),
            boosts={k: v for k, v in d.get("boosts", {}).items() if v},
            volatiles=[Effect.from_json(v) for v in d.get("volatiles", [])],
            unmodelled_volatiles=list(d.get("unmodelledVolatiles", [])),
            is_mega=bool(d.get("isMega")),
            active_index=d.get("activeIndex"),
            trapped=bool(d.get("trapped")),
            newly_switched=bool(d.get("newlySwitched")),
            last_move=d.get("lastMove"),
            locked_move=d.get("lockedMove"),
            move_last_turn_failed=bool(d.get("moveLastTurnFailed")),
            times_attacked=int(d.get("timesAttacked", 0)),
            active_move_actions=int(d.get("activeMoveActions", 0)),
            ability_state=dict(d.get("abilityState", {})),
            stats_override=dict(d["statsOverride"]) if d.get("statsOverride") else None,
            transformed=bool(d.get("transformed")),
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "slot": self.slot,
            "species": self.species,
            "baseSpecies": self.base_species,
            "types": list(self.types),
            "level": self.level,
            "gender": self.gender,
            "ability": self.ability,
            "item": self.item,
            "baseItem": self.base_item,
            "nature": self.nature,
            "sp": dict(self.sp) if self.sp is not None else None,
            "moves": [m.to_json() for m in self.moves],
            "hp": self.hp,
            "maxhp": self.maxhp,
            "fainted": self.fainted,
            "status": self.status,
            "boosts": dict(self.boosts),
            "volatiles": [v.to_json() for v in self.volatiles],
            "unmodelledVolatiles": list(self.unmodelled_volatiles),
            "isMega": self.is_mega,
            "activeIndex": self.active_index,
            "trapped": self.trapped,
            "newlySwitched": self.newly_switched,
            "lastMove": self.last_move,
            "lockedMove": self.locked_move,
            "moveLastTurnFailed": self.move_last_turn_failed,
            "timesAttacked": self.times_attacked,
            "activeMoveActions": self.active_move_actions,
            "abilityState": dict(self.ability_state),
            "transformed": self.transformed,
        }
        if self.stats_override is not None:
            out["statsOverride"] = dict(self.stats_override)
        if self.status_duration is not None:
            out["statusDuration"] = self.status_duration
        if self.status_counter is not None:
            out["statusCounter"] = self.status_counter
        return out


@dataclass(slots=True)
class Side:
    id: str
    name: str
    active: list[int | None]
    pokemon: list[Pokemon]
    side_conditions: list[Effect] = field(default_factory=list)
    slot_conditions: list[list[Effect]] = field(default_factory=list)
    #: Mega Evolution is a once-per-battle side resource.
    mega_used: bool = False
    #: Party slots holding a mega stone that matches their species.
    mega_capable_slots: list[int] = field(default_factory=list)

    def active_pokemon(self) -> list[Pokemon | None]:
        return [None if s is None else self.pokemon[s] for s in self.active]

    def bench(self) -> list[Pokemon]:
        return [p for p in self.pokemon if not p.is_active and not p.fainted]

    def side_condition(self, cid: str) -> Effect | None:
        for c in self.side_conditions:
            if c.id == cid:
                return c
        return None

    def has_side_condition(self, cid: str) -> bool:
        return self.side_condition(cid) is not None

    def copy(self) -> Side:
        return Side(
            id=self.id,
            name=self.name,
            active=list(self.active),
            pokemon=[p.copy() for p in self.pokemon],
            side_conditions=[c.copy() for c in self.side_conditions],
            slot_conditions=[[c.copy() for c in group] for group in self.slot_conditions],
            mega_used=self.mega_used,
            mega_capable_slots=list(self.mega_capable_slots),
        )

    @staticmethod
    def from_json(d: dict[str, Any]) -> Side:
        return Side(
            id=d["id"],
            name=d.get("name", d["id"]),
            active=list(d["active"]),
            pokemon=[Pokemon.from_json(p) for p in d["pokemon"]],
            side_conditions=[Effect.from_json(c) for c in d.get("sideConditions", [])],
            slot_conditions=[
                [Effect.from_json(c) for c in group] for group in d.get("slotConditions", [])
            ],
            mega_used=bool(d.get("megaUsed")),
            mega_capable_slots=list(d.get("megaCapableSlots", [])),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "active": list(self.active),
            "pokemon": [p.to_json() for p in self.pokemon],
            "sideConditions": [c.to_json() for c in self.side_conditions],
            "slotConditions": [[c.to_json() for c in g] for g in self.slot_conditions],
            "megaUsed": self.mega_used,
            "megaCapableSlots": list(self.mega_capable_slots),
        }


@dataclass(slots=True)
class Field:
    weather: str | None = None
    weather_duration: int | None = None
    terrain: str | None = None
    terrain_duration: int | None = None
    pseudo_weather: list[Effect] = field(default_factory=list)

    def has_pseudo_weather(self, pid: str) -> bool:
        return any(p.id == pid for p in self.pseudo_weather)

    @property
    def trick_room(self) -> bool:
        return self.has_pseudo_weather("trickroom")

    def copy(self) -> Field:
        return Field(
            weather=self.weather,
            weather_duration=self.weather_duration,
            terrain=self.terrain,
            terrain_duration=self.terrain_duration,
            pseudo_weather=[p.copy() for p in self.pseudo_weather],
        )

    @staticmethod
    def from_json(d: dict[str, Any]) -> Field:
        return Field(
            weather=d.get("weather"),
            weather_duration=d.get("weatherDuration"),
            terrain=d.get("terrain"),
            terrain_duration=d.get("terrainDuration"),
            pseudo_weather=[Effect.from_json(p) for p in d.get("pseudoWeather", [])],
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "weather": self.weather,
            "terrain": self.terrain,
            "pseudoWeather": [p.to_json() for p in self.pseudo_weather],
        }
        if self.weather_duration is not None:
            out["weatherDuration"] = self.weather_duration
        if self.terrain_duration is not None:
            out["terrainDuration"] = self.terrain_duration
        return out


def _swap_source_slot(source_slot: str | None) -> str | None:
    """Flips the side in a ``source_slot``, in either of the two encodings in use."""
    if not source_slot:
        return source_slot
    if len(source_slot) >= 2 and source_slot[0] in "01" and source_slot[1].isdigit():
        return ("1" if source_slot[0] == "0" else "0") + source_slot[1:]
    if len(source_slot) >= 3 and source_slot[0] == "p" and source_slot[1] in "12":
        return "p" + ("2" if source_slot[1] == "1" else "1") + source_slot[2:]
    return source_slot


@dataclass(slots=True)
class Position:
    format: str
    sides: list[Side]
    turn: int = 1
    field: Field = field(default_factory=Field)
    request_state: str = "move"
    ended: bool = False
    winner: str | None = None

    def copy(self) -> Position:
        return Position(
            format=self.format,
            sides=[s.copy() for s in self.sides],
            turn=self.turn,
            field=self.field.copy(),
            request_state=self.request_state,
            ended=self.ended,
            winner=self.winner,
        )

    def swapped(self) -> Position:
        """The same battle seen from the other seat.

        Swapping ``sides`` is not enough, and getting that wrong looks exactly like a bug
        in the resolver. Effects carry a ``source_slot`` naming who applied them -- the
        Pokemon whose Infestation is trapping you, the one whose Leech Seed is draining you
        -- and those references are *side-indexed*. Leave them alone and the mirrored
        position says the trap belongs to the side that is now trapped, which changes
        whether it is released. That difference is worth about seven points on a cell.

        Two encodings are in the field, because two writers produce them: ``"01"`` (side
        digit, slot digit) from the resolver, and Showdown's ``"p1a"`` from a dumped
        battle. Both are remapped; anything else is left as-is rather than guessed at.

        Used by the antisymmetry checks, where a genuine mirror is the whole point: the
        value function must answer ``1 - V`` and the resolver must produce the mirrored
        turn, and neither claim can be tested with an approximate swap.
        """
        out = self.copy()
        out.sides = [out.sides[1], out.sides[0]]
        out.sides[0].id, out.sides[1].id = out.sides[1].id, out.sides[0].id
        out.sides[0].name, out.sides[1].name = out.sides[1].name, out.sides[0].name
        for effect in out.effects():
            effect.source_slot = _swap_source_slot(effect.source_slot)
        return out

    def effects(self) -> list[Effect]:
        """Every Effect the position holds, wherever it lives.

        Enumerated in one place so a new container cannot be silently missed by the
        things that have to walk them all.
        """
        out: list[Effect] = list(self.field.pseudo_weather)
        for side in self.sides:
            out.extend(side.side_conditions)
            for slot in side.slot_conditions:
                out.extend(slot)
            for mon in side.pokemon:
                out.extend(mon.volatiles)
        return out

    @staticmethod
    def from_json(d: dict[str, Any]) -> Position:
        return Position(
            format=d["format"],
            sides=[Side.from_json(s) for s in d["sides"]],
            turn=d.get("turn", 1),
            field=Field.from_json(d.get("field", {})),
            request_state=d.get("requestState", "move"),
            ended=bool(d.get("ended")),
            winner=d.get("winner"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "turn": self.turn,
            "field": self.field.to_json(),
            "sides": [s.to_json() for s in self.sides],
            "requestState": self.request_state,
            "ended": self.ended,
            "winner": self.winner,
        }

    @staticmethod
    def load(path: str | Path) -> Position:
        with Path(path).open(encoding="utf-8") as fh:
            return Position.from_json(json.load(fh))

    def dump(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(self.to_json(), fh, indent=2)
            fh.write("\n")


def validate_position(pos: Position, active_per_side: int) -> list[str]:
    """Structural checks on a hand-written or generated position.

    Returns a list of problems rather than raising, so a CLI can print all of them at
    once.
    """
    problems: list[str] = []
    if len(pos.sides) != 2:
        problems.append(f"expected 2 sides, got {len(pos.sides)}")
    for side in pos.sides:
        if len(side.active) != active_per_side:
            problems.append(
                f"{side.id}: expected {active_per_side} active slots, got {len(side.active)}"
            )
        slots = [p.slot for p in side.pokemon]
        if slots != list(range(len(side.pokemon))):
            problems.append(f"{side.id}: pokemon slots must be 0..n-1 in order, got {slots}")
        for idx, s in enumerate(side.active):
            if s is None:
                continue
            if not 0 <= s < len(side.pokemon):
                problems.append(f"{side.id}: active[{idx}] = {s} is out of range")
                continue
            mon = side.pokemon[s]
            if mon.active_index != idx:
                problems.append(
                    f"{side.id}: {mon.species} activeIndex={mon.active_index} but sits in "
                    f"active slot {idx}"
                )
        seen = [s for s in side.active if s is not None]
        if len(seen) != len(set(seen)):
            problems.append(f"{side.id}: the same party slot appears twice in active")
        for mon in side.pokemon:
            if not 0 <= mon.hp <= mon.maxhp:
                problems.append(f"{side.id}/{mon.species}: hp {mon.hp} outside 0..{mon.maxhp}")
            if mon.fainted != (mon.hp == 0):
                problems.append(
                    f"{side.id}/{mon.species}: fainted={mon.fainted} disagrees with hp={mon.hp}"
                )
            for stat in mon.boosts:
                if stat not in BOOST_IDS:
                    problems.append(f"{side.id}/{mon.species}: unknown boost {stat!r}")
            for stat, value in mon.boosts.items():
                if not -6 <= value <= 6:
                    problems.append(f"{side.id}/{mon.species}: boost {stat}={value} outside -6..6")
            if mon.sp is not None:
                for stat in mon.sp:
                    if stat not in STAT_IDS:
                        problems.append(f"{side.id}/{mon.species}: unknown stat {stat!r} in sp")
    return problems
