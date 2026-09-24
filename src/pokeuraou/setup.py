"""Building a position from a written-down scenario, without the simulator.

The analysis tool has to start from what a player can see: two open team sheets and the
state of the field. That is a JSON file, not a Showdown battle, so this module is the only
path from text to :class:`~pokeuraou.position.Position`.

The asymmetry between the two sides is the whole point. Our own spread is known, so our
Pokemon get exact stats. The opponent's is not, so their entries carry no ``sp`` at all
and the position is deliberately *unusable* until a belief fills it in -- ``stats_array``
raises rather than assuming, and :func:`with_spreads` is the only way to satisfy it. A
hidden HP is given as a fraction, because a fraction is what the game shows and an
absolute HP would silently smuggle in the max HP the spread was supposed to hide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .hpdisplay import Band, band, uses_floor_display
from .position import Effect, Field, MoveSlot, Pokemon, Position, Side
from .regulation import STAT_IDS, Regulation, load_regulation, to_id
from .stats import nature_multipliers, stats_from_sp

#: Slot keys are (side index, active slot index), as everywhere else in the project.
SlotKey = tuple[int, int]


class ScenarioError(ValueError):
    pass


@dataclass(slots=True)
class Scenario:
    """A parsed scenario: a position plus what is unknown about it."""

    reg: Regulation
    position: Position
    #: Party index -> revealed nature, for every Pokemon whose spread is hidden.
    hidden: dict[tuple[int, int], str] = field(default_factory=dict)
    #: Party index -> the (percentage, bar colour) the scenario reported, for the same
    #: Pokemon. The exact HP is not knowable from this alone: see :meth:`band_for`.
    hp_display: dict[tuple[int, int], tuple[int, str | None]] = field(default_factory=dict)
    #: The scenario's `observations` list, unparsed. `observe.parse_observations` turns
    #: these into likelihoods; keeping them raw here stops this module from having to
    #: know about the belief layer.
    raw_observations: list[dict[str, Any]] = field(default_factory=list)
    source: str = "<inline>"

    def band_for(self, party_key: tuple[int, int], maxhp: int) -> Band:
        """Exact HP values consistent with what the scenario says was displayed.

        Depends on max HP, which depends on the spread, so the band is per belief
        particle rather than per Pokemon -- a 62% bar on a 250 HP Pokemon is a wider
        band than on a 160 HP one.
        """
        percent, colour = self.hp_display.get(party_key, (100, None))
        return band(
            percent, maxhp, colour=colour, floor_rule=uses_floor_display(self.reg)
        )

    def hidden_actives(self) -> list[SlotKey]:
        """(side, active slot) for the hidden Pokemon that are on the field."""
        out: list[SlotKey] = []
        for side_index, side in enumerate(self.position.sides):
            for slot, party_index in enumerate(side.active):
                if (side_index, party_index) in self.hidden:
                    out.append((side_index, slot))
        return out

    def party_index(self, key: SlotKey) -> int:
        side_index, slot = key
        return self.position.sides[side_index].active[slot]


def _read_hp(entry: dict[str, Any]) -> tuple[int | None, int, str | None]:
    """(absolute HP, displayed percentage, bar colour) from a team entry.

    A percentage may carry the bar colour Showdown appends at 20% and 50% -- "50%g" means
    strictly above half -- because that is a bit of information the game gives away and
    throwing it out would make the band wider than it is.
    """
    if "hp" not in entry and "hpPercent" not in entry:
        # Absent means full, which is a *percentage* of 100 rather than an absolute 100:
        # a hidden Pokemon must not be handed an absolute HP by omission.
        return None, 100, None
    raw = entry.get("hp", entry.get("hpPercent"))
    if isinstance(raw, str):
        text = raw.strip()
        colour: str | None = None
        if text and text[-1].isalpha():
            colour, text = text[-1], text[:-1]
        text = text.rstrip("%").strip()
        try:
            percent = int(round(float(text)))
        except ValueError as exc:
            raise ScenarioError(
                f"cannot read hp {raw!r}; give an integer percentage such as \"62%\", "
                "optionally with the bar colour (\"50%g\")"
            ) from exc
        if not 0 <= percent <= 100:
            raise ScenarioError(f"hp {raw!r} is not a percentage")
        return None, percent, colour
    value = int(raw)
    if value < 0:
        raise ScenarioError(f"hp {raw!r} is negative")
    return value, 100 if value else 0, None


def _moves(reg: Regulation, entry: dict[str, Any]) -> list[MoveSlot]:
    raw = entry.get("moves") or []
    if not raw:
        raise ScenarioError(f"{entry.get('species')} has no moves")
    out: list[MoveSlot] = []
    for name in raw:
        move_id = to_id(name)
        move = reg.moves.get(move_id)
        if move is None:
            raise ScenarioError(
                f"{name!r} is not legal in {reg.meta.format_id}; the regulation config is the "
                "authority on the move pool"
            )
        out.append(MoveSlot(id=move_id, pp=move.start_pp, maxpp=move.start_pp))
    return out


def _make_pokemon(
    reg: Regulation, index: int, entry: dict[str, Any], *, known: bool
) -> tuple[Pokemon, str, tuple[int, str | None]]:
    species_id = to_id(entry.get("species", ""))
    species = reg.species.get(species_id)
    if species is None:
        raise ScenarioError(f"{entry.get('species')!r} is not legal in {reg.meta.format_id}")
    nature = str(entry.get("nature", "")).strip()
    if not nature:
        raise ScenarioError(
            f"{species.name} has no nature; the Champions team sheet reveals it, so "
            "leaving it out would discard information the player actually has"
        )
    ability = to_id(entry.get("ability", ""))
    item = to_id(entry["item"]) if entry.get("item") else None

    sp: dict[str, int] | None = None
    hp = maxhp = 0
    absolute, percent, colour = _read_hp(entry)
    if not known and absolute is not None and absolute > 0:
        raise ScenarioError(
            f"{species.name}'s hp {absolute} looks absolute; the opponent's HP is only "
            "ever shown as a percentage, and an absolute number would smuggle in the max "
            "HP the spread is meant to hide. Give \"62%\" (optionally \"50%g\" for the "
            "bar colour)."
        )
    if known:
        raw_sp = entry.get("sp")
        if raw_sp is None:
            raise ScenarioError(
                f"{species.name} is on our own side and has no sp; our own spread is "
                "known and has to be given"
            )
        sp = {k: int(raw_sp.get(k, 0)) for k in STAT_IDS}
        base = np.array(species.base_stats, dtype=np.int64)
        spread = np.array([[sp[k] for k in STAT_IDS]], dtype=np.int64)
        stats = stats_from_sp(
            reg, base, spread, nature_multipliers(reg, [nature]), level=50
        )
        maxhp = int(stats[0, 0])
        if absolute is not None:
            hp = min(absolute, maxhp)
        elif percent <= 0:
            hp = 0
        else:
            # Our own max HP is known, so a percentage either pins down one value or it
            # does not. If it does not, asking for the number is better than picking one:
            # we are not actually uncertain about our own Pokemon.
            options = band(
                percent, maxhp, colour=colour, floor_rule=uses_floor_display(reg)
            )
            if options.width > 1:
                raise ScenarioError(
                    f"{species.name} is on our own side and \"{percent}%\" of {maxhp} is "
                    f"{options.low}..{options.high}; give the exact HP, which we know"
                )
            hp = options.low

    mon = Pokemon(
        slot=index,
        species=species_id,
        base_species=species_id,
        types=species.types,
        ability=ability,
        nature=nature,
        moves=_moves(reg, entry),
        hp=hp,
        maxhp=maxhp,
        item=item,
        base_item=item,
        sp=sp,
        active_index=None,
        status=entry.get("status"),
        boosts={k: int(v) for k, v in (entry.get("boosts") or {}).items() if v},
        fainted=bool(entry.get("fainted")) or percent <= 0,
    )
    return mon, nature, (percent, colour)


def _field(raw: dict[str, Any]) -> Field:
    pseudo = [
        Effect(id=to_id(p), duration=5)
        if isinstance(p, str)
        else Effect(id=to_id(p["id"]), duration=p.get("duration", 5))
        for p in (raw.get("pseudoWeather") or [])
    ]
    return Field(
        weather=to_id(raw["weather"]) if raw.get("weather") else None,
        weather_duration=raw.get("weatherDuration", 5 if raw.get("weather") else None),
        terrain=to_id(raw["terrain"]) if raw.get("terrain") else None,
        terrain_duration=raw.get("terrainDuration", 5 if raw.get("terrain") else None),
        pseudo_weather=pseudo,
    )


def parse_scenario(data: dict[str, Any], *, source: str = "<inline>") -> Scenario:
    """Reads a scenario. Side 0 is ours and its spreads are required."""
    format_id = data.get("regulation")
    if not format_id:
        raise ScenarioError("scenario needs a \"regulation\" (a configs/regulations id)")
    reg = load_regulation(str(format_id))

    sides_raw = data.get("sides")
    if not isinstance(sides_raw, list) or len(sides_raw) != 2:
        raise ScenarioError("scenario needs exactly two sides; side 0 is ours")

    hidden: dict[tuple[int, int], str] = {}
    displays: dict[tuple[int, int], tuple[int, str | None]] = {}
    sides: list[Side] = []
    for side_index, side_raw in enumerate(sides_raw):
        known = side_index == 0 or bool(side_raw.get("spreadsKnown"))
        team = side_raw.get("team") or []
        if len(team) != 4:
            raise ScenarioError(
                f"side {side_index} has {len(team)} Pokemon; the selection phase is out of "
                "scope, so a scenario is the four that were brought"
            )
        mons: list[Pokemon] = []
        actives: list[int] = []
        for index, entry in enumerate(team):
            mon, nature, display = _make_pokemon(reg, index, entry, known=known)
            if not known:
                hidden[(side_index, index)] = nature
                displays[(side_index, index)] = display
            if entry.get("active"):
                actives.append(index)
            mons.append(mon)
        if len(actives) != 2:
            raise ScenarioError(
                f"side {side_index} has {len(actives)} Pokemon marked \"active\": true; "
                "a doubles position has exactly two"
            )
        for slot, party_index in enumerate(actives):
            mons[party_index].active_index = slot
        mega_slots = [
            m.slot for m in mons if reg.mega_target(m.species, m.item) is not None
        ]
        conditions = [
            Effect(id=to_id(c), duration=None) if isinstance(c, str)
            else Effect(id=to_id(c["id"]), duration=c.get("duration"), layers=c.get("layers"))
            for c in (side_raw.get("sideConditions") or [])
        ]
        sides.append(
            Side(
                id=str(side_raw.get("id") or f"p{side_index + 1}"),
                name=str(side_raw.get("name") or f"p{side_index + 1}"),
                active=actives,
                pokemon=mons,
                side_conditions=conditions,
                slot_conditions=[[], []],
                mega_used=bool(side_raw.get("megaUsed")),
                mega_capable_slots=mega_slots,
            )
        )

    position = Position(
        format=reg.meta.format_id,
        sides=sides,
        turn=int(data.get("turn", 1)),
        field=_field(data.get("field") or {}),
    )
    observations = data.get("observations") or []
    if not isinstance(observations, list):
        raise ScenarioError("\"observations\" must be a list")
    return Scenario(
        reg=reg,
        position=position,
        hidden=hidden,
        hp_display=displays,
        raw_observations=list(observations),
        source=source,
    )


def load_scenario(path: str | Path) -> Scenario:
    text = Path(path).read_text(encoding="utf-8")
    return parse_scenario(json.loads(text), source=str(path))


def with_spreads(
    scenario: Scenario,
    assignment: dict[tuple[int, int], np.ndarray],
    hp: dict[tuple[int, int], int] | None = None,
) -> Position:
    """A copy of the position with the given SP filled into the hidden Pokemon.

    ``assignment`` is keyed by (side, *party* index) and holds a length-6 SP vector.
    Every hidden Pokemon must be assigned: a position with a half-filled belief would
    resolve for some actions and raise for others, which is a worse failure than raising
    for all of them.
    """
    missing = set(scenario.hidden) - set(assignment)
    if missing:
        raise ScenarioError(
            f"no spread assigned for {sorted(missing)}; every hidden Pokemon needs one"
        )

    pos = scenario.position.copy()
    for (side_index, party_index), spread in assignment.items():
        mon = pos.sides[side_index].pokemon[party_index]
        species = scenario.reg.species[mon.species]
        values = np.asarray(spread, dtype=np.int64).reshape(1, len(STAT_IDS))
        stats = stats_from_sp(
            scenario.reg,
            np.array(species.base_stats, dtype=np.int64),
            values,
            nature_multipliers(scenario.reg, [mon.nature]),
            level=50,
        )
        mon.sp = {k: int(values[0, i]) for i, k in enumerate(STAT_IDS)}
        mon.maxhp = int(stats[0, 0])
        if mon.fainted:
            mon.hp = 0
            continue
        key = (side_index, party_index)
        chosen = None if hp is None else hp.get(key)
        if chosen is not None:
            mon.hp = max(0, min(int(chosen), mon.maxhp))
            continue
        # No exact HP supplied: take the bottom of the band. It is a *choice inside a
        # known set*, not an estimate, and the CLI enumerates the set instead of relying
        # on this.
        mon.hp = scenario.band_for(key, mon.maxhp).low
    return pos


__all__ = [
    "Scenario",
    "ScenarioError",
    "SlotKey",
    "load_scenario",
    "parse_scenario",
    "with_spreads",
]
