"""Two faints and one Pokemon left: Showdown asks for both slots and lets one pass (IKA-257).

`battle.ts` keeps every `switchFlag` while `canSwitch(side)` is not 0, so the request is
`forceSwitch: [true, true]`, and `side.ts` `clearChoice` allows
`forcedPasses = canSwitchOut - min(canSwitchOut, canSwitchIn)` passes on those slots. The
port's replacement phase noted every pass on an owed slot as "replacement owed ... but none
was chosen", so each such phase in generation carried a note Showdown would not: 62 in the
600 games of the g600b trial run, all of this shape. A pass the bench can fill is still
noted -- Showdown refuses it (`Can't pass: You need to switch in a Pokemon`).

Setup: turn 1 Gengar explodes beside a protecting Chandelure and Garchomp comes in; turn 2
Garchomp and Chandelure both explode. Every foe is a Ghost, so nothing else is hit, and
Incineroar is the one Pokemon left.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import PassAction, SideAction, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import replacements_needed, resolve_replacements
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

OWED = "replacement owed at"


def _mon(species: str, ability: str, moves: list[str]) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20},
    )


TEAM_A = [
    _mon("Gengar", "Cursed Body", ["explosion", "protect", "shadowball", "sludgebomb"]),
    _mon("Chandelure", "Flash Fire", ["explosion", "protect", "shadowball", "flamethrower"]),
    _mon("Garchomp", "Rough Skin", ["explosion", "dragonclaw", "protect", "rockslide"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "protect"]),
]
TEAM_B = [
    _mon("Froslass", "Cursed Body", ["protect", "shadowball", "icebeam", "willowisp"]),
    _mon("Polteageist", "Cursed Body", ["protect", "shadowball", "storedpower", "strengthsap"]),
    _mon("Sableye", "Prankster", ["protect", "shadowball", "foulplay", "willowisp"]),
    _mon("Dragapult", "Clear Body", ["protect", "shadowball", "dragondarts", "uturn"]),
]


def _force(request: dict | None) -> tuple[bool, ...]:
    return tuple(bool(f) for f in ((request or {}).get("forceSwitch") or []))


def _start(oracle: Oracle, *, one_left: bool):  # noqa: ANN202
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    if one_left:
        handle.step(["move 1, move 2", "move 1, move 1"])
        handle.step(["switch 3", None])
    handle.step(["move 1, move 1", "move 1, move 1"])
    assert handle.choice_errors == [], handle.choice_errors
    return handle


def _owed_notes(unmodelled: tuple[str, ...]) -> list[str]:
    return [note for note in unmodelled if note.startswith(OWED)]


def test_one_pokemon_for_two_faints_leaves_one_slot_empty(reg, oracle: Oracle) -> None:  # noqa: ANN001
    handle = _start(oracle, one_left=True)
    before = Position.from_json(handle.position)
    asked = _force(handle.requests[0])
    bench = [m.species for m in before.sides[0].pokemon if not m.fainted and not m.is_active]
    handle.step(["pass, pass", None])
    refused = list(handle.choice_errors)
    handle.close()
    assert asked == (True, True)
    assert bench == ["incineroar"]
    assert refused and "Can't pass" in refused[0], refused

    assert replacements_needed(before, reg) == ((True, True), (False, False))
    options = switch_actions_after_faint(reg, before, 0, [True, True])
    assert sorted(o.to_choice() for o in options) == ["pass, switch 4", "switch 4, pass"]
    foe = SideAction(slots=(PassAction(slot=0), PassAction(slot=1)))
    for option in options:
        after = resolve_replacements(reg, before, [option, foe])
        assert _owed_notes(after.unmodelled) == [], (option.to_choice(), after.unmodelled)
        filled = option.slots.index(next(s for s in option.slots if not isinstance(s, PassAction)))
        side = after.position.sides[0]
        assert side.pokemon[side.active[filled]].species == "incineroar"


def test_showdown_takes_either_slot_for_the_last_pokemon(oracle: Oracle) -> None:
    for choice in ("switch 4, pass", "pass, switch 4"):
        handle = _start(oracle, one_left=True)
        handle.step([choice, None])
        errors = list(handle.choice_errors)
        handle.close()
        assert errors == [], (choice, errors)


def test_a_pass_the_bench_can_fill_is_still_noted(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """The control: two faints, two on the bench. Showdown refuses a pass, the port notes it."""
    handle = _start(oracle, one_left=False)
    before = Position.from_json(handle.position)
    asked = _force(handle.requests[0])
    handle.step(["switch 3, pass", None])
    refused = list(handle.choice_errors)
    handle.close()
    assert asked == (True, True)
    assert refused and "Can't pass" in refused[0], refused

    options = switch_actions_after_faint(reg, before, 0, [True, True])
    assert options and all(not any(isinstance(s, PassAction) for s in o.slots) for o in options)
    one = SideAction(slots=(options[0].slots[0], PassAction(slot=1)))
    foe = SideAction(slots=(PassAction(slot=0), PassAction(slot=1)))
    after = resolve_replacements(reg, before, [one, foe])
    assert _owed_notes(after.unmodelled) == ["replacement owed at p1[1] but none was chosen"]
    both = resolve_replacements(reg, before, [options[0], foe])
    assert _owed_notes(both.unmodelled) == []
