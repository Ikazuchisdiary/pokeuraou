"""End-of-turn abilities the port ignores are noted on a turn they fire, not only on a hit (IKA-317).

Moody's shape (IKA-308), for the rest of the abilities Showdown a5df827 fires at the end of
the turn (an `onResidual`, or a residual damage or heal) and the port does not apply. Their
only note was the one every hit carries for an ignored ability (`damage::unmodelled_effects`),
so a turn their holder spent on status moves, with nobody hitting it, missed the effect and
said nothing. `rust/src/ability_notes.rs` notes each where it fires:

    poisonheal:  onDamage(damage, target, source, effect) {
                     if (effect.id === 'psn' || effect.id === 'tox') { this.heal(...); return false; }
    hydration:   onResidual(pokemon) {
                     if (pokemon.status && ['raindance', 'primordialsea'].includes(...)) cureStatus();
    liquidooze:  onSourceTryHeal(damage, target, source, effect) {
                     if (['drain', 'leechseed', 'strengthsap'].includes(effect.id)) {
                         this.damage(damage); return 0; }
    opportunist: onFoeAfterBoost(...) -- copied after the foe's move, or at the residual
    hungerswitch: onResidual(pokemon) { ... pokemon.formeChange(targetForme); }
    forecast:    onWeatherChange(pokemon) { ... formeChange('Castform-Sunny') ... }

Each case plays in Showdown first, so the turn is one the ability acts on (or, for the
controls, does not); the positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`), which says nothing on any of them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Pokemon, Position

from ._port import Budget, require_binary, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)

#: The holder: 1 Swords Dance, 2 Protect, 3 Rain Dance, 4 Sunny Day.
HOLDER = ["swordsdance", "protect", "raindance", "sunnyday"]
#: The partner and the bench: 1 Helping Hand, 2 Rain Dance, 3 Sunny Day, 4 Protect.
OTHER = ["helpinghand", "raindance", "sunnyday", "protect"]
#: The foes: 1 Toxic, 2 Leech Seed, 3 Swords Dance, 4 Helping Hand.
FOE = ["toxic", "leechseed", "swordsdance", "helpinghand"]


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 32, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=None, sp=spread)


def _holder(pos: Position) -> Pokemon:
    return pos.sides[0].pokemon[pos.sides[0].active[0]]


def _foe_a(pos: Position) -> Pokemon:
    return pos.sides[1].pokemon[pos.sides[1].active[0]]


@dataclass(frozen=True)
class Case:
    species: str
    ability: str
    ours: str
    theirs: str
    #: What Showdown shows when the ability acted, read off (before, after).
    acted: Callable[[Position, Position], bool]
    fires: bool
    note: str | None


def _unhurt_poisoned(before: Position, after: Position) -> bool:
    mon = _holder(after)
    return mon.status == "tox" and mon.hp == mon.maxhp


def _rain_cured(before: Position, after: Position) -> bool:
    mon = _holder(after)
    return mon.status is None and mon.hp == mon.maxhp


def _seeder_hurt(before: Position, after: Position) -> bool:
    return _foe_a(after).hp < _foe_a(before).hp


def _copied_boost(before: Position, after: Position) -> bool:
    return _holder(after).boosts.get("atk", 0) == 2


def _form_changed(before: Position, after: Position) -> bool:
    return _holder(after).species != _holder(before).species


POISON_HEAL = "ability: poisonheal (poison heals 1/8 instead of hurting, not applied)"
HYDRATION = "ability: hydration (status cured in rain, not applied)"
LIQUID_OOZE = "ability: liquidooze (drain hurts instead of healing, not applied)"
OPPORTUNIST = "ability: opportunist (copying foes' boosts not applied)"
HUNGER_SWITCH = "ability: hungerswitch (end-of-turn form change not applied)"
FORECAST = "ability: forecast (form change with the weather not applied)"

#: Our choices: the holder's status move, the partner's Protect. Theirs: foe A's move on the
#: holder, foe B's Helping Hand.
CASES: dict[str, Case] = {
    "poisonheal-toxic": Case(
        "Gliscor", "Poison Heal", "move 1, move 4", "move 1 1, move 4 -1",
        _unhurt_poisoned, True, POISON_HEAL,
    ),
    "hydration-rain": Case(
        "Vaporeon", "Hydration", "move 3, move 4", "move 1 1, move 4 -1",
        _rain_cured, True, HYDRATION,
    ),
    "liquidooze-leechseed": Case(
        "Swalot", "Liquid Ooze", "move 1, move 4", "move 2 1, move 4 -1",
        _seeder_hurt, True, LIQUID_OOZE,
    ),
    "opportunist-swordsdance": Case(
        "Espathra", "Opportunist", "move 2, move 4", "move 3, move 4 -1",
        _copied_boost, True, OPPORTUNIST,
    ),
    "hungerswitch": Case(
        "Morpeko", "Hunger Switch", "move 2, move 4", "move 3, move 4 -1",
        _form_changed, True, HUNGER_SWITCH,
    ),
    "forecast-sun": Case(
        "Castform", "Forecast", "move 2, move 3", "move 3, move 4 -1",
        _form_changed, True, FORECAST,
    ),
    # Controls: the holder unpoisoned, and the Toxic on a Pokemon without Poison Heal.
    "control-poisonheal-no-status": Case(
        "Gliscor", "Poison Heal", "move 1, move 4", "move 3, move 4 -1",
        _unhurt_poisoned, False, None,
    ),
    "control-no-holder": Case(
        "Gliscor", "Hyper Cutter", "move 1, move 4", "move 1 1, move 4 -1",
        _unhurt_poisoned, False, None,
    ),
}


def _play(oracle: Oracle, case: Case) -> tuple[Position, list[str], Position]:
    mine = [
        _mon(case.species, case.ability, HOLDER, 10),
        _mon("Swampert", "Damp", OTHER, 20),
        _mon("Milotic", "Marvel Scale", OTHER, 0),
    ]
    foes = [_mon("Garchomp", "Rough Skin", FOE, 30), _mon("Incineroar", "Blaze", FOE, 30)]
    handle = oracle.create(FORMAT_ID, mine, foes, policy=RandomnessPolicy())
    handle.step(["team 123", "team 12"])
    before = Position.from_json(handle.position)
    handle.step([case.ours, case.theirs])
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    return before, [case.ours, case.theirs], after


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choices[side] in menu, (choices[side], sorted(menu))
        out.append(menu[choices[side]])
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_end_of_turn_ability_is_noted_when_it_fires(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    require_binary()
    case = CASES[name]
    before, choices, after = _play(oracle, case)
    # Showdown first: the turn is one the ability acted on, or (a control) one it did not.
    assert case.acted(before, after) is case.fires, (name, _holder(after), _foe_a(after).hp)
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    ours = {note for note in result.unmodelled if note.startswith("ability: ")}
    expected = {case.note} if case.note else set()
    assert ours == expected, (name, sorted(result.unmodelled))
