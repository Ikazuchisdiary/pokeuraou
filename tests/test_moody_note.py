"""Moody's end-of-turn boosts are noted on every turn its holder is there for (IKA-308).

Showdown a5df827, data/abilities.ts::

    moody: {
        onResidualOrder: 28,
        onResidualSubOrder: 2,
        onResidual(pokemon) {
            // +2 to one stat below 6 and -1 to another above -6, drawn at random
            this.boost(boost, pokemon, pokemon);
        },
    },

The port does not apply them (a random draw of two stats), and its only note was the one
every hit carries for an ability it ignores (`damage::unmodelled_effects`): a turn the
holder spent on status moves, with nobody hitting it, missed the boosts and said nothing
(diff_turn seed 3 battle 88 turn 2 in IKA-256). The note is now `moves::residuals`'s, where
Moody fires. Each case plays in Showdown first, so the turn is one Moody fires on (or, for
the controls, does not); the positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, require_binary, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

NOTE = "ability: moody (end-of-turn +2/-1 not applied)"
STATS = ("atk", "def", "spa", "spd", "spe")

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 32, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=None, sp=spread)


#: The holder: 1 Protect, 2 Swords Dance, 3 Helping Hand, 4 Dragon Claw.
HOLDER = ["protect", "swordsdance", "helpinghand", "dragonclaw"]
#: The partner and the bench: 1 Protect, 2 Helping Hand, 3 Growl, 4 Dragon Claw.
OTHER = ["protect", "helpinghand", "growl", "dragonclaw"]
#: The foes: 1 Swords Dance, 2 Helping Hand, 3 Dragon Claw, 4 Protect.
FOE = ["swordsdance", "helpinghand", "dragonclaw", "protect"]


def _teams(ability: str) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        _mon("Scovillain", ability, HOLDER, 10),
        _mon("Swampert", "Damp", OTHER, 20),
        _mon("Milotic", "Marvel Scale", OTHER, 0),
    ]
    theirs = [
        _mon("Garchomp", "Rough Skin", FOE, 30),
        _mon("Incineroar", "Blaze", FOE, 30),
    ]
    return mine, theirs


#: The foes change nothing: A Swords Dance, B Helping Hand on A.
QUIET = "move 1, move 2 -1"

#: name -> (the holder's ability, our choices, theirs, whether Moody fires in Showdown, the note).
CASES: dict[str, tuple[str, str, str, bool, bool]] = {
    # The case the note was missing from: status moves only, nobody hit.
    "status-moves-only": ("Moody", "move 1, move 1", QUIET, True, True),
    "swords-dance-only": ("Moody", "move 2, move 2 -1", QUIET, True, True),
    # A hit on the holder: noted before too (on the hit), and still.
    "hit-by-a-foe": ("Moody", "move 3 -2, move 1", "move 3 1, move 2 -1", True, True),
    # Controls: no Moody on the field, and the holder switched out before the residual.
    "control-no-moody": ("Damp", "move 1, move 1", QUIET, False, False),
    "control-holder-switched-out": ("Moody", "switch 3, move 1", QUIET, False, False),
}


def _boosts(pos: Position, species: str) -> tuple[int, ...]:
    mon = next(m for m in pos.sides[0].pokemon if m.species == species)
    return tuple(mon.boosts.get(stat, 0) for stat in STATS)


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], Position]:
    ability, ours, theirs, _fires, _noted = CASES[name]
    mine, foes = _teams(ability)
    handle = oracle.create(FORMAT_ID, mine, foes, policy=RandomnessPolicy())
    handle.step(["team 123", "team 12"])
    before = Position.from_json(handle.position)
    handle.step([ours, theirs])
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    return before, [ours, theirs], after


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choices[side] in menu, (choices[side], sorted(menu))
        out.append(menu[choices[side]])
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_moody_is_noted_when_it_fires(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    require_binary()
    _ability, _ours, _theirs, fires, noted = CASES[name]
    before, choices, after = _play(oracle, name)
    # Showdown first: the case is a turn Moody fires on (its holder's boosts moved, and no
    # move of this turn boosts it except Swords Dance's +2 Attack), or one it does not.
    moved = _boosts(after, "scovillain")
    if "swords-dance" in name:
        moved = (moved[0] - 2, *moved[1:])
    assert (moved != (0,) * len(STATS)) is fires, (name, moved)
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert (NOTE in result.unmodelled) is noted, (name, sorted(result.unmodelled))
    # The note on the hit (`defender.ability:moody`) is gone: this one says it where it fires.
    assert not any("ability:moody" in note for note in result.unmodelled), result.unmodelled
