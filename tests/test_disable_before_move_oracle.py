"""Disable stops the disabled move at `BeforeMove`, not only on the next menu (IKA-325).

Showdown a5df827, data/moves.ts, and the champions mod's override of the handler::

    disable: {
        condition: {
            onBeforeMovePriority: 7,
            // data/mods/champions/moves.ts
            onBeforeMove(attacker, defender, move) {
                if (!(move.isZ && move.isZOrMaxPowered) && move.id === this.effectState.move &&
                    !move.flags['cantusetwice']) {
                    this.add('cant', attacker, 'Disable', move);
                    return false;
                }
            },
            onDisableMove(pokemon) { ... pokemon.disableMove(moveSlot.id) ... },
        },
    },

The port kept the volatile and the menu's half (`MoveSlot.disabled`), so a move already on
its way went on: a charging move's second turn, which `twoturnmove`'s `onLockMove` fires
whatever the menu says (IKA-316's seed 14 battle 68 turn 4, a Solar Beam), and a move chosen
before a faster Disable landed. Each case plays in Showdown and holds the port's one outcome
to it; the positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`), and the controls are a Disable on
another move and a Disable that lands after the move.
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

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)
VOLATILES = frozenset({"disable", "twoturnmove"})
STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 32, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=None, sp=spread)


def _teams(user_spe: int = 0, disabler_spe: int = 32) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [
        # 1 Solar Beam, 2 Dragon Claw, 3 Protect, 4 Heat Wave
        _mon("Charizard", "Blaze", ["solarbeam", "dragonclaw", "protect", "heatwave"], user_spe),
        # 1 Protect, 2 Swords Dance
        _mon("Swampert", "Damp", ["protect", "swordsdance", "bulkup", "helpinghand"], 20),
    ]
    theirs = [
        # 1 Disable, 2 Protect (Levitate: Cursed Body would disable on its own)
        _mon("Gengar", "Levitate", ["disable", "protect", "shadowball", "helpinghand"], disabler_spe),
        # 1 Protect, 2 Helping Hand, 3 Bulk Up
        _mon("Incineroar", "Blaze", ["protect", "helpinghand", "bulkup", "flareblitz"], 10),
    ]
    return mine, theirs


#: Solar Beam charged at Incineroar; the foes protect.
CHARGE = ["move 1 2, move 1", "move 2, move 1"]
#: Dragon Claw at Incineroar, which lends Gengar a hand.
CLAW = ["move 2 2, move 2", "move 2, move 2 -1"]
#: The foes' turn: Gengar disables Charizard, Incineroar bulks up.
DISABLE = "move 1 1, move 3"

#: name -> (teams, setup turns, the compared turn (Showdown's, then ours when it differs),
#: a log line that shows what the case is about, whether that line must be absent).
CASES: dict[str, tuple] = {
    "a-charged-solar-beam": (
        _teams(), [CHARGE], ["move 1, move 2", DISABLE], "move 1, move 2",
        "|cant|p1a: Charizard|Disable|Solar Beam", True,
    ),
    "a-move-chosen-before-the-disable": (
        _teams(), [CLAW], ["move 2 2, move 2", DISABLE], None,
        "|cant|p1a: Charizard|Disable|Dragon Claw", True,
    ),
    # Controls: Disable on the move last used, and another chosen; Disable after the move.
    "control-another-move": (
        _teams(), [CLAW], ["move 4, move 2", DISABLE], None,
        "|move|p1a: Charizard|Heat Wave", True,
    ),
    "control-a-slower-disable": (
        _teams(user_spe=32, disabler_spe=0), [CLAW], ["move 2 2, move 2", DISABLE], None,
        "|cant|p1a: Charizard|Disable", False,
    ),
}


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (
            mon.hp,
            mon.fainted,
            mon.status if not mon.fainted else None,
            tuple(mon.boosts.get(stat, 0) for stat in STATS) if not mon.fainted else (),
            tuple(sorted(v.id for v in mon.volatiles if v.id in VOLATILES)) if not mon.fainted else (),
            tuple(m.pp for m in mon.moves),
            mon.move_last_turn_failed if not mon.fainted else None,
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choices[side] in menu, (choices[side], sorted(menu))
        out.append(menu[choices[side]])
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    require_binary()
    (mine, theirs), setup, choices, ours, shown, present = CASES[name]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs_now = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    # Showdown first: the turn is one Disable stops the move on (or, a control, does not).
    assert any(shown in line for line in log) is present, (name, log)
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    port_choices = [ours or choices[0], choices[1]]
    result = resolve_turn(reg, before, _actions(reg, before, port_choices), budget=BUDGET)
    assert len(result.branches) == 1, result.branches
    assert _state(result.branches[0].position) == theirs_now, name
