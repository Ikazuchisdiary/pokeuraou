"""Heatproof halves the burn's residual damage (IKA-328).

Showdown a5df827 (no champions override of either)::

    // data/conditions.ts
    brn: { onResidualOrder: 10, onResidual(pokemon) { this.damage(pokemon.baseMaxhp / 16); } },
    // data/abilities.ts
    heatproof: {
        onSourceModifyAtk / onSourceModifySpA: Fire 0.5x,
        onDamage(damage, target, source, effect) {
            if (effect && effect.id === 'brn') { return damage / 2; }
        },
    },

`spreadDamage` floors to at least 1 before the `Damage` event and again after it. The port
named Heatproof for its Fire half (so it is `modelled.rs`'s and noted nowhere) and took the
full sixteenth for the burn: a silent error, and Sinistcha can hold it in M-C. Each case plays
in Showdown and holds the port's one outcome to it; the positive control is the exe before
this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`), and the controls are a
burned Pokemon without Heatproof and a Heatproof one not burned.
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


def _mon(species: str, ability: str, moves: list[str], spe: int, hp: int = 32) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": hp, "atk": 20, "def": 10, "spa": 0, "spd": 4, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=None, sp=spread)


#: name -> (holder species, ability, its HP points, Gengar's move: 1 Will-O-Wisp or 2 Protect,
#: whether Showdown's burn damage is the halved sixteenth).
CASES: dict[str, tuple] = {
    "sinistcha": ("Sinistcha", "Heatproof", 32, "move 1 1", True),
    "sinistcha-fewer-hp": ("Sinistcha", "Heatproof", 5, "move 1 1", True),
    "masterpiece": ("Sinistcha-Masterpiece", "Heatproof", 20, "move 1 1", True),
    # Controls: burned without Heatproof; Heatproof and not burned.
    "control-hospitality": ("Sinistcha", "Hospitality", 32, "move 1 1", False),
    "control-not-burned": ("Sinistcha", "Heatproof", 32, "move 2", False),
}


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (mon.hp, mon.fainted, mon.status)
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
def test_the_burn_damage_matches_showdown(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    require_binary()
    species, ability, hp, wisp, halved = CASES[name]
    mine = [
        _mon(species, ability, ["swordsdance", "protect", "helpinghand", "bulkup"], 0, hp),
        _mon("Swampert", "Damp", ["protect", "swordsdance", "bulkup", "helpinghand"], 20),
    ]
    theirs = [
        _mon("Gengar", "Levitate", ["willowisp", "protect", "shadowball", "helpinghand"], 32),
        _mon("Milotic", "Marvel Scale", ["protect", "swordsdance", "recover", "icebeam"], 10),
    ]
    choices = ["move 1, move 1", f"{wisp}, move 1"]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    handle.close()
    # Showdown first: the burn took a sixteenth, or half of one.
    holder = after.sides[0].pokemon[after.sides[0].active[0]]
    sixteenth = max(1, holder.maxhp // 16)
    if holder.status == "brn":
        assert holder.maxhp - holder.hp == (max(1, sixteenth // 2) if halved else sixteenth), name
    else:
        assert holder.hp == holder.maxhp and not halved, name
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert len(result.branches) == 1, result.branches
    assert _state(result.branches[0].position) == _state(after), name
