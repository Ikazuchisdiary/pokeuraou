"""A resist berry reads the hit's type after a skin or the move's own change (IKA-326).

Showdown a5df827, data/items.ts (no champions override of the berries)::

    roseliberry: {
        onSourceModifyDamage(damage, source, target, move) {
            if (move.type === 'Fairy' && target.getMoveHitData(move).typeMod > 0) {
                ... if (target.eatItem()) { ... return this.chainModify(0.5); }
    chilanberry: {
        onSourceModifyDamage(damage, source, target, move) {
            if (move.type === 'Normal' && (!target.volatiles['substitute'] || ...)) {
                if (target.eatItem()) { ... return this.chainModify(0.5); }

`move.type` is the active move's, after `ModifyType`: Pixilate's Quick Attack is Fairy
(data/abilities.ts, pixilate `onModifyType`), Normalize's Flamethrower Normal, Weather Ball
in the sun Fire. The port halved the damage by the calculator's type but decided whether the
berry was eaten by the move's declared type, so the Pixilate Quick Attack into a Roseli Berry
left the berry (IKA-316's seed 16 battle 52 turn 7) and a Chilan Berry was eaten instead.
Each case plays in Showdown and holds the port's one outcome to it; the positive control is
the exe before this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`), and the
controls are a berry the declared type already decided.
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
#: The holder's moves: it Swords Dances (move 1).
STILL = ["swordsdance", "protect", "helpinghand", "bulkup"]


def _mon(species: str, ability: str, moves: list[str], spe: int, item: str | None = None) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 32, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: name -> (attacker species, ability, its move (move 1), holder species, berry, eaten).
CASES: dict[str, tuple] = {
    "pixilate-roseli": ("Sylveon", "Pixilate", "quickattack", "Garchomp", "Roseli Berry", True),
    "pixilate-chilan": ("Sylveon", "Pixilate", "quickattack", "Garchomp", "Chilan Berry", False),
    "aerilate-coba": ("Salamence", "Aerilate", "quickattack", "Venusaur", "Coba Berry", True),
    "refrigerate-yache": ("Glalie", "Refrigerate", "quickattack", "Garchomp", "Yache Berry", True),
    "galvanize-wacan": ("Raichu", "Galvanize", "quickattack", "Gyarados", "Wacan Berry", True),
    "liquidvoice-passho": ("Primarina", "Liquid Voice", "hypervoice", "Incineroar", "Passho Berry", True),
    "normalize-chilan": ("Persian", "Normalize", "flamethrower", "Garchomp", "Chilan Berry", True),
    "weatherball-sun-occa": ("Ninetales", "Drought", "weatherball", "Venusaur", "Occa Berry", True),
    # Controls: no type change, the berry the declared type decides.
    "control-moonblast-roseli": ("Sylveon", "Cute Charm", "moonblast", "Garchomp", "Roseli Berry", True),
    "control-quickattack-chilan": ("Sylveon", "Cute Charm", "quickattack", "Garchomp", "Chilan Berry", True),
    "control-normalize-occa": ("Persian", "Normalize", "flamethrower", "Scizor", "Occa Berry", False),
}


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (mon.hp, mon.fainted, mon.item)
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
def test_the_berry_reads_the_hit_type(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    require_binary()
    species, ability, move, holder, berry, eaten = CASES[name]
    mine = [
        _mon(species, ability, [move, "protect", "helpinghand", "bulkup"], 32),
        _mon("Swampert", "Damp", ["protect", "swordsdance", "bulkup", "helpinghand"], 20),
    ]
    theirs = [
        _mon(holder, "Pressure", STILL, 0, berry),
        _mon("Milotic", "Marvel Scale", ["protect", "swordsdance", "recover", "icebeam"], 10),
    ]
    spread = move == "hypervoice"
    choices = ["move 1, move 1" if spread else "move 1 1, move 1", "move 1, move 1"]
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    log = list(handle.log)
    handle.close()
    # Showdown first: the berry went (or stayed) as the case says.
    held = after.sides[1].pokemon[after.sides[1].active[0]].item
    assert (held is None) is eaten, (name, held, log)
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert len(result.branches) == 1, result.branches
    assert _state(result.branches[0].position) == _state(after), name
