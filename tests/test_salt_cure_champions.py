"""Salt Cure's residual is the champions mod's, not the base game's (IKA-159).

The mod rewrites the volatile's residual (vendor/pokemon-showdown/data/mods/champions/
moves.ts, `saltcure`):

    onResidual(pokemon) {
        this.damage(pokemon.baseMaxhp / (pokemon.hasType(['Water', 'Steel']) ? 8 : 16));
    },

so a sixteenth, and an eighth for Water and Steel types -- half of the base game's 1/8 and
1/4, which the resolver and the port both still used. Against a 150 HP target Showdown took
9 and Python 18.

Every case is played by Showdown first and the Python turn, and the port's, are held to its
HP. One target of each kind: Incineroar (neither type), Rotom-Wash (Water), Corviknight
(Steel). The control is a Body Press into the same Incineroar -- the same Garganacl, the
same turn, no Salt Cure -- which was right before and has to stay right.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int, item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
        item=item,
    )


TEAM_A = [
    _mon("Garganacl", "Purifying Salt", ["saltcure", "bodypress", "protect", "recover"], 20),
    _mon("Kingambit", "Defiant", ["protect", "kowtowcleave", "suckerpunch", "swordsdance"], 21),
]

TEAM_B = [
    _mon("Incineroar", "Blaze", ["bulkup", "protect", "knockoff", "flareblitz"], 2),
    _mon("Rotom-Wash", "Levitate", ["nastyplot", "protect", "hydropump", "thunderbolt"], 3),
    _mon("Corviknight", "Unnerve", ["bulkup", "protect", "bravebird", "roost"], 4),
    _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake", "dragonclaw"], 5),
]

#: name -> (side 1's team order, side 0's choice, the residual's denominator, or None for
#: no Salt Cure). Side 1's lead sets up; its partner, Garchomp, Protects.
CASES = {
    "neither-type": ("team 1423", "move 1 1, move 1", 16),
    "water": ("team 2413", "move 1 1, move 1", 8),
    "steel": ("team 3412", "move 1 1, move 1", 8),
    "control-body-press": ("team 1423", "move 2 1, move 1", None),
}
SIDE_B = "move 1, move 1"

BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1} {mon.species}": (mon.hp, mon.maxhp)
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict[str, tuple], list[str]]:
    order, choice, _denominator = CASES[name]
    choices = [choice, SIDE_B]
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 12", order])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    return before, choices, theirs, log


def _check_showdown(name: str, before: Position, log: list[str]) -> None:
    """Showdown's own residual line is the mod's fraction -- the positive control."""
    denominator = CASES[name][2]
    target = before.sides[1].pokemon[0]
    # The log carries each line twice, in exact HP and in percent; read the exact one.
    exact = [
        line
        for line in log
        if line.startswith("|-damage|p2a") and line.split("|")[3].endswith(f"/{target.maxhp}")
    ]
    residual = [line for line in exact if "Salt Cure" in line]
    if denominator is None:
        assert residual == [], residual
        return
    assert len(residual) == 1, log
    hits = [int(line.split("|")[3].split("/")[0]) for line in exact if "Salt Cure" not in line]
    after = int(residual[0].split("|")[3].split("/")[0])
    assert hits[-1] - after == target.maxhp // denominator, (residual, target.maxhp)


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_takes_the_mods_fraction(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    """The port plays Showdown's turn, and every branch it has is Showdown's position."""
    before, choices, theirs, log = _play(oracle, name)
    _check_showdown(name, before, log)
    # Showdown's positions carry its final stats, which the port refuses as it would a
    # transformed Pokemon's. Nobody here is transformed, so they follow from the spreads.
    for side in before.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    actions = _actions(reg, before, choices)
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    assert there.branches
    for index in range(len(there.branches)):
        chosen = node.resolve(before, actions, BUDGET, select=index)
        assert chosen is not None and chosen.position is not None
        assert _state(chosen.position) == theirs, (name, index)
