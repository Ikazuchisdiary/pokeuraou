"""A non-Ghost Curse takes no target (IKA-168).

Curse's dex entry is `target: "normal"`, but Showdown never asks a non-Ghost for one:
`Pokemon.getMoves` (sim/pokemon.ts) writes the request's target as

    case 'curse': if (!this.hasType('Ghost')) target = 'self';

and `Side.chooseMove` validates the choice against that request target, so "move N 1"
is refused ("You can't choose a target for Curse") and "move N" is the only legal
string. A Ghost's request keeps "normal". The enumeration read the dump's target alone
and offered the non-Ghost user the two foe targets and not the untargeted choice.

Until Showdown 2345119 the dump carried the same rule as `nonGhostTarget: "self"`;
a5df827 dropped the key (the champions mod pins the target in `onModifyMove` and the
queue instead), so the rule is written here as Showdown writes it, on the current types.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import MoveAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from .conftest import FORMAT_ID
from .test_actions import _assert_enumeration_matches, _synthetic_position

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str]) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP))


TEAM_A = [
    _mon("Snorlax", "Thick Fat", ["curse", "protect", "bodyslam", "rest"]),
    _mon("Gengar", "Cursed Body", ["curse", "protect", "shadowball", "sludgebomb"]),
    _mon("Garchomp", "Rough Skin", ["earthquake", "dragonclaw", "protect", "rockslide"]),
    _mon("Kingambit", "Defiant", ["protect", "ironhead", "suckerpunch", "swordsdance"]),
]
TEAM_B = [
    _mon("Gengar", "Cursed Body", ["curse", "protect", "shadowball", "sludgebomb"]),
    _mon("Snorlax", "Thick Fat", ["curse", "protect", "bodyslam", "rest"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "knockoff", "protect"]),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"]),
]


def _curse_targets(reg, pos: Position, side: int, slot: int) -> set[int | None]:  # noqa: ANN001
    return {
        piece.target
        for action in side_actions(reg, pos, side)
        for piece in action.slots
        if isinstance(piece, MoveAction) and piece.slot == slot and piece.move_id == "curse"
    }


@pytest.mark.oracle
def test_showdown_asks_a_non_ghost_for_no_target(oracle: Oracle) -> None:
    """The fact, with its control: the Ghost beside the Snorlax keeps a target."""
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    targets = [
        {m["id"]: m.get("target") for m in active["moves"]}["curse"]
        for active in handle.requests[0]["active"]
    ]
    assert targets == ["self", "normal"], targets
    probed = {r["choice"]: r["ok"] for r in handle.probe(0, ["move 1, move 2", "move 1 1, move 2"])}
    assert probed == {"move 1, move 2": True, "move 1 1, move 2": False}, probed
    handle.close()


@pytest.mark.oracle
def test_the_enumeration_is_what_showdown_accepts(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """Both sides lead a non-Ghost and a Ghost Curse user; every string is probed."""
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    pos = Position.from_json(handle.position)
    for side in (0, 1):
        _assert_enumeration_matches(reg, handle, pos, side)
    handle.close()


def test_the_rule_follows_the_current_types(reg) -> None:  # noqa: ANN001
    """No oracle: Showdown's `hasType` reads the types as they are now."""
    pos = _synthetic_position(reg, TEAM_A)
    snorlax, gengar = (pos.sides[0].pokemon[i] for i in pos.sides[0].active)
    assert (snorlax.species, gengar.species) == ("snorlax", "gengar")
    assert _curse_targets(reg, pos, 0, 0) == {None}
    assert _curse_targets(reg, pos, 0, 1) == {1, 2}
    # A Snorlax turned Ghost (Trick-or-Treat, a Protean user) chooses a target; a Gengar
    # that lost the type (Soak) does not.
    snorlax.types = ("Normal", "Ghost")
    gengar.types = ("Water",)
    assert _curse_targets(reg, pos, 0, 0) == {1, 2}
    assert _curse_targets(reg, pos, 0, 1) == {None}
