"""Expanding Force spreads under Psychic Terrain, Air Balloon pops, Terrain Pulse needs the
ground (IKA-205).

Showdown a5df827, data/moves.ts and data/items.ts (the champions mod overrides none)::

    expandingforce: {
        onBasePower(basePower, source) {
            if (this.field.isTerrain('psychicterrain') && source.isGrounded()) {
                return this.chainModify(1.5);
            }
        },
        onModifyMove(move, source, target) {
            if (this.field.isTerrain('psychicterrain') && source.isGrounded()) {
                move.target = 'allAdjacentFoes';
            }
        },
        target: "normal",
    },
    terrainpulse: {
        onModifyType(move, pokemon) { if (!pokemon.isGrounded()) return; switch (this.field.terrain) ... },
        onModifyMove(move, pokemon) {
            if (this.field.terrain && pokemon.isGrounded()) move.basePower *= 2;
        },
    },
    airballoon: {
        onDamagingHit(damage, target, source, move) {
            this.add('-enditem', target, 'Air Balloon');
            target.item = '';
            ...
        },
        onAfterSubDamage(damage, target, source, effect) {
            if (effect.effectType === 'Move') { ...the same }
        },
    },

`useMoveInner` (sim/battle-actions.ts) runs `ModifyMove` before `getMoveTargets`, so from a
grounded user under Psychic Terrain the move is a spread move from the start: it is not
redirected, it hits both foes whichever target was chosen (the ally included), it takes
the spread 0.75 when two foes are there (`spreadHit`, set from the targets before Protect
drops any), and Wide Guard (`move.target === 'allAdjacentFoes'`) stops it. The request
still asks for a target, so the menu keeps one choice per target.

The port had only the 1.5, and without the grounded check; the balloon never popped, so a
Ground move after any hit still missed its holder; Terrain Pulse took the terrain's type and
doubled power from a user in the air.

Every case is played by Showdown first, and asserts what Showdown did before the port is
held to it: the HP, item and faint of all four active Pokemon after the turn.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Pokemon, Position

from ._port import Budget
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

FAST = {"hp": 20, "atk": 32, "def": 0, "spa": 32, "spd": 0, "spe": 32}
SLOW = {"hp": 32, "atk": 32, "def": 0, "spa": 0, "spd": 0, "spe": 0}
BULKY = {"hp": 32, "atk": 0, "def": 32, "spa": 0, "spd": 32, "spe": 0}
#: The ids this file is about: the port's notes must not name them.
IDS = ("expandingforce", "airballoon", "terrainpulse")


def _mon(species: str, ability: str, moves: list[str], sp: dict, item: str | None = None) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves,
                   sp=dict(sp), item=item)


def _p(pos: Position, side: int, slot: int) -> Pokemon:
    return pos.sides[side].pokemon[pos.sides[side].active[slot]]


def _loaded(raw: dict) -> Position:
    pos = Position.from_json(raw)
    for side in pos.sides:
        for mon in side.pokemon:
            mon.trapped = False
    return pos


def _chosen(reg, pos: Position, step: tuple[str, str]):  # noqa: ANN001, ANN202
    """The menu's actions for the choices. The menu offers a `normal` move no ally target
    (actions.py), so "move 1 -2" is built from the menu's "move 1 1" with the target moved."""
    chosen = []
    for side, choice in enumerate(step):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        at_ally = choice.startswith("move 1 -2,")
        key = choice.replace("move 1 -2,", "move 1 1,") if at_ally else choice
        assert key in menu, (choice, sorted(menu))
        action = menu[key]
        if at_ally:
            action = replace(action, slots=(replace(action.slots[0], target=-2), *action.slots[1:]))
            assert action.to_choice() == choice
        chosen.append(action)
    return chosen


def _board(pos: Position) -> list[tuple[str, int, str | None, bool]]:
    return [
        (_p(pos, side, slot).species, _p(pos, side, slot).hp, _p(pos, side, slot).item,
         _p(pos, side, slot).fainted)
        for side in (0, 1) for slot in (0, 1)
    ]


# ---------------------------------------------------------------------------
# The Pokemon.

#: Psychic Surge puts the terrain up before turn 1; Inner Focus is the control.
INDEEDEE = _mon("Indeedee", "Psychic Surge", ["expandingforce", "protect", "terrainpulse", "calmmind"], FAST)
PLAIN_INDEEDEE = replace(INDEEDEE, ability="Inner Focus")
FLOATING_INDEEDEE = replace(INDEEDEE, item="airballoon")
#: Not Dark: Expanding Force aimed at it has to land when there is no terrain.
GUARD = _mon("Hippowdon", "Sand Force", ["protect", "calmmind", "slackoff", "highhorsepower"], SLOW)
#: Faster than Indeedee, to knock p2b out before Expanding Force moves.
CHOMP = _mon("Garchomp", "Rough Skin", ["stompingtantrum", "protect", "dragonclaw", "earthquake"], FAST)
#: Fire/Rock, 4x weak to Ground.
FRAIL = _mon("Arcanine-Hisui", "Intimidate", ["calmmind", "protect", "flareblitz", "rockslide"],
             {"hp": 0, "def": 0, "spd": 0})
P2A = _mon("Milotic", "Marvel Scale", ["calmmind", "protect", "substitute", "icebeam"], BULKY)
P2B = _mon("Sylveon", "Pixilate", ["calmmind", "protect", "wideguard", "followme"], BULKY)
FLOATING_P2A = replace(P2A, item="airballoon")
#: Slower than Milotic's partner's attacker, for the Ground hit after the balloon pops.
HIPPO = _mon("Hippowdon", "Sand Force", ["highhorsepower", "protect", "slackoff", "calmmind"], SLOW)

P1_BENCH = [
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], SLOW),
    _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"], SLOW),
]
P2_BENCH = [
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"], SLOW),
    _mon("Gengar", "Cursed Body", ["shadowball", "protect", "sludgebomb", "hypnosis"], SLOW),
]


def _hurt(b: Position, a: Position, side: int, slot: int) -> bool:
    return _p(a, side, slot).hp < _p(b, side, slot).hp


def _item(a: Position, side: int, slot: int) -> str | None:
    return _p(a, side, slot).item


@dataclasses.dataclass(frozen=True)
class Case:
    p1: tuple[TeamSet, TeamSet]
    p2: tuple[TeamSet, TeamSet]
    #: Every step's (p1, p2) choices; the port is held to the last one.
    steps: tuple[tuple[str, str], ...]
    #: What Showdown did, read from the positions around the last step.
    fact: Callable[[Position, Position], bool]


CALM = "move 1, move 1"  # p2: Calm Mind x2


def _cases() -> dict[str, Case]:
    ef = (INDEEDEE, GUARD)
    plain = (PLAIN_INDEEDEE, GUARD)
    p2 = (P2A, P2B)
    return {
        # Expanding Force under Psychic Terrain from a grounded user: both foes, whichever
        # target was chosen.
        "ef, terrain, at p2a": Case(ef, p2, (("move 1 1, move 1", CALM),),
                                    lambda b, a: _hurt(b, a, 1, 0) and _hurt(b, a, 1, 1)),
        "ef, terrain, at p2b": Case(ef, p2, (("move 1 2, move 1", CALM),),
                                    lambda b, a: _hurt(b, a, 1, 0) and _hurt(b, a, 1, 1)),
        "ef, terrain, at the ally": Case(ef, p2, (("move 1 -2, move 2", CALM),),
                                         lambda b, a: _hurt(b, a, 1, 0) and _hurt(b, a, 1, 1)
                                         and not _hurt(b, a, 0, 1)),
        # p2b's Protect does not undo the spread 0.75 on p2a.
        "ef, terrain, p2b protects": Case(ef, p2, (("move 1 1, move 1", "move 1, move 2"),),
                                          lambda b, a: _hurt(b, a, 1, 0) and not _hurt(b, a, 1, 1)),
        "ef, terrain, wide guard": Case(ef, p2, (("move 1 1, move 1", "move 1, move 3"),),
                                        lambda b, a: not _hurt(b, a, 1, 0) and not _hurt(b, a, 1, 1)),
        "ef, terrain, follow me": Case(ef, p2, (("move 1 1, move 1", "move 1, move 4"),),
                                       lambda b, a: _hurt(b, a, 1, 0) and _hurt(b, a, 1, 1)),
        # The user in the air: one target, and neither the 1.5 nor the terrain's 1.3.
        "ef, terrain, balloon user": Case((FLOATING_INDEEDEE, GUARD), p2, (("move 1 1, move 1", CALM),),
                                          lambda b, a: _hurt(b, a, 1, 0) and not _hurt(b, a, 1, 1)),
        # One foe left: no spread 0.75 (Garchomp knocks p2b out first).
        "ef, terrain, one foe left": Case((INDEEDEE, CHOMP), (P2A, FRAIL),
                                          (("move 1 1, move 1 2", CALM),),
                                          lambda b, a: _hurt(b, a, 1, 0) and _p(a, 1, 1).fainted),
        # Controls: no terrain, the move is single-target and redirectable.
        "ef, no terrain, at p2a": Case(plain, p2, (("move 1 1, move 1", CALM),),
                                       lambda b, a: _hurt(b, a, 1, 0) and not _hurt(b, a, 1, 1)),
        "ef, no terrain, at the ally": Case(plain, p2, (("move 1 -2, move 2", CALM),),
                                            lambda b, a: _hurt(b, a, 0, 1) and not _hurt(b, a, 1, 0)),
        "ef, no terrain, wide guard": Case(plain, p2, (("move 1 1, move 1", "move 1, move 3"),),
                                           lambda b, a: _hurt(b, a, 1, 0)),
        "ef, no terrain, follow me": Case(plain, p2, (("move 1 1, move 1", "move 1, move 4"),),
                                          lambda b, a: _hurt(b, a, 1, 1) and not _hurt(b, a, 1, 0)),
        # Terrain Pulse: Psychic and doubled from a grounded user, Normal and 50 from the air.
        "terrain pulse, grounded": Case(ef, p2, (("move 3 1, move 1", CALM),),
                                        lambda b, a: _hurt(b, a, 1, 0)),
        "terrain pulse, balloon user": Case((FLOATING_INDEEDEE, GUARD), p2, (("move 3 1, move 1", CALM),),
                                            lambda b, a: _hurt(b, a, 1, 0)),
        "terrain pulse, no terrain": Case(plain, p2, (("move 3 1, move 1", CALM),),
                                          lambda b, a: _hurt(b, a, 1, 0)),
        # Air Balloon: any damaging hit pops it, and a Ground move then lands.
        "balloon, hit": Case((CHOMP, HIPPO), (FLOATING_P2A, P2B), (("move 3 1, move 2", CALM),),
                             lambda b, a: _hurt(b, a, 1, 0) and _item(a, 1, 0) is None),
        "balloon, hit then ground": Case((CHOMP, HIPPO), (FLOATING_P2A, P2B), (("move 3 1, move 1 1", CALM),),
                                         lambda b, a: _item(a, 1, 0) is None),
        "balloon, under a substitute": Case(
            (CHOMP, HIPPO), (FLOATING_P2A, P2B),
            (("move 2, move 2", "move 3, move 1"), ("move 3 1, move 3", CALM)),
            lambda b, a: not _hurt(b, a, 1, 0) and _item(b, 1, 0) == "airballoon"
            and _item(a, 1, 0) is None),
        # Controls: a Ground move misses the holder and leaves it; no hit leaves it.
        "balloon, ground only": Case((CHOMP, HIPPO), (FLOATING_P2A, P2B), (("move 1 1, move 2", CALM),),
                                     lambda b, a: not _hurt(b, a, 1, 0) and _item(a, 1, 0) == "airballoon"),
        "balloon, no hit": Case((CHOMP, HIPPO), (FLOATING_P2A, P2B), (("move 2, move 2", CALM),),
                                lambda b, a: _item(a, 1, 0) == "airballoon"),
        "no balloon, hit then ground": Case((CHOMP, HIPPO), p2, (("move 3 1, move 1 1", CALM),),
                                            lambda b, a: _hurt(b, a, 1, 0)),
    }


CASES = _cases()


def _play(oracle: Oracle, case: Case) -> tuple[dict, dict]:
    handle = oracle.create(FORMAT_ID, [*case.p1, *P1_BENCH], [*case.p2, *P2_BENCH],
                           policy=RandomnessPolicy(damage_roll=0))
    handle.step(["team 1234", "team 1234"])
    before = handle.position
    for step in case.steps:
        before = handle.position
        handle.step(list(step))
        assert handle.choice_errors == [], handle.choice_errors
    after = handle.position
    handle.close()
    return before, after


@pytest.mark.parametrize("name", sorted(CASES))
def test_showdown(oracle: Oracle, name: str) -> None:
    before, after = _play(oracle, CASES[name])
    b, a = Position.from_json(before), Position.from_json(after)
    assert CASES[name].fact(b, a), (_board(b), _board(a))


def test_showdown_spread_hits_the_same_whichever_target(oracle: Oracle) -> None:
    """The chosen target does not matter once the move spreads (the menu keeps all three)."""
    boards = {n: _board(Position.from_json(_play(oracle, CASES[n])[1]))
              for n in ("ef, terrain, at p2a", "ef, terrain, at p2b")}
    assert boards["ef, terrain, at p2a"] == boards["ef, terrain, at p2b"], boards


def test_showdown_balloon_lets_the_ground_move_land(oracle: Oracle) -> None:
    hit = Position.from_json(_play(oracle, CASES["balloon, hit"])[1])
    both = Position.from_json(_play(oracle, CASES["balloon, hit then ground"])[1])
    assert _p(both, 1, 0).hp < _p(hit, 1, 0).hp


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, port, name: str) -> None:  # noqa: ANN001
    from ._port_showdown import port_turn, port_weights

    case = CASES[name]
    before, after = _play(oracle, case)
    start = _loaded(before)
    chosen = _chosen(reg, start, case.steps[-1])
    pinned = port_turn(port, start, chosen)
    assert _board(pinned) == _board(Position.from_json(after)), name
    notes = port_weights(port, start, chosen, Budget.deterministic(0)).get("unmodelled") or []
    assert [n for n in notes if any(i in n for i in IDS)] == []
