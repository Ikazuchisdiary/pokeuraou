"""A foeSide move lays its side condition on the foe's side (IKA-165).

Showdown gives a side move one target, and the condition goes to that target's side
(sim/battle-actions.ts `moveHit`):

    if (moveData.sideCondition) {
        hitResult = target.side.addSideCondition(moveData.sideCondition, source, move);

and the target is chosen by `getMoveTargets` (sim/pokemon.ts):

    case 'foeSide': case 'allySide': case 'allyTeam':
        if (!move.target.startsWith('foe')) targets.push(...this.alliesAndSelf());
        if (!move.target.startsWith('ally')) targets.push(...this.foes(true));

so Stealth Rock, Spikes, Toxic Spikes and Sticky Web (target `foeSide`) land on the foe's
side, and Tailwind and the screens (target `allySide`) on the user's. The resolver and the
port put every one on the user's side, so a Spikes user spiked itself.

Each case plays two turns in Showdown. Turn 1: Glimmora uses the move while everyone else
Protects (or Helping Hands). Turn 2: side 1's Incineroar switches to Hippowdon -- grounded,
neither Poison nor Steel, so every hazard reaches it -- and nothing attacks. The Python turn,
and the port's, are held to Showdown's side conditions after turn 1 and, played on from
their own turn 1, to its HP, status and boosts after turn 2. The controls are Tailwind and
Reflect from the same Glimmora, which were right before and have to stay right.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.resolve import Budget, resolve_turn

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
    )


WHIMSICOTT = _mon("Whimsicott", "Chlorophyll", ["helpinghand", "protect", "tailwind", "reflect"], 21)
TEAM_A = [
    _mon("Glimmora", "Corrosion", ["stealthrock", "spikes", "toxicspikes", "protect"], 20),
    WHIMSICOTT,
]
#: Sticky Web, and the controls: the same user, an allySide move.
TEAM_A_WEB = [
    _mon("Glimmora", "Corrosion", ["stickyweb", "protect", "tailwind", "reflect"], 20),
    WHIMSICOTT,
]

TEAM_B = [
    _mon("Incineroar", "Blaze", ["protect", "bulkup", "knockoff", "flareblitz"], 2),
    _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake", "dragonclaw"], 5),
    _mon("Hippowdon", "Sand Force", ["protect", "slackoff", "earthquake", "stealthrock"], 3),
    _mon("Rotom-Wash", "Levitate", ["protect", "nastyplot", "hydropump", "thunderbolt"], 4),
]

#: name -> (side 0's team, its turn-1 choice, the condition, whose side it belongs on).
CASES = {
    "stealthrock": (TEAM_A, "move 1, move 1 -1", "stealthrock", 1),
    "spikes": (TEAM_A, "move 2, move 1 -1", "spikes", 1),
    "toxicspikes": (TEAM_A, "move 3, move 1 -1", "toxicspikes", 1),
    "stickyweb": (TEAM_A_WEB, "move 1, move 1 -1", "stickyweb", 1),
    "control-tailwind": (TEAM_A_WEB, "move 3, move 1 -1", "tailwind", 0),
    "control-reflect": (TEAM_A_WEB, "move 4, move 1 -1", "reflect", 0),
}
TURN1_B = "move 1, move 1"
#: Turn 2: side 0 Protects twice (neither did on turn 1), side 1's Incineroar goes out for
#: Hippowdon and Garchomp sets up. Nothing takes damage but from a hazard.
TURN2_A = "move {protect}, move 2"
TURN2_B = "switch 3, move 2"

BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _conditions(pos: Position) -> list[list[tuple]]:
    return [sorted((c.id, c.layers or 1) for c in side.side_conditions) for side in pos.sides]


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1} {mon.species}": (mon.hp, mon.status, tuple(sorted((mon.boosts or {}).items())))
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _turn2_a(name: str) -> str:
    team = CASES[name][0]
    return TURN2_A.format(protect=team[0].moves.index("protect") + 1)


def _play(oracle: Oracle, name: str) -> tuple[Position, Position, Position]:
    team, choice, _cid, _owner = CASES[name]
    handle = oracle.create(FORMAT_ID, team, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 12", "team 1234"])
    before = Position.from_json(handle.position)
    handle.step([choice, TURN1_B])
    assert handle.choice_errors == [], handle.choice_errors
    middle = Position.from_json(handle.position)
    handle.step([_turn2_a(name), TURN2_B])
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    handle.close()
    return before, middle, after


def _check_showdown(name: str, middle: Position, after: Position) -> None:
    """Showdown's own side is the move's target's -- the positive control."""
    _team, _choice, cid, owner = CASES[name]
    conditions = _conditions(middle)
    assert cid in [c for c, _ in conditions[owner]], conditions
    assert cid not in [c for c, _ in conditions[1 - owner]], conditions
    hippowdon = next(m for m in after.sides[1].pokemon if m.species == "hippowdon")
    fresh = hippowdon.hp == hippowdon.maxhp and not hippowdon.status and not any(
        (hippowdon.boosts or {}).values()
    )
    # A hazard shows on the switch-in; an allySide move on side 0 leaves it untouched.
    assert fresh == (owner == 0), (name, hippowdon)


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


def _only(result) -> Position:  # noqa: ANN001
    assert len(result.branches) == 1, [b.events for b in result.branches]
    return result.branches[0].position


@pytest.mark.parametrize("name", sorted(CASES))
def test_side_condition_lands_on_the_targets_side(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    before, middle, after = _play(oracle, name)
    _check_showdown(name, middle, after)
    choice = CASES[name][1]

    mine = _only(resolve_turn(reg, before, _actions(reg, before, [choice, TURN1_B]), budget=BUDGET))
    assert _conditions(mine) == _conditions(middle), (name, _conditions(mine), _conditions(middle))

    turn2 = [_turn2_a(name), TURN2_B]
    played_on = _only(resolve_turn(reg, mine, _actions(reg, mine, turn2), budget=BUDGET))
    assert _state(played_on) == _state(after), (name, _state(played_on), _state(after))
    assert _conditions(played_on) == _conditions(after), name


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


def _clear_stats(pos: Position) -> Position:
    # Showdown's positions carry its final stats, which the port refuses as it would a
    # transformed Pokemon's. Nobody here is transformed, so they follow from the spreads.
    for side in pos.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    return pos


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_lays_it_on_the_targets_side(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    """The port plays Showdown's two turns too, and matches Python's positions."""
    before, middle, after = _play(oracle, name)
    before = _clear_stats(before)
    choice = CASES[name][1]
    node = rustnode.node_for(reg)
    assert node is not None

    actions = _actions(reg, before, [choice, TURN1_B])
    there = node.resolve(before, actions, BUDGET, select=0)
    assert there is not None and there.position is not None, "the port refused turn 1"
    mine = there.position
    assert _conditions(mine) == _conditions(middle), (name, _conditions(mine), _conditions(middle))

    turn2 = [_turn2_a(name), TURN2_B]
    played = node.resolve(mine, _actions(reg, mine, turn2), BUDGET, select=0)
    assert played is not None and played.position is not None, "the port refused turn 2"
    assert _state(played.position) == _state(after), (name, _state(played.position), _state(after))

    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    here = _only(resolve_turn(reg, before, actions, budget=BUDGET))
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    assert here.to_json() == mine.to_json(), name
