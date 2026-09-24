"""Hazards laid by a hit, and a hazard move that has nothing left to add (IKA-173).

Stone Axe and Ceaseless Edge lay their hazard from the move's own `onAfterHit`
(vendor/pokemon-showdown/data/moves.ts, `stoneaxe` and `ceaselessedge`):

    onAfterHit(target, source, move) {
        if (!move.hasSheerForce) {
            for (const side of source.side.foeSidesWithConditions()) {
                side.addSideCondition('stealthrock');

and `spreadMoveHit` calls it for every target that took numeric damage, after
`DamagingHit` and only while the user stands (sim/battle-actions.ts):

    if (moveData.onAfterHit && pokemon.hp) {
        for (const t of damagedTargets) {
            this.battle.singleEvent('AfterHit', moveData, {}, t, pokemon, move);

So a hit lays it on the user's foe's side -- whoever it hit, an ally included, and a
target it knocked out included -- and a miss, a Protect or Sheer Force lays nothing.
The resolver and the port laid nothing at all.

Toxic Debris (data/abilities.ts) lays Toxic Spikes on the attacker's side, or on the
attacker's foe's side when the attacker is an ally, from `onDamagingHit` -- which runs
before the faint is processed, so a Glimmora knocked out by the hit still lays them:

    const side = source.isAlly(target) ? source.side.foe : source.side;

The resolver and the port did it only for a foe's hit on a Glimmora that survived.

A hazard move with nothing to add fails: `addSideCondition` returns false for a condition
that is up and has no `onSideRestart`, and Spikes and Toxic Spikes return false from it
at three and two layers (sim/side.ts, data/moves.ts). The move then fails, which is
`moveLastTurnResult === false` -- what Stomping Tantrum reads. Tailwind and the screens
fail the same way. The resolver and the port let every one of them succeed.

Each case plays Showdown's turns and holds Python's, and the port's, played on from their
own previous turn, to Showdown's side conditions, HP, status, boosts and failed-move flags
after every turn. A case that lays a hazard ends with side 1's Incineroar switching out
for Hippowdon, so the hazard shows on the switch-in; the controls leave it untouched.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, SideAction, side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget
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
KLEAVOR = _mon("Kleavor", "Sharpness", ["stoneaxe", "protect", "xscissor", "closecombat"], 19)
KLEAVOR_SHEER = _mon("Kleavor", "Sheer Force", ["stoneaxe", "protect", "xscissor", "closecombat"], 19)
SAMUROTT = _mon("Samurott-Hisui", "Sharpness", ["ceaselessedge", "protect", "razorshell", "aquajet"], 19)
GLIMMORA_DEBRIS = _mon("Glimmora", "Toxic Debris", ["protect", "powergem", "earthpower", "sludgebomb"], 20)
GARCHOMP_A = _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake", "dragonclaw"], 5)
GLIMMORA_HAZARDS = _mon("Glimmora", "Toxic Debris", ["stealthrock", "protect", "spikes", "toxicspikes"], 20)

TEAM_B = [
    _mon("Incineroar", "Blaze", ["protect", "bulkup", "knockoff", "flareblitz"], 2),
    _mon("Garchomp", "Rough Skin", ["protect", "swordsdance", "earthquake", "dragonclaw"], 5),
    _mon("Hippowdon", "Sand Force", ["protect", "slackoff", "earthquake", "stealthrock"], 3),
    _mon("Rotom-Wash", "Levitate", ["protect", "nastyplot", "hydropump", "thunderbolt"], 4),
]

#: Turn 2 of a hazard case: side 0 Protects twice, side 1's Incineroar goes out for
#: Hippowdon and Garchomp sets up. Nothing takes damage but from a hazard.
SWITCH_IN = ("move 2, move 2", "switch 3, move 2")


class Case:
    def __init__(
        self,
        team: list[TeamSet],
        turns: list[tuple[str, str]],
        conditions: list[list[tuple[str, int]]],
        *,
        accuracy: str = "hit",
        failed: str | None = None,
    ) -> None:
        self.team = team
        self.turns = turns
        #: Showdown's side conditions after the last turn -- the positive control.
        self.conditions = conditions
        self.accuracy = accuracy
        #: The side-0 species whose move fails on the last turn, if any.
        self.failed = failed


ROCK_P2 = [[], [("stealthrock", 1)]]
NONE = [[], []]
T1_STONE_AXE = ("move 1 2, move 1 -1", "move 2, move 2")  # into Garchomp; Helping Hand

CASES = {
    "stoneaxe": Case([KLEAVOR, WHIMSICOTT], [T1_STONE_AXE, SWITCH_IN], ROCK_P2),
    # Knocked out by the hit: numeric damage all the same.
    "stoneaxe-ko": Case([KLEAVOR, WHIMSICOTT], [("move 1 1, move 1 -1", "move 2, move 2")], ROCK_P2),
    # Into its own partner: the rocks still go to the foe's side.
    "stoneaxe-ally": Case(
        [KLEAVOR, WHIMSICOTT], [("move 1 -2, move 1 -1", "move 2, move 2"), SWITCH_IN], ROCK_P2
    ),
    "ceaselessedge": Case(
        [SAMUROTT, WHIMSICOTT],
        [("move 1 1, move 1 -1", "move 2, move 2"), SWITCH_IN],
        [[], [("spikes", 1)]],
    ),
    "control-stoneaxe-miss": Case(
        [KLEAVOR, WHIMSICOTT], [T1_STONE_AXE, SWITCH_IN], NONE, accuracy="miss"
    ),
    "control-stoneaxe-protect": Case(
        [KLEAVOR, WHIMSICOTT], [("move 1 2, move 1 -1", "move 2, move 1"), SWITCH_IN], NONE
    ),
    "control-stoneaxe-sheerforce": Case([KLEAVOR_SHEER, WHIMSICOTT], [T1_STONE_AXE, SWITCH_IN], NONE),
    # Kleavor's X-Scissor into its partner Glimmora: Toxic Spikes on the foe's side.
    "toxicdebris-ally": Case(
        [GLIMMORA_DEBRIS, KLEAVOR],
        [("move 2 1, move 3 -1", "move 1, move 2"), ("move 1, move 2", "switch 3, move 2")],
        [[], [("toxicspikes", 1)]],
    ),
    # The partner's Earthquake, the way a recorded game would meet it: Glimmora is knocked
    # out and the Toxic Spikes go to the foe's side.
    "toxicdebris-ally-earthquake": Case(
        [GLIMMORA_DEBRIS, GARCHOMP_A],
        [("move 2 1, move 3", "move 1, move 1")],
        [[], [("toxicspikes", 1)]],
    ),
    # Garchomp's Earthquake knocks Glimmora out: Toxic Spikes on Garchomp's side.
    "toxicdebris-ko": Case(
        [GLIMMORA_DEBRIS, WHIMSICOTT],
        [("move 2 1, move 2", "move 1, move 3")],
        [[], [("toxicspikes", 1)]],
    ),
    # A foe's Knock Off into a Glimmora that survives: right before, and has to stay right.
    "control-toxicdebris-foe": Case(
        [GLIMMORA_DEBRIS, WHIMSICOTT],
        [("move 2 2, move 2", "move 3 1, move 1"), ("move 1, move 1 -1", "switch 3, move 2")],
        [[], [("toxicspikes", 1)]],
    ),
    "stealthrock-twice": Case(
        [GLIMMORA_HAZARDS, WHIMSICOTT],
        [("move 1, move 2", "move 1, move 2"), ("move 1, move 1 -1", "move 2, move 1")],
        ROCK_P2,
        failed="glimmora",
    ),
    "toxicspikes-thrice": Case(
        [GLIMMORA_HAZARDS, WHIMSICOTT],
        [
            ("move 4, move 2", "move 1, move 2"),
            ("move 4, move 1 -1", "move 2, move 1"),
            ("move 4, move 2", "move 1, move 2"),
        ],
        [[], [("toxicspikes", 2)]],
        failed="glimmora",
    ),
    "spikes-four": Case(
        [GLIMMORA_HAZARDS, WHIMSICOTT],
        [
            ("move 3, move 2", "move 1, move 2"),
            ("move 3, move 1 -1", "move 2, move 1"),
            ("move 3, move 2", "move 1, move 2"),
            ("move 3, move 1 -1", "move 2, move 1"),
        ],
        [[], [("spikes", 3)]],
        failed="glimmora",
    ),
    "tailwind-twice": Case(
        [GLIMMORA_HAZARDS, WHIMSICOTT],
        [("move 2, move 3", "move 1, move 2"), ("move 3, move 3", "move 2, move 1")],
        [[("tailwind", 1)], [("spikes", 1)]],
        failed="whimsicott",
    ),
}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)
#: The miss control: accuracy enumerated, and the miss is the second branch.
BUDGET_ACCURACY = replace(BUDGET, enumerate_accuracy=True)


def _conditions(pos: Position) -> list[list[tuple]]:
    return [sorted((c.id, c.layers or 1) for c in side.side_conditions) for side in pos.sides]


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1} {mon.species}": (
            mon.hp,
            mon.status,
            tuple(sorted((mon.boosts or {}).items())),
            # Showdown's `clearVolatile` on the faint forgets `moveLastTurnResult`; the
            # resolver keeps it, and a fainted Pokemon never reads it.
            mon.move_last_turn_failed and not mon.fainted,
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> list[Position]:
    """Showdown's position before the first turn and after every turn."""
    case = CASES[name]
    handle = oracle.create(FORMAT_ID, case.team, TEAM_B, policy=RandomnessPolicy(accuracy=case.accuracy))
    handle.step(["team 12", "team 1234"])
    out = [Position.from_json(handle.position)]
    for choices in case.turns:
        handle.step(list(choices))
        assert handle.choice_errors == [], (name, choices, handle.choice_errors)
        out.append(Position.from_json(handle.position))
    handle.close()
    return out


def _check_showdown(name: str, positions: list[Position]) -> None:
    """Showdown's own answer: the positive control for every case."""
    case = CASES[name]
    last = positions[-1]
    assert _conditions(last) == case.conditions, (name, _conditions(last))
    failed = sorted(m.species for m in last.sides[0].pokemon if m.move_last_turn_failed)
    assert failed == ([case.failed] if case.failed else []), (name, failed)
    if case.turns[-1][1].startswith("switch 3"):
        hippowdon = next(m for m in last.sides[1].pokemon if m.species == "hippowdon")
        fresh = hippowdon.hp == hippowdon.maxhp and not hippowdon.status
        assert fresh == (case.conditions[1] == []), (name, hippowdon)


def _side_action(reg, pos: Position, side: int, choice: str) -> SideAction:  # noqa: ANN001
    listed = side_actions(reg, pos, side)
    for action in listed:
        if action.to_choice() == choice:
            return action
    # A single-target move into the partner is legal in Showdown but not in the action
    # list, so it is built from the listed move with the target swapped.
    slots = []
    for index, part in enumerate(choice.split(", ")):
        same = [a.slots[index] for a in listed if a.slots[index].to_choice() == part]
        if same:
            slots.append(same[0])
            continue
        _move, move_index, target = part.split()
        listed_move = next(
            a.slots[index]
            for a in listed
            if isinstance(a.slots[index], MoveAction) and a.slots[index].move_index == int(move_index)
        )
        slots.append(replace(listed_move, target=int(target)))
    built = SideAction(slots=tuple(slots))
    assert built.to_choice() == choice, (built.to_choice(), choice)
    return built


def _actions(reg, pos: Position, choices: tuple[str, str]) -> list:  # noqa: ANN001
    return [_side_action(reg, pos, side, choices[side]) for side in (0, 1)]


def _budget_and_branch(name: str, turn: int) -> tuple[Budget, int, int]:
    """The budget, the branch Showdown's policy took, and how many there are."""
    if CASES[name].accuracy == "miss" and turn == 0:
        return BUDGET_ACCURACY, 1, 2
    return BUDGET, 0, 1


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
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
def test_the_port_plays_showdowns_turns(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    """The port plays the same turns and matches Showdown."""
    showdown = _play(oracle, name)
    node = rustnode.node_for(reg)
    assert node is not None
    mine = [_clear_stats(showdown[0])]
    for turn, choices in enumerate(CASES[name].turns):
        budget, branch, count = _budget_and_branch(name, turn)
        there = node.resolve(mine[-1], _actions(reg, mine[-1], choices), budget, select=branch)
        assert there is not None and there.position is not None, (name, turn, "the port refused")
        assert len(there.branches) == count, (name, turn, there.branches)
        mine.append(there.position)
        assert _conditions(mine[-1]) == _conditions(showdown[turn + 1]), (
            name, turn, _conditions(mine[-1]), _conditions(showdown[turn + 1])
        )
        assert _state(mine[-1]) == _state(showdown[turn + 1]), (
            name, turn, _state(mine[-1]), _state(showdown[turn + 1])
        )
