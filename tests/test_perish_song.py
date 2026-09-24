"""Perish Song, played against Showdown for four turns (IKA-172).

`data/moves.ts`, perishsong:

    onHitField(target, source, move) {
        let result = false;
        let message = false;
        for (const pokemon of this.getAllActive()) {
            if (this.runEvent('Invulnerability', pokemon, source, move) === false) {
                ...
            } else if (this.runEvent('TryHit', pokemon, source, move) === null) {
                result = true;
            } else if (!pokemon.volatiles['perishsong']) {
                pokemon.addVolatile('perishsong');
                ...
                result = true;
        ...
        if (!result) return false;
    },
    condition: { duration: 4, onEnd(target) { ... target.faint(); }, onResidualOrder: 24, ... }

Soundproof's `onTryHit` is the `null` (`target !== source && move.flags['sound']`), and it
is `breakable`, so a Mold Breaker singer reaches it. The volatile is cleared by a switch.

The resolver has this; the port did not: its `apply_status_move` had no Perish Song, while
`modelled.rs` listed the move as fully modelled, so the port's turn left nobody counting
down and nothing was reported. Generation advances the game through the port, and the
recorded games show it: value-gen11L's pool has Perish Song used and no volatile after it.

Each case plays four turns in Showdown. Turn 1: side 0's slot a sings while its partner
and side 1 Protect -- Perish Song has no `protect` flag, so it goes through. Turns 2 to 4:
everyone uses Helping Hand, which moves nothing, but for the switch and repeat cases. The
Python turn and the port's are each played on from their own previous turn and held to
Showdown's HP, faint, Perish counter and failed-move flag after every turn.
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

#: Every set carries the same four moves, so one choice string means the same thing on
#: every Pokemon. Not legal sets: the bridge does not check learnsets, and this is the way
#: to put Perish Song on a Soundproof or Mold Breaker user.
MOVES = ["helpinghand", "protect", "perishsong", "splash"]


def _mon(species: str, ability: str, spe: int) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=list(MOVES),
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
    )


POLITOED = _mon("Politoed", "Damp", 10)
WHIMSICOTT = _mon("Whimsicott", "Chlorophyll", 21)
KOMMOO = _mon("Kommo-o", "Soundproof", 12)
PANGORO = _mon("Pangoro", "Mold Breaker", 6)
INCINEROAR = _mon("Incineroar", "Blaze", 2)
GARCHOMP = _mon("Garchomp", "Rough Skin", 5)
HIPPOWDON = _mon("Hippowdon", "Sand Force", 3)
ROTOM = _mon("Rotom-Wash", "Levitate", 4)

TEAM_B = [INCINEROAR, GARCHOMP, HIPPOWDON, ROTOM]
TEAM_B_SOUNDPROOF = [INCINEROAR, KOMMOO, HIPPOWDON, ROTOM]

SING = "move 3, move 2"
PROTECT_BOTH = "move 2, move 2"
HELP = "move 1 -2, move 1 -1"

#: name -> (side 0's team, side 1's team, the four turns' choices per side).
CASES: dict[str, tuple[list[TeamSet], list[TeamSet], list[tuple[str, str]]]] = {
    # All four count down and all four faint at the end of turn 4.
    "all-four": ([POLITOED, WHIMSICOTT], TEAM_B, [(SING, PROTECT_BOTH)] + [(HELP, HELP)] * 3),
    # A Soundproof foe is not given it and is still standing.
    "soundproof-foe": (
        [POLITOED, WHIMSICOTT], TEAM_B_SOUNDPROOF, [(SING, PROTECT_BOTH)] + [(HELP, HELP)] * 3,
    ),
    # Soundproof guards the singer's partner as well: `target !== source`.
    "soundproof-ally": ([POLITOED, KOMMOO], TEAM_B, [(SING, PROTECT_BOTH)] + [(HELP, HELP)] * 3),
    # ...but not the singer itself.
    "soundproof-singer": (
        [KOMMOO, WHIMSICOTT], TEAM_B, [(SING, PROTECT_BOTH)] + [(HELP, HELP)] * 3,
    ),
    # Soundproof is `breakable`: a Mold Breaker singer reaches it.
    "mold-breaker": (
        [PANGORO, WHIMSICOTT], TEAM_B_SOUNDPROOF, [(SING, PROTECT_BOTH)] + [(HELP, HELP)] * 3,
    ),
    # Incineroar switches out on turn 2: the volatile goes with the switch and Hippowdon,
    # who comes in, never has one.
    "switch-out": (
        [POLITOED, WHIMSICOTT],
        TEAM_B,
        [(SING, PROTECT_BOTH), (HELP, "switch 3, move 1 -1"), (HELP, HELP), (HELP, HELP)],
    ),
    # Sung again on turns 2 and 4 with everyone already counting: nobody is given a new
    # counter, and the move fails (`if (!result) return false`). Turn 3's Helping Hand on a
    # partner who has not moved yet succeeds in between, so the flag has to go down again.
    "sung-twice": (
        [POLITOED, WHIMSICOTT],
        TEAM_B,
        [
            (SING, PROTECT_BOTH),
            ("move 3, move 1 -1", HELP),
            ("move 1 -2, move 2", HELP),
            ("move 3, move 1 -1", HELP),
        ],
    ),
}

BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


#: Whose failed-move flag is compared. Only the singer's, in the one case about it: the
#: Helping Hand filler fails in Showdown when its partner has already moved
#: (`if (!target.newlySwitched && !this.queue.willMove(target)) return false`), which
#: tests/test_helping_hand_fails.py holds to Showdown (IKA-184); it is not this test's question.
FLAG_OF = {"sung-twice": "p1 politoed"}


def _state(pos: Position, name: str = "") -> dict[str, tuple]:
    """Per Pokemon, on the field or not: HP, fainted, Perish counter (and one failed flag)."""
    out = {}
    for index, side in enumerate(pos.sides):
        for mon in side.pokemon:
            perish = mon.volatile("perishsong")
            key = f"p{index + 1} {mon.species}"
            out[key] = (mon.hp, mon.fainted, None if perish is None else perish.duration)
            if FLAG_OF.get(name) == key and not mon.fainted:
                out[key] += (mon.move_last_turn_failed,)
    return out


def _play(oracle: Oracle, name: str) -> list[Position]:
    team_a, team_b, turns = CASES[name]
    handle = oracle.create(FORMAT_ID, team_a, team_b, policy=RandomnessPolicy())
    handle.step(["team 12", "team 1234"])
    positions = [Position.from_json(handle.position)]
    for choice_a, choice_b in turns:
        handle.step([choice_a, choice_b])
        assert handle.choice_errors == [], (name, handle.choice_errors)
        positions.append(Position.from_json(handle.position))
    handle.close()
    return positions


def _check_showdown(name: str, positions: list[Position]) -> None:
    """What Showdown itself did -- the positive control that the case is the one named."""
    after_1 = _state(positions[1])
    counting = sorted(k for k, v in after_1.items() if v[2] is not None)
    end = _state(positions[-1])
    fainted = sorted(k for k, v in end.items() if v[1])
    expected_counting = {
        "all-four": ["p1 politoed", "p1 whimsicott", "p2 garchomp", "p2 incineroar"],
        "soundproof-foe": ["p1 politoed", "p1 whimsicott", "p2 incineroar"],
        "soundproof-ally": ["p1 politoed", "p2 garchomp", "p2 incineroar"],
        "soundproof-singer": ["p1 kommoo", "p1 whimsicott", "p2 garchomp", "p2 incineroar"],
        "mold-breaker": ["p1 pangoro", "p1 whimsicott", "p2 incineroar", "p2 kommoo"],
        "switch-out": ["p1 politoed", "p1 whimsicott", "p2 garchomp", "p2 incineroar"],
        "sung-twice": ["p1 politoed", "p1 whimsicott", "p2 garchomp", "p2 incineroar"],
    }[name]
    assert counting == expected_counting, (name, after_1)
    assert all(after_1[k][2] == 3 for k in counting), after_1
    expected_fainted = {
        "switch-out": ["p1 politoed", "p1 whimsicott", "p2 garchomp"],
    }.get(name, expected_counting)
    assert fainted == expected_fainted, (name, end)
    if name == "switch-out":
        assert end["p2 incineroar"][2] is None and end["p2 hippowdon"][2] is None, end
    if name == "sung-twice":
        assert _state(positions[2], name)["p1 politoed"][3], "Showdown fails the second Perish Song"
        assert not _state(positions[1], name)["p1 politoed"][3], "but not the first"
        # Turn 4's failure is not there to see: the singer faints at the end of it, and
        # Showdown's `clearVolatile` resets the flag of a fainted Pokemon.
        flags = [_state(p, name)["p1 politoed"][3:] for p in positions[1:]]
        assert flags == [(False,), (True,), (False,), ()], flags
        assert all(v[2] == 2 for k, v in _state(positions[2]).items() if k in counting)


def _actions(reg, pos: Position, choices: tuple[str, str]) -> list:  # noqa: ANN001
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


def _clear_stats(pos: Position) -> Position:
    # Showdown's positions carry its final stats, which the port refuses as it would a
    # transformed Pokemon's. Nobody here is transformed, so they follow from the spreads.
    for side in pos.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    return pos


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_sings_it_too(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    """The port plays Showdown's four turns, and matches Showdown's position turn by turn."""
    positions = _play(oracle, name)
    _check_showdown(name, positions)
    node = rustnode.node_for(reg)
    assert node is not None
    mine = _clear_stats(positions[0])
    for turn, choices in enumerate(CASES[name][2], start=1):
        actions = _actions(reg, mine, choices)
        there = node.resolve(mine, actions, BUDGET, select=0)
        assert there is not None and there.position is not None, f"the port refused turn {turn}"
        assert there.unmodelled == [] or all("perish" not in u for u in there.unmodelled), there.unmodelled
        assert len(there.branches) == 1, there.branches
        assert _state(there.position, name) == _state(positions[turn], name), (
            name, turn, _state(there.position, name), _state(positions[turn], name),
        )
        mine = there.position
