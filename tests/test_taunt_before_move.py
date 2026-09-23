"""Taunt stops a status move chosen the same turn, and lasts a turn longer on a Pokemon
that has already moved, played against Showdown (IKA-188).

`data/moves.ts`, taunt (the champions mod does not override it):

    condition: {
        duration: 3,
        onStart(target) {
            if (target.activeTurns && !this.queue.willMove(target)) {
                this.effectState.duration!++;
            }
            ...
        },
        onDisableMove(pokemon) { ... every Status move but Me First ... },
        onBeforeMovePriority: 5,
        onBeforeMove(attacker, defender, move) {
            if (!(move.isZ && move.isZOrMaxPowered) && move.category === 'Status' && move.id !== 'mefirst') {
                this.add('cant', attacker, 'move: Taunt', move);
                return false;
            }
        },
    },

`onDisableMove` only shapes the next request, so a Pokemon taunted by a faster (or a
Prankster) Taunt still uses the status move it chose -- the resolver and the port let it:
the Tailwind went up, its PP was spent, and a Choice Scarf locked. Showdown stops it in
`BeforeMove`, at priority 5: after sleep and freeze (10) and flinch (8), before confusion
(3) and paralysis (1), so the confusion's try is not spent. No PP, no lock, and
`moveThisTurnResult = false`.

A Taunt on a Pokemon that has already moved this turn is 4 long (the turn's own residual
takes one, three whole turns remain); on one that came in this turn it is 3, although it
has no move queued either -- `activeTurns` is 0. The resolver had 3 for all of them.

Each turn is resolved from Showdown's own position before it and held to Showdown's
position after it, by Python and by the port.
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


def _mon(species: str, ability: str, moves: list[str], spe: int, item: str | None = None) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe},
        item=item,
    )


GRIMM_MOVES = ["taunt", "spiritbreak", "protect", "reflect"]
PRANKSTER = _mon("Grimmsnarl", "Prankster", GRIMM_MOVES, 1)
SLOW_TAUNT = _mon("Grimmsnarl", "Frisk", GRIMM_MOVES, 1)
MILOTIC = _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "confuseray"], 2)
INCINEROAR = _mon("Incineroar", "Blaze", ["fakeout", "flareblitz", "partingshot", "darkestlariat"], 3)
SYLVEON = _mon("Sylveon", "Pixilate", ["hypervoice", "protect", "moonblast", "wish"], 4)
WHIM_MOVES = ["tailwind", "dazzlinggleam", "encore", "protect"]
WHIMSICOTT = _mon("Whimsicott", "Chlorophyll", WHIM_MOVES, 30)
SCARF_WHIMSICOTT = _mon("Whimsicott", "Chlorophyll", WHIM_MOVES, 30, "Choice Scarf")
GARCHOMP = _mon("Garchomp", "Rough Skin", ["dragonclaw", "earthquake", "protect", "swordsdance"], 1)

TEAM_A = [PRANKSTER, MILOTIC, INCINEROAR, SYLVEON]
SLOW_A = [SLOW_TAUNT, MILOTIC, INCINEROAR, SYLVEON]
TEAM_B = [WHIMSICOTT, GARCHOMP, INCINEROAR, SYLVEON]
SCARF_B = [SCARF_WHIMSICOTT, GARCHOMP, INCINEROAR, SYLVEON]

TAUNT = "move 1 1, move 2"
TAILWIND = "move 1, move 3"
GLEAM = "move 2, move 3"

#: name -> (team A, team B, the turns' choices per side).
CASES: dict[str, tuple[list[TeamSet], list[TeamSet], list[tuple[str, str]]]] = {
    # Prankster Taunt, then the Tailwind Whimsicott chose: `cant`, no Tailwind, no PP.
    "status-after-taunt": (TEAM_A, TEAM_B, [(TAUNT, TAILWIND)]),
    # The same under a Choice Scarf: no lock either (`onModifyMove` is never reached).
    "scarf-status-after-taunt": (TEAM_A, SCARF_B, [(TAUNT, TAILWIND)]),
    # Confused on turn 1, taunted on turn 2: Taunt's 5 is above confusion's 3, so the try
    # is not spent -- `time` stays 2.
    "confused-status-after-taunt": (
        TEAM_A, TEAM_B, [("move 3, move 4 1", "move 2, move 3"), (TAUNT, TAILWIND)],
    ),
    # Control: an attack after the Taunt goes through.
    "attack-after-taunt": (TEAM_A, TEAM_B, [(TAUNT, GLEAM)]),
    # A slow Taunt on a Whimsicott that has already moved: 4 long, so 3 after the turn.
    "taunt-after-moved": (SLOW_A, TEAM_B, [(TAUNT, TAILWIND), ("move 3, move 2", GLEAM)]),
    # Control: the one that came in this turn has no move queued but `activeTurns` 0: 3.
    "taunt-on-switched-in": (SLOW_A, TEAM_B, [(TAUNT, "switch 3, move 3")]),
}

EXPECTED_TAUNT = {
    # The taunt's duration on side 1's slot 0 after the last turn, from Showdown's log.
    "status-after-taunt": 2,
    "scarf-status-after-taunt": 2,
    "confused-status-after-taunt": 2,
    "attack-after-taunt": 2,
    "taunt-after-moved": 2,  # 4, then two residuals
    "taunt-on-switched-in": 2,
}

#: Turns that only reach the position the case is about. Confuse Ray's turn is one: the
#: resolver leaves the length unrolled (IKA-177), where Showdown's position has `time`.
SETUP_TURNS = {"confused-status-after-taunt": 1}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_status_checks=False
).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    """Per Pokemon: HP, fainted, failed flag, PP, the taunt's length, the Choice lock and
    confusion's `time`; per side, its conditions."""
    out: dict[str, tuple] = {}
    for index, side in enumerate(pos.sides):
        out[f"p{index + 1} side"] = tuple(sorted((c.id, c.duration) for c in side.side_conditions))
        for mon in side.pokemon:
            taunt = mon.volatile("taunt")
            lock = mon.volatile("choicelock")
            confusion = mon.volatile("confusion")
            out[f"p{index + 1} {mon.species}"] = (
                mon.hp,
                mon.fainted,
                None if mon.fainted else mon.move_last_turn_failed,
                tuple(m.pp for m in mon.moves),
                taunt.duration if taunt is not None else None,
                lock.move if lock is not None else None,
                confusion.extra.get("time") if confusion is not None else None,
            )
    return out


def _play(oracle: Oracle, name: str) -> tuple[list[Position], list[str]]:
    team_a, team_b, turns = CASES[name]
    handle = oracle.create(FORMAT_ID, team_a, team_b, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    positions = [Position.from_json(handle.position)]
    for choice_a, choice_b in turns:
        handle.step([choice_a, choice_b])
        assert handle.choice_errors == [], (name, handle.choice_errors)
        positions.append(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    return positions, log


def _check_showdown(name: str, positions: list[Position], log: list[str]) -> None:
    """The positive control: each case is the one its name says, in Showdown's own terms."""
    blocked = any(line.startswith("|cant|p2a: Whimsicott|move: Taunt|Tailwind") for line in log)
    assert blocked == (name in {"status-after-taunt", "scarf-status-after-taunt",
                                "confused-status-after-taunt"}), (name, log)
    target = positions[-1].sides[1].active_pokemon()[0]
    assert target is not None
    held = target.volatile("taunt")
    assert held is not None and held.duration == EXPECTED_TAUNT[name], (name, held)
    if name == "taunt-after-moved":
        first = positions[1].sides[1].active_pokemon()[0]
        after_first = first.volatile("taunt") if first is not None else None
        assert after_first is not None and after_first.duration == 3
    if name == "confused-status-after-taunt":
        held = target.volatile("confusion")
        assert held is not None and held.extra.get("time") == 2


def _actions(reg, pos: Position, choices: tuple[str, str]) -> list:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


def _only(result) -> Position:  # noqa: ANN001
    assert len(result.branches) == 1, [b.events for b in result.branches]
    return result.branches[0].position


@pytest.mark.parametrize("name", sorted(CASES))
def test_taunt_against_showdown(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    positions, log = _play(oracle, name)
    _check_showdown(name, positions, log)
    for turn, choices in enumerate(CASES[name][2], start=1):
        if turn <= SETUP_TURNS.get(name, 0):
            continue
        start = positions[turn - 1]
        mine = _only(resolve_turn(reg, start, _actions(reg, start, choices), budget=BUDGET))
        assert _state(mine) == _state(positions[turn]), (
            name, turn, _state(mine), _state(positions[turn]),
        )


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
def test_the_port_stops_it_too(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    """The port plays Showdown's turns, and matches Python's position turn by turn."""
    positions, _log = _play(oracle, name)
    for turn, choices in enumerate(CASES[name][2], start=1):
        if turn <= SETUP_TURNS.get(name, 0):
            continue
        start = _clear_stats(positions[turn - 1])
        node = rustnode.node_for(reg)
        assert node is not None
        actions = _actions(reg, start, choices)
        there = node.resolve(start, actions, BUDGET, select=0)
        assert there is not None and there.position is not None, f"the port refused turn {turn}"

        os.environ[rustnode.ENV_ENABLE] = "0"
        rustnode.reset()
        here = _only(resolve_turn(reg, start, actions, budget=BUDGET))
        os.environ[rustnode.ENV_ENABLE] = "1"
        rustnode.reset()

        assert _state(there.position) == _state(positions[turn]), (
            name, turn, _state(there.position), _state(positions[turn]),
        )
        assert here.to_json() == there.position.to_json(), (name, turn)
