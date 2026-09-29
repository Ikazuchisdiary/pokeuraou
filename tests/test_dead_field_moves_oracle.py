"""`narrow.dead_field_move` against Showdown (IKA-395).

A field move whose field is already up fails in Showdown (`-fail`); Trick Room, Magic Room,
Wonder Room (a recast ends them), Spikes and Toxic Spikes (they stack) do not. But a failed
field move is not dominated by the other moves the way a dead Fake Out is, so the function
grades the failure (records/IKA-395.md), and each grade is held to what Showdown does:

* Sucker Punch fails against a foe's status move even when that move fails;
* Stomping Tantrum doubles after a failed move;
* Brick Break takes a screen down before a slower recast, which then works;
* two allies casting the same field: the second fails, unless the first is flinched.

The positive control is the module before this change (no `dead_field_move`): every test
here fails with an AttributeError on it. The controls that the comparison can fail are the
rooms and the hazards, and the turn before the field is up.
"""

from __future__ import annotations

import pytest

from pokeuraou import narrow
from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

FILL = ["protect", "helpinghand", "substitute", "irondefense"]
QUIET = "move 4, move 4"


def _mon(species: str, ability: str, moves: list[str], spe: int, item: str | None = None) -> TeamSet:
    spread = {"hp": 32, "atk": 20, "def": 20, "spa": 20, "spd": 20, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


def _foes(second: list[str] | None = None, spe: int = 0) -> list[TeamSet]:
    return [
        _mon("Garchomp", "Rough Skin", FILL, 2),
        _mon("Incineroar", "Blaze", second or FILL, spe),
    ]


class _Game:
    def __init__(self, oracle: Oracle, mine: list[TeamSet], theirs: list[TeamSet]) -> None:
        self.handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
        self.handle.step(["team 12", "team 12"])

    def position(self) -> Position:
        return Position.from_json(self.handle.position)

    def turn(self, mine: str, theirs: str = QUIET) -> list[str]:
        self.handle.step([mine, theirs])
        assert self.handle.choice_errors == [], self.handle.choice_errors
        return list(self.handle.log)

    def close(self) -> None:
        self.handle.close()


def _verdict(reg, pos: Position, choice: str, side: int = 0):  # noqa: ANN001, ANN202
    action = next(a for a in side_actions(reg, pos, side) if a.to_choice() == choice)
    return narrow.dead_field_move(reg, pos, side, action), narrow.dead_field_moves(reg, pos, side, action)


def _failed(log: list[str], who: str) -> bool:
    return any(line.startswith(f"|-fail|{who}") for line in log)


FIELD_MOVES = [
    "tailwind", "reflect", "lightscreen", "safeguard", "sunnyday", "raindance", "sandstorm",
    "snowscape", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain", "gravity",
    "fairylock",
]
#: Recasting these works in Showdown (a room ends, hazards stack).
NOT_DEAD = ["trickroom", "magicroom", "wonderroom", "spikes", "toxicspikes"]


def _caster(move: str, ability: str = "Inner Focus", snow: bool = False) -> list[TeamSet]:
    # No other weather or terrain move on the field unless asked: it would make the field
    # changeable (that grade is tested on its own below).
    others = ["snowscape", "protect", "irondefense"] if snow else ["irondefense", "protect", "helpinghand"]
    return [
        _mon("Alakazam", ability, [move, *others], 20),
        _mon("Kangaskhan", "Inner Focus", FILL, 10),
    ]


@pytest.mark.parametrize("move", FIELD_MOVES)
def test_a_recast_fails_in_showdown_and_is_reported_dead(reg, oracle: Oracle, move: str) -> None:  # noqa: ANN001
    game = _Game(oracle, _caster(move), _foes())
    try:
        first = game.turn("move 1, move 4")
        assert not _failed(first, "p1a"), f"{move} should work the first time: {first}"
        pos = game.position()
        verdict, per_slot = _verdict(reg, pos, "move 1, move 4")
        second = game.turn("move 1, move 4")
        assert _failed(second, "p1a"), f"showdown did not fail the recast of {move}: {second}"
        assert verdict == narrow.DEAD, (move, verdict, per_slot)
    finally:
        game.close()


@pytest.mark.parametrize("move", FIELD_MOVES)
def test_control_the_first_cast_is_not_reported(reg, oracle: Oracle, move: str) -> None:  # noqa: ANN001
    """The comparison can fail: before the field is up nothing is dead."""
    game = _Game(oracle, _caster(move), _foes())
    try:
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        log = game.turn("move 1, move 4")
        assert not _failed(log, "p1a"), log
        assert verdict is None, (move, verdict)
    finally:
        game.close()


def test_aurora_veil_fails_without_snow_and_after_it_is_up(reg, oracle: Oracle) -> None:  # noqa: ANN001
    game = _Game(oracle, _caster("auroraveil", snow=True), _foes())
    try:
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        assert _failed(game.turn("move 1, move 4"), "p1a"), "Aurora Veil without snow must fail"
        assert verdict == narrow.DEAD
        game.turn("move 2, move 4")  # Snowscape
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        assert not _failed(game.turn("move 1, move 4"), "p1a"), "with snow it works"
        assert verdict is None
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        assert _failed(game.turn("move 1, move 4"), "p1a")
        assert verdict == narrow.DEAD
    finally:
        game.close()


@pytest.mark.parametrize("move", NOT_DEAD)
def test_control_a_recast_that_works_is_never_reported(reg, oracle: Oracle, move: str) -> None:  # noqa: ANN001
    game = _Game(oracle, _caster(move), _foes())
    try:
        game.turn("move 1, move 4")
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        second = game.turn("move 1, move 4")
        assert not _failed(second, "p1a"), f"showdown failed the recast of {move}: {second}"
        assert verdict is None, (move, verdict)
    finally:
        game.close()


def test_sucker_punch_is_dodged_by_the_failed_cast(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """The failed Tailwind is not "nothing": a foe's Sucker Punch fails against it, and hits
    a Shadow Ball. The verdict says so instead of `dead`."""
    foes = _foes(["suckerpunch", "protect", "helpinghand", "irondefense"])
    mine = [
        _mon("Alakazam", "Inner Focus", ["tailwind", "shadowball", "protect", "irondefense"], 20),
        _mon("Kangaskhan", "Inner Focus", FILL, 10),
    ]
    game = _Game(oracle, mine, foes)
    try:
        game.turn("move 1, move 4")
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        log = game.turn("move 1, move 4", "move 4, move 1 1")
        assert _failed(log, "p2b"), f"Sucker Punch should fail against a failed Tailwind: {log}"
        assert verdict == narrow.DEAD_BUT_DODGES_SUCKER_PUNCH
    finally:
        game.close()
    control = _Game(oracle, mine, foes)
    try:
        control.turn("move 1, move 4")
        log = control.turn("move 2 1, move 4", "move 4, move 1 1")
        hit = any(line.startswith("|move|p2b: Incineroar|Sucker Punch|p1a") for line in log)
        assert hit and not _failed(log, "p2b"), log
    finally:
        control.close()


def _tantrum_damage(oracle: Oracle, second: str) -> tuple[int, Position]:
    mine = [
        _mon("Kangaskhan", "Inner Focus", ["tailwind", "stompingtantrum", "protect", "irondefense"], 20),
        _mon("Alakazam", "Inner Focus", FILL, 10),
    ]
    game = _Game(oracle, mine, _foes())
    try:
        game.turn("move 1, move 4")
        game.turn(second)
        pos = game.position()
        log = game.turn("move 2 1, move 4")
        hp = [
            line for line in log if line.startswith("|-damage|p2a: Garchomp") and "/" in line.split("|")[3]
        ]
        after = int(hp[0].split("|")[3].split("/")[0])
        return after, pos
    finally:
        game.close()


def test_stomping_tantrum_doubles_after_the_failed_cast(reg, oracle: Oracle) -> None:  # noqa: ANN001
    after_failed, _ = _tantrum_damage(oracle, "move 1, move 4")
    after_worked, _ = _tantrum_damage(oracle, "move 4, move 4")
    lost_failed, lost_worked = 215 - after_failed, 215 - after_worked
    assert lost_failed > 1.6 * lost_worked, (lost_failed, lost_worked)
    mine = [
        _mon("Kangaskhan", "Inner Focus", ["tailwind", "stompingtantrum", "protect", "irondefense"], 20),
        _mon("Alakazam", "Inner Focus", FILL, 10),
    ]
    game = _Game(oracle, mine, _foes())
    try:
        game.turn("move 1, move 4")
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        assert verdict == narrow.DEAD_BUT_FEEDS_STOMPING_TANTRUM
    finally:
        game.close()


@pytest.mark.parametrize("breaker", ["brickbreak", "psychicfangs"])
def test_a_faster_foe_can_take_the_screen_down_first(reg, oracle: Oracle, breaker: str) -> None:  # noqa: ANN001
    """The recast of Light Screen works after the foe's screen breaker: `dead-but-changeable`."""
    mine = [
        _mon("Snorlax", "Thick Fat", ["lightscreen", "shadowball", "protect", "irondefense"], 0),
        _mon("Kangaskhan", "Inner Focus", FILL, 1),
    ]
    game = _Game(oracle, mine, _foes([breaker, "protect", "helpinghand", "irondefense"], spe=32))
    try:
        game.turn("move 1, move 4")
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        log = game.turn("move 1, move 4", "move 4, move 1 1")
        assert not _failed(log, "p1a"), f"the recast should work after {breaker}: {log}"
        assert verdict == narrow.DEAD_BUT_CHANGEABLE
    finally:
        game.close()


def test_the_weather_flips_and_the_recast_works(reg, oracle: Oracle) -> None:  # noqa: ANN001
    mine = [
        _mon("Snorlax", "Thick Fat", ["sunnyday", "shadowball", "protect", "irondefense"], 0),
        _mon("Kangaskhan", "Inner Focus", FILL, 1),
    ]
    game = _Game(oracle, mine, _foes(["raindance", "protect", "helpinghand", "irondefense"], spe=32))
    try:
        game.turn("move 1, move 4")
        pos = game.position()
        verdict, _ = _verdict(reg, pos, "move 1, move 4")
        log = game.turn("move 1, move 4", "move 4, move 1")
        assert not _failed(log, "p1a"), log
        assert verdict == narrow.DEAD_BUT_CHANGEABLE
    finally:
        game.close()


def test_two_allies_casting_the_same_field_is_a_hedge(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """The second Tailwind fails -- unless the first ally is flinched by a foe's Fake Out."""
    mine = [
        _mon("Alakazam", "Magic Guard", ["tailwind", "shadowball", "protect", "irondefense"], 20),
        _mon("Kangaskhan", "Early Bird", ["tailwind", "protect", "helpinghand", "irondefense"], 10),
    ]
    plain = _Game(oracle, mine, _foes())
    try:
        pos = plain.position()
        verdict, per_slot = _verdict(reg, pos, "move 1, move 1")
        log = plain.turn("move 1, move 1")
        assert _failed(log, "p1b") and not _failed(log, "p1a"), log
        assert per_slot == [None, narrow.DEAD_IF_FIRST_ACTS] and verdict == narrow.DEAD_IF_FIRST_ACTS
    finally:
        plain.close()
    flinched = _Game(oracle, mine, _foes(["fakeout", "protect", "helpinghand", "irondefense"], spe=32))
    try:
        log = flinched.turn("move 1, move 1", "move 4, move 1 1")
        assert any(line.startswith("|cant|p1a: Alakazam|flinch") for line in log), log
        assert not _failed(log, "p1b"), f"the second Tailwind works when the first is flinched: {log}"
    finally:
        flinched.close()
