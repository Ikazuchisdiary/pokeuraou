"""Moonlight, Synthesis and Morning Sun, played against Showdown (IKA-187).

`data/moves.ts`, moonlight (synthesis and morningsun are the same body):

    onHit(pokemon) {
        let factor = 0.5;
        switch (pokemon.effectiveWeather(undefined, true)) {
        case 'sunnyday': case 'desolateland': factor = 0.667; break;
        case 'raindance': case 'primordialsea': case 'sandstorm': case 'hail': case 'snowscape':
            factor = 0.25; break;
        }
        const success = !!this.heal(this.modify(pokemon.maxhp, factor));

and `sim/battle.ts`:

    modify(value, numerator, denominator = 1) {
        const modifier = tr(numerator * 4096 / denominator);
        return tr((tr(value * modifier) + 2048 - 1) / 4096);
    }

So the amount is `(maxhp * m + 2047) >> 12` with m = 2048, 2732 or 1024: a half is rounded
*down* -- a quarter of 170 heals 42, half of 171 heals 85. Python rounded it half up, as
`_round_fraction` does for Recover's `heal` field (`Math.round(baseMaxhp * heal[0] /
heal[1])` in `battle-actions.ts`, which really is half up), and healed 43 and 86. The port
did not heal at all: `modelled.rs` lists the three as fully modelled, the dump has no
`heal` field for them (the amount is in `onHit`), and `apply_status_move` never named them.
`tools/port_gate_audit.py` let it through because it set every fully-modelled arm aside
without asking whether the dump says the move has custom code.

Turn 1: side 1's Garchomp Iron Heads side 0's Clefable (Magic Guard, so the weather does not
chip it) while its partner, the weather setter, Protects. Turn 2: Clefable uses the move,
everyone else Helping Hands. The Python turn and the port's, both from Showdown's position
after turn 1, are held to Showdown's HP of every Pokemon after turn 2.

The controls: no weather at an even max HP (both roundings agree, so old Python passed and
only the port failed), and Recover at an odd max HP, which has to stay rounded half up.
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

FILL = ["ironhead", "protect", "helpinghand"]

SETTERS = {
    "none": ("Whimsicott", "Prankster"),
    "sun": ("Torkoal", "Drought"),
    "rain": ("Politoed", "Drizzle"),
    "sand": ("Hippowdon", "Sand Stream"),
    "snow": ("Abomasnow", "Snow Warning"),
}


def _mon(species: str, ability: str, moves: list[str], spe: int, atk: int = 20, hp: int = 20) -> TeamSet:
    defence = 0 if species == "Clefable" else 10
    return TeamSet(
        species=species, ability=ability, nature="Serious", moves=moves,
        sp={"hp": hp, "atk": atk, "def": defence, "spa": 20, "spd": 10, "spe": spe},
    )


#: name -> (the move Clefable uses, the weather, Clefable's HP points, the heal Showdown shows).
#: HP points 0 give a max HP of 170, 1 give 171.
CASES: dict[str, tuple[str, str, int, int]] = {}
for _move in ("moonlight", "synthesis", "morningsun"):
    for _weather, _heal in (("none", 85), ("sun", 113), ("rain", 42), ("sand", 42), ("snow", 42)):
        CASES[f"{_move}-{_weather}"] = (_move, _weather, 0, _heal)
#: Half of 171 is 85.5: `modify` gives 85, `Math.round` 86.
CASES["moonlight-none-odd-max-hp"] = ("moonlight", "none", 1, 85)
#: Recover's `heal: [1, 2]` is `Math.round`: 86.
CASES["control-recover-odd-max-hp"] = ("recover", "none", 1, 86)
CASES["control-recover-in-rain"] = ("recover", "rain", 0, 85)

TURN_1 = ["move 3 -2, move 2", "move 1 1, move 2"]
TURN_2 = ["move 4, move 3 -1", "move 3 -2, move 3 -1"]

BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _teams(name: str) -> tuple[list[TeamSet], list[TeamSet]]:
    move, weather, hp, _heal = CASES[name]
    species, ability = SETTERS[weather]
    # The move is in the fourth slot, so `move 4` is it and `move 3` Helping Hand.
    mine = [
        _mon("Clefable", "Magic Guard", [*FILL, move], 5, hp=hp),
        _mon(species, ability, [*FILL, "splash"], 4),
    ]
    theirs = [
        _mon("Garchomp", "Rough Skin", [*FILL, "splash"], 20, atk=32),
        _mon("Incineroar", "Blaze", [*FILL, "splash"], 3),
        _mon("Rotom-Wash", "Levitate", [*FILL, "splash"], 2),
    ]
    return mine, theirs


def _hp(pos: Position) -> dict[str, int]:
    return {
        f"p{index + 1}.{mon.species}": mon.hp for index, side in enumerate(pos.sides) for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, dict[str, int]]:
    mine, theirs = _teams(name)
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    handle.step(TURN_1)
    assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(TURN_2)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    log = list(handle.log)
    handle.close()
    clef = lambda pos: next(m for m in pos.sides[0].pokemon if m.species == "clefable")  # noqa: E731
    healed = clef(after).hp - clef(before).hp
    assert healed == CASES[name][3], f"{name}: Showdown healed {healed}, not {CASES[name][3]}: {log}"
    assert clef(before).hp < clef(before).maxhp - CASES[name][3], "Clefable too healthy to show the heal"
    for side in before.sides:
        for party in side.pokemon:
            # The port declines any position with a stats override (it reads it as a
            # Transform); both engines compute the stats from the set.
            party.stats_override = None
    return before, _hp(after)


def _actions(reg, pos: Position) -> list:  # noqa: ANN001
    return [next(a for a in side_actions(reg, pos, s) if a.to_choice() == TURN_2[s]) for s in (0, 1)]


@pytest.mark.parametrize("name", sorted(CASES))
def test_python_matches_showdown(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    before, theirs = _play(oracle, name)
    result = resolve_turn(reg, before, _actions(reg, before), budget=BUDGET)
    assert len(result.branches) == 1, [b.events for b in result.branches]
    ours = _hp(result.branches[0].position)
    assert ours == theirs, f"{name}: showdown {theirs} != python {ours}"


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    before, theirs = _play(oracle, name)
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, _actions(reg, before), BUDGET, select=0)
    assert there is not None and there.position is not None, "the port refused the turn"
    assert not there.unmodelled, there.unmodelled
    rust_now = _hp(there.position)
    assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
