"""A `randomNormal` move hits a foe drawn at random, every turn (IKA-178).

vendor/pokemon-showdown/sim/battle.ts, `getTarget`::

    if (move.target !== 'randomNormal' && this.validTargetLoc(targetLoc, pokemon, move.target)) {
        ...
    }
    return this.getRandomTarget(pokemon, move);

and `getRandomTarget` ends in `pokemon.side.randomFoe()`, which is `sample(foes())` -- the
foes still standing (sim/side.ts). `runMove` calls `getTarget` every time the move is used,
so a rampage's locked turns draw again: `side.chooseMove` hands the lock
`lastMoveTargetLoc`, and `getTarget` ignores it for this target type. `getMoveTargets`
then runs `RedirectTarget` in doubles, so Follow Me and Rage Powder pull the move in.

Outrage, Petal Dance, Raging Fury, Thrash, Uproar and Struggle are the regulation's
`randomNormal` moves. Both engines hit the first foe standing, every time, and skipped the
redirection. Now each foe standing is a branch of equal weight, under a budget that
enumerates the secondaries and is not the oracle's pinned one; that one, and a budget that
collapses the random ranges, take the first foe, as `multihit_counts` and `_roll_rampage`
do (the second with an `unmodelled` note).

The oracle answers `sample` with its first value, or with its last under
`RandomnessPolicy(sample='last')`, which is how the second foe is reached.
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

SP = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": 20}


def _mon(species: str, ability: str, moves: list[str]) -> TeamSet:
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, sp=dict(SP))


US = [
    _mon("Garchomp", "Rough Skin", ["outrage", "protect", "earthquake", "swordsdance"]),
    _mon("Sylveon", "Pixilate", ["calmmind", "protect", "moonblast", "wish"]),
    _mon("Milotic", "Marvel Scale", ["recover", "protect", "scald", "icebeam"]),
    _mon("Incineroar", "Intimidate", ["fakeout", "flareblitz", "partingshot", "darkestlariat"]),
]
BENCH = [
    _mon("Kingambit", "Defiant", ["kowtowcleave", "protect", "suckerpunch", "ironhead"]),
    _mon("Charizard", "Blaze", ["heatwave", "airslash", "protect", "solarbeam"]),
]
#: Calm Mind all round: nothing touches the HP an Outrage deals, so the turn's only chance is
#: the foe it hits. No sand either (Sand Force), and every Speed differs.
FOES = [
    _mon("Hippowdon", "Sand Force", ["calmmind", "protect", "earthquake", "yawn"]),
    _mon("Milotic", "Marvel Scale", ["calmmind", "protect", "scald", "toxic"]),
    *BENCH,
]
#: Clefable's Follow Me draws the Outrage whichever foe was drawn.
POWDER = [
    _mon("Hippowdon", "Sand Force", ["calmmind", "protect", "earthquake", "yawn"]),
    _mon("Clefable", "Magic Guard", ["followme", "protect", "moonblast", "calmmind"]),
    *BENCH,
]
STEP = ["move 1, move 1", "move 1, move 1"]
#: Every source of chance enumerated but the crit, at the oracle's damage roll. The
#: rampage's length is on Showdown's position, so it does not branch.
BUDGET = replace(Budget.exact(), enumerate_crit=False).with_fixed_roll(0)
PINNED = Budget.deterministic()


def _play(oracle: Oracle, foes: list[TeamSet], sample: str, turns: int) -> tuple[list[Position], list]:
    """Showdown's positions before the first Outrage and after each, and each turn's rolls.
    The rampage lasts three turns (`multihit='max'`), so turn 2 is locked and unconfused."""
    handle = oracle.create(
        FORMAT_ID, US, foes, policy=RandomnessPolicy(multihit="max", sample=sample)
    )
    handle.step(["team 1234", "team 1234"])
    positions = [Position.from_json(handle.position)]
    rolls = []
    for _ in range(turns):
        handle.step(STEP)
        assert handle.choice_errors == [], handle.choice_errors
        positions.append(Position.from_json(handle.position))
        rolls.append(list(handle.rolls))
    handle.close()
    return positions, rolls


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{slot}.{mon.species}": (mon.hp, mon.status, mon.has_volatile("lockedmove"))
        for index, side in enumerate(pos.sides)
        for slot, mon in enumerate(side.pokemon)
    }


def _key(state: dict[str, tuple]) -> tuple:
    return tuple(sorted(state.items()))


def _actions(reg, pos: Position) -> list:  # noqa: ANN001
    """`STEP`, or on a side with one Pokemon standing, its first slot's part of it."""
    chosen = []
    for side in (0, 1):
        menu = side_actions(reg, pos, side)
        exact = [a for a in menu if a.to_choice() == STEP[side]]
        first = STEP[side].split(",")[0]
        chosen.append(exact[0] if exact else next(a for a in menu if a.to_choice().split(",")[0] == first))
    return chosen


def _python(reg, start: Position, budget: Budget) -> dict[tuple, float]:  # noqa: ANN001
    result = resolve_turn(reg, start, _actions(reg, start), budget=budget)
    assert not result.suspended
    out: dict[tuple, float] = {}
    for branch in result.branches:
        key = _key(_state(branch.position))
        out[key] = out.get(key, 0.0) + branch.probability
    return out


def _port(reg, start: Position, budget: Budget) -> dict[tuple, float]:  # noqa: ANN001
    start = Position.from_json(start.to_json())
    # Showdown's stats ride along as an override, which the port reads as a Transform.
    for side in start.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    actions = _actions(reg, start)
    node = rustnode.node_for(reg)
    assert node is not None
    weights = node.resolve(start, actions, budget).branches
    out: dict[tuple, float] = {}
    for index, weight in enumerate(weights):
        picked = node.resolve(start, actions, budget, select=index)
        assert picked is not None and picked.position is not None, "the port refused the turn"
        key = _key(_state(picked.position))
        out[key] = out.get(key, 0.0) + weight
    return out


def _mixed(hit_first: Position, hit_last: Position, species: str) -> dict[str, tuple]:
    """`hit_first` with one foe's HP taken from `hit_last`: the position where that foe,
    not the other, was hit this turn."""
    state = _state(hit_first)
    for key, value in _state(hit_last).items():
        if key.startswith("p2.") and key.endswith(species):
            state[key] = value
    return state


def _cases(oracle: Oracle) -> dict[str, tuple[Position, dict[tuple, float]]]:
    """name -> (Showdown's position before the turn, Showdown's outcomes and weights)."""
    first, first_rolls = _play(oracle, FOES, "first", 2)
    last, _ = _play(oracle, FOES, "last", 2)
    powder_first, _ = _play(oracle, POWDER, "first", 1)
    powder_last, _ = _play(oracle, POWDER, "last", 1)
    assert first[0].to_json() == last[0].to_json()
    # The locked turn draws again: its rolls include a `sample` of both foes.
    assert any(r["kind"] == "sample" and len(r["values"]) == 2 for r in first_rolls[1])
    # The first turn: each foe is a half.
    turn1 = {_key(_state(first[1])): 0.5, _key(_state(last[1])): 0.5}
    # The locked turn, from the position where Hippowdon took the first: Hippowdon again
    # (Showdown's own turn 2), or Milotic, which takes what it took on the other game's turn 1.
    turn2 = {
        _key(_state(first[2])): 0.5,
        _key(_mixed(first[2], last[1], "milotic") | _only_hippowdon(first[1])): 0.5,
    }
    powder = {_key(_state(powder_first[1])): 1.0}
    assert _state(powder_last[1]) == _state(powder_first[1]), "Follow Me draws either draw"
    return {
        "turn 1": (first[0], turn1),
        "turn 2 (locked)": (first[1], turn2),
        "follow me": (powder_first[0], powder),
    }


def _only_hippowdon(pos: Position) -> dict[str, tuple]:
    return {k: v for k, v in _state(pos).items() if k.startswith("p2.") and k.endswith("hippowdon")}


NAMES = ["turn 1", "turn 2 (locked)", "follow me"]


@pytest.fixture(scope="module")
def cases(oracle: Oracle):  # noqa: ANN201
    return _cases(oracle)


@pytest.mark.oracle
def test_showdown_hits_each_foe(cases) -> None:  # noqa: ANN001
    """The facts: the two draws hit different foes, on the first turn and the locked one."""
    for name in ("turn 1", "turn 2 (locked)"):
        _start, want = cases[name]
        assert len(want) == 2, name


@pytest.mark.oracle
@pytest.mark.parametrize("name", NAMES)
def test_python_branches_the_foe(reg, cases, name: str) -> None:  # noqa: ANN001
    start, want = cases[name]
    ours = _python(reg, start, BUDGET)
    assert ours.keys() == want.keys(), (name, ours, want)
    for key, weight in want.items():
        assert ours[key] == pytest.approx(weight), name


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.oracle
@pytest.mark.parametrize("name", NAMES)
def test_the_port_branches_the_foe(reg, cases, bridged: None, name: str) -> None:  # noqa: ANN001
    start, want = cases[name]
    ours = _port(reg, start, BUDGET)
    assert ours.keys() == want.keys(), (name, ours, want)
    for key, weight in want.items():
        assert ours[key] == pytest.approx(weight), name


# ---------------------------------------------------------------------------
# Controls: where there is nothing to draw, or the budget does not draw.


def _pinned_want(oracle: Oracle, name: str) -> tuple[Position, dict[tuple, float]]:
    foes = POWDER if name == "follow me" else FOES
    turns = 2 if name == "turn 2 (locked)" else 1
    first, _ = _play(oracle, foes, "first", turns)
    return first[-2], {_key(_state(first[-1])): 1.0}


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["turn 1", "turn 2 (locked)"])
def test_the_pinned_budget_takes_the_first_foe(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """The oracle's `sample` answers the first foe standing, and so does the pinned budget."""
    start, want = _pinned_want(oracle, name)
    assert _python(reg, start, PINNED) == pytest.approx(want)


@pytest.mark.oracle
@pytest.mark.parametrize("name", ["turn 1", "turn 2 (locked)"])
def test_the_port_pinned_takes_the_first_foe(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    start, want = _pinned_want(oracle, name)
    assert _port(reg, start, PINNED) == pytest.approx(want)


def _one_foe(oracle: Oracle) -> Position:
    """Showdown's first position with Milotic knocked out beforehand: one foe standing."""
    first, _ = _play(oracle, FOES, "first", 0)
    pos = first[0]
    milotic = pos.sides[1].pokemon[pos.sides[1].active[1]]
    milotic.hp, milotic.fainted, milotic.status = 0, True, "fnt"
    return pos


@pytest.mark.oracle
def test_one_foe_standing_is_not_a_draw(reg, oracle: Oracle, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ANN001
    """Python and the port: one outcome, no note, and Hippowdon takes it."""
    pos = _one_foe(oracle)
    result = resolve_turn(reg, pos, _actions(reg, pos), budget=BUDGET)
    assert len(result.branches) == 1
    assert not any("randomNormal" in note for note in result.unmodelled)
    hippowdon = result.branches[0].position.sides[1].pokemon[pos.sides[1].active[0]]
    assert hippowdon.hp < hippowdon.maxhp
    if rustnode.binary_path().exists():
        monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
        rustnode.reset()
        try:
            assert _port(reg, pos, BUDGET) == pytest.approx(_python(reg, pos, BUDGET))
        finally:
            rustnode.reset()


@pytest.mark.oracle
def test_a_collapsed_budget_notes_the_draw(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """`enumerate_secondary` off and not pinned: the first foe, with the note."""
    first, _ = _play(oracle, FOES, "first", 1)
    collapsed = replace(BUDGET, enumerate_secondary=False)
    result = resolve_turn(reg, first[0], _actions(reg, first[0]), budget=collapsed)
    assert {_key(_state(b.position)) for b in result.branches} == {_key(_state(first[1]))}
    assert any("randomNormal" in note for note in result.unmodelled), result.unmodelled
