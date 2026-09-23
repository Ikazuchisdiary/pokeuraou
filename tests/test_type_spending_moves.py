"""Double Shock and Burn Up spend their user's type (IKA-162).

vendor/pokemon-showdown/data/moves.ts, `doubleshock` (and `burnup` with Fire):

    onTryMove(pokemon, target, move) {
        if (pokemon.hasType('Electric')) return;
        this.add('-fail', pokemon, 'move: Double Shock');
        this.attrLastMove('[still]');
        return null;
    },
    self: {
        onHit(pokemon) {
            pokemon.setType(pokemon.getTypes(true).map(type => type === "Electric" ? "???" : type));
            ...

So a Pawmot that lands one is `???/Fighting` from then on and its next Double Shock fails
-- after the PP is spent, and with `null`, which is not the `false` Stomping Tantrum reads.
`self` runs from `selfDrops`, only for a target the move reached: a Protect or an immunity
leaves the type alone. The type comes back when the Pokemon leaves the field: switching out
and fainting both end in `clearVolatile`, whose `setSpecies(this.baseSpecies)` puts the
species' types back. Neither engine did any of it -- the dump keeps `self` as `{}` and
names only `onTryMove` in `customHooks`, so nothing reported the move as unmodelled either.

Each case is played by Showdown first, and the Python turn and the port's turn are both
held to it. The controls -- a Protect, a Ground-type target, and a Ground move into a
Pawmot that still has its Electric type -- were right before and have to stay right.
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


def _mon(species: str, ability: str, moves: list[str], spe: int, atk: int = 20) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        sp={"hp": 32, "atk": atk, "def": 32, "spa": 20, "spd": 2, "spe": spe},
    )


# Pawmot is the only Double Shock user in M-C; Arcanine is one of the eight Burn Up users.
TEAM_A = [
    _mon("Pawmot", "Iron Fist", ["doubleshock", "protect", "thunderpunch", "closecombat"], 32),
    _mon("Arcanine", "Justified", ["burnup", "protect", "extremespeed", "flareblitz"], 30),
    _mon("Garchomp", "Sand Veil", ["earthquake", "protect", "rockslide", "dragonclaw"], 8),
    _mon("Sylveon", "Pixilate", ["protect", "calmmind", "hypervoice", "wish"], 1),
]
# Toxapex takes the hits; the slower Garchomp answers with a Ground move after them, so the
# same turn reads the type the move just spent. Its Attack is low enough that Pawmot lives.
TEAM_B = [
    _mon("Toxapex", "Regenerator", ["protect", "recover", "scald", "toxic"], 0),
    _mon("Garchomp", "Sand Veil", ["highhorsepower", "protect", "dragonclaw", "rockslide"], 0, atk=0),
    _mon("Milotic", "Marvel Scale", ["scald", "protect", "recover", "icebeam"], 0),
    _mon("Gastrodon", "Storm Drain", ["earthpower", "protect", "recover", "icebeam"], 0),
]

#: Pawmot lands a Double Shock on Toxapex and nobody else does anything to it.
SPEND = ["move 1 1, move 2", "move 2, move 2"]
#: The same with Arcanine's Burn Up.
SPEND_FIRE = ["move 2, move 1 1", "move 2, move 2"]

#: name -> (turns played first, the turn compared, a Showdown log line that shows the case).
CASES = {
    # Double Shock, then High Horsepower into the Pawmot that is now ???/Fighting: neutral.
    "double-shock-spends-electric": (
        [],
        ["move 1 1, move 2", "move 2, move 1 1"],
        "|-start|p1a: Pawmot|typechange|???/Fighting|[from] move: Double Shock",
    ),
    # Burn Up, then Scald into the Arcanine that is now ???: neutral.
    "burn-up-spends-fire": (
        [],
        ["move 2, move 1 1", "move 3 2, move 2"],
        "|-start|p1b: Arcanine|typechange|???|[from] move: Burn Up",
    ),
    # No Electric type, no Double Shock: the PP goes and nothing is hit.
    "a-second-double-shock-fails": (
        [SPEND],
        ["move 1 1, move 3 1", "move 2, move 3 2"],
        "|-fail|p1a: Pawmot|move: Double Shock",
    ),
    "a-second-burn-up-fails": (
        [SPEND_FIRE],
        ["move 3 1, move 1 1", "move 2, move 3 1"],
        "|-fail|p1b: Arcanine|move: Burn Up",
    ),
    # The benched Pawmot is Electric/Fighting again.
    "switching-out-restores-the-type": (
        [SPEND],
        ["switch 3, move 3 1", "move 2, move 3 2"],
        "|switch|p1a: Garchomp|",
    ),
    # A Pawmot that faints as ???/Fighting is Electric/Fighting in the faint.
    "fainting-restores-the-type": (
        [["move 1 1, move 2", "move 2, move 1 1"]],
        ["move 1 1, move 3 1", "move 3 1, move 1 1"],
        "|faint|p1a: Pawmot",
    ),
    # Controls: the move reaches no one, so `self` never runs.
    "control-protected-target": (
        [],
        ["move 1 1, move 2", "move 1, move 3 2"],
        "|-activate|p2a: Toxapex|move: Protect",
    ),
    "control-immune-target": (
        [],
        ["move 1 2, move 2", "move 2, move 3 2"],
        "|-immune|p2b: Garchomp",
    ),
    # Control: a Ground move into a Pawmot that has not spent its type is super effective.
    "control-ground-into-electric": (
        [],
        ["move 3 1, move 2", "move 2, move 1 1"],
        "|-supereffective|p1a: Pawmot",
    ),
}

#: No crit, no chance-based secondary, the maximum roll -- what the oracle's policy pins.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    # Whether last turn's move failed is read on side 0 only, where the two users are: a
    # Double Shock that fails returns `null`, which Stomping Tantrum does not count. Side
    # 1's Recover at full HP is a `false`, recorded since IKA-171 and held to Showdown in
    # tests/test_after_move_oracle.py, not here.
    return {
        f"p{index + 1}.{mon.species}": (
            tuple(mon.types),
            mon.hp,
            tuple(slot.pp for slot in mon.moves),
            mon.move_last_turn_failed if index == 0 else None,
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], dict, list[str]]:
    setup, choices, _line = CASES[name]
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    return before, choices, theirs, log


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    return [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]) for side in (0, 1)
    ]


@pytest.mark.parametrize("name", sorted(CASES))
def test_python_spends_the_type_as_showdown_does(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    before, choices, theirs, log = _play(oracle, name)
    shown = any(line.startswith(CASES[name][2]) for line in log)
    assert shown, f"Showdown did not do what the case says: {log}"

    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    outcomes = [(b.position, b.events) for b in result.branches]
    outcomes += [(s.position, ["(suspended)"]) for s in result.suspended]
    # High Horsepower is 95% accurate. Showdown's policy lets it hit, so the branches where
    # every move hit are the ones compared.
    ours = [(pos, events) for pos, events in outcomes if not any(e.endswith("missed") for e in events)]
    assert ours, "no Python outcome where every move hit"
    for pos, events in ours:
        ours_now = _state(pos)
        assert ours_now == theirs, f"{name}: showdown {theirs} != python {ours_now}; " + " / ".join(events)


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
def test_the_port_spends_the_type_as_showdown_does(
    reg, oracle: Oracle, bridged: None, name: str  # noqa: ANN001
) -> None:
    before, choices, theirs, _log = _play(oracle, name)
    # Showdown's position carries each Pokemon's stats, and the port declines any position
    # with an override (it reads that as a Transform). Both engines compute the same stats
    # from the spreads.
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    actions = _actions(reg, before, choices)
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    # The branches come back in Python's order, so Python's events say which ones are the
    # branch where every move hit. Only the order is taken from Python; the state each of
    # those branches holds is compared with Showdown's.
    here = resolve_turn(reg, before, actions, budget=BUDGET)
    assert [b.probability for b in here.branches] == pytest.approx(there.branches, abs=1e-12)
    hit = [
        index
        for index, branch in enumerate(here.branches)
        if not any(e.endswith("missed") for e in branch.events)
    ]
    assert hit, "no outcome where every move hit"
    for index in hit:
        chosen = node.resolve(before, actions, BUDGET, select=index)
        assert chosen is not None and chosen.position is not None
        rust_now = _state(chosen.position)
        assert rust_now == theirs, f"{name}, branch {index}: showdown {theirs} != rust {rust_now}"
