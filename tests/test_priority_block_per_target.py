"""Armor Tail and Queenly Majesty stop a priority move only when it is aimed at their side,
after the move has started (IKA-158).

Showdown's three abilities are one ``onFoeTryMove`` (vendor/pokemon-showdown/data/abilities.ts,
armortail / dazzling / queenlymajesty):

    const targetAllExceptions = ['perishsong', 'flowershield', 'rototiller'];
    if (move.target === 'foeSide' || (move.target === 'all' && !targetAllExceptions.includes(move.id))) {
        return;
    }
    const holder = this.effectState.target;
    if ((source.isAlly(holder) || move.target === 'all') && move.priority > 0.1) {
        ...; return false;
    }

``TryMove`` runs in ``useMoveInner`` (sim/battle-actions.ts:486) as
``runEvent('TryMove', pokemon, target, move)``, so the handler's ``source`` is the move's
target -- the last of ``getMoveTargets`` -- and it runs after ``runMove`` has spent the PP.
The ability is ``breakable``, so a Mold Breaker move ignores it (``suppressingAbility``).

The resolver and the port stopped the move in ``_can_act`` / ``can_act``, before it started,
whenever a live foe had one of the abilities and the move was not self- or ally-side
targeted. That spent no PP, stopped a priority move aimed at the user's own ally, stopped a
Prankster Spikes (``foeSide``), and ignored Mold Breaker.

Every case is played by Showdown first and the Python turn is held to its HP, boosts, PP,
volatiles, last move, side conditions and weather. It is more than PP: ``runMove`` also
bumps ``activeMoveActions`` before ``TryMove`` (sim/battle-actions.ts:217), so a Fake Out
the old code stopped was not counted as the user's first move and could land the next turn.
The bridge does not export that counter; the two-turn test below reads it from Showdown's
refusal of the second Fake Out instead. The controls, which the old code also passed, are a
Prankster Sunny Day (``all``, never stopped) and a Prankster Tailwind (``allySide``, never
stopped). A Bullet Punch into the holder from a Pokemon without Mold Breaker is stopped in
both -- the positive control for the Mold Breaker case -- and the old code still failed it
on PP, as it did every case it stopped.

The Spikes case compared everything but the side conditions while the resolver and the
port laid every ``sideCondition`` on the user's own side, a separate defect pinned by a
strict xfail at the bottom. IKA-165 lays a foeSide one on the foe's side, so the case is
compared whole and the test below it passes.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from pokeuraou import rustnode
from pokeuraou.actions import MoveAction, SideAction, side_actions
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


TEAM_A = [
    _mon("Incineroar", "Intimidate", ["fakeout", "protect", "flareblitz", "uturn"], 4),
    _mon("Whimsicott", "Prankster", ["cottonspore", "protect", "tailwind", "sunnyday"], 30),
    _mon("Pangoro", "Mold Breaker", ["bulletpunch", "bulkup", "knockoff", "closecombat"], 6),
    _mon("Klefki", "Prankster", ["spikes", "protect", "thunderwave", "playrough"], 8),
    _mon("Lucario", "Inner Focus", ["bulletpunch", "protect", "aurasphere", "closecombat"], 7),
    # Not a legal set: no Prankster Pokemon in the regulation learns Perish Song. It is the
    # only way to reach the `targetAllExceptions` branch.
    _mon("Meowstic", "Prankster", ["perishsong", "protect", "psychic", "reflect"], 9),
]

TEAM_B = [
    _mon("Farigiraf", "Armor Tail", ["calmmind", "protect", "psychic", "hypervoice"], 10),
    _mon("Garchomp", "Rough Skin", ["swordsdance", "protect", "earthquake", "dragonclaw"], 12),
    _mon("Tsareena", "Queenly Majesty", ["protect", "tropkick", "uturn", "highjumpkick"], 14),
    _mon("Corviknight", "Unnerve", ["bulkup", "protect", "bravebird", "roost"], 16),
]

#: name -> (team orders, choices, the start of the Showdown log line that says what happened).
CASES = {
    # Fake Out into Farigiraf's partner: `source.isAlly(holder)`, stopped after the PP.
    "fakeout-into-holders-partner": (
        ["team 1234", "team 1234"],
        ["move 1 2, move 2", "move 1, move 1"],
        "|cant|p2a: Farigiraf|ability: Armor Tail|Fake Out",
    ),
    # Fake Out into the user's own ally: its target is not on the holder's side.
    "fakeout-into-own-ally": (
        ["team 1324", "team 1234"],
        ["move 1 -2, move 2", "move 1, move 1"],
        "|move|p1a: Incineroar|Fake Out|p1b: Pangoro",
    ),
    # Mold Breaker: the ability is `breakable`, so Pangoro's Bullet Punch lands.
    "mold-breaker": (
        ["team 3124", "team 1234"],
        ["move 1 1, move 2", "move 1, move 1"],
        "|move|p1a: Pangoro|Bullet Punch|p2a: Farigiraf",
    ),
    # Positive control for the Mold Breaker case: the same Bullet Punch from Lucario is
    # stopped (the old code agreed on HP and still spent no PP).
    "no-mold-breaker": (
        ["team 5123", "team 1234"],
        ["move 1 1, move 2", "move 1, move 1"],
        "|cant|p2a: Farigiraf|ability: Armor Tail|Bullet Punch",
    ),
    # Prankster Cotton Spore (`allAdjacentFoes`): stopped as a whole, after the PP.
    "prankster-spread": (
        ["team 2134", "team 1234"],
        ["move 1, move 2", "move 1, move 1"],
        "|cant|p2a: Farigiraf|ability: Armor Tail|Cotton Spore",
    ),
    # Prankster Spikes (`foeSide`): the handler returns first, so the Spikes go down.
    "prankster-foeside": (
        ["team 4123", "team 1234"],
        ["move 1, move 2", "move 1, move 1"],
        "|-sidestart|p2: ",
    ),
    # Control: Prankster Sunny Day (`all`, not an exception) is never stopped.
    "control-prankster-all": (
        ["team 2134", "team 1234"],
        ["move 4, move 2", "move 1, move 1"],
        "|-weather|SunnyDay",
    ),
    # Control: Prankster Tailwind (`allySide`) is never stopped.
    "control-prankster-allyside": (
        ["team 2134", "team 1234"],
        ["move 3, move 2", "move 1, move 1"],
        "|-sidestart|p1: ",
    ),
    # Prankster Perish Song is one of the `all` moves the handler does stop.
    "prankster-perish-song": (
        ["team 6123", "team 1234"],
        ["move 1, move 2", "move 1, move 1"],
        "|cant|p2a: Farigiraf|ability: Armor Tail|Perish Song",
    ),
    # Queenly Majesty is the same handler: Fake Out into Tsareena's partner is stopped.
    "queenly-majesty": (
        ["team 1234", "team 3214"],
        ["move 1 2, move 2", "move 1, move 1"],
        "|cant|p2a: Tsareena|ability: Queenly Majesty|Fake Out",
    ),
}

#: Cases whose side conditions are not compared (see the module docstring). None since
#: IKA-165, which put Spikes on the foe's side.
HAZARD_CASES: frozenset[str] = frozenset()

BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


def _state(pos: Position, *, sides: bool = True) -> dict[str, tuple]:
    out: dict[str, tuple] = {
        f"p{index + 1} {mon.species}": (
            mon.hp,
            tuple(sorted((k, v) for k, v in mon.boosts.items() if v)),
            tuple((m.id, m.pp) for m in mon.moves),
            tuple(sorted(v.id for v in mon.volatiles)),
            # `runMove` sets this before `TryMove`, so a stopped move is still the last one.
            mon.last_move,
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }
    for index, side in enumerate(pos.sides):
        if sides:
            out[f"p{index + 1} side"] = tuple(sorted(c.id for c in side.side_conditions))
    out["weather"] = (pos.field.weather,)
    return out


def _play(
    oracle: Oracle, name: str, *, sides: bool | None = None
) -> tuple[Position, list[str], dict[str, tuple], list[str]]:
    orders, choices, _line = CASES[name]
    if sides is None:
        sides = name not in HAZARD_CASES
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(orders)
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    theirs = _state(Position.from_json(handle.position), sides=sides)
    log = list(handle.log)
    handle.close()
    return before, choices, theirs, log


#: Incineroar Fake Outs its own partner Pangoro, which Bulk Ups.
ALLY_FAKE_OUT = SideAction(slots=(MoveAction(0, 1, "fakeout", -2), MoveAction(1, 2, "bulkup", None)))


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        found = [a for a in side_actions(reg, pos, side) if a.to_choice() == choices[side]]
        if not found and choices[side] == ALLY_FAKE_OUT.to_choice():
            # The action generator never aims a `normal` move at an ally, so the search
            # cannot choose this; the resolver still has to play it as Showdown does.
            found = [ALLY_FAKE_OUT]
        assert found, (side, choices[side], [a.to_choice() for a in side_actions(reg, pos, side)])
        out.append(found[0])
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_priority_blocking_ability_stops_moves_at_its_side(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    before, choices, theirs, log = _play(oracle, name)
    assert any(line.startswith(CASES[name][2]) for line in log), (
        f"Showdown did not do what the case says: {log}"
    )

    sides = name not in HAZARD_CASES
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert result.branches
    for branch in result.branches:
        mine = _state(branch.position, sides=sides)
        assert mine == theirs, f"{name}: showdown {theirs} != python {mine}; " + " / ".join(branch.events)


#: Turn 2 after "fakeout-into-holders-partner": Incineroar Fake Outs its own partner
#: Whimsicott, which uses Tailwind.
SECOND_TURN = ["move 1 -2, move 3", "move 1, move 1"]
SECOND_TURN_ACTION = SideAction(slots=(MoveAction(0, 1, "fakeout", -2), MoveAction(1, 3, "tailwind", None)))


def test_a_stopped_fake_out_was_still_the_first_move(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """The stopped Fake Out counted: Showdown will not take a second one.

    The champions mod disables Fake Out once `activeMoveActions` is non-zero
    (data/mods/champions/moves.ts), so the second turn's choice is refused outright. The
    old resolver never reached `_use_move`, left the counter at 0, and let the same Fake
    Out land on turn 2. The new one counts it and fails the second Fake Out.
    """
    orders, choices, _line = CASES["fakeout-into-holders-partner"]
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(orders)
    before = Position.from_json(handle.position)
    handle.step(choices)
    handle.step(SECOND_TURN)
    refused = list(handle.choice_errors)
    handle.close()
    assert any("Fake Out is disabled" in error for error in refused), refused

    first = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert len(first.branches) == 1
    middle = first.branches[0].position
    incineroar = next(mon for mon in middle.sides[0].pokemon if mon.species == "incineroar")
    assert incineroar.active_move_actions == 1
    foe = next(a for a in side_actions(reg, middle, 1) if a.to_choice() == SECOND_TURN[1])
    second = resolve_turn(reg, middle, [SECOND_TURN_ACTION, foe], budget=BUDGET)
    assert second.branches
    whimsicott = next(mon for mon in middle.sides[0].pokemon if mon.species == "whimsicott")
    for branch in second.branches:
        after = next(mon for mon in branch.position.sides[0].pokemon if mon.species == "whimsicott")
        assert after.hp == whimsicott.hp, " / ".join(branch.events)


def test_spikes_land_on_the_foes_side(reg, oracle: Oracle) -> None:  # noqa: ANN001
    before, choices, theirs, _log = _play(oracle, "prankster-foeside", sides=True)
    assert theirs["p2 side"] == ("spikes",)
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    for branch in result.branches:
        assert _state(branch.position) == theirs


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
def test_the_port_stops_moves_at_the_holders_side(
    reg,
    oracle: Oracle,
    bridged: None,
    name: str,  # noqa: ANN001
) -> None:
    """The port plays Showdown's turn too, and matches Python branch for branch."""
    before, choices, theirs, _log = _play(oracle, name)
    # Showdown's positions carry its final stats, which the port refuses as it would a
    # transformed Pokemon's. Nobody here is transformed, so they follow from the spreads.
    for side in before.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    actions = _actions(reg, before, choices)
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, BUDGET)
    assert there is not None, "the port refused the turn"
    for index in range(len(there.branches)):
        chosen = node.resolve(before, actions, BUDGET, select=index)
        assert chosen is not None and chosen.position is not None
        assert _state(chosen.position, sides=name not in HAZARD_CASES) == theirs, (name, index)

    budget = replace(Budget.matrix(), enumerate_accuracy=True)
    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    here = resolve_turn(reg, before, actions, budget=budget)
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    there = node.resolve(before, actions, budget)
    assert there is not None, "the port refused the turn"
    mine = [b.probability for b in here.branches]
    assert len(mine) == len(there.branches)
    assert all(abs(x - y) < 1e-12 for x, y in zip(mine, there.branches, strict=True))
    for index, branch in enumerate(here.branches):
        chosen = node.resolve(before, actions, budget, select=index)
        assert chosen is not None and chosen.position is not None
        assert chosen.position.to_json() == branch.position.to_json(), (name, index)
