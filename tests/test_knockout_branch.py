"""The knock-out branch (`Budget.enumerate_knockouts`, IKA-359).

`Budget.matrix()` pins the median roll and never crits, so a hit that knocks out only on a
high roll or a crit reads as never knocking out. With `enumerate_knockouts` a single hit
whose knock-out the 2 x 16 (crit, roll) draws disagree on forks in two: knocked out and
not, each carried by one real draw and weighted by its class's share.

It adds no rule, so it is held to the enumerated game rather than to Showdown directly:
the budget that enumerates every crit and every roll (which the oracle tests hold to
Showdown) must reach each of the two positions, and give the knock-out the same chance.
Both properties are checked on one hand-built turn with a single damaging hit:
Charizard's Air Slash into the other Charizard, everyone else setting up or charging.

Positive control: against the port before this change (`POKEURAOU_RUST_NODE_BIN` at a
build of master 3d9e499) the field is not read, the hit does not fork, and the first test
fails.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import TeamSet
from pokeuraou.position import Position
from pokeuraou.regulation import Regulation

from ._port import Budget, TurnResult, resolve_turn
from .test_actions import _synthetic_position

#: Nothing but the hit's own draws: no accuracy, secondary, status or speed-tie branch.
QUIET = Budget(
    damage_rolls=-1 - 8,
    enumerate_crit=False,
    enumerate_accuracy=False,
    enumerate_status_checks=False,
    enumerate_secondary=False,
    enumerate_speed_ties=False,
    max_branches=64,
)
KNOCKOUT = replace(QUIET, enumerate_knockouts=True)
#: The same turn with every crit and every roll enumerated: the reference.
ENUMERATED = replace(QUIET, damage_rolls=16, enumerate_crit=True, max_branches=10_000)


def _turn(reg: Regulation, team: list[TeamSet]) -> tuple[Position, list]:
    """Air Slash into the foe's Charizard; the other three set up or charge."""
    pos = _synthetic_position(reg, team)
    assert [pos.sides[s].pokemon[0].species for s in (0, 1)] == ["charizard", "charizard"]

    def pick(side: int, choice: str):  # noqa: ANN202
        by = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in by, sorted(by)
        return by[choice]

    return pos, [pick(0, "move 2 1, move 4"), pick(1, "move 4 1, move 4")]


def _foe_hp(result: TurnResult) -> dict[int, float]:
    out: dict[int, float] = {}
    for b in result.branches:
        hp = b.position.sides[1].pokemon[0].hp
        out[hp] = out.get(hp, 0.0) + float(b.probability)
    return out


def _key(pos: Position) -> str:
    return json.dumps(pos.to_json(), sort_keys=True)


@pytest.fixture
def uncertain(reg: Regulation, team_a: list[TeamSet]) -> tuple[Position, list, float]:
    """The turn, with the target's HP set where some draws knock out and some do not;
    and the enumerated chance of the knock-out."""
    pos, actions = _turn(reg, team_a)
    full = pos.sides[1].pokemon[0].hp
    dealt = sorted(full - hp for hp in _foe_hp(resolve_turn(reg, pos, actions, budget=ENUMERATED)))
    assert len(dealt) >= 8, dealt  # rolls and crits give many damage numbers
    # Above the median roll's damage: the pinned roll alone never knocks out.
    pinned = full - next(iter(_foe_hp(resolve_turn(reg, pos, actions, budget=QUIET))))
    target_hp = next(d for d in dealt if d > pinned)
    pos.sides[1].pokemon[0].hp = target_hp
    chance = _foe_hp(resolve_turn(reg, pos, actions, budget=ENUMERATED)).get(0, 0.0)
    assert 0.0 < chance < 1.0, chance
    return pos, actions, chance


def test_an_uncertain_knockout_forks_with_its_enumerated_chance(
    reg: Regulation, uncertain: tuple[Position, list, float]
) -> None:
    pos, actions, chance = uncertain
    pinned = resolve_turn(reg, pos, actions, budget=QUIET)
    assert _foe_hp(pinned).get(0, 0.0) == 0.0, "the pinned median alone never knocks out"

    forked = resolve_turn(reg, pos, actions, budget=KNOCKOUT)
    hp = _foe_hp(forked)
    assert len(forked.branches) == 2, hp
    assert hp.get(0, 0.0) == pytest.approx(chance, abs=1e-12)
    assert sum(hp.values()) == pytest.approx(1.0, abs=1e-12)

    # Both classes are real draws: positions the enumerated turn reaches.
    reached = {_key(b.position) for b in resolve_turn(reg, pos, actions, budget=ENUMERATED).branches}
    for b in forked.branches:
        assert _key(b.position) in reached
    # The survivor is the pinned roll itself (no crit, the median), as without the fork.
    survivor = [b for b in forked.branches if b.position.sides[1].pokemon[0].hp > 0]
    assert _key(survivor[0].position) == _key(pinned.branches[0].position)


def test_a_certain_hit_does_not_fork(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Full HP (no draw knocks out) and 1 HP (every draw does): the turn is the budget's own."""
    pos, actions = _turn(reg, team_a)
    for hp in (pos.sides[1].pokemon[0].hp, 1):
        pos.sides[1].pokemon[0].hp = hp
        plain = resolve_turn(reg, pos, actions, budget=QUIET)
        forked = resolve_turn(reg, pos, actions, budget=KNOCKOUT)
        assert [(_key(b.position), b.probability) for b in forked.branches] == [
            (_key(b.position), b.probability) for b in plain.branches
        ], hp


def test_a_sash_at_full_hp_does_not_fork(reg: Regulation, uncertain: tuple[Position, list, float]) -> None:
    """Deal-damage spares a Focus Sash at full HP, so no draw knocks out."""
    pos, actions, _ = uncertain
    target = pos.sides[1].pokemon[0]
    target.maxhp = target.hp
    target.item = "focussash"
    plain = resolve_turn(reg, pos, actions, budget=QUIET)
    forked = resolve_turn(reg, pos, actions, budget=KNOCKOUT)
    assert len(forked.branches) == len(plain.branches) == 1


def test_the_field_crosses_only_when_on() -> None:
    from pokeuraou.rustnode import dump_budget

    assert "enumerateKnockouts" not in dump_budget(Budget.matrix())
    assert dump_budget(replace(Budget.matrix(), enumerate_knockouts=True))["enumerateKnockouts"]
    assert not Budget.matrix().enumerate_knockouts
