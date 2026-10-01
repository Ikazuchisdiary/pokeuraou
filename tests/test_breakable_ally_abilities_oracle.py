"""Mold Breaker passes Friend Guard and Aura Break (IKA-414).

Showdown a5df827 (no champions override of either)::

    // sim/battle.ts, runEvent: a `breakable` ability's handler is skipped while
    suppressingAbility(effectHolder) {
        return this.activePokemon && this.activePokemon.isActive &&
            (this.activePokemon !== target || this.gen < 8) &&
            this.activeMove && this.activeMove.ignoreAbility && !target?.hasItem('Ability Shield');
    }
    // data/abilities.ts
    friendguard: { onAnyModifyDamage(...) { ... return this.chainModify(0.75); }, flags: { breakable: 1 } },
    aurabreak:   { onAnyTryPrimaryHit(target, source, move) { move.hasAuraBreak = true; },
                   flags: { breakable: 1 } },

Both act on a *third* Pokemon's hit, so the port's damage layer (which skipped the defender's
own ability for a Mold Breaker, but kept the ally's Friend Guard and the field's Aura Break)
left them on: Mold Breaker, Teravolt and Turboblaze users hit an ally of a Friend Guard holder
for 0.75x and a Fairy or Dark move met the broken 0.75x aura instead of the full 1.33x.
Ability Shield on the holder keeps its ability. Each case plays in Showdown and holds the
port's one outcome to it. The positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`); the controls are a user without Mold
Breaker (the port was right already) and the Shield holder.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, require_binary, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 32, "atk": 20, "def": 10, "spa": 20, "spd": 4, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: name -> (user's ability, user's ally ability, target's ally ability, target's ally item, move).
#: The user's ally and the target's ally are the two bystanders; the target has Immunity.
CASES: dict[str, tuple] = {
    "friend-guard-moldbreaker": ("Mold Breaker", "Pressure", "Friend Guard", None, "ironhead"),
    "friend-guard-teravolt": ("Teravolt", "Pressure", "Friend Guard", None, "ironhead"),
    "friend-guard-shield": ("Mold Breaker", "Pressure", "Friend Guard", "Ability Shield", "ironhead"),
    "friend-guard-control": ("Intimidate", "Pressure", "Friend Guard", None, "ironhead"),
    "aura-break-moldbreaker": ("Mold Breaker", "Fairy Aura", "Aura Break", None, "moonblast"),
    "aura-break-shield": ("Mold Breaker", "Fairy Aura", "Aura Break", "Ability Shield", "moonblast"),
    "aura-break-control": ("Intimidate", "Fairy Aura", "Aura Break", None, "moonblast"),
}


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (mon.hp, mon.fainted, mon.status)
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choices[side] in menu, (choices[side], sorted(menu))
        out.append(menu[choices[side]])
    return out


def _teams(name: str) -> tuple[list[TeamSet], list[TeamSet], list[str]]:
    user, user_ally, ally, item, move = CASES[name]
    mine = [
        _mon("Excadrill", user, [move, "protect", "helpinghand", "bulkup"], 20),
        _mon("Sylveon", user_ally, ["protect", "swordsdance", "bulkup", "helpinghand"], 10),
    ]
    theirs = [
        _mon("Snorlax", "Immunity", ["protect", "rest", "bulkup", "helpinghand"], 0),
        _mon("Clefable", ally, ["protect", "swordsdance", "bulkup", "helpinghand"], 0, item),
    ]
    return mine, theirs, ["move 1 1, move 1", "move 3, move 2"]


def _play(oracle: Oracle, name: str):  # noqa: ANN202
    mine, theirs, choices = _teams(name)
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    handle.close()
    return before, after, choices


def _target_loss(after: Position) -> int:
    target = after.sides[1].pokemon[after.sides[1].active[0]]
    return target.maxhp - target.hp


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_hit_matches_showdown(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    require_binary()
    before, after, choices = _play(oracle, name)
    assert _target_loss(after) > 0, name
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    result = resolve_turn(reg, before, _actions(reg, before, choices), budget=BUDGET)
    assert len(result.branches) == 1, result.branches
    assert _state(result.branches[0].position) == _state(after), name


@pytest.mark.parametrize(
    ("passed", "kept"),
    [
        ("friend-guard-moldbreaker", "friend-guard-control"),
        ("friend-guard-teravolt", "friend-guard-control"),
        ("aura-break-moldbreaker", "aura-break-control"),
    ],
)
def test_showdown_moves_with_the_breaker(oracle: Oracle, passed: str, kept: str) -> None:
    """Positive control for the comparison: the reference's own answer differs between the
    Mold Breaker case and the control, and the Shield holder's answer is the control's."""
    losses = {name: _target_loss(_play(oracle, name)[1]) for name in {passed, kept}}
    shield = passed.rsplit("-", 1)[0] + "-shield"
    if shield in CASES:
        losses[shield] = _target_loss(_play(oracle, shield)[1])
        assert losses[shield] == losses[kept], losses
    # Friend Guard's 0.75x is gone; Aura Break's broken 0.75x aura is back to 1.33x.
    assert losses[passed] > losses[kept], losses
