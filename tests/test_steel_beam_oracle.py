"""Steel Beam's half-HP recoil (`mindBlownRecoil`) in the port, played against Showdown (IKA-406).

Showdown a5df827 (the champions mod overrides neither the move nor `applyRecoilDamage`)::

    data/moves.ts
      steelbeam: accuracy: 95, basePower: 140, category: "Special", target: "normal",
                 mindBlownRecoil: true,
                 onMoveFail(target, source, move) {
                     if (move.multihit) return;
                     this.damage(Math.round(source.maxhp / 2), source, source,
                                 this.dex.conditions.get('Steel Beam'));
                 }
    sim/battle-actions.ts applyRecoilDamage (1379-1395)
      else if (move.mindBlownRecoil || move.chloroblastRecoil) recoilDamage = Math.round(pokemon.maxhp / 2);
      const effect = move.mindBlownRecoil ? this.dex.conditions.get(move.name) : 'recoil';
      this.battle.damage(recoilDamage, pokemon, pokemon, effect);
    sim/battle-actions.ts useMoveInner (526): `if (!moveResult) { ...singleEvent('MoveFail'...) }`
    data/abilities.ts  magicguard: `effect.effectType !== 'Move'` blocks;  rockhead: `effect.id === 'recoil'`

The user loses half its max HP, rounded up, after a move that dealt damage (a doll's counts), and
after one that failed in `trySpreadMoveHit` (a miss, Protect, a semi-invulnerable target). A move
with no target never gets there. The effect is `dex.conditions.get('Steel Beam')`, a Condition (not
a Move, which the effect's name suggests): Showdown's Magic Guard stops it (the log shows no recoil)
and Rock Head does not. The port refused every turn with the move ("move field
mindBlownRecoil: steelbeam"); the positive control is the exe before this change
(`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`). The port's misses are a branch, so the
accuracy cases compare Showdown's miss with the port's branches (and the hit branch is checked
to differ, so the comparison can fail).
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

VOLATILES = frozenset({"substitute"})


def _mon(
    species: str, ability: str, moves: list[str], spe: int, item: str | None = None, **sp: int
) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 20, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    spread.update(sp)
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=item, sp=spread)


#: The user: 1 Steel Beam, 2 Protect, 3 Nasty Plot, 4 Substitute.
BEAM = ["steelbeam", "protect", "nastyplot", "substitute"]
#: The partner: Protect only matters.
PARTNER = ["protect", "swordsdance", "calmmind"]
#: Foe A: 1 Dig (first: a locked move is always choice 1), 2 Substitute, 3 Protect, 4 Swords Dance
#: (nothing that hurts the user).
FOE_A = ["dig", "substitute", "protect", "swordsdance"]
FOE_B = ["protect", "swordsdance", "taunt", "substitute"]


def _teams(
    ability: str = "Sturdy", foe_a: str = "Garchomp", foe_a_ability: str = "Sand Veil", **sp: int
) -> tuple[list[TeamSet], list[TeamSet]]:
    mine = [_mon("Aggron", ability, BEAM, 32, **sp), _mon("Kangaskhan", "Inner Focus", PARTNER, 10)]
    theirs = [
        _mon(foe_a, foe_a_ability, FOE_A, 30, **({"hp": 0} if foe_a == "Ditto" else {})),
        _mon("Incineroar", "Blaze", FOE_B, 0),
        _mon("Milotic", "Marvel Scale", FOE_B, 0),
    ]
    return mine, theirs


BEAM_AT_A = "move 1 1, move 2"
#: Garchomp and Incineroar do nothing to anyone.
IDLE = "move 4, move 2"
PROTECT_A = "move 3, move 2"
SUBSTITUTE_A = "move 2, move 2"
DIG = "move 1 2, move 2"
#: A reference line for each case: the user's recoil, from the move's own condition.
RECOIL = "[from] steelbeam"

#: name -> (teams, setup turns, the compared turn, a Showdown log line that shows the case,
#:          the policy's accuracy).
CASES: dict[str, tuple] = {
    "a-hit-costs-half-rounded-up": (_teams(), [], [BEAM_AT_A, IDLE], RECOIL, "hit"),
    "a-hit-costs-half-of-an-even-hp": (_teams(hp=21), [], [BEAM_AT_A, IDLE], RECOIL, "hit"),
    "magic-guard-stops-the-recoil-of-a-hit": (
        _teams("Magic Guard"), [], [BEAM_AT_A, IDLE], "|-damage|p2a: Garchomp", "hit",
    ),
    "a-hit-through-rock-head": (_teams("Rock Head"), [], [BEAM_AT_A, IDLE], RECOIL, "hit"),
    "a-hit-that-faints-the-target-still-recoils": (
        _teams(foe_a="Ditto", foe_a_ability="Limber"), [["move 3, move 2", IDLE]], [BEAM_AT_A, IDLE],
        "|faint|p2a: Ditto", "hit",
    ),
    "a-miss-costs-half": (_teams(), [], [BEAM_AT_A, IDLE], RECOIL, "miss"),
    "magic-guard-stops-the-recoil-of-a-miss": (
        _teams("Magic Guard"), [], [BEAM_AT_A, IDLE], "|-miss|p1a: Aggron|p2a: Garchomp", "miss",
    ),
    "a-protected-target-costs-half": (_teams(), [], [BEAM_AT_A, PROTECT_A], RECOIL, "hit"),
    "a-substitute-takes-it-and-the-user-still-recoils": (
        _teams(), [["move 3, move 2", SUBSTITUTE_A]], [BEAM_AT_A, IDLE], RECOIL, "hit",
    ),
    "a-digging-target-costs-half": (
        _teams(foe_a="Snorlax", foe_a_ability="Thick Fat"), [["move 3, move 2", DIG]],
        [BEAM_AT_A, "move 1, move 2"], "|-miss|p1a: Aggron|p2a: Snorlax", "hit",
    ),
    # Controls: the same user where the move is not used or recoils nothing.
    "control-a-user-that-does-not-attack-loses-nothing": (
        _teams(), [], ["move 3, move 2", IDLE], "|-boost|p1a: Aggron|spa|2", "hit",
    ),
}

BUDGET = replace(
    Budget.exact(), enumerate_crit=False, enumerate_secondary=False, enumerate_accuracy=False
).with_fixed_roll(0)


def _state(pos: Position) -> dict[str, tuple]:
    return {
        f"p{index + 1}.{mon.species}": (
            mon.hp,
            mon.fainted,
            mon.item,
            tuple(sorted(v.id for v in mon.volatiles if v.id in VOLATILES)) if not mon.fainted else (),
        )
        for index, side in enumerate(pos.sides)
        for mon in side.pokemon
    }


def _play(
    oracle: Oracle, teams: tuple, setup: list, choices: list[str], policy: RandomnessPolicy
) -> tuple[Position, dict, list[str]]:
    mine, theirs = teams
    handle = oracle.create(FORMAT_ID, mine, theirs, policy=policy)
    handle.step(["team 12", "team 12"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = _state(Position.from_json(handle.position))
    log = list(handle.log)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    return before, after, log


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choices[side] in menu, (choices[side], sorted(menu))
        out.append(menu[choices[side]])
    return out


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_port_matches_showdown(reg, oracle: Oracle, bridged: None, name: str) -> None:  # noqa: ANN001
    teams, setup, choices, shown, accuracy = CASES[name]
    before, theirs, log = _play(oracle, teams, setup, choices, RandomnessPolicy(accuracy=accuracy))
    assert any(shown in line for line in log), f"Showdown did not do what {name} says: {log}"
    node = rustnode.node_for(reg)
    assert node is not None
    actions = _actions(reg, before, choices)
    if accuracy == "hit":
        there = node.resolve(before, actions, BUDGET)
        assert there is not None, f"the port refused the turn: {node.refusal}"
        assert len(there.branches) == 1, there.branches
        picked = node.resolve(before, actions, BUDGET, select=0)
        assert picked is not None and picked.position is not None
        rust_now = _state(picked.position)
        assert rust_now == theirs, f"{name}: showdown {theirs} != rust {rust_now}"
        return
    budget = replace(BUDGET, enumerate_accuracy=True)
    there = node.resolve(before, actions, budget)
    assert there is not None, f"the port refused the turn: {node.refusal}"
    states = []
    for index in range(len(there.branches)):
        picked = node.resolve(before, actions, budget, select=index)
        assert picked is not None and picked.position is not None
        states.append(_state(picked.position))
    assert theirs in states, f"{name}: showdown {theirs} is none of the port's branches {states}"
    assert len({tuple(sorted(s.items())) for s in states}) >= 2, "the hit and the miss should differ"


def test_the_half_rounds_up_for_an_odd_hp_only(oracle: Oracle) -> None:
    """`Math.round(maxhp / 2)`: one Aggron whose half rounds up (a floor would differ) and one
    whose half is whole, so the two hit cases hold the rounding and not just the fraction."""
    remainders = set()
    for name in ("a-hit-costs-half-rounded-up", "a-hit-costs-half-of-an-even-hp"):
        teams, setup, choices, _shown, accuracy = CASES[name]
        before, _after, _log = _play(oracle, teams, setup, choices, RandomnessPolicy(accuracy=accuracy))
        remainders.add(before.sides[0].pokemon[0].maxhp % 2)
    assert remainders == {0, 1}, remainders
