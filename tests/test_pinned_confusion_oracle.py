"""Under the differential tools' pinned policy a confused Pokemon hits itself, as Showdown's
pinned oracle does (IKA-325).

Showdown a5df827, data/conditions.ts (no champions override)::

    confusion: {
        onBeforeMovePriority: 3,
        onBeforeMove(pokemon) {
            ...
            this.add('-activate', pokemon, 'confusion');
            if (!this.randomChance(33, 100)) {
                return;
            }
            ... this.damage(damage, pokemon, pokemon, activeMove) ...
            return false;
        },
    },

The oracle's `installPolicy` answers every `randomChance(n, 100)` as an accuracy roll: `hit`,
the policy `diff_turn` and `diverge_report` use, so its confused Pokemon hits itself every
time. `Budget.deterministic` collapsed the check to "it acts", so every confused turn in those
tools diverged without a note -- IKA-316 met one, an Electro Shot fired from its charge; the
charge is not what matters (the plain case below is the same). A searched budget, which
branches the self-hit at 33%, is unchanged: that is the control. The positive control is the
exe before this change (`POKEURAOU_RUST_NODE_BIN=<old exe> pytest this-file`).
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position

from ._port import Budget, require_binary, resolve_turn
from .conftest import FORMAT_ID

pytestmark = pytest.mark.oracle


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    """Not legal sets: the bridge does not check learnsets or abilities."""
    spread = {"hp": 32, "atk": 20, "def": 10, "spa": 20, "spd": 10, "spe": spe}
    return TeamSet(species=species, ability=ability, nature="Serious", moves=moves, item=None, sp=spread)


MINE = [
    # 1 Flash Cannon, 2 Electro Shot
    _mon("Archaludon", "Sturdy", ["flashcannon", "electroshot", "protect", "dracometeor"], 0),
    # 1 Protect, 2 Swords Dance
    _mon("Milotic", "Marvel Scale", ["protect", "swordsdance", "recover", "icebeam"], 0),
]
THEIRS = [
    # 1 Confuse Ray, 2 Protect (Levitate: Cursed Body could disable on the hit)
    _mon("Gengar", "Levitate", ["confuseray", "protect", "shadowball", "helpinghand"], 32),
    # 1 Protect, 2 Bulk Up
    _mon("Incineroar", "Blaze", ["protect", "bulkup", "helpinghand", "flareblitz"], 10),
]

#: name -> (setup turns, Showdown's choices, ours when they differ).
CASES: dict[str, tuple] = {
    "flash-cannon": ([], ["move 1 1, move 1", "move 1 1, move 1"], None),
    # Electro Shot charged last turn; its second turn is Showdown's `move 1`, the port's `move 2`.
    "charged-electro-shot": (
        [["move 2 1, move 1", "move 2, move 1"]],
        ["move 1, move 2", "move 1 1, move 2"],
        ["move 2, move 2", "move 1 1, move 2"],
    ),
}


def _actions(reg, pos: Position, choices: list[str]) -> list:  # noqa: ANN001
    out = []
    for side in (0, 1):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choices[side] in menu, (choices[side], sorted(menu))
        out.append(menu[choices[side]])
    return out


def _play(oracle: Oracle, name: str) -> tuple[Position, list[str], Position, list[str]]:
    setup, choices, ours = CASES[name]
    handle = oracle.create(FORMAT_ID, MINE, THEIRS, policy=RandomnessPolicy())
    handle.step(["team 12", "team 12"])
    for turn in setup:
        handle.step(turn)
        assert handle.choice_errors == [], handle.choice_errors
    before = Position.from_json(handle.position)
    handle.step(choices)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    log = list(handle.log)
    handle.close()
    for side in before.sides:
        for party in side.pokemon:
            party.stats_override = None
    return before, ours or choices, after, log


def _hp(pos: Position) -> dict[str, int]:
    return {f"p{i + 1}.{mon.species}": mon.hp for i, side in enumerate(pos.sides) for mon in side.pokemon}


@pytest.mark.parametrize("name", sorted(CASES))
def test_pinned_confusion_hits_itself_as_showdown_does(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    require_binary()
    before, ours, after, log = _play(oracle, name)
    assert any("[from] confusion" in line for line in log), log
    result = resolve_turn(reg, before, _actions(reg, before, ours), budget=Budget.deterministic(0))
    assert len(result.branches) == 1, result.branches
    assert _hp(result.branches[0].position) == _hp(after), name


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_searched_budget_still_branches_the_self_hit(reg, oracle: Oracle, name: str) -> None:  # noqa: ANN001
    """The control: not pinned, the self-hit is a 33% branch, as before."""
    require_binary()
    before, ours, _after, _log = _play(oracle, name)
    result = resolve_turn(reg, before, _actions(reg, before, ours), budget=Budget.exact().with_fixed_roll(0))
    user = before.sides[0].pokemon[before.sides[0].active[0]]
    hurt = sum(
        b.probability for b in result.branches
        if b.position.sides[0].pokemon[before.sides[0].active[0]].hp < user.hp
    )
    assert hurt == pytest.approx(0.33), name
