"""How many times a [2, 5] multi-hit move hits (IKA-160).

Showdown's `hitStepMoveHitLoop` draws the count for a [2, 5] move in generation 5 and later
with `sample([2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5])` -- 35-35-15-15
(vendor/pokemon-showdown/sim/battle-actions.ts, and the champions mod's own copy in
data/mods/champions/scripts.ts). The resolver had 1/3, 1/3, 1/6, 1/6, the older
`[2, 2, 3, 3, 4, 5]`. Skill Link's `onModifyMove` replaces the range with its upper end
before anything is drawn, and the resolver never read it, so Mega Heracross's and
Toucannon's Bullet Seed hit twice in the pinned differential test and 2-5 in the search.

The distribution is read off the simulator: the oracle records the values `sample` was
handed, so the test holds the constant to Showdown's array rather than to a comment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.regulation import Regulation

from ._port import Budget, resolve_turn
from .conftest import FORMAT_ID


def _mon(species: str, ability: str, moves: list[str], spe: int) -> TeamSet:
    return TeamSet(
        species=species,
        ability=ability,
        nature="Serious",
        moves=moves,
        item=None,
        sp={"hp": 20, "atk": 20, "def": 10, "spa": 0, "spd": 10, "spe": spe},
    )


# p1a Garchomp (Rough Skin) Scale Shots the Charizard across from it and p1b Toucannon
# (Skill Link) Bullet Seeds the Kingambit, which takes half and survives five hits. The
# other side only Swords Dances. No species twice on a side: the port reads that as a
# transformed Pokemon and refuses the turn.
TEAM_A = [
    _mon("Garchomp", "Rough Skin", ["scaleshot", "protect"], 10),
    _mon("Toucannon", "Skill Link", ["bulletseed", "protect"], 9),
    _mon("Kingambit", "Defiant", ["protect", "ironhead"], 8),
    _mon("Sylveon", "Pixilate", ["protect", "hypervoice"], 7),
]
TEAM_B = [
    _mon("Charizard", "Blaze", ["swordsdance", "protect"], 2),
    _mon("Kingambit", "Defiant", ["swordsdance", "protect"], 1),
    _mon("Sylveon", "Pixilate", ["protect", "calmmind"], 1),
    _mon("Garchomp", "Rough Skin", ["protect", "swordsdance"], 3),
]
CHOICES = ["move 1 1, move 1 2", "move 1, move 1"]

#: What the oracle's policy pins: no crit, no chance secondary, the maximum roll, and the
#: first element of every `sample`.
BUDGET = replace(Budget.exact(), enumerate_crit=False, enumerate_secondary=False).with_fixed_roll(0)


@pytest.mark.oracle
def test_the_constant_is_the_array_the_simulator_samples(oracle: Oracle) -> None:
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    handle.step(CHOICES)
    assert handle.choice_errors == [], handle.choice_errors
    samples = [[int(v) for v in r["values"]] for r in handle.rolls if r["kind"] == "sample"]
    handle.close()
    # One draw, for Garchomp's Scale Shot: Skill Link's move has no range left to draw.
    assert len(samples) == 1, samples
    counts = Counter(samples[0])
    theirs = {hits: n / len(samples[0]) for hits, n in sorted(counts.items())}
    assert theirs == pytest.approx({2: 0.35, 3: 0.35, 4: 0.15, 5: 0.15}), samples[0]


@pytest.mark.oracle
def test_the_port_counts_the_hits_the_simulator_samples(reg: Regulation, oracle: Oracle) -> None:
    """The port's branches on Scale Shot's count carry Showdown's own weights, and Skill
    Link's Bullet Seed does not branch at all (IKA-210; was Python's `multihit_counts`).

    Accuracy is pinned so that the count is the only chance left: each Charizard HP is one
    count, the more hits the less HP. The pre-IKA-160 port has 1/3, 1/3, 1/6, 1/6.
    """
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    handle.step(CHOICES)
    samples = [[int(v) for v in r["values"]] for r in handle.rolls if r["kind"] == "sample"]
    handle.close()
    assert len(samples) == 1, samples
    theirs = {hits: n / len(samples[0]) for hits, n in sorted(Counter(samples[0]).items())}
    register_mega_stones(reg)
    actions = [
        next(a for a in side_actions(reg, before, side) if a.to_choice() == CHOICES[side])
        for side in (0, 1)
    ]
    budget = replace(Budget.matrix(), enumerate_accuracy=False)
    result = resolve_turn(reg, before, actions, budget=budget)
    assert not result.suspended
    charizard = result.collapse(lambda p: p.sides[1].pokemon[0].hp)
    weights = [charizard[hp] for hp in sorted(charizard, reverse=True)]
    by_count = dict(zip(sorted(theirs), weights, strict=True))
    assert by_count == pytest.approx(theirs), (charizard, theirs)
    kingambit = {b.position.sides[1].pokemon[1].hp for b in result.branches}
    assert len(kingambit) == 1, kingambit


# ---------------------------------------------------------------------------
# The port against Showdown, not against Python (IKA-207).


@pytest.mark.oracle
def test_the_ports_skill_link_agrees_with_the_simulator(reg: Regulation, oracle: Oracle, port) -> None:  # noqa: ANN001
    """`test_skill_link_agrees_with_the_simulator` with the port's pinned outcome."""
    from ._port_showdown import port_turn

    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    handle.step(CHOICES)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    log = list(handle.log)
    handle.close()
    assert "|-hitcount|p2b: Kingambit|5" in log, log
    theirs = [m.hp for m in after.sides[1].pokemon]
    actions = [
        next(a for a in side_actions(reg, before, side) if a.to_choice() == CHOICES[side])
        for side in (0, 1)
    ]
    ours = port_turn(port, before, actions)
    assert [m.hp for m in ours.sides[1].pokemon] == theirs
