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

from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.oracle import Oracle, RandomnessPolicy, TeamSet
from pokeuraou.position import Position
from pokeuraou.regulation import Regulation
from pokeuraou.resolve import MULTIHIT_2_5, Budget, multihit_counts, resolve_turn

from .conftest import FORMAT_ID


def _two_to_five(reg: Regulation) -> str:
    return next(k for k, m in sorted(reg.moves.items()) if m.raw.get("multihit") == [2, 5])


def test_a_two_to_five_move_is_35_35_15_15(reg: Regulation) -> None:
    move = reg.moves[_two_to_five(reg)]
    spread = dict(multihit_counts(move, Budget.exact()))
    assert spread == pytest.approx({2: 0.35, 3: 0.35, 4: 0.15, 5: 0.15})
    assert sum(spread.values()) == pytest.approx(1.0)
    # The matrix budget enumerates it too; a pinned one takes the minimum.
    assert dict(multihit_counts(move, Budget.matrix())) == spread
    assert multihit_counts(move, Budget.deterministic(0)) == [(2, 1.0)]


def test_skill_link_hits_the_upper_end_under_every_budget(reg: Regulation) -> None:
    move = reg.moves[_two_to_five(reg)]
    for budget in (Budget.exact(), Budget.matrix(), Budget.deterministic(0)):
        assert multihit_counts(move, budget, "skilllink") == [(5, 1.0)], budget
        # Any other ability is the distribution.
        assert multihit_counts(move, budget, "keeneye") == multihit_counts(move, budget)
    # A fixed count is not a range, and Skill Link leaves it alone.
    for fixed in ("dualwingbeat", "doublehit"):
        if fixed in reg.moves:
            assert multihit_counts(reg.moves[fixed], Budget.exact(), "skilllink") == [(2, 1.0)]


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
    assert theirs == pytest.approx(dict(MULTIHIT_2_5)), (
        f"Showdown samples {samples[0]}; the resolver has {MULTIHIT_2_5}"
    )


@pytest.mark.oracle
def test_skill_link_agrees_with_the_simulator(reg: Regulation, oracle: Oracle) -> None:
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    before = Position.from_json(handle.position)
    handle.step(CHOICES)
    assert handle.choice_errors == [], handle.choice_errors
    after = Position.from_json(handle.position)
    log = list(handle.log)
    handle.close()
    # Showdown's own counts: the pinned two for Garchomp (the control), five for Skill Link.
    assert "|-hitcount|p2a: Charizard|2" in log, log
    assert "|-hitcount|p2b: Kingambit|5" in log, log

    theirs = [m.hp for m in after.sides[1].pokemon]
    actions = [
        next(a for a in side_actions(reg, before, side) if a.to_choice() == CHOICES[side])
        for side in (0, 1)
    ]
    result = resolve_turn(reg, before, actions, budget=BUDGET)
    ours = [
        [m.hp for m in b.position.sides[1].pokemon]
        for b in result.branches
        if not any(e.endswith("missed") for e in b.events)
    ]
    assert ours, "no Python outcome where every move hit"
    for hp in ours:
        assert hp == theirs, f"showdown {theirs} != python {hp}"


@pytest.mark.oracle
def test_the_port_counts_the_hits_as_python_does(reg: Regulation, oracle: Oracle) -> None:
    """The same turn through the Rust port, branch by branch.

    Under the matrix budget Garchomp's Scale Shot count is enumerated, so its weights
    are the 35-35-15-15 ones; under the pinned budget only Skill Link's five hits differ
    from the old answer. The pre-IKA-160 port fails both.
    """
    binary = rustnode.binary_path()
    if not binary.exists():
        pytest.skip(f"no Rust binary at {binary}; `cargo build --release`")
    handle = oracle.create(FORMAT_ID, TEAM_A, TEAM_B, policy=RandomnessPolicy())
    handle.step(["team 1234", "team 1234"])
    pos = Position.from_json(handle.position)
    handle.close()
    # Showdown's positions carry its final stats, which the port refuses as it would a
    # transformed Pokemon's. Nobody here is transformed, so they follow from the spreads.
    for side in pos.sides:
        for mon in side.pokemon:
            mon.stats_override = None
    register_mega_stones(reg)
    actions = [
        next(a for a in side_actions(reg, pos, side) if a.to_choice() == CHOICES[side])
        for side in (0, 1)
    ]
    node = rustnode.RustNode(reg)
    try:
        for budget in (Budget.matrix(), BUDGET):
            here = resolve_turn(reg, pos, actions, budget=budget)
            there = node.resolve(pos, actions, budget)
            assert there is not None, "the port refused the turn"
            assert there.branches == pytest.approx(
                [b.probability for b in here.branches], abs=1e-12
            ), budget
            for index, branch in enumerate(here.branches):
                chosen = node.resolve(pos, actions, budget, select=index)
                assert chosen is not None and chosen.position is not None
                assert chosen.position.to_json() == branch.position.to_json(), (budget, index)
        # Not vacuous: the matrix budget does branch on the count.
        assert len(resolve_turn(reg, pos, actions, budget=Budget.matrix()).branches) > 1
    finally:
        node.close()


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
