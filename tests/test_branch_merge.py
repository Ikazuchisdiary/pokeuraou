"""The lossless branch merge.

Three things have to hold, and they fail in different ways.

*It folds what it should.* The sixteen damage rolls of a guaranteed knock-out are one
position, and integer HP collapses the survivors too. That is the saving.

*It folds nothing else.* Two branches merge only when everything that decides the rest of
the turn is equal, so the distribution over successor positions is unchanged. Checked here
against the unmerged tree with `to_json` as the key -- deliberately not the resolver's own
comparison, which is what is on trial.

The turns are the port's (IKA-210); the audit that every field of Python's `_Turn` is
either compared or named as ignored went with Python's merge.

*It never costs exactness.* Fewer branches means the cap binds less often and each action
gets more of the resolution budget, so a turn that was exact without the merge must be
exact with it. The reverse would be the optimisation paying for itself with accuracy.
"""

from __future__ import annotations

import json
from dataclasses import replace

from pokeuraou.actions import side_actions
from pokeuraou.oracle import TeamSet
from pokeuraou.position import Position
from pokeuraou.regulation import Regulation

from ._port import Budget, TurnResult, resolve_turn
from .test_actions import _synthetic_position

#: Enough branching to be worth merging, small enough that neither arm hits the cap: four
#: rolls per hit and nothing else enumerated, so a four-action turn is at most 256
#: branches unmerged, and a cap far above that.
ROOMY = Budget(
    damage_rolls=4,
    enumerate_crit=False,
    enumerate_accuracy=False,
    enumerate_status_checks=False,
    enumerate_secondary=False,
    max_branches=10_000,
)


def distribution(result: TurnResult) -> dict[str, float]:
    """Successor position -> probability, keyed by the wire format rather than by `==`."""
    out: dict[str, float] = {}
    for branch in result.branches:
        key = json.dumps(branch.position.to_json(), sort_keys=True)
        out[key] = out.get(key, 0.0) + float(branch.probability)
    for pause in result.suspended:
        key = "paused " + json.dumps(pause.position.to_json(), sort_keys=True)
        out[key] = out.get(key, 0.0) + float(pause.probability)
    return out


def pairs(reg: Regulation, pos: Position, count: int = 6) -> list[tuple]:
    """A spread of action pairs: both sides attacking, in the order the engine lists them."""
    ours = list(side_actions(reg, pos, 0))
    theirs = list(side_actions(reg, pos, 1))
    step = max(1, len(ours) // count)
    return [(ours[i], theirs[(i * 7) % len(theirs)]) for i in range(0, len(ours), step)][:count]


def test_a_guaranteed_knockout_stops_being_sixteen_branches(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """The measurement the issue was opened on, as a test.

    Every other source of chance is off, so the only thing that can branch is the damage
    roll -- and against a target the roll cannot save, every roll is the same position.
    """
    pos = _synthetic_position(reg, team_a)
    for mon in pos.sides[1].pokemon:
        mon.hp = 1
    # Only the damage roll may branch, and the cap is the shipped one. Unmerged, four
    # damaging actions are 16^4 states and the cap takes what it can; merged, the rolls
    # that decided nothing are gone and the turn fits.
    rolls_only = Budget(
        enumerate_crit=False,
        enumerate_accuracy=False,
        enumerate_status_checks=False,
        enumerate_secondary=False,
        enumerate_speed_ties=False,
    )
    ours, theirs = pairs(reg, pos, count=6)[0]

    plain = resolve_turn(
        reg, pos, [ours, theirs], budget=replace(rolls_only, merge_duplicates=False)
    )
    merged = resolve_turn(reg, pos, [ours, theirs], budget=rolls_only)

    outcomes = len(merged.branches) + len(merged.suspended)
    assert outcomes < len(plain.branches) + len(plain.suspended)
    assert merged.exact, "the whole turn fits once the duplicate rolls are folded"
    assert not plain.exact, (
        "without the merge this turn does not fit the cap -- if it now does, the fixture "
        "stopped being the case the issue was opened on"
    )


def _close(a: dict[str, float], b: dict[str, float]) -> bool:
    keys = set(a) | set(b)
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) < 1e-12 for k in keys)


def test_merging_leaves_the_distribution_alone(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Same turn, same probabilities over the same positions, merged or not."""
    pos = _synthetic_position(reg, team_a)
    checked = 0
    for ours, theirs in pairs(reg, pos):
        for budget in (Budget.matrix(), ROOMY):
            plain = resolve_turn(
                reg, pos, [ours, theirs], budget=replace(budget, merge_duplicates=False)
            )
            merged = resolve_turn(reg, pos, [ours, theirs], budget=budget)
            if not (plain.exact and merged.exact):
                # The cap bound in one of them, so they are answering different questions:
                # one of the two dropped branches and renormalised. Compared elsewhere.
                continue
            checked += 1
            assert _close(distribution(merged), distribution(plain)), (
                f"{ours.to_choice()} vs {theirs.to_choice()} under {budget}: the merge "
                "changed the distribution"
            )
            assert abs(merged.total_probability - 1.0) < 1e-9
    assert checked >= 4, "too few exact turns to have tested anything"


def test_merging_never_makes_a_turn_less_exact(reg: Regulation, team_a: list[TeamSet]) -> None:
    """Merging can only take pressure off the cap, never add to it."""
    pos = _synthetic_position(reg, team_a)
    for ours, theirs in pairs(reg, pos):
        # The shipped pair plus one that caps hard on purpose: the property is about what
        # happens when the cap binds, and `Budget.exact()` on a mirror is minutes.
        for budget in (Budget.matrix(), Budget.fast(), replace(ROOMY, max_branches=24)):
            plain = resolve_turn(
                reg, pos, [ours, theirs], budget=replace(budget, merge_duplicates=False)
            )
            merged = resolve_turn(reg, pos, [ours, theirs], budget=budget)
            assert merged.exact or not plain.exact, (
                f"{ours.to_choice()} vs {theirs.to_choice()} under {budget}: exact "
                "without the merge and inexact with it"
            )
            assert len(merged.branches) <= len(plain.branches) or not plain.exact, (
                "an exact turn cannot gain branches by folding duplicates"
            )


def test_a_speed_tie_that_ends_the_same_way_is_one_outcome(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Ties are resolved above the queue, so their duplicates meet only at the end.

    The fixture is a mirror, so every pair is tied and both orders are enumerated. Where
    the two orders end in the same position -- both sides act, neither faints -- that is
    one outcome with the weight of both, not two outcomes at half each.
    """
    pos = _synthetic_position(reg, team_a)
    tied = 0
    for ours, theirs in pairs(reg, pos):
        budget = Budget.matrix()
        plain = resolve_turn(
            reg, pos, [ours, theirs], budget=replace(budget, merge_duplicates=False)
        )
        merged = resolve_turn(reg, pos, [ours, theirs], budget=budget)
        if not (plain.exact and merged.exact):
            continue
        if len(merged.branches) < len(plain.branches):
            tied += 1
            assert _close(distribution(merged), distribution(plain))
    assert tied, "the mirror should have produced at least one tie that folds"
