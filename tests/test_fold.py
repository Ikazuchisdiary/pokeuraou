"""The fold: a turn's leaves and the tree that turns their values into the turn's value.

Chance is an `Average` and a mid-turn replacement a `BestOf` (`pokeuraou.fold`). The port
builds the tree (`port.turn_leaves`, and the folds it sends with an encoded node); these
tests hold what it builds to what the search needs of it. Moved from test_resolve.py, where
they folded Python's resolver's turns (IKA-210).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from pokeuraou.actions import MoveAction, PassAction, SideAction, side_actions
from pokeuraou.fold import Average, BestOf
from pokeuraou.oracle import TeamSet
from pokeuraou.position import MoveSlot
from pokeuraou.regulation import Regulation

from ._port import Budget, resolve_turn, resume_alternatives, turn_expectation, turn_leaves
from .test_actions import _synthetic_position


def test_summarising_a_suspended_turn_is_refused(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """A partial expectation is worse than an error, because it looks like a number.

    Averaging over `branches` while a suspension holds part of the mass gives a value short
    by that fraction and perfectly ordinary-looking. Every caller has to answer the
    replacement first.
    """
    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.moves[0] = MoveSlot(id="uturn", pp=20, maxpp=20)
    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="uturn", target=1),
            PassAction(slot=1),
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    assert result.suspended

    with pytest.raises(ValueError, match="suspended"):
        result.expected(lambda _p: 0.0)

    # ...and the fold does produce a number, once the choice is part of it.
    value, _flags = turn_expectation(reg, result, lambda _p: 0.5)
    assert value == pytest.approx(0.5)


def test_the_mid_turn_replacement_is_a_choice_and_not_an_average(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """The interrupted side gets its best option, not the mean of all of them.

    This is the whole reason `TurnLeaves` exists. Which Pokemon comes in is a decision, so
    averaging over the bench would price a Parting Shot as if the player brought in
    something at random. Zero-sum means side 0 maximises and side 1 minimises, and the
    direction is asserted both ways because getting the sign backwards would be invisible
    in aggregate.
    """
    pos = _synthetic_position(reg, team_a)
    mover = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mover.moves[0] = MoveSlot(id="uturn", pp=20, maxpp=20)
    ours = SideAction(
        slots=(
            MoveAction(slot=0, move_index=1, move_id="uturn", target=1),
            PassAction(slot=1),
        )
    )
    theirs = side_actions(reg, pos, 1)[0]
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.deterministic(8))
    assert result.suspended

    chooser, alternatives = resume_alternatives(reg, result.suspended[0])
    assert chooser == 0
    assert len(alternatives) >= 2, "a bench of two is the point of the test"

    # Score each candidate by which species ended up in the slot, so the values are
    # distinct and the best one is known independently of the value function.
    plan = turn_leaves(reg, result)
    species = [
        (p.sides[0].pokemon[p.sides[0].active[0]].species if p.sides[0].active[0] is not None else "")
        for p in plan.positions
    ]
    scores = {name: float(i + 1) for i, name in enumerate(sorted(set(species)))}
    values = [scores[name] for name in species]

    assert plan.value(values) == pytest.approx(max(scores.values())), (
        "side 0 chooses, so the fold must take the option it likes most"
    )
    # Flip the fold's owner and the same leaves must produce the worst value instead.
    flipped = replace(
        plan,
        root=Average(
            parts=[
                (w, BestOf(chooser=1, options=node.options) if isinstance(node, BestOf) else node)
                for w, node in plan.root.parts
            ]
        ),
    )
    assert flipped.value(values) == pytest.approx(min(scores.values()))
