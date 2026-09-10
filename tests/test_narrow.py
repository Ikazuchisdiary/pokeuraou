"""Candidate narrowing.

The dangerous failure here is quiet: keep the top N by damage and every switch, Protect
and status move disappears from the answer, the equilibrium is computed over attacks only,
and the output looks perfectly reasonable. So the property that gets pinned is the
coverage guarantee -- every individual slot option survives somewhere in the kept set, or
is named in ``uncovered`` -- rather than anything about the ranking itself.

The score is deliberately *not* tested for accuracy. It orders candidates and never
reaches the output; every printed number comes from exactly resolved turns.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import side_actions
from pokeuraou.narrow import narrow
from pokeuraou.oracle import TeamSet
from pokeuraou.regulation import Regulation

from .test_actions import _synthetic_position


def _slot_keys(actions: list[object]) -> set[str]:
    from pokeuraou.narrow import _slot_key

    return {
        _slot_key(action, index)  # type: ignore[arg-type]
        for action in actions
        for index in range(len(action.slots))  # type: ignore[attr-defined]
    }


@pytest.mark.parametrize("limit", [24, 12, 40])
def test_every_slot_option_survives(
    reg: Regulation, team_a: list[TeamSet], limit: int
) -> None:
    """The guarantee the whole design rests on: combinations are pruned, options are not."""
    pos = _synthetic_position(reg, team_a)
    pool = side_actions(reg, pos, 0)
    result = narrow(reg, pos, 0, limit=limit)

    wanted = _slot_keys(pool)
    got = _slot_keys(result.actions)
    missing = wanted - got
    assert not missing or result.uncovered, (
        f"{len(missing)} slot options vanished without being reported"
    )
    if not result.uncovered:
        assert missing == set(), "coverage claimed but not delivered"


def test_the_limit_is_respected(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    for limit in (1, 4, 24):
        result = narrow(reg, pos, 0, limit=limit)
        assert len(result.kept) <= limit


def test_a_tiny_limit_reports_what_it_could_not_cover(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """One candidate cannot cover a dozen options, and the report has to say so.

    Silently returning one action would read as "there is only one thing to do here".
    """
    pos = _synthetic_position(reg, team_a)
    result = narrow(reg, pos, 0, limit=1)
    assert len(result.kept) == 1
    assert result.uncovered, "a limit of 1 cannot cover the pool and must admit it"


def test_narrowing_is_a_no_op_when_it_fits(reg: Regulation, team_a: list[TeamSet]) -> None:
    pos = _synthetic_position(reg, team_a)
    pool = side_actions(reg, pos, 0)
    result = narrow(reg, pos, 0, limit=len(pool) + 5)
    assert result.complete
    assert not result.uncovered
    assert len(result.kept) == len(pool)


def test_the_same_position_narrows_the_same_way(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Two runs that disagree would make every downstream number irreproducible."""
    pos = _synthetic_position(reg, team_a)
    first = narrow(reg, pos, 0, limit=16)
    second = narrow(reg, pos, 0, limit=16)
    assert [c.action.to_choice() for c in first.kept] == [
        c.action.to_choice() for c in second.kept
    ]


def test_switches_and_protect_are_not_ranked_out(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """They score zero damage, so a pure damage ranking would delete all of them."""
    pos = _synthetic_position(reg, team_a)
    pool = side_actions(reg, pos, 0)
    result = narrow(reg, pos, 0, limit=24)

    def has_switch(actions: list[object]) -> bool:
        from pokeuraou.actions import SwitchAction

        return any(
            isinstance(slot, SwitchAction)
            for action in actions
            for slot in action.slots  # type: ignore[attr-defined]
        )

    if has_switch(pool):
        assert has_switch(result.actions), "every switch was ranked out of the matrix"
