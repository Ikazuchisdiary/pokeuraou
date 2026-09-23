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


def test_a_fake_out_that_cannot_work_is_not_offered_as_a_candidate() -> None:
    """Not worth a candidate slot where it is legal, and not legal in champions.

    In the champions dex Fake Out's `onDisableMove` takes it off the request once its user
    has moved (IKA-166), so `side_actions` no longer offers it; `drop_dead_actions` is the
    guard for a pool that still holds one (a dex without the hook, where Showdown fails it
    at execution). A move that provably does nothing is dominated by every other move, and
    the search was not merely spending one of its 24 slots on one: in a recorded game the
    equilibrium put 41.6% of a node's weight on a Fake Out that could not fire, because the
    value function's cells do not know the move is dead.
    """
    from pokeuraou.actions import side_actions
    from pokeuraou.damage import register_mega_stones
    from pokeuraou.narrow import drop_dead_actions
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import load_roster

    roster = load_roster("rizabanadohido")
    reg = roster.reg
    register_mega_stones(reg)
    incineroar = roster.sets[5]
    assert "fakeout" in incineroar.moves
    sets = [incineroar, roster.sets[1], roster.sets[2], roster.sets[3]]
    pos = position_from_sets(reg, sets, sets)

    def fake_outs(position) -> int:  # noqa: ANN001
        pool = side_actions(reg, position, 0)
        kept = drop_dead_actions(reg, position, 0, pool)
        return sum(
            1
            for action in kept
            for slot_action in action.slots
            if getattr(slot_action, "move_id", None) == "fakeout"
        )

    # Turn one out: it works, so it must be offered.
    assert pos.sides[0].pokemon[0].active_move_actions == 0
    assert fake_outs(pos) > 0

    # The Fake Out combinations of the first turn out, kept for the pool checks below.
    pool = side_actions(reg, pos, 0)
    only = [
        action
        for action in pool
        if any(getattr(s, "move_id", None) == "fakeout" for s in action.slots)
    ]
    assert only, "the position must offer at least one Fake Out combination"

    # After one move action the champions request disables it (IKA-166), and a pool that
    # still holds it drops it: `onTry` would fail it, since the counter is bumped first.
    pos.sides[0].pokemon[0].active_move_actions = 1
    assert fake_outs(pos) == 0
    kept = drop_dead_actions(reg, pos, 0, pool)
    assert kept and not any(action in only for action in kept)

    # The last action is never dropped: an empty list becomes Struggle, which is a
    # different and illegal action.
    assert drop_dead_actions(reg, pos, 0, only) == only


def test_the_scorer_prices_last_respects_at_its_real_power() -> None:
    """Variable base power has to reach the *scorer*, not only the resolver.

    Last Respects is 50 + 50 per fainted ally, from a `basePowerCallback` -- the declared
    `basePower: 50` in the move data is only its floor. The resolver built a MoveContext
    and got this right; `narrow` and `observe` called the calculator without one and took
    the default, where no ally has ever fainted. So the ranking that decides which actions
    the search even considers priced a 200-power move at 50, and the belief layer inferred
    spreads from the same wrong number.
    """
    from pokeuraou.damage import calculate, register_mega_stones
    from pokeuraou.priors import SampledSet
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import load_roster
    from pokeuraou.view import battler, field_state, move_context

    roster = load_roster("rizabanadohido")
    reg = roster.reg
    register_mega_stones(reg)
    basculegion = SampledSet(
        species="basculegion",
        ability="adaptability",
        item="lifeorb",
        nature="Adamant",
        sp={"hp": 8, "atk": 32, "spe": 26},
        moves=["lastrespects", "wavecrash", "aquajet", "protect"],
    )
    pos = position_from_sets(
        reg, list(roster.sets[:4]), [basculegion, *roster.sets[1:4]]
    )
    attacker = battler(reg, pos.sides[1].pokemon[0])
    defender = battler(reg, pos.sides[0].pokemon[0])

    def damage() -> int:
        result = calculate(
            reg,
            attacker,
            defender,
            "lastrespects",
            field_state(pos, reg),
            defender_side=0,
            spread=False,
            # The context a scorer can build from the position alone -- which is the
            # thing that was missing.
            move_ctx=move_context(pos, 1, 0),
        )
        return int(result.rolls.max())

    fresh = damage()
    # Faint two of *their* bench, not their actives: emptying an active slot changes the
    # position rather than the move's power, which is a different measurement.
    for index in (2, 3):
        pos.sides[1].pokemon[index].fainted = True
        pos.sides[1].pokemon[index].hp = 0
    after = damage()
    assert after >= 2.5 * fresh, (
        f"two fainted allies must roughly triple it: {fresh} -> {after}"
    )
