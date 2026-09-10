"""Effective Speed and action order.

Speed is checked against Showdown directly (``tools/diff_speed.py``) rather than inferred
from a wrong turn order, because an order divergence could be a Speed bug or an ordering
bug and the two need different fixes. The order itself is then checked separately
(``tools/diff_order.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.battler import Battler, FieldState
from pokeuraou.oracle import ORACLE_JS
from pokeuraou.regulation import STAT_IDS, Regulation
from pokeuraou.speed import (
    ORDER_MEGA,
    ORDER_MOVE,
    ORDER_SWITCH,
    QueuedAction,
    action_overriding_effects,
    effective_speed,
    fractional_priority,
    move_priority,
    order_groups,
    speed_changing_effects,
)

from . import _diff_order_entry as diff_order
from . import _diff_speed_entry as diff_speed

#: Measured against the pinned Showdown commit. Speed has no known divergence, so this is
#: zero rather than a tolerance.
MAX_SPEED_DIVERGENCE = 0.0

#: Turn order likewise, once the turns Showdown re-sorts mid-turn are excluded (those are
#: unpredictable from the pre-turn state by construction, and the resolver re-sorts).
MAX_ORDER_DIVERGENCE = 0.0


def make_battler(
    reg: Regulation,
    species: str,
    *,
    spe: int,
    ability: str = "noability",
    item: str | None = None,
    status: str | None = None,
    boosts: dict[str, int] | None = None,
    volatiles: frozenset[str] = frozenset(),
) -> Battler:
    """A Battler with a chosen Speed, for testing the modifier chain in isolation."""
    entry = reg.species[species]
    stats = np.zeros((1, len(STAT_IDS)), dtype=np.int64)
    stats[0, STAT_IDS.index("spe")] = spe
    stats[0, 0] = 200
    return Battler(
        species=entry.id,
        types=entry.types,
        ability=ability,
        item=item,
        level=50,
        stats=stats,
        hp=np.array([200]),
        maxhp=np.array([200]),
        boosts=boosts or {},
        status=status,
        volatiles=volatiles,
    )


def test_speed_is_the_raw_stat_when_nothing_applies(reg: Regulation) -> None:
    mon = make_battler(reg, "kingambit", spe=100)
    assert int(effective_speed(reg, mon, FieldState(), frozenset())[0]) == 100


def test_tailwind_doubles_speed(reg: Regulation) -> None:
    mon = make_battler(reg, "kingambit", spe=100)
    assert int(effective_speed(reg, mon, FieldState(), frozenset({"tailwind"}))[0]) == 200


def test_choice_scarf_is_one_and_a_half(reg: Regulation) -> None:
    mon = make_battler(reg, "kingambit", spe=101, item="choicescarf")
    # 101 * 1.5 in Showdown's fixed point: (101 * 6144 + 2047) >> 12 = 151
    assert int(effective_speed(reg, mon, FieldState(), frozenset())[0]) == 151


def test_paralysis_halves_after_the_chain(reg: Regulation) -> None:
    """Paralysis runs at handler priority -101, so it applies to the chained result.

    Halving first and then doubling would give a different answer, which is why the order
    matters rather than just the factors.
    """
    plain = make_battler(reg, "kingambit", spe=101, status="par")
    assert int(effective_speed(reg, plain, FieldState(), frozenset())[0]) == 50

    with_tailwind = make_battler(reg, "kingambit", spe=101, status="par")
    got = int(effective_speed(reg, with_tailwind, FieldState(), frozenset({"tailwind"}))[0])
    assert got == 101  # (101 * 2) * 50 // 100
    assert got != 100  # what halving before doubling would give


def test_quick_feet_replaces_the_paralysis_penalty(reg: Regulation) -> None:
    mon = make_battler(reg, "kingambit", spe=100, ability="quickfeet", status="par")
    assert int(effective_speed(reg, mon, FieldState(), frozenset())[0]) == 150


def test_weather_speed_abilities_need_their_weather(reg: Regulation) -> None:
    mon = make_battler(reg, "venusaur", spe=100, ability="chlorophyll")
    assert int(effective_speed(reg, mon, FieldState(), frozenset())[0]) == 100
    sun = FieldState(weather="sunnyday")
    assert int(effective_speed(reg, mon, sun, frozenset())[0]) == 200


def test_cloud_nine_on_an_ally_suppresses_a_speed_ability(reg: Regulation) -> None:
    """Weather suppression is field-wide, so an ally holding Cloud Nine counts."""
    mon = make_battler(reg, "venusaur", spe=100, ability="chlorophyll")
    sun = FieldState(weather="sunnyday", active_abilities=(("chlorophyll", "cloudnine"), ()))
    assert int(effective_speed(reg, mon, sun, frozenset())[0]) == 100


def test_unburden_needs_the_item_to_have_been_lost(reg: Regulation) -> None:
    """Holding nothing is not enough: Showdown adds a volatile when the item goes."""
    never_held = make_battler(reg, "sneasler", spe=100, ability="unburden", item=None)
    assert int(effective_speed(reg, never_held, FieldState(), frozenset())[0]) == 100

    lost_it = make_battler(
        reg, "sneasler", spe=100, ability="unburden", item=None,
        volatiles=frozenset({"unburden"}),
    )
    assert int(effective_speed(reg, lost_it, FieldState(), frozenset())[0]) == 200


def test_boosts_use_showdown_integer_ratios(reg: Regulation) -> None:
    mon = make_battler(reg, "kingambit", spe=100, boosts={"spe": 1})
    assert int(effective_speed(reg, mon, FieldState(), frozenset())[0]) == 150
    mon = make_battler(reg, "kingambit", spe=100, boosts={"spe": -1})
    assert int(effective_speed(reg, mon, FieldState(), frozenset())[0]) == 66


def test_priority_from_abilities_and_terrain(reg: Regulation) -> None:
    # Tailwind's own priority is 0; Prankster adds one because it is a status move.
    prankster = make_battler(reg, "whimsicott", spe=100, ability="prankster")
    assert reg.moves["tailwind"].priority == 0
    assert move_priority(reg, "tailwind", prankster, FieldState()) == 1
    assert move_priority(reg, "moonblast", prankster, FieldState()) == 0

    plain = make_battler(reg, "kingambit", spe=100)
    assert move_priority(reg, "suckerpunch", plain, FieldState()) == 1
    assert move_priority(reg, "trickroom", plain, FieldState()) == -7

    grassy = FieldState(terrain="grassyterrain")
    if "grassyglide" in reg.moves:
        assert move_priority(reg, "grassyglide", plain, grassy) == 1
        assert move_priority(reg, "grassyglide", plain, FieldState()) == 0


def test_fractional_priority_is_a_branch_only_when_random(reg: Regulation) -> None:
    plain = make_battler(reg, "kingambit", spe=100)
    assert fractional_priority(reg, "ironhead", plain) == [(0.0, 1.0)]

    claw = make_battler(reg, "kingambit", spe=100, item="quickclaw")
    outcomes = fractional_priority(reg, "ironhead", claw)
    assert len(outcomes) == 2
    assert sum(p for _, p in outcomes) == pytest.approx(1.0)

    stall = make_battler(reg, "kingambit", spe=100, ability="stall")
    assert fractional_priority(reg, "ironhead", stall) == [(-0.1, 1.0)]


def _queued(side: int, slot: int, *, priority: int, speed: int, order: int = ORDER_MOVE) -> QueuedAction:
    return QueuedAction(
        side=side, slot=slot, kind="move" if order == ORDER_MOVE else "switch",
        order=order, priority=priority, fractional=0.0,
        speed=np.array([speed]), move_id="tackle",
    )


def test_order_is_by_bracket_then_priority_then_speed() -> None:
    actions = [
        _queued(0, 0, priority=0, speed=100),
        _queued(1, 0, priority=1, speed=50),
        _queued(0, 1, priority=0, speed=200),
        _queued(1, 1, priority=0, speed=150, order=ORDER_SWITCH),
    ]
    groups = order_groups(actions, trick_room=False)
    assert len(groups) == 1
    # Switch bracket first, then the priority move, then the two by Speed.
    assert groups[0].order == (3, 1, 2, 0)
    assert groups[0].ties == ()


def test_mega_resolves_between_switches_and_moves() -> None:
    actions = [
        _queued(0, 0, priority=0, speed=200),
        QueuedAction(
            side=0, slot=0, kind="mega", order=ORDER_MEGA, priority=0, fractional=0.0,
            speed=np.array([200]),
        ),
        _queued(1, 0, priority=0, speed=10, order=ORDER_SWITCH),
    ]
    groups = order_groups(actions, trick_room=False)
    assert groups[0].order == (2, 1, 0)


def test_trick_room_reverses_the_speed_comparison() -> None:
    actions = [_queued(0, 0, priority=0, speed=200), _queued(1, 0, priority=0, speed=50)]
    assert order_groups(actions, trick_room=False)[0].order == (0, 1)
    assert order_groups(actions, trick_room=True)[0].order == (1, 0)


def test_trick_room_does_not_reorder_across_priority() -> None:
    actions = [
        _queued(0, 0, priority=1, speed=1),
        _queued(1, 0, priority=0, speed=999),
    ]
    assert order_groups(actions, trick_room=True)[0].order == (0, 1)


def test_exact_speed_ties_are_reported_not_resolved() -> None:
    actions = [_queued(0, 0, priority=0, speed=100), _queued(1, 0, priority=0, speed=100)]
    groups = order_groups(actions, trick_room=False)
    assert len(groups) == 1
    assert groups[0].ties == ((0, 1),)


def test_a_belief_over_speeds_gives_several_orders() -> None:
    """Different candidate spreads can order the same actions differently.

    This is the reason ``order_groups`` returns groups at all: collapsing to one order
    would throw away exactly the uncertainty a doubles turn hinges on.
    """
    ours = _queued(0, 0, priority=0, speed=120)
    theirs = QueuedAction(
        side=1, slot=0, kind="move", order=ORDER_MOVE, priority=0, fractional=0.0,
        speed=np.array([100, 130, 120]), move_id="tackle",
    )
    groups = order_groups([ours, theirs], trick_room=False)
    orders = {g.order for g in groups}
    assert orders == {(0, 1), (1, 0)}
    tied = [g for g in groups if g.ties]
    assert tied and tied[0].count == 1  # the particle with an exact tie


def test_speed_changing_effects_are_detected() -> None:
    lines = [
        "|move|p1a: X|Tailwind|",
        "|-sidestart|p1: p1|move: Tailwind",
        "|-unboost|p2a: Y|spe|1",
    ]
    assert speed_changing_effects(lines) == {"tailwind", "unboost:spe"}
    assert speed_changing_effects(["|move|p1a: X|Tackle|p2a: Y"]) == set()


def test_action_overriding_effects_are_detected() -> None:
    assert action_overriding_effects(["|-start|p2a: Y|move: Encore"]) == {"encore"}
    assert action_overriding_effects(["|move|p1a: X|Tackle|p2a: Y"]) == set()


@pytest.mark.oracle
def test_speed_matches_showdown(reg: Regulation) -> None:
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    del reg
    report = diff_speed.run(battles=10, seed=1, max_turns=10)
    assert report.compared > 150
    assert report.divergence_rate <= MAX_SPEED_DIVERGENCE, report.render()


@pytest.mark.oracle
def test_turn_order_matches_showdown(reg: Regulation) -> None:
    if not ORACLE_JS.exists():
        pytest.skip("oracle not built")
    del reg
    report = diff_order.run(battles=10, seed=1, max_turns=10)
    assert report.compared > 40
    assert report.divergence_rate <= MAX_ORDER_DIVERGENCE, report.render()
