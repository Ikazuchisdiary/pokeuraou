"""IKA-419g: the Baton hold, a flag that is off by default.

What these tests hold:

- **the filter**: `side_actions` leaves out the holder's Baton Pass rows while its Special
  Attack and Special Defense are not both at +1 or more, for the held side only, and not at
  all without the hold (positive and negative controls at both edges of the gate);
- **the drives**: the Calm Mind and Baton Pass drive and the screen drive (IKA-444).

The seats (`pool_match_game`) and the check of recorded games (`check_hold_baton`) went with
the board's Python game loop in IKA-458 stage 3.7.
"""

from __future__ import annotations

import pytest

from pokeuraou import actions
from pokeuraou.actions import (
    BATON_HOLD,
    MoveAction,
    SideAction,
    baton_gate_open,
    hold_baton,
    side_actions,
)
from pokeuraou.position import Effect, MoveSlot, Pokemon, Position, Side

MOVES = ["batonpass", "calmmind", "protect", "moonblast"]

#: The species the board's Baton hold held (the removed `poolplay.HOLD_BATON_SPECIES`).
HOLD_BATON_SPECIES = "espathra"


def _mon(slot: int, species: str, boosts: dict[str, int] | None = None) -> Pokemon:
    return Pokemon(
        slot=slot, species=species, base_species=species, types=(), ability="x", nature="Serious",
        moves=[MoveSlot(m, 8, 8) for m in MOVES], hp=100, maxhp=100, boosts=dict(boosts or {}),
    )


def _position(boosts: dict[str, int] | None = None, *, fainted: bool = False) -> Position:
    holder = _mon(0, HOLD_BATON_SPECIES, boosts)
    holder.fainted = fainted
    side0 = Side(id="p1", name="a", active=[0, 1], pokemon=[holder, _mon(1, "raichu"), _mon(2, "x")])
    side1 = Side(
        id="p2", name="b", active=[0, 1],
        pokemon=[_mon(0, HOLD_BATON_SPECIES, boosts), _mon(1, "y"), _mon(2, "z")],
    )
    return Position(format="f", sides=[side0, side1])


@pytest.fixture
def menus(monkeypatch):  # noqa: ANN001, ANN201
    """Slot 0 chooses among the four MOVES, slot 1 between Baton Pass and Protect: 8 joint rows."""

    def slot_actions(_reg, _pos, _side, slot, **_kw):  # noqa: ANN001, ANN003, ANN202
        ids = MOVES if slot == 0 else ["batonpass", "protect"]
        return [
            MoveAction(slot=slot, move_index=i + 1, move_id=m, target=None)
            for i, m in enumerate(ids)
        ]

    monkeypatch.setattr(actions, "slot_actions", slot_actions)


def _batons(rows: list[SideAction], slot: int = 0) -> int:
    """Rows in which ``slot`` uses Baton Pass (the other slot's Baton Pass is no one's hold)."""
    return sum(
        1 for a in rows
        if any(isinstance(s, MoveAction) and s.slot == slot and s.move_id == "batonpass"
               for s in a.slots)
    )


def test_the_gate_opens_only_at_plus_one_in_both_special_stats() -> None:
    assert not baton_gate_open(_mon(0, "e", {}))
    assert not baton_gate_open(_mon(0, "e", {"spa": 1}))
    assert not baton_gate_open(_mon(0, "e", {"spd": 1}))
    assert not baton_gate_open(_mon(0, "e", {"spa": 3, "spd": 0}))
    assert not baton_gate_open(_mon(0, "e", {"spa": 1, "spd": -1}))
    assert not baton_gate_open(_mon(0, "e", {"def": 2, "spe": 2}))
    assert baton_gate_open(_mon(0, "e", {"spa": 1, "spd": 1}))
    assert baton_gate_open(_mon(0, "e", {"spa": 2, "spd": 6}))


def test_side_actions_drops_the_held_baton_rows_and_only_those(menus) -> None:  # noqa: ANN001
    pos = _position({"spa": 1})  # Special Defense still 0: shut
    unheld = side_actions(None, pos, 0)
    assert len(unheld) == 8 and _batons(unheld) == 2  # positive control: rows exist
    with hold_baton(HOLD_BATON_SPECIES, 0):
        held = side_actions(None, pos, 0)
        other_side = side_actions(None, pos, 1)  # the hold is side 0's
    assert len(held) == 6 and _batons(held) == 0
    assert [a.to_choice() for a in held] == [
        a.to_choice() for a in unheld if _batons([a]) == 0
    ]
    # The ally slot's own Baton Pass is not the holder's: its rows stay (3 of the 6).
    assert _batons(held, slot=1) == 3 and _batons(unheld, slot=1) == 4
    assert len(other_side) == 8 and _batons(other_side) == 2
    assert BATON_HOLD[0] is None  # the context restored it


def test_side_actions_keeps_them_once_the_gate_is_open_or_the_holder_is_off(menus) -> None:  # noqa: ANN001
    with hold_baton(HOLD_BATON_SPECIES, 0):
        opened = side_actions(None, _position({"spa": 1, "spd": 1}), 0)
        assert len(opened) == 8 and _batons(opened) == 2
        # The holder fainted, or is another species: nothing to hold.
        assert _batons(side_actions(None, _position(fainted=True), 0)) == 2
    with hold_baton("not-the-species", 0):
        assert _batons(side_actions(None, _position(), 0)) == 2
    with hold_baton(None, 0):  # a None species changes nothing
        assert _batons(side_actions(None, _position(), 0)) == 2
    assert _batons(side_actions(None, _position(), 0)) == 2  # no hold at all


# IKA-444: the drive. With a hold and `BATON_DRIVE`, the holder's slot keeps only Calm Mind
# rows while its gate is shut and only Baton Pass rows once it is open (a teammate on the
# bench); the partner slot stays free; nothing changes without the drive or without a hold.


def _uses(rows: list[SideAction], move: str, slot: int = 0) -> int:
    return sum(
        1 for a in rows
        if any(isinstance(s, MoveAction) and s.slot == slot and s.move_id == move for s in a.slots)
    )


@pytest.fixture
def drive():  # noqa: ANN201
    actions.BATON_DRIVE[0] = True
    try:
        yield
    finally:
        actions.BATON_DRIVE[0] = False


def test_the_drive_uses_calm_mind_while_the_gate_is_shut(menus, drive) -> None:  # noqa: ANN001, ARG001
    pos = _position({"spa": 1})  # shut
    with hold_baton(HOLD_BATON_SPECIES, 0):
        driven = side_actions(None, pos, 0)
        other_side = side_actions(None, pos, 1)
    assert len(driven) == 2 and _uses(driven, "calmmind") == 2
    assert _uses(driven, "batonpass", slot=1) == 1  # the partner still chooses
    assert len(other_side) == 8  # side 1 holds nothing


def test_the_drive_uses_baton_pass_once_the_gate_is_open(menus, drive) -> None:  # noqa: ANN001, ARG001
    pos = _position({"spa": 1, "spd": 1})
    with hold_baton(HOLD_BATON_SPECIES, 0):
        driven = side_actions(None, pos, 0)
    assert len(driven) == 2 and _batons(driven) == 2


def test_the_drive_does_not_pass_to_an_empty_bench(menus, drive) -> None:  # noqa: ANN001, ARG001
    pos = _position({"spa": 1, "spd": 1})
    pos.sides[0].pokemon[2].fainted = True  # nobody left to receive it
    with hold_baton(HOLD_BATON_SPECIES, 0):
        rows = side_actions(None, pos, 0)
    assert len(rows) == 8  # open gate, no bench: the hold's own menu, unsteered


def test_without_the_drive_or_the_hold_nothing_is_steered(menus) -> None:  # noqa: ANN001
    shut, opened = _position({"spa": 1}), _position({"spa": 1, "spd": 1})
    with hold_baton(HOLD_BATON_SPECIES, 0):
        assert len(side_actions(None, shut, 0)) == 6  # the hold alone (negative control)
        assert len(side_actions(None, opened, 0)) == 8
    actions.BATON_DRIVE[0] = True
    try:
        assert len(side_actions(None, shut, 0)) == 8  # drive without a hold does nothing
    finally:
        actions.BATON_DRIVE[0] = False


def test_the_drive_leaves_a_menu_without_calm_mind_alone(menus, drive, monkeypatch) -> None:  # noqa: ANN001, ARG001
    pos = _position({})
    monkeypatch.setattr(actions, "BATON_DRIVE_SETUP", "nastyplot")  # not on the menu
    with hold_baton(HOLD_BATON_SPECIES, 0):
        rows = side_actions(None, pos, 0)
    assert len(rows) == 6 and _batons(rows) == 0  # the hold still applies


# IKA-444: the screen drive. On the held side, Raichu sets a screen the side lacks; the other
# slot and the other side stay as they were; nothing happens without the hold or once both are up.

RAICHU_MOVES = ["fakeout", "reflect", "lightscreen", "risingvoltage"]


@pytest.fixture
def raichu_menus(monkeypatch):  # noqa: ANN001, ANN201
    """Slot 0 chooses among MOVES, slot 1 (Raichu) among RAICHU_MOVES: 16 joint rows."""

    def slot_actions(_reg, _pos, _side, slot, **_kw):  # noqa: ANN001, ANN003, ANN202
        ids = MOVES if slot == 0 else RAICHU_MOVES
        return [
            MoveAction(slot=slot, move_index=i + 1, move_id=m, target=None)
            for i, m in enumerate(ids)
        ]

    monkeypatch.setattr(actions, "slot_actions", slot_actions)


@pytest.fixture
def screens():  # noqa: ANN201
    actions.SCREEN_DRIVE[0] = "raichu"
    try:
        yield
    finally:
        actions.SCREEN_DRIVE[0] = None


def test_the_screen_drive_sets_a_missing_screen(raichu_menus, screens) -> None:  # noqa: ANN001, ARG001
    pos = _position({"spa": 1, "spd": 1})  # gate open: the holder is not steered without BATON_DRIVE
    with hold_baton(HOLD_BATON_SPECIES, 0):
        rows = side_actions(None, pos, 0)
        other_side = side_actions(None, pos, 1)
    assert len(rows) == 8  # 4 holder moves x 2 screens
    assert _uses(rows, "reflect", slot=1) == 4 and _uses(rows, "lightscreen", slot=1) == 4
    assert len(other_side) == 16  # side 1 has no Raichu steered and holds nothing


def test_the_screen_drive_sets_only_the_screen_still_missing(raichu_menus, screens) -> None:  # noqa: ANN001, ARG001
    pos = _position({"spa": 1, "spd": 1})
    pos.sides[0].side_conditions.append(Effect(id="reflect", duration=5))
    with hold_baton(HOLD_BATON_SPECIES, 0):
        rows = side_actions(None, pos, 0)
    assert len(rows) == 4 and _uses(rows, "lightscreen", slot=1) == 4
    pos.sides[0].side_conditions.append(Effect(id="lightscreen", duration=5))
    with hold_baton(HOLD_BATON_SPECIES, 0):
        assert len(side_actions(None, pos, 0)) == 16  # both up: Raichu is free again


def test_the_screen_drive_matches_mega_raichu_and_does_nothing_without_a_hold(raichu_menus, screens) -> None:  # noqa: ANN001, ARG001
    pos = _position({"spa": 1, "spd": 1})
    pos.sides[0].pokemon[1].species = "raichumegax"
    with hold_baton(HOLD_BATON_SPECIES, 0):
        assert len(side_actions(None, pos, 0)) == 8
    assert len(side_actions(None, pos, 0)) == 16  # no hold: nothing steered (negative control)


def test_the_screen_and_baton_drives_together(raichu_menus, screens, drive) -> None:  # noqa: ANN001, ARG001
    pos = _position({"spa": 1})  # gate shut: Calm Mind
    with hold_baton(HOLD_BATON_SPECIES, 0):
        rows = side_actions(None, pos, 0)
    assert len(rows) == 2 and _uses(rows, "calmmind") == 2
    assert _uses(rows, "reflect", slot=1) == 1 and _uses(rows, "lightscreen", slot=1) == 1
