"""IKA-419f: the forced Mega Evolution of the user's "line 1" (`play_game`'s ``force_mega``).

What these tests hold: `_forced_mega_choice` plays only a row that Mega Evolves the species,
from the mixture conditioned on those rows, else the first in the menu; it waits while the
species is off the field and says so when the menu has no such row.

The rest of line 1 (the fixed selection `draw_line1`, the seats of `pool_match_game` and the
check `check_line1`) went with the board's Python game loop in IKA-458 stage 3.7.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou.actions import MoveAction, SideAction, SwitchAction
from pokeuraou.selfplay import _forced_mega_choice

# ---------------------------------------------------------------- the forced Mega


def _position(*, mega_used: bool = False, on_field: bool = True, fainted: bool = False):  # noqa: ANN202
    mon = SimpleNamespace(species="raichu", fainted=fainted)
    other = SimpleNamespace(species="espathra", fainted=False)
    side = SimpleNamespace(
        active=[0 if on_field else 2, 1], pokemon=[mon, other, SimpleNamespace(
            species="politoed", fainted=False)], mega_used=mega_used,
    )
    return SimpleNamespace(sides=[side, side])


def _menu(*, with_mega: bool = True) -> list[SideAction]:
    def move(slot: int, mega: bool) -> MoveAction:
        return MoveAction(slot=slot, move_index=1, move_id="thunderbolt", target=1, mega=mega)

    other = MoveAction(slot=1, move_index=1, move_id="protect", target=None)
    rows = [
        SideAction((move(0, False), other)),
        SideAction((move(0, True), other)) if with_mega else SideAction((move(0, False), other)),
        SideAction((SwitchAction(slot=0, party_index=3, species="politoed"), other)),
        SideAction((move(0, True), other)) if with_mega else SideAction((move(0, False), other)),
    ]
    return rows


def test_the_forced_mega_is_drawn_from_the_mixture_conditioned_on_mega_rows() -> None:
    menu = _menu()
    strategy = np.array([0.5, 0.1, 0.1, 0.3])  # rows 1 and 3 Mega: 3 : 1 among them
    counts = {1: 0, 3: 0}
    for seed in range(400):
        index, note = _forced_mega_choice(
            _position(), 0, "raichu", menu, strategy, np.random.default_rng(seed)
        )
        assert note["rule"] == "mixture" and note["mass"] == pytest.approx(0.4)
        counts[index] += 1
    assert 0.18 < counts[1] / 400 < 0.32
    assert menu[1].declares_mega and menu[3].declares_mega


def test_the_forced_mega_takes_the_first_mega_row_when_they_carry_no_probability() -> None:
    index, note = _forced_mega_choice(
        _position(), 0, "raichu", _menu(), np.array([0.6, 0.0, 0.4, 0.0]),
        np.random.default_rng(0),
    )
    assert index == 1 and note["rule"] == "menu-order" and note["mass"] == 0.0


def test_the_forced_mega_waits_and_reports_when_there_is_none() -> None:
    strategy = np.array([0.25, 0.25, 0.25, 0.25])
    rng = np.random.default_rng(0)
    # Raichu is on the bench: not its turn yet, and the caller keeps waiting.
    assert _forced_mega_choice(_position(on_field=False), 0, "raichu", _menu(), strategy, rng) is None
    assert _forced_mega_choice(_position(fainted=True), 0, "raichu", _menu(), strategy, rng) is None
    # On the field but the menu holds no Mega row (used already): the search's own draw stands.
    index, note = _forced_mega_choice(
        _position(mega_used=True), 0, "raichu", _menu(with_mega=False), strategy, rng
    )
    assert index is None and note["rule"] == "unavailable"
    # A Mega row of ANOTHER species' slot does not count: the species is on slot 0 only.
    other_slot = [
        SideAction((MoveAction(slot=1, move_index=1, move_id="x", target=None, mega=True),))
    ]
    index, note = _forced_mega_choice(_position(), 0, "raichu", other_slot, np.array([1.0]), rng)
    assert index is None and note["rule"] == "unavailable"

