"""IKA-419g: the Baton hold, a flag that is off by default.

What these tests hold:

- **the filter**: `side_actions` leaves out the holder's Baton Pass rows while its Special
  Attack and Special Defense are not both at +1 or more, for the held side only, and not at
  all without the hold (positive and negative controls at both edges of the gate);
- **the seats**: `pool_match_game` hands ``hold_baton`` to the side whose arm has it and whose
  six holds the species, and to no other; line 1's fixed back two reach the picks;
- **the check**: `check_hold_baton` re-reads recorded menus and stops a game whose held node
  still listed or played a Baton Pass, a seat that should not have held, or a tally that
  differs from the menus.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou import actions, poolplay
from pokeuraou.actions import (
    BATON_HOLD,
    MoveAction,
    SideAction,
    baton_gate_open,
    hold_baton,
    side_actions,
)
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import (
    HOLD_BATON_SPECIES,
    PoolArm,
    SolvedSelections,
    check_hold_baton,
    draw_line1,
    pool_match_game,
)
from pokeuraou.position import MoveSlot, Pokemon, Position, Side
from pokeuraou.selfplay import GameRecord

from .test_pool_match import _recorder
from .test_poolplay import _stub, _variants, _write_pool

MOVES = ["batonpass", "calmmind", "protect", "moonblast"]


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


# ---------------------------------------------------------------- the seats


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _arms(pool, *, hold: bool, back=None):  # noqa: ANN001, ANN202
    def arm(name: str, on: bool) -> PoolArm:
        return PoolArm(
            name=name, evaluate=_stub,
            solver=SolvedSelections(pool.reg, pool.teams, _stub, tag=name),
            limit=2, rank_by_leaf=True, rank_fill="refs2", hold_baton=on,
            line1=back is not None, line1_back=back,
        )

    return arm("tested", hold), arm("other", False)


def _play(pool, arms, monkeypatch, games: int = 6):  # noqa: ANN001, ANN202
    species = [s.species for s in pool.teams[0].sets]
    monkeypatch.setattr(poolplay, "HOLD_BATON_SPECIES", species[1])
    monkeypatch.setattr(poolplay, "LINE1_LEADS", (species[0], species[1]))
    monkeypatch.setattr(poolplay, "LINE1_BACK_FALLBACK", (species[2], species[3]))
    monkeypatch.setattr(poolplay, "LINE1_MEGA", species[0])
    flags: list[dict] = []
    fake = _recorder([])

    def recording(reg, rng, own, foe, label, **kwargs):  # noqa: ANN001, ANN003, ANN202
        flags.append({"hold": kwargs["hold_baton"]})
        return fake(reg, rng, own, foe, label, **kwargs)

    monkeypatch.setattr(poolplay, "play_game", recording)
    out = []
    for game in range(games):
        for which in (0, 1):
            _record, sides = pool_match_game(
                pool.reg, pool, arms, seed=7, game_index=game, which=which, hide_bench=True,
                line_only_with=[species[0]],
            )
            out.append((which, sides))
    return species, flags, out


def test_the_hold_reaches_the_tested_arms_seat_only_and_nothing_changes_when_off(  # noqa: ANN201
    pool, monkeypatch  # noqa: ANN001
):
    species, off_flags, off = _play(pool, _arms(pool, hold=False), monkeypatch)
    assert all(f["hold"] == (None, None) for f in off_flags)
    species, on_flags, on = _play(pool, _arms(pool, hold=True), monkeypatch)
    assert len(on) == 12
    for (which, _sides), flags in zip(on, on_flags, strict=True):
        for side in (0, 1):
            assert flags["hold"][side] == (species[1] if side == which else None)
    # Picks are the same with the hold on or off (the hold is not a selection rule).
    for (_w, on_sides), (_w2, off_sides) in zip(on, off, strict=True):
        assert on_sides["picks"] == off_sides["picks"]


def test_line1_fixed_back_reaches_the_picks_and_leaves_the_stream_alone(  # noqa: ANN201
    pool, monkeypatch  # noqa: ANN001
):
    species = [s.species for s in pool.teams[0].sets]
    back = (species[4], species[2])
    _sp, _fl, fixed = _play(pool, _arms(pool, hold=True, back=back), monkeypatch)
    for which, sides in fixed:
        picks = sides["picks"][which]
        six = [s.species for s in pool.teams[sides["teams"][which]].sets]
        assert [six[i] for i in picks[2:]] == list(back)
        assert sides["lines"][which]["rule"] == "fixed"
    # The draw takes one number from the stream in every rule.
    entry = SimpleNamespace(selections=((0, 1, 2, 3),), our_mixture=lambda **_: np.array([1.0]),
                            their_mixture=lambda _i, **_: np.array([1.0]))
    kw = {"epsilon": 0.0, "temperature": 1.0}
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    draw_line1(entry, 0, species, a, fixed_back=back, **kw)
    draw_line1(entry, 0, species, b, **kw)
    assert a.random() == b.random()
    with pytest.raises(ValueError):
        draw_line1(entry, 0, species, np.random.default_rng(3), fixed_back=("nobody", species[2]), **kw)


# --------------------------------------------------------------------- the check


def _game(*, menu: list[str], chosen: str, boosts: dict[str, int], tally: dict | None = None,
          hold: str | None = HOLD_BATON_SPECIES) -> GameRecord:
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="pool")
    record.hold_baton = [hold, None]
    opened = baton_gate_open(_mon(0, "e", boosts))
    record.baton_hold = [
        {"held": 0 if opened else 1, "held_legal": 0 if opened else 1, "open": int(opened),
         "open_menu": int(opened and any(c.startswith("move 1") for c in menu)), "leaks": 0}
        if tally is None else tally,
        None,
    ]
    record.decisions = [SimpleNamespace(
        kind="move", turn=1, position=_position(boosts).to_json(),
        own_actions=menu, foe_actions=[], own_chosen=chosen, foe_chosen="move 2",
    )]
    return record


def _bucket() -> dict:
    return {"games": 0, "held": 0, "held_legal": 0, "open": 0, "open_menu": 0, "baton_open": 0}


def test_check_hold_baton_passes_a_held_game_and_counts(  # noqa: ANN201
):
    bucket = _bucket()
    record = _game(menu=["move 2, move 1", "move 3, move 2"], chosen="move 2, move 1",
                   boosts={"spa": 1})
    check_hold_baton(record, 0, True, ["x", HOLD_BATON_SPECIES], bucket)
    assert bucket["games"] == 1 and bucket["held"] == 1 and bucket["open"] == 0
    # An open node may list and play Baton Pass: counted, not refused.
    record = _game(menu=["move 1, move 1", "move 2, move 1"], chosen="move 1, move 1",
                   boosts={"spa": 1, "spd": 1})
    check_hold_baton(record, 0, True, ["x", HOLD_BATON_SPECIES], bucket)
    assert bucket["open"] == 1 and bucket["open_menu"] == 1 and bucket["baton_open"] == 1


def test_check_hold_baton_refuses_a_leak_a_wrong_seat_and_a_wrong_tally() -> None:
    species = ["x", HOLD_BATON_SPECIES]
    # Negative controls, each breaking the named shape.
    leaked_menu = _game(menu=["move 1, move 1", "move 2, move 1"], chosen="move 2, move 1",
                        boosts={"spa": 1})
    with pytest.raises(AssertionError, match="menu lists a Baton Pass"):
        check_hold_baton(leaked_menu, 0, True, species, _bucket())
    played = _game(menu=["move 2, move 1"], chosen="move 1, move 1", boosts={"spa": 1})
    with pytest.raises(AssertionError, match="Baton Pass was played"):
        check_hold_baton(played, 0, True, species, _bucket())
    # The flag was meant for this seat and the game did not hold.
    unheld = _game(menu=["move 2, move 1"], chosen="move 2, move 1", boosts={}, hold=None)
    with pytest.raises(AssertionError, match="expected True"):
        check_hold_baton(unheld, 0, True, species, _bucket())
    # The seat the flag was not meant for (no such species in its six) must not carry it.
    clean = _game(menu=["move 2, move 1"], chosen="move 2, move 1", boosts={})
    with pytest.raises(AssertionError, match="expected False"):
        check_hold_baton(clean, 0, True, ["x", "y"], _bucket())
    with pytest.raises(AssertionError, match="expected False"):
        check_hold_baton(clean, 0, False, species, _bucket())
    # The game's own tally says something the menus do not.
    wrong = _game(menu=["move 2, move 1"], chosen="move 2, move 1", boosts={},
                  tally={"held": 5, "held_legal": 5, "open": 0, "open_menu": 0, "leaks": 0})
    with pytest.raises(AssertionError, match="held"):
        check_hold_baton(wrong, 0, True, species, _bucket())
    counted = _game(menu=["move 2, move 1"], chosen="move 2, move 1", boosts={},
                    tally={"held": 1, "held_legal": 1, "open": 0, "open_menu": 0, "leaks": 1})
    with pytest.raises(AssertionError, match="leaks"):
        check_hold_baton(counted, 0, True, species, _bucket())
