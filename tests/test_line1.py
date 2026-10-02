"""IKA-419f: the user's "line 1" as a pair of flags that are off by default.

What these tests hold:

- **the selection**: `draw_line1` leads Raichu then Espathra, draws the back two from the
  equilibrium conditioned on the rows that bring both, and falls back to the fixed pair when
  those rows carry no probability (negative controls: the conditioned frequencies, and the
  fallback only on zero mass);
- **the forcing**: `_forced_mega_choice` plays only a row that Mega Evolves the species, from
  the mixture conditioned on those rows, else the first in the menu; it waits while the
  species is off the field and says so when the menu has no such row;
- **the seats**: `pool_match_game` applies line 1 to the side its arm and ``line_only_with``
  name and to no other, and with the flags off the picks and the arguments of `play_game`
  are the ones without them;
- **the check**: `check_line1` stops a game whose seat got what it should not have.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from pokeuraou import poolplay
from pokeuraou.actions import MoveAction, SideAction, SwitchAction
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, check_line1, draw_line1, pool_match_game
from pokeuraou.selfplay import GameRecord, _forced_mega_choice

from .test_pool_match import _recorder
from .test_poolplay import _stub, _variants, _write_pool

SIX = ["raichu", "espathra", "archaludon", "whimsicott", "politoed", "charizard"]


def _entry(weights: dict[tuple[int, ...], float]) -> SimpleNamespace:
    """A solve over every ordered four of six, with the given weight on some rows (both
    sides read the same weights)."""
    selections = tuple(itertools.permutations(range(6), 4))
    mixture = np.array([weights.get(row, 0.0) for row in selections], dtype=np.float64)
    return SimpleNamespace(
        selections=selections,
        our_mixture=lambda **_: mixture,
        their_mixture=lambda _index, **_: mixture,
    )


KW = {"epsilon": 0.0, "temperature": 1.0}


def test_line1_leads_and_draws_the_back_from_the_conditioned_equilibrium() -> None:
    both_a = (0, 2, 1, 3)       # Raichu, Archaludon, Espathra, Whimsicott: back 2, 3
    both_b = (4, 1, 0, 5)       # back Politoed, Charizard
    other = (0, 2, 3, 4)        # no Espathra
    entry = _entry({both_a: 0.3, both_b: 0.1, other: 0.6})
    counts = {(2, 3): 0, (4, 5): 0}
    for seed in range(400):
        picks, note = draw_line1(entry, 0, SIX, np.random.default_rng(seed), **KW)
        assert picks[:2] == (0, 1)
        assert note["rule"] == "equilibrium"
        assert note["mass"] == pytest.approx(0.4)
        counts[picks[2:]] += 1
    # 0.3 : 0.1 inside the conditioned rows, so 3 : 1 and not the raw 0.3 / 0.1 of the whole
    assert sum(counts.values()) == 400
    assert 0.68 < counts[(2, 3)] / 400 < 0.82
    # The same draw on side 1 reads the other mixture function.
    picks, note = draw_line1(entry, 1, SIX, np.random.default_rng(0), **KW)
    assert picks[:2] == (0, 1) and note["rule"] == "equilibrium"


def test_line1_falls_back_to_the_fixed_back_pair_only_when_the_mass_is_zero() -> None:
    no_both = _entry({(0, 2, 3, 4): 1.0})
    picks, note = draw_line1(no_both, 0, SIX, np.random.default_rng(1), **KW)
    assert picks == (0, 1, 2, 4)  # Archaludon, Politoed
    assert note["rule"] == "fallback" and note["reason"] == "zero mass"
    # A row that holds both but brings a back member outside the candidates cannot be used
    # either; the candidates here are all four others, so the same row counts when it is there.
    picks, note = draw_line1(_entry({(0, 1, 5, 3): 1.0}), 0, SIX, np.random.default_rng(1), **KW)
    assert picks == (0, 1, 5, 3) and note["rule"] == "equilibrium"
    picks, note = draw_line1(None, 0, SIX, np.random.default_rng(1), **KW)
    assert picks == (0, 1, 2, 4) and note["reason"] == "no solve"


def test_line1_takes_the_same_draw_from_the_stream_whichever_rule_applies() -> None:
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    draw_line1(_entry({(0, 1, 2, 3): 1.0}), 0, SIX, rng_a, **KW)
    draw_line1(_entry({(0, 2, 3, 4): 1.0}), 0, SIX, rng_b, **KW)
    assert rng_a.random() == rng_b.random()


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


# --------------------------------------------------------------------- the seats


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _arms(pool, *, line1: bool):  # noqa: ANN001, ANN202
    def arm(name: str, flag: bool) -> PoolArm:
        return PoolArm(
            name=name, evaluate=_stub,
            solver=SolvedSelections(pool.reg, pool.teams, _stub, tag=name),
            limit=2, rank_by_leaf=True, rank_fill="refs2", line1=flag,
        )

    return arm("tested", line1), arm("other", False)


def _play(pool, arms, monkeypatch, *, only_with, games: int = 6):  # noqa: ANN001, ANN202
    species = [s.species for s in pool.teams[0].sets]
    monkeypatch.setattr(poolplay, "LINE1_LEADS", (species[0], species[1]))
    monkeypatch.setattr(poolplay, "LINE1_BACK_CANDIDATES", tuple(species[2:]))
    monkeypatch.setattr(poolplay, "LINE1_BACK_FALLBACK", (species[2], species[3]))
    monkeypatch.setattr(poolplay, "LINE1_MEGA", species[0])
    flags: list[dict] = []
    fake = _recorder([])

    def recording(reg, rng, own, foe, label, **kwargs):  # noqa: ANN001, ANN003, ANN202
        flags.append({"force_mega": kwargs["force_mega"]})
        return fake(reg, rng, own, foe, label, **kwargs)

    monkeypatch.setattr(poolplay, "play_game", recording)
    out = []
    for game in range(games):
        for which in (0, 1):
            _record, sides = pool_match_game(
                pool.reg, pool, arms, seed=7, game_index=game, which=which,
                hide_bench=True, line_only_with=only_with,
            )
            out.append((which, sides))
    return species, flags, out


def test_line1_reaches_the_tested_arms_seat_only_and_nothing_changes_when_off(  # noqa: ANN201
    pool, monkeypatch  # noqa: ANN001
):
    species, forced_off, off = _play(pool, _arms(pool, line1=False), monkeypatch, only_with=None)
    assert all(f["force_mega"] == (None, None) for f in forced_off)
    assert all(sides["lines"] == (None, None) for _which, sides in off)

    species, forced_on, on = _play(pool, _arms(pool, line1=True), monkeypatch, only_with=None)
    assert len(on) == 12
    for (which, sides), flags in zip(on, forced_on, strict=True):
        # The tested arm sits at side `which`; only that side got line 1.
        for side in (0, 1):
            assert (sides["lines"][side] is not None) == (side == which)
            assert flags["force_mega"][side] == (species[0] if side == which else None)
        six = [s.species for s in pool.teams[sides["teams"][which]].sets]
        assert tuple(sides["picks"][which][:2]) == (six.index(species[0]), six.index(species[1]))
    # The side without it draws what it drew with the flags off, game for game.
    for (which, on_sides), (_w, off_sides) in zip(on, off, strict=True):
        assert on_sides["picks"][1 - which] == off_sides["picks"][1 - which]
    # Positive control: line 1 did change the tested side's four somewhere.
    assert any(
        on_sides["picks"][which] != off_sides["picks"][which]
        for (which, on_sides), (_w, off_sides) in zip(on, off, strict=True)
    )


def test_line_only_with_keeps_line1_off_a_side_without_the_species(  # noqa: ANN201
    pool, monkeypatch  # noqa: ANN001
):
    _, _flags, none_named = _play(
        pool, _arms(pool, line1=True), monkeypatch, only_with=["not-a-species"]
    )
    assert all(sides["lines"] == (None, None) for _which, sides in none_named)
    species = [s.species for s in pool.teams[0].sets]
    _, _flags, named = _play(pool, _arms(pool, line1=True), monkeypatch, only_with=[species[0]])
    assert all(sides["lines"][which] is not None for which, sides in named)


# --------------------------------------------------------------------- the check


def _played(*, force: str | None, forced: dict | None, chosen: str = "move 1 1 mega, move 1"):  # noqa: ANN202
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="pool")
    record.force_mega = [force, None]
    record.forced_mega = [forced, None]
    record.decisions = [SimpleNamespace(kind="move", turn=1, own_chosen=chosen, foe_chosen="")]
    sides = {"lines": ({"rule": "equilibrium", "mass": 0.5}, None), "picks": ((0, 1, 2, 3), ())}
    return record, sides


def test_check_line1_passes_a_forced_game_and_stops_a_wrong_seat() -> None:
    bucket = {"games": 0, "selection": {}, "forced": {}}
    record, sides = _played(force="raichu", forced={"turn": 1, "rule": "mixture", "mass": 0.5})
    check_line1(record, sides, 0, True, None, SIX, bucket)
    assert bucket["games"] == 1 and bucket["forced"] == {"mixture": 1}
    # The flag was meant for this side and the game's record says nothing was forced.
    broken, sides = _played(force=None, forced=None)
    with pytest.raises(AssertionError):
        check_line1(broken, sides, 0, True, None, SIX, bucket)
    # The Mega was not declared on the forcing turn.
    broken, sides = _played(
        force="raichu", forced={"turn": 1, "rule": "mixture", "mass": 0.5}, chosen="move 1 1"
    )
    with pytest.raises(AssertionError):
        check_line1(broken, sides, 0, True, None, SIX, bucket)
    # The seat the flag was not meant for (species not named) must not carry it.
    with pytest.raises(AssertionError):
        check_line1(record, sides, 0, True, ["charizard-not-here"], SIX, bucket)
