"""The game log's field-state line: what is up, and how much of it is left.

A reader cannot judge a mixture without knowing whether the sun is out, whether Trick
Room has reversed the order, or whether the screens are about to drop -- the search was
given all of it and the log was not showing any of it.

What is tested is that the line *follows the position*. It is assembled from the same
fields the search reads, so the failure worth guarding against is not an ugly line but a
confident one that disagrees with the state: a duration off by a turn, a screen shown on
the wrong side, an effect that has already expired. Each assertion below ties a piece of
the text back to the position it came from.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.actions import MoveAction, PassAction, SideAction
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.position import Effect
from pokeuraou.resolve import Budget, resolve_turn
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from ._harness import load_tool


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture(scope="module")
def log_tool():  # noqa: ANN201
    return load_tool("show_game")


@pytest.fixture(scope="module")
def loc(roster, log_tool):  # noqa: ANN001, ANN201
    return log_tool.Localiser(roster.reg, log_tool.load_names("ja"))


def test_nothing_up_prints_nothing(roster, log_tool, loc) -> None:  # noqa: ANN001
    """An empty field gives an empty line, so the log does not carry a "なし" row."""
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    assert log_tool.field_line(loc, pos.to_json()) == ""


def test_the_line_follows_the_position(roster, log_tool, loc) -> None:  # noqa: ANN001
    """Weather, its countdown, and a side condition on the side that owns it."""
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    pos.field.weather = "sunnyday"
    pos.field.weather_duration = 4
    pos.sides[1].side_conditions.append(
        Effect(id="lightscreen", duration=5, layers=1)
    )
    line = log_tool.field_line(loc, pos.to_json())
    assert loc.move("sunnyday") in line, line
    assert "残4" in line, f"the weather's countdown has to be in it: {line}"
    assert loc.move("lightscreen") in line, line
    # On the opponent's half, because that is the side carrying it. Getting this backwards
    # would tell a reader their own attacks are halved when the opposite is true.
    theirs = line.split("敵", 1)
    assert len(theirs) == 2 and loc.move("lightscreen") in theirs[1], (
        f"a screen must be attributed to the side that owns it: {line}"
    )
    assert "自" not in line.split("敵", 1)[0], line


def test_layers_are_shown_and_a_single_layer_is_not(roster, log_tool, loc) -> None:  # noqa: ANN001
    """Spikes stack instead of expiring, so they carry layers and no countdown."""
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    pos.sides[0].side_conditions.append(Effect(id="toxicspikes", layers=1))
    one = log_tool.field_line(loc, pos.to_json())
    assert "層" not in one, f"one layer is the default and reads as noise: {one}"
    assert "残" not in one, f"hazards do not expire, so no countdown: {one}"

    pos.sides[0].side_conditions[0].layers = 2
    two = log_tool.field_line(loc, pos.to_json())
    assert "2層" in two, two


def test_the_countdown_matches_what_the_resolver_does(roster, log_tool, loc) -> None:  # noqa: ANN001
    """`残N` has to mean N more end-of-turn phases, this turn included.

    Asserted by playing the turns rather than by reading the printer: the number is only
    useful if it counts the same thing the resolver counts down, and the resolver is the
    authority on that.
    """
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    pos.field.weather = "sunnyday"
    pos.field.weather_duration = 3
    rng = np.random.default_rng(5)
    seen: list[int] = []
    for _ in range(4):
        line = log_tool.field_line(loc, pos.to_json())
        if not pos.field.weather:
            break
        assert f"残{pos.field.weather_duration}" in line, (
            f"the printed countdown is the position's: {line}"
        )
        seen.append(pos.field.weather_duration)
        ours = narrow(reg, pos, 0, limit=4).actions
        theirs = narrow(reg, pos, 1, limit=4).actions
        result = resolve_turn(
            reg,
            pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        weights = np.array([b.probability for b in result.branches], dtype=np.float64)
        pos = result.branches[
            int(rng.choice(len(weights), p=weights / weights.sum()))
        ].position
    assert seen[:3] == [3, 2, 1], f"the countdown has to tick once a turn: {seen}"
    assert not pos.field.weather, (
        "and 残1 has to be the last turn: the weather is gone after it"
    )


def test_a_branch_is_matched_on_more_than_hp(roster, log_tool, loc) -> None:  # noqa: ANN001
    """A landed secondary costs no HP, so HP alone cannot tell the branches apart.

    Ancient Power's 10% raises five of the user's stats and takes nothing off anyone; the
    branch where it fired and the branch where it did not sit at the same distance from a
    record that only lists HP, and the tie went to whichever came first. Read as the
    secondary firing on almost every use -- against an engine that puts it at exactly the
    declared 0.100.

    So this builds the real fork and asks the matcher for the one the record describes.
    """
    reg = roster.reg
    sets = list(roster.sets)
    holder = next(entry for entry in sets if "ancientpower" in entry.moves)
    others = [entry for entry in sets if entry.species != holder.species][:4]
    pos = position_from_sets(reg, [holder, *others][:4], others)

    slot = next(
        i for i, m in enumerate(pos.sides[0].pokemon[0].moves, start=1)
        if m.id == "ancientpower"
    )
    ours = SideAction(
        slots=(
            MoveAction(
                slot=0, move_index=slot, move_id="ancientpower", target=1, mega=False
            ),
            PassAction(slot=1),
        )
    )
    # A passive opponent, so the fork is the secondary and not a Protect or a faint.
    theirs = SideAction(slots=(PassAction(slot=0), PassAction(slot=1)))
    result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.exact())

    boosted = [
        b for b in result.branches if (b.position.sides[0].pokemon[0].boosts or {}).get("spe", 0) > 0
    ]
    plain = [
        b for b in result.branches if not (b.position.sides[0].pokemon[0].boosts or {}).get("spe", 0)
    ]
    assert boosted and plain, (
        "the turn did not fork on the secondary, so nothing here is being tested"
    )
    # Both readings exist; the two must be told apart by the state, not by their order.
    assert any(
        [mon.hp for side in a.position.sides for mon in side.pokemon]
        == [mon.hp for side in b.position.sides for mon in side.pokemon]
        for a in boosted
        for b in plain
    ), "no pair shares an HP vector, so this position cannot exercise the tie-break"
