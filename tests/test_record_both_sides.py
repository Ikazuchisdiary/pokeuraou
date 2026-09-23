"""IKA-127: each decision records what each side's search was conditioned on.

Under a hidden bench the two sides solve different games, and the record used to keep
side 0's value and the true position only. Now it also keeps, per decision, each side's
shown identities and side 1's own value, and per game both lead pairs. These tests hold
the recorded "shown" to the replay a reader would otherwise have had to do, and show the
replay is not vacuous: the carried identity is what the board alone forgets.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from pokeuraou import selfplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import identity
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import play_game, replay_shown
from pokeuraou.teams import load_roster


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    roster = load_roster("rizabanadohido")
    return reg, list(roster.sets)[:6]


def _payload(record) -> dict:  # noqa: ANN001
    # Through JSON, as a reader gets it.
    return json.loads(json.dumps(record.to_json(objective="hp-share", search_limit=4)))


@pytest.fixture(scope="module")
def hidden_game(setup) -> dict:  # noqa: ANN001
    """A few turns of an unscripted hidden-bench game, as its JSON."""
    reg, sheet = setup
    own = sheet[:4]
    foe = [sheet[i] for i in (1, 0, 4, 5)]
    record = play_game(
        reg, np.random.default_rng(3), own, foe, "test",
        search_limit=3, max_turns=4, sheets=(sheet, sheet),
    )
    return _payload(record)


def test_the_recorded_shown_is_the_replayed_shown(hidden_game) -> None:  # noqa: ANN001
    game = hidden_game
    assert game["information"] == "hidden-bench"
    decisions = game["decisions"]
    assert len(decisions) >= 3, "the game ended before it could say anything"
    recorded = [d["shownIdentities"] for d in decisions]
    assert recorded == replay_shown(game)
    # Something was hidden, or the comparison is the open game's in disguise.
    assert any(len(shown[1]) < 4 for shown in recorded), recorded


def test_side_ones_own_value_is_recorded_where_it_solved_one(hidden_game) -> None:  # noqa: ANN001
    for decision in hidden_game["decisions"]:
        if decision["kind"] == "selfswitch":
            assert "foeSearchValue" not in decision
            continue
        value = decision["foeSearchValue"]
        # Side 0's units: a share in [0, 1], not the negated game's [-1, 0].
        assert 0.0 <= value <= 1.0, value
        assert 0.0 <= decision["searchValue"] <= 1.0


def test_in_a_hidden_mirror_side_one_believes_the_mirror_of_side_zero(setup) -> None:  # noqa: ANN001
    """Same four, same sheet, same leads: side 1's game is side 0's with the seats swapped.

    So side 1's win probability as side 1 believes it equals side 0's as side 0 believes
    it, and in side 0's units the two recorded values sum to one. A sign or units error in
    `foeSearchValue` breaks the sum; so does reading side 0's answer twice, unless the
    value is exactly one half -- which the hidden game does not give here.
    """
    reg, sheet = setup
    record = play_game(
        reg, np.random.default_rng(0), sheet[:4], sheet[:4], "test",
        search_limit=2, max_turns=1, sheets=(sheet, sheet),
    )
    first = record.decisions[0]
    assert first.kind == "move" and first.turn == 1
    assert first.foe_search_value is not None
    assert first.search_value + first.foe_search_value == pytest.approx(1.0, abs=1e-6)
    assert abs(first.search_value - 0.5) > 1e-4, (
        "the value is one half, so this cannot tell side 1's answer from side 0's"
    )


def test_the_open_game_shows_the_whole_four_and_records_no_second_value(setup) -> None:  # noqa: ANN001
    reg, sheet = setup
    own = sheet[:4]
    foe = [sheet[i] for i in (1, 0, 4, 5)]
    record = play_game(
        reg, np.random.default_rng(3), own, foe, "test",
        search_limit=3, max_turns=2, open_information=True,
    )
    game = _payload(record)
    whole = [sorted(identity(m) for m in side.pokemon) for side in
             selfplay.position_from_sets(reg, own, foe).sides]
    assert game["decisions"]
    for decision in game["decisions"]:
        assert decision["shownIdentities"] == whole
        assert "foeSearchValue" not in decision
    assert replay_shown(game) == [d["shownIdentities"] for d in game["decisions"]]


def test_leads_are_the_turn_one_pairs_in_active_order(setup, hidden_game) -> None:  # noqa: ANN001
    reg, sheet = setup
    assert hidden_game["leads"] == [
        [identity(m) for m in selfplay.position_from_sets(reg, sheet[:4], sheet[:4])
         .sides[0].pokemon[:2]],
        ["venusaur", "charizard"],
    ]


# ------------------------------------------------ the carry the board alone forgets
#
# The IKA-117 shape: side 1's Charizard goes back for Garchomp at full HP, so at turn 2
# nothing on the board says it was ever out. The record must still say it was shown.

TURN_ONE = {0: "move 4, move 4", 1: "switch 3, move 4"}


def _action(reg, position, side, choice):  # noqa: ANN001, ANN202
    from pokeuraou.actions import side_actions

    for action in side_actions(reg, position, side):
        if action.to_choice() == choice:
            return action
    raise AssertionError(f"side {side} has no {choice!r} here")


def test_a_lead_that_went_back_stays_shown_in_the_record(setup, monkeypatch) -> None:  # noqa: ANN001
    from pokeuraou.actions import side_actions
    from pokeuraou.hidden import seen_identities

    reg, sheet = setup

    def scripted(reg, position, side, *, limit, rank=None):  # noqa: ANN001, ANN202, ARG001
        if position.turn == 1:
            actions = [_action(reg, position, side, TURN_ONE[side])]
        else:
            actions = side_actions(reg, position, side)[:1]
        return types.SimpleNamespace(actions=actions)

    monkeypatch.setattr(selfplay, "narrow", scripted)
    record = play_game(
        reg, np.random.default_rng(0), sheet[:4], sheet[:4], "test",
        sheets=(sheet, sheet), max_turns=1,
    )
    game = _payload(record)
    second = next(d for d in game["decisions"] if d["turn"] == 2)
    assert second["shownIdentities"][1] == ["charizard", "garchomp", "venusaur"]
    assert replay_shown(game) == [d["shownIdentities"] for d in game["decisions"]]

    # Positive control: the board alone has forgotten the Charizard, so a replay that
    # started at turn 2 -- or a record that kept slots -- would say otherwise.
    from pokeuraou.position import Position

    board = Position.from_json(second["position"])
    assert sorted(seen_identities(board, 1)) == ["garchomp", "venusaur"]
    late = dict(game, decisions=[second])
    assert replay_shown(late)[0][1] != second["shownIdentities"][1]
