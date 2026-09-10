"""Reading a scenario, and refusing to guess what it does not say.

Every one of these is a refusal test. The scenario file is the only place a user can
introduce a wrong premise, and a premise that gets silently repaired -- a missing nature
defaulted, an absolute HP accepted for a Pokemon whose max HP is unknown, a spread guessed
for the opponent -- produces output that looks right and is not. So the parser is strict
and the tests pin what it rejects.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.setup import ScenarioError, parse_scenario, with_spreads

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn1.json"


@pytest.fixture(scope="module")
def raw() -> dict:
    if not EXAMPLE.exists():
        pytest.skip(f"{EXAMPLE.name} missing")
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_our_spreads_are_known_and_the_opponents_are_not(raw: dict) -> None:
    scenario = parse_scenario(copy.deepcopy(raw))
    for mon in scenario.position.sides[0].pokemon:
        assert mon.sp is not None, "our own spread is known and has to be used"
        assert mon.maxhp > 0
    for mon in scenario.position.sides[1].pokemon:
        assert mon.sp is None, "the opponent's spread is the hidden state"
    assert len(scenario.hidden) == 4
    # The nature is revealed by the team sheet, so it is recorded even for hidden spreads.
    assert all(nature for nature in scenario.hidden.values())


def test_a_position_with_a_hidden_spread_cannot_be_resolved(raw: dict) -> None:
    """The refusal that keeps a guessed spread out of the pipeline entirely."""
    from pokeuraou.view import battler

    scenario = parse_scenario(copy.deepcopy(raw))
    mon = scenario.position.sides[1].pokemon[scenario.position.sides[1].active[0]]
    with pytest.raises(ValueError, match="belief layer"):
        battler(scenario.reg, mon)


def test_our_own_side_must_supply_a_spread(raw: dict) -> None:
    data = copy.deepcopy(raw)
    del data["sides"][0]["team"][0]["sp"]
    with pytest.raises(ScenarioError, match="own spread is known"):
        parse_scenario(data)


def test_an_absolute_hp_is_refused_for_a_hidden_pokemon(raw: dict) -> None:
    """An absolute HP would smuggle in the max HP the spread was meant to hide."""
    data = copy.deepcopy(raw)
    data["sides"][1]["team"][0]["hp"] = 155
    with pytest.raises(ScenarioError, match="looks absolute"):
        parse_scenario(data)


def test_a_percentage_hp_is_accepted(raw: dict) -> None:
    data = copy.deepcopy(raw)
    data["sides"][1]["team"][0]["hp"] = "62%"
    scenario = parse_scenario(data)
    key = (1, 0)
    assert scenario.hp_display[key] == (62, None)


def test_a_missing_nature_is_refused(raw: dict) -> None:
    data = copy.deepcopy(raw)
    del data["sides"][1]["team"][0]["nature"]
    with pytest.raises(ScenarioError, match="no nature"):
        parse_scenario(data)


def test_an_illegal_move_is_refused(raw: dict) -> None:
    """The regulation config is the authority; the code holds no move pool of its own."""
    data = copy.deepcopy(raw)
    data["sides"][0]["team"][0]["moves"] = ["Terastal Blast"]
    with pytest.raises(ScenarioError, match="not legal"):
        parse_scenario(data)


def test_a_team_that_is_not_four_is_refused(raw: dict) -> None:
    data = copy.deepcopy(raw)
    data["sides"][0]["team"] = data["sides"][0]["team"][:3]
    with pytest.raises(ScenarioError, match="4 Pokemon|3 Pokemon"):
        parse_scenario(data)


def test_exactly_two_actives_per_side(raw: dict) -> None:
    data = copy.deepcopy(raw)
    data["sides"][0]["team"][2]["active"] = True
    with pytest.raises(ScenarioError, match="active"):
        parse_scenario(data)


def test_with_spreads_needs_every_hidden_pokemon(raw: dict) -> None:
    """A half-filled belief would resolve some actions and raise on others."""
    scenario = parse_scenario(copy.deepcopy(raw))
    partial = {next(iter(scenario.hidden)): np.zeros(6, dtype=np.int64)}
    with pytest.raises(ScenarioError, match="no spread assigned"):
        with_spreads(scenario, partial)


def test_with_spreads_picks_an_hp_the_game_could_have_displayed(raw: dict) -> None:
    """Not `round(maxhp * 0.5)`: that value often displays as 49% or 51%.

    Champions floors, so the HP consistent with a 50% bar sits at the bottom of the band.
    A point estimate taken from the fraction is off by a point, which is the margin a KO
    turns on.
    """
    from pokeuraou.hpdisplay import displayed_percent

    data = copy.deepcopy(raw)
    data["sides"][1]["team"][0]["hp"] = "50%"
    scenario = parse_scenario(data)
    assignment = {
        key: np.array([32, 0, 0, 0, 0, 0], dtype=np.int64) for key in scenario.hidden
    }
    pos = with_spreads(scenario, assignment)
    mon = pos.sides[1].pokemon[0]
    assert mon.sp == {"hp": 32, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
    assert mon.maxhp > 0
    assert displayed_percent(mon.hp, mon.maxhp) == 50

    # And an explicit choice from the band is honoured.
    band = scenario.band_for((1, 0), mon.maxhp)
    pos = with_spreads(scenario, assignment, {(1, 0): band.high})
    assert pos.sides[1].pokemon[0].hp == band.high


def test_our_own_percentage_must_pin_down_one_value(raw: dict) -> None:
    """We know our own HP, so a percentage that leaves a choice is a missing number."""
    data = copy.deepcopy(raw)
    data["sides"][0]["team"][0]["hp"] = "50%"
    try:
        parse_scenario(data)
    except ScenarioError as exc:
        assert "give the exact HP" in str(exc)


def test_the_bar_colour_is_read(raw: dict) -> None:
    """At 20% and 50% the bar colour is free information the game hands over."""
    data = copy.deepcopy(raw)
    data["sides"][1]["team"][0]["hp"] = "50%g"
    scenario = parse_scenario(data)
    assert scenario.hp_display[(1, 0)] == (50, "g")
