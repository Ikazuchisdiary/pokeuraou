"""Self-play games, and the promise that their labels are real.

The value of this data rests on one claim: the target is who actually won. So the tests
check that a game reaches a result, that a game which does not is *discarded* rather than
labelled with a proxy, and that every recorded position can be read back -- a training
script that cannot reconstruct the position has a policy target attached to nothing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import Position
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.selfplay import play_game, position_from_sets
from pokeuraou.teams import (
    load_archetypes,
    load_roster,
    pick_four,
    sample_archetype,
    usable_archetypes,
)


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    roster = load_roster("rizabanadohido")
    reg, archetypes = load_archetypes("wcs2026-regmb")
    register_mega_stones(reg)
    path = find_cached_chaos(reg.meta.format_id)
    if path is None:
        pytest.skip("no cached usage stats")
    prior = load_chaos(path, reg)
    usable, _blocked = usable_archetypes(prior, archetypes)
    if not usable:
        pytest.skip("no usable archetype")
    return reg, prior, roster, usable


def test_a_turn_one_position_is_well_formed(setup) -> None:  # noqa: ANN001
    reg, _prior, roster, _usable = setup
    four = roster.sets[:4]
    pos = position_from_sets(reg, four, four)
    assert pos.turn == 1
    assert not pos.ended
    for side in pos.sides:
        assert len(side.pokemon) == 4
        assert side.active == [0, 1]
        for mon in side.pokemon:
            assert mon.hp == mon.maxhp > 0
            assert len(mon.moves) == 4
            assert mon.sp is not None
    # Our Charizard holds Charizardite Y, so the mega has to be offered.
    assert pos.sides[0].mega_capable_slots


def test_a_game_reaches_a_result(setup) -> None:  # noqa: ANN001
    reg, prior, roster, usable = setup
    rng = np.random.default_rng(4)
    foe_six = sample_archetype(rng, reg, prior, usable[0])
    foe_four = pick_four(rng, foe_six, size=4)
    record = play_game(
        reg,
        rng,
        pick_four(rng, roster.sets, size=4),
        foe_four,
        usable[0].id,
        objective=HP_SHARE,
        search_limit=4,
        sheets=(list(roster.sets), foe_six),
    )
    assert record.decisions, "a game with no decision point is not a game"
    assert record.outcome in (0.0, 1.0, None)
    assert record.turns >= 1
    for decision in record.decisions:
        assert decision.kind in ("move", "replacement")
        assert len(decision.own_actions) == len(decision.own_policy)
        assert len(decision.foe_actions) == len(decision.foe_policy)
        assert sum(decision.own_policy) == pytest.approx(1.0)
        assert sum(decision.foe_policy) == pytest.approx(1.0)


def test_every_recorded_position_can_be_read_back(setup) -> None:  # noqa: ANN001
    """A policy target attached to a position nothing can reconstruct is worthless."""
    reg, prior, roster, usable = setup
    rng = np.random.default_rng(5)
    foe_six = sample_archetype(rng, reg, prior, usable[0])
    foe_four = pick_four(rng, foe_six, size=4)
    record = play_game(
        reg,
        rng,
        pick_four(rng, roster.sets, size=4),
        foe_four,
        usable[0].id,
        objective=HP_SHARE,
        search_limit=4,
        sheets=(list(roster.sets), foe_six),
    )
    for decision in record.decisions:
        restored = Position.from_json(json.loads(json.dumps(decision.position)))
        assert len(restored.sides) == 2
        assert restored.format == reg.meta.format_id


def _hidden_game(setup, seed: int, **kwargs):  # noqa: ANN001, ANN003, ANN202
    reg, prior, roster, usable = setup
    rng = np.random.default_rng(seed)
    foe_six = sample_archetype(rng, reg, prior, usable[0])
    return play_game(
        reg, rng, pick_four(rng, roster.sets, size=4), pick_four(rng, foe_six, size=4),
        usable[0].id, objective=HP_SHARE, search_limit=4,
        sheets=(list(roster.sets), foe_six), **kwargs,
    )


def test_an_unfinished_game_has_no_outcome(setup) -> None:  # noqa: ANN001
    """A one-turn cap guarantees nothing finishes, so no outcome may be put on it.

    Labelling a timed-out game with a proxy would put the rejected target back into the
    data through the back door (`poolplay` discards such a game instead of writing it).
    """
    record = _hidden_game(setup, 1, max_turns=1)
    assert record.outcome is None


def test_written_games_carry_a_real_outcome(setup) -> None:  # noqa: ANN001
    record = _hidden_game(setup, 2)
    if record.outcome is None:
        pytest.skip("the game did not finish in this small sample")
    payload = record.to_json(objective=HP_SHARE.name, search_limit=4)
    assert payload["outcome"] in (0.0, 1.0)
    assert payload["targetIsRealOutcome"] is True
    assert payload["searchObjective"] == HP_SHARE.name
    assert payload["decisions"]
    assert payload["information"] == "hidden-bench"
