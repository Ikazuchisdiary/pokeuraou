"""Self-play games, and the promise that their labels are real.

The value of this data rests on one claim: the target is who actually won. So the tests
check that a written game carries a real result, and that every recorded position can be
read back -- a training script that cannot reconstruct the position has a policy target
attached to nothing. The games are the Rust generation worker's (IKA-463).
"""

from __future__ import annotations

import json

import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.position import Position
from pokeuraou.priors import find_cached_chaos, load_chaos
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import (
    load_archetypes,
    load_roster,
    usable_archetypes,
)

from ._worker_games import worker_games


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


# ----------------------------------------------------------- the generation worker's games
#
# IKA-463: these were games of `selfplay.play_game`; generation's games are the Rust worker's
# now (`tests/_worker_games.py`). The worker writes a finished game only, so a game the turn cap
# stops never reaches a record (the old test of that case read `play_game`'s record before it
# was discarded).


@pytest.fixture(scope="module")
def worker(tmp_path_factory):  # noqa: ANN001, ANN201
    return worker_games(tmp_path_factory.mktemp("worker"), 3)


def test_a_game_reaches_a_result(worker) -> None:  # noqa: ANN001
    _pool, games = worker
    assert len(games) == 3, "the worker wrote fewer games than it was given"
    for game in games:
        assert game["decisions"], "a game with no decision point is not a game"
        assert game["outcome"] in (0.0, 1.0)
        assert game["targetIsRealOutcome"] is True
        assert game["information"] == "hidden-bench"
        assert game["turns"] >= 1
        for decision in game["decisions"]:
            assert decision["kind"] in ("move", "replacement", "selfswitch")
            if decision["kind"] == "selfswitch":
                continue
            assert len(decision["ownActions"]) == len(decision["ownPolicy"])
            assert len(decision["foeActions"]) == len(decision["foePolicy"])
            assert sum(decision["ownPolicy"]) == pytest.approx(1.0)
            assert sum(decision["foePolicy"]) == pytest.approx(1.0)


def test_every_recorded_position_can_be_read_back(worker) -> None:  # noqa: ANN001
    """A policy target attached to a position nothing can reconstruct is worthless."""
    pool, games = worker
    for game in games:
        for decision in game["decisions"]:
            restored = Position.from_json(json.loads(json.dumps(decision["position"])))
            assert len(restored.sides) == 2
            assert restored.format == pool.reg.meta.format_id
