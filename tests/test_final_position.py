"""IKA-87: a record keeps the position its game stopped at, and why it stopped.

The decisions stop before the last turn is resolved, so a record written before this
could say who won but not who was left standing, or on what HP. These tests hold the
recorded final position to the last decision it follows -- it is one of the branches the
recorded actions resolve to from that decision's position -- and show that a reader of
a record without it (every pool before IKA-87) reads it as it always did.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.actions import side_actions
from pokeuraou.damage import register_mega_stones
from pokeuraou.position import Position
from pokeuraou.regulation import load_regulation
from pokeuraou.selfplay import (
    END_REASONS,
    GameRecord,
    _close_record,
    final_position,
    play_game,
    replay_shown,
)
from pokeuraou.teams import load_roster
from tests._harness import load_tool

from ._port import Budget, resolve_turn

encode_dataset = load_tool("encode_dataset")


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    roster = load_roster("rizabanadohido")
    return reg, list(roster.sets)[:6]


def _payload(record) -> dict:  # noqa: ANN001
    return json.loads(json.dumps(record.to_json(objective="hp-share", search_limit=2)))


def _play(setup, *, seed: int, max_turns: int, hidden: bool) -> dict:  # noqa: ANN001
    reg, sheet = setup
    own = sheet[:4]
    foe = [sheet[i] for i in (1, 0, 4, 5)]
    record = play_game(
        reg, np.random.default_rng(seed), own, foe, "test",
        search_limit=2, max_turns=max_turns,
        sheets=(sheet, sheet) if hidden else None,
        open_information=not hidden,
    )
    return _payload(record)


@pytest.fixture(scope="module")
def finished(setup) -> dict:  # noqa: ANN001
    """A hidden-bench game played to its result."""
    return _play(setup, seed=5, max_turns=40, hidden=True)


@pytest.fixture(scope="module")
def capped(setup) -> dict:  # noqa: ANN001
    """An open game stopped by the cap (`max_turns * 2` decision steps) after a move."""
    return _play(setup, seed=8, max_turns=3, hidden=False)


def _board(pos: dict) -> list:
    """What the final position is for: each Pokemon's HP and faint, and the result."""
    return [
        [(m["species"], m["hp"], m["fainted"]) for m in side["pokemon"]]
        for side in pos["sides"]
    ] + [pos["turn"], pos["ended"], pos["winner"]]


def _branches(reg, decision: dict) -> list[list]:  # noqa: ANN001
    """Every board the last move decision's recorded actions resolve to."""
    pos = Position.from_json(decision["position"])
    menus = [{a.to_choice(): a for a in side_actions(reg, pos, i)} for i in (0, 1)]
    chosen = [menus[0][decision["ownChosen"]], menus[1][decision["foeChosen"]]]
    result = resolve_turn(reg, pos, chosen, budget=Budget.exact())
    return [_board(b.position.to_json()) for b in result.branches]


def _last_move(game: dict) -> dict:
    last = game["decisions"][-1]
    assert last["kind"] == "move", "a self-switch ended the turn; pick another seed"
    return last


def test_a_finished_game_records_who_was_left(finished) -> None:  # noqa: ANN001
    game = finished
    assert game["endReason"] == "wipeout"
    assert game["outcome"] in (0.0, 1.0)
    final = final_position(game)
    assert final is not None and final.ended
    wiped = [i for i in (0, 1) if all(m.fainted and m.hp == 0 for m in final.sides[i].pokemon)]
    assert wiped, "a wipe-out with nobody on zero"
    # Side 0 won exactly when side 1 is the one emptied (or emptied last).
    assert (final.winner == final.sides[0].id) == (game["outcome"] == 1.0)
    if len(wiped) == 1:
        assert wiped[0] == (1 if game["outcome"] == 1.0 else 0)
    # Someone was still standing, or it would be a mutual knockout, which this seed is not.
    assert any(not m.fainted for side in final.sides for m in side.pokemon)
    assert game["turns"] == final.turn


def test_the_final_position_follows_the_last_decision(setup, finished) -> None:  # noqa: ANN001
    reg, _ = setup
    last = _last_move(finished)
    boards = _branches(reg, last)
    assert _board(finished["finalPosition"]) in boards


def test_a_capped_game_records_the_board_the_cap_left(setup, capped) -> None:  # noqa: ANN001
    reg, _ = setup
    game = capped
    assert game["endReason"] == "turn-cap"
    assert game["outcome"] is None
    final = final_position(game)
    assert final is not None and not final.ended
    assert _board(game["finalPosition"]) in _branches(reg, _last_move(game))


def test_the_check_is_not_vacuous(setup, finished) -> None:  # noqa: ANN001
    """The last decision's own board is not among its branches: the check can fail."""
    reg, _ = setup
    last = _last_move(finished)
    assert _board(last["position"]) not in _branches(reg, last)
    # And the board before it (a turn earlier) does not pass for the final one either.
    earlier = [d for d in finished["decisions"] if d["kind"] == "move"][-2]
    assert _board(earlier["position"]) not in _branches(reg, last)


def test_every_reason_is_a_named_one(finished, capped) -> None:  # noqa: ANN001
    assert {finished["endReason"], capped["endReason"]} <= set(END_REASONS)


def test_the_reason_reads_the_board(finished) -> None:  # noqa: ANN001
    """The two reasons no seed here reaches: a draw, and a turn with nothing after it."""
    board = final_position(finished)
    assert board is not None
    reasons = {}
    for name, winner, preset in (
        ("wipeout", board.winner, None),
        ("draw", None, None),
        ("unresolved", board.winner, "unresolved"),
    ):
        record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
        record.end_reason = preset
        pos = Position.from_json(finished["finalPosition"])
        pos.winner = winner
        _close_record(record, pos)
        reasons[name] = record.end_reason
        assert record.final_position == pos.to_json()
    assert reasons == {"wipeout": "wipeout", "draw": "draw", "unresolved": "unresolved"}
    # A record that never closed writes neither key.
    bare = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
    payload = bare.to_json(objective="hp-share", search_limit=2)
    assert "finalPosition" not in payload and "endReason" not in payload


def _old(game: dict) -> dict:
    """The same game as a record from before IKA-87."""
    return {k: v for k, v in game.items() if k not in ("finalPosition", "endReason")}


def test_an_old_record_reads_as_before(finished) -> None:  # noqa: ANN001
    old = _old(finished)
    assert final_position(old) is None
    assert replay_shown(old) == replay_shown(finished)


def test_encode_dataset_reads_both_forms(tmp_path: Path, finished) -> None:  # noqa: ANN001
    args = argparse.Namespace(kinds=None, limit=0, force=True, regulation=None, chunk=4096)
    out = {}
    for name, game in (("new", finished), ("old", _old(finished))):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "games.jsonl").write_bytes(
            (json.dumps(game, ensure_ascii=False) + "\n").encode("utf-8")
        )
        dataset, _meta = encode_dataset.encode_dir(directory, args)
        out[name] = dataset
    assert len(out["new"]) == len(finished["decisions"])
    for field in ("outcome", "turn", "game"):
        a = getattr(out["new"], field, None)
        b = getattr(out["old"], field, None)
        if a is not None:
            assert np.array_equal(np.asarray(a), np.asarray(b)), field
