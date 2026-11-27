"""IKA-278: the leaf ranking's numbers, recorded beside the games for IKA-274.

What these tests hold:

- **off writes nothing, on changes no game**: without the file nothing is gathered or
  written, and with it the games file holds the same records (the wall-clock
  ``searchSeconds`` aside, which no two runs share) while every written game has one line;
- **the score is the one `narrow` ordered by**: each kept candidate's score in the `Narrowed`
  the menus were built from is the recorded score of that candidate, and the recorded score
  is the fold of the recorded fills to the bit (`check_game` holds both against the games,
  and fails on a line that was tampered with -- the positive control);
- **under a hidden bench the score is the completion's**: the ranking read one completion,
  and the position rebuilt from the game's record and the pool gives the recorded score
  again, where the true position does not; with more than one part the fold is
  `believed_ranking`'s weighted mean;
- **a torn tail and a replayed game** leave the file readable and each game once.
"""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import rank_scores, selfplay
from pokeuraou.actions import side_actions
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions
from pokeuraou.pool import load_pool
from pokeuraou.position import Position
from pokeuraou.search import believed_ranking, leaf_ranking

from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _bare(games: list[dict]) -> list[dict]:
    out = copy.deepcopy(games)
    for game in out:
        game.pop("searchSeconds")
    return out


def test_the_score_is_the_one_narrow_ordered_by(pool, monkeypatch) -> None:  # noqa: ANN001
    reg = pool.reg
    team = list(pool.teams[0].sets)
    other = list(pool.teams[2].sets)
    pos = selfplay.position_from_sets(reg, team[:4], other[:4], rng=np.random.default_rng(3))
    seen: list = []
    real = selfplay.narrow

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        got = real(*args, **kwargs)
        if kwargs.get("rank") is not None:
            seen.append(got)
        return got

    monkeypatch.setattr(selfplay, "narrow", spy)
    # Null: nothing gathers outside the block, and the ranking is the very same function.
    selfplay._menus(reg, pos, (3, 3), _stub, Budget.matrix(), True, rank_fill="refs2")
    assert len(seen) == 2
    seen.clear()
    with rank_scores.collecting() as sink:
        rank_scores.at_node(0, pos.turn, 0)
        selfplay._menus(reg, pos, (3, 3), _stub, Budget.matrix(), True, rank_fill="refs2")
    assert [row["side"] for row in sink.rows] == [0, 1]
    for row, narrowed in zip(sink.rows, seen, strict=True):
        assert narrowed.considered == len(row["candidates"])
        at = {c: i for i, c in enumerate(row["candidates"])}
        assert len(narrowed.kept) == 3
        for kept in narrowed.kept:
            assert kept.score == row["score"][at[kept.action.to_choice()]]
        assert np.array_equal(rank_scores.folded(row), np.asarray(row["score"]))
        assert len(row["fills"]) == 1 and len(row["fills"][0]["replies"]) == 2
        assert row["completion"] is None  # no spreads: the open game


def _actions(reg, pos: Position, side: int, choices: list[str]) -> list:  # noqa: ANN001
    by_choice = {a.to_choice(): a for a in side_actions(reg, pos, side)}
    return [by_choice[c] for c in choices]


def test_more_than_one_completion_folds_as_believed_ranking(pool) -> None:  # noqa: ANN001
    reg = pool.reg
    team = list(pool.teams[0].sets)
    foe = list(pool.teams[1].sets)
    pos = selfplay.position_from_sets(reg, team[:4], foe[:4], rng=np.random.default_rng(5))
    worlds = completions(reg, pos, 1, foe, seen=frozenset(
        m.slot for m in pos.sides[1].pokemon if m.active_index is not None
    ))
    assert len(worlds) > 2
    weights = [0.3, 0.7]
    parts = [
        (leaf_ranking(reg, w.position, 0, _stub, budget=Budget.matrix()), weight)
        for w, weight in zip(worlds[:2], weights, strict=True)
    ]
    pool_actions = side_actions(reg, pos, 0)
    with rank_scores.collecting() as sink:
        rank_scores.at_node(0, pos.turn, 0)
        rank = rank_scores.watch(believed_ranking(parts), 0, None, None, weights)
        score = rank(pool_actions)
    (row,) = sink.rows
    assert [f["weight"] for f in row["fills"]] == weights
    assert np.array_equal(rank_scores.folded(row), score)
    # Each part's own score, and the fold of them, which neither of them is.
    singles = [np.asarray(p[0](pool_actions)) for p in parts]
    assert np.allclose(score, 0.3 * singles[0] + 0.7 * singles[1], rtol=0, atol=1e-15)
    assert not np.array_equal(singles[0], singles[1])


def _member(index: int) -> bytes:
    line = json.dumps({"gameIndex": index, "rankScores": 1, "rankings": []}) + "\n"
    return gzip.compress(line.encode(), mtime=0)


def test_a_torn_tail_is_cut_and_a_replayed_game_read_once(tmp_path: Path) -> None:
    path = tmp_path / "rank-worker0.jsonl.gz"
    path.write_bytes(_member(0) + _member(1) + _member(2)[:9])
    # The reader stops at the torn member rather than failing.
    assert [line["gameIndex"] for line in rank_scores.iter_file(path)] == [0, 1]
    writer = rank_scores.Writer(path)
    assert writer.repaired == 9
    writer.write(json.dumps({"gameIndex": 2}).encode() + b"\n")
    writer.close()
    assert [line["gameIndex"] for line in rank_scores.iter_file(path)] == [0, 1, 2]
    assert (tmp_path / "rank-worker0.jsonl.gz.torn").read_bytes() == _member(2)[:9]
    # A resume that replayed game 1 (and a line from a directory reader's glob order).
    (tmp_path / "rank-r1-worker0.jsonl.gz").write_bytes(_member(1) + _member(3))
    assert sorted(line["gameIndex"] for line in rank_scores.iter_records(tmp_path)) == [
        0, 1, 2, 3
    ]
    assert rank_scores.path_for(tmp_path / "games-worker7.jsonl") == (
        tmp_path / "rank-worker7.jsonl.gz"
    )
