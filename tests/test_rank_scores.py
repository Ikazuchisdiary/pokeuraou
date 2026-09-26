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
from pokeuraou.payoff import HP_SHARE
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import SolvedSelections, generate_pool
from pokeuraou.position import Position
from pokeuraou.search import believed_ranking, leaf_ranking

from ._harness import load_tool
from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _generate(pool, out: Path, ranks: Path | None, indices=(0, 1, 2)):  # noqa: ANN001, ANN202
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    stats = generate_pool(
        pool.reg, pool, games=0, hide_bench=True, seed=278, out=out, solver=solver,
        evaluate=_stub, objective=HP_SHARE, search_limit=3, max_turns=8,
        rank_by_leaf=True, indices=list(indices), rank_scores_out=ranks,
    )
    games = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln]
    return stats, games


@pytest.fixture(scope="module")
def played(pool, tmp_path_factory):  # noqa: ANN001, ANN201
    """The same three games, once without the rank file and once with it."""
    base = tmp_path_factory.mktemp("played")
    (base / "off").mkdir()
    (base / "on").mkdir()
    _, off = _generate(pool, base / "off" / "games-worker0.jsonl", None)
    stats, on = _generate(
        pool, base / "on" / "games-worker0.jsonl", base / "on" / "rank-worker0.jsonl.gz"
    )
    lines = list(rank_scores.iter_records(base / "on"))
    return {"base": base, "off": off, "on": on, "stats": stats, "lines": lines}


def _bare(games: list[dict]) -> list[dict]:
    out = copy.deepcopy(games)
    for game in out:
        game.pop("searchSeconds")
    return out


def test_off_writes_nothing_and_on_changes_no_game(played) -> None:  # noqa: ANN001
    assert played["off"], "no game finished; the comparison would be vacuous"
    assert list(played["base"].joinpath("off").iterdir()) == [
        played["base"] / "off" / "games-worker0.jsonl"
    ]
    assert _bare(played["on"]) == _bare(played["off"])
    lines = played["lines"]
    assert [line["gameIndex"] for line in lines] == [g["gameIndex"] for g in played["on"]]
    assert played["stats"]["rank_scores_bytes"] == (
        played["base"] / "on" / "rank-worker0.jsonl.gz"
    ).stat().st_size
    moves = sum(d["kind"] == "move" for g in played["on"] for d in g["decisions"])
    # Both sides of every move decision, from the one construction self-play makes.
    assert sum(len(line["rankings"]) for line in lines) == 2 * moves > 0
    assert all(row["agent"] == 0 for line in lines for row in line["rankings"])


def test_every_ranking_matches_its_decision_and_a_tampered_one_does_not(played) -> None:  # noqa: ANN001
    by_index = {g["gameIndex"]: g for g in played["on"]}
    for line in played["lines"]:
        assert rank_scores.check_game(by_index[line["gameIndex"]], line) == []

    # Positive controls: one value of one fill moved, and one menu's two scores swapped.
    line = copy.deepcopy(played["lines"][0])
    row = line["rankings"][0]
    row["fills"][0]["values"][0][0] += 1e-3
    game = by_index[line["gameIndex"]]
    assert any("fold" in p for p in rank_scores.check_game(game, line))

    line = copy.deepcopy(played["lines"][0])
    for row in line["rankings"]:
        decision = game["decisions"][row["decision"]]
        menu = decision["ownActions"] if row["side"] == 0 else decision["foeActions"]
        at = [row["candidates"].index(c) for c in menu]
        scores = [row["score"][i] for i in at]
        if len(set(scores)) > 1:
            hi, lo = at[scores.index(max(scores))], at[scores.index(min(scores))]
            row["score"][hi], row["score"][lo] = row["score"][lo], row["score"][hi]
            break
    else:
        pytest.fail("no menu with two different scores to swap")
    assert any("order" in p for p in rank_scores.check_game(game, line))


def test_the_check_tool_reads_a_run_directory(played) -> None:  # noqa: ANN001
    tool = load_tool("rank_scores_check")
    found = tool.check_dir(played["base"] / "on")
    assert found["games"] == found["matched"] == len(played["on"]) > 0
    assert found["problems"] == found["missing"] == found["extra"] == 0
    assert found["guessed"] > 0
    # Null: the run without the file is all games without their rankings.
    off = tool.check_dir(played["base"] / "off")
    assert off["missing"] == off["games"] == len(played["off"]) and off["matched"] == 0


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
    selfplay._menus(reg, pos, (3, 3), _stub, Budget.matrix(), True)
    assert len(seen) == 2
    seen.clear()
    with rank_scores.collecting() as sink:
        rank_scores.at_node(0, pos.turn, 0)
        selfplay._menus(reg, pos, (3, 3), _stub, Budget.matrix(), True)
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


def test_a_hidden_ranking_is_its_completion_and_the_record_rebuilds_it(pool, played) -> None:  # noqa: ANN001
    reg = pool.reg
    by_index = {g["gameIndex"]: g for g in played["on"]}
    guessed = moved = 0
    for line in played["lines"]:
        game = by_index[line["gameIndex"]]
        for row in line["rankings"]:
            completion = row["completion"]
            assert completion is not None, "a hidden-bench game ranks from a completion"
            assert len(row["fills"]) == 1 and row["fills"][0]["weight"] == 1.0
            if not completion["slots"]:
                continue
            guessed += 1
            rebuilt = rank_scores.completion_position(reg, pool, game, row)
            truth = Position.from_json(game["decisions"][row["decision"]]["position"])
            pool_actions = _actions(reg, truth, row["side"], row["candidates"])
            again = leaf_ranking(reg, rebuilt, row["side"], _stub, budget=Budget.matrix())
            assert np.array_equal(again(pool_actions), np.asarray(row["score"]))
            # Control: the true bench is another position, and it scores otherwise.
            true_rank = leaf_ranking(reg, truth, row["side"], _stub, budget=Budget.matrix())
            moved += not np.array_equal(true_rank(pool_actions), np.asarray(row["score"]))
    assert guessed > 0, "no ranking read a guessed bench; the check would be vacuous"
    assert moved > 0, "the true bench scored the same everywhere; the rebuild proves nothing"


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


def test_the_cli_passes_the_file_and_says_so(pool, monkeypatch, capsys, tmp_path) -> None:  # noqa: ANN001
    import pokeuraou.pool as pool_module
    from pokeuraou import poolplay

    tool = load_tool("selfplay")
    called: dict = {}

    def fake_generate(reg, pool_arg, **kwargs):  # noqa: ANN001, ANN003, ANN202
        called.update(kwargs)
        return {"games": 0, "finished": 0, "discarded_unfinished": 0, "wins": 0,
                "decisions": 0, "turns": 0, "mirror_games": 0, "mirror_wins": 0,
                "mirror_draws": 0, "path": "x"}

    monkeypatch.setattr(pool_module, "load_pool", lambda name: pool)
    monkeypatch.setattr(poolplay, "generate_pool", fake_generate)
    monkeypatch.setattr(tool, "build_leaf", lambda args, reg: (_stub, "value:stub"))
    out = tmp_path / "games-worker4.jsonl"
    # refs2 named: the shipped fill would load the default Q (IKA-338), not this test's point.
    base = ["selfplay.py", "--pool", "p", "--games", "1", "--out", str(out),
            "--rank-fill", "refs2", "--rank-leaf"]
    monkeypatch.setattr("sys.argv", base)
    tool.main()
    assert called["rank_scores_out"] is None  # null: off unless asked
    assert "rank scores" not in capsys.readouterr().err
    monkeypatch.setattr("sys.argv", [*base, "--record-rank-scores"])
    tool.main()
    assert called["rank_scores_out"] == tmp_path / "rank-worker4.jsonl.gz"
    assert f"rank scores: recording every leaf ranking to {tmp_path / 'rank-worker4.jsonl.gz'}" in (
        capsys.readouterr().err
    )
    # Without a leaf ranking there is nothing to record, and the roster path has no file.
    monkeypatch.setattr("sys.argv", [*base[:-1], "--record-rank-scores"])
    with pytest.raises(SystemExit):
        tool.main()
    monkeypatch.setattr("sys.argv", ["selfplay.py", "--hide-bench", "--rank-leaf",
                                     "--record-rank-scores"])
    with pytest.raises(SystemExit):
        tool.main()
