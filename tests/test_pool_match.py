"""IKA-259: an M-C match -- two arms, both seats drawn from one pool.

What these tests hold:

- **the pair**: game ``i`` is the pair and seats `draw_pair` gives ``[seed, i]``, the same
  in both seats of the game; only the ARMS swap;
- **the seat swap**: every per-arm setting (leaf, width, ranking, selection, belief)
  follows the arm into whichever side it sits, and the tested arm is at side ``which``;
- **each arm solves its own selection**: an arm's four and its belief about the opponent
  come from its own solver over its own leaf, never from the other arm's; stores are per
  leaf and a store solved for another leaf stops the run;
- **the null**: identical arms play the same inputs in both seats (and, played out, the
  same game), so a pair scores exactly 0.5;
- **the records pair up** the way `pokeuraou.sprt` reads them, through the worker CLI.

The pool is `test_poolplay`'s three variants of our M-B roster; the leaves are
antisymmetric stubs, and all but one test replace `play_game` with a recorder.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pokeuraou import poolplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import draw_pair, load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.selfplay import GameRecord
from pokeuraou.sprt import PairTail

from ._harness import load_tool
from .test_poolplay import _stub, _variants, _write_pool


def _other_stub(positions):  # noqa: ANN001, ANN202
    """Antisymmetric too, and a different opinion: the bench counts double, not the leads."""
    out = np.empty(len(positions), dtype=np.float64)
    for i, pos in enumerate(positions):
        strength = [
            sum(m.hp * (1 if m.active_index is not None else 2) for m in side.pokemon)
            for side in pos.sides
        ]
        out[i] = 1.0 / (1.0 + np.exp(-(strength[0] - strength[1]) / 40.0))
    return out


class Counted:
    """A leaf that counts the positions it is asked about."""

    def __init__(self, fn):  # noqa: ANN001
        self.fn = fn
        self.positions = 0

    def __call__(self, positions):  # noqa: ANN001, ANN204
        self.positions += len(positions)
        return self.fn(positions)


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _recorder(calls: list[dict]):  # noqa: ANN202
    def fake(reg, rng, own, foe, label, **kwargs):  # noqa: ANN001, ANN003, ANN202
        priors = kwargs["bench_prior"]
        calls.append({
            "own": [s.species for s in own],
            "foe": [s.species for s in foe],
            "label": label,
            "evaluate": kwargs["evaluate"],
            "limits": kwargs["search_limit"],
            "ranks": kwargs["rank_by_leaf"],
            "selection": kwargs["selection"],
            "priors": None if priors is None else [
                None if p is None else (p.species, p.probabilities) for p in priors
            ],
            "one_agent": kwargs["one_agent"],
            "state": rng.bit_generator.state["state"],
        })
        record = GameRecord(own_team=[], foe_team=[], foe_archetype=label)
        record.outcome = float(rng.random() < 0.5)
        record.turns = 1
        return record

    return fake


def _arms(pool, leaf_a, leaf_b, *, b_solves: bool = True):  # noqa: ANN001, ANN202
    a = PoolArm(
        name="arm-a", evaluate=leaf_a,
        solver=SolvedSelections(pool.reg, pool.teams, leaf_a, tag="a"),
        limit=2, rank_by_leaf=True,
    )
    b = PoolArm(
        name="arm-b" if b_solves else "hp-share",
        evaluate=leaf_b if b_solves else None,
        solver=SolvedSelections(pool.reg, pool.teams, leaf_b, tag="b") if b_solves else None,
        limit=3, rank_by_leaf=False,
    )
    return a, b


def _play(pool, arms, monkeypatch, *, seed: int, games: range):  # noqa: ANN001, ANN202
    calls: list[dict] = []
    monkeypatch.setattr(poolplay, "play_game", _recorder(calls))
    sides = []
    for game in games:
        for which in (0, 1):
            _, meta = pool_match_game(
                pool.reg, pool, arms, seed=seed, game_index=game, which=which,
                hide_bench=True,
            )
            sides.append(meta)
    return calls, sides


# ------------------------------------------------------------------------ the pair


def test_both_seats_of_a_game_are_its_seeded_pair(pool, monkeypatch) -> None:  # noqa: ANN001
    arms = _arms(pool, _stub, _other_stub)
    calls, sides = _play(pool, arms, monkeypatch, seed=259, games=range(8))
    teams = []
    for game in range(8):
        k, a, b = draw_pair(np.random.default_rng([259, game]), pool.pairs)
        for which in (0, 1):
            meta = sides[2 * game + which]
            assert (meta["pair"], meta["teams"]) == (k, (a, b))
            call = calls[2 * game + which]
            assert list(call["selection"][0]) == [s.species for s in pool.teams[a].sets]
            assert list(call["selection"][1]) == [s.species for s in pool.teams[b].sets]
        # The game's generator starts where it does in both seats.
        assert calls[2 * game]["state"] == calls[2 * game + 1]["state"]
        teams.append((a, b))
    # Positive control: the games are not all one pair.
    assert len(set(teams)) > 1


# ------------------------------------------------------------------- the seat swap


def test_every_setting_follows_its_arm_into_either_seat(pool, monkeypatch) -> None:  # noqa: ANN001
    arms = _arms(pool, _stub, _other_stub, b_solves=False)
    calls, sides = _play(pool, arms, monkeypatch, seed=3, games=range(4))
    for index, (call, meta) in enumerate(zip(calls, sides, strict=True)):
        which = index % 2
        tested, other = which, 1 - which
        assert call["evaluate"][tested] is _stub and call["evaluate"][other] is None
        assert call["limits"][tested] == 2 and call["limits"][other] == 3
        assert call["ranks"][tested] is True and call["ranks"][other] is False
        assert meta["leaves"][tested] == "arm-a" and meta["leaves"][other] == "hp-share"
        assert meta["selections"][tested] == "solved"
        assert meta["selections"][other] == "uniform"
        # bench_prior[s] is read by side 1 - s: the tested arm holds a belief about the
        # other side, the uniform arm holds none.
        assert call["priors"][other] is not None and call["priors"][tested] is None
        assert meta["beliefs"][tested] == "solved" and meta["beliefs"][other] == "uniform"
        assert call["one_agent"] is False


# --------------------------------------------------- each arm solves its own selection


def test_each_arm_draws_and_believes_from_its_own_solve(pool, monkeypatch) -> None:  # noqa: ANN001
    leaf_a, leaf_b = Counted(_stub), Counted(_other_stub)
    arms = _arms(pool, leaf_a, leaf_b)
    calls, sides = _play(pool, arms, monkeypatch, seed=11, games=range(6))
    assert leaf_a.positions > 0 and leaf_b.positions > 0
    assert arms[0].solver.solves > 0 and arms[1].solver.solves > 0
    differs = 0
    for index, (call, meta) in enumerate(zip(calls, sides, strict=True)):
        which = index % 2
        a, b = meta["teams"]
        for side in (0, 1):
            arm = arms[0] if side == which else arms[1]
            entry = arm.solver.entry(a, b)
            mixture = (
                entry.our_mixture(epsilon=0.0, temperature=1.0)
                if side == 0 else entry.their_mixture(0, epsilon=0.0, temperature=1.0)
            )
            pick = meta["picks"][side]
            assert mixture[list(map(tuple, entry.selections)).index(tuple(pick))] > 0
            # This side's belief about the other side, from this arm's solve.
            about = 1 - side
            believed = call["priors"][about][1]
            own_view = entry.our_mixture(epsilon=0.0, temperature=1.0) if about == 0 \
                else entry.their_mixture(0, epsilon=0.0, temperature=1.0)
            assert np.allclose(believed, own_view)
            other = (arms[1] if arm is arms[0] else arms[0]).solver.entry(a, b)
            other_view = other.our_mixture(epsilon=0.0, temperature=1.0) if about == 0 \
                else other.their_mixture(0, epsilon=0.0, temperature=1.0)
            differs += not np.allclose(believed, other_view)
    # Positive control: the two leaves do disagree about some selection game, so the
    # assertion above could have caught a belief taken from the wrong arm.
    assert differs > 0


def test_an_arm_reading_another_leafs_store_stops(pool, tmp_path) -> None:  # noqa: ANN001
    store = tmp_path / "selection-arm-a"
    SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="sha|value:arm-a").entry(0, 1)
    other = SolvedSelections(
        pool.reg, pool.teams, _other_stub, store=store, tag="sha|value:arm-b"
    )
    with pytest.raises(ValueError, match="another pool or another leaf"):
        other.entry(0, 1)


def test_a_second_writer_of_the_same_pair_keeps_the_first_file(pool, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """Windows refuses the rename while the first worker's file is open (seen in the first
    real run: both seats of a game solve the pair at once). The answer is the same, so the
    second writer drops its copy; with no file there the error still stands."""
    store = tmp_path / "store"
    first = SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="t")
    first.entry(0, 1)
    written = (store / "000-001.json").read_bytes()

    def refuse(src, dst):  # noqa: ANN001, ANN202
        raise PermissionError(5, "Access is denied", str(dst))

    monkeypatch.setattr(poolplay.os, "replace", refuse)
    second = SolvedSelections(pool.reg, pool.teams, _stub, store=None, tag="t")
    second.store = store
    second._write(0, 1, second._solve(0, 1))
    assert (store / "000-001.json").read_bytes() == written
    assert not list(store.glob("*.tmp"))
    # Control: the same refusal with no file in place is not swallowed.
    with pytest.raises(PermissionError):
        second._write(0, 2, second._solve(0, 2))


def test_a_read_during_another_workers_rename_waits_for_it(pool, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """The reading half of the race (IKA-82's first board match lost a worker to it): an
    open refused while the file is being replaced is retried, and a refusal that never
    ends is still an error."""
    store = tmp_path / "store"
    SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="t").entry(0, 1)
    real = poolplay.Path.read_bytes
    refusals = {"left": 3}

    def busy(self):  # noqa: ANN001, ANN202
        if self.name == "000-001.json" and refusals["left"] > 0:
            refusals["left"] -= 1
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(poolplay.Path, "read_bytes", busy)
    reader = SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="t")
    reader.entry(0, 1)
    assert reader.loaded == 1 and reader.solves == 0 and refusals["left"] == 0
    # Control: a file that stays locked is not read as missing or swallowed.
    refusals["left"] = 10_000
    stuck = SolvedSelections(pool.reg, pool.teams, _stub, store=store, tag="t")
    with pytest.raises(PermissionError):
        stuck.entry(0, 1)


# ----------------------------------------------------------------------- the null


def test_identical_arms_see_identical_inputs_in_both_seats(pool, monkeypatch) -> None:  # noqa: ANN001
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    same = PoolArm(name="arm", evaluate=_stub, solver=solver, limit=2, rank_by_leaf=True)
    calls, _ = _play(pool, (same, same), monkeypatch, seed=5, games=range(6))
    for game in range(6):
        first, second = calls[2 * game], calls[2 * game + 1]
        assert first["selection"] == second["selection"]
        assert first["priors"] == second["priors"]
        assert first["state"] == second["state"]
    # Positive control: two different arms are not identical across the seats.
    calls, _ = _play(pool, _arms(pool, _stub, _other_stub), monkeypatch, seed=5,
                     games=range(6))
    assert any(calls[2 * g]["selection"] != calls[2 * g + 1]["selection"] for g in range(6))


def test_identical_arms_play_the_same_game_in_both_seats(pool) -> None:  # noqa: ANN001
    """Played out at width 2 with hp-share leaves: each pair of records is byte-identical."""
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    same = PoolArm(name="arm", evaluate=None, solver=solver, limit=2, rank_by_leaf=False)
    finished = 0
    for game in range(2):
        records = []
        for which in (0, 1):
            record, _ = pool_match_game(
                pool.reg, pool, (same, same), seed=17, game_index=game, which=which,
                hide_bench=True,
            )
            payload = record.to_json(objective="hp-share", search_limit=(2, 2))
            payload.pop("searchSeconds")
            records.append(payload)
        assert records[0] == records[1]
        finished += records[0]["outcome"] is not None
    assert finished, "no game finished; the comparison would be vacuous"


# ------------------------------------------------------------------ the worker CLI


def test_the_worker_writes_records_that_pair_up(pool, tmp_path, monkeypatch, capsys) -> None:  # noqa: ANN001
    tool = load_tool("pool_match")
    leaf = Counted(_stub)
    monkeypatch.setattr(tool, "build_leaves", lambda args, encoder: (leaf, None, "stub", None))
    monkeypatch.setattr(poolplay, "play_game", _recorder([]))
    out = tmp_path / "m" / "games-worker0.jsonl"
    tool.main([
        "--pool", str(_write_pool(tmp_path / "pool.json", _variants())),
        "--value", "unused.pt", "--hide-bench", "--games", "5", "--seed", "9",
        "--games-out", str(out), "--limit", "2",
    ])
    records = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln]
    assert len(records) == 10
    for record in records:
        which = record["seatIndex"]
        source = record["provenance"]
        assert source["kind"] == "pool-match"
        assert source["seat"] == f"stub = side {which}"
        assert source["leaves"][which] == "stub" and source["leaves"][1 - which] == "hp-share"
        assert source["books"][which] == "solved" and source["books"][1 - which] == "uniform"
        assert source["beliefs"][which] == "solved"
        assert source["information"] == ["hidden-bench", "hidden-bench"]
    tail = PairTail(out.parent)
    tail.poll()
    assert [index for index, _ in tail.complete()] == list(range(5))
    err = capsys.readouterr().err
    assert "echo, stub = side 0" in err and "echo, stub = side 1" in err
    assert "selection of tested stub" in err
