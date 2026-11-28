"""IKA-458 stage 3.7: the generation worker is `pokeuraou-damage selfplay-worker`.

The Python loop that played generation's games (`poolplay.generate_pool`) was removed, so these
tests hold the Rust worker's games by themselves, on the tests' own inference server (a leaf and
a Q of stand-ins, no GPU):

- **a seed is a set of games**: the same seed and indices give the same records, whether one
  worker plays them in a row or two workers split them; another seed gives other games (the
  control that the comparison can fail);
- **the store changes no game**: a run that reads every pair's solve from the store another run
  wrote solves nothing, leaves the files untouched and writes the same records;
- **the record says solved and carries the pair**: the selection's fields, the pool's teams
  and pair, both seats' mixtures; a pairs subset draws only its pairs and records the index in
  `pool.pairs`;
- **a mirror draws both seats from one mixture** at value 0.5;
- **a flag the Rust worker does not play stops the config** (`rustworker.unsupported`).

Until stage 3.7 these were `test_poolplay.py`'s tests of `generate_pool`, and this file compared
the Rust worker's records with it key for key (records/IKA-458.md sections 16.5 and 18).
"""

from __future__ import annotations

import json
import re
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import qrank, rustworker
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.inference import RemoteValue
from pokeuraou.pool import load_pool

from ._served import encoded_of, hp_share_model, serve
from .test_poolplay import _roster_team, _variants, _write_pool
from .test_q_rank import _ServedStub

LEAF = "value:stub"
GAMES = 3
WIDTH = 4


def _loaded(path: Path):  # noqa: ANN202
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    return _loaded(_write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants()))


@pytest.fixture(scope="module")
def mirror_pool(tmp_path_factory):  # noqa: ANN001, ANN201
    return _loaded(_write_pool(tmp_path_factory.mktemp("mirror") / "p.json", [_roster_team()]))


def _leaf_model(encoder: Encoder):  # noqa: ANN202
    """HP share, and the picked four's max HP apart so that turn 1 is not a flat 0.5 (a selection
    game whose every cell ties has no vertex to agree on)."""
    share = hp_share_model(encoder)
    k = encoder.mon_names.index("maxhp_scaled")

    def model(arrays: dict, rows: int) -> np.ndarray:
        del rows
        encoded = encoded_of(arrays)
        mon = encoded.mon.astype(np.float64)
        bulk = np.where(encoded.mask > 0, mon[..., k], 0.0).sum(axis=-1)
        return share(encoded) + 0.05 * np.tanh(bulk[:, 0] - bulk[:, 1])

    return model


@pytest.fixture(scope="module")
def served(pool):  # noqa: ANN001, ANN201
    encoder = Encoder(pool.reg)
    server, address = serve(
        {"value": _leaf_model(encoder)},
        arms={"value": ["stub.pt"]},
        q_models={"q": _ServedStub(encoder.vocab.fingerprint())},
    )
    yield address
    server.shutdown()
    server.server_close()


def _args(address: str | None, **changed) -> Namespace:  # noqa: ANN003
    base = dict(
        inference=address, rank_leaf=True, rank_fill="q-nocover", seed=4581, max_turns=40,
        explore_epsilon=0.25, explore_temperature=0.5, limit=WIDTH,
    )
    base.update(changed)
    return Namespace(**base)


def _config(pool, address: str, tmp: Path, store: Path, *, pairs=None, **changed) -> Path:  # noqa: ANN001, ANN003
    encoder = Encoder(pool.reg)
    leaf = RemoteValue(address, "value", encoder)
    q = qrank.RemoteQ(address, "q", encoder)
    qrank.install(q)
    try:
        config = rustworker.build_config(
            _args(address, **changed), pool, leaf, LEAF, qrank.record_fields({"": q}), pairs, store,
        )
    finally:
        qrank._INSTALLED.pop("", None)  # noqa: SLF001
        leaf.close()
        q.close()
    path = tmp / "worker-config.json"
    rustworker.write_config(config, path)
    return path


def _play(config: Path, address: str, out: Path, first: int, games: int) -> str:
    """Runs the worker over games ``first..first+games``; its stdout and stderr."""
    done = subprocess.run(  # noqa: S603
        rustworker.command(config, address, out, first_game=first, games=games),
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "handoffs 0" in done.stderr, done.stderr
    return done.stdout + done.stderr


def _records(out: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in out.read_bytes().decode("utf-8").splitlines() if line]
    return {r["gameIndex"]: {k: v for k, v in r.items() if k != "searchSeconds"} for r in rows}


def _solves(log: str) -> tuple[int, int]:
    """(solved here, read from the store) off the worker's selection line."""
    found = re.search(r"selection: (\d+) solves \((\d+) reused, (\d+) read from", log)
    assert found, log
    return int(found.group(1)), int(found.group(3))


def test_a_seed_and_its_indices_are_the_same_games(pool, served, tmp_path) -> None:  # noqa: ANN001
    whole = tmp_path / "whole"
    config = _config(pool, served, whole, whole / "store")
    _play(config, served, whole / "g.jsonl", 0, GAMES)
    split = tmp_path / "split"
    config = _config(pool, served, split, split / "store")
    _play(config, served, split / "w1.jsonl", 1, GAMES - 1)
    _play(config, served, split / "w0.jsonl", 0, 1)
    a = _records(whole / "g.jsonl")
    b = {**_records(split / "w0.jsonl"), **_records(split / "w1.jsonl")}
    assert len(a) == GAMES, "no game finished; the comparison would be vacuous"
    assert a == b
    # The control: another seed is other games.
    other = tmp_path / "other"
    config = _config(pool, served, other, other / "store", seed=4582)
    _play(config, served, other / "g.jsonl", 0, GAMES)
    c = _records(other / "g.jsonl")
    assert all(c[i]["decisions"] != a[i]["decisions"] for i in a if i in c)


def test_the_store_changes_no_game_and_is_read(pool, served, tmp_path) -> None:  # noqa: ANN001
    store = tmp_path / "store"
    config = _config(pool, served, tmp_path / "a", store)
    solved, read = _solves(_play(config, served, tmp_path / "a" / "g.jsonl", 0, GAMES))
    assert solved > 0 and read == 0
    stamps = {p.name: p.stat().st_mtime_ns for p in store.glob("*.json")}
    assert len(stamps) == solved
    config = _config(pool, served, tmp_path / "b", store)
    solved_again, read_again = _solves(_play(config, served, tmp_path / "b" / "g.jsonl", 0, GAMES))
    assert solved_again == 0 and read_again == solved
    assert {p.name: p.stat().st_mtime_ns for p in store.glob("*.json")} == stamps
    assert _records(tmp_path / "a" / "g.jsonl") == _records(tmp_path / "b" / "g.jsonl")


def test_the_record_says_solved_and_carries_the_pair(pool, served, tmp_path) -> None:  # noqa: ANN001
    wanted = [(0, 1), (1, 2)]
    config = _config(pool, served, tmp_path, tmp_path / "store", pairs=wanted)
    _play(config, served, tmp_path / "g.jsonl", 0, 6)
    records = _records(tmp_path / "g.jsonl")
    assert len(records) == 6
    ids = [t.id for t in pool.teams]
    for record in records.values():
        assert record["selectionSource"] == "solved"
        assert record["pool"]["id"] == "test-pool"
        a, b = (ids.index(t) for t in record["pool"]["teams"])
        assert (min(a, b), max(a, b)) in wanted
        assert pool.pairs[record["pool"]["pair"]] == (min(a, b), max(a, b))
        assert record["pool"]["mirror"] == (a == b)
        assert len(record["ownSelectionPolicy"]) == 90
        assert sum(record["foeSelectionMixture"]) == pytest.approx(1.0)
        assert record["qModel"] == ["stub-q.pt"]


def test_mirror_games_draw_both_seats_from_one_mixture(mirror_pool, tmp_path) -> None:  # noqa: ANN001
    encoder = Encoder(mirror_pool.reg)
    server, address = serve(
        {"value": _leaf_model(encoder)}, arms={"value": ["stub.pt"]},
        q_models={"q": _ServedStub(encoder.vocab.fingerprint())},
    )
    try:
        config = _config(mirror_pool, address, tmp_path, tmp_path / "store")
        _play(config, address, tmp_path / "g.jsonl", 0, 2)
    finally:
        server.shutdown()
        server.server_close()
    records = _records(tmp_path / "g.jsonl")
    assert len(records) == 2
    for record in records.values():
        assert record["pool"]["mirror"]
        assert record["selectionValue"] == pytest.approx(0.5, abs=1e-9)
        assert record["ownSelectionMixture"] == record["foeSelectionMixture"]
        assert record["ownSelectionPolicy"] == record["foeSelectionPolicy"]


def test_a_game_the_turn_cap_stops_is_not_written(pool, served, tmp_path) -> None:  # noqa: ANN001
    """The worker writes finished games only: at ``max_turns=1`` no game ends, the worker plays
    the turns and writes nothing, and its last lines count every game as discarded
    (worker.rs, the `outcome` that is None)."""
    config = _config(pool, served, tmp_path, tmp_path / "store", max_turns=1)
    log = _play(config, served, tmp_path / "g.jsonl", 0, GAMES)
    assert _records(tmp_path / "g.jsonl") == {}
    assert f"finished 0, discarded unfinished {GAMES}" in log, log
    # The games were played (the run is not one that wrote nothing for want of a move).
    moves = re.search(r"rust worker: move nodes (\d+)", log)
    assert moves and int(moves.group(1)) >= GAMES, log


class _MirroredQ(_ServedStub):
    """A stand-in Q that is the same game from either seat: ``Q(i, j) = 1 - Q'(j, i)`` where
    ``Q'`` is the matrix with the seats swapped. `_stub_matrix` is not, so in a mirror it gives
    the two seats different menus and their values do not mirror.

    IKA-470: a ranking game, ``Q(i, j) = 0.5 + 0.4 tanh((s_i - s_j) / 2)`` with one strength
    ``s`` per action (a fixed random weighting of the action's and the port's features, the same
    weighting on both seats). The best action is a dominant row, so the Q solve has one pure
    equilibrium and the candidates' scores (their best reply to it) are strictly ordered by ``s``
    with no tie: the menu is the four strongest actions, the same four on both seats, on any
    platform. A game whose solve is mixed (the sines this stood for before) makes the scores of the
    actions on the equilibrium's support equal up to rounding, and when the support is wider than
    the menu, which four a seat keeps is the rounding's choice. CI run 46128754110 (Linux) failed
    on this test with a sum of 1.075; that rounding is the cause we believe, and it was not
    reproduced (the Windows build passed). With the sines no game had the same menu on both seats
    (0 of 16), and the sum held only by chance."""

    def __call__(self, arrays):  # noqa: ANN001, ANN204
        self.calls += 1

        def strength(acts: np.ndarray, feats: np.ndarray) -> np.ndarray:
            a = acts.reshape(len(acts), -1).astype(np.float64)
            f = feats.reshape(len(feats), -1).astype(np.float64)
            weights = np.random.default_rng(470)
            wa, wf = weights.normal(size=a.shape[1]), weights.normal(size=f.shape[1])
            raw = a @ wa + f @ wf
            return 3.0 * raw / max(float(np.abs(raw).max()), 1e-9)

        s0 = strength(arrays["acts0"], arrays["feats0"])
        s1 = strength(arrays["acts1"], arrays["feats1"])
        return 0.5 + 0.4 * np.tanh(0.5 * (s0[:, None] - s1[None, :]))


@pytest.fixture(scope="module")
def mirrored_games(mirror_pool, tmp_path_factory):  # noqa: ANN001, ANN201
    """Six games of one team in both seats, on `_MirroredQ`."""
    tmp = tmp_path_factory.mktemp("mirrored")
    encoder = Encoder(mirror_pool.reg)
    server, address = serve(
        {"value": _leaf_model(encoder)}, arms={"value": ["stub.pt"]},
        q_models={"q": _MirroredQ(encoder.vocab.fingerprint())},
    )
    try:
        config = _config(mirror_pool, address, tmp, tmp / "store")
        _play(config, address, tmp / "g.jsonl", 0, 6)
    finally:
        server.shutdown()
        server.server_close()
    rows = [json.loads(line) for line in (tmp / "g.jsonl").read_bytes().decode().splitlines()]
    assert len(rows) == 6
    return rows


def test_a_mirror_s_two_values_sum_to_one(mirrored_games) -> None:  # noqa: ANN001
    """Where both seats picked the same four and the same leads, side 1's game at turn 1 is side
    0's with the seats swapped, so side 1's value of its own game equals side 0's, and in side 0's
    units the two recorded values sum to one (`foeSearchValue` is ``-foe_value``, playgame.rs). A
    sign or units error breaks the sum; so does writing side 0's value twice, unless it is one
    half, which this hidden game does not give."""
    # IKA-470: the same four, not only the same team and leads. The port's HiGHS may end on
    # another vertex of the selection game than scipy's did, so a game with the same leads and
    # other backs is possible, and it is not a mirror (the other backs make other menus).
    same = [r for r in mirrored_games
            if r["ownTeam"] == r["foeTeam"] and r["leads"][0] == r["leads"][1]
            and sorted(r["ownPick"]) == sorted(r["foePick"])]
    # 5 of the 6 games stay after the filter; the floor keeps the filter from emptying the test.
    assert len(same) >= 3, (
        f"{len(same)} of {len(mirrored_games)} games put the same four and leads in both seats")
    for record in same:
        first = record["decisions"][0]
        assert first["kind"] == "move" and first["turn"] == 1
        # A mirror, checked and not assumed: both seats kept the same four actions (the strength
        # ranking of `_MirroredQ` leaves no tie to choose them by), and so the sum below holds.
        assert first["ownActions"] == first["foeActions"], "the seats' menus are not a mirror"
        assert first["searchValue"] + first["foeSearchValue"] == pytest.approx(1.0, abs=1e-9)
        assert abs(first["searchValue"] - 0.5) > 1e-6, (
            "the value is one half, so this cannot tell side 1's answer from side 0's"
        )


def test_search_seconds_split_in_half(mirrored_games) -> None:  # noqa: ANN001
    """Both seats' search is one node a move, so the worker books half of each move's seconds to
    each seat (worker.rs): ``searchSeconds`` is two equal shares."""
    for record in mirrored_games:
        own, foe = record["searchSeconds"]
        assert own == foe and own > 0.0, record["searchSeconds"]


def test_a_flag_the_rust_worker_does_not_play_is_named() -> None:
    assert rustworker.unsupported(_args("h:1")) == []
    assert rustworker.unsupported(_args("h:1", rank_fill="refs2")) == [
        "--rank-fill refs2 (only a q-nocover fill)"
    ]
    assert rustworker.unsupported(_args("h:1", rank_leaf=False)) == [
        "the damage ranking (no --rank-leaf)"
    ]
    assert rustworker.unsupported(_args(None)) == [
        "a leaf not on an inference server (--inference)"
    ]


def test_profile_stages_reads_the_worker_s_last_line(pool, served, tmp_path) -> None:  # noqa: ANN001
    """`tools/profile_stages.py` reads the Rust worker's own last line from its log (the worker
    writes no timing report): the regular expression is held to what worker.rs prints."""
    from ._harness import load_tool

    profile = load_tool("profile_stages")
    config = _config(pool, served, tmp_path, tmp_path / "store")
    log = _play(config, served, tmp_path / "g.jsonl", 0, 2)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "worker0.log").write_bytes(log.encode("utf-8"))
    found = profile.rust_workers(tmp_path)
    assert found["workers"] == 1 and found["moves"] > 0 and found["play"] > 0, found
    assert {"prepare", "solve", "write", "replacements", "selfswitches", "handoffs"} <= set(found)
    # The control: a log whose line is not the worker's reads as nothing.
    (tmp_path / "logs" / "worker0.log").write_bytes(log.replace("rust worker:", "worker:").encode())
    assert profile.rust_workers(tmp_path) == {}
