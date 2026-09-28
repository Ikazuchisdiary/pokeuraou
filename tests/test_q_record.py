"""IKA-340: a game's record names the Q its rank fill ranked by, and the file's sha256.

Before, M-C generation wrote no Q into its records -- only the worker log's echo said
which one it was -- and a match record named the file but not its contents, so a Q
retrained under the same name was indistinguishable. What these tests hold:

- **generation writes the Q**: `generate_pool` puts ``qModel`` (the file names, the form a
  match record already had) and ``qModelSha256`` (each file's sha256, same order) into
  every record, after ``pool``; the rest of the record is unchanged, and without a Q
  nothing is added;
- **the worker hands it over**: `selfplay.py --pool` gives `generate_pool` the Q it
  installed, named and hashed;
- **a served Q is hashed by the server**, which holds the file (`qrank.RemoteQ.digests`);
- **a match record carries the sha256 too** (`pool_match`), beside the ``qModel`` it had.

Nothing here needs torch, CUDA or data/: the Q is a stand-in over a small file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pokeuraou import poolplay, qrank
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.payoff import HP_SHARE
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import SolvedSelections, generate_pool

from .test_poolplay import _fake_play, _stub, _variants, _write_pool
from .test_q_rank import _ServedStub
from .test_shipped_rank_fill import FakeQ, _board_worker, _generation_worker

Q_BYTES = b"IKA-340: a stand-in Q file"
Q_SHA = hashlib.sha256(Q_BYTES).hexdigest()


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture
def q_file(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    """A default Q with known bytes; `FakeQ` never loads it, `file_sha256` reads it."""
    path = tmp_path / "models" / "q-mc0.pt"
    path.parent.mkdir()
    path.write_bytes(Q_BYTES)
    FakeQ.loaded = []
    monkeypatch.setattr(qrank, "default_q", lambda: path)
    monkeypatch.setattr(qrank, "LocalQ", FakeQ)
    return path


@pytest.fixture
def pool_file(tmp_path):  # noqa: ANN001, ANN201
    return str(_write_pool(tmp_path / "pool.json", _variants()))


def _records(pool, out: Path, monkeypatch, q_record):  # noqa: ANN001, ANN202
    monkeypatch.setattr(poolplay, "play_game", _fake_play([]))
    generate_pool(
        pool.reg, pool, games=0, hide_bench=True, seed=340, out=out,
        solver=SolvedSelections(pool.reg, pool.teams, _stub), objective=HP_SHARE,
        indices=list(range(4)), q_record=q_record,
    )
    return [json.loads(line) for line in out.read_bytes().decode("utf-8").splitlines() if line]


def test_generation_records_name_the_q_and_its_sha256(pool, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    fields = {"qModel": ["q-mc0.pt"], "qModelSha256": [Q_SHA]}
    named = _records(pool, tmp_path / "q.jsonl", monkeypatch, fields)
    bare = _records(pool, tmp_path / "bare.jsonl", monkeypatch, None)
    assert len(named) == len(bare) == 4
    for with_q, without in zip(named, bare, strict=True):
        assert with_q["qModel"] == ["q-mc0.pt"] and with_q["qModelSha256"] == [Q_SHA]
        # After the pool block, before the index; the game itself is unchanged.
        keys = list(with_q)
        assert keys.index("pool") < keys.index("qModel") < keys.index("gameIndex")
        assert "qModel" not in without and "qModelSha256" not in without
        assert {k: v for k, v in with_q.items() if not k.startswith("qModel")} == without


def test_the_generation_worker_hands_over_the_q_it_installed(
    monkeypatch, pool_file, tmp_path, q_file  # noqa: ANN001
) -> None:
    called = _generation_worker(monkeypatch, pool_file, tmp_path, ["--rank-leaf"])
    assert called["q_record"] == {"qModel": ["q-mc0.pt"], "qModelSha256": [Q_SHA]}
    assert qrank.file_sha256(q_file) == Q_SHA
    # refs2 ranks by no Q, and names none.
    called = _generation_worker(monkeypatch, pool_file, tmp_path,
                                ["--rank-leaf", "--rank-fill", "refs2"])
    assert called["q_record"] == {}


def test_a_served_q_is_named_by_the_server_s_sha256(pool) -> None:  # noqa: ANN001
    from pokeuraou.inference import serve

    encoder = Encoder(pool.reg)
    served = _ServedStub(encoder.vocab.fingerprint())
    served.digests = [Q_SHA]
    server, address = serve({}, q_models={"q": served})
    try:
        remote = qrank.RemoteQ(address, "q", encoder)
        try:
            assert remote.digests() == [Q_SHA]
            assert qrank.record_fields({"": remote}) == {
                "qModel": ["stub-q.pt"], "qModelSha256": [Q_SHA]
            }
        finally:
            remote.close()
    finally:
        server.shutdown()


def test_record_fields_keep_the_order_of_the_names() -> None:
    class Named:
        def __init__(self, name: str, digest: str) -> None:
            self.name, self.digest = name, digest

        def describe(self) -> list[str]:
            return [self.name]

        def digests(self) -> list[str]:
            return [self.digest]

    got = qrank.record_fields({"b": Named("b.pt", "2"), "": Named("a.pt", "1")})
    assert got == {"qModel": ["a.pt", "b=b.pt"], "qModelSha256": ["1", "2"]}
    assert qrank.record_fields({}) == {}


def test_a_match_record_carries_the_sha256_beside_the_name(
    tmp_path, monkeypatch, q_file  # noqa: ANN001
) -> None:
    _board_worker(tmp_path, monkeypatch, ["--rank-leaf", "--baseline-rank-leaf"])
    lines = (tmp_path / "m" / "games-worker0.jsonl").read_bytes().decode("utf-8").splitlines()
    games = [json.loads(line) for line in lines if line]
    assert games
    for game in games:
        assert game["qModel"] == ["q-mc0.pt"] and game["qModelSha256"] == [Q_SHA]
