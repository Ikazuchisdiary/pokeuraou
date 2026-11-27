"""IKA-340: a game's record names the Q its rank fill ranked by, and the file's sha256.

Before, M-C generation wrote no Q into its records -- only the worker log's echo said
which one it was -- and a match record named the file but not its contents, so a Q
retrained under the same name was indistinguishable. What these tests hold:

- **the worker hands it over**: `selfplay.py` gives the Rust worker's config the Q it
  installed, named and hashed (the config puts ``qModel`` and ``qModelSha256`` into every
  record, after ``pool``: `test_rust_board_worker.py` holds the template's keys);
- **a served Q is hashed by the server**, which holds the file (`qrank.RemoteQ.digests`).

Nothing here needs torch, CUDA or data/: the Q is a stand-in over a small file.
"""

from __future__ import annotations

import hashlib

import pytest

from pokeuraou import qrank
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.pool import load_pool

from .test_poolplay import _variants, _write_pool
from .test_q_rank import _ServedStub
from .test_shipped_rank_fill import FakeQ, _generation_worker

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


def test_the_generation_worker_hands_over_the_q_it_installed(
    monkeypatch, pool_file, tmp_path, q_file  # noqa: ANN001
) -> None:
    called = _generation_worker(monkeypatch, pool_file, tmp_path, ["--rank-leaf"])
    assert called["q_record"] == {"qModel": ["q-mc0.pt"], "qModelSha256": [Q_SHA]}
    assert qrank.file_sha256(q_file) == Q_SHA


def test_a_served_q_is_named_by_the_server_s_sha256(pool) -> None:  # noqa: ANN001
    from ._served import serve

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
