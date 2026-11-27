"""Games for the tests that read a game record: the Rust generation worker's, on the stand-in server.

Until IKA-463 these tests made their games with `selfplay.play_game`. The Rust worker
(`pokeuraou-damage selfplay-worker`) is the only loop that plays generation's games now, so its
records are the records the readers (the encoder, the game page, `final_position`, ...) meet.

`worker_games` plays ``games`` games of the three-team test pool (`test_poolplay._variants`) on a
stand-in inference server (a leaf of HP share with the fours' bulk apart, the stand-in Q) and
hands back the records the worker wrote, in game order. The worker writes finished games only: a
game the turn cap stops is not written (records/IKA-458.md section 16.2).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pokeuraou.encode import Encoder

from ._served import serve
from .test_poolplay import _variants, _write_pool
from .test_q_rank import _ServedStub
from .test_rust_worker import _config, _leaf_model, _loaded, _play


@contextlib.contextmanager
def served_worker_games(tmp: Path, games: int, *, seed: int = 4581, first: int = 0,
                        **changed: Any) -> Iterator[tuple[Any, list[dict], str]]:  # noqa: ANN401
    """(the pool, the records of games ``first..first+games`` the worker finished, the stand-in
    server's address), the server up until the block ends: for a test that asks the same leaf
    and Q the games were played with (the leaf is the server's arm ``value``, the Q its ``q``)."""
    tmp.mkdir(parents=True, exist_ok=True)
    pool = _loaded(_write_pool(tmp / "p.json", _variants()))
    encoder = Encoder(pool.reg)
    server, address = serve(
        {"value": _leaf_model(encoder)}, arms={"value": ["stub.pt"]},
        q_models={"q": _ServedStub(encoder.vocab.fingerprint())},
    )
    try:
        config = _config(pool, address, tmp, tmp / "store", seed=seed, **changed)
        out = tmp / "games-worker0.jsonl"
        _play(config, address, out, first, games)
        rows = [json.loads(line) for line in out.read_bytes().decode("utf-8").splitlines() if line]
        yield pool, sorted(rows, key=lambda r: r["gameIndex"]), address
    finally:
        server.shutdown()
        server.server_close()


def worker_games(tmp: Path, games: int, **kwargs: Any) -> tuple[Any, list[dict]]:  # noqa: ANN401
    """(the pool, the records of the games the worker finished); the server is stopped."""
    with served_worker_games(tmp, games, **kwargs) as (pool, rows, _address):
        return pool, rows
