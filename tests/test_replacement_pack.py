"""IKA-473: a replacement node's completions packed into fewer requests keep every job's values.

`replacenode::score_jobs` packs consecutive completions while they and their sum stay in one
graph bucket of the inference server. With `POKEURAOU_REPLACEMENT_PACK=0` it sends one request a
completion, as before. On the stand-in server (a per-row leaf), the worker's games are the same
either way, and the packed run sends fewer requests (the control that the packing ran).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pokeuraou.encode import Encoder

from ._served import serve
from .test_poolplay import _variants, _write_pool
from .test_q_rank import _ServedStub
from .test_rust_worker import _config, _leaf_model, _loaded, _play


def _games(tmp: Path, monkeypatch: pytest.MonkeyPatch, pack: str) -> tuple[dict[int, dict], list[int]]:
    """(the records by game, the row count of every leaf request) of 12 games."""
    monkeypatch.setenv("POKEURAOU_REPLACEMENT_PACK", pack)
    tmp.mkdir(parents=True, exist_ok=True)
    pool = _loaded(_write_pool(tmp / "p.json", _variants()))
    encoder = Encoder(pool.reg)
    leaf = _leaf_model(encoder)
    asked: list[int] = []

    def counted(arrays: dict, rows: int) -> np.ndarray:
        asked.append(rows)
        return leaf(arrays, rows)

    server, address = serve(
        {"value": counted}, arms={"value": ["stub.pt"]},
        q_models={"q": _ServedStub(encoder.vocab.fingerprint())},
    )
    try:
        out = tmp / "games-worker0.jsonl"
        _play(_config(pool, address, tmp, tmp / "store", seed=4731), address, out, 0, 12)
    finally:
        server.shutdown()
        server.server_close()
    rows = [json.loads(line) for line in out.read_bytes().decode("utf-8").splitlines() if line]
    return {r["gameIndex"]: {k: v for k, v in r.items() if k != "searchSeconds"} for r in rows}, asked


def test_packed_completions_give_each_job_the_values_it_got_alone(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    packed, packed_asked = _games(tmp_path / "packed", monkeypatch, "1")
    alone, alone_asked = _games(tmp_path / "alone", monkeypatch, "0")
    assert packed, "no game finished"
    assert packed == alone
    # The control: the packing ran. Fewer requests, the same rows.
    assert len(packed_asked) < len(alone_asked), (len(packed_asked), len(alone_asked))
    assert sum(packed_asked) == sum(alone_asked)


def test_an_unknown_packing_value_stops_the_node(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    with pytest.raises(AssertionError, match="POKEURAOU_REPLACEMENT_PACK"):
        _games(tmp_path / "bad", monkeypatch, "yes")
