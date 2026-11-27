"""IKA-458 stage 3.7: the generation worker is `pokeuraou-damage selfplay-worker`.

What these tests hold, on the tests' own inference server (a leaf and a Q of stand-ins, no GPU):

- **the Rust worker writes the games the Python worker wrote**: the same seed and indices played
  by `generate_pool` (the Python loop over `prepGame` and `playGame`) and by the Rust worker give
  the same records, key for key and in the same key order, apart from the clocks
  (`searchSeconds`), with the selection of every pair solved in each (a fresh store each);
- **the Rust worker's selection solves are the Python ones**: the same pairs, the same tag and
  teams, the same strategies and values to the last few bits (the sums run in another order);
- **the Python worker reads a store the Rust worker wrote** and the other way round
  (`BookEntry.from_json`; `prep.rs` read Python's files since stage 3.5);
- **a flag the Rust worker does not play stops the config** (`rustworker.unsupported`).
"""

from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from pokeuraou import qrank, rustnode, rustworker
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.inference import RemoteValue
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import SolvedSelections, generate_pool
from pokeuraou.selection_book import BookEntry

from ._served import encoded_of, hp_share_model, serve
from .test_poolplay import _variants, _write_pool
from .test_q_rank import _ServedStub

LEAF = "value:stub"
GAMES = 3
WIDTH = 4


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


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


def _args(address: str, **changed) -> Namespace:  # noqa: ANN003
    base = dict(
        inference=address, rank_leaf=True, rank_fill="q-nocover", bench_drop="none",
        deepen="none", depth=1, solve_restricted=False, seed=4581, max_turns=40,
        explore_epsilon=0.25, explore_temperature=0.5, limit=WIDTH,
        uniform_selection=False, no_selection_memo=False, record_rank_scores=False,
    )
    base.update(changed)
    return Namespace(**base)


def _python_games(pool, address: str, out: Path, store: Path) -> list[dict]:  # noqa: ANN001
    encoder = Encoder(pool.reg)
    leaf = RemoteValue(address, "value", encoder)
    q = qrank.RemoteQ(address, "q", encoder)
    qrank.install(q)
    rustnode.hold_positions()
    try:
        solver = SolvedSelections(
            pool.reg, pool.teams, leaf, model=LEAF, store=store, tag=f"{pool.sha256}|{LEAF}",
        )
        generate_pool(
            pool.reg, pool, games=0, seed=4581, out=out, solver=solver, evaluate=leaf,
            search_limit=WIDTH, max_turns=40, leaf=LEAF, rank_by_leaf=True,
            rank_fill="q-nocover", indices=list(range(GAMES)),
            q_record=qrank.record_fields({"": q}),
        )
    finally:
        rustnode.hold_positions(False)
        qrank._INSTALLED.pop("", None)  # noqa: SLF001
        leaf.close()
        q.close()
    return [json.loads(line) for line in out.read_bytes().decode("utf-8").splitlines() if line]


def _rust_games(pool, address: str, tmp: Path, store: Path) -> list[dict]:  # noqa: ANN001
    encoder = Encoder(pool.reg)
    leaf = RemoteValue(address, "value", encoder)
    q = qrank.RemoteQ(address, "q", encoder)
    qrank.install(q)
    try:
        config = rustworker.build_config(
            _args(address), pool, leaf, LEAF, qrank.record_fields({"": q}), None, store,
        )
    finally:
        qrank._INSTALLED.pop("", None)  # noqa: SLF001
        leaf.close()
        q.close()
    path = tmp / "worker-config.json"
    rustworker.write_config(config, path)
    out = tmp / "rust.jsonl"
    done = subprocess.run(  # noqa: S603
        rustworker.command(path, address, out, first_game=0, games=GAMES),
        capture_output=True, text=True, timeout=600, check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "handoffs 0" in done.stderr, done.stderr
    return [json.loads(line) for line in out.read_bytes().decode("utf-8").splitlines() if line]


def _without_clock(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in ("searchSeconds", "engine")}


def test_the_rust_worker_writes_the_python_worker_s_games(pool, served, tmp_path) -> None:  # noqa: ANN001
    python = _python_games(pool, served, tmp_path / "python.jsonl", tmp_path / "store-p")
    rust = _rust_games(pool, served, tmp_path, tmp_path / "store-r")
    assert len(python) == GAMES, "no game finished; the comparison would be vacuous"
    by_index = {r["gameIndex"]: r for r in rust}
    assert sorted(by_index) == [r["gameIndex"] for r in python]
    for want in python:
        got = by_index[want["gameIndex"]]
        assert list(got) == list(want)
        # The decisions, picks and outcome; the selection's mixtures to the last bits (the
        # Rust solve sums its expected losses in another order than numpy's BLAS).
        for key in ("ownSelectionPolicy", "foeSelectionPolicy", "ownSelectionMixture",
                    "foeSelectionMixture"):
            assert np.allclose(got.pop(key), want.pop(key), rtol=0, atol=1e-12), key
        assert abs(got.pop("selectionValue") - want.pop("selectionValue")) < 1e-12
        got_d, want_d = got.pop("decisions"), want.pop("decisions")
        assert [(d["kind"], d["ownChosen"], d["foeChosen"]) for d in got_d] == [
            (d["kind"], d["ownChosen"], d["foeChosen"]) for d in want_d
        ]
        assert _without_clock(got) == _without_clock(want)
    # The stores: the same pairs, the same answers.
    files = sorted(p.name for p in (tmp_path / "store-p").glob("*.json"))
    assert files and files == sorted(p.name for p in (tmp_path / "store-r").glob("*.json"))
    for name in files:
        a = json.loads((tmp_path / "store-p" / name).read_bytes())
        b = json.loads((tmp_path / "store-r" / name).read_bytes())
        assert (a["tag"], a["pair"], a["teams"]) == (b["tag"], b["pair"], b["teams"])
        ea, eb = BookEntry.from_json(a["entry"]), BookEntry.from_json(b["entry"])
        assert ea.selections == eb.selections and ea.key == eb.key and ea.player == eb.player
        assert np.allclose(ea.our_strategy, eb.our_strategy, rtol=0, atol=1e-12)
        assert np.allclose(ea.their_strategies[0], eb.their_strategies[0], rtol=0, atol=1e-12)
        assert np.allclose(ea.our_ev_loss, eb.our_ev_loss, rtol=0, atol=1e-12)
        assert abs(ea.value - eb.value) < 1e-12
        assert a["entry"]["classSets"] == b["entry"]["classSets"]
        assert a["entry"]["notes"] == b["entry"]["notes"]


def test_each_worker_reads_the_other_s_store(pool, served, tmp_path) -> None:  # noqa: ANN001
    """A store the Rust worker solved is read by the Python worker and the other way round: the
    second run of each solves nothing."""
    _rust_games(pool, served, tmp_path, tmp_path / "store")
    solved = sorted(p.name for p in (tmp_path / "store").glob("*.json"))
    python = _python_games(pool, served, tmp_path / "python.jsonl", tmp_path / "store")
    assert len(python) == GAMES
    assert sorted(p.name for p in (tmp_path / "store").glob("*.json")) == solved
    stamps = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "store").glob("*.json")}
    _rust_games(pool, served, tmp_path, tmp_path / "store")
    assert {p.name: p.stat().st_mtime_ns for p in (tmp_path / "store").glob("*.json")} == stamps


def test_a_flag_the_rust_worker_does_not_play_is_named() -> None:
    assert rustworker.unsupported(_args("h:1")) == []
    assert rustworker.unsupported(_args("h:1", deepen="m8")) == ["--deepen m8"]
    assert rustworker.unsupported(_args("h:1", rank_fill="refs2")) == [
        "--rank-fill refs2 (only a q-nocover fill)"
    ]
    assert rustworker.unsupported(_args(None)) == [
        "a leaf not on an inference server (--inference)"
    ]
