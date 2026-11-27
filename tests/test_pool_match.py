"""IKA-259: an M-C board's selection stores and its pair filter.

What these tests hold:

- **each arm solves its own selection**: stores are per leaf and a store solved for another
  leaf stops the run; two workers writing or reading one pair at once settle on one file;
- `--pairs-focus` keeps a focus team against the field only.

The board's games are the Rust worker's since IKA-458 stage 3.7 (`test_rust_board_worker.py`
holds the seats, the null of identical arms and the records that pair up).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import poolplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import SolvedSelections
from pokeuraou.regulation import load_regulation
from pokeuraou.standin import RowStub

from ._harness import load_tool
from .test_poolplay import _stub, _variants, _write_pool


def _bench_double(rows):  # noqa: ANN001, ANN202
    strength = (rows.hp * np.where(rows.active, 1, 2)).sum(axis=-1)
    return 1.0 / (1.0 + np.exp(-(strength[:, 0] - strength[:, 1]) / 40.0))


#: Antisymmetric too, and a different opinion: the bench counts double, not the leads (served:
#: the port scores the selection game on the server, IKA-463).
_other_stub = RowStub(load_regulation("gen9championsvgc2026regmb"), _bench_double)


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded



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


def test_focus_pairs_keep_a_focus_team_against_the_field_only() -> None:
    """IKA-443: candidates (4, 5) against the pool (0-3); candidate against candidate is out."""
    pool_match = load_tool("pool_match")

    pairs = [(a, b) for a in range(6) for b in range(a, 6)]
    got = pool_match.focus_pairs(pairs, {4, 5}, {0, 1, 2, 3})
    assert got == [(0, 4), (0, 5), (1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
    every = pool_match.focus_pairs(pairs, {0}, None)
    assert every == [(0, b) for b in range(6)]  # the mirror too; --pairs-no-mirror drops it
