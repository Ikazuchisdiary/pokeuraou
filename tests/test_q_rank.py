"""IKA-274: the root menu ranked by a learned Q (`pokeuraou.qrank`, rank fills ``q`` / ``q-nocover``).

What these tests hold:

- **the label**: ``q`` keeps the cover, ``q-nocover`` does not, and neither fills a cell
  with the leaf;
- **the ranking**: each candidate is scored by its value against the other side's half of
  the solve of Q over both whole pools, and `_menus` builds the menu from that score --
  with the cover or without, as the label says -- and counts every ranking (`QRANKED`);
  without a Q installed it stops rather than falling back;
- **the served Q**: a Q arm on the inference server answers the same matrix the worker
  would compute from the same arrays, through the op ``q``, and says what it is;
- **it follows its arm into either seat**: in a pool match only the ``q`` arm's rankings
  are counted, in both seats;
- **torch**: a Q loaded here and the same file served answer the same matrix to the bit.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import qhead, qrank, selfplay
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game
from pokeuraou.search import parse_rank_fill, rank_fill_covers

from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


@pytest.fixture()
def pos(pool):  # noqa: ANN001, ANN201
    team = list(pool.teams[0].sets)
    return selfplay.position_from_sets(
        pool.reg, team[:4], team[:4], rng=np.random.default_rng(0)
    )


def _stub_matrix(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """A made-up Q of the request's arrays: reads the actions and the port's features."""
    a0 = arrays["acts0"].reshape(len(arrays["acts0"]), -1).astype(np.float64).sum(1)
    a1 = arrays["acts1"].reshape(len(arrays["acts1"]), -1).astype(np.float64).sum(1)
    f0 = arrays["feats0"].astype(np.float64).sum(1)
    f1 = arrays["feats1"].astype(np.float64).sum(1)
    x = np.sin(0.37 * a0[:, None] + 0.11 * f0[:, None] - 0.23 * a1[None, :] - 0.07 * f1[None, :])
    return 0.5 + 0.4 * x


class _StubQ:
    """`LocalQ`'s interface over `_stub_matrix`."""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.calls = 0

    def describe(self) -> list[str]:
        return ["stub-q.pt"]

    def matrix(self, reg, pos, pools):  # noqa: ANN001, ANN201
        self.calls += 1
        return _stub_matrix(qrank._pool_arrays(reg, self.encoder, pos, pools, True))  # noqa: SLF001


@pytest.fixture()
def stub_q(pool):  # noqa: ANN001, ANN201
    model = _StubQ(Encoder(pool.reg))
    qrank.install(model)
    yield model
    qrank._INSTALLED.clear()  # noqa: SLF001


def _q_scores(reg, pos, side):  # noqa: ANN001, ANN202
    """What the ranking should score: value against the other side's half of the solve."""
    pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
    m = _stub_matrix(qrank._pool_arrays(reg, Encoder(reg), pos, pools, True))  # noqa: SLF001
    e = solve(m)
    return m @ e.col_strategy if side == 0 else -(e.row_strategy @ m)


def test_the_q_labels_rank_by_q_and_say_whether_the_menu_is_covered() -> None:
    assert parse_rank_fill("q") == (0, False)
    assert parse_rank_fill("q-nocover") == (0, False)
    assert rank_fill_covers("q") and not rank_fill_covers("q-nocover")
    assert qrank.is_q("q") and qrank.is_q("q-nocover")
    assert not qrank.is_q("refs2") and not qrank.is_q("refs2-nocover")
    for bad in ("q-fast", "q2", "qnocover", "q-nocover-fast"):
        with pytest.raises(ValueError, match="rank fill"):
            parse_rank_fill(bad)


def test_the_root_menu_is_ranked_against_the_foes_half_of_the_q_solve(
    pool, pos, stub_q  # noqa: ANN001
) -> None:
    reg = pool.reg
    budget = Budget.matrix()
    expect = {}
    for cover in (True, False):
        expect[cover] = tuple(
            narrow(
                reg, pos, side, limit=4, cover=cover,
                rank=lambda p, _s=None, side=side: _q_scores(reg, pos, side),
            ).actions
            for side in (0, 1)
        )
    assert expect[True] != expect[False]  # the cover can be seen in the menus
    # Positive control: the Q's order is not the leaf ranking's.
    leafy = selfplay._menus(reg, pos, (4, 4), _stub, budget, True)  # noqa: SLF001
    assert leafy != expect[True]

    for label, cover in (("q", True), ("q-nocover", False)):
        before = dict(qrank.QRANKED.get(label, {"rankings": 0, "cells": 0}))
        calls = stub_q.calls
        menus = selfplay._menus(reg, pos, (4, 4), _stub, budget, True, rank_fill=label)  # noqa: SLF001
        assert menus == expect[cover]
        # One Q per side's ranking, over both whole pools; no leaf cell.
        assert stub_q.calls - calls == 2
        got = qrank.QRANKED[label]
        assert got["rankings"] - before["rankings"] == 2
        size = len(qhead.legal_pool(reg, pos, 0)) * len(qhead.legal_pool(reg, pos, 1))
        assert got["cells"] - before["cells"] == 2 * size


def test_a_q_label_without_a_q_stops(pool, pos) -> None:  # noqa: ANN001
    qrank._INSTALLED.clear()  # noqa: SLF001
    with pytest.raises(RuntimeError, match="needs a Q"):
        selfplay._menus(  # noqa: SLF001
            pool.reg, pos, (4, 4), _stub, Budget.matrix(), True, rank_fill="q-nocover"
        )


class _ServedStub:
    """`qrank.served_q`'s interface over `_stub_matrix`."""

    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint
        self.properties = True
        self.files = ["stub-q.pt"]
        self.held = 0.0
        self.calls = 0

    def __call__(self, arrays):  # noqa: ANN001, ANN204
        self.calls += 1
        return _stub_matrix(arrays)


def test_a_served_q_answers_what_the_worker_would_compute(pool, pos) -> None:  # noqa: ANN001
    from pokeuraou.inference import serve

    reg = pool.reg
    encoder = Encoder(reg)
    served = _ServedStub(encoder.vocab.fingerprint())
    server, address = serve({}, q_models={"q": served})
    try:
        remote = qrank.RemoteQ(address, "q", encoder)
        try:
            assert remote.describe() == ["stub-q.pt"]
            pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
            got = remote.matrix(reg, pos, pools)
            want = _stub_matrix(qrank._pool_arrays(reg, encoder, pos, pools, True))  # noqa: SLF001
            assert got.shape == (len(pools[0]), len(pools[1]))
            assert np.array_equal(got, want)
            assert served.calls == 1 and remote.calls == 1
            # The features crossed too: without them the made-up Q answers otherwise.
            bare = qrank._pool_arrays(reg, encoder, pos, pools, True)  # noqa: SLF001
            bare["feats0"] = np.zeros_like(bare["feats0"])
            assert not np.array_equal(_stub_matrix(bare), got)
        finally:
            remote.close()
        with pytest.raises(RuntimeError, match="no Q arm"):
            qrank.RemoteQ(address, "other", encoder)
    finally:
        server.shutdown()


def _arm(fill: str, solver) -> PoolArm:  # noqa: ANN001
    return PoolArm(name="arm", evaluate=_stub, solver=solver, limit=2, rank_by_leaf=True,
                   rank_fill=fill)


def _rankings() -> dict[str, int]:
    return {k: v["rankings"] for k, v in qrank.QRANKED.items()}


def test_the_q_ranking_follows_its_arm_into_either_seat(pool, stub_q) -> None:  # noqa: ANN001, ARG001
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    tested, other = _arm("q-nocover", solver), _arm("refs2-nocover", solver)
    for which in (0, 1):
        before = _rankings()
        record, _sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=274, game_index=0, which=which,
            hide_bench=True, max_turns=2,
        )
        assert record.rank_fill[which] == "q-nocover"
        assert record.rank_fill[1 - which] == "refs2-nocover"
        moves = sum(d.kind == "move" for d in record.decisions)
        assert moves > 0
        after = _rankings()
        # The q arm ranks both sides' menus of its own construction, once per move decision.
        assert after["q-nocover"] - before.get("q-nocover", 0) == 2 * moves
        assert after.get("refs2-nocover", 0) == before.get("refs2-nocover", 0)


def test_a_q_loaded_here_and_served_answer_alike(pool, pos, tmp_path) -> None:  # noqa: ANN001
    torch = pytest.importorskip("torch")
    from pokeuraou.inference import serve

    reg = pool.reg
    encoder = Encoder(reg)
    torch.manual_seed(274)
    config = qhead.QConfig(properties=True)
    table = qhead.move_table(reg, encoder.vocab)
    net = qhead.build_net(encoder, config, table)
    path = tmp_path / "q.pt"
    torch.save(
        {
            "state": net.state_dict(), "config": qhead.config_dict(config),
            "vocab_fingerprint": encoder.vocab.fingerprint(),
            "regulation": reg.meta.format_id, "move_table": table,
        },
        path,
    )
    local = qrank.LocalQ(path, encoder)
    pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
    here = local.matrix(reg, pos, pools)
    assert here.shape == (len(pools[0]), len(pools[1]))
    assert np.all((here > 0) & (here < 1))
    server, address = serve({}, q_models=qrank.load_q_arms({"q": path}, "cpu"))
    try:
        remote = qrank.RemoteQ(address, "q", encoder)
        try:
            assert remote.describe() == ["q.pt"]
            assert np.array_equal(remote.matrix(reg, pos, pools), here)
        finally:
            remote.close()
    finally:
        server.shutdown()
