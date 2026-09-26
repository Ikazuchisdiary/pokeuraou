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

Stage 3:

- **one round trip**: both sides' requests of a menu go together (`qrank.prefetch`, op
  ``q_many``), and the games are the games of one request per ranking;
- **a Q per arm**: ``q-nocover.NAME`` ranks by the Q of that name, so two arms of a match
  can rank by two Qs;
- **CUDA graphs**: the server's graphed forward pass (`qrank.QGraphs`) is the eager
  pass's matrix to the bit.
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

    def __init__(self, encoder: Encoder, name: str = "stub-q.pt") -> None:
        self.encoder = encoder
        self.name = name
        self.calls = 0
        self.trips = 0

    def describe(self) -> list[str]:
        return [self.name]

    def matrix(self, reg, pos, pools):  # noqa: ANN001, ANN201
        self.calls += 1
        self.trips += 1
        return _stub_matrix(qrank._pool_arrays(reg, self.encoder, pos, pools, True))  # noqa: SLF001

    def matrices(self, reg, asks):  # noqa: ANN001, ANN201
        self.trips += 1
        self.calls += len(asks)
        return [
            _stub_matrix(qrank._pool_arrays(reg, self.encoder, pos, pools, True))  # noqa: SLF001
            for pos, pools in asks
        ]


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
    for bad in ("q-fast", "q2", "qnocover", "q-nocover-fast", "q.", "q.Long", "q.long-nocover"):
        with pytest.raises(ValueError, match="rank fill"):
            parse_rank_fill(bad)
    # Stage 3: a Q of another name, the cover as the label says.
    assert parse_rank_fill("q-nocover.long") == (0, False)
    assert rank_fill_covers("q.long") and not rank_fill_covers("q-nocover.long")
    assert qrank.q_name("q-nocover.long") == "long" and qrank.q_name("q-nocover") == ""


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
        trips = stub_q.trips
        menus = selfplay._menus(reg, pos, (4, 4), _stub, budget, True, rank_fill=label)  # noqa: SLF001
        assert menus == expect[cover]
        # One Q per side's ranking, over both whole pools; no leaf cell. Nothing is hidden
        # here, so both sides read one position and one pair of pools: one request, in
        # one round trip (stage 3).
        assert stub_q.calls - calls == 1
        assert stub_q.trips - trips == 1
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
            # Stage 3: several in one round trip (op q_many), each its own answer.
            other = selfplay.position_from_sets(
                reg, list(pool.teams[1].sets)[:4], list(pool.teams[0].sets)[:4],
                rng=np.random.default_rng(1),
            )
            pools2 = (qhead.legal_pool(reg, other, 0), qhead.legal_pool(reg, other, 1))
            both = remote.matrices(reg, [(pos, pools), (other, pools2)])
            assert np.array_equal(both[0], want)
            want2 = _stub_matrix(qrank._pool_arrays(reg, encoder, other, pools2, True))  # noqa: SLF001
            assert not np.array_equal(want2, want)
            assert np.array_equal(both[1], want2)
            assert served.calls == 3 and remote.calls == 3 and remote.trips == 2
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


def _game(record) -> object:  # noqa: ANN001
    """A record as JSON without its clocks."""

    def strip(x: object) -> object:
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if "Seconds" not in k}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    return strip(record.to_json(objective="test", search_limit=2))


def test_one_round_trip_plays_the_games_of_one_request_per_ranking(
    pool, stub_q, monkeypatch  # noqa: ANN001
) -> None:
    """Stage 3: `prefetch` sends both sides' requests together; the games do not move."""
    solver = SolvedSelections(pool.reg, pool.teams, _stub)
    tested, other = _arm("q-nocover", solver), _arm("refs2-nocover", solver)
    got = {}
    for how in ("together", "apart"):
        if how == "apart":
            # Positive control: without the prefetch every ranking is its own trip.
            monkeypatch.setattr(qrank, "prefetch", lambda *_a, **_k: {})
        trips, calls = stub_q.trips, stub_q.calls
        records = [
            pool_match_game(
                pool.reg, pool, (tested, other), seed=274, game_index=g, which=which,
                hide_bench=True, max_turns=3,
            )[0]
            for g in (0,)
            for which in (0, 1)
        ]
        got[how] = (records, stub_q.trips - trips, stub_q.calls - calls)
    together, apart = got["together"], got["apart"]
    assert [_game(r) for r in together[0]] == [_game(r) for r in apart[0]]
    # The same requests (hidden benches: two views, two requests a menu), half the trips.
    assert together[2] == apart[2] > 0
    assert 2 * together[1] == apart[1]


def test_each_arm_ranks_by_the_q_its_label_names(pool) -> None:  # noqa: ANN001
    """Stage 3: ``q-nocover`` and ``q-nocover.b`` in one match rank by two Qs."""
    encoder = Encoder(pool.reg)
    plain, named = _StubQ(encoder, "a.pt"), _StubQ(encoder, "b.pt")
    qrank.install(plain)
    qrank.install(named, "b")
    try:
        solver = SolvedSelections(pool.reg, pool.teams, _stub)
        tested, other = _arm("q-nocover", solver), _arm("q-nocover.b", solver)
        before = _rankings()
        record, _sides = pool_match_game(
            pool.reg, pool, (tested, other), seed=274, game_index=0, which=1,
            hide_bench=True, max_turns=2,
        )
        assert record.rank_fill == ["q-nocover.b", "q-nocover"]
        after = _rankings()
        moves = sum(d.kind == "move" for d in record.decisions)
        for label in ("q-nocover", "q-nocover.b"):
            assert after[label] - before.get(label, 0) == 2 * moves
        # Each arm's requests went to its own Q, and only there (one or two a menu: two
        # when the sides' views differ).
        assert 0 < plain.calls <= 2 * moves and 0 < named.calls <= 2 * moves
        assert plain.trips == named.trips == moves
        assert qrank.describe_installed({"": plain, "b": named}) == ["a.pt", "b=b.pt"]
        with pytest.raises(RuntimeError, match="needs the Q 'c'"):
            qrank.installed("c")
    finally:
        qrank._INSTALLED.clear()  # noqa: SLF001


def _untrained_q(reg, encoder, path):  # noqa: ANN001, ANN202
    import torch

    torch.manual_seed(274)
    config = qhead.QConfig(properties=True)
    table = qhead.move_table(reg, encoder.vocab)
    net = qhead.build_net(encoder, config, table)
    torch.save(
        {
            "state": net.state_dict(), "config": qhead.config_dict(config),
            "vocab_fingerprint": encoder.vocab.fingerprint(),
            "regulation": reg.meta.format_id, "move_table": table,
        },
        path,
    )
    return path


def test_a_q_loaded_here_and_served_answer_alike(pool, pos, tmp_path) -> None:  # noqa: ANN001
    pytest.importorskip("torch")
    from pokeuraou.inference import serve

    reg = pool.reg
    encoder = Encoder(reg)
    path = _untrained_q(reg, encoder, tmp_path / "q.pt")
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


def test_the_graphed_q_is_the_eager_q_to_the_bit(pool, tmp_path) -> None:  # noqa: ANN001
    """Stage 3: `QGraphs` (trunk graph, a graph per side and pool size, eager pair head)
    answers `qhead.q_matrix`'s matrix to the bit, over pool sizes met in any order."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a card")
    reg = pool.reg
    encoder = Encoder(reg)
    net = qhead.load_q(_untrained_q(reg, encoder, tmp_path / "q.pt"), "cuda")
    device = torch.device("cuda")
    graphs = qrank.QGraphs(net, device)
    teams = [list(t.sets) for t in pool.teams]
    requests = []
    for i in range(3):
        pos = selfplay.position_from_sets(
            reg, teams[i % len(teams)][:4], teams[(i + 1) % len(teams)][:4],
            rng=np.random.default_rng(i),
        )
        full = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
        # Every pair of a few sizes, met in a shuffled order, so a side's graph is often
        # captured after the other side's graph of a size it meets later: the case a pool
        # shared between the graphs got wrong (a later capture's output under an earlier
        # one's scratch).
        for c0 in (None, 3, 9, 17, 30):
            for c1 in (None, 3, 9, 17, 30):
                pools = tuple(p[:c] if c else p for p, c in zip(full, (c0, c1), strict=True))
                requests.append(qrank._pool_arrays(reg, encoder, pos, pools, True))  # noqa: SLF001
    order = np.random.default_rng(274).permutation(len(requests))
    requests = [requests[i] for i in order]
    for arrays in requests + requests[::-1]:
        want = qhead.q_matrix(net, arrays, device)
        got = graphs.matrix(arrays)
        assert got is not None and np.array_equal(got, want)
    assert graphs.failed is None
    assert graphs.replays == 2 * len(requests)
    assert graphs.captured == 1 + len({(s, len(a[f"acts{s}"])) for a in requests for s in (0, 1)})
