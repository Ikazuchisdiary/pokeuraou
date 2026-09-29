"""IKA-389 (`portmenus`): a read's children's Q menus built in the port are `_q_menus`' to
the bit -- the legal pools action for action and in order, the Q's arrays byte for byte, the
Q's requests to the inference server one for one, the scores (numpy's BLAS, the mean where
the game has no answer) and the menus.

The server is `inference.serve` with a numpy stand-in Q that records every request (each
array's name, dtype, shape and bytes, and the batch it came in) and whose matrix moves with
the batch's size, as a CUDA answer does (IKA-307): so the same menus need the same requests.
Its games tie across their support up to the last bits, as a Q's do, so the order among the
support is decided by the products' last bits. No torch.
"""

from __future__ import annotations

import base64
import hashlib

import numpy as np
import pytest

from pokeuraou import deepen, ladder, portmenus, portserved, qhead, qrank, rustnode
from pokeuraou.actions import side_actions
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.equilibrium import EquilibriumError, solve
from pokeuraou.inference import serve
from pokeuraou.position import Position
from pokeuraou.teams import load_roster

from .test_ladder import LEAF, _played
from .test_ladder_pool import _node, _same
from .test_portserved import _chomp_gambit


class QStub:
    """A Q in the server: each request's arrays kept (with its batch's size), and a matrix of
    a smooth pair score of the two actions' rows that moves with the batch's size."""

    files = ["q-stub"]
    digests = ["0" * 64]
    properties = True

    def __init__(self, fingerprint: str) -> None:
        self.fingerprint = fingerprint
        self.log: list[tuple[int, str]] = []

    def _one(self, arrays: dict[str, np.ndarray], size: int) -> np.ndarray:
        digest = hashlib.sha256()
        for name in (*qhead.POSITION_ARRAYS, "acts0", "acts1", "feats0", "feats1"):
            array = arrays[name]
            digest.update(f"{name}{array.dtype.str}{array.shape}".encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        self.log.append((size, digest.hexdigest()))
        base = float(arrays["mon"].astype(np.float64).sum()) * 1e-4
        w = np.arange(1, 15, dtype=np.float64)

        def vec(side: int) -> np.ndarray:
            acts = arrays[f"acts{side}"].reshape(len(arrays[f"acts{side}"]), -1).astype(np.float64)
            feats = arrays[f"feats{side}"].astype(np.float64)
            return np.tanh(acts @ w * 0.01 + feats.sum(axis=1) * 0.05 + base)

        u, v = vec(0), vec(1)
        # Cyclic in the difference, so the games mix (as a Q's do).
        d = u[:, None] - v[None, :]
        logit = np.sin(9.0 * d) + 0.2 * d
        return 1.0 / (1.0 + np.exp(-logit)) + size * 1e-12

    def __call__(self, arrays: dict[str, np.ndarray]) -> np.ndarray:
        return self._one(arrays, 1)

    def batch(self, requests: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
        return [self._one(r, len(requests)) for r in requests]


def _variant(pos: Position, edit) -> Position:  # noqa: ANN001
    data = pos.to_json()
    edit(data)
    return Position.from_json(data)


def _active(data: dict, side: int, slot: int = 0) -> dict:
    return data["sides"][side]["pokemon"][data["sides"][side]["active"][slot]]


def _volatile(vid: str, **extra) -> dict:  # noqa: ANN003
    return {"id": vid, **extra}


def _variants(pos: Position) -> list[Position]:
    """The rules `slot_actions` and `_usable_move_slots` read, each set on the position."""
    first = pos.sides[0].pokemon[pos.sides[0].active[0]].moves[0].id
    edits = [
        lambda d: _active(d, 0)["volatiles"].append(_volatile("taunt")),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("encore", move=first)),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("disable", move=first)),
        lambda d: (_active(d, 0)["volatiles"].append(_volatile("torment")),
                   _active(d, 0).update(lastMove=first)),
        lambda d: (_active(d, 0)["volatiles"].append(_volatile("choicelock", move=first)),
                   _active(d, 0).update(item="choicescarf")),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("throatchop")),
        lambda d: _active(d, 1)["volatiles"].append(_volatile("imprison")),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("partiallytrapped")),
        lambda d: _active(d, 0).update(trapped=True),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("mustrecharge")),
        lambda d: _active(d, 0)["volatiles"].append(_volatile("lockedmove", move=first)),
        lambda d: _active(d, 0)["volatiles"].append(
            _volatile("twoturnmove", move=first, extra={"targetLoc": 1})),
        lambda d: _active(d, 0).update(lockedMove=first),
        lambda d: [m.update(pp=0) for m in _active(d, 0)["moves"]],
        lambda d: d["sides"][0].update(megaUsed=True),
        lambda d: _active(d, 0, 1).update(fainted=True, hp=0),
        lambda d: _active(d, 1).update(ability="shadowtag"),
        lambda d: _active(d, 1).update(ability="magnetpull"),
        lambda d: (_active(d, 1).update(ability="arenatrap"),
                   d["field"]["pseudoWeather"].append({"id": "gravity"})),
        lambda d: _active(d, 0).update(item="shedshell", trapped=False),
        lambda d: _active(d, 0).update(activeMoveActions=2),
        lambda d: d["sides"][1]["active"].__setitem__(1, None),
    ]
    return [_variant(pos, edit) for edit in edits]


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _positions(roster) -> list[tuple]:  # noqa: ANN001
    reg = roster.reg
    played = _played(roster)
    out = [(reg, p) for p in played]
    out += [(reg, v) for p in played[:2] for v in _variants(p)]
    mc_reg, chomp = _chomp_gambit()
    out += [(mc_reg, chomp)] + [(mc_reg, v) for v in _variants(chomp)]
    return out


def _by_reg(items: list[tuple]) -> dict:
    grouped: dict = {}
    for reg, pos in items:
        grouped.setdefault(reg.meta.format_id, (reg, []))[1].append(pos)
    return grouped


def test_the_ports_legal_pools_are_pythons(roster) -> None:  # noqa: ANN001
    """`side_actions` and `qhead.legal_pool` of each position, action by action in order.
    The positive control: the variants move the pools (a rule the port did not read would
    leave its pool as the base position's)."""
    moved = 0
    for reg, positions in _by_reg(_positions(roster)).values():
        node = rustnode.require_node(reg)
        raw = node.port_lists("legal", positions, each={"raw": True})["pools"]
        legal = node.port_lists("legal", positions)["pools"]
        for pos, raw_pools, pools in zip(positions, raw, legal, strict=True):
            for side in (0, 1):
                assert [portmenus._action(a) for a in raw_pools[side]] == side_actions(reg, pos, side)
                assert [portmenus._action(a) for a in pools[side]] == qhead.legal_pool(reg, pos, side)
        base = [a.to_choice() for a in side_actions(reg, positions[0], 0)]
        moved += sum(1 for p in positions[1:] if [a.to_choice() for a in side_actions(reg, p, 0)] != base)
    assert moved >= 20, moved


def test_the_ports_q_arrays_are_pythons(roster) -> None:  # noqa: ANN001
    """`qrank._pool_arrays` of each position with its legal pools, byte for byte."""
    compared = 0
    for reg, positions in _by_reg(_positions(roster)).values():
        encoder = Encoder(reg)
        got = rustnode.require_node(reg).port_lists(
            "qArrays", positions, extra={"properties": True, "megaFromSlots": False})["arrays"]
        for pos, arrays in zip(positions, got, strict=True):
            pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
            if not pools[0] or not pools[1]:
                continue
            want = qrank._pool_arrays(reg, encoder, pos, pools, True)
            assert sorted(arrays) == sorted(want)
            for name, array in want.items():
                assert arrays[name]["dtype"] == array.dtype.str
                assert list(arrays[name]["shape"]) == list(array.shape)
                assert base64.b64decode(arrays[name]["data"]) == np.ascontiguousarray(array).tobytes()
            compared += 1
    assert compared >= 20


def test_the_ports_scores_are_numpys(roster) -> None:  # noqa: ANN001
    """The candidates' scores: `solve`'s `row_ev` and minus its `col_ev` (numpy's BLAS), and
    the means against every reply where the game has no answer -- to the bit."""
    library = portmenus.blas()
    assert library is not None, "numpy's BLAS is not an OpenBLAS the port can load here"
    rng = np.random.default_rng(7)
    games = [rng.random((int(rng.integers(2, 90)), int(rng.integers(2, 90)))) for _ in range(150)]
    games += [np.round(g, 1) for g in games[:40]] + [rng.random((30, 1)), rng.random((1, 30))]
    request = {"kind": "qScores", "blas": library, "games": [
        {"rows": g.shape[0], "cols": g.shape[1],
         "data": base64.b64encode(g.astype("<f8").tobytes()).decode()} for g in games]}
    node = rustnode.require_node(roster.reg)
    solved = node._exchange(request)["answers"]  # noqa: SLF001
    means = node._exchange({**request, "mean": True})["answers"]  # noqa: SLF001

    def unpack(text: str) -> np.ndarray:
        return np.frombuffer(base64.b64decode(text), "<f8")

    for g, got, mean in zip(games, solved, means, strict=True):
        m, n = g.shape
        np.testing.assert_array_equal(unpack(mean["rows"]), g.mean(axis=1))
        np.testing.assert_array_equal(unpack(mean["cols"]), -g.mean(axis=0))
        try:
            e = solve(g)
            want = (np.asarray(e.row_ev), -np.asarray(e.col_ev))
        except (EquilibriumError, ValueError):
            want = (g.mean(axis=1), -g.mean(axis=0))
        # One score orders nothing, and numpy takes `dot` rather than `gemv` there.
        if m > 1:
            np.testing.assert_array_equal(unpack(got["rows"]), want[0])
        if n > 1:
            np.testing.assert_array_equal(unpack(got["cols"]), want[1])


@pytest.fixture
def served(roster):  # noqa: ANN001, ANN201
    reg = roster.reg
    encoder = Encoder(reg)
    stub = QStub(encoder.vocab.fingerprint())
    server, address = serve({}, q_models={"qstub": stub})
    model = qrank.RemoteQ(address, "qstub", encoder)
    saved = dict(qrank._INSTALLED)  # noqa: SLF001
    qrank.install(model)
    yield stub, model
    qrank._INSTALLED.clear()  # noqa: SLF001
    qrank._INSTALLED.update(saved)  # noqa: SLF001
    model.close()
    server.shutdown()
    portmenus.ON[0] = False
    portserved.ON[0] = False


def _both(reg, positions, width, stub):  # noqa: ANN001, ANN202
    portmenus.ON[0] = False
    del stub.log[:]
    python = deepen._q_menus(reg, positions, width)
    python_log = list(stub.log)
    portmenus.ON[0] = True
    del stub.log[:]
    before = dict(portmenus.COUNTS)
    ported = deepen._q_menus(reg, positions, width)
    portmenus.ON[0] = False
    assert stub.log == python_log
    assert ported == python
    return portmenus.COUNTS["asked"] - before["asked"]


@pytest.mark.parametrize("chunk", [deepen.Q_MENU_CHUNK, 3])
def test_the_ports_menus_are_q_menus(roster, served, monkeypatch, chunk) -> None:  # noqa: ANN001
    """Every child's two menus and every request the Q was sent, at the default chunk and at
    a chunk of 3 (several requests a call). The positive control: the positions the port
    ranked, and games whose support ties (the order among it is the products' last bits)."""
    stub, _model = served
    monkeypatch.setattr(deepen, "Q_MENU_CHUNK", chunk)
    reg = roster.reg
    positions = _played(roster) + [v for p in _played(roster)[:2] for v in _variants(p)]
    asked = 0
    for width in (1, 4, 12):
        asked += _both(reg, positions, width, stub)
    assert asked >= 3 * (len(positions) - 3), asked
    ties = 0
    for pos in positions:
        pools = (qhead.legal_pool(reg, pos, 0), qhead.legal_pool(reg, pos, 1))
        if pools[0] and pools[1]:
            e = solve(stub._one(qrank._pool_arrays(reg, Encoder(reg), pos, pools, True), 1))
            support = e.row_ev[e.row_strategy > 1e-9]
            ties += int(len(support) > 1 and np.ptp(support) < 1e-12)
    assert ties >= 5, ties


def test_a_ladder_read_is_the_same_read(roster, served) -> None:  # noqa: ANN001
    """A ladder read whose children's menus are the Q's (``k``) reads the same with the port
    building them: every rung, the counted work and the notes. The positive control: the
    children the port ranked (`here["portMenus"]`)."""
    stub, _model = served
    reg = roster.reg
    stages = ladder.parse_ladder("d2r2b3k3+d2r3bak4x")
    for pos in _played(roster)[:2]:
        ours, theirs, items, matrices, weights, start = _node(reg, pos, 5)
        saved = ladder._POOL
        ladder._POOL = None
        try:
            reads = []
            for on in (False, True):
                portmenus.ON[0] = on
                del stub.log[:]
                reads.append((ladder.read(reg, 0, ours, theirs, items, matrices, weights, start,
                                          LEAF, budget=Budget.matrix(), stages=stages, budget_ms=None),
                              list(stub.log)))
        finally:
            ladder._POOL = saved
            portmenus.ON[0] = False
        (python, python_log), (ported, ported_log) = reads
        _same(ported, python)
        assert ported_log == python_log
        assert python.here["portMenus"] == 0 and ported.here["portMenus"] > 0

