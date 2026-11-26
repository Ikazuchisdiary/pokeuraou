"""IKA-457 stage 2.4 (`moveNode`): a move node's menus and the node over them in one request to
the port, against the two requests of stage 2.3 and 1.4 (`_menus`, then `belief_payoffs`).

What these hold, on one `inference.serve` that is both the leaf's server (`test_portserved`'s
numpy stand-in, keeping every request's rows and bytes) and the Q's (`test_port_menus`' stand-in):

- **the same menus and the same node**: both menus action for action, every completion's matrix
  to the bit (the last place off the SkylakeX core, as `test_belief_fold_port` allows), the
  games the request solved, the notes -- with the bench hidden and with nothing hidden;
- **the same requests**: the Q was sent the same requests and the leaf the same rows, each in
  the same order (the Q's first);
- the positive control is the node the port decided (`portlp.COUNTS["decided"]`); the control
  that the comparison can fail is another Q; a fault leaves both to Python.

No torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import portlp, qrank, rustnode
from pokeuraou.beliefnode import belief_payoffs
from pokeuraou.budget import Budget
from pokeuraou.encode import Encoder
from pokeuraou.equilibrium import solve_bayesian
from pokeuraou.hidden import completions
from pokeuraou.inference import RemoteValue, serve
from pokeuraou.selfplay import _decide, _menus

from .test_ladder import _played
from .test_port_menus import QStub, roster  # noqa: F401 - fixture
from .test_portlp import openblas_core
from .test_portserved import Stub

FILL = "q-nocover"
EXACT = openblas_core() == "SkylakeX"


def _equal(x, y) -> bool:  # noqa: ANN001
    return bool(np.array_equal(x, y)) if EXACT else bool(np.allclose(x, y, rtol=0, atol=1e-12))


class OtherQ(QStub):
    """Another Q: the same matrix with its rows in the other order."""

    def _one(self, arrays: dict[str, np.ndarray], size: int) -> np.ndarray:
        return super()._one(arrays, size)[::-1].copy()


@pytest.fixture
def flags_off():  # noqa: ANN201
    was = (portlp.DECIDE[0], portlp.MENUS[0], portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0],
           portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0], rustnode._SPAN_BLOCKS[0])  # noqa: SLF001
    yield
    (portlp.DECIDE[0], portlp.MENUS[0], portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0],
     portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0]) = was[:8]
    rustnode._SPAN_BLOCKS[0] = was[8]  # noqa: SLF001


def _serve(roster, q_stub):  # noqa: ANN001, ANN202, F811
    reg = roster.reg
    encoder = Encoder(reg)
    q_stub.fingerprint = encoder.vocab.fingerprint()
    leaf_stub = Stub()
    server, address = serve({"stub": leaf_stub}, q_models={"qstub": q_stub})
    model = qrank.RemoteQ(address, "qstub", encoder)
    saved = dict(qrank._INSTALLED)  # noqa: SLF001
    qrank.install(model)
    leaf = RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20)

    def close() -> None:
        leaf.close()
        qrank._INSTALLED.clear()  # noqa: SLF001
        qrank._INSTALLED.update(saved)  # noqa: SLF001
        model.close()
        server.shutdown()

    return leaf_stub, leaf, close


def _nodes(roster):  # noqa: ANN001, ANN202, F811
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    out = []
    for pos in _played(roster):
        everything = frozenset(range(len(pos.sides[0].pokemon)))
        out.append((reg, pos, {s: completions(reg, pos, s, sheet, seen=everything) for s in (0, 1)}))
        hidden = {s: completions(reg, pos, s, sheet, seen=frozenset({0, 1})) for s in (0, 1)}
        if any(len(items) > 1 for items in hidden.values()):
            out.append((reg, pos, hidden))
    return out


def _two_roads(reg, pos, spreads, limits, q_stub, leaf_stub, leaf, *, fused):  # noqa: ANN001, ANN202
    portlp.DECIDE[0] = False
    portlp.set_exact(True)
    if fused:
        portlp.set_decide(True)
    del q_stub.log[:], leaf_stub.log[:]
    used: dict = {}
    got = _decide(
        reg, pos, limits, leaf, leaf, Budget.matrix(), True, None, spreads, True, (0, 0), (1, 1),
        ("lp", "lp"), False, "heaviest", used, FILL,
    )
    if got is not None:
        ours, theirs, node = got
    else:
        ours, theirs = _menus(
            reg, pos, limits, leaf, Budget.matrix(), True, None, spreads, used=used, rank_fill=FILL
        )
        node = belief_payoffs(
            reg, pos, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads, solve=(0, 1)
        )
    return ours, theirs, node, list(q_stub.log), list(leaf_stub.log), used, got is not None


def _games(node, spreads, side):  # noqa: ANN001, ANN202
    weights = np.asarray([item.weight for item in spreads[1 - side]], dtype=np.float64)
    return [m if side == 0 else -m.T for m in node.matrices[side]], weights


@pytest.mark.parametrize("limits", [(12, 12), (4, 7)])
def test_the_menus_and_the_node_are_one_request(roster, flags_off, limits) -> None:  # noqa: ANN001, F811
    q_stub = QStub("")
    leaf_stub, leaf, close = _serve(roster, q_stub)
    try:
        decided = hidden = 0
        for reg, pos, spreads in _nodes(roster):
            before = portlp.COUNTS["decided"]
            python = _two_roads(reg, pos, spreads, limits, q_stub, leaf_stub, leaf, fused=False)
            ported = _two_roads(reg, pos, spreads, limits, q_stub, leaf_stub, leaf, fused=True)
            assert not python[6] and ported[6], "the port did not decide the node: vacuous"
            assert portlp.COUNTS["decided"] - before == 1
            assert ported[0] == python[0] and ported[1] == python[1], "the menus differ"
            assert ported[3] == python[3], "the Q was sent other requests"
            assert ported[4] == python[4], "the leaf was sent other rows"
            assert ported[5] == python[5], "the view each side ranked from differs"
            a, b = python[2], ported[2]
            for side in (0, 1):
                assert len(a.matrices[side]) == len(b.matrices[side])
                assert all(_equal(x, y) for x, y in zip(a.matrices[side], b.matrices[side], strict=True))
            assert a.unmodelled == b.unmodelled
            assert set(b.solved) == {0, 1}
            for side in (0, 1):
                games, weights = _games(a if EXACT else b, spreads, side)
                want = solve_bayesian(games, weights)
                assert b.solved[side].value == want.value
                assert np.array_equal(b.solved[side].row_strategy, want.row_strategy)
            decided += 1
            hidden += int(any(len(items) > 1 for items in spreads.values()))
        assert decided >= 3 and hidden >= 1
    finally:
        close()


def test_a_fault_leaves_the_menus_and_the_node_to_python(roster, flags_off, monkeypatch) -> None:  # noqa: ANN001, F811
    q_stub = QStub("")
    leaf_stub, leaf, close = _serve(roster, q_stub)
    try:
        reg, pos, spreads = _nodes(roster)[0]
        want = _two_roads(reg, pos, spreads, (12, 12), q_stub, leaf_stub, leaf, fused=False)
        monkeypatch.setattr(
            rustnode.RustNode, "move_node", lambda self, request: {"fault": "injected"}  # noqa: ARG005
        )
        faults, decided = portlp.COUNTS["faults"], portlp.COUNTS["decided"]
        got = _two_roads(reg, pos, spreads, (12, 12), q_stub, leaf_stub, leaf, fused=True)
        assert not got[6]
        assert portlp.COUNTS["faults"] == faults + 1 and portlp.COUNTS["decided"] == decided
        assert got[0] == want[0] and got[1] == want[1]
    finally:
        close()


def test_another_q_is_other_menus_in_the_fused_request(roster, flags_off) -> None:  # noqa: ANN001, F811
    """The check above is not vacuous: a Q with its rows the other way gives other menus."""
    seen = []
    for q_stub in (QStub(""), OtherQ("")):
        leaf_stub, leaf, close = _serve(roster, q_stub)
        try:
            seen.append([
                _two_roads(reg, pos, spreads, (4, 4), q_stub, leaf_stub, leaf, fused=True)[0]
                for reg, pos, spreads in _nodes(roster)
            ])
        finally:
            close()
    assert seen[0] != seen[1]
