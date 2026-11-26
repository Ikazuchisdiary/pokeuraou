"""IKA-457 stage 2.3 (`rootmenus`): a move node's two menus by the Q, built in the port, are
`selfplay._menus`' to the bit -- the menus action for action in order, the Q's requests to the
inference server one for one (the `q` / `q_many` request and every array's bytes), the cut at
each width and the tally of the options a coverless menu leaves off (`narrow.COVERLESS`).

The server is `inference.serve` with `test_port_menus`' numpy stand-in Q. Both views of the
node are tried: the node itself (nothing hidden: one request), and the heaviest completion of
each side's bench (two requests). The positive control is the node the port took
(`portlp.COUNTS["menus"]`); the control that the comparison can fail is another Q. A fault from
the port leaves the menus to Python.

No torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import narrow, portlp, qrank, rustnode
from pokeuraou.budget import Budget
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.inference import serve
from pokeuraou.selfplay import _menus

from .test_ladder import LEAF, _played
from .test_port_menus import QStub, roster  # noqa: F401 - fixture

FILL = "q-nocover"


class OtherQ(QStub):
    """Another Q: the same matrix with its rows in the other order."""

    def _one(self, arrays: dict[str, np.ndarray], size: int) -> np.ndarray:
        return super()._one(arrays, size)[::-1].copy()


@pytest.fixture
def flags_off():  # noqa: ANN201
    was = (portlp.MENUS[0], portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0],
           portlp.FOLD[0], portlp.GEN[0], rustnode._SPAN_BLOCKS[0])  # noqa: SLF001
    yield
    (portlp.MENUS[0], portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0],
     portlp.FOLD[0], portlp.GEN[0]) = was[:7]
    rustnode._SPAN_BLOCKS[0] = was[7]  # noqa: SLF001


def _install(roster, stub):  # noqa: ANN001, ANN202, F811
    encoder = Encoder(roster.reg)
    stub.fingerprint = encoder.vocab.fingerprint()
    server, address = serve({}, q_models={"qstub": stub})
    model = qrank.RemoteQ(address, "qstub", encoder)
    saved = dict(qrank._INSTALLED)  # noqa: SLF001
    qrank.install(model)

    def close() -> None:
        qrank._INSTALLED.clear()  # noqa: SLF001
        qrank._INSTALLED.update(saved)  # noqa: SLF001
        model.close()
        server.shutdown()

    return model, close


@pytest.fixture
def served(roster, flags_off):  # noqa: ANN001, ANN201, F811
    stub = QStub("")
    model, close = _install(roster, stub)
    yield stub, model
    close()


def _build(reg, pos, limits, spreads, stub, *, menus):  # noqa: ANN001, ANN202
    portlp.MENUS[0] = False
    portlp.set_node(True)
    if menus:
        portlp.set_menus(True)
    del stub.log[:]
    before = {k: dict(v) for k, v in narrow.COVERLESS.items()}
    used: dict = {}
    got = _menus(
        reg, pos, limits, LEAF, Budget.matrix(), True, None, spreads,
        used=used, rank_fill=FILL,
    )
    after = narrow.COVERLESS.get(FILL, {"menus": 0, "dropping": 0, "dropped": 0})
    was = before.get(FILL, {"menus": 0, "dropping": 0, "dropped": 0})
    tally = {k: after[k] - was[k] for k in after}
    return got, list(stub.log), tally, used


def _nodes(roster):  # noqa: ANN001, ANN202, F811
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    out = []
    for pos in _played(roster):
        out.append((reg, pos, None))
        hidden = {s: completions(reg, pos, s, sheet, seen=frozenset({0, 1})) for s in (0, 1)}
        if any(len(items) > 1 for items in hidden.values()):
            out.append((reg, pos, hidden))
    return out


@pytest.mark.parametrize("limits", [(12, 12), (4, 7), (1, 30)])
def test_the_ports_menus_are_menus(roster, served, limits) -> None:  # noqa: ANN001, F811
    stub, _model = served
    took = hidden_nodes = cut = 0
    for reg, pos, spreads in _nodes(roster):
        before = portlp.COUNTS["menus"]
        python = _build(reg, pos, limits, spreads, stub, menus=False)
        ported = _build(reg, pos, limits, spreads, stub, menus=True)
        took += portlp.COUNTS["menus"] - before
        assert portlp.COUNTS["menus"] - before == 1, "the port did not build the menus: vacuous"
        assert ported[0] == python[0], "the menus differ"
        assert ported[1] == python[1], "the Q was sent other requests"
        assert ported[2] == python[2], "the coverless tally differs"
        assert ported[3] == python[3], "the view each side ranked from differs"
        hidden_nodes += int(spreads is not None)
        cut += int(python[2]["dropped"] > 0)
    assert took >= 3 and hidden_nodes >= 1
    if limits[0] < 12:
        assert cut >= 1, "no menu was cut: the tally of the options left off is not held"


def test_two_views_are_two_requests(roster, served) -> None:  # noqa: ANN001, F811
    stub, _model = served
    two = 0
    for reg, pos, spreads in _nodes(roster):
        if spreads is None:
            continue
        ported = _build(reg, pos, (12, 12), spreads, stub, menus=True)
        python = _build(reg, pos, (12, 12), spreads, stub, menus=False)
        assert ported[1] == python[1]
        two += int(len(python[1]) == 2)
    assert two >= 1, "no node ranked from two views: the q_many request is not held"


def test_a_fault_leaves_the_menus_to_python(roster, served, monkeypatch) -> None:  # noqa: ANN001, F811
    stub, _model = served
    reg, pos, spreads = _nodes(roster)[0]
    want = _build(reg, pos, (12, 12), spreads, stub, menus=False)
    monkeypatch.setattr(
        rustnode.RustNode, "root_menus", lambda self, request: {"fault": "injected"}  # noqa: ARG005
    )
    faults, built = portlp.COUNTS["faults"], portlp.COUNTS["menus"]
    got = _build(reg, pos, (12, 12), spreads, stub, menus=True)
    assert portlp.COUNTS["faults"] == faults + 1 and portlp.COUNTS["menus"] == built
    assert got[0] == want[0] and got[1] == want[1]


def test_another_q_is_other_menus(roster, flags_off) -> None:  # noqa: ANN001, F811
    """The check above is not vacuous: a Q with its rows the other way round gives other menus."""
    menus_of = []
    for stub in (QStub(""), OtherQ("")):
        _model, close = _install(roster, stub)
        try:
            menus_of.append([
                _build(reg, pos, (4, 4), spreads, stub, menus=True)[0]
                for reg, pos, spreads in _nodes(roster)
            ])
        finally:
            close()
    assert menus_of[0] != menus_of[1]
