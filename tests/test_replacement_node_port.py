"""A replacement node solved in one request to the port (IKA-457 stage 2.2, `replacementNode`)
against the node Python reads one completion at a time (`_do_replacement_node`).

What these hold, on the inference server's numpy stand-in of `test_portserved`, which keeps each
request's rows and the bytes of every array, and on the recorded mid-game replacement of
`test_hidden_search`:

- **the same node**: the decision the node records (the menus, both mixtures, the values, the
  choices) and the position it leads to are the Python road's to the bit, with the bench hidden
  and with nothing hidden, and with one leaf for both sides or a leaf each.
- **the same requests**: the server is sent the same rows (their count and the bytes of every
  array) in the same order -- side 0's completions, then side 1's.
- The positive control is the node the port took (`portlp.COUNTS["replaced"]`); the control that
  the comparison can fail is another server. A fault from the port leaves the node to Python.

No torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import portlp, rustnode
from pokeuraou.encode import Encoder
from pokeuraou.inference import RemoteValue, serve
from pokeuraou.selfplay import GameRecord, _do_replacement_node

from .test_hidden_search import SEEN_ALL, _fixture_sheets, replacement  # noqa: F401 - fixture
from .test_portlp import openblas_core
from .test_portserved import Stub

#: The LP is the same HiGHS either way; the numpy order is the SkylakeX one only on that core.
EXACT = openblas_core() == "SkylakeX"


@pytest.fixture(scope="module")
def server():  # noqa: ANN201
    stub = Stub()
    shut, address = serve({"stub": stub})
    yield stub, address
    shut.shutdown()


@pytest.fixture
def flags_off():  # noqa: ANN201
    was = (portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0],
           portlp.GEN[0], rustnode._SPAN_BLOCKS[0])  # noqa: SLF001
    yield
    (portlp.REPLACE[0], portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0],
     portlp.GEN[0]) = was[:6]
    rustnode._SPAN_BLOCKS[0] = was[6]  # noqa: SLF001


def _run(reg, position, owed, sheets, shown, leaves, *, replace):  # noqa: ANN001, ANN202
    portlp.REPLACE[0] = False
    portlp.set_node(True)
    if replace:
        portlp.set_replace(True)
    record = GameRecord(own_team=[], foe_team=[], foe_archetype="test")
    after = _do_replacement_node(
        reg, np.random.default_rng(3), position.copy(), owed, record, leaves,
        sheets=sheets, shown=shown,
    )
    assert record.decisions, "the node recorded nothing"
    return record.decisions[0], after, record


def _same(a, b) -> bool:  # noqa: ANN001
    compare = np.array_equal if EXACT else (lambda x, y: np.allclose(x, y, rtol=0, atol=1e-12))
    return (
        a.own_actions == b.own_actions and a.foe_actions == b.foe_actions
        and compare(a.own_policy, b.own_policy) and compare(a.foe_policy, b.foe_policy)
        and (a.search_value == b.search_value if EXACT else abs(a.search_value - b.search_value) < 1e-12)
        and a.own_chosen == b.own_chosen and a.foe_chosen == b.foe_chosen
    )


def _compare(replacement, server, *, shown, distinct):  # noqa: ANN001, ANN202, F811
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    stub, address = server
    with (
        RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as first,
        RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as second,
    ):
        leaves = (first, second if distinct else first)
        runs = []
        for replace in (False, True):
            del stub.log[:]
            before = portlp.COUNTS["replaced"]
            decision, after, record = _run(
                reg, position, owed, sheets, shown, leaves, replace=replace
            )
            runs.append((decision, after, list(stub.log), portlp.COUNTS["replaced"] - before, record))
    (python, py_after, py_log, ran0, _), (ported, port_after, port_log, ran1, record) = runs
    assert (ran0, ran1) == (0, 1), "the port did not take the node: the comparison is vacuous"
    assert port_log and sum(rows for rows, _digest in port_log) > 0
    assert port_log == py_log, "the server was sent other requests"
    assert _same(python, ported)
    assert port_after.to_json() == py_after.to_json()
    assert record.decisions[0] is ported, "the node recorded another decision"
    return port_log


@pytest.mark.parametrize("distinct", [False, True])
def test_a_hidden_bench_replacement_is_the_ports(replacement, server, flags_off, distinct) -> None:  # noqa: ANN001, F811
    log = _compare(
        replacement, server, shown=[frozenset({0, 1}), frozenset({0, 1})], distinct=distinct
    )
    assert len(log) > 2, "one request a completion at least"


def test_a_replacement_with_nothing_hidden_is_the_ports(replacement, server, flags_off) -> None:  # noqa: ANN001, F811
    _compare(replacement, server, shown=[SEEN_ALL, SEEN_ALL], distinct=False)


def test_a_fault_leaves_the_node_to_python(replacement, server, flags_off, monkeypatch) -> None:  # noqa: ANN001, F811
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    stub, address = server
    shown = [frozenset({0, 1}), frozenset({0, 1})]
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        want, want_after, _ = _run(reg, position, owed, sheets, shown, (leaf, leaf), replace=False)
        monkeypatch.setattr(
            rustnode.RustNode, "replacement_node", lambda self, request: {"fault": "injected"}  # noqa: ARG005
        )
        faults, replaced = portlp.COUNTS["faults"], portlp.COUNTS["replaced"]
        got, got_after, _ = _run(reg, position, owed, sheets, shown, (leaf, leaf), replace=True)
    assert portlp.COUNTS["faults"] == faults + 1 and portlp.COUNTS["replaced"] == replaced
    assert _same(want, got)
    assert got_after.to_json() == want_after.to_json()


def test_another_server_is_another_node(replacement, flags_off) -> None:  # noqa: ANN001, F811
    """The check above is not vacuous: values scaled by the server move the mixtures."""
    reg, position, owed, data = replacement
    sheets = _fixture_sheets(reg, position, data)
    shown = [frozenset({0, 1}), frozenset({0, 1})]
    decisions = []
    for scale in (1.0, 0.31):
        stub = Stub()
        original = Stub.__call__

        stub.__class__ = type(
            "Scaled", (Stub,), {"__call__": lambda self, a, r, s=scale, o=original: o(self, a, r) * s}
        )
        shut, address = serve({"stub": stub})
        try:
            with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
                decisions.append(
                    _run(reg, position, owed, sheets, shown, (leaf, leaf), replace=True)[0]
                )
        finally:
            shut.shutdown()
    assert decisions[0].search_value != decisions[1].search_value
