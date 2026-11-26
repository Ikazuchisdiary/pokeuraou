"""A belief node planned, filled, stacked, scored and folded in one request to the port (IKA-455
stage 1.4, `beliefNode`) against the node Python plans (stage 1.3's `belief_payoffs`).

What these hold, on the inference server's numpy stand-in of `test_portserved`, which keeps each
request's rows and the bytes of every array:

- **the same plan**: the port's `reaches_bench`, `_reaching`, `_borrow` and `_entering` lay the
  node's blocks out in Python's order and cut and patch them as Python does -- so the requests the
  server is sent are the same to the byte (the order of the blocks is the order of the rows), and
  the matrices, the Bayesian answers, the counts (`shared`, `redone`) and the notes are the same
  to the bit. Turn one, a mid-game node (a switch, a Mega, damage: cells borrowed from another
  completion's fill) and one where the side has mega evolved.
- **the same counts**: the encoder's note of the rows patched, and the rule the port says it
  encoded with, are counted as the Python road counts them.
- The positive control is the nodes the port planned (`portlp.COUNTS["nodes"]`) and that Python
  made no fill of its own; the control that the comparison can fail is another server.

No torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import portlp, rustnode
from pokeuraou.beliefnode import belief_payoffs
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.inference import RemoteValue, serve
from pokeuraou.narrow import narrow

from ._port import Budget
from .test_beliefnode import midgame, setup  # noqa: F401 - fixtures
from .test_portserved import Stub


@pytest.fixture(scope="module")
def server():  # noqa: ANN201
    stub = Stub()
    shut, address = serve({"stub": stub})
    yield stub, address
    shut.shutdown()


@pytest.fixture
def flags_off():  # noqa: ANN201
    was = (portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0],
           portlp.rustnode._SPAN_BLOCKS[0])
    yield
    portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0] = was[:4]
    portlp.rustnode._SPAN_BLOCKS[0] = was[4]


def _node(reg, position, ours, theirs, spreads, leaf, *, node):  # noqa: ANN001, ANN202
    portlp.NODE[0] = False
    portlp.set_stack(True)
    if node:
        portlp.set_node(True)
    return belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads, solve=(0, 1)
    )


def _same(a, b) -> bool:  # noqa: ANN001
    return (
        all(
            len(a.matrices[s]) == len(b.matrices[s])
            and all(np.array_equal(x, y) for x, y in zip(a.matrices[s], b.matrices[s], strict=True))
            for s in (0, 1)
        )
        and all(
            a.solved[s].value == b.solved[s].value
            and np.array_equal(a.solved[s].row_strategy, b.solved[s].row_strategy)
            for s in (0, 1)
        )
        and (a.shared, a.redone, a.unmodelled) == (b.shared, b.redone, b.unmodelled)
    )


KEYS = ("patched side=rebuilt", "rust can_mega=holder")


def _compare(reg, position, ours, theirs, spreads, server, *, batch=8192):  # noqa: ANN001, ANN202
    stub, address = server
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20, batch_size=batch) as leaf:
        used = leaf.encoder.used
        runs = []
        for node in (False, True):
            del stub.log[:]
            before = {k: used.get(k, 0) for k in KEYS}
            nodes_before = portlp.COUNTS["nodes"]
            fills_before = rustnode.PORT_WAITED[1]
            got = _node(reg, position, ours, theirs, spreads, leaf, node=node)
            runs.append((
                got, list(stub.log), {k: used.get(k, 0) - before[k] for k in KEYS},
                portlp.COUNTS["nodes"] - nodes_before, rustnode.PORT_WAITED[1] - fills_before,
            ))
    (stacked, stacked_log, stacked_notes, ran0, _), (planned, planned_log, planned_notes, ran1, _) = runs
    assert (ran0, ran1) == (0, 1), "the port did not plan the node: the comparison is vacuous"
    assert planned_log and sum(rows for rows, _digest in planned_log) > 0
    assert planned_log == stacked_log, "the server was sent other requests"
    assert _same(stacked, planned)
    assert planned_notes == stacked_notes and stacked_notes["patched side=rebuilt"] > 0
    return planned_log


@pytest.mark.parametrize("batch", [8192, 100])
def test_a_turn_one_node_is_planned_in_the_port(setup, server, flags_off, batch) -> None:  # noqa: ANN001, F811
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=10).actions
    theirs = narrow(reg, position, 1, limit=10).actions
    spreads = {s: completions(reg, position, s, sheet) for s in (0, 1)}
    log = _compare(reg, position, ours, theirs, spreads, server, batch=batch)
    if batch == 100:
        assert len(log) > 1, "the batch was not cut: the chunking is not held"


@pytest.mark.parametrize("spent", [False, True])
def test_a_midgame_node_is_planned_in_the_port(midgame, server, flags_off, spent) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, spreads, sheet, seen = midgame
    if spent:
        position = position.copy()
        position.sides[0].mega_used = True
        spreads = {s: completions(reg, position, s, sheet, seen=seen[s]) for s in (0, 1)}
    _compare(reg, position, ours, theirs, spreads, server)


def test_the_comparison_can_fail(setup, flags_off) -> None:  # noqa: ANN001, F811
    """Another server's values are other matrices: the check above is not vacuous."""
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=10).actions
    theirs = narrow(reg, position, 1, limit=10).actions
    spreads = {s: completions(reg, position, s, sheet) for s in (0, 1)}
    nodes = []
    for scale in (0.013, 0.02):
        stub = Stub()
        original = stub.__call__.__func__

        def call(arrays, rows, original=original, stub=stub, scale=scale):  # noqa: ANN001, ANN202
            return original(stub, arrays, rows) * (1.0 + scale)

        stub.__class__ = type("Scaled", (Stub,), {"__call__": lambda self, a, r, call=call: call(a, r)})
        shut, address = serve({"stub": stub})
        try:
            with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
                nodes.append(_node(reg, position, ours, theirs, spreads, leaf, node=True))
        finally:
            shut.shutdown()
    assert not _same(nodes[0], nodes[1])
