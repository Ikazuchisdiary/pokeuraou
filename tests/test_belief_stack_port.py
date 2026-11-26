"""A belief node stacked, scored and folded in the port (IKA-455 stage 1.3, `beliefServed`)
against the node Python stacks (`beliefnode._stacked`) and the port folds (stage 1.2).

What these hold:

- **the same requests to the server**: the inference server here is the numpy stand-in of
  `test_portserved`, which keeps each request's rows and the bytes of every array. The port
  lays the node's blocks (held fills, cut and patched as `_patched` patches them) into the
  leaf's block itself; the requests it sends are the ones Python's stack sends, to the byte,
  and cut where `RemoteValue` cuts (a small `batch_size` too).
- **the same matrices and Bayesian answers**, to the bit, for a turn-1 node and a mid-game
  node (a switch, a Mega, damage) -- and one where the side has already mega evolved, so the
  patch clears `can_mega` and rebuilds the side vector from it.
- The positive controls are the nodes the port stacked (`portlp.COUNTS["stacked"]`) and the
  rows the server was asked for through it; the control that the comparison can fail is a node
  scored by another server.

No torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import portlp
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
def stack_off():  # noqa: ANN201
    was = (portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0], portlp.rustnode._SPAN_BLOCKS[0])
    yield
    portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0] = was[:3]
    portlp.rustnode._SPAN_BLOCKS[0] = was[3]


def _node(reg, position, ours, theirs, spreads, leaf, *, stack):  # noqa: ANN001, ANN202
    portlp.set_fold(True)
    portlp.STACK[0] = False
    if stack:
        portlp.set_stack(True)
    return belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads, solve=(0, 1)
    )


def _same(a, b) -> bool:  # noqa: ANN001
    same = all(
        len(a.matrices[s]) == len(b.matrices[s])
        and all(np.array_equal(x, y) for x, y in zip(a.matrices[s], b.matrices[s], strict=True))
        for s in (0, 1)
    )
    return same and all(
        a.solved[s].value == b.solved[s].value
        and np.array_equal(a.solved[s].row_strategy, b.solved[s].row_strategy)
        for s in (0, 1)
    )


PATCHED = "patched side=rebuilt"


def test_a_node_that_goes_back_to_python_counts_its_patches_once(  # noqa: ANN001
    setup, server, stack_off, monkeypatch  # noqa: F811
) -> None:
    """With stage 1.3 on but the port unable to take the node, Python stacks it and `_patched`
    alone counts the rows it patched (the descriptions made on the way count nothing)."""
    from pokeuraou import beliefnode

    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=10).actions
    theirs = narrow(reg, position, 1, limit=10).actions
    spreads = {s: completions(reg, position, s, sheet) for s in (0, 1)}
    stub, address = server
    counts = []
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        for stack, refuse in ((False, False), (True, True)):
            if refuse:
                monkeypatch.setattr(beliefnode, "_fold_request", lambda *a, **k: None)
            before = leaf.encoder.used.get(PATCHED, 0)
            _node(reg, position, ours, theirs, spreads, leaf, stack=stack)
            counts.append(leaf.encoder.used.get(PATCHED, 0) - before)
    assert counts[0] > 0 and counts[0] == counts[1], counts


def _compare(reg, position, ours, theirs, spreads, server, *, batch=8192):  # noqa: ANN001, ANN202
    stub, address = server
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20, batch_size=batch) as leaf:
        used = leaf.encoder.used
        del stub.log[:]
        patched_before = used.get(PATCHED, 0)
        folded = _node(reg, position, ours, theirs, spreads, leaf, stack=False)
        folded_log = list(stub.log)
        folded_patched = used.get(PATCHED, 0) - patched_before
        before = portlp.COUNTS.get("stacked", 0)
        del stub.log[:]
        patched_before = used.get(PATCHED, 0)
        stacked = _node(reg, position, ours, theirs, spreads, leaf, stack=True)
        stacked_log = list(stub.log)
        stacked_patched = used.get(PATCHED, 0) - patched_before
        ran = portlp.COUNTS.get("stacked", 0) - before
    assert ran == 1, "the port did not stack the node: the comparison is vacuous"
    # The rows patched are counted once on either road (`encoder.note`).
    assert folded_patched > 0 and stacked_patched == folded_patched
    assert folded_log and sum(rows for rows, _digest in stacked_log) > 0
    assert stacked_log == folded_log, "the server was sent other requests"
    assert _same(folded, stacked)
    assert (folded.shared, folded.redone) == (stacked.shared, stacked.redone)
    return stacked_log


@pytest.mark.parametrize("batch", [8192, 100])
def test_a_turn_one_node_is_stacked_in_the_port(setup, server, stack_off, batch) -> None:  # noqa: ANN001, F811
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=10).actions
    theirs = narrow(reg, position, 1, limit=10).actions
    spreads = {s: completions(reg, position, s, sheet) for s in (0, 1)}
    log = _compare(reg, position, ours, theirs, spreads, server, batch=batch)
    if batch == 100:
        assert len(log) > 1, "the batch was not cut: the chunking is not held"


@pytest.mark.parametrize("spent", [False, True])
def test_a_midgame_node_is_stacked_in_the_port(midgame, server, stack_off, spent) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, spreads, sheet, seen = midgame
    if spent:
        # The side has mega evolved: a patched bench row cannot hold a stone, and the side
        # vector is rebuilt from the rows that are left.
        position = position.copy()
        position.sides[0].mega_used = True
        encoder = Encoder(reg)
        used = encoder.side_names.index("mega_used")
        assert encoder.encode_positions([position], bind=False).side[0, 0, used] > 0
        spreads = {s: completions(reg, position, s, sheet, seen=seen[s]) for s in (0, 1)}
    _compare(reg, position, ours, theirs, spreads, server)


def test_the_comparison_can_fail(setup, stack_off) -> None:  # noqa: ANN001, F811
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
                nodes.append(_node(reg, position, ours, theirs, spreads, leaf, stack=True))
        finally:
            shut.shutdown()
    assert not _same(nodes[0], nodes[1])
