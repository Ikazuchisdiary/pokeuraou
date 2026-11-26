"""A node with nothing hidden in the port's whole-node request (IKA-457 stage 2.1, `portlp.EXACT`)
against the definition (`beliefnode._per_completion`, which fills in the port, scores in Python and
solves the games in another crossing).

What these hold, on the inference server's numpy stand-in of `test_portserved`, which keeps each
request's rows and the bytes of every array:

- **the same node**: with every slot seen each side's one completion is the position itself
  (`exact`, `root`), and the request of stage 1.4 -- the one fill, the stack, the server's values,
  the fold and the games -- gives the matrices, the Bayesian answers and the
  notes of `_per_completion`, to the bit. Turn one, and a mid-game node (a switch, a Mega, damage).
- **the same requests**: the rows the server is sent (their count and the bytes of every array) are
  the ones the definition sends.
- The positive control is the node the port took (`portlp.COUNTS["exact"]`) and the flag off taking
  none; the control that the comparison can fail is another server.
- A port that faults on the request leaves the node to `_per_completion`, as for a hidden bench.

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
def flags_off():  # noqa: ANN201
    was = (portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0],
           portlp.rustnode._SPAN_BLOCKS[0], dict(portlp.COUNTS))
    yield
    portlp.EXACT[0], portlp.NODE[0], portlp.STACK[0], portlp.FOLD[0], portlp.GEN[0] = was[:5]
    portlp.rustnode._SPAN_BLOCKS[0] = was[5]


def _node(reg, position, ours, theirs, spreads, leaf, *, exact):  # noqa: ANN001, ANN202
    portlp.EXACT[0] = False
    portlp.NODE[0] = False
    portlp.set_node(True)
    if exact:
        portlp.set_exact(True)
    return belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads, solve=(0, 1)
    )


def _same_matrices(a, b) -> bool:  # noqa: ANN001
    """The matrices to the bit, and the notes. `shared` and `redone` are left out: they are cells
    counted for the report alone (nothing reads them), and a node with nothing hidden counts them
    as the whole-node request does for a hidden one -- every clean cell of every completion shared
    (200 here) -- where `_per_completion` counts the second side's copy of the one position as
    shared and the first side's resolution as redone (100 and 100)."""
    same = all(
        len(a.matrices[s]) == len(b.matrices[s])
        and all(np.array_equal(x, y) for x, y in zip(a.matrices[s], b.matrices[s], strict=True))
        for s in (0, 1)
    )
    return same and a.unmodelled == b.unmodelled


def _same(definition, in_port, spreads) -> bool:  # noqa: ANN001
    """`_same_matrices`, and the games the port solved in the request against `solve_bayesian` of
    the definition's matrices (the definition solves them afterwards, in `belief_solve`)."""
    from pokeuraou.equilibrium import solve_bayesian

    if not _same_matrices(definition, in_port):
        return False
    assert set(in_port.solved) == {0, 1}, "the request solved no game: the check is vacuous"
    for side in (0, 1):
        weights = np.asarray([item.weight for item in spreads[1 - side]], dtype=np.float64)
        games = [m if side == 0 else -m.T for m in definition.matrices[side]]
        want = solve_bayesian(games, weights)
        got = in_port.solved[side]
        if not (got.value == want.value and np.array_equal(got.row_strategy, want.row_strategy)):
            return False
    return True


def _everything_seen(reg, position, sheet, seen):  # noqa: ANN001, ANN202
    spreads = {s: completions(reg, position, s, sheet, seen=seen[s]) for s in (0, 1)}
    assert all(len(items) == 1 and items[0].exact and items[0].position is position
               for items in spreads.values()), "a side still has a hidden slot"
    return spreads


def _compare(reg, position, ours, theirs, spreads, server, *, batch=8192):  # noqa: ANN001, ANN202
    stub, address = server
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20, batch_size=batch) as leaf:
        runs = []
        for exact in (False, True):
            del stub.log[:]
            before = portlp.COUNTS["exact"]
            got = _node(reg, position, ours, theirs, spreads, leaf, exact=exact)
            runs.append((got, list(stub.log), portlp.COUNTS["exact"] - before))
    (definition, definition_log, ran0), (in_port, port_log, ran1) = runs
    assert (ran0, ran1) == (0, 1), "the port did not take the node: the comparison is vacuous"
    assert port_log and sum(rows for rows, _digest in port_log) > 0
    assert port_log == definition_log, "the server was sent other requests"
    assert _same(definition, in_port, spreads)
    return port_log


@pytest.mark.parametrize("batch", [8192, 100])
def test_a_turn_one_node_with_nothing_hidden_is_the_ports(setup, server, flags_off, batch) -> None:  # noqa: ANN001, F811
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=10).actions
    theirs = narrow(reg, position, 1, limit=10).actions
    everything = frozenset(range(len(position.sides[0].pokemon)))
    spreads = _everything_seen(reg, position, sheet, {0: everything, 1: everything})
    log = _compare(reg, position, ours, theirs, spreads, server, batch=batch)
    if batch == 100:
        assert len(log) > 1, "the batch was not cut: the chunking is not held"


def test_a_midgame_node_with_nothing_hidden_is_the_ports(midgame, server, flags_off) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, _spreads, sheet, _seen = midgame
    everything = frozenset(range(len(position.sides[0].pokemon)))
    spreads = _everything_seen(reg, position, sheet, {0: everything, 1: everything})
    _compare(reg, position, ours, theirs, spreads, server)


def test_the_flag_off_takes_no_such_node(setup, server, flags_off) -> None:  # noqa: ANN001, F811
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=6).actions
    theirs = narrow(reg, position, 1, limit=6).actions
    everything = frozenset(range(len(position.sides[0].pokemon)))
    spreads = _everything_seen(reg, position, sheet, {0: everything, 1: everything})
    stub, address = server
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        before = portlp.COUNTS["exact"], portlp.COUNTS["nodes"]
        _node(reg, position, ours, theirs, spreads, leaf, exact=False)
        assert (portlp.COUNTS["exact"], portlp.COUNTS["nodes"]) == before


def test_a_fault_leaves_the_node_to_the_definition(setup, server, flags_off, monkeypatch) -> None:  # noqa: ANN001, F811
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=6).actions
    theirs = narrow(reg, position, 1, limit=6).actions
    everything = frozenset(range(len(position.sides[0].pokemon)))
    spreads = _everything_seen(reg, position, sheet, {0: everything, 1: everything})
    stub, address = server
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        want = _node(reg, position, ours, theirs, spreads, leaf, exact=False)

        def faulty(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
            raise portlp.PortFault("injected")

        from pokeuraou import beliefnode

        monkeypatch.setattr(beliefnode, "_node_in_port", faulty)
        faults, nodes = portlp.COUNTS["faults"], portlp.COUNTS["nodes"]
        got = _node(reg, position, ours, theirs, spreads, leaf, exact=True)
    assert portlp.COUNTS["faults"] == faults + 1 and portlp.COUNTS["nodes"] == nodes
    assert _same_matrices(want, got)


def test_the_comparison_can_fail(setup, flags_off) -> None:  # noqa: ANN001, F811
    """Another server's values are other matrices: the check above is not vacuous."""
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=10).actions
    theirs = narrow(reg, position, 1, limit=10).actions
    everything = frozenset(range(len(position.sides[0].pokemon)))
    spreads = _everything_seen(reg, position, sheet, {0: everything, 1: everything})
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
                nodes.append(_node(reg, position, ours, theirs, spreads, leaf, exact=True))
        finally:
            shut.shutdown()
    assert not _same_matrices(nodes[0], nodes[1])
    assert np.isfinite(nodes[0].matrices[0][0]).all()
