"""A port that answers a fault (IKA-455): the node is solved in Python, the fault is counted.

`beliefNode`, `beliefServed`, `beliefFold` and the LPs of stage 1.1 answer `{"fault": text}`
where the port cannot do what it was asked (a fill it does not hold, a request it cannot read).
That is not a reason to restart the process, which `error` is (`port.ask`), and not a reason to
stop the game: the caller does the same thing in Python. What these hold:

- **the answer is the Python road's, to the bit**, whichever request faults and at every level
  (the node, the stack, the fold, the games); the count of faults is the number that faulted.
- **a ticket the port does not hold is a fault**, not a crash: a node whose fills another node's
  reference fill cleared (a process shared by two threads, or one restarted in between) is
  answered by Python.
- the control that the counting can fail: with nothing injected, no fault is counted.

No torch: the server is `test_portserved`'s numpy stand-in.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import port, portlp, rustnode
from pokeuraou.beliefnode import belief_payoffs
from pokeuraou.budget import Budget
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.inference import RemoteValue, serve
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.search import belief_solve

from .test_beliefnode import setup  # noqa: F401 - fixture
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


def _inject(monkeypatch, kinds, *, node=False):  # noqa: ANN001, ANN202
    """The port answers a fault to a request of these kinds (`node`: to `beliefNode`)."""
    real_lp, real_node = rustnode.RustNode.lp, rustnode.RustNode.belief_node

    def lp(self, request):  # noqa: ANN001, ANN202
        if request.get("kind") in kinds or (request.get("kind") is None and "lp" in kinds):
            return {"fault": "injected"}
        return real_lp(self, request)

    def belief_node(self, request):  # noqa: ANN001, ANN202
        return {"fault": "injected"} if node else real_node(self, request)

    monkeypatch.setattr(rustnode.RustNode, "lp", lp)
    monkeypatch.setattr(rustnode.RustNode, "belief_node", belief_node)


def _same_node(a, b) -> bool:  # noqa: ANN001
    return (
        all(
            len(a.matrices[s]) == len(b.matrices[s])
            and all(np.array_equal(x, y) for x, y in zip(a.matrices[s], b.matrices[s], strict=True))
            for s in (0, 1)
        )
        and (a.shared, a.redone, a.unmodelled) == (b.shared, b.redone, b.unmodelled)
    )


def _hidden(setup):  # noqa: ANN001, ANN202, F811
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=10).actions
    theirs = narrow(reg, position, 1, limit=10).actions
    spreads = {s: completions(reg, position, s, sheet) for s in (0, 1)}
    return reg, position, ours, theirs, spreads


def _run(reg, position, ours, theirs, spreads, leaf, *, node, stack=True, fold=True):  # noqa: ANN001, ANN202
    portlp.NODE[0] = portlp.STACK[0] = portlp.FOLD[0] = portlp.GEN[0] = False
    portlp.rustnode._SPAN_BLOCKS[0] = False
    if fold:
        portlp.set_fold(True)
    if stack:
        portlp.set_stack(True)
    if node:
        portlp.set_node(True)
    return belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads, solve=(0, 1)
    )


@pytest.mark.parametrize(
    ("kinds", "node", "faults"),
    [
        ((), True, 1),  # the node: the stack road runs
        (("beliefServed",), True, 2),  # and then the stack faults: Python stacks, the port folds
        (("beliefServed", "beliefFold"), True, 3),  # and the fold: Python does all of it
    ],
)
def test_a_node_the_port_faults_on_is_the_python_roads(  # noqa: ANN001, PLR0913
    setup, server, flags_off, monkeypatch, kinds, node, faults  # noqa: F811
) -> None:
    reg, position, ours, theirs, spreads = _hidden(setup)
    stub, address = server
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        python = _run(reg, position, ours, theirs, spreads, leaf, node=False, stack=False, fold=False)
        before = portlp.COUNTS["faults"]
        _inject(monkeypatch, kinds, node=node)
        got = _run(reg, position, ours, theirs, spreads, leaf, node=True)
        counted = portlp.COUNTS["faults"] - before
    assert counted == faults, counted
    assert _same_node(python, got)


def test_nothing_injected_counts_nothing(setup, server, flags_off) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, spreads = _hidden(setup)
    _stub, address = server
    with RemoteValue(address, "stub", Encoder(reg), buffer_bytes=64 << 20) as leaf:
        before = portlp.COUNTS["faults"], portlp.COUNTS["nodes"]
        _run(reg, position, ours, theirs, spreads, leaf, node=True)
        assert portlp.COUNTS["faults"] == before[0]
        assert portlp.COUNTS["nodes"] == before[1] + 1, "the port did not plan the node"


def test_games_the_port_faults_on_are_solved_in_python(setup, flags_off, monkeypatch) -> None:  # noqa: ANN001, F811
    """Stage 1.1's games (`belief_solve`'s Bayesian games, one crossing): a fault is Python's
    solve, the same answers to the bit."""
    reg, position, ours, theirs, spreads = _hidden(setup)
    leaves = {0: HP_SHARE.batch, 1: HP_SHARE.batch}
    portlp.set_gen(False)
    python = belief_solve(reg, position, ours, theirs, spreads, leaves, budget=Budget.matrix())
    portlp.set_gen(True)
    _inject(monkeypatch, ("lp",))
    before = portlp.COUNTS["faults"]
    got = belief_solve(reg, position, ours, theirs, spreads, leaves, budget=Budget.matrix())
    assert portlp.COUNTS["faults"] - before == 1
    for side in (0, 1):
        assert got[side].value == python[side].value
        assert np.array_equal(got[side].strategy, python[side].strategy)


def test_the_qs_games_the_port_faults_on_are_solved_in_python(setup, flags_off, monkeypatch) -> None:  # noqa: ANN001, F811
    """`qrank.prefetch`'s games (stage 1.1): on a fault `rank` solves them itself."""
    from pokeuraou import qhead, qrank

    from .test_gen_port_lp import _MadeUpQ

    reg, _sheet, position = setup

    def rankings():  # noqa: ANN202
        model = _MadeUpQ()
        given = qrank.prefetch(reg, position, (position, position), model)
        return [
            qrank.q_ranking(reg, position, side, model, "q-nocover", given.get(side))(
                qhead.legal_pool(reg, position, side)
            )
            for side in (0, 1)
        ]

    portlp.set_gen(False)
    python = rankings()
    portlp.set_gen(True)
    _inject(monkeypatch, ("lp",))
    before = portlp.COUNTS["faults"]
    got = rankings()
    assert portlp.COUNTS["faults"] - before == 1
    for a, b in zip(python, got, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_ticket_the_port_does_not_hold_is_a_fault_and_not_a_crash(setup, flags_off) -> None:  # noqa: ANN001, F811
    """A node's fills are held until the next node's reference fill (or a restarted process)
    drops them: a request that names one then is answered by a fault -- the road back to Python
    -- and the process goes on (the next request is answered)."""
    reg, _sheet, _position = setup
    request = {
        "kind": "beliefServed",
        "server": {"address": "127.0.0.1:1", "model": "m", "shm": "s", "bytes": 1},
        "patch": {"can_mega": 0, "maxhp_scaled": 1, "hp_fraction": 2, "fainted": 3, "mega_used": 0,
                  "mega_available": 1, "alive_fraction": 2, "team_hp_fraction": 3, "hp_scale": 250.0},
        "parts": [{"ticket": 987654321, "rows": None}],
    }
    answer = port.ask(reg, lambda node: node.lp(request))
    assert "no fill held as 987654321" in answer["fault"], answer
    assert "error" not in answer
    # The process is the same one and answers on.
    assert port.ask(reg, lambda node: node.lp({"kind": "lp", "games": []}))["answers"] == []
