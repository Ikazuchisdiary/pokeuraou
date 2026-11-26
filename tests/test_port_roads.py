"""The port's roads are the only roads (IKA-460 stage 4).

A game's nodes (the hidden bench's node, the replacement node, the menus, the turn's advance)
are asked of the port on the inference server's leaf. What these hold, on `tests/_served.py`'s
stand-in server (no torch, no GPU):

- **a request the port faults on stops the game with its words** (`portlp.PortFault`: the
  request's name and what the port said), and is not solved again in Python. The control that
  the check can fail: with nothing injected, nothing is raised and the node ran in the port.
- **a leaf held in the process is refused** (`beliefnode.UnservedLeaf`) with a sentence that
  says what to start, where the old road scored it here.
- **the roads ran**: `portlp.COUNTS` moves on a node, the positive control the old flag tests
  read.

What the old tests of each stage compared against the Python road (to the bit) is gone with the
road: records/IKA-460.md lists each and what replaced it.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import beliefnode, port, portlp, rustnode
from pokeuraou.beliefnode import UnservedLeaf, belief_payoffs
from pokeuraou.budget import Budget
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE

from ._served import hp_share_leaf
from .test_beliefnode import setup  # noqa: F401 - fixture


def _node(setup):  # noqa: ANN001, ANN202, F811
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=6).actions
    theirs = narrow(reg, position, 1, limit=6).actions
    spreads = {s: completions(reg, position, s, sheet) for s in (0, 1)}
    return reg, position, ours, theirs, spreads


def test_the_node_is_the_ports(setup) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, spreads = _node(setup)
    before = dict(portlp.COUNTS)
    got = belief_payoffs(
        reg, position, ours, theirs, hp_share_leaf(), budget=Budget.matrix(), spreads=spreads,
        solve=(0, 1),
    )
    assert portlp.COUNTS["nodes"] == before["nodes"] + 1
    assert portlp.COUNTS["crossings"] == before["crossings"] + 1
    assert set(got.solved) == {0, 1}


def test_a_fault_of_the_port_stops_with_its_words(setup, monkeypatch) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, spreads = _node(setup)
    leaf = hp_share_leaf()
    # The control: nothing injected, nothing raised.
    belief_payoffs(reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads)
    monkeypatch.setattr(rustnode.RustNode, "belief_node", lambda self, request: {"fault": "injected"})
    with pytest.raises(portlp.PortFault, match="beliefNode.*injected") as caught:
        belief_payoffs(reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads)
    assert caught.value.where == "beliefNode" and caught.value.why == "injected"


def test_the_lps_fault_stops_too(setup, monkeypatch) -> None:  # noqa: ANN001, F811
    reg, *_ = setup
    games = [np.array([[0.5, 0.2], [0.1, 0.4]])]
    assert not isinstance(portlp.solve_many(reg, games)[0], Exception)
    monkeypatch.setattr(rustnode.RustNode, "lp", lambda self, request: {"fault": "injected"})
    with pytest.raises(portlp.PortFault, match="`lp`.*injected"):
        portlp.solve_many(reg, games)


def test_the_advance_of_a_turn_stops_on_a_fault(setup, monkeypatch) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, _spreads = _node(setup)
    rng = np.random.default_rng(1)
    chosen = [ours[0], theirs[0]]
    assert port.advance(reg, position, chosen, Budget.exact(), rng).index is not None
    monkeypatch.setattr(rustnode.RustNode, "advance_turn", lambda self, *a, **k: "injected")
    with pytest.raises(portlp.PortFault, match="advanceTurn.*injected"):
        port.advance(reg, position, chosen, Budget.exact(), rng)


@pytest.mark.unserved
def test_a_leaf_held_here_is_refused(setup) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, spreads = _node(setup)
    with pytest.raises(UnservedLeaf, match="inference.RemoteValue"):
        belief_payoffs(
            reg, position, ours, theirs, HP_SHARE.batch, budget=Budget.matrix(), spreads=spreads
        )
    with pytest.raises(UnservedLeaf):
        beliefnode.served_leaf(lambda positions: np.zeros(len(positions)))
    # A served leaf is accepted as itself.
    leaf = hp_share_leaf()
    assert beliefnode.served_leaf(leaf) is leaf
    assert beliefnode.served_leaf(leaf.__call__) is leaf
    assert isinstance(leaf.encoder, Encoder)
