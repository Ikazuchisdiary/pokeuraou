"""A belief node folded in the port (IKA-455 stage 1.2, `rust/src/beliefnode.rs`) against the
Python loop it replaces.

What these hold:

- **the same matrices**: `belief_payoffs` with `portlp.FOLD` on gives every completion's matrix
  to the bit, for a turn-1 node and a mid-game node (a switch, a Mega, damage: the dirty cells
  of the reference's, of a completion's own fill and borrowed from another completion's).
  The positive controls are the folds the port reports, the `packed` spans and the borrowed
  cells it was sent; the control that the comparison can fail is a node scored by another leaf.
- **the same Bayesian answers**: the games solved in the same crossing are `solve_bayesian`'s,
  to the bit, for both sides.

No torch: the leaves are the stand-ins of `test_beliefnode`.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import beliefnode, portlp
from pokeuraou.beliefnode import belief_payoffs
from pokeuraou.encode import Encoder
from pokeuraou.equilibrium import solve_bayesian
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow

from ._port import Budget
from .test_beliefnode import _Leaf, _SlotLeaf, midgame, setup  # noqa: F401 - fixtures


@pytest.fixture
def fold_off():  # noqa: ANN201
    was_fold, was_gen, was_blocks = portlp.FOLD[0], portlp.GEN[0], portlp.rustnode._SPAN_BLOCKS[0]
    yield
    portlp.set_fold(was_fold)
    portlp.set_gen(was_gen)
    portlp.rustnode._SPAN_BLOCKS[0] = was_blocks


def _both(reg, position, ours, theirs, spreads, leaf, solve=(0, 1)):  # noqa: ANN001, ANN201
    budget = Budget.matrix()
    portlp.set_fold(False)
    portlp.rustnode._SPAN_BLOCKS[0] = False
    python = belief_payoffs(reg, position, ours, theirs, leaf, budget=budget, spreads=spreads)
    portlp.set_fold(True)
    ported = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=budget, spreads=spreads, solve=solve
    )
    return python, ported


def _same(a, b) -> bool:  # noqa: ANN001
    return all(
        len(a.matrices[s]) == len(b.matrices[s])
        and all(np.array_equal(x, y) for x, y in zip(a.matrices[s], b.matrices[s], strict=True))
        for s in (0, 1)
    )


def _games(node, spreads, side):  # noqa: ANN001, ANN202
    weights = np.asarray([item.weight for item in spreads[1 - side]], dtype=np.float64)
    return [m if side == 0 else -m.T for m in node.matrices[side]], weights


def test_a_turn_one_node_is_folded_to_the_bit(setup, fold_off) -> None:  # noqa: ANN001, F811
    reg, sheet, position = setup
    ours = narrow(reg, position, 0, limit=10).actions
    theirs = narrow(reg, position, 1, limit=10).actions
    spreads = {s: completions(reg, position, s, sheet) for s in (0, 1)}
    before = portlp.counts()["folded"]
    python, ported = _both(reg, position, ours, theirs, spreads, _Leaf(Encoder(reg)))
    assert portlp.counts()["folded"] > before, "the port did not fold: the comparison is vacuous"
    assert python.shared > 0 and _same(python, ported)
    assert (python.shared, python.redone) == (ported.shared, ported.redone)
    assert python.unmodelled == ported.unmodelled
    # Its games are the Python solves, to the bit.
    assert set(ported.solved) == {0, 1}
    for side in (0, 1):
        games, weights = _games(python, spreads, side)
        want = solve_bayesian(games, weights)
        got = ported.solved[side]
        assert got.value == want.value
        assert np.array_equal(got.row_strategy, want.row_strategy)
        assert all(np.array_equal(a, b) for a, b in zip(got.col_strategies, want.col_strategies, strict=True))
    # The comparison can fail: another leaf's matrices are other numbers.
    other, _ = _both(reg, position, ours, theirs, spreads, _SlotLeaf(Encoder(reg)))
    assert not _same(python, other)


def test_a_midgame_node_is_folded_to_the_bit(midgame, fold_off, monkeypatch) -> None:  # noqa: ANN001, F811
    reg, position, ours, theirs, spreads, _sheet, _seen = midgame
    packed: list[int] = []
    real = beliefnode._packed
    monkeypatch.setattr(beliefnode, "_packed", lambda spans: packed.append(len(spans)) or real(spans))
    python, ported = _both(reg, position, ours, theirs, spreads, _SlotLeaf(Encoder(reg)))
    assert python.shared > 0 and python.redone > 0
    # The borrowed-from-a-completion road ran (spans re-indexed to another block).
    assert packed, "no cell was borrowed from another completion: that road is not held"
    assert _same(python, ported)
    for side in (0, 1):
        games, weights = _games(python, spreads, side)
        want = solve_bayesian(games, weights)
        got = ported.solved[side]
        assert got.value == want.value and np.array_equal(got.row_strategy, want.row_strategy)
