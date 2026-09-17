"""Sharing one resolution across completions, and the claim that lets it.

The claim is that an unrevealed benched Pokemon takes no part in a turn it is not switched
into, so the cells where none of them reaches the field resolve identically in every
completion and their leaves differ only in who is on the bench. If that is wrong -- a
mechanic that counts party members, a feature that reads the bench -- the payoffs stay
plausible and are wrong, and nothing downstream could notice.

So the test is the definition: `belief_payoffs` against a matrix per completion resolved
from scratch, every cell, exactly. Not almost: a payoff that differs in its last places is
an equilibrium that differs, and the whole point of this path is that it changes nothing
but the time.

It needs the Rust port, because the fast path is only taken when the port is there to be
compared against. Without it the test skips rather than passing vacuously -- a green run
on a machine where the code under test never executed is worse than a red one.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.beliefnode import belief_payoffs, reaches_bench
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.regulation import load_regulation
from pokeuraou.resolve import Budget, batched_payoff
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster


@pytest.fixture(scope="module")
def setup():  # noqa: ANN201
    os.environ.setdefault("POKEURAOU_RUST_NODE", "1")
    if not rustnode.available():
        pytest.skip("the shared path is only taken with the port; nothing to compare")
    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    roster = load_roster("rizabanadohido")
    sheet = list(roster.sets)[:6]
    position = position_from_sets(reg, sheet[:4], sheet[:4])
    return reg, sheet, position


class _Leaf:
    """A deterministic stand-in for the learned net, scored from the encoding.

    It has to score `Encoded` rather than positions, because that is what selects the
    shared path -- an evaluator without `from_encoded` falls back and the test would
    compare the slow path with itself. The arithmetic is arbitrary and reproducible; what
    matters is that it reads every per-Pokemon array, so a patch that missed one shows up.
    """

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def from_encoded(self, encoded) -> np.ndarray:  # noqa: ANN001
        parts = (
            encoded.species.reshape(len(encoded), -1).sum(axis=1) * 0.7,
            encoded.ability.reshape(len(encoded), -1).sum(axis=1) * 0.3,
            encoded.item.reshape(len(encoded), -1).sum(axis=1) * 0.11,
            encoded.moves.reshape(len(encoded), -1).sum(axis=1) * 0.013,
            encoded.mon.reshape(len(encoded), -1).sum(axis=1).astype(np.float64) * 0.017,
            encoded.mask.reshape(len(encoded), -1).sum(axis=1).astype(np.float64) * 0.5,
        )
        raw = sum(parts)
        return 1.0 / (1.0 + np.exp(-(raw % 7.0) + 3.0))

    def __call__(self, positions: list) -> np.ndarray:
        return self.from_encoded(self.encoder.encode_positions(positions))


def _menus(reg, position, limit):  # noqa: ANN001, ANN202
    return (
        narrow(reg, position, 0, limit=limit).actions,
        narrow(reg, position, 1, limit=limit).actions,
    )


def test_the_shared_node_equals_a_matrix_per_completion(setup) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 10)
    leaf = _Leaf(Encoder(reg))
    budget = Budget.matrix()
    spreads = {
        0: completions(reg, position, 0, sheet),
        1: completions(reg, position, 1, sheet),
    }
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=budget, spreads=spreads
    )
    assert node.shared > 0, "the fast path did not run; the comparison would be vacuous"

    for side, items in spreads.items():
        built = node.matrices[1 - side]
        assert len(built) == len(items)
        for index, item in enumerate(items):
            expected, _notes = batched_payoff(
                reg, item.position, ours, theirs, leaf, budget=budget
            )
            np.testing.assert_allclose(
                built[index], expected, rtol=0, atol=0,
                err_msg=f"side {side} completion {index} ({', '.join(item.species)})",
            )


def test_it_shares_most_of_the_node(setup) -> None:  # noqa: ANN001
    """If almost nothing is shared the path is correct and pointless, and a change that
    made it so would otherwise pass every other test here."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 10)
    leaf = _Leaf(Encoder(reg))
    spreads = {
        0: completions(reg, position, 0, sheet),
        1: completions(reg, position, 1, sheet),
    }
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads
    )
    cells = len(ours) * len(theirs)
    per_completion = sum(len(items) for items in spreads.values()) * cells
    assert node.redone < 0.4 * per_completion, (
        f"{node.redone} of {per_completion} cell-resolutions repeated; "
        "the sharing is not paying for itself"
    )


def test_a_switch_into_a_hidden_slot_is_not_shared(setup) -> None:  # noqa: ANN001
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 24)
    hidden = {0: (2, 3), 1: (2, 3)}
    mask = reaches_bench(reg, ours, theirs, hidden)
    # 1-based in the action, 0-based in the slot. Spelt out here because comparing them
    # directly is what the first version did, and the mask was wrong in both directions.
    wanted = [
        j for j, action in enumerate(theirs)
        if any(index - 1 in hidden[1] for index in action.switch_indices)
    ]
    assert wanted, "this position offers no switch into a hidden slot to test with"
    for j in wanted:
        assert mask[:, j].all(), f"column {j} switches into a hidden slot and was shared"


def test_nothing_hidden_costs_nothing(setup) -> None:  # noqa: ANN001
    """Every slot seen means one completion a side and the old node, unchanged."""
    reg, sheet, position = setup
    ours, theirs = _menus(reg, position, 8)
    leaf = _Leaf(Encoder(reg))
    seen = frozenset({0, 1, 2, 3})
    spreads = {
        side: completions(reg, position, side, sheet, seen=seen) for side in (0, 1)
    }
    node = belief_payoffs(
        reg, position, ours, theirs, leaf, budget=Budget.matrix(), spreads=spreads
    )
    expected, _notes = batched_payoff(
        reg, position, ours, theirs, leaf, budget=Budget.matrix()
    )
    for side in (0, 1):
        assert len(node.matrices[side]) == 1
        np.testing.assert_allclose(node.matrices[side][0], expected, rtol=0, atol=0)
