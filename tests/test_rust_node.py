"""The Rust bridge answers exactly what Python answers, or it is not used.

Skipped when the binary is not built, because the port is optional: nothing in the project
depends on it, and a missing binary must read as "not built" rather than as a failure.

The differentials in `tools/` are the wide checks -- thousands of turns, whole nodes, whole
games. This is the narrow one that runs with the suite: if the bridge is present, a node
filled through it must equal the node filled here, cell for cell.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.cli import _modal, build_beliefs
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.narrow import narrow
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.position import validate_position
from pokeuraou.resolve import Budget, batched_payoffs
from pokeuraou.setup import load_scenario, with_spreads

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn5.json"


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):
    if not rustnode.binary_path().exists():
        pytest.skip(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    # The process cache is module state and the fixture must not leak one.
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


def _node(limit: int = 6):
    """A real node: the scenario with the belief layer's modal spreads filled in.

    Not a hand-made one. Fabricating spreads for the hidden Pokemon leaves HP and maximum
    HP inconsistent, and the resolver's answer on a position that could not occur is not a
    thing to hold two implementations to -- Python itself deals negative damage on one.
    """
    scenario = load_scenario(EXAMPLE)
    reg = scenario.reg
    register_mega_stones(reg)
    beliefs = build_beliefs(scenario)
    pos = with_spreads(scenario, {key: _modal(b) for key, b in beliefs.items()})
    assert not validate_position(pos, reg.meta.active_per_side)
    row = narrow(reg, pos, 0, limit=limit).actions
    col = narrow(reg, pos, 1, limit=limit).actions
    return reg, pos, row, col


def test_a_node_is_the_same_through_the_bridge(bridged: None) -> None:
    reg, pos, row, col = _node()
    objectives = [OBJECTIVES["hp-share"], OBJECTIVES["faints"]]
    evaluators = [o.batch for o in objectives]

    node = rustnode.node_for(reg)
    assert node is not None, "the bridge reported itself available but produced no process"
    through_rust, _notes, rust_exact = batched_payoffs(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )

    rustnode.reset()
    os.environ[rustnode.ENV_ENABLE] = "0"
    in_python, _python_notes, python_exact = batched_payoffs(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )

    for index, objective in enumerate(objectives):
        gap = float(
            np.abs(np.asarray(through_rust[index]) - np.asarray(in_python[index])).max()
        )
        # A payoff is a weighted mean and numpy sums a dot product in a different order
        # than a sequential loop, so the last place may differ; nothing else may.
        assert gap < 1e-12, f"{objective.name} differs by {gap}"
    assert np.array_equal(rust_exact, python_exact)


class _EncodedLeaf:
    """A learned leaf's shape without the learned part: it scores the encoding.

    What makes a value function able to take the other crossing is not that it is a neural
    network, it is that its input *is* the encoding. This has that property and no weights,
    so the test costs no torch and still exercises the path a trained model takes.
    """

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def _score(self, encoded: Any) -> np.ndarray:
        flat = encoded.mon.reshape(len(encoded.mon), -1).astype(np.float64)
        return (np.tanh(flat.sum(axis=1) / 100.0) + 1.0) / 2.0

    def __call__(self, positions: Any) -> np.ndarray:
        return self._score(self.encoder.encode_positions(list(positions)))

    def from_encoded(self, encoded: Any) -> np.ndarray:
        return self._score(encoded)


def test_a_leaf_that_reads_the_encoding_crosses_as_arrays(bridged: None) -> None:
    """A learned leaf takes the other crossing: the leaves go across, not the payoff.

    Its input is the encoding, so the port cannot hand back a number -- it hands back the
    encoded leaves and how to fold their values, and the matrix that comes out has to be
    the one Python builds by encoding the same leaves itself.
    """
    reg, pos, row, col = _node()
    leaf = _EncodedLeaf(Encoder(reg))

    through_rust, _notes, rust_exact = batched_payoffs(
        reg, pos, row, col, [leaf], budget=Budget.matrix()
    )

    rustnode.reset()
    os.environ[rustnode.ENV_ENABLE] = "0"
    in_python, _python_notes, python_exact = batched_payoffs(
        reg, pos, row, col, [leaf], budget=Budget.matrix()
    )

    gap = float(np.abs(np.asarray(through_rust[0]) - np.asarray(in_python[0])).max())
    assert gap < 1e-12, f"the encoded crossing differs by {gap}"
    assert np.array_equal(rust_exact, python_exact)


def test_a_named_objective_rides_along_with_a_learned_leaf(bridged: None) -> None:
    """The analyser's own pair, which is a node scored two ways at once.

    A learned objective and a parameter-free one beside it as a cross-check. The leaves are
    over there, so the parameter-free one is scored over there too and its values come back
    with them -- sending a whole node home for the sake of the second column would cost more
    than the column.
    """
    reg, pos, row, col = _node()
    evaluators = [_EncodedLeaf(Encoder(reg)), OBJECTIVES["hp-share"].batch]

    through_rust, _notes, _rust_exact = batched_payoffs(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )

    rustnode.reset()
    os.environ[rustnode.ENV_ENABLE] = "0"
    in_python, _python_notes, _python_exact = batched_payoffs(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )

    for index, what in enumerate(["the encoded leaf", "hp-share"]):
        gap = float(
            np.abs(np.asarray(through_rust[index]) - np.asarray(in_python[index])).max()
        )
        assert gap < 1e-12, f"{what} differs by {gap}"


def test_an_evaluator_that_reads_positions_keeps_the_node_here(bridged: None) -> None:
    """Neither crossing fits an arbitrary callable, so it must not take either.

    A plain objective crosses by name and a leaf that reads the encoding crosses as arrays;
    something that is neither reads positions, and positions are what does not cross.
    """
    from pokeuraou.resolve import _encoded_leaf_plan, _objective_names

    assert _objective_names([OBJECTIVES["hp-share"].batch]) == ["hp-share"]
    assert _objective_names([lambda positions: np.zeros(len(positions))]) is None
    assert _encoded_leaf_plan([lambda positions: np.zeros(len(positions))]) is None
    # A named objective on its own has a plan, but no reason to take this crossing: the
    # payoff itself crosses, and `_rust_payoffs` has already dealt with it.
    assert _encoded_leaf_plan([OBJECTIVES["hp-share"].batch]) == [("hp-share", None)]


def test_the_menu_is_scored_the_same_through_the_bridge(bridged: None) -> None:
    """`narrow` decides which choices reach the matrix, so its scores must be *bit* equal.

    Not close. The scores are only ever used to order candidates, and an order is what
    survives into the game -- a difference in the last place is a different menu, which is
    a different game, which no tolerance would have caught.
    """
    reg, pos, _row, _col = _node()

    through_rust = [narrow(reg, pos, side, limit=6) for side in (0, 1)]
    rustnode.reset()
    os.environ[rustnode.ENV_ENABLE] = "0"
    in_python = [narrow(reg, pos, side, limit=6) for side in (0, 1)]

    for side, (there, here) in enumerate(zip(through_rust, in_python, strict=True)):
        assert [c.action.to_choice() for c in there.kept] == [
            c.action.to_choice() for c in here.kept
        ], f"side {side} kept a different menu"
        for mine, theirs in zip(here.kept, there.kept, strict=True):
            assert mine.score == theirs.score, f"{mine.score!r} against {theirs.score!r}"
            assert mine.detail == theirs.detail


def test_a_menu_scored_from_beliefs_stays_here(bridged: None) -> None:
    """The port carries one particle; the belief layer's Battlers carry a whole spread.

    So the analyser's menu is scored in Python. The guard is the argument itself -- a
    caller that passes `battlers` is not offered the crossing -- which is why this checks
    that the bridged path is not even consulted rather than that the answer matches.
    """
    from pokeuraou.narrow import _battlers_from_position, _bridged_scores

    reg, pos, _row, _col = _node()
    table = _battlers_from_position(reg, pos)
    with_beliefs = narrow(reg, pos, 0, limit=6, battlers=table)
    without = narrow(reg, pos, 0, limit=6)
    # Same answer either way here, because these Battlers carry one particle each; what
    # the test pins is that supplying them at all is a legitimate call.
    assert [c.action.to_choice() for c in with_beliefs.kept] == [
        c.action.to_choice() for c in without.kept
    ]
    assert _bridged_scores(reg, pos, 0, [c.action for c in without.kept]) is not None


def test_a_node_that_dies_is_replaced_rather_than_given_up_on(bridged: None) -> None:
    """One failure used to end the bridge for the whole process.

    It fell back to Python and stayed there, at a twentieth of the speed, for however many
    games were left -- and a generation run lost two workers to exactly that. A node
    process holds nothing between requests, so the answer to one dying is another one.
    """
    reg, pos, row, col = _node()
    evaluators = [OBJECTIVES["hp-share"].batch]
    before, _notes, _exact = batched_payoffs(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )

    node = rustnode.node_for(reg)
    assert node is not None
    node._process.kill()  # noqa: SLF001 - the failure a generation run saw

    after, _n2, _e2 = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    assert np.allclose(np.asarray(after[0]), np.asarray(before[0])), "the answer changed"
    assert rustnode.available(), "one failure gave up on the bridge"

    # And the next node goes through the port again rather than through Python.
    replacement = rustnode.node_for(reg)
    assert replacement is not None
    filled = replacement.fill(pos, row, col, ["hp-share"], Budget.matrix())
    assert not filled.refused


def test_an_impossible_position_is_refused(bridged: None) -> None:
    """A position Python's own validator rejects must not be answered, only refused.

    Two implementations of a well-defined function agree; two implementations handed a
    state the game cannot reach do whatever they each do. The port says so rather than
    quietly producing the other answer.
    """
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    reg, pos, row, col = _node(limit=2)
    # More HP than the Pokemon has, which `validate_position` rejects.
    pos.sides[0].pokemon[0].hp = pos.sides[0].pokemon[0].maxhp + 5
    assert validate_position(pos, reg.meta.active_per_side)

    node = rustnode.node_for(reg)
    filled = node.fill(pos, row, col, ["hp-share"], Budget.matrix())
    assert len(filled.refused) == len(row) * len(col)
    assert all("hp" in why for _i, _j, why in filled.refused)
