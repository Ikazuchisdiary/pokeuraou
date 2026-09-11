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

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.cli import _modal, build_beliefs
from pokeuraou.damage import register_mega_stones
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


def test_a_learned_leaf_is_not_sent_across(bridged: None) -> None:
    """An evaluator that is not a plain objective must keep the node in Python.

    Its input is the leaves, and a node's leaves do not cross a process boundary. The
    guard is that `_objective_names` finds no name, so this checks the guard rather than
    the consequence.
    """
    from pokeuraou.resolve import _objective_names

    assert _objective_names([OBJECTIVES["hp-share"].batch]) == ["hp-share"]
    assert _objective_names([lambda positions: np.zeros(len(positions))]) is None


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
