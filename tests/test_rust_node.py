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

from pokeuraou import resolve as resolve_module
from pokeuraou import rustnode
from pokeuraou.actions import side_actions
from pokeuraou.cli import _modal, build_beliefs
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.narrow import narrow
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.position import validate_position
from pokeuraou.resolve import Budget, batched_payoffs, resolve_turn, turn_leaves
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


def test_asking_for_some_cells_answers_those_cells(bridged: None) -> None:
    """A restricted fill must equal the whole one, cell for cell, on both paths.

    This is what lets a caller solve a node without resolving all of it: the equilibrium
    needs about a fifth of a wide matrix, and the rest is work nobody reads. What must not
    happen is a cell answering differently because of who it was asked alongside.
    """
    reg, pos, row, col = _node()
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]
    wanted = [(0, 0), (1, 2), (2, 1), (0, 3)]
    wanted = [(i, j) for i, j in wanted if i < len(row) and j < len(col)]

    for bridge in ("1", "0"):
        os.environ[rustnode.ENV_ENABLE] = bridge
        rustnode.reset()
        whole, _n, whole_exact = batched_payoffs(
            reg, pos, row, col, evaluators, budget=Budget.matrix()
        )
        some, _n2, some_exact = batched_payoffs(
            reg, pos, row, col, evaluators, budget=Budget.matrix(), cells=wanted
        )
        for index in range(len(evaluators)):
            for i, j in wanted:
                assert some[index][i, j] == whole[index][i, j], (
                    f"cell {(i, j)} differs with the bridge {bridge}"
                )
                assert some_exact[i, j] == whole_exact[i, j]


def test_the_arrays_take_both_roads_and_are_the_same_bytes(bridged: None) -> None:
    """An encoded node's arrays cross through shared memory, or down the pipe, unchanged.

    Two things at once, and the first is why the second means anything. The header has to
    say which road it took -- `grow` for the node that asks for a block, `shm` once there
    is one, `pipe` for a process holding none -- because a run that quietly fell back to
    the pipe would pass an equality check by never testing anything. Then the arrays from
    the two roads have to be *identical*: the same function writes the same bytes in the
    same order into a different sink, and nothing about a node's value may depend on which.
    """
    reg, pos, row, col = _node()

    def both(blocks: bool) -> tuple[list[str], Any]:
        rustnode.reset()
        child = rustnode.node_for(reg)
        assert child is not None
        child._shm_off = not blocks  # noqa: SLF001 - the arm the test is here to pick
        roads: list[str] = []
        real = rustnode.EncodedNode.unpack

        def watch(header: dict, body: bytearray):  # noqa: ANN202
            roads.append(str(header.get("via")))
            return real(header, body)

        rustnode.EncodedNode.unpack = staticmethod(watch)
        try:
            # Twice: the first node of a process is the one that asks for a block, and
            # the second is the one that finds it already there.
            filled = None
            for _ in range(2):
                filled = child.fill_encoded(
                    pos, row, col, Budget.matrix(), ["hp-share"], None
                )
            return roads, filled
        finally:
            rustnode.EncodedNode.unpack = staticmethod(real)

    through_block, with_block = both(blocks=True)
    through_pipe, down_pipe = both(blocks=False)
    assert through_block == ["grow", "shm"], "the block was offered and never used"
    assert through_pipe == ["pipe", "pipe"], "a block was used by a process holding none"

    for name in ("species", "ability", "item", "moves", "mon", "mask", "side", "field"):
        mine = getattr(with_block.encoded, name)
        theirs = getattr(down_pipe.encoded, name)
        assert mine.dtype == theirs.dtype and mine.shape == theirs.shape
        assert np.array_equal(mine, theirs), f"{name} differs between the two roads"
    assert np.array_equal(
        with_block.leaf_values["hp-share"], down_pipe.leaf_values["hp-share"]
    )
    assert with_block.spans == down_pipe.spans
    assert with_block.folded == down_pipe.folded


def _references(tree: dict) -> int:
    """How many leaf references a fold tree holds -- one per `add_leaf` call that built it."""
    if "leaf" in tree:
        return 1
    if "best" in tree:
        return sum(_references(option) for option in tree["options"])
    return sum(_references(part) for _weight, part in tree["avg"])


def _cell_values(filled: Any) -> dict[tuple[int, int], float]:
    values = np.asarray(filled.leaf_values["hp-share"], dtype=np.float64)
    out = {
        (i, j): float(values[indices] @ np.asarray(weights))
        for i, j, indices, weights in filled.spans
        if weights
    }
    for i, j, root in filled.folded:
        out[(i, j)] = resolve_module.fold_value(resolve_module._fold_from_json(root), values)  # noqa: SLF001
    return out


def test_leaf_sharing_switched_off_stores_every_leaf_and_changes_no_cell(
    bridged: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`POKEURAOU_RUST_LEAF_SHARING=0` is IKA-62's positive control, so it has to be one.

    With sharing on, the node must keep fewer leaves than it was offered -- otherwise the
    switch below compares two identical runs and proves nothing. With it off, every offer
    is kept, and every cell folds to the same value: sharing is a storage decision only.
    """
    reg, pos, row, col = _node()

    def fill() -> Any:
        rustnode.reset()
        child = rustnode.node_for(reg)
        assert child is not None
        return child.fill_encoded(pos, row, col, Budget.matrix(), ["hp-share"], None)

    shared = fill()
    monkeypatch.setenv("POKEURAOU_RUST_LEAF_SHARING", "0")
    unshared = fill()

    offered = sum(len(indices) for _i, _j, indices, _w in shared.spans) + sum(
        _references(root) for _i, _j, root in shared.folded
    )
    assert len(shared.encoded.species) < offered, "sharing never fired on this node"
    assert len(unshared.encoded.species) == offered
    assert _cell_values(shared) == _cell_values(unshared)


def test_a_resumed_turn_is_resolved_on_the_turns_own_budget(bridged: None) -> None:
    """A turn paused by Parting Shot resumes at the resolution Python gives it (IKA-140).

    Every other test here runs `Budget.matrix()`, whose pinned roll is never narrowed, so
    none of them can see what budget a resumed turn gets. `Budget()` can: Kowtow Cleave's
    and Close Combat's rolls are live branches by the time Incineroar's Parting Shot pauses
    the turn, so that step runs on a narrowed budget. Python resumes the turn on the
    budget it was asked for; the port used to resume it on the narrowed one, and made 110
    leaves of this cell where Python makes 210.
    """
    reg, pos, _row, col = _node()
    ours = next(
        a for a in side_actions(reg, pos, 0) if a.to_choice() == "move 2 1, move 2 1"
    )
    theirs = next(a for a in col if a.to_choice() == "move 1 2, move 2")
    budget = Budget()

    python = resolve_turn(reg, pos, [ours, theirs], budget=budget)
    assert python.suspended, "this cell no longer pauses; it tests nothing"
    python_leaves = len(turn_leaves(reg, python).positions)

    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill_encoded(pos, [ours], [theirs], budget, ["hp-share"], None)
    assert not filled.refused
    assert [(i, j) for i, j, _root in filled.folded] == [(0, 0)]
    assert _references(filled.folded[0][2]) == python_leaves

    objectives = [OBJECTIVES["hp-share"], OBJECTIVES["faints"]]
    evaluators = [o.batch for o in objectives]
    through_rust, _notes, _exact = batched_payoffs(
        reg, pos, [ours], [theirs], evaluators, budget=budget
    )
    rustnode.reset()
    os.environ[rustnode.ENV_ENABLE] = "0"
    in_python, _python_notes, _python_exact = batched_payoffs(
        reg, pos, [ours], [theirs], evaluators, budget=budget
    )
    for index, objective in enumerate(objectives):
        gap = abs(float(through_rust[index][0][0]) - float(in_python[index][0][0]))
        assert gap < 1e-12, f"{objective.name} differs by {gap}"


def test_a_node_is_the_same_through_the_bridge_under_the_fast_budget(bridged: None) -> None:
    """A whole node under `Budget.fast()`, whose branch budget is divided as a turn unfolds.

    `Budget.matrix()` pins the roll and never narrows, so the node test above cannot see
    the path that narrows or the budget a paused turn resumes on (IKA-146). This node adds
    to the narrowed menu the rows and columns where Parting Shot pauses a turn after the
    rolls have already branched: IKA-140's line moved 12 cells of scenario-turn5's whole
    menu under this budget, and these rows and columns carry all 12 -- cell
    `move 2 1, move 2 1` / `move 1 2, move 1` by 5.3e-4 in hp-share and 0.036 in faints.
    """
    reg, pos, row, col = _node()
    menu = {0: side_actions(reg, pos, 0), 1: side_actions(reg, pos, 1)}

    def pick(side: int, choices: list[str]) -> list[Any]:
        return [next(a for a in menu[side] if a.to_choice() == c) for c in choices]

    def joined(narrowed: list[Any], added: list[Any]) -> list[Any]:
        return list({a.to_choice(): a for a in narrowed + added}.values())

    pausing_rows = ["move 2 1, move 2 1", "move 2 1, move 2 2", "move 2 2, move 2 1"]
    pausing_rows += ["move 2 2, move 2 2", "move 4 2, move 2 1", "move 4 2, move 2 2"]
    row = joined(row, pick(0, pausing_rows))
    col = joined(col, pick(1, ["move 1 2, move 1", "move 3 2, move 1"]))
    budget = Budget.fast()

    # What makes the node worth holding to: a cell that narrows before it pauses.
    cell = pick(0, ["move 2 1, move 2 1"]) + pick(1, ["move 1 2, move 1"])
    python_turn = resolve_turn(reg, pos, cell, budget=budget)
    assert python_turn.suspended, "this cell no longer pauses; it tests nothing"
    assert "resolution narrowed to fit the branch budget" in python_turn.reductions

    objectives = [OBJECTIVES["hp-share"], OBJECTIVES["faints"]]
    evaluators = [o.batch for o in objectives]
    through_rust, _notes, rust_exact = batched_payoffs(
        reg, pos, row, col, evaluators, budget=budget
    )
    rustnode.reset()
    os.environ[rustnode.ENV_ENABLE] = "0"
    in_python, _python_notes, python_exact = batched_payoffs(
        reg, pos, row, col, evaluators, budget=budget
    )
    for index, objective in enumerate(objectives):
        gap = np.abs(np.asarray(through_rust[index]) - np.asarray(in_python[index]))
        assert float(gap.max()) < 1e-12, (
            f"{objective.name} differs by {float(gap.max())} on {int((gap > 1e-12).sum())} cells"
        )
    # The exact mask as well (IKA-151). Under a stratified roll Python marks a turn inexact
    # for "damage rolls stratified"; the port had no such reduction and called every cell
    # of this node exact that nothing else made inexact. The comparison means something
    # only if the node has a cell the stratified roll is the *only* reason for.
    only_stratified = resolve_turn(reg, pos, [row[0], col[0]], budget=budget)
    assert set(only_stratified.reductions) == {"damage rolls stratified"}, (
        f"cell (0, 0) no longer tests the demotion alone: {only_stratified.reductions}"
    )
    rust_exact, python_exact = np.asarray(rust_exact), np.asarray(python_exact)
    assert not python_exact[0, 0]
    assert np.array_equal(rust_exact, python_exact), (
        f"the exact mask differs on {int((rust_exact != python_exact).sum())} of "
        f"{python_exact.size} cells ({int((rust_exact & ~python_exact).sum())} exact in "
        "the port only)"
    )


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


def test_the_refused_cells_are_filled_in_one_call(
    bridged: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node's refused cells are filled together, not one at a time.

    Measured on 2026-09-20: 6.0% of a generation's leaf rows were arriving in 94.9% of its
    forward passes, because each cell the port declined was filled by calling
    `batched_payoffs` on a 1x1 node of its own -- a forward pass for a handful of leaves,
    too small to amortise a kernel launch. `cells=` was already there; only using it was
    missing.

    The guard is the count of calls rather than the time, because the time belongs to the
    machine and the count is the property that made it slow.
    """
    reg, pos, row, col = _node()
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]

    # Slow Start is implemented here and refused there, so every cell comes back named
    # and the tail is the only thing that fills this node.
    mine = pos.sides[0].active_pokemon()[0]
    assert mine is not None
    mine.ability = "slowstart"
    assert not validate_position(pos, reg.meta.active_per_side)

    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    expected, _notes, expected_exact = batched_payoffs(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )

    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert len(filled.refused) == len(row) * len(col), "the port was meant to refuse these"

    asked: list[object] = []
    real = resolve_module.batched_payoffs

    def counting(*args: Any, **kwargs: Any):  # noqa: ANN202
        asked.append(kwargs.get("cells"))
        return real(*args, **kwargs)

    monkeypatch.setattr(resolve_module, "batched_payoffs", counting)
    rustnode.reset()
    got, _n2, got_exact = counting(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )

    assert len(asked) == 2, f"{len(asked) - 1} calls filled the tail, not 1"
    assert asked[0] is None
    assert asked[1] is not None and len(asked[1]) == len(row) * len(col)
    for index in range(len(evaluators)):
        # Bit-identical here, and it has to be: these cells were resolved and scored in
        # Python on both runs, so nothing summed anything in a different order.
        assert np.array_equal(np.asarray(got[index]), np.asarray(expected[index]))
    assert np.array_equal(np.asarray(got_exact), np.asarray(expected_exact))


def test_a_white_herb_holder_is_not_refused(bridged: None) -> None:
    """The effect crossed with the port; only the item list was never told about it.

    `check_white_herb` and its four call sites landed on 2026-09-12 and `item_handled` did
    not list the item, so every position a holder could be involved in was refused for an
    effect that was already implemented -- 63% of the cells generation refused.
    """
    reg, pos, row, col = _node()
    mine = pos.sides[0].active_pokemon()[0]
    assert mine is not None
    mine.item = "whiteherb"
    assert not validate_position(pos, reg.meta.active_per_side)
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]

    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert not [why for _i, _j, why in filled.refused if "whiteherb" in why]

    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    expected, _notes, _e = batched_payoffs(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    got, _n2, _e2 = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    for index in range(len(evaluators)):
        # Not bit-identical: a cell is a weighted mean and the port sums it in its own
        # order, which is worth about 1e-16. A wrong effect is worth 1e-03.
        assert np.allclose(
            np.asarray(got[index]), np.asarray(expected[index]), rtol=0, atol=1e-12
        )


def _turn_differences(node: Any, reg: Any, pos: Any, a: Any, b: Any) -> list[str]:
    """One cell resolved by both engines, compared branch by branch.

    The matrix is a weighted mean, so a branch weight that is wrong and a branch weight
    that is right can give the same cell whenever the branches score alike. This compares
    the thing itself: every weight, every suspended weight, the notes, and every branch's
    position -- the same equality `pokeuraou-damage turns` holds a fixture to.
    """
    budget = Budget.matrix()
    here = resolve_module.resolve_turn(reg, pos, [a, b], budget=budget)
    there = node.resolve(pos, [a, b], budget)
    if there is None:
        return ["the port refused the turn"]
    wrong: list[str] = []
    mine = [branch.probability for branch in here.branches]
    if len(mine) != len(there.branches) or any(
        abs(x - y) > 1e-12 for x, y in zip(mine, there.branches, strict=False)
    ):
        wrong.append(f"branch weights: python {mine}, rust {there.branches}")
    paused = [s.probability for s in here.suspended]
    if len(paused) != len(there.suspended) or any(
        abs(x - y) > 1e-12 for x, y in zip(paused, there.suspended, strict=False)
    ):
        wrong.append(f"suspended weights: python {paused}, rust {there.suspended}")
    if sorted(here.unmodelled) != sorted(there.unmodelled):
        wrong.append(f"notes: python {sorted(here.unmodelled)}, rust {sorted(there.unmodelled)}")
    if not wrong:
        for index, branch in enumerate(here.branches):
            chosen = node.resolve(pos, [a, b], budget, select=index)
            assert chosen is not None and chosen.position is not None
            if chosen.position.to_json() != branch.position.to_json():
                wrong.append(f"branch {index} position differs")
    return wrong


def test_a_quick_claw_holder_is_resolved_over_there_and_rolls_the_same(bridged: None) -> None:
    """The claw is `fractional_priority`'s, in both engines; only the gate was never told.

    A cell the claw does not move is no evidence -- its two queue branches reach the same
    states and merge back into one -- so the control comes first: the cells whose Python
    answer changes when the claw is taken away again. Those are held to the port branch by
    branch, because a wrong 20% can still average to the right cell (IKA-70).

    The holder is Incineroar at p1b and not Kingambit at p1a on purpose. The first action
    queued was the one place Python weighed the claw right, and the first version of this
    test sat there and passed while every other slot rolled it half the time.
    """
    reg, pos, row, col = _node()
    mine = pos.sides[0].active_pokemon()[1]
    assert mine is not None and mine.species == "incineroar"
    mine.item = "quickclaw"
    assert not validate_position(pos, reg.meta.active_per_side)
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]

    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    expected, _notes, _e = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    bare = pos.copy()
    bare.sides[0].active_pokemon()[1].item = None
    without, _n0, _e0 = batched_payoffs(reg, bare, row, col, evaluators, budget=Budget.matrix())
    moved = np.argwhere(np.abs(np.asarray(expected[0]) - np.asarray(without[0])) > 1e-12)
    assert len(moved), "the claw changed no cell, so agreeing here would say nothing"

    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert not filled.refused, f"refused: {sorted({why for _i, _j, why in filled.refused})}"
    got, _n2, _e2 = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    for index in range(len(evaluators)):
        assert np.allclose(
            np.asarray(got[index]), np.asarray(expected[index]), rtol=0, atol=1e-12
        )
    for i, j in moved:
        assert not _turn_differences(node, reg, pos, row[i], col[j]), (int(i), int(j))


def test_a_claw_holders_status_move_rolls_over_there_too(bridged: None) -> None:
    """Quick Claw fires on a status move in Showdown, and so in both engines (IKA-145).

    Both skipped the claw on every status move, so `tools/diff_node.py` agreed while both
    were wrong. The cells held here are Incineroar's Parting Shot with the claw, and the
    control comes first: the Python answer on them must move when the claw is taken away,
    or agreeing would say nothing -- before the fix neither engine rolled it there.
    """
    reg, pos, _row, col = _node()
    mine = pos.sides[0].active_pokemon()[1]
    assert mine is not None and mine.species == "incineroar"
    mine.item = "quickclaw"
    assert not validate_position(pos, reg.meta.active_per_side)
    row = [
        choice
        for choice in side_actions(reg, pos, 0)
        if getattr(choice.slots[1], "move_id", None) == "partingshot"
        and not getattr(choice.slots[0], "mega", False)
    ][:4]
    assert row, "no Parting Shot choice for the holder"
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]

    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    expected, _notes, _e = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    bare = pos.copy()
    bare.sides[0].active_pokemon()[1].item = None
    without, _n0, _e0 = batched_payoffs(reg, bare, row, col, evaluators, budget=Budget.matrix())
    moved = np.argwhere(np.abs(np.asarray(expected[0]) - np.asarray(without[0])) > 1e-12)
    assert len(moved), "the claw moved no Parting Shot cell, so agreeing would say nothing"

    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    for i, j in moved:
        assert not _turn_differences(node, reg, pos, row[i], col[j]), (int(i), int(j))
    got, _n2, _e2 = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    for index in range(len(evaluators)):
        assert np.allclose(
            np.asarray(got[index]), np.asarray(expected[index]), rtol=0, atol=1e-12
        )


def test_a_focus_band_holder_falls_the_same_way_over_there(bridged: None) -> None:
    """Neither engine branches the band's 1-in-10: both let the hit land and say so.

    Python reports `survival chance not branched:focusband` and faints the holder. The
    port used to refuse the turn at that point instead, which was a line nobody could reach
    while the gate refused every holder -- and the first thing listing the item would have
    made reachable. So the cells held to the port here are the ones that take that path:
    a lethal hit on the holder, which is what Close Combat into a Kingambit on 120 HP is.
    """
    reg, pos, row, col = _node()
    mine = pos.sides[0].active_pokemon()[0]
    assert mine is not None and mine.species == "kingambit"
    mine.item = "focusband"
    assert not validate_position(pos, reg.meta.active_per_side)
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]
    note = "survival chance not branched:focusband"

    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    expected, notes, _e = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    fired = [
        (i, j)
        for i in range(len(row))
        for j in range(len(col))
        if note
        in resolve_module.resolve_turn(reg, pos, [row[i], col[j]], budget=Budget.matrix()).unmodelled
    ]
    assert note in notes and fired, "no cell reached the band, so agreeing would say nothing"

    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert not filled.refused, f"refused: {sorted({why for _i, _j, why in filled.refused})}"
    assert note in filled.unmodelled
    got, rust_notes, _e2 = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    assert rust_notes == notes
    for index in range(len(evaluators)):
        assert np.allclose(
            np.asarray(got[index]), np.asarray(expected[index]), rtol=0, atol=1e-12
        )
    for i, j in fired:
        assert not _turn_differences(node, reg, pos, row[i], col[j]), (i, j)


@pytest.mark.parametrize("ability", ["disguise", "iceface"])
def test_disguise_and_ice_face_are_refused_by_name(bridged: None, ability: str) -> None:
    """The port zeroes the hit and nothing else, so a holder is refused, and says why.

    Python also busts the forme and takes Mimikyu's 1/8. An answer from the port would be
    a wrong one, which is worse than a refused cell -- Python fills those. The reason is
    held exactly: the gate's own refusal reads `ability: disguise`, so a test that only
    counted refusals would pass with the named check deleted, and would keep passing the
    day someone lists the ability in `ability_handled` (IKA-71).
    """
    reg, pos, row, col = _node()
    mine = pos.sides[0].active_pokemon()[0]
    assert mine is not None
    mine.ability = ability
    assert not validate_position(pos, reg.meta.active_per_side)

    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share"], Budget.matrix())
    assert len(filled.refused) == len(row) * len(col)
    assert {why for _i, _j, why in filled.refused} == {
        f"ability: {ability} (forme change and 1/8 not ported)"
    }


def test_stance_change_takes_the_forme_over_there_too(bridged: None) -> None:
    """The forme decides the stats, so a port that skipped it read the wrong Pokemon.

    The holder does not have to be Aegislash: neither implementation checks the species
    before changing the forme, so pinning the ability on whoever is in front tests exactly
    the arithmetic `change_forme` has to match -- species, types, maximum HP, and the HP
    carried across the change.
    """
    reg, pos, row, col = _node()
    mine = pos.sides[0].active_pokemon()[0]
    assert mine is not None
    mine.ability = "stancechange"
    assert not validate_position(pos, reg.meta.active_per_side)
    evaluators = [OBJECTIVES["hp-share"].batch, OBJECTIVES["faints"].batch]

    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert not [why for _i, _j, why in filled.refused if "stancechange" in why]

    os.environ[rustnode.ENV_ENABLE] = "0"
    rustnode.reset()
    expected, _notes, _e = batched_payoffs(
        reg, pos, row, col, evaluators, budget=Budget.matrix()
    )
    os.environ[rustnode.ENV_ENABLE] = "1"
    rustnode.reset()
    got, _n2, _e2 = batched_payoffs(reg, pos, row, col, evaluators, budget=Budget.matrix())
    for index in range(len(evaluators)):
        # Not bit-identical: a cell is a weighted mean and the port sums it in its own
        # order, which is worth about 1e-16. A wrong effect is worth 1e-03.
        assert np.allclose(
            np.asarray(got[index]), np.asarray(expected[index]), rtol=0, atol=1e-12
        )


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


def _after_the_stone_holder_switches_in():  # noqa: ANN202
    """Charizard, the side's only stone, starts at party slot 2 and is switched in.

    Resolved rather than assembled, because `_do_switch` is what renumbers the slots.
    """
    from pokeuraou.actions import MoveAction, SwitchAction, side_actions
    from pokeuraou.regulation import load_regulation
    from pokeuraou.resolve import resolve_turn
    from pokeuraou.selfplay import position_from_sets
    from pokeuraou.teams import load_roster

    reg = load_regulation("gen9championsvgc2026regmb")
    register_mega_stones(reg)
    sheet = {entry.species: entry for entry in load_roster("rizabanadohido").sets}
    own = [sheet[n] for n in ("venusaur", "sylveon", "charizard", "garchomp")]
    foe = [sheet[n] for n in ("incineroar", "toxapex", "garchomp", "venusaur")]
    before = position_from_sets(reg, own, foe)
    ours = next(
        action
        for action in side_actions(reg, before, 0)
        if isinstance(action.slots[0], SwitchAction)
        and action.slots[0].species == "charizard"
        and isinstance(action.slots[1], MoveAction)
        and action.slots[1].move_id == "detect"
    )
    theirs = next(
        action
        for action in side_actions(reg, before, 1)
        if all(
            isinstance(one, MoveAction) and one.move_id in ("protect", "banefulbunker")
            for one in action.slots
        )
    )
    turn = resolve_turn(reg, before, [ours, theirs], budget=Budget.exact())
    after = turn.branches[0].position
    assert after.sides[0].mega_capable_slots == [2]
    assert after.sides[0].pokemon[0].species == "charizard", "the premise: it moved"
    return reg, after


def test_the_port_puts_can_mega_on_the_stone_holder_after_a_switch(bridged: None) -> None:
    """`encode.rs` read `side.mega_capable_slots` as well, and was wrong the same way.

    Both implementations applied one rule, which is why `diff_encode.py` agreed with both
    while both were wrong (IKA-121). So the port is not compared with Python here: every
    leaf it encodes is asked whether `can_mega` stands exactly on the rows whose species
    and item are a mega pairing, on a side that has not spent its mega -- and whether
    `mega_available` says the same of the side. The node starts right after the holder
    came in, so its leaves have it off the number the side kept.
    """
    reg, pos = _after_the_stone_holder_switches_in()
    row = narrow(reg, pos, 0, limit=6).actions
    col = narrow(reg, pos, 1, limit=6).actions
    node = rustnode.node_for(reg)
    assert node is not None
    encoded = node.fill_encoded(pos, row, col, Budget.matrix(), [], None).encoded

    encoder = Encoder(reg)
    species_of = {index: sid for sid, index in encoder.vocab.species.items()}
    item_of = {index: iid for iid, index in encoder.vocab.items.items()}
    can_mega = encoder.mon_names.index("can_mega")
    is_mega = encoder.mon_names.index("is_mega")
    used = encoder.side_names.index("mega_used")
    available = encoder.side_names.index("mega_available")
    holder = encoder.vocab.species["charizard"]
    moved = 0
    for b in range(len(encoded)):
        for s in range(2):
            spent = bool(encoded.side[b, s, used])
            any_holder = False
            for p in range(encoder.mons_per_side):
                if not encoded.mask[b, s, p]:
                    continue
                sid = species_of.get(int(encoded.species[b, s, p]), "")
                holds = reg.mega_target(sid, item_of.get(int(encoded.item[b, s, p]))) is not None
                any_holder |= holds
                want = holds and not spent and not encoded.mon[b, s, p, is_mega]
                assert encoded.mon[b, s, p, can_mega] == float(want), (b, s, p, sid)
                moved += int(want and int(encoded.species[b, s, p]) == holder and p != 2)
            assert encoded.side[b, s, available] == float(any_holder and not spent), (b, s)
    assert moved, "no leaf had the holder off its old number, so nothing was tested"


def test_the_port_applies_the_old_can_mega_rule_only_when_asked(bridged: None) -> None:
    """`rules=` travels per request, so one node process serves both arms (IKA-141).

    The same node, asked twice from one process: under revision 1's rule `can_mega` stands
    on whatever row holds the side's recorded slot numbers, and the header says the rule was
    applied; asked without it, the current rule, as the IKA-121 test above checks in full.
    The positive control is that the two answers differ on this node at all.
    """
    from pokeuraou.encode import EncodingRules

    reg, pos = _after_the_stone_holder_switches_in()
    row = narrow(reg, pos, 0, limit=6).actions
    col = narrow(reg, pos, 1, limit=6).actions
    node = rustnode.node_for(reg)
    assert node is not None
    old = node.fill_encoded(
        pos, row, col, Budget.matrix(), [], None, rules=EncodingRules(mega_from_slots=True)
    )
    new = node.fill_encoded(pos, row, col, Budget.matrix(), [], None, rules=EncodingRules())
    plain = node.fill_encoded(pos, row, col, Budget.matrix(), [], None)
    assert old.mega_from_slots is True
    assert new.mega_from_slots is False and plain.mega_from_slots is False

    encoder = Encoder(reg)
    can_mega = encoder.mon_names.index("can_mega")
    is_mega = encoder.mon_names.index("is_mega")
    used = encoder.side_names.index("mega_used")
    available = encoder.side_names.index("mega_available")
    numbered = [list(side.mega_capable_slots) for side in pos.sides]
    for b in range(len(old.encoded)):
        for s in range(2):
            spent = bool(old.encoded.side[b, s, used])
            assert old.encoded.side[b, s, available] == float(bool(numbered[s]) and not spent)
            for p in range(encoder.mons_per_side):
                if not old.encoded.mask[b, s, p]:
                    continue
                want = p in numbered[s] and not spent and not old.encoded.mon[b, s, p, is_mega]
                assert old.encoded.mon[b, s, p, can_mega] == float(want), (b, s, p)
    assert not np.array_equal(old.encoded.mon, new.encoded.mon), "the rules agree here"
    np.testing.assert_array_equal(new.encoded.mon, plain.encoded.mon)
    moved = np.argwhere(old.encoded.mon != new.encoded.mon)
    assert set(moved[:, -1].tolist()) == {can_mega}


def test_two_leaves_in_one_process_each_get_their_own_rule(bridged: None) -> None:
    """The per-arm switch through `batched_payoffs`, as a match worker calls it (IKA-141).

    Two stand-in leaves in one process, one per rule, fill the same node. Each leaf's
    encoder books the rule the port *says* it applied, and the payoffs differ -- so the
    rule reached the port per leaf, and not once for the process.
    """
    from pokeuraou.encode import EncodingRules

    reg, pos = _after_the_stone_holder_switches_in()
    row = narrow(reg, pos, 0, limit=6).actions
    col = narrow(reg, pos, 1, limit=6).actions

    class Leaf:
        def __init__(self, encoder: Encoder) -> None:
            self.encoder = encoder

        def from_encoded(self, encoded: Any) -> np.ndarray:  # noqa: ANN401
            n = len(encoded)
            weights = np.linspace(0.1, 1.9, encoded.mon[0].size)
            raw = encoded.mon.reshape(n, -1).astype(np.float64) @ weights
            return 1.0 / (1.0 + np.exp(-(raw % 5.0) + 2.5))

        def __call__(self, positions: list) -> np.ndarray:
            return self.from_encoded(self.encoder.encode_positions(positions))

    old = Leaf(Encoder(reg, rules=EncodingRules(mega_from_slots=True)))
    new = Leaf(Encoder(reg))
    old_payoff, _notes, _exact = batched_payoffs(reg, pos, row, col, [old], budget=Budget.matrix())
    new_payoff, _notes, _exact = batched_payoffs(reg, pos, row, col, [new], budget=Budget.matrix())
    assert "rust can_mega=slots" in old.encoder.used, old.encoder.used
    assert "rust can_mega=holder" not in old.encoder.used, old.encoder.used
    assert "rust can_mega=holder" in new.encoder.used, new.encoder.used
    assert "rust can_mega=slots" not in new.encoder.used, new.encoder.used
    assert not np.array_equal(old_payoff[0], new_payoff[0])
    # A mixture that disagrees cannot share one fill, so the port is not asked for it.
    assert resolve_module._rust_encoded_payoffs(
        reg, pos, row, col, [old, new], Budget.matrix()
    ) is None
