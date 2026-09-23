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
