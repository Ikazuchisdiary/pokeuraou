"""The port's node commands answer what the port's own turn answers, cell for cell.

A node reaches the port three ways (`pokeuraou.port.batched_payoffs`): a ported objective is
*filled* over there (`fill`, folded by `turn_value`), a learned leaf gets the node's leaves
*encoded* over there with the folds to take back (`fill_encoded`), and anything else is
resolved over there one cell at a time -- the `turn` command, the `alternatives` command for a
pause -- and folded here (`port.turn_leaves`, `port._scored_here`). Those are three pieces
of code, and the third is the definition: a cell is its turn's outcomes, weighted, with each
mid-turn replacement chosen by the side that owes it.

Until IKA-210 this file held the first two to Python's resolver. Python is going away
(IKA-204), so the reference is now the port's own turn: `fill` and `fill_encoded` against
`_scored_here`, and the `resolve` command's branches against the `turn` command's. The
differentials in `tools/` are the wide checks; this is the narrow one that runs with the
suite. The suite assumes the binary: without one every test here fails.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pokeuraou import port, rustnode
from pokeuraou.actions import side_actions
from pokeuraou.budget import Budget
from pokeuraou.cli import _modal, build_beliefs
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.fold import _fold_from_json, fold_value
from pokeuraou.narrow import narrow
from pokeuraou.payoff import OBJECTIVES
from pokeuraou.position import validate_position
from pokeuraou.setup import load_scenario, with_spreads

from ._port import resolve_turn

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scenario-turn5.json"


@pytest.fixture()
def bridged(monkeypatch: pytest.MonkeyPatch):
    if not rustnode.binary_path().exists():
        pytest.fail(f"no Rust binary at {rustnode.binary_path()}; `cargo build --release`")
    monkeypatch.setenv(rustnode.ENV_ENABLE, "1")
    # The process cache is module state and the fixture must not leak one.
    rustnode.reset()
    yield
    rustnode.reset()
    os.environ.pop(rustnode.ENV_ENABLE, None)


def _node(limit: int = 6):
    """A real node: the scenario with the belief layer's modal spreads filled in.

    Not a hand-made one. Fabricating spreads for the hidden Pokemon leaves HP and maximum
    HP inconsistent, and a resolver's answer on a position that could not occur is not a
    thing to hold two roads to.
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


def _by_turn(
    reg: Any, pos: Any, row: list, col: list, evaluators: list, budget: Budget,
    cells: list | None = None,
) -> tuple[list[np.ndarray], set[str], np.ndarray]:
    """The definition: every cell from the `turn` command (and `alternatives` for a pause),
    its leaves scored here by `evaluators` and folded here. `port.batched_payoffs`' road for
    an evaluator the port cannot score, taken on purpose for any evaluator."""
    return port._scored_here(reg, pos, row, col, evaluators, budget, cells)  # noqa: SLF001


def _filled(
    reg: Any, pos: Any, row: list, col: list, evaluators: list, budget: Budget,
    cells: list | None = None,
) -> tuple[list[np.ndarray], set[str], np.ndarray]:
    """`port.batched_payoffs`: `fill` for named objectives, `fill_encoded` for a leaf."""
    return port.batched_payoffs(reg, pos, row, col, evaluators, budget=budget, cells=cells)


def _assert_same(got: list, want: list, what: list[str], atol: float = 1e-12) -> None:
    for index, name in enumerate(what):
        gap = np.abs(np.asarray(got[index]) - np.asarray(want[index]))
        # A payoff is a weighted mean and the roads sum it in different orders, so the last
        # place may differ; nothing else may. A wrong effect is worth 1e-3.
        assert float(gap.max()) < atol, (
            f"{name} differs by {float(gap.max())} on {int((gap > atol).sum())} cells"
        )


def _objectives() -> tuple[list, list[str]]:
    objectives = [OBJECTIVES["hp-share"], OBJECTIVES["faints"]]
    return [o.batch for o in objectives], [o.name for o in objectives]


def _pausing_cells(reg: Any, pos: Any, row: list, col: list, budget: Budget) -> list:
    """The cells whose turn stops for a mid-turn replacement: where the folds differ from a
    plain mean, so a node that has none cannot tell a wrong fold from a right one."""
    return [
        (i, j)
        for i in range(len(row))
        for j in range(len(col))
        if port.turn(reg, pos, [row[i], col[j]], budget).suspended
    ]


def test_a_node_is_its_turns_folded(bridged: None) -> None:
    """`fill` against the definition, both objectives, every cell, and the exact mask."""
    reg, pos, row, col = _node()
    evaluators, names = _objectives()
    filled, notes, exact = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    turned, turn_notes, turn_exact = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    _assert_same(filled, turned, names)
    assert np.array_equal(exact, turn_exact)
    assert notes == turn_notes


#: Rows where Incineroar's Parting Shot pauses a turn with the rest of the queue still to
#: run (IKA-146's), so a resumed turn has something left to get wrong.
PAUSING_ROWS = [
    "move 2 1, move 2 1", "move 2 1, move 2 2", "move 2 2, move 2 1",
    "move 2 2, move 2 2", "move 4 2, move 2 1", "move 4 2, move 2 2",
]


def _with_parting_shot(reg: Any, pos: Any, row: list) -> list:
    """The narrowed row, plus the Parting Shot choices that pause a turn."""
    menu = {a.to_choice(): a for a in side_actions(reg, pos, 0)}
    extra = [menu[choice] for choice in PAUSING_ROWS]
    return list({a.to_choice(): a for a in [*row, *extra]}.values())


def _choice_matters(reg: Any, pos: Any, row: list, col: list, cells: list) -> bool:
    """Whether some pause among `cells` offers replacements worth different hp-shares: a
    node where every option is worth the same cannot tell choosing from averaging."""
    for i, j in cells:
        plan = port.turn_leaves(reg, port.turn(reg, pos, [row[i], col[j]], Budget.matrix(), full=True))
        values = OBJECTIVES["hp-share"].batch(plan.positions)
        for _weight, part in plan.root.parts:
            options = getattr(part, "options", None)
            if options and len({round(fold_value(o, values), 12) for o in options}) > 1:
                return True
    return False


def test_a_node_with_replacements_is_its_turns_folded(bridged: None) -> None:
    """`fill`'s `turn_value` folds a pause by the chooser's best option; so does the
    definition, over the `alternatives` command's turns. Both objectives, every cell.

    Incineroar is made the fastest on the field so its Parting Shot pauses the turn before
    the foes move: the replacement then takes their hits, and which one comes in is worth
    something. Paused last, every option is worth the same and a fold that averaged the
    bench would pass."""
    reg, pos, row, col = _node()
    pos.sides[0].active_pokemon()[1].boosts["spe"] = 6
    row = _with_parting_shot(reg, pos, row)
    evaluators, names = _objectives()
    cells = _pausing_cells(reg, pos, row, col, Budget.matrix())
    assert cells, "no cell pauses"
    assert _choice_matters(reg, pos, row, col, cells), "no pause offers a choice worth making"
    filled, notes, exact = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    turned, turn_notes, turn_exact = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    _assert_same(filled, turned, names)
    assert np.array_equal(exact, turn_exact)
    assert notes == turn_notes


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
    the one built here by encoding the `turn` command's leaves in Python. The row carries
    the Parting Shot cells, so the folds the port sends are exercised too.
    """
    reg, pos, row, col = _node()
    row = _with_parting_shot(reg, pos, row)
    leaf = _EncodedLeaf(Encoder(reg))
    encoded, _notes, exact = _filled(reg, pos, row, col, [leaf], Budget.matrix())
    turned, _turn_notes, turn_exact = _by_turn(reg, pos, row, col, [leaf], Budget.matrix())
    _assert_same(encoded, turned, ["the encoded leaf"])
    assert np.array_equal(exact, turn_exact)


def test_a_named_objective_rides_along_with_a_learned_leaf(bridged: None) -> None:
    """The analyser's own pair, which is a node scored two ways at once.

    A learned objective and a parameter-free one beside it as a cross-check. The leaves are
    over there, so the parameter-free one is scored over there too and its values come back
    with them -- sending a whole node home for the sake of the second column would cost more
    than the column.
    """
    reg, pos, row, col = _node()
    evaluators = [_EncodedLeaf(Encoder(reg)), OBJECTIVES["hp-share"].batch]
    encoded, _notes, _exact = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    turned, _turn_notes, _turn_exact = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    _assert_same(encoded, turned, ["the encoded leaf", "hp-share"])


def test_an_evaluator_that_reads_positions_keeps_the_node_here(bridged: None) -> None:
    """Neither crossing fits an arbitrary callable, so it must not take either.

    A plain objective crosses by name and a leaf that reads the encoding crosses as arrays;
    something that is neither reads positions, and positions are what does not cross.
    """
    assert port.objective_names([OBJECTIVES["hp-share"].batch]) == ["hp-share"]
    assert port.objective_names([lambda positions: np.zeros(len(positions))]) is None
    assert port.encoded_leaf_plan([lambda positions: np.zeros(len(positions))]) is None
    # A named objective on its own has a plan, but no reason to take this crossing: the
    # payoff itself crosses, by `fill`.
    assert port.encoded_leaf_plan([OBJECTIVES["hp-share"].batch]) == [("hp-share", None)]


def test_the_menu_is_scored_the_same_through_the_bridge(bridged: None) -> None:
    """`narrow` decides which choices reach the matrix, so its scores must be *bit* equal.

    Not close. The scores are only ever used to order candidates, and an order is what
    survives into the game -- a difference in the last place is a different menu, which is
    a different game, which no tolerance would have caught. (Switched off, `narrow` scores
    with `damage.calculate`, not with a resolver.)
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
    """A restricted fill must equal the whole one, cell for cell, on every road.

    This is what lets a caller solve a node without resolving all of it: the equilibrium
    needs about a fifth of a wide matrix, and the rest is work nobody reads. What must not
    happen is a cell answering differently because of who it was asked alongside.
    """
    reg, pos, row, col = _node()
    evaluators, _names = _objectives()
    wanted = [(0, 0), (1, 2), (2, 1), (0, 3)]
    wanted = [(i, j) for i, j in wanted if i < len(row) and j < len(col)]

    for road in (_filled, _by_turn):
        whole, _n, whole_exact = road(reg, pos, row, col, evaluators, Budget.matrix())
        some, _n2, some_exact = road(
            reg, pos, row, col, evaluators, Budget.matrix(), cells=wanted
        )
        for index in range(len(evaluators)):
            for i, j in wanted:
                assert some[index][i, j] == whole[index][i, j], (
                    f"cell {(i, j)} differs on {road.__name__}"
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
        out[(i, j)] = fold_value(_fold_from_json(root), values)
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
    """A turn paused by Parting Shot resumes at the resolution it was asked for (IKA-140).

    Every other test here runs `Budget.matrix()`, whose pinned roll is never narrowed, so
    none of them can see what budget a resumed turn gets. `Budget()` can: Kowtow Cleave's
    and Close Combat's rolls are live branches by the time Incineroar's Parting Shot pauses
    the turn, so that step runs on a narrowed budget. The turn resumes on the budget it was
    asked for; `fill_encoded` used to resume it on the narrowed one, and made 110 leaves of
    this cell where the turn makes 210. The reference is the `turn` command's pause resumed
    by the `alternatives` command and flattened here (`port.turn_leaves`).
    """
    reg, pos, _row, col = _node()
    ours = next(
        a for a in side_actions(reg, pos, 0) if a.to_choice() == "move 2 1, move 2 1"
    )
    theirs = next(a for a in col if a.to_choice() == "move 1 2, move 2")
    budget = Budget()

    whole = port.turn(reg, pos, [ours, theirs], budget, full=True)
    assert whole.pauses, "this cell no longer pauses; it tests nothing"
    turn_leaves = len(port.turn_leaves(reg, whole).positions)

    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill_encoded(pos, [ours], [theirs], budget, ["hp-share"], None)
    assert not filled.refused
    assert [(i, j) for i, j, _root in filled.folded] == [(0, 0)]
    assert _references(filled.folded[0][2]) == turn_leaves

    evaluators, names = _objectives()
    got, _notes, _exact = _filled(reg, pos, [ours], [theirs], evaluators, budget)
    want, _turn_notes, _turn_exact = _by_turn(reg, pos, [ours], [theirs], evaluators, budget)
    _assert_same(got, want, names)


def test_a_node_is_its_turns_folded_under_the_fast_budget(bridged: None) -> None:
    """A whole node under `Budget.fast()`, whose branch budget is divided as a turn unfolds.

    `Budget.matrix()` pins the roll and never narrows, so the node test above cannot see
    the path that narrows or the budget a paused turn resumes on (IKA-146). This node adds
    to the narrowed menu the rows and columns where Parting Shot pauses a turn after the
    rolls have already branched: IKA-140's line moved 12 cells of scenario-turn5's whole
    menu under this budget, and these rows and columns carry all 12.

    The exact mask as well (IKA-151): under a stratified roll a turn is inexact, and the
    mask has to say so on both roads. (Python's `reductions` named why; the port reports
    `exact` alone, so the premise is held as "some cell is inexact, and one that pauses".)
    """
    reg, pos, row, col = _node()
    menu = {0: side_actions(reg, pos, 0), 1: side_actions(reg, pos, 1)}

    def pick(side: int, choices: list[str]) -> list[Any]:
        return [next(a for a in menu[side] if a.to_choice() == c) for c in choices]

    def joined(narrowed: list[Any], added: list[Any]) -> list[Any]:
        return list({a.to_choice(): a for a in narrowed + added}.values())

    row = joined(row, pick(0, PAUSING_ROWS))
    col = joined(col, pick(1, ["move 1 2, move 1", "move 3 2, move 1"]))
    budget = Budget.fast()

    # What makes the node worth holding to: a cell that narrows before it pauses.
    cell = pick(0, ["move 2 1, move 2 1"]) + pick(1, ["move 1 2, move 1"])
    paused = port.turn(reg, pos, cell, budget)
    assert paused.suspended and not paused.exact, "this cell no longer narrows and pauses"

    evaluators, names = _objectives()
    got, notes, exact = _filled(reg, pos, row, col, evaluators, budget)
    want, turn_notes, turn_exact = _by_turn(reg, pos, row, col, evaluators, budget)
    _assert_same(got, want, names)
    assert notes == turn_notes
    exact, turn_exact = np.asarray(exact), np.asarray(turn_exact)
    assert not turn_exact.all(), "no cell is inexact, so the mask says nothing"
    assert np.array_equal(exact, turn_exact), (
        f"the exact mask differs on {int((exact != turn_exact).sum())} of "
        f"{turn_exact.size} cells ({int((exact & ~turn_exact).sum())} exact in fill only)"
    )


def test_a_node_that_dies_is_replaced_rather_than_given_up_on(bridged: None) -> None:
    """One failure used to end the bridge for the whole process.

    It fell back to Python and stayed there, at a twentieth of the speed, for however many
    games were left -- and a generation run lost two workers to exactly that. A node
    process holds nothing between requests, so the answer to one dying is another one
    (`port.ask`).
    """
    reg, pos, row, col = _node()
    evaluators = [OBJECTIVES["hp-share"].batch]
    before, _notes, _exact = _filled(reg, pos, row, col, evaluators, Budget.matrix())

    node = rustnode.node_for(reg)
    assert node is not None
    node._process.kill()  # noqa: SLF001 - the failure a generation run saw

    after, _n2, _e2 = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    assert np.array_equal(np.asarray(after[0]), np.asarray(before[0])), "the answer changed"
    assert rustnode.available(), "one failure gave up on the bridge"

    replacement = rustnode.node_for(reg)
    assert replacement is not None and replacement is not node
    filled = replacement.fill(pos, row, col, ["hp-share"], Budget.matrix())
    assert not filled.refused


def test_a_refused_cell_stops_the_node(bridged: None) -> None:
    """A cell the port refuses is a stop naming why, on every road (IKA-209).

    It used to be filled by Python's resolver, the whole tail in one call. There is none
    now, so a node with a refused cell raises rather than coming back partly filled or
    filled by something else. Ice Face is refused by name (its intact forme is in neither
    regulation); the control is the same node without it, which every road answers.
    """
    reg, pos, row, col = _node(limit=3)
    evaluators, _names = _objectives()
    leaf = _EncodedLeaf(Encoder(reg))
    for road, asked in ((_filled, evaluators), (_filled, [leaf]), (_by_turn, evaluators)):
        road(reg, pos, row, col, asked, Budget.matrix())

    mine = pos.sides[0].active_pokemon()[0]
    assert mine is not None
    mine.ability = "iceface"
    assert not validate_position(pos, reg.meta.active_per_side)
    for road, asked in ((_filled, evaluators), (_filled, [leaf]), (_by_turn, evaluators)):
        with pytest.raises(port.PortRefused, match="iceface"):
            road(reg, pos, row, col, asked, Budget.matrix())


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
    evaluators, names = _objectives()

    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert not [why for _i, _j, why in filled.refused if "whiteherb" in why]

    got, _n, _e = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    want, _n2, _e2 = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    _assert_same(got, want, names)


def _turn_differences(node: Any, reg: Any, pos: Any, a: Any, b: Any) -> list[str]:
    """One cell asked of two commands, compared branch by branch.

    The matrix is a weighted mean, so a branch weight that is wrong and a branch weight
    that is right can give the same cell whenever the branches score alike. This compares
    the thing itself: every weight, every suspended weight, the notes, and every branch's
    position -- from the `turn` command (what the rule tests and the definition read) and
    from the `resolve` command (what generation draws a game's next position from).
    """
    budget = Budget.matrix()
    here = port.turn(reg, pos, [a, b], budget, full=True)
    there = node.resolve(pos, [a, b], budget)
    if there is None:
        return ["the port refused the turn"]
    wrong: list[str] = []
    mine = [branch.probability for branch in here.outcomes]
    if len(mine) != len(there.branches) or any(
        abs(x - y) > 1e-12 for x, y in zip(mine, there.branches, strict=False)
    ):
        wrong.append(f"branch weights: turn {mine}, resolve {there.branches}")
    paused = [s.probability for s in here.pauses]
    if len(paused) != len(there.suspended) or any(
        abs(x - y) > 1e-12 for x, y in zip(paused, there.suspended, strict=False)
    ):
        wrong.append(f"suspended weights: turn {paused}, resolve {there.suspended}")
    if sorted(here.unmodelled) != sorted(there.unmodelled):
        wrong.append(f"notes: turn {sorted(here.unmodelled)}, resolve {sorted(there.unmodelled)}")
    if not wrong:
        for index, branch in enumerate(here.outcomes):
            chosen = node.resolve(pos, [a, b], budget, select=index)
            assert chosen is not None and chosen.position is not None
            if chosen.position.to_json() != branch.position.to_json():
                wrong.append(f"branch {index} position differs")
    return wrong


def _moved_by(reg: Any, pos: Any, bare: Any, row: list, col: list) -> np.ndarray:
    """The cells whose hp-share `fill` moves between `pos` and `bare`: the control that the
    thing taken away in `bare` is exercised by this node at all."""
    evaluators = [OBJECTIVES["hp-share"].batch]
    with_it, _n, _e = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    without, _n0, _e0 = _filled(reg, bare, row, col, evaluators, Budget.matrix())
    return np.argwhere(np.abs(np.asarray(with_it[0]) - np.asarray(without[0])) > 1e-12)


def test_a_quick_claw_holder_is_resolved_over_there_and_rolls_the_same(bridged: None) -> None:
    """The claw is `fractional_priority`'s; only the gate was never told (IKA-70).

    A cell the claw does not move is no evidence -- its two queue branches reach the same
    states and merge back into one -- so the control comes first: the cells whose answer
    changes when the claw is taken away again. Those are held branch by branch across the
    two commands, because a wrong 20% can still average to the right cell.

    The holder is Incineroar at p1b and not Kingambit at p1a on purpose. The first action
    queued was the one place Python weighed the claw right, and the first version of this
    test sat there and passed while every other slot rolled it half the time.
    """
    reg, pos, row, col = _node()
    mine = pos.sides[0].active_pokemon()[1]
    assert mine is not None and mine.species == "incineroar"
    mine.item = "quickclaw"
    assert not validate_position(pos, reg.meta.active_per_side)
    bare = pos.copy()
    bare.sides[0].active_pokemon()[1].item = None
    moved = _moved_by(reg, pos, bare, row, col)
    assert len(moved), "the claw changed no cell, so agreeing here would say nothing"

    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert not filled.refused, f"refused: {sorted({why for _i, _j, why in filled.refused})}"
    evaluators, names = _objectives()
    got, _n, _e = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    want, _n2, _e2 = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    _assert_same(got, want, names)
    for i, j in moved:
        assert not _turn_differences(node, reg, pos, row[i], col[j]), (int(i), int(j))


def test_a_claw_holders_status_move_rolls_over_there_too(bridged: None) -> None:
    """Quick Claw fires on a status move in Showdown, and so in the port (IKA-145).

    Both engines skipped the claw on every status move, so `tools/diff_node.py` agreed while
    both were wrong. The cells held here are Incineroar's Parting Shot with the claw, and
    the control comes first: the answer on them must move when the claw is taken away, or
    agreeing would say nothing.
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
    bare = pos.copy()
    bare.sides[0].active_pokemon()[1].item = None
    moved = _moved_by(reg, pos, bare, row, col)
    assert len(moved), "the claw moved no Parting Shot cell, so agreeing would say nothing"

    node = rustnode.node_for(reg)
    assert node is not None
    for i, j in moved:
        assert not _turn_differences(node, reg, pos, row[i], col[j]), (int(i), int(j))
    evaluators, names = _objectives()
    got, _n, _e = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    want, _n2, _e2 = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    _assert_same(got, want, names)


def test_a_focus_band_holder_falls_the_same_way_over_there(bridged: None) -> None:
    """The band's 1-in-10 is not branched: the hit lands and the turn says so.

    The port reports `survival chance not branched:focusband` and faints the holder. It
    used to refuse the turn at that point instead, which was a line nobody could reach
    while the gate refused every holder -- and the first thing listing the item would have
    made reachable. So the cells held here are the ones that take that path: a lethal hit
    on the holder, which is what Close Combat into a Kingambit on 120 HP is.
    """
    reg, pos, row, col = _node()
    mine = pos.sides[0].active_pokemon()[0]
    assert mine is not None and mine.species == "kingambit"
    mine.item = "focusband"
    assert not validate_position(pos, reg.meta.active_per_side)
    evaluators, names = _objectives()
    note = "survival chance not branched:focusband"
    fired = [
        (i, j)
        for i in range(len(row))
        for j in range(len(col))
        if note in port.turn(reg, pos, [row[i], col[j]], Budget.matrix()).unmodelled
    ]
    assert fired, "no cell reached the band, so agreeing would say nothing"

    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert not filled.refused, f"refused: {sorted({why for _i, _j, why in filled.refused})}"
    assert note in filled.unmodelled
    got, notes, _e = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    want, turn_notes, _e2 = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    assert notes == turn_notes
    _assert_same(got, want, names)
    for i, j in fired:
        assert not _turn_differences(node, reg, pos, row[i], col[j]), (i, j)


def _feint_node() -> tuple[Any, Any, list, list]:
    """Kingambit with Feint into a side that Protects, and one that puts up Wide Guard.

    The recorded menus have no Protect in the column, so a Feint there would break
    nothing and agreeing on it would say nothing (IKA-58). The row is every Feint choice
    of Kingambit's, the column Protect or Wide Guard choices of the foe's.
    """
    from pokeuraou.position import MoveSlot

    reg, pos, _row, _col = _node()
    mine = pos.sides[0].active_pokemon()[0]
    assert mine is not None and mine.species == "kingambit"
    mine.moves[0] = MoveSlot(id="feint", pp=16, maxpp=16)
    foe = pos.sides[1].active_pokemon()[1]
    assert foe is not None
    foe.moves[1] = MoveSlot(id="wideguard", pp=16, maxpp=16)
    assert not validate_position(pos, reg.meta.active_per_side)
    # Incineroar's hit has to follow Feint into the same foe, or the break exposes nothing
    # a payoff can see; one choice with a different target keeps the other kind of cell.
    feints = [
        choice
        for choice in side_actions(reg, pos, 0)
        if getattr(choice.slots[0], "move_id", None) == "feint"
        and getattr(choice.slots[1], "move_id", None) in ("flareblitz", "throatchop")
    ]
    row = [c for c in feints if c.slots[0].target == c.slots[1].target][:5] + [
        c for c in feints if c.slots[0].target != c.slots[1].target
    ][:1]
    guards = [
        choice
        for choice in side_actions(reg, pos, 1)
        if any(
            getattr(slot, "move_id", None) in ("protect", "wideguard") for slot in choice.slots
        )
    ]
    # Sneasler's Protect first. The other guard is Sinistcha's, a Ghost that Feint cannot
    # touch, so since IKA-153 a Feint into it breaks nothing.
    sneasler_protects = [
        c for c in guards if getattr(c.slots[0], "move_id", None) == "protect"
    ]
    col = sneasler_protects[:4] + [c for c in guards if c not in sneasler_protects][:4]
    assert row and col
    return reg, pos, row, col


def test_feint_breaks_the_guard_the_same_way_over_there(bridged: None) -> None:
    """`breaksProtect` (IKA-61), the same on every road.

    The port refused every Feint turn until IKA-61. The control comes first: the cells
    whose turn breaks a guard, read off the port's own trace (`... broke protect on p2a`,
    IKA-215) -- some move the payoff (a Protect broken, the hit lands), some only the
    position (Wide Guard gone, `stall` reset), which no payoff sees. Every one is held
    across the two commands branch by branch, since a wrong break can still average to the
    right cell.
    """
    reg, pos, row, col = _feint_node()
    fired = [
        (i, j)
        for i in range(len(row))
        for j in range(len(col))
        if any(
            " broke " in line
            for branch in resolve_turn(
                reg, pos, [row[i], col[j]], budget=Budget.matrix(), events=True
            ).branches
            for line in branch.events
        )
    ]
    assert fired, "no Feint broke a guard, so agreeing would say nothing"

    node = rustnode.node_for(reg)
    assert node is not None
    filled = node.fill(pos, row, col, ["hp-share", "faints"], Budget.matrix())
    assert not filled.refused, f"refused: {sorted({why for _i, _j, why in filled.refused})}"
    evaluators, names = _objectives()
    got, _n, _e = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    want, _n2, _e2 = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    _assert_same(got, want, names)
    for i, j in fired:
        assert not _turn_differences(node, reg, pos, row[i], col[j]), (i, j)


@pytest.mark.parametrize("ability", ["disguise", "iceface"])
def test_disguise_and_ice_face_are_refused_by_name(bridged: None, ability: str) -> None:
    """Ice Face is refused by name, and says why; Disguise is answered.

    Until IKA-208 both were refused: the port zeroed the hit and nothing else, where
    Python also busts the forme and takes Mimikyu's 1/8. The port busts it now
    (`moves::bust_disguise`, held to Showdown by test_disguise_order and
    test_disguise_afterhit). Ice Face stays refused, because its intact forme `eiscue` is
    in neither regulation. The reason is held exactly: the gate's own refusal would read
    `ability: iceface`, so a test that only counted refusals would pass with the named
    check deleted (IKA-71).
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
    if ability == "disguise":
        assert not filled.refused, sorted({why for _i, _j, why in filled.refused})
        return
    assert len(filled.refused) == len(row) * len(col)
    assert {why for _i, _j, why in filled.refused} == {
        f"ability: {ability} (its intact forme is not in the regulation)"
    }


def test_stance_change_takes_the_forme_over_there_too(bridged: None) -> None:
    """The forme decides the stats, so a road that skipped it read the wrong Pokemon.

    The holder does not have to be Aegislash: the port does not check the species before
    changing the forme, so pinning the ability on whoever is in front tests the arithmetic
    `change_forme` does -- species, types, maximum HP, and the HP carried across the
    change -- on `fill` and on the definition alike.
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

    got, _n, _e = _filled(reg, pos, row, col, evaluators, Budget.matrix())
    want, _n2, _e2 = _by_turn(reg, pos, row, col, evaluators, Budget.matrix())
    _assert_same(got, want, [o.name for o in (OBJECTIVES["hp-share"], OBJECTIVES["faints"])])


def test_an_impossible_position_is_refused(bridged: None) -> None:
    """A position `validate_position` rejects must not be answered, only refused.

    A state the game cannot reach has no right answer to hold anything to. The port says so
    rather than quietly producing one.
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

    Resolved rather than assembled, because the switch is what renumbers the slots.
    """
    from pokeuraou.actions import MoveAction, SwitchAction
    from pokeuraou.regulation import load_regulation
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
    turn = port.turn(reg, before, [ours, theirs], Budget.exact(), full=True)
    after = turn.outcomes[0].position
    assert after.sides[0].mega_capable_slots == [2]
    assert after.sides[0].pokemon[0].species == "charizard", "the premise: it moved"
    return reg, after


def test_the_port_puts_can_mega_on_the_stone_holder_after_a_switch(bridged: None) -> None:
    """`encode.rs` read `side.mega_capable_slots` as well, and was wrong the same way.

    Both implementations applied one rule, which is why `diff_encode.py` agreed with both
    while both were wrong (IKA-121). So the port is not compared with anything here: every
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
    old_payoff, _notes, _exact = _filled(reg, pos, row, col, [old], Budget.matrix())
    new_payoff, _notes, _exact = _filled(reg, pos, row, col, [new], Budget.matrix())
    assert "rust can_mega=slots" in old.encoder.used, old.encoder.used
    assert "rust can_mega=holder" not in old.encoder.used, old.encoder.used
    assert "rust can_mega=holder" in new.encoder.used, new.encoder.used
    assert "rust can_mega=slots" not in new.encoder.used, new.encoder.used
    assert not np.array_equal(old_payoff[0], new_payoff[0])
    # A mixture that disagrees cannot share one fill, so the port is not asked for one:
    # the node is resolved over there and scored here, each leaf by its own encoder.
    real = port._encoded  # noqa: SLF001

    def refuse(*_args: Any) -> None:
        raise AssertionError("a mixture of rules was sent to one fill_encoded")

    port._encoded = refuse  # noqa: SLF001
    try:
        mixed, _notes, _exact = _filled(reg, pos, row, col, [old, new], Budget.matrix())
    finally:
        port._encoded = real  # noqa: SLF001
    _assert_same(mixed, [old_payoff[0], new_payoff[0]], ["revision 1", "current"])
