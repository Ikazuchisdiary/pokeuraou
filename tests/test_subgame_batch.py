"""A depth-2 pass scores all its sub-games in one call to the leaf (IKA-291).

`_refined_value` called the leaf once per sub-game, so a hidden depth-2 decision paid a
call per refined cell and branch (438 a game in M-C generation against 45 at depth 1,
IKA-111). `_refine_cells` fills every sub-game of the pass first and scores them together.

What must not move is any value. A learned leaf's answer for a row depends on how many
rows share its call (on the card, 37 different answers for the same eight rows between 8 and
988 rows), so the leaf here does the same on purpose: every value carries a trace of its
batch's size. The batched pass equals the per-sub-game pass to the bit only if every
sub-game is still scored as a block of its own size -- and the control that stacks the
blocks into one call shows that the comparison sees it when they are not.

Until IKA-463 the pass was read through the fixed depth-2 search (open and hidden); that
search is gone, so the tests call `_refine_cells` as the ladder does.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import port
from pokeuraou import search as search_mod
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from ._port import resolve_turn


class _SizedLeaf:
    """A deterministic stand-in for the net whose answers depend on the batch's size.

    It reads every array the net reads (as `test_beliefnode._Leaf`), and adds 1e-9 per row
    of the call, the way a card's answer moves with the number of rows in its call. It
    counts the calls it is asked, by kind.
    """

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.calls = {"from_encoded": 0, "segments": 0, "positions": 0}
        self.blocks = 0

    def _score(self, encoded) -> np.ndarray:  # noqa: ANN001
        n = len(encoded.species)
        parts = (
            encoded.species.reshape(n, -1).sum(axis=1) * 0.7,
            encoded.ability.reshape(n, -1).sum(axis=1) * 0.3,
            encoded.item.reshape(n, -1).sum(axis=1) * 0.11,
            encoded.moves.reshape(n, -1).sum(axis=1) * 0.013,
            encoded.mon.reshape(n, -1).sum(axis=1).astype(np.float64) * 0.017,
            encoded.mask.reshape(n, -1).sum(axis=1).astype(np.float64) * 0.5,
            encoded.side.reshape(n, -1).astype(np.float64)
            @ np.linspace(0.3, 1.7, encoded.side[0].size),
            encoded.field.reshape(n, -1).astype(np.float64)
            @ np.linspace(0.2, 1.1, encoded.field.shape[1]),
        )
        raw = sum(parts)
        return 1.0 / (1.0 + np.exp(-(raw % 7.0) + 3.0)) * (1.0 - 1e-6) + n * 1e-9

    def from_encoded(self, encoded) -> np.ndarray:  # noqa: ANN001
        self.calls["from_encoded"] += 1
        return self._score(encoded)

    def from_encoded_segments(self, segments) -> list[np.ndarray]:  # noqa: ANN001
        self.calls["segments"] += 1
        self.blocks += len(segments)
        return [self._score(segment) for segment in segments]

    def __call__(self, positions: list) -> np.ndarray:
        self.calls["positions"] += 1
        return self._score(self.encoder.encode_positions(positions))

    def total(self) -> int:
        return sum(self.calls.values())


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _per_sub_game(reg, cells, evaluate, *, budget, sub_limit, sub_branches, shares=None, child_q=None):  # noqa: ANN001, ANN202, ARG001, E501
    """The pass as it was before IKA-291: `_refined_value` per cell, a call per sub-game."""
    assert child_q is None, "the per-sub-game pass builds children by damage only (IKA-362)"
    return [
        search_mod._refined_value(
            reg, pos, ours, theirs, evaluate, budget=budget,
            sub_limit=sub_limit, sub_branches=sub_branches,
        )
        for pos, ours, theirs in cells
    ]


def _stacked_scoring(evaluate, segments):  # noqa: ANN001, ANN202
    """The fault the batching must not make: every block in one call of the summed size."""
    from pokeuraou.beliefnode import _stacked

    stacked, starts = _stacked([(len(s.species), (lambda s=s: s)) for s in segments], segments[0])
    values = np.asarray(evaluate.__self__.from_encoded(stacked), dtype=np.float64)
    return [values[start : start + len(s.species)] for start, s in zip(starts, segments, strict=True)]


def _positions(roster) -> list:  # noqa: ANN001
    """Turn 1 and a few turns into one game of two different fours."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    pos = position_from_sets(reg, sheet[:4], sheet[2:6])
    rng = np.random.default_rng(11)
    out = []
    for _ in range(3):
        if pos.ended:
            break
        out.append(pos.copy())
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        result = resolve_turn(
            reg, pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        pos = max(result.branches, key=lambda b: b.probability).position
    return out


def _hidden(roster, leaf, pos, rect: int = 3) -> list:  # noqa: ANN001
    """One `_refine_cells` pass, as a hidden-bench read asks it (the ladder; until IKA-463
    `search._restricted_belief`): the ``rect`` x ``rect`` corner of the menus in every
    completion of side 1's bench, each cell with its `TurnShare` across the completions.
    Each cell's value, its notes and its count of sub-games."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    ours = narrow(reg, pos, 0, limit=6).actions
    theirs = narrow(reg, pos, 1, limit=6).actions
    turns: dict = {}
    clean: dict = {}
    cells, shares = [], []
    for item in completions(reg, pos, 1, sheet):
        for i in range(min(rect, len(ours))):
            for j in range(min(rect, len(theirs))):
                cells.append((item.position, ours[i], theirs[j]))
                if item.exact or not item.slots:
                    shares.append(None)
                    continue
                if (i, j) not in clean:
                    clean[(i, j)] = not search_mod._cell_reaches_bench(
                        reg, ours[i], theirs[j], 1, tuple(item.slots), item.position
                    )
                shares.append(search_mod.TurnShare(
                    cache=turns, key=(i, j), side=1, slots=tuple(item.slots), clean=clean[(i, j)],
                ))
    found = search_mod._refine_cells(
        reg, cells, leaf.__call__, budget=Budget.matrix(), shares=shares,
        sub_limit=search_mod.DEFAULT_SUB_LIMIT, sub_branches=search_mod.DEFAULT_SUB_BRANCHES,
    )
    return [(value, sorted(notes), solved) for value, notes, solved in found]


_pass = _hidden


def _all_answers(roster, positions) -> tuple[list, int, int]:  # noqa: ANN001
    """The pass's answers on every position, and the leaf's calls and sub-games."""
    leaf = _SizedLeaf(Encoder(roster.reg))
    answers: list = []
    subgames = 0
    for pos in positions:
        got = _pass(roster, leaf, pos)
        subgames += sum(solved for _value, _notes, solved in got)
        answers.append(got)
    return answers, leaf.total(), subgames


def test_the_batched_pass_is_the_per_sub_game_pass_to_the_bit(roster, monkeypatch) -> None:  # noqa: ANN001
    """A pass over the open position's cells and its completions' cells, on turn 1 and later
    turns: values, notes and counts are the same as `_refined_value` per cell gives -- with a
    leaf whose answers move with batch size."""
    positions = _positions(roster)
    assert len(positions) >= 2
    batched, _calls, subgames = _all_answers(roster, positions)
    assert subgames > 20, "the pass must actually refine for this to test anything"
    monkeypatch.setattr(search_mod, "_refine_cells", _per_sub_game)
    reference, _calls, _subgames = _all_answers(roster, positions)
    assert batched == reference


def test_stacking_the_sub_games_into_one_call_is_caught(roster, monkeypatch) -> None:  # noqa: ANN001
    """The positive control: score the pass's blocks as one batch -- the cheaper thing the
    issue first asked for -- and the answers are not the per-sub-game ones."""
    pos = _positions(roster)[0]
    reference = _pass(roster, _SizedLeaf(Encoder(roster.reg)), pos)
    monkeypatch.setattr(port, "score_segments", _stacked_scoring)
    stacked = _pass(roster, _SizedLeaf(Encoder(roster.reg)), pos)
    assert stacked != reference


def test_a_pass_calls_the_leaf_once(roster, monkeypatch) -> None:  # noqa: ANN001
    """The count, with its positive control: the per-sub-game pass (the code before
    IKA-291) calls the leaf once per sub-game; the batched pass once, whatever the number of
    sub-games in it (a few more where it holds more than `GATHER_ROWS` rows)."""
    pos = _positions(roster)[0]
    batched_leaf = _SizedLeaf(Encoder(roster.reg))
    subgames = sum(solved for _value, _notes, solved in _pass(roster, batched_leaf, pos))

    monkeypatch.setattr(search_mod, "_refine_cells", _per_sub_game)
    before_leaf = _SizedLeaf(Encoder(roster.reg))
    _pass(roster, before_leaf, pos)

    assert subgames >= 20
    # Before: one call per sub-game and none batched.
    assert before_leaf.calls["segments"] == 0
    assert before_leaf.calls["from_encoded"] >= subgames
    # After: the pass's sub-games in a few calls of segments, and no call of its own.
    assert batched_leaf.calls["segments"] * 5 < subgames
    assert batched_leaf.blocks >= subgames
    assert batched_leaf.calls["from_encoded"] + batched_leaf.calls["positions"] == 0
    assert batched_leaf.total() * 5 < before_leaf.total()


def test_a_pass_cut_by_gather_rows_is_the_same_pass(roster, monkeypatch) -> None:  # noqa: ANN001
    """A pass whose sub-games are scored in several calls (the memory cap) is the same pass:
    a sub-game is its own block wherever the calls are cut."""
    pos = _positions(roster)[0]
    whole = _pass(roster, _SizedLeaf(Encoder(roster.reg)), pos)
    monkeypatch.setattr(search_mod, "GATHER_ROWS", 1)
    leaf = _SizedLeaf(Encoder(roster.reg))
    cut = _pass(roster, leaf, pos)
    assert cut == whole
    assert leaf.calls["segments"] > 4
