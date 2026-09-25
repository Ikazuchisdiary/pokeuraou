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
from pokeuraou.search import belief_solve, search
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


def _per_sub_game(reg, cells, evaluate, *, budget, sub_limit, sub_branches):  # noqa: ANN001, ANN202
    """The pass as it was before IKA-291: `_refined_value` per cell, a call per sub-game."""
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


def _hidden(roster, leaf, pos):  # noqa: ANN001, ANN202
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    ours = narrow(reg, pos, 0, limit=6).actions
    theirs = narrow(reg, pos, 1, limit=6).actions
    spreads = {side: completions(reg, pos, side, sheet) for side in (0, 1)}
    return belief_solve(
        reg, pos, ours, theirs, spreads, {0: leaf.__call__, 1: leaf.__call__},
        budget=Budget.matrix(), depth=2, refine=2, passes=2,
    )


def _open(roster, leaf, pos, restricted: bool):  # noqa: ANN001, ANN202
    reg = roster.reg
    ours = narrow(reg, pos, 0, limit=6).actions
    theirs = narrow(reg, pos, 1, limit=6).actions
    return search(
        reg, pos, ours, theirs, leaf.__call__, budget=Budget.matrix(),
        depth=2, refine=2, passes=2, solve_restricted=restricted,
    )


def _hidden_answers(answers) -> list:  # noqa: ANN001
    out = []
    for side in sorted(answers):
        a = answers[side]
        out.append((
            side, a.strategy.tobytes(), a.value, tuple(r.tobytes() for r in a.replies),
            sorted(a.unmodelled), a.refined, a.subgames, a.converged, a.optimism,
        ))
    return out


def _open_answer(result) -> tuple:  # noqa: ANN001
    eq = result.equilibrium
    return (
        eq.row_strategy.tobytes(), eq.col_strategy.tobytes(), eq.value,
        result.payoff.tobytes(), sorted(result.unmodelled), result.refined,
        result.subgames, result.converged, result.optimism, result.restricted,
    )


def _all_answers(roster, positions) -> tuple[list, int, int]:  # noqa: ANN001
    """Every depth-2 reading's answers on the positions, and the leaf's calls and sub-games."""
    leaf = _SizedLeaf(Encoder(roster.reg))
    answers: list = []
    subgames = 0
    for pos in positions:
        hidden = _hidden(roster, leaf, pos)
        subgames += sum(a.subgames for a in hidden.values())
        answers.append(_hidden_answers(hidden))
        for restricted in (True, False):
            opened = _open(roster, leaf, pos, restricted)
            subgames += opened.subgames
            answers.append(_open_answer(opened))
    return answers, leaf.total(), subgames


def test_the_batched_pass_is_the_per_sub_game_pass_to_the_bit(roster, monkeypatch) -> None:  # noqa: ANN001
    """Hidden depth 2 (both sides) and the open game's two depth-2 readings, on turn 1 and
    later turns: strategies, values, replies, payoffs, notes and counts are the same bytes
    as `_refined_value` per cell gives -- with a leaf whose answers move with batch size."""
    positions = _positions(roster)
    assert len(positions) >= 2
    batched, _calls, subgames = _all_answers(roster, positions)
    assert subgames > 20, "the depth-2 readings must actually refine for this to test anything"
    monkeypatch.setattr(search_mod, "_refine_cells", _per_sub_game)
    reference, _calls, _subgames = _all_answers(roster, positions)
    assert batched == reference


def test_stacking_the_sub_games_into_one_call_is_caught(roster, monkeypatch) -> None:  # noqa: ANN001
    """The positive control: score the pass's blocks as one batch -- the cheaper thing the
    issue first asked for -- and the answers are not the per-sub-game ones."""
    positions = _positions(roster)[:1]
    reference_leaf = _SizedLeaf(Encoder(roster.reg))
    reference = _hidden_answers(_hidden(roster, reference_leaf, positions[0]))
    monkeypatch.setattr(port, "score_segments", _stacked_scoring)
    stacked_leaf = _SizedLeaf(Encoder(roster.reg))
    stacked = _hidden_answers(_hidden(roster, stacked_leaf, positions[0]))
    assert stacked != reference


def test_a_hidden_depth_2_decision_calls_the_leaf_once_a_pass(roster, monkeypatch) -> None:  # noqa: ANN001
    """The count, with its positive control: the per-sub-game pass (the code before
    IKA-291) calls the leaf once per sub-game; the batched pass once per pass that has
    new cells, whatever the number of sub-games in it."""
    pos = _positions(roster)[0]
    batched_leaf = _SizedLeaf(Encoder(roster.reg))
    batched = _hidden(roster, batched_leaf, pos)
    subgames = sum(a.subgames for a in batched.values())

    monkeypatch.setattr(search_mod, "_refine_cells", _per_sub_game)
    before_leaf = _SizedLeaf(Encoder(roster.reg))
    _hidden(roster, before_leaf, pos)

    assert subgames >= 20
    # Before: the node's own call(s), then one per sub-game and none batched.
    assert before_leaf.calls["segments"] == 0
    assert before_leaf.calls["from_encoded"] >= subgames
    # After: the node's own call(s), then one call per pass per side (two passes, two sides),
    # or a few more where a pass holds more than `GATHER_ROWS` rows.
    node_calls = batched_leaf.calls["from_encoded"] + batched_leaf.calls["positions"]
    assert batched_leaf.calls["segments"] * 5 < subgames
    assert batched_leaf.blocks >= subgames
    assert node_calls < before_leaf.calls["from_encoded"] - subgames + 2
    assert batched_leaf.total() * 5 < before_leaf.total()


def test_a_pass_cut_by_gather_rows_is_the_same_pass(roster, monkeypatch) -> None:  # noqa: ANN001
    """A pass whose sub-games are scored in several calls (the memory cap) is the same pass:
    a sub-game is its own block wherever the calls are cut."""
    pos = _positions(roster)[0]
    whole = _hidden_answers(_hidden(roster, _SizedLeaf(Encoder(roster.reg)), pos))
    monkeypatch.setattr(search_mod, "GATHER_ROWS", 1)
    leaf = _SizedLeaf(Encoder(roster.reg))
    cut = _hidden_answers(_hidden(roster, leaf, pos))
    assert cut == whole
    assert leaf.calls["segments"] > 4
