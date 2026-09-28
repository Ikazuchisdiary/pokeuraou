"""IKA-365: a refined cell's children go to the leaf in one call, and nothing moves.

`deepen._expand` filled and scored a cell's branches one at a time -- through the inference
server that was a round trip a branch, plus one for the ended branches, a few dozen rows
each, and a served deepening spent its time in the trips. It now fills every branch in one
crossing (`port.pending_payoffs`) and scores the children and the ended positions in one
`port.score_segments` call, each block in a call of its own size.

The leaf here (`_SizedLeaf`) answers with a trace of its call's size, as a card does, so a
block scored in a call of another size would show. The reference is the old road itself:
`_expand_scored` turned off, which is the per-branch `batched_payoff` and the separate
`evaluate(ended)`. The control stacks the blocks into one call and must be seen.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import deepen as deepen_mod
from pokeuraou import port
from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.encode import Encoder
from pokeuraou.search import search
from pokeuraou.teams import load_roster

from .test_deepen_ahead import _belief, _menus, _same_belief, _same_tree
from .test_hidden_depth2 import _played
from .test_subgame_batch import _SizedLeaf, _stacked_scoring


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _old_road(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(deepen_mod, "_expand_scored", lambda *_a, **_k: None)


def test_a_cell_is_one_call_and_the_old_road_to_the_bit(roster, monkeypatch) -> None:  # noqa: ANN001
    sized = _SizedLeaf(Encoder(roster.reg))
    leaf = sized.__call__
    cells = 0
    new: list = []
    for pos in _played(roster)[:2]:
        for side in (0, 1):
            new.append(_belief(roster, pos, side, leaf, 400, 0))
    calls_new = dict(sized.calls)
    blocks_new = sized.blocks
    _old_road(monkeypatch)
    sized.calls = {k: 0 for k in sized.calls}
    sized.blocks = 0
    at = 0
    for pos in _played(roster)[:2]:
        for side in (0, 1):
            old = _belief(roster, pos, side, leaf, 400, 0)
            _same_belief(old[0], new[at][0])
            _same_tree(old[1], new[at][1])
            cells += sum(1 for _n, _c, took in new[at][1][1:] if took)
            at += 1
    # Positive control: the new road scored the cells' children in `segments` calls, one
    # a cell, where the old one made a `from_encoded` call a branch.
    assert cells > 20
    assert calls_new["from_encoded"] == 0 and 0 < calls_new["segments"] <= cells
    assert sized.calls["segments"] == 0 and sized.calls["from_encoded"] >= cells
    assert blocks_new >= sized.calls["from_encoded"]


def test_stacking_a_cells_blocks_is_seen(roster, monkeypatch) -> None:  # noqa: ANN001
    """The control: the blocks of a cell in one call of the summed size move the tree."""
    leaf = _SizedLeaf(Encoder(roster.reg)).__call__
    pos = _played(roster)[1]
    good = _belief(roster, pos, 0, leaf, 400, 0)
    monkeypatch.setattr(port, "score_segments", _stacked_scoring)
    bad = _belief(roster, pos, 0, leaf, 400, 0)
    with pytest.raises(AssertionError):
        _same_belief(good[0], bad[0])
        _same_tree(good[1], bad[1])


def test_an_open_root_is_the_old_road(roster, monkeypatch) -> None:  # noqa: ANN001
    """Both benches seen (`deepen_root`): the same search on either road."""
    reg = roster.reg
    leaf = _SizedLeaf(Encoder(reg)).__call__
    answers: dict[str, list] = {"new": [], "old": []}
    for road in ("new", "old"):
        if road == "old":
            _old_road(monkeypatch)
        for pos in _played(roster)[:2]:
            ours, theirs = _menus(reg, pos)
            answers[road].append(search(reg, pos, ours, theirs, leaf, budget=Budget.matrix(),
                                        deepen=300))
    expanded = 0
    for a, b in zip(answers["new"], answers["old"], strict=True):
        assert np.array_equal(a.payoff, b.payoff)
        assert np.array_equal(a.equilibrium.row_strategy, b.equilibrium.row_strategy)
        assert a.equilibrium.value == b.equilibrium.value
        assert a.deepened == b.deepened
        expanded += a.deepened.expanded
    assert expanded > 10
