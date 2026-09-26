"""The node's binary wire says what its JSON said (IKA-302).

An encoded node's spans -- each cell's leaf indices and weights, the fold back to a matrix --
used to be the large part of the JSON header. They now come in the body as a binary block
(`encoded_node::span_block` over there, `rustnode._binary_spans` here). The JSON road is kept
on request (`jsonSpans`) for this file alone, which holds the two to the same lists: the
same cells in the same order, the same leaf indices, and every weight the same double bit
for bit. Both crossings take the block: one node (`fill_encoded`) and many in one line
(`fill_encoded_many`, a depth-2 pass's sub-games), each through the shared block and down
the pipe.

The comparison is shown able to fail (`test_the_comparison_sees_a_broken_block`): a block
read with one field out of place is caught.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pytest

from pokeuraou import rustnode
from pokeuraou.budget import Budget

from .test_rust_node import _node, _pausing_cells, _with_parting_shot, bridged  # noqa: F401


def _exact(spans: list) -> list:
    """Spans with each weight as its bit pattern, so 'equal' means the same double."""
    return [
        (i, j, list(indices), [struct.pack("<d", w) for w in weights])
        for i, j, indices, weights in spans
    ]


def _both(child: Any, pos: Any, row: list, col: list) -> tuple[Any, Any]:  # noqa: ANN401
    binary = child.fill_encoded(pos, row, col, Budget.matrix(), ["hp-share"], None)
    text = child.fill_encoded(pos, row, col, Budget.matrix(), ["hp-share"], None, json_spans=True)
    return binary, text


@pytest.mark.parametrize("blocks", [True, False], ids=["shm", "pipe"])
def test_binary_spans_are_the_json_spans(bridged: None, blocks: bool) -> None:  # noqa: F811
    reg, pos, row, col = _node()
    rustnode.reset()
    child = rustnode.node_for(reg)
    assert child is not None
    child._shm_off = not blocks  # noqa: SLF001 - the road the test is here to pick
    seen: list[bool] = []
    real = rustnode.EncodedNode.unpack

    def watch(header: dict, body: bytearray):  # noqa: ANN202
        seen.append("spanCount" in header)
        return real(header, body)

    rustnode.EncodedNode.unpack = staticmethod(watch)
    try:
        binary, text = _both(child, pos, row, col)
    finally:
        rustnode.EncodedNode.unpack = staticmethod(real)
    # The positive control of the road itself: the first came as a block, the second as JSON.
    assert seen == [True, False]
    assert binary.spans, "a node with no spans compares nothing"
    assert sum(len(w) for _i, _j, _x, w in binary.spans) > len(binary.spans)
    assert _exact(binary.spans) == _exact(text.spans)
    assert binary.folded == text.folded
    assert binary.exact == text.exact
    # And the rest of the body is where it was: the leaf values behind the arrays.
    assert np.array_equal(binary.leaf_values["hp-share"], text.leaf_values["hp-share"])
    for name in ("species", "mon", "field"):
        assert np.array_equal(getattr(binary.encoded, name), getattr(text.encoded, name))


def test_a_pausing_node_keeps_its_folds_beside_the_block(bridged: None) -> None:  # noqa: F811
    """A cell whose turn paused is a fold tree in the header; the rest are spans."""
    reg, pos, row, col = _node()
    pos.sides[0].active_pokemon()[1].boosts["spe"] = 6
    row = _with_parting_shot(reg, pos, row)
    assert _pausing_cells(reg, pos, row, col, Budget.matrix()), "no cell pauses"
    child = rustnode.node_for(reg)
    binary, text = _both(child, pos, row, col)
    assert binary.folded, "no folded cell"
    assert _exact(binary.spans) == _exact(text.spans)
    assert binary.folded == text.folded


@pytest.mark.parametrize("blocks", [True, False], ids=["shm", "pipe"])
def test_many_nodes_in_one_line_cut_their_blocks_apart(bridged: None, blocks: bool) -> None:  # noqa: F811
    reg, pos, row, col = _node()
    rustnode.reset()
    child = rustnode.node_for(reg)
    child._shm_off = not blocks  # noqa: SLF001
    asks = [(pos, row, col), (pos, row[:3], col[:4]), (pos, row[2:], col)]
    many = child.fill_encoded_many(asks, Budget.matrix())
    for node, (p, r, c) in zip(many, asks, strict=True):
        alone = child.fill_encoded(p, r, c, Budget.matrix(), [], None, json_spans=True)
        assert _exact(node.spans) == _exact(alone.spans)
        assert node.folded == alone.folded
        assert np.array_equal(node.encoded.mon, alone.encoded.mon)


def test_the_comparison_sees_a_broken_block(bridged: None) -> None:  # noqa: F811
    """The positive control: a reader with the rows and the lengths swapped is caught."""
    reg, pos, row, col = _node()
    child = rustnode.node_for(reg)
    real = rustnode._binary_spans  # noqa: SLF001

    def broken(header: dict, body: Any):  # noqa: ANN202, ANN401
        spans = real(header, body)
        return [(j, i, indices, weights) for i, j, indices, weights in spans]

    rustnode._binary_spans = broken  # noqa: SLF001
    try:
        binary, text = _both(child, pos, row, col)
    finally:
        rustnode._binary_spans = real  # noqa: SLF001
    assert _exact(binary.spans) != _exact(text.spans)
