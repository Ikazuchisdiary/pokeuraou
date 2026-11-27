"""IKA-282: `belief_solve` is asked for the sides it is read on.

Under a hidden bench, two arms with different leaves build a menu each, and each call of
`belief_solve` is asked for the side it is read on. The board games that counted this
(two leaves solve one side each, one leaf solves once) went with the board's Python game loop
in IKA-458 stage 3.7; the Rust board worker solves per side (`playgame.rs`, records/IKA-458.md
section 17.2). What stays: asking for no side stops. `_other` and `_other_served` are the second
leaf other tests import.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import search
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool

from ._served import steep_leaf
from .test_poolplay import _stub, _variants, _write_pool


@pytest.fixture(scope="module")
def pool(tmp_path_factory):  # noqa: ANN001, ANN201
    path = _write_pool(tmp_path_factory.mktemp("pool") / "p.json", _variants())
    loaded = load_pool(path)
    register_mega_stones(loaded.reg)
    return loaded


def _other(positions):  # noqa: ANN001, ANN202
    """A second leaf that disagrees with `_stub`: the active pair only, a steeper curve."""
    out = np.empty(len(positions), dtype=np.float64)
    for i, pos in enumerate(positions):
        strength = [
            sum(m.hp for m in side.pokemon if m.active_index is not None)
            + 0.25 * sum(m.hp for m in side.pokemon if m.active_index is None)
            for side in pos.sides
        ]
        out[i] = 1.0 / (1.0 + np.exp(-(strength[0] - strength[1]) / 60.0))
    return out


_other_served = steep_leaf()


def test_asking_for_no_side_stops(pool) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="no side"):
        search.belief_solve(pool.reg, None, [], [], {}, {0: _stub, 1: _other},
                            budget=None, sides=())
