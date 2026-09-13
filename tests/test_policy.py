"""The learned ordering's feature layout, and the shape it hands `narrow`.

Two things are worth pinning and neither is about the model's quality.

The first is that the rows built at search time are the rows the model was trained on.
`features_for` and `ids_for` used to live in `tools/policy_dataset.py`, where only training
called them; now the search calls them too, and a model is only correct if both callers
build the same arithmetic. A width that drifts by one column would load, run and rank -- by
numbers that mean something else.

The second is that `policy_ranking` satisfies the contract `narrow` has with a ranker: one
score per candidate, in the order the candidates were given, whatever else is true. A
ranker that returned a short array would silently truncate the menu, and a menu that is
quietly short is the failure this whole module exists to avoid.

The model itself is not loaded here. `load_policy` needs torch and a trained file, and what
it does with them is measured on the board rather than asserted in a test.
"""

from __future__ import annotations

import numpy as np

from pokeuraou.actions import side_actions
from pokeuraou.narrow import narrow
from pokeuraou.oracle import TeamSet
from pokeuraou.policy import (
    FEATURES,
    IDS,
    MOVE_IDS,
    SPECIES_IDS,
    features_for,
    ids_for,
    policy_ranking,
)
from pokeuraou.regulation import Regulation

from .test_actions import _synthetic_position


def test_feature_width_matches_the_declared_layout(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """One row per action, `FEATURES` wide. The model's first layer is sized from this."""
    pos = _synthetic_position(reg, team_a)
    actions = side_actions(reg, pos, 0)
    rows = features_for(reg, pos, 0, actions, [0.0] * len(actions))
    assert rows.shape == (len(actions), FEATURES)
    assert rows.dtype == np.float32
    assert np.isfinite(rows).all()


def test_identity_columns_cover_the_layout_exactly(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """Every id column is embedded by exactly one table, or the rest of the row shifts.

    `load_policy` splits `ids_for`'s columns between the species table and the move table
    and concatenates the two. A column claimed by both, or by neither, changes the width of
    what reaches the trunk -- which fails loudly on a trained model and silently on a
    retrained one.
    """
    assert sorted(SPECIES_IDS + MOVE_IDS) == list(range(IDS))


def test_ids_are_within_the_vocabularies(reg: Regulation, team_a: list[TeamSet]) -> None:
    """An index past the embedding table is an exception at search time, not a bad rank."""
    from pokeuraou.encode import Encoder

    encoder = Encoder(reg)
    pos = _synthetic_position(reg, team_a)
    actions = side_actions(reg, pos, 0)
    ids = ids_for(encoder, pos, 0, actions)
    assert ids.shape == (len(actions), IDS)
    assert ids.min() >= 0
    assert ids[:, SPECIES_IDS].max() <= len(encoder.vocab.species)
    assert ids[:, MOVE_IDS].max() <= len(encoder.vocab.moves)


def test_scores_shift_the_score_block_only(reg: Regulation, team_a: list[TeamSet]) -> None:
    """The damage score enters four columns; handing different scores moves only those.

    `narrow` computes the damage candidates anyway and hands them over, so the score block
    is the one part of a row that is free. If a change there leaked into the action's own
    columns the model would be reading the ordering it is supposed to replace.
    """
    pos = _synthetic_position(reg, team_a)
    actions = side_actions(reg, pos, 0)
    flat = features_for(reg, pos, 0, actions, [0.0] * len(actions))
    varied = features_for(
        reg, pos, 0, actions, list(np.linspace(0.0, 1.0, len(actions)))
    )
    base = 2 * 12 + 4  # SLOT_FEATURES twice, then PAIR_FEATURES
    assert (flat[:, :base] == varied[:, :base]).all()
    assert (flat[:, base + 4 :] == varied[:, base + 4 :]).all()
    assert not (flat[:, base : base + 4] == varied[:, base : base + 4]).all()


def test_policy_ranking_answers_narrow_in_narrows_shape(
    reg: Regulation, team_a: list[TeamSet]
) -> None:
    """One score per candidate, in order, and the menu follows it.

    Two stand-ins score the pool by its position, one each way, so the ordering asked for
    is unambiguous and has nothing to do with damage. The menu is not simply the top of
    that order -- coverage takes its picks first -- so what is asserted is that the
    ranker's own favourite survives and that reversing the ranking changes the menu.
    """
    pos = _synthetic_position(reg, team_a)
    seen: list[int] = []

    def ascending(position, side, actions, scored=None):
        seen.append(len(actions))
        return np.arange(len(actions), dtype=np.float64)

    def descending(position, side, actions, scored=None):
        return -np.arange(len(actions), dtype=np.float64)

    pool = side_actions(reg, pos, 0)
    assert policy_ranking(ascending, pos, 0)(pool).shape == (len(pool),)
    assert policy_ranking(ascending, pos, 0)([]).shape == (0,)

    up = narrow(reg, pos, 0, limit=24, rank=policy_ranking(ascending, pos, 0))
    down = narrow(reg, pos, 0, limit=24, rank=policy_ranking(descending, pos, 0))
    assert len(up.actions) <= 24
    chosen = {a.to_choice() for a in up.actions}
    assert pool[-1].to_choice() in chosen          # the ranker's own favourite
    assert chosen != {a.to_choice() for a in down.actions}
    # `narrow` asked once, over the whole legal pool -- not per candidate.
    assert seen and max(seen) == len(pool)
