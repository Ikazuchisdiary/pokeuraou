"""The TD target mixes two arrays that must stay in the same orientation and scale.

`outcome` is 1.0 when side 0 won; `search_value` is the equilibrium value of a matrix
whose maximiser is side 0. Both are side-0 relative, so the mix is a plain convex
combination and any flip in it would be a bug that a validation number could not see --
the model would simply fit a target that means the opposite for half the rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.value import td_target

from .test_concat_datasets import shard


def dataset_with(outcome: list[float], searched: list[float]):  # noqa: ANN201
    data = shard(len(outcome), games=len(outcome), foe_names=("worlds",), outcome=0.0)
    data.outcome = np.array(outcome, dtype=np.float32)
    data.search_value = np.array(searched, dtype=np.float32)
    return data


def test_lambda_zero_is_the_outcome_exactly() -> None:
    data = dataset_with([1.0, 0.0, 1.0], [0.3, 0.9, 0.5])
    assert np.array_equal(td_target(data, 0.0), data.outcome)


def test_lambda_one_is_the_search_value_exactly() -> None:
    data = dataset_with([1.0, 0.0, 1.0], [0.3, 0.9, 0.5])
    assert np.allclose(td_target(data, 1.0), data.search_value)


def test_the_mix_is_convex_and_keeps_side_zero_orientation() -> None:
    data = dataset_with([1.0, 0.0], [0.25, 0.75])
    mixed = td_target(data, 0.4)
    assert np.allclose(mixed, [0.6 + 0.4 * 0.25, 0.4 * 0.75])
    # A won game never gets a target below its search value, nor above 1: a flip would
    # push the first row under 0.5 and the second above it.
    assert mixed[0] > mixed[1]


def test_a_lambda_outside_the_unit_interval_is_refused() -> None:
    data = dataset_with([1.0, 0.0], [0.25, 0.75])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        td_target(data, 1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        td_target(data, -0.1)


def test_a_dataset_without_search_values_is_refused() -> None:
    data = dataset_with([1.0, 0.0], [0.25, 0.75])
    data.search_value = np.zeros(0, np.float32)
    with pytest.raises(ValueError, match="search_value"):
        td_target(data, 0.4)
