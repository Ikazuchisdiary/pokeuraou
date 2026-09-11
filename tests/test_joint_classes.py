"""The heaviest joint spread classes, taken without building the rest.

`joint_classes` used to build the whole Cartesian product and sort it: two hidden Pokemon
with about 1,150 spread-and-HP entries each is 1.36 million combinations materialised,
sorted, and thrown away except for four. The replacement walks the lattice with a heap.

An optimisation that changes the answer is not one, so this compares it against the
product on cases small enough to enumerate -- including the ties, which the old code broke
by `itertools.product` order through a stable sort, and which the new one has to break the
same way or a reported belief class changes name.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from pokeuraou.cli import _heaviest_combinations


def entry(weight: float, label: str):
    """One slot entry, shaped as `joint_classes` builds them."""
    return (weight, (0, 0), np.zeros(6, dtype=np.int64), 100, label)


def brute_force(per_slot, limit):
    combos = list(itertools.product(*per_slot)) if per_slot else [()]
    ordered = sorted(
        combos, key=lambda combo: -float(np.prod([e[0] for e in combo])) if combo else 0.0
    )
    return ordered[:limit]


def labels(combos):
    return [tuple(e[4] for e in combo) for combo in combos]


@pytest.mark.parametrize("limit", [1, 2, 4, 7, 50])
def test_matches_the_product_on_small_cases(limit: int) -> None:
    per_slot = [
        [entry(0.5, "a0"), entry(0.3, "a1"), entry(0.15, "a2"), entry(0.05, "a3")],
        [entry(0.6, "b0"), entry(0.25, "b1"), entry(0.1, "b2"), entry(0.05, "b3")],
    ]
    assert labels(_heaviest_combinations(per_slot, limit)) == labels(
        brute_force(per_slot, limit)
    )


def test_ties_break_in_product_order() -> None:
    """Equal weights must come out in the order the product would have produced them.

    The old code sorted a materialised list with Python's stable sort, so among equal
    weights the earlier combination won -- and the earlier one is the one the last axis
    reaches first. A heap that broke ties any other way would rename a belief class the
    report prints.
    """
    per_slot = [
        [entry(0.5, "a0"), entry(0.5, "a1")],
        [entry(0.25, "b0"), entry(0.25, "b1")],
    ]
    assert labels(_heaviest_combinations(per_slot, 4)) == [
        ("a0", "b0"),
        ("a0", "b1"),
        ("a1", "b0"),
        ("a1", "b1"),
    ]


def test_three_slots() -> None:
    per_slot = [
        [entry(0.7, "a0"), entry(0.3, "a1")],
        [entry(0.6, "b0"), entry(0.4, "b1")],
        [entry(0.9, "c0"), entry(0.1, "c1")],
    ]
    assert labels(_heaviest_combinations(per_slot, 5)) == labels(brute_force(per_slot, 5))


def test_edges() -> None:
    assert _heaviest_combinations([], 4) == [()]
    assert _heaviest_combinations([[], [entry(1.0, "b")]], 4) == []
    single = [[entry(1.0, "only")]]
    assert labels(_heaviest_combinations(single, 4)) == [("only",)]


def test_it_does_not_build_the_product() -> None:
    """Two slots of a thousand entries is a million combinations; this must not make them.

    The check is the clock rather than a counter: the product would take seconds, and this
    is asked for four of them.
    """
    import time

    per_slot = [
        [entry(1.0 / (i + 1), f"a{i}") for i in range(1200)],
        [entry(1.0 / (i + 1), f"b{i}") for i in range(1200)],
    ]
    started = time.perf_counter()
    out = _heaviest_combinations(per_slot, 4)
    assert len(out) == 4
    assert time.perf_counter() - started < 0.5
