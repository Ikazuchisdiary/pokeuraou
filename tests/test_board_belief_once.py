"""IKA-282: a match whose arms build different menus solves each belief node once.

Under a hidden bench, two arms with different leaves and a leaf ranking build a menu
each, so `play_game` calls `belief_solve` twice a decision -- once over side 0's menu,
once over side 1's. Each call used to build both sides' nodes and solve both LPs, and the
game read `answers[0]` of the first and `foe_answers[1]` of the second: half of the
matrix, the dirty fill and the LP were built and thrown away. Now each call is asked for
the side it is read on.

What these tests hold:

- **half the work**: per decision, two `belief_solve` calls asking one side each, two
  nodes and two LPs where the old shape built four of each (counted, with the old shape as
  the positive control of the counter);
- **the same game**: the record is the one the old shape plays, field for field apart
  from the search seconds;
- **a mix-up is caught**: a solve that builds the asked side's node with the other arm's
  leaf plays a different game, so the comparison above could have failed;
- **one leaf is untouched**: identical arms still solve once a decision, both sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou import beliefnode, equilibrium, search, selfplay
from pokeuraou.damage import register_mega_stones
from pokeuraou.pool import load_pool
from pokeuraou.poolplay import PoolArm, SolvedSelections, pool_match_game

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


REAL = search.belief_solve


def _old_shape(*args, sides=(0, 1), **kwargs):  # noqa: ANN002, ANN003, ANN202, ARG001
    """What `belief_solve` did before IKA-282: both sides, whatever is read."""
    return REAL(*args, **kwargs)


def _mixed_up(reg, pos, ours, theirs, spreads, evaluators, *, sides=(0, 1), **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    """The fault the one-sided build could have: the asked side scored by the other leaf."""
    if tuple(sides) != (0, 1) and evaluators[0] is not evaluators[1]:
        evaluators = {0: evaluators[1], 1: evaluators[0]}
    return REAL(reg, pos, ours, theirs, spreads, evaluators, sides=sides, **kwargs)


def _counting(monkeypatch, solve):  # noqa: ANN001, ANN202
    """Counts, inside `belief_solve` only, its calls, the nodes it builds and its LPs.

    The replacement node and the selection solve their own Bayesian games; they are not
    what this is about, so an LP is counted only while a `belief_solve` is running.
    """
    counts = {"solve": [], "node": 0, "lp": 0}
    inside = [False]

    def counted_solve(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        counts["solve"].append(tuple(kwargs.get("sides", (0, 1))))
        inside[0] = True
        try:
            return solve(*args, **kwargs)
        finally:
            inside[0] = False

    real_node, real_lp = beliefnode.belief_payoffs, equilibrium.solve_bayesian

    def counted_node(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        counts["node"] += inside[0]
        return real_node(*args, **kwargs)

    def counted_lp(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        counts["lp"] += inside[0]
        return real_lp(*args, **kwargs)

    monkeypatch.setattr(selfplay, "belief_solve", counted_solve)
    # `belief_solve` imports both at call time, so the module attributes are what it gets.
    monkeypatch.setattr(beliefnode, "belief_payoffs", counted_node)
    monkeypatch.setattr(equilibrium, "solve_bayesian", counted_lp)
    return counts


def _play(pool, arms, which):  # noqa: ANN001, ANN202
    record, _ = pool_match_game(
        pool.reg, pool, arms, seed=282, game_index=0, which=which,
        hide_bench=True, max_turns=4,
    )
    payload = record.to_json(objective="value:arm", search_limit=(4, 4))
    payload.pop("searchSeconds", None)
    moves = sum(d.kind == "move" for d in record.decisions)
    return payload, moves


def _arms(pool):  # noqa: ANN001, ANN202
    a = PoolArm(name="a", evaluate=_stub, solver=SolvedSelections(pool.reg, pool.teams, _stub),
                limit=4, rank_by_leaf=True)
    b = PoolArm(name="b", evaluate=_other,
                solver=SolvedSelections(pool.reg, pool.teams, _other),
                limit=4, rank_by_leaf=True)
    return a, b


@pytest.mark.parametrize("which", [0, 1])
def test_two_leaves_solve_one_side_each_and_play_the_same_game(pool, monkeypatch, which) -> None:  # noqa: ANN001
    arms = _arms(pool)

    with monkeypatch.context() as patch:
        old = _counting(patch, _old_shape)
        before, moves = _play(pool, arms, which)
    assert moves > 0
    # Positive control of the counter: the old shape builds four nodes and four LPs.
    assert len(old["solve"]) == 2 * moves
    assert old["node"] == 4 * moves and old["lp"] == 4 * moves

    with monkeypatch.context() as patch:
        new = _counting(patch, REAL)
        after, moves_after = _play(pool, arms, which)
    assert moves_after == moves
    assert new["solve"] == [(0,), (1,)] * moves, "each call asks for the side it is read on"
    assert new["node"] == 2 * moves, "half the nodes"
    assert new["lp"] == 2 * moves, "half the LPs"

    assert after == before, "reading one side must not change the game"

    with monkeypatch.context() as patch:
        _counting(patch, _mixed_up)
        wrong, _ = _play(pool, arms, which)
    assert wrong != before, (
        "a side scored by the other arm's leaf played the same game, so the comparison "
        "above could not have failed"
    )


def test_one_leaf_still_solves_once_for_both_sides(pool, monkeypatch) -> None:  # noqa: ANN001
    a, _ = _arms(pool)
    with monkeypatch.context() as patch:
        counts = _counting(patch, REAL)
        _, moves = _play(pool, (a, a), 0)
    assert moves > 0
    assert counts["solve"] == [(0, 1)] * moves
    assert counts["node"] == moves and counts["lp"] == 2 * moves


def test_asking_for_no_side_stops(pool) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="no side"):
        search.belief_solve(pool.reg, None, [], [], {}, {0: _stub, 1: _other},
                            budget=None, sides=())
