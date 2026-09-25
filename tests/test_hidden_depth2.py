"""Restricted depth 2 inside a hidden-bench node (IKA-111).

`belief_solve` had no depth, so `play_game` refused depth 2 with `sheets`, and depth 2 did
not move a single decision where the agent ships. These pin what the new reading claims:

- with one completion of weight 1 it IS the open game's restricted depth 2 -- the same
  rectangle, the same strategy, the same guaranteed value -- so the hidden reading is the
  open one generalised, not a different search that happens to share a name;
- with a real belief it refines cells per completion and moves the answer (the positive
  control: zero refined cells at depth 1, and on master, where the call could not be made);
- depth 1 is the old answer to the bit, and a game at depth 1 records nothing new;
- `play_game` takes depth 2 under `sheets` only in the restricted reading, and the arm's
  setting reaches its own side in both seats of a pool match.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.budget import Budget
from pokeuraou.damage import register_mega_stones
from pokeuraou.hidden import completions
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import Position
from pokeuraou.search import belief_solve, search
from pokeuraou.selfplay import play_game, position_from_sets
from pokeuraou.teams import load_roster

from ._port import resolve_turn

LEAF = HP_SHARE.batch
SEEN_ALL = frozenset({0, 1, 2, 3})


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _played(roster, turns: int = 4, seed: int = 3) -> list[Position]:  # noqa: ANN001
    """Positions from one game of two different fours (a mirror is 0.5 at every depth
    under hp-share, and a test of "the number moved" would pass on nothing)."""
    reg = roster.reg
    pos = position_from_sets(reg, list(roster.sets[:4]), list(roster.sets[2:6]))
    rng = np.random.default_rng(seed)
    out: list[Position] = []
    for _ in range(turns):
        if pos.ended:
            break
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            break
        out.append(pos.copy())
        result = resolve_turn(
            reg, pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        pos = max(result.branches, key=lambda b: b.probability).position
    return out


def _menus(reg, pos, limit=6):  # noqa: ANN001, ANN202
    return (
        narrow(reg, pos, 0, limit=limit).actions,
        narrow(reg, pos, 1, limit=limit).actions,
    )


def _exact(reg, pos, sheet):  # noqa: ANN001, ANN202
    """Both benches seen: one completion of weight 1 per side, the position itself."""
    return {side: completions(reg, pos, side, sheet, seen=SEEN_ALL) for side in (0, 1)}


def test_one_completion_is_the_open_restricted_depth_2(roster) -> None:  # noqa: ANN001
    """The correctness check: nothing hidden, so the Bayesian rectangle has one column set
    and the answer must be `search(depth=2, solve_restricted=True)`'s, for both sides."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    checked = 0
    for pos in _played(roster):
        ours, theirs = _menus(reg, pos)
        if not ours or not theirs:
            continue
        spreads = _exact(reg, pos, sheet)
        assert len(spreads[0]) == 1 and len(spreads[1]) == 1
        budget = Budget.matrix()
        opened = search(
            reg, pos, ours, theirs, LEAF, budget=budget,
            depth=2, refine=2, passes=2, solve_restricted=True,
        )
        hidden = belief_solve(
            reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF}, budget=budget,
            depth=2, refine=2, passes=2,
        )
        if opened.restricted == (0, 0):
            continue
        mine = hidden[0]
        assert mine.classes == 1
        assert mine.refined == opened.refined
        assert mine.converged == opened.converged
        assert mine.value == pytest.approx(float(opened.equilibrium.value), abs=1e-9)
        assert mine.optimism == pytest.approx(opened.optimism, abs=1e-9)
        np.testing.assert_allclose(mine.strategy, opened.equilibrium.row_strategy, atol=1e-9)
        np.testing.assert_allclose(mine.replies[0], opened.equilibrium.col_strategy, atol=1e-9)
        # Side 1 solves the same rectangle from its own chair: its strategy is the open
        # search's column strategy.
        np.testing.assert_allclose(
            hidden[1].strategy, opened.equilibrium.col_strategy, atol=1e-9
        )
        assert hidden[1].refined == opened.refined
        checked += 1
    assert checked >= 2, f"only {checked} positions compared"


def test_depth_2_refines_per_completion_and_moves_the_answer(roster) -> None:  # noqa: ANN001
    """The positive control: a real belief (turn 1, both benches unseen) refines cells in
    every completion's column set, and the answer is not the depth-1 one."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    pos = position_from_sets(reg, sheet[:4], sheet[2:6])
    ours, theirs = _menus(reg, pos)
    spreads = {side: completions(reg, pos, side, sheet) for side in (0, 1)}
    assert len(spreads[1]) > 1
    budget = Budget.matrix()
    shallow = belief_solve(
        reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF}, budget=budget
    )
    deep = belief_solve(
        reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF}, budget=budget,
        depth=2, refine=2, passes=2,
    )
    for side in (0, 1):
        assert shallow[side].refined == 0
        assert deep[side].refined > 0
        assert deep[side].subgames > 0
        assert deep[side].classes == shallow[side].classes
        assert deep[side].strategy.sum() == pytest.approx(1.0)
        assert all(y.sum() == pytest.approx(1.0) for y in deep[side].replies)
        # The value is the guarantee over every completion's every column at the prices
        # in hand, so it is at most the restricted game's own value.
        assert deep[side].optimism >= -1e-9
        assert any("hidden-bench strategy read from the restricted" in n
                   for n in deep[side].unmodelled)
    moved = any(
        not np.allclose(deep[s].strategy, shallow[s].strategy, atol=1e-9)
        or abs(deep[s].value - shallow[s].value) > 1e-9
        for s in (0, 1)
    )
    assert moved, "depth 2 refined cells and changed nothing on either side"


def test_depth_2_per_side(roster) -> None:  # noqa: ANN001
    """`depth` is per side: (2, 1) refines side 0's game only, and side 1's answer is
    the depth-1 one to the bit."""
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    pos = position_from_sets(reg, sheet[:4], sheet[2:6])
    ours, theirs = _menus(reg, pos)
    spreads = {side: completions(reg, pos, side, sheet) for side in (0, 1)}
    budget = Budget.matrix()
    shallow = belief_solve(reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF}, budget=budget)
    mixed = belief_solve(
        reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF}, budget=budget,
        depth=(2, 1), refine=2, passes=1,
    )
    assert mixed[0].refined > 0
    assert mixed[1].refined == 0
    np.testing.assert_array_equal(mixed[1].strategy, shallow[1].strategy)
    assert mixed[1].value == shallow[1].value
    # The side-0 depth-2 notes did not leak into side 1's shared set.
    assert not any("hidden-bench strategy" in n for n in mixed[1].unmodelled)


def test_depth_1_is_the_old_answer_to_the_bit(roster) -> None:  # noqa: ANN001
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    pos = position_from_sets(reg, sheet[:4], sheet[2:6])
    ours, theirs = _menus(reg, pos)
    spreads = {side: completions(reg, pos, side, sheet) for side in (0, 1)}
    budget = Budget.matrix()
    before = belief_solve(reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF}, budget=budget)
    after = belief_solve(
        reg, pos, ours, theirs, spreads, {0: LEAF, 1: LEAF}, budget=budget, depth=1
    )
    for side in (0, 1):
        np.testing.assert_array_equal(after[side].strategy, before[side].strategy)
        assert after[side].value == before[side].value
        assert after[side].unmodelled == before[side].unmodelled
        assert after[side].refined == 0


def _game(roster, **kwargs):  # noqa: ANN001, ANN003, ANN202
    reg = roster.reg
    sheet = list(roster.sets)[:6]
    return play_game(
        reg, np.random.default_rng(11), sheet[:4], sheet[2:6], "probe",
        search_limit=4, max_turns=3, sheets=(sheet, sheet), **kwargs,
    )


def test_a_hidden_game_plays_depth_2_and_records_it(roster) -> None:  # noqa: ANN001
    shallow = _game(roster)
    deep = _game(roster, depth=(2, 1), solve_restricted=(True, False))
    assert deep.information == "hidden-bench"
    assert deep.depth == [2, 1] and deep.solve_restricted == [True, False]
    notes = [n for n in deep.unmodelled if "hidden-bench strategy read" in n]
    assert notes, "a depth-2 hidden game refined nothing"
    assert not any("hidden-bench strategy read" in n for n in shallow.unmodelled)
    as_json = deep.to_json(objective="hp-share", search_limit=4)
    assert as_json["depth"] == [2, 1] and as_json["solveRestricted"] == [True, False]
    plain = shallow.to_json(objective="hp-share", search_limit=4)
    assert "depth" not in plain and "solveRestricted" not in plain


def test_hidden_depth_2_needs_the_restricted_reading(roster) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="hidden bench"):
        _game(roster, depth=(2, 2))
    with pytest.raises(ValueError, match="hidden bench"):
        _game(roster, depth=(2, 1), solve_restricted=(True, True))
    with pytest.raises(ValueError, match="hidden bench"):
        _game(roster, depth=3, solve_restricted=True)


def test_a_pool_arm_carries_its_depth_to_its_own_side_in_both_seats(  # noqa: ANN001
    monkeypatch, tmp_path
) -> None:
    """A per-arm setting must reach the arm in BOTH seats (`pool_match_game` swaps them)."""
    from pokeuraou import poolplay

    seen: list[dict] = []

    def fake_play_game(*_args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.append(kwargs)
        raise RuntimeError("stop")

    monkeypatch.setattr(poolplay, "play_game", fake_play_game)
    deep = poolplay.PoolArm(name="d", evaluate=None, solver=None, limit=4,
                            rank_by_leaf=False, depth=2, solve_restricted=True)
    flat = poolplay.PoolArm(name="f", evaluate=None, solver=None, limit=4,
                            rank_by_leaf=False)
    from pokeuraou.pool import load_pool

    from .test_pool_match import _variants, _write_pool

    pool = load_pool(_write_pool(tmp_path / "p.json", _variants()))
    register_mega_stones(pool.reg)
    for which in (0, 1):
        with pytest.raises(RuntimeError, match="stop"):
            poolplay.pool_match_game(
                pool.reg, pool, (deep, flat), seed=1, game_index=0, which=which,
                hide_bench=True, max_turns=2, epsilon=0.0, temperature=1.0,
            )
    assert seen[0]["depth"] == (2, 1) and seen[0]["solve_restricted"] == (True, False)
    assert seen[1]["depth"] == (1, 2) and seen[1]["solve_restricted"] == (False, True)
