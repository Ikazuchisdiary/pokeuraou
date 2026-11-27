"""The open game's depth-1 search, a refined cell's value, and the leaf ranking.

- the undeepened `search` is `batched_payoff` and `solve`, bit for bit;
- the refined value of a cell (`search._refine_cells`, what the ladder reads a deep cell
  with) is the next position's equilibrium value, checked against solving that position
  directly.

The fixed depth-2 search (``search(depth=2)``) and its restricted reading were removed in
IKA-463, with the tests of their passes, rectangles and seat identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.port import batched_payoff
from pokeuraou.position import Position
from pokeuraou.search import leaf_ranking, search
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from ._port import Budget, resolve_turn

#: The parameter-free objective in batch form stands in for a learned value function.
#: Every claim here is about the search, not about the leaf, and a leaf with no weights
#: keeps the test from depending on which model happens to be on disk.
LEAF = HP_SHARE.batch


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _played(roster, turns: int = 5, seed: int = 3) -> list[Position]:  # noqa: ANN001
    """Positions from a real game, one per turn.

    Turn 1 is not enough: it has no statuses, no boosts, no weather and a full bench, and
    those are exactly the states where one more ply might see something the value
    function does not.

    The two sides are deliberately *different* fours. A mirror is the right fixture for
    testing symmetry and the wrong one for testing that a number moved: hp-share on a
    perfectly symmetric position is exactly 0.5 at every depth, so a first draft of this
    file concluded that refinement changed nothing when it had changed nothing it could.
    """
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
            reg,
            pos,
            [ours[int(rng.integers(len(ours)))], theirs[int(rng.integers(len(theirs)))]],
            budget=Budget.exact(),
        )
        if result.suspended or not result.branches:
            break
        weights = np.array([b.probability for b in result.branches], dtype=np.float64)
        pos = result.branches[
            int(rng.choice(len(weights), p=weights / weights.sum()))
        ].position
    return out


def test_depth_one_is_the_old_search_exactly(roster) -> None:  # noqa: ANN001
    """Same matrix, same equilibrium, bit for bit, on positions from a real game."""
    reg = roster.reg
    positions = _played(roster)
    assert len(positions) >= 3, "the game ended too early to test anything"
    for pos in positions:
        ours = narrow(reg, pos, 0, limit=8).actions
        theirs = narrow(reg, pos, 1, limit=8).actions
        if not ours or not theirs:
            continue
        expected, notes = batched_payoff(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix()
        )
        got = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
        assert np.array_equal(got.payoff, expected), "the matrix must be identical"
        assert got.unmodelled == notes
        assert got.refined == 0 and got.subgames == 0, "depth 1 refines nothing"
        reference = solve(expected)
        assert got.equilibrium.value == reference.value
        assert np.array_equal(got.equilibrium.row_strategy, reference.row_strategy)
        assert np.array_equal(got.equilibrium.col_strategy, reference.col_strategy)


def test_a_refined_cell_is_the_next_position_s_equilibrium(roster) -> None:  # noqa: ANN001
    """The refined number is recomputed by hand from the same definition.

    Resolve the cell's turn, solve each resulting position's own matrix, average over the
    branch probabilities. If `_refine_cells`' value does not equal that, the number a deep
    cell is read at is not the thing its docstring says it is.
    """
    from pokeuraou.search import _refine_cells  # noqa: PLC2701

    reg = roster.reg
    positions = _played(roster)
    verified = 0
    for pos in positions:
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            continue
        shallow = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix())
        # The cell both sides weight most, as the depth-1 answer plays it.
        i = int(np.argmax(shallow.equilibrium.row_strategy))
        j = int(np.argmax(shallow.equilibrium.col_strategy))
        [(value, _notes, _solved)] = _refine_cells(
            reg, [(pos, ours[i], theirs[j])], LEAF, budget=Budget.matrix(), sub_limit=8,
            sub_branches=3,
        )
        if value is None:
            continue
        result = resolve_turn(reg, pos, [ours[i], theirs[j]], budget=Budget.matrix())
        branches = sorted(result.branches, key=lambda b: -b.probability)[:3]
        weights = np.array([b.probability for b in branches], dtype=np.float64)
        weights /= weights.sum()
        values = []
        for branch in branches:
            if branch.position.ended:
                values.append(float(LEAF([branch.position])[0]))
                continue
            sub_row = narrow(reg, branch.position, 0, limit=8).actions
            sub_col = narrow(reg, branch.position, 1, limit=8).actions
            sub, _notes = batched_payoff(
                reg, branch.position, sub_row, sub_col, LEAF, budget=Budget.matrix()
            )
            values.append(float(solve(sub).value))
        by_hand = float(np.array(values) @ weights)
        assert value == pytest.approx(by_hand, abs=1e-12), (
            f"cell ({i},{j}) is {value}, the definition gives {by_hand}"
        )
        # The control: the cell's own depth-1 value is not the refined one.
        assert abs(float(shallow.payoff[i, j]) - by_hand) > 1e-9
        verified += 1
    assert verified >= 1, "no cell was refined, so nothing was verified"


def test_a_leaf_ranking_keeps_the_coverage_guarantee(roster) -> None:  # noqa: ANN001
    """Changing the order must not change what narrowing promises.

    The guarantee is that every individual slot option appears somewhere in the kept set,
    so nothing is eliminated outright -- a Protect or a Fake Out scores no damage and
    would vanish from a pure ranking. `rank` replaces the ordering and nothing else.

    Asserted at a budget wide enough to cover everything, because that is where the
    guarantee is a guarantee. At a tight budget the cover is greedy and takes the
    best-scoring candidate among those covering the most, so a different order genuinely
    leaves a different set uncovered -- a first draft of this test asserted the sets were
    equal and failed on exactly that, which is the algorithm working rather than breaking.
    What must still hold at a tight budget is that the same budget is spent.
    """
    reg = roster.reg
    positions = _played(roster)
    checked = 0
    for pos in positions:
        for side in (0, 1):
            ranking = leaf_ranking(reg, pos, side, LEAF, budget=Budget.matrix())
            wide = narrow(reg, pos, side, limit=99, rank=ranking)
            assert wide.uncovered == (), (
                f"nothing should be left out at a budget of 99: {wide.uncovered}"
            )
            plain = narrow(reg, pos, side, limit=8)
            ranked = narrow(reg, pos, side, limit=8, rank=ranking)
            assert len(ranked.kept) == len(plain.kept), "the same budget is spent"
            assert ranked.considered == plain.considered
            assert len(ranked.uncovered) == len(ranked.uncovered_options)
            checked += 1
    assert checked >= 6, f"only {checked} narrowings checked"


def test_a_leaf_ranking_actually_reorders(roster) -> None:  # noqa: ANN001
    """The two scores disagree, which is the entire reason the option exists.

    If the leaf ordered candidates the way expected damage does, there would be nothing to
    measure and nothing to fix -- so this fails loudly rather than letting a head-to-head
    spend two hours discovering the two arms were the same arm.
    """
    reg = roster.reg
    positions = _played(roster)
    differed = 0
    compared = 0
    for pos in positions:
        for side in (0, 1):
            plain = narrow(reg, pos, side, limit=6)
            ranked = narrow(
                reg,
                pos,
                side,
                limit=6,
                rank=leaf_ranking(reg, pos, side, LEAF, budget=Budget.matrix()),
            )
            if plain.complete:
                continue  # nothing was narrowed, so nothing could be reordered
            compared += 1
            before = [a.to_choice() for a in plain.actions]
            after = [a.to_choice() for a in ranked.actions]
            differed += int(before != after)
    assert compared >= 2, f"only {compared} positions were narrowed at all"
    assert differed >= 1, "the leaf ranking never changed the menu on any position"


def test_the_leaf_ranking_scores_what_it_says_it_scores(roster) -> None:  # noqa: ANN001
    """The score of a candidate is the leaf's mean over the reference replies.

    Recomputed from the definition rather than compared against itself: the ranking is
    about to decide which actions the solver ever sees, and a score that is not the
    quantity its docstring names would move that choice for reasons nobody could state.
    """
    reg = roster.reg
    pos = _played(roster)[1]
    for side in (0, 1):
        rank = leaf_ranking(reg, pos, side, LEAF, budget=Budget.matrix(), references=2)
        pool = narrow(reg, pos, side, limit=99).actions[:5]
        got = np.asarray(rank(pool), dtype=np.float64)
        replies = narrow(reg, pos, 1 - side, limit=8).actions[:2]
        ours, theirs = (pool, replies) if side == 0 else (replies, pool)
        payoff, _notes = batched_payoff(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix()
        )
        # Side 0's payoff either way, so side 1 wants it small and its score is negated.
        want = payoff.mean(axis=1) if side == 0 else -payoff.mean(axis=0)
        assert np.allclose(got, want, atol=1e-12), f"side {side}: {got} != {want}"
