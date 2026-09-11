"""The two-ply refinement: that depth 1 is unchanged, and that depth 2 changes something.

The first of those is the load-bearing one. A depth parameter whose lowest setting is not
byte-for-byte the previous search makes every comparison against it meaningless -- the
head-to-head would be measuring the refactor as much as the depth, and there would be no
way to tell the two apart afterwards.

The rest pin the properties the module's docstring claims, because they are the ones a
reader would otherwise have to take on trust:

- refined cells come from the equilibrium's support, not from anywhere else;
- a cell that cannot be refined keeps its depth-1 value rather than an invented one;
- the refined value of a cell is the next position's equilibrium value, which is checked
  against solving that position directly;
- the cost is bounded by `refine`, because an unbounded "refine what matters" is how a
  search that was supposed to cost twice as much ends up costing thirty times.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.equilibrium import solve
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import Position
from pokeuraou.resolve import Budget, batched_payoff, resolve_turn
from pokeuraou.search import leaf_ranking, search
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

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
        got = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=1)
        assert np.array_equal(got.payoff, expected), "the matrix must be identical"
        assert got.unmodelled == notes
        assert got.refined == 0 and got.subgames == 0, "depth 1 refines nothing"
        reference = solve(expected)
        assert got.equilibrium.value == reference.value
        assert np.array_equal(got.equilibrium.row_strategy, reference.row_strategy)
        assert np.array_equal(got.equilibrium.col_strategy, reference.col_strategy)


def test_one_pass_refines_only_the_depth_one_support(roster) -> None:  # noqa: ANN001
    """With a single pass, every cell that changed is one the depth-1 equilibrium weighted.

    One pass, because that is the only setting where the claim is checkable from outside.
    With more, a later pass refines the support of an equilibrium that no longer exists by
    the time the caller sees a result -- see the test below, which is about exactly that.
    """
    reg = roster.reg
    positions = _played(roster)
    checked = 0
    for pos in positions:
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            continue
        shallow = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=1)
        deep = search(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix(),
            depth=2, refine=2, passes=1,
        )
        rows = shallow.equilibrium.row_strategy > 1e-9
        cols = shallow.equilibrium.col_strategy > 1e-9
        outside = ~np.outer(rows, cols)
        changed = shallow.payoff != deep.payoff
        assert not (changed & outside).any(), (
            f"a cell outside the depth-1 support changed: {np.argwhere(changed & outside)}"
        )
        assert deep.refined <= 2 * 2, f"refine=2 is at most four cells: {deep.refined}"
        checked += 1
    assert checked >= 3, f"only {checked} positions searched"


def test_a_second_pass_can_reach_outside_the_first_support(roster) -> None:  # noqa: ANN001
    """Refining moves the equilibrium, and the next pass follows it.

    This is the double-oracle step doing its job rather than a leak: an action the depth-1
    matrix ranked below the support can rise once the supported cells carry real values,
    and it is then refined in turn. A search that never did this would be refining the
    depth-1 answer's cells and calling the result depth 2.

    Asserted as a property of at least one position rather than of every position, because
    whether the support moves at all is a fact about the position, not about the search.
    """
    reg = roster.reg
    positions = _played(roster)
    moved = 0
    for pos in positions:
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            continue
        shallow = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=1)
        one = search(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix(),
            depth=2, refine=2, passes=1,
        )
        two = search(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix(),
            depth=2, refine=2, passes=2,
        )
        assert two.refined >= one.refined, "a second pass cannot un-refine a cell"
        assert two.refined <= 2 * 2 * 2, f"still bounded: {two.refined}"
        rows = shallow.equilibrium.row_strategy > 1e-9
        cols = shallow.equilibrium.col_strategy > 1e-9
        outside = ~np.outer(rows, cols)
        if ((shallow.payoff != two.payoff) & outside).any():
            moved += 1
    assert moved >= 1, (
        "no position's support moved between passes; either the fixture is too easy or "
        "the iteration is not iterating"
    )


def test_a_refined_cell_is_the_next_position_s_equilibrium(roster) -> None:  # noqa: ANN001
    """The refined number is recomputed by hand from the same definition.

    Resolve the cell's turn, solve each resulting position's own matrix, average over the
    branch probabilities. If the search's cell does not equal that, the number it puts in
    the matrix is not the thing its docstring says it is.
    """
    reg = roster.reg
    positions = _played(roster)
    verified = 0
    for pos in positions:
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            continue
        shallow = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=1)
        deep = search(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=2, refine=1
        )
        changed = np.argwhere(shallow.payoff != deep.payoff)
        for i, j in changed:
            result = resolve_turn(
                reg, pos, [ours[int(i)], theirs[int(j)]], budget=Budget.matrix()
            )
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
            assert deep.payoff[i, j] == pytest.approx(by_hand, abs=1e-12), (
                f"cell ({i},{j}) is {deep.payoff[i, j]}, the definition gives {by_hand}"
            )
            verified += 1
    assert verified >= 1, "no cell was refined, so nothing was verified"


def test_an_unrefinable_cell_keeps_its_depth_one_value(roster) -> None:  # noqa: ANN001
    """A cell the refinement declines leaves the matrix alone.

    `refine` is set past the number of actions so every supported cell is attempted; the
    ones that cannot be refined -- a turn suspended for a mid-turn replacement, a position
    with no legal reply -- must come back unchanged rather than as a guess.
    """
    reg = roster.reg
    positions = _played(roster)
    for pos in positions:
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            continue
        shallow = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=1)
        deep = search(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix(),
            depth=2, refine=99, passes=1,
        )
        same = shallow.payoff == deep.payoff
        # One pass, so the refined set is the depth-1 support and nothing else may move.
        rows = shallow.equilibrium.row_strategy > 1e-9
        cols = shallow.equilibrium.col_strategy > 1e-9
        outside = ~np.outer(rows, cols)
        assert same[outside].all(), (
            "a cell outside the support changed: "
            f"{np.argwhere(~same & outside)}"
        )
        # And inside it, a cell the refinement declined is left exactly as it was, which
        # is only visible as "unchanged" -- so the count is what carries the claim.
        assert deep.refined <= int(np.outer(rows, cols).sum()), (
            f"more cells refined than the support has: {deep.refined}"
        )


def test_the_search_reports_what_it_approximated(roster) -> None:  # noqa: ANN001
    """A refined result says so, and says whether the support settled."""
    reg = roster.reg
    positions = _played(roster)
    saw_refinement = False
    for pos in positions:
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        if not ours or not theirs:
            continue
        deep = search(
            reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=2, refine=2
        )
        if not deep.refined:
            continue
        saw_refinement = True
        assert any("depth-2" in note for note in deep.unmodelled), (
            f"a mixed-depth matrix has to be declared: {sorted(deep.unmodelled)}"
        )
        if not deep.converged:
            assert any("settled" in note for note in deep.unmodelled)
    assert saw_refinement, "no position was refined, so the reporting was not exercised"


def test_depth_two_keeps_the_seat_identity(roster) -> None:  # noqa: ANN001
    """Swapping the sides must swap the answer, at depth 2 as at depth 1.

        V(P) + V(swap(P)) = 1

    hp-share is antisymmetric by construction and neither the resolver nor the LP has any
    business knowing which seat it works for, so the identity holds exactly at depth 1 --
    `test_symmetry.py` is built on it. Depth 2 adds a step that could break it without
    breaking anything else: the refinement picks cells by equilibrium probability, and a
    tie broken by index rather than by something seat-agnostic would refine a different
    set of cells in the two orientations and quietly give one seat a better search.

    That is not hypothetical. A speed tie broken by side index was worth nine points in a
    mirror, and every match result this search produces is confounded if it recurs here.
    """
    reg = roster.reg
    positions = _played(roster)
    assert len(positions) >= 3, "the game ended too early to test anything"
    worst = 0.0
    checked = 0
    for pos in positions:
        mirror = pos.swapped()
        ours = narrow(reg, pos, 0, limit=6).actions
        theirs = narrow(reg, pos, 1, limit=6).actions
        swapped_row = narrow(reg, mirror, 0, limit=6).actions
        swapped_col = narrow(reg, mirror, 1, limit=6).actions
        if not ours or not theirs or not swapped_row or not swapped_col:
            continue
        forward = search(reg, pos, ours, theirs, LEAF, budget=Budget.matrix(), depth=2)
        backward = search(
            reg, mirror, swapped_row, swapped_col, LEAF, budget=Budget.matrix(), depth=2
        )
        total = forward.equilibrium.value + backward.equilibrium.value
        worst = max(worst, abs(total - 1.0))
        checked += 1
    assert checked >= 3, f"only {checked} positions checked"
    # The same 2e-3 tolerance test_symmetry.py uses: the one known asymmetry left is a
    # same-species Speed tie in the residual phase, measured at 0.001 and reported rather
    # than resolved. Anything larger is new, and it would be the search that added it.
    assert worst < 2e-3, f"depth 2 broke the seat identity by {worst:.4f}"


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
