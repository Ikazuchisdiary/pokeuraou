"""Seat symmetry: swapping the two sides must swap the answer and nothing else.

Side 0 is always our roster and side 1 is always a tournament team, so a seat effect does
not cancel out in the training data -- it is a constant added to every label. And it cannot
be found by playing: 420 games per arm put the estimate at +-2.7 points, which was not
enough to tell -0.07 from zero. It can be found exactly, because ``hp-share`` is
antisymmetric by construction and the resolver has no business knowing which seat it is
working for:

    V(P, [a, b]) + V(swap(P), [b, a]) = 1

for *any* position and any action pair. That identity found two real defects, and both are
regression-tested here:

- **a mutual knockout went to side 0 every time.** The old code looped over the sides
  assigning a winner per wiped side, so with both wiped the second iteration overwrote the
  first. Showdown's ``checkWin`` gives gen 5+ the side of the Pokemon that fainted *last*,
  and ends the battle at the faint that empties a side rather than at the end of the turn;
  both readings are "whichever wipe-out completed last wins".
- **the residual phase ran in side order.** Showdown speed-sorts ``getAllActive()`` for
  ``eachEvent('Residual')``, so a burn and a trap on our side used to resolve before their
  poison in *both* orientations, and a faint the residuals caused landed on whichever seat
  we happened to hold. Worth 4.8 points on one measured cell.

What remains, deliberately: an exact Speed tie between two actives of the same species
falls back to a side-dependent order, because Showdown breaks such ties at random and the
residual phase does not branch. It is reported through ``unmodelled`` rather than hidden,
and it was 1 of 2400 sampled action pairs at a magnitude of 0.001.

Positions here are built by *playing*, several turns in, rather than constructed by hand:
statuses, boosts, weather, traps and depleted benches appear in the proportions the search
really meets them, and every defect above lived past turn 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokeuraou.actions import MoveAction, SideAction
from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.payoff import HP_SHARE
from pokeuraou.position import Position
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

from ._port import Budget, resolve_turn, turn_expectation

REGULATION = "gen9championsvgc2026regmb"


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _played_positions(roster, turns: int = 6, seed: int = 3) -> list[Position]:  # noqa: ANN001
    """Positions from a real game, one per turn, mirror on both sides.

    A mirror keeps the game going long enough to accumulate state -- and makes any
    asymmetry doubly visible, since the two sides are otherwise identical.
    """
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
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
        pos = result.branches[int(rng.choice(len(weights), p=weights / weights.sum()))].position
    return out


def test_swapping_the_sides_swaps_the_answer(roster) -> None:  # noqa: ANN001
    """The identity, on positions a game actually reached.

    Tolerance is 2e-3, not zero: the one known asymmetry left is a same-species Speed tie
    in the residual phase, which is reported through `unmodelled` and was measured at
    0.001. Anything larger is a new defect, and the two this file documents were 1.0 and
    0.048.
    """
    reg = roster.reg
    positions = _played_positions(roster)
    assert len(positions) >= 4, "the game ended too early to test anything"
    rng = np.random.default_rng(11)
    checked = 0
    for pos in positions:
        row = narrow(reg, pos, 0, limit=6).actions
        col = narrow(reg, pos, 1, limit=6).actions
        if not row or not col:
            continue
        mirror = pos.swapped()
        mirror_row = {a.to_choice(): a for a in narrow(reg, mirror, 0, limit=6).actions}
        mirror_col = {a.to_choice(): a for a in narrow(reg, mirror, 1, limit=6).actions}
        for _ in range(4):
            ours = row[int(rng.integers(len(row)))]
            theirs = col[int(rng.integers(len(col)))]
            a, b = ours.to_choice(), theirs.to_choice()
            # The swapped position's own narrowing must offer the swapped candidates.
            assert a in mirror_col, f"narrowing is seat-dependent: {a} missing"
            assert b in mirror_row, f"narrowing is seat-dependent: {b} missing"
            forward, _ = turn_expectation(
                reg, resolve_turn(reg, pos, [ours, theirs], budget=Budget.matrix()), HP_SHARE
            )
            backward, _ = turn_expectation(
                reg,
                resolve_turn(
                    reg, mirror, [mirror_row[b], mirror_col[a]], budget=Budget.matrix()
                ),
                HP_SHARE,
            )
            checked += 1
            assert forward + backward == pytest.approx(1.0, abs=2e-3), (
                f"turn {pos.turn}, ours {a} vs theirs {b}: "
                f"{forward:.6f} + {backward:.6f}"
            )
    assert checked >= 8


def test_a_mutual_knockout_goes_to_the_side_that_ran_out_last(roster) -> None:  # noqa: ANN001
    """Showdown's rule, and the one the old code got backwards for side 0.

    Built rather than played: a double wipe-out is rare enough that no sampled game
    reaches one, which is exactly why it went unnoticed. Both sides are down to a single
    Incineroar at 1 HP and both use Flare Blitz. The faster one knocks the other out and
    then dies to its own recoil, so the *last* faint is the attacker's -- and the win is
    the attacker's too.

    Asserted in both orientations, with no branch skipped: an earlier version of this test
    let the "not a double wipe-out" case fall through, and passed while checking nothing.

    The budget has to enumerate the Speed tie, because two identical Incineroars are one.
    ``matrix`` and ``exact`` branch on it and come out at 0.5 from either seat;
    ``deterministic``, which pins every roll for the differential harness, breaks the tie
    by side index and would report a win for side 0 from both seats. That is a known
    property of a pinned budget rather than a seat bias in the resolver -- play and
    analysis both enumerate -- and it is why this test does not use it.
    """
    reg = roster.reg
    # Incineroar is the only member with a recoil move, so it has to be the survivor.
    sets = [roster.sets[5], roster.sets[1], roster.sets[2], roster.sets[3]]
    assert "flareblitz" in sets[0].moves

    def one_each() -> Position:
        pos = position_from_sets(reg, sets, sets)
        for side in pos.sides:
            for mon in side.pokemon[1:]:
                mon.hp = 0
                mon.fainted = True
                mon.status = "fnt"
            side.active = [0, None]
            side.pokemon[0].hp = 1
        return pos

    action = SideAction(
        slots=(MoveAction(slot=0, move_index=0, move_id="flareblitz", target=1),)
    )
    pos = one_each()
    result = resolve_turn(reg, pos, [action, action], budget=Budget.matrix(), events=True)
    assert result.branches
    for branch in result.branches:
        after = branch.position
        wiped = [
            index
            for index, side in enumerate(after.sides)
            if all(m.fainted for m in side.pokemon)
        ]
        assert wiped == [0, 1], f"the setup must wipe both sides, got {wiped}"
        assert after.ended
        faints = [line for line in branch.events if line.endswith("fainted")]
        assert len(faints) == 2, branch.events
        expected = after.sides[0].id if faints[-1].startswith("p1") else after.sides[1].id
        assert after.winner == expected, (
            f"last faint {faints[-1]!r} must win; events {branch.events}"
        )

    # The identity: the same turn from the other seat must give the other winner, so the
    # two values sum to one. This is the assertion the old code failed -- it named side 0
    # in both orientations, for a residual of exactly +1.
    forward, _ = turn_expectation(reg, result, HP_SHARE)
    mirrored = resolve_turn(
        reg, one_each().swapped(), [action, action], budget=Budget.matrix()
    )
    backward, _ = turn_expectation(reg, mirrored, HP_SHARE)
    assert forward + backward == pytest.approx(1.0, abs=1e-12)
    # The tie is enumerated, so each seat wins exactly half of the branches. The old code
    # gave both seats 1.0 -- a residual of exactly +1, the largest a cell can have.
    assert forward == pytest.approx(0.5) and backward == pytest.approx(0.5)


def test_a_one_sided_knockout_goes_to_the_survivor(roster) -> None:  # noqa: ANN001
    """The other half of Showdown's `checkWin`, on the port: one side out, the other wins,
    from either seat -- and a turn in which nobody runs out leaves the battle going.

    Was a table over Python's `settle_outcome` (IKA-210: the resolver's own function goes
    with it). The both-out half is the mutual knockout above; the "arrived already empty,
    no order recorded" row was a Python-only input no turn produces.
    """
    reg = roster.reg
    sets = [roster.sets[2], roster.sets[0], roster.sets[1], roster.sets[3]]
    assert sets[0].species == "garchomp"

    def last_each(hp_behind: int) -> Position:
        pos = position_from_sets(reg, sets, sets)
        for side in pos.sides:
            for mon in side.pokemon[1:]:
                mon.hp = 0
                mon.fainted = True
                mon.status = "fnt"
            side.active = [0, None]
        pos.sides[1].pokemon[0].hp = hp_behind
        return pos

    claw = SideAction(
        slots=(MoveAction(slot=0, move_index=1, move_id="dragonclaw", target=1),)
    )
    # Stomping Tantrum cannot take a full-HP Garchomp, so side 0 survives in every
    # branch (and, unlike Rock Slide, cannot flinch it out of its Dragon Claw).
    slide = SideAction(
        slots=(MoveAction(slot=0, move_index=2, move_id="stompingtantrum", target=1),)
    )

    for flipped in (False, True):
        pos = last_each(1)
        actions = [claw, slide]
        if flipped:
            pos, actions = pos.swapped(), [slide, claw]
        survivor = 1 if flipped else 0
        result = resolve_turn(reg, pos, actions, budget=Budget.matrix())
        assert result.branches and not result.suspended
        for branch in result.branches:
            after = branch.position
            assert all(m.fainted for m in after.sides[1 - survivor].pokemon)
            assert after.ended and after.winner == after.sides[survivor].id, flipped

    # Nobody out: the battle goes on (the Garchomp behind has its full HP this time).
    full = last_each(1)
    full.sides[1].pokemon[0].hp = full.sides[1].pokemon[0].maxhp
    result = resolve_turn(reg, full, [slide, slide], budget=Budget.matrix())
    assert result.branches
    for branch in result.branches:
        assert not branch.position.ended and branch.position.winner is None


def test_swapping_relocates_the_effects_that_name_a_side(roster) -> None:  # noqa: ANN001
    """`Position.swapped` must move `source_slot`, or the mirror is not a mirror.

    This is what made the first run of the identity report a seven-point violation that
    was not there: the trap said its owner was side 0, the swap left that alone, and the
    mirrored turn released a trap the real one kept.
    """
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    from pokeuraou.position import Effect

    pos.sides[1].pokemon[0].volatiles.append(
        Effect(id="partiallytrapped", duration=4, source_slot="00")
    )
    pos.sides[0].pokemon[1].volatiles.append(
        Effect(id="leechseed", source_slot="p2b")
    )
    flipped = pos.swapped()

    trap = flipped.sides[0].pokemon[0].volatile("partiallytrapped")
    assert trap is not None
    assert trap.source_slot == "10", "the trapper is now on side 1"
    seed = flipped.sides[1].pokemon[1].volatile("leechseed")
    assert seed is not None
    assert seed.source_slot == "p1b", "Showdown-style references flip too"
    # And swapping twice is the identity.
    again = flipped.swapped()
    assert again.sides[1].pokemon[0].volatile("partiallytrapped").source_slot == "00"
    assert again.sides[0].pokemon[1].volatile("leechseed").source_slot == "p2b"


def test_a_poison_type_never_misses_toxic(roster) -> None:  # noqa: ANN001
    """From gen 8, Toxic used by a Poison type cannot miss -- and the move data hides it.

    `toxic` still says `accuracy: 90`; the exemption lives in `Scripts#tryMoveHit`, behind
    a comment in the move's own entry pointing at it. Reading the dump alone leaves a
    Poison type's Toxic failing one time in ten, which is what our Toxapex was doing in
    every generated game.

    Both directions are asserted from one position: Toxapex is Poison/Water, and giving it
    an explicit non-Poison typing -- the field Protean and Soak write -- must bring the
    miss branch back. A test that only checked the Poison case would pass just as well if
    the rule had been written as "Toxic never misses".
    """
    reg = roster.reg
    poisoner = roster.sets[4]
    assert poisoner.species == "toxapex" and "toxic" in poisoner.moves
    sets = [poisoner, roster.sets[1], roster.sets[2], roster.sets[3]]
    index = poisoner.moves.index("toxic")

    def outcome(types: tuple[str, ...] | None) -> set[bool]:
        pos = position_from_sets(reg, sets, sets)
        if types is not None:
            pos.sides[0].pokemon[0].types = types
        # The target is the other side's Toxapex, and a Poison type cannot be poisoned: the
        # Toxic that hits it fails, and so does the one that misses (Showdown rolls the 90
        # first, then `trySetStatus` fails), so the two branches are one state and merge --
        # the miss branch would vanish for a reason that has nothing to do with the user's
        # typing (IKA-171). A non-Poison target keeps the hit (it is badly poisoned) and
        # the miss apart.
        pos.sides[1].pokemon[0].types = ("Water",)
        ours = SideAction(
            slots=(
                MoveAction(slot=0, move_index=index, move_id="toxic", target=1),
                MoveAction(slot=1, move_index=3, move_id="protect", target=None),
            )
        )
        # The target must not be behind a Protect: blocking the very move under test is
        # how a test like this passes while checking nothing. Infestation is Toxapex's
        # attacking move, so slot 0 stands there and takes it.
        theirs = SideAction(
            slots=(
                MoveAction(slot=0, move_index=0, move_id="infestation", target=1),
                MoveAction(slot=1, move_index=3, move_id="protect", target=None),
            )
        )
        result = resolve_turn(reg, pos, [ours, theirs], budget=Budget.exact(), events=True)
        return {
            any("Toxic" in line and "missed" in line for line in branch.events)
            for branch in result.branches
        }

    assert outcome(None) == {False}, "a Poison type's Toxic must not have a miss branch"
    assert True in outcome(("Water",)), "without the typing the 90% must come back"
