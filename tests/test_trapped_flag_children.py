"""Showdown's `trapped` flag belongs to the root, not to its children (IKA-175).

`Pokemon.trapped` is Showdown's own verdict, read from an oracle position, and
`actions._is_trapped` takes it before anything else. `Pokemon.copy` carries it into every
position the resolver builds, so a search started from Showdown kept a Pokemon trapped in
every child -- after its Shadow Tag trapper had fainted, say. Showdown clears it in
`endTurn` (`sim/battle.ts`)::

    pokemon.trapped = pokemon.maybeTrapped = false;
    this.runEvent('TrapPokemon', pokemon);

and a benched Pokemon lost it in `clearVolatile`. The resolver now drops it at the same
point, in Python (`resolve._clear_trapped`, after the residuals and after the faint
replacements) and in the port (`run_queue`), and a child's trap is read off the position
alone, which is what generation has always done.

Every turn here is the port's (IKA-210); the pre-IKA-175 binary returns the flag set.
"""

from __future__ import annotations

import pytest

from pokeuraou.actions import SwitchAction, side_actions, switch_actions_after_faint
from pokeuraou.oracle import Oracle, RandomnessPolicy
from pokeuraou.position import Effect, Position

from ._port import Budget, replacements_needed, resolve_replacements, resolve_turn
from .conftest import FORMAT_ID
from .test_trap_sources import (
    MEGA_TURN,
    QUIET_B,
    SUBJECTS,
    TAGGERS,
    _hand_built,
    _lead,
)


def _switches(reg, pos: Position, side: int, slot: int) -> set[int]:  # noqa: ANN001
    return {
        a.slots[slot].party_index
        for a in side_actions(reg, pos, side)
        if isinstance(a.slots[slot], SwitchAction)
    }


def _choose(reg, pos: Position, choices: list[str]):  # noqa: ANN001, ANN202
    chosen = []
    for side, choice in enumerate(choices):
        menu = {a.to_choice(): a for a in side_actions(reg, pos, side)}
        assert choice in menu, (choice, sorted(menu))
        chosen.append(menu[choice])
    return chosen


def _flags(pos: Position) -> list[bool]:
    return [m.trapped for s in pos.sides for m in s.pokemon]


# Garchomp Swords Dances and Milotic Recovers; Hippowdon Slack Offs and Milotic Recovers.
# Nothing is random and nothing traps.
QUIET_TURN = ["move 1, move 1", "move 1, move 1"]


def _flagged(reg):  # noqa: ANN001, ANN202
    """A hand-built position with no trap source, and Showdown's flag on p1's first slot."""
    pos = _hand_built(reg, SUBJECTS["control"], QUIET_B)
    pos.sides[0].pokemon[pos.sides[0].active[0]].trapped = True
    return pos


def test_the_root_flag_decides_the_root(reg) -> None:  # noqa: ANN001
    """`_is_trapped`'s first check: at the root, Showdown's verdict is the verdict."""
    pos = _flagged(reg)
    assert _switches(reg, pos, 0, 0) == set()
    assert _switches(reg, pos, 0, 1), "the partner carries no flag"
    # The positive control: the flag is all that traps it.
    pos.sides[0].pokemon[pos.sides[0].active[0]].trapped = False
    assert _switches(reg, pos, 0, 0) == {3, 4}


def test_a_child_does_not_inherit_the_flag(reg) -> None:  # noqa: ANN001
    pos = _flagged(reg)
    bench = pos.sides[1].pokemon[2]
    assert not bench.is_active
    bench.trapped = True  # a benched Pokemon has none in Showdown; dropped all the same
    result = resolve_turn(reg, pos, _choose(reg, pos, QUIET_TURN), budget=Budget.matrix())
    assert result.branches and not result.suspended
    assert pos.sides[0].pokemon[0].trapped, "the root itself is not touched"
    for branch in result.branches:
        # The menu first: that is the behaviour; the flag is why.
        assert _switches(reg, branch.position, 0, 0) == {3, 4}
        assert not any(_flags(branch.position)), _flags(branch.position)


def test_a_trap_on_the_position_survives_into_the_child(reg) -> None:  # noqa: ANN001
    """The control: with Ingrain on it, the child is trapped by the position instead."""
    pos = _flagged(reg)
    mon = pos.sides[0].pokemon[pos.sides[0].active[0]]
    mon.volatiles.append(Effect(id="ingrain"))
    result = resolve_turn(reg, pos, _choose(reg, pos, QUIET_TURN), budget=Budget.matrix())
    for branch in result.branches:
        child = branch.position
        assert not any(_flags(child))
        assert child.sides[0].pokemon[0].has_volatile("ingrain")
        assert _switches(reg, child, 0, 0) == set()
        assert _switches(reg, child, 0, 1), "the partner is free"


def test_the_replacements_drop_it_too(reg) -> None:  # noqa: ANN001
    """A root at a faint replacement: `endTurn` runs after it, so the next turn is clean."""
    pos = _flagged(reg)
    gone = pos.sides[1].pokemon[pos.sides[1].active[0]]
    gone.hp = 0
    gone.fainted = True
    needed = replacements_needed(pos)
    choices = [
        switch_actions_after_faint(reg, pos, side, list(needed[side]))[0] for side in range(2)
    ]
    result = resolve_replacements(reg, pos, choices)
    assert result.position.sides[1].pokemon[0].species != gone.species, "a replacement came in"
    assert _switches(reg, result.position, 0, 0) == {3, 4}
    assert not any(_flags(result.position))


# ---------------------------------------------------------------------------
# From Showdown: Mega Gengar's Shadow Tag traps both of p1's actives.

#: p1's Garchomp Dragon Claws Gengar (at 1 HP, so it faints); Milotic Recovers. Gengar
#: Shadow Balls Garchomp, Castform Protects.
KO_TURN = ["move 4 1, move 1", "move 2 1, move 1"]
#: Everybody Protects; Gengar stays, and so does its trap.
PROTECT_TURN = ["move 2, move 2", "move 1, move 1"]


def _tagged_root(oracle: Oracle) -> Position:
    handle = oracle.create(
        FORMAT_ID, _lead(SUBJECTS["control"]), TAGGERS, policy=RandomnessPolicy()
    )
    handle.step(["team 1234", "team 1234"])
    handle.step(MEGA_TURN)
    assert handle.choice_errors == [], handle.choice_errors
    pos = Position.from_json(handle.position)
    handle.close()
    assert [pos.sides[0].pokemon[i].trapped for i in pos.sides[0].active] == [True, True]
    return pos


@pytest.mark.oracle
def test_a_fainted_trapper_frees_the_child(reg, oracle: Oracle) -> None:  # noqa: ANN001
    root = _tagged_root(oracle)
    assert _switches(reg, root, 0, 0) == set() and _switches(reg, root, 0, 1) == set()
    gengar = root.sides[1].pokemon[root.sides[1].active[0]]
    assert gengar.species == "gengarmega" and gengar.ability == "shadowtag"
    gengar.hp = 1
    result = resolve_turn(reg, root, _choose(reg, root, KO_TURN), budget=Budget.matrix())
    assert result.branches
    for branch in result.branches:
        child = branch.position
        assert child.sides[1].pokemon[child.sides[1].active[0]].fainted
        assert _switches(reg, child, 0, 0) and _switches(reg, child, 0, 1)
        assert not any(_flags(child)), _flags(child)


@pytest.mark.oracle
def test_a_standing_trapper_still_traps_the_child(reg, oracle: Oracle) -> None:  # noqa: ANN001
    """The control: Shadow Tag is read off the child, not off the flag."""
    root = _tagged_root(oracle)
    result = resolve_turn(reg, root, _choose(reg, root, PROTECT_TURN), budget=Budget.matrix())
    assert result.branches
    for branch in result.branches:
        child = branch.position
        assert not any(_flags(child))
        assert _switches(reg, child, 0, 0) == set() and _switches(reg, child, 0, 1) == set()

