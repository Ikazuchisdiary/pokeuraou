"""`Pokemon.copy` shares three dicts, and these tests are the reason that is allowed.

Copying a position is the resolver's most frequent operation -- one per branch per action,
about half a million per generated game -- and three of the dicts it copied are write-once:
a Pokemon's SP spread is fixed for the battle, `ability_state` is only ever read through
`.get`, and `stats_override` is written when Transform copies a target and read from then
on. Sharing them with the original removed three dict constructions per Pokemon.

That is safe exactly as long as nothing writes into them, and "nothing writes into them"
is a claim about every line of the resolver, present and future. Asserting it by reading
the code would hold only until the next commit, so it is asserted by *execution*: the
three dicts are replaced with read-only proxies and real turns are resolved on top. A
write into a shared dict then raises instead of leaking a mutation from one branch of the
search into its sibling -- a defect that would otherwise show up as nothing more than
slightly wrong numbers.

The second test covers the other direction. Every *mutable* field that is genuinely copied
must come out as a distinct object, and the list of fields is read from the dataclass
rather than typed out here, so a field added later is covered without anyone remembering
to come back.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import numpy as np
import pytest

from pokeuraou.damage import register_mega_stones
from pokeuraou.narrow import narrow
from pokeuraou.position import Pokemon, Position
from pokeuraou.resolve import Budget, resolve_turn
from pokeuraou.selfplay import position_from_sets
from pokeuraou.teams import load_roster

#: The fields `Pokemon.copy` deliberately shares instead of copying, and why each is
#: allowed to be shared. Anything not listed here that holds a mutable container must be
#: copied, which is what `test_every_mutable_field_is_copied` checks.
SHARED = {
    "sp": "a spread is fixed for the battle; only ever replaced wholesale",
    "ability_state": "read through .get, never assigned into",
    "stats_override": "written once by Transform, read afterwards",
}


@pytest.fixture(scope="module")
def roster():  # noqa: ANN201
    loaded = load_roster("rizabanadohido")
    register_mega_stones(loaded.reg)
    return loaded


def _freeze_shared(pos: Position) -> None:
    """Make the three shared dicts refuse writes, on every Pokemon in the position."""
    for side in pos.sides:
        for mon in side.pokemon:
            for name in SHARED:
                value = getattr(mon, name)
                if isinstance(value, dict):
                    setattr(mon, name, MappingProxyType(value))


def test_resolving_turns_never_writes_into_a_shared_dict(roster) -> None:  # noqa: ANN001
    """Several turns of three real games, with the shared dicts read-only.

    Mirrors, because they keep a game alive long enough to accumulate the states that
    would plausibly want to write somewhere: statuses, boosts, weather, a depleted bench.
    `Budget.matrix()` rather than `exact()` so that every branch of every turn -- the
    secondaries, the accuracy rolls, the critical hits -- is resolved on a copied
    position, which is where an aliased write would do its damage.

    Three seeds rather than one because a single game can end, or suspend on a
    self-switching move, after two turns; the count asserted at the end is what keeps a
    game that dies early from passing as a test that checked something.
    """
    reg = roster.reg
    resolved = 0
    for seed in (3, 5, 11):
        sets = list(roster.sets[:4])
        pos = position_from_sets(reg, sets, sets)
        _freeze_shared(pos)
        rng = np.random.default_rng(seed)
        for _ in range(6):
            if pos.ended:
                break
            ours = narrow(reg, pos, 0, limit=6).actions
            theirs = narrow(reg, pos, 1, limit=6).actions
            if not ours or not theirs:
                break
            for row in ours[:3]:
                for col in theirs[:3]:
                    # A TypeError from MappingProxyType here is the point of the test.
                    resolve_turn(reg, pos, [row, col], budget=Budget.matrix())
                    resolved += 1
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
    assert resolved >= 60, f"only {resolved} cells resolved; the games ended too early"


def test_the_shared_dicts_are_the_same_object_after_a_copy(roster) -> None:  # noqa: ANN001
    """The sharing itself, so the optimisation cannot be undone without a test saying so."""
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    mon = pos.sides[0].pokemon[0]
    assert mon.sp is not None, "the fixture's own spreads should be known"
    clone = mon.copy()
    for name, reason in SHARED.items():
        original, copied = getattr(mon, name), getattr(clone, name)
        if original is None:
            continue
        assert copied is original, f"{name} is copied again; it is shared because {reason}"


def test_every_mutable_field_is_copied(roster) -> None:  # noqa: ANN001
    """Anything mutable and not in `SHARED` must be a distinct object after `copy()`.

    Read from `dataclasses.fields`, so a mutable field added to `Pokemon` later is
    covered by this test on the day it is added rather than on the day it corrupts a
    branch.
    """
    reg = roster.reg
    sets = list(roster.sets[:4])
    pos = position_from_sets(reg, sets, sets)
    mon = pos.sides[0].pokemon[0]
    mon.boosts["atk"] = 1
    mon.unmodelled_volatiles.append("something")
    clone = mon.copy()

    checked = []
    for field in dataclasses.fields(Pokemon):
        if field.name in SHARED:
            continue
        value = getattr(mon, field.name)
        if not isinstance(value, (dict, list, set, bytearray)):
            continue  # tuples, strings and numbers are immutable; sharing them is free
        assert getattr(clone, field.name) is not value, (
            f"{field.name} is shared with the original but is mutable: a write in one "
            f"branch of the search would appear in its siblings"
        )
        checked.append(field.name)
    assert "moves" in checked and "boosts" in checked, f"only checked {checked}"
