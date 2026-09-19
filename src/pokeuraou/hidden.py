"""What the opponent has not shown yet, and the positions it could be.

Champions shows open team sheets, so the opponent's *six* is public. Which **four** they
brought is not: two lead, and the other two arrive when they arrive. Counting it on
recorded games, **47.4% of all move decisions and 100% of opening decisions** were taken
by a search that had been handed the answer -- side 1's whole four sits in the position
from turn 1, and `narrow` enumerates switches to Pokemon nobody has seen.

:mod:`pokeuraou.belief` already does this job for the other hidden thing, the SP spread,
and the shape here is the same: enumerate what is consistent with what has been seen,
weight it, and let the caller average over it. The difference is that a spread is a number
the position carries and an unrevealed bench member is a *whole Pokemon*, so the
enumeration produces positions rather than particles.

## What is public and what is not

Public: the six on the sheet, that they brought four, and how many of those four are still
unseen -- the count is on the screen. So the uncertainty is exactly "which of the sheet
members we have not met are in those slots", and there are at most C(4,2) = 6 of them.
That is small enough to average over exactly rather than sample.

## Replacing an unseen Pokemon is lossless

A Pokemon that has never been active carries no state: full HP, no status, no boosts, no
volatiles, its moves at full PP. So a completion does not edit a Pokemon, it builds a
fresh one from the sheet -- the same call `position_from_sets` makes -- and the result is
a position that could really have arisen. Nothing downstream needs to know it was
substituted, and in particular the value function is shown a position of exactly the kind
it was trained on rather than a masked one it has never seen.

What this does *not* do is decide how to use the completions. Averaging a payoff matrix
over them is one answer, and the one a solver wants, since the mixture it reports should
be the mixture against the belief rather than against a guess; playing the most likely one
is another. This module only makes the set, and says what each member is worth.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from .position import Pokemon, Position, Side

if TYPE_CHECKING:  # pragma: no cover - imported for types only; selfplay imports us
    from .priors import SampledSet
    from .regulation import Regulation


@dataclass(frozen=True, slots=True)
class Completion:
    """One way the unseen slots could be filled, and how much belief it carries."""

    position: Position
    #: Sheet species placed into `slots`, in that order. The identity of the guess, so a
    #: caller can report what it averaged over rather than only the average.
    species: tuple[str, ...]
    slots: tuple[int, ...]
    weight: float
    #: True when nothing was hidden, so this is the position itself. A caller that wants
    #: to know whether it is paying for uncertainty asks this rather than counting.
    exact: bool = False


def seen_slots(
    position: Position, side_index: int, so_far: frozenset[int] = frozenset()
) -> frozenset[int]:
    """Slots the opponent has shown by the time this position is on the board.

    `so_far` carries what earlier turns revealed, because a Pokemon that came in and went
    back out is still known and the position alone no longer says so. A caller that has
    not been tracking passes nothing and gets the conservative answer: everything the
    position itself gives away.

    Conservative means *over*-revealing. Treating a hidden Pokemon as seen costs only the
    thing this module exists to remove, and treating a seen one as hidden would have the
    search reason about a Pokemon the opponent has already shown -- a mistake that reads
    as a bad evaluation rather than as a bookkeeping bug.
    """
    side = position.sides[side_index]
    shown = set(so_far)
    for mon in side.pokemon:
        if (
            mon.active_index is not None
            or mon.fainted
            or mon.hp != mon.maxhp
            or mon.status is not None
            or any(mon.boosts.values())
            or mon.volatiles
            or mon.is_mega
        ):
            shown.add(mon.slot)
    return frozenset(shown)


def shown_species(
    position: Position, side_index: int, seen: frozenset[int]
) -> set[str]:
    """Species ids that side has revealed, with each Mega's base form alongside it.

    A Mega on the board is `charizardmegay` and its sheet entry is `charizard`, so
    matching on `species` alone leaves the base form available to fill a hidden slot and
    the belief gains completions holding a second copy of a Pokemon already out. With one
    Mega and two unseen slots that was 4 of 10 completions -- 40% of the mass on worlds
    Species Clause forbids -- and it never tripped `completions`' arity check, because an
    extra candidate makes the sheet look MORE able to explain the board, not less.
    `baseSpecies` arrives from Showdown as a display name, so both go through `to_id`.

    Lives here rather than inside `completions` because the weights that price those
    completions have to condition on the same set, and two spellings of "what has been
    seen" is how the belief and its weights come apart.
    """
    from .regulation import to_id

    out: set[str] = set()
    for mon in position.sides[side_index].pokemon:
        if mon.slot not in seen:
            continue
        out.add(to_id(mon.species))
        out.add(to_id(mon.base_species))
    return out


def completions(
    reg: Regulation,
    position: Position,
    side_index: int,
    sheet: Sequence[SampledSet],
    *,
    seen: frozenset[int] | None = None,
    weights: dict[tuple[str, ...], float] | None = None,
) -> list[Completion]:
    """Every position consistent with what has been seen of `side_index`'s team.

    `sheet` is that side's six. Members whose species is already on the board are the ones
    they brought and showed; the rest are the candidates, and the unseen slots are filled
    from them in every combination. The order within a combination is not a degree of
    freedom worth spending: two unseen bench slots are interchangeable until one of them
    is switched in, and the positions they produce differ only by a relabelling.

    `weights` prices the combinations, keyed by the sorted species tuple -- the opponent's
    selection equilibrium marginalised onto their back two is what should fill it. Missing
    or all-zero means uniform, which is the honest prior when no book is loaded.

    A side with nothing hidden returns the position itself, weight 1, `exact`. That is the
    52.6% of decisions where this costs nothing, and it is returned rather than special
    cased so a caller has one code path.
    """
    from .selfplay import _make_pokemon  # circular at module scope: selfplay imports us

    side = position.sides[side_index]
    shown = seen_slots(position, side_index) if seen is None else seen
    hidden = [mon.slot for mon in side.pokemon if mon.slot not in shown]
    if not hidden:
        return [Completion(position, (), (), 1.0, exact=True)]

    from .regulation import to_id

    on_board = shown_species(position, side_index, shown)
    candidates = [entry for entry in sheet if to_id(entry.species) not in on_board]
    if len(candidates) < len(hidden):
        # The sheet cannot explain the board, so something upstream handed us the wrong
        # six. Every caller's fallback would be to carry on with the truth it already
        # holds, which is the leak this module exists to close; better to say so.
        raise ValueError(
            f"{len(hidden)} unseen slots on side {side_index} but only "
            f"{len(candidates)} sheet members unaccounted for"
        )

    made: list[Completion] = []
    for choice in combinations(range(len(candidates)), len(hidden)):
        picked = [candidates[index] for index in choice]
        key = tuple(sorted(entry.species for entry in picked))
        made.append(
            Completion(
                position=substitute(reg, position, side_index, hidden, picked, _make_pokemon),
                species=tuple(entry.species for entry in picked),
                slots=tuple(hidden),
                weight=float((weights or {}).get(key, 0.0)),
            )
        )
    total = sum(item.weight for item in made)
    if total <= 0.0:
        share = 1.0 / len(made)
        return [
            Completion(item.position, item.species, item.slots, share) for item in made
        ]
    return [
        Completion(item.position, item.species, item.slots, item.weight / total)
        for item in made
    ]


def substitute(
    reg: Regulation,
    position: Position,
    side_index: int,
    slots: Sequence[int],
    sets: Sequence[SampledSet],
    make: Callable[[Regulation, int, SampledSet, int | None], Pokemon],
) -> Position:
    """The position with `slots` rebuilt from `sets`, everything else untouched."""
    original = position.sides[side_index]
    replacement = {
        slot: make(reg, slot, entry, None)
        for slot, entry in zip(slots, sets, strict=True)
    }
    mons = [replacement.get(mon.slot, mon).copy() for mon in original.pokemon]
    # Mega capability follows the species and the item, so it cannot be carried over from
    # a side that held a different Pokemon in that slot.
    side = Side(
        id=original.id,
        name=original.name,
        active=list(original.active),
        pokemon=mons,
        side_conditions=[effect.copy() for effect in original.side_conditions],
        slot_conditions=[
            [effect.copy() for effect in condition]
            for condition in original.slot_conditions
        ],
        mega_capable_slots=[
            mon.slot for mon in mons if reg.mega_target(mon.species, mon.item) is not None
        ],
        mega_used=original.mega_used,
    )
    sides = list(position.sides)
    sides[side_index] = side
    swapped = position.copy()
    swapped.sides = sides
    return swapped


__all__ = ["Completion", "completions", "seen_slots", "substitute"]
