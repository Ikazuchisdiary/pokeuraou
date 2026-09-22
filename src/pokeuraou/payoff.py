"""What the equilibrium is an equilibrium *of*.

A matrix game needs a number in every cell, and at this milestone there is no trained
value function to supply one. The temptation is a weighted heuristic -- so much per KO,
so much per point of HP, a bonus for hazards -- and that is exactly the kind of number
this project refuses to print: the weights would be invented, and every frequency and EV
loss downstream would inherit them without saying so.

So the objectives here are *parameter-free*. Each is a single stated quantity with no
tunable constants, mapped into [0, 1] so it can stand where a win probability will later
stand. None of them is a win probability, and the CLI says so next to every number it
prints.

    hp-share   the side's share of the total HP still on the field and bench
    faints     the side's share of the Pokemon still standing

Neither is the truth. hp-share treats a Pokemon on 1 HP as nearly whole, which
overvalues chip damage; faints ignores damage entirely until it kills, which undervalues
setting up a KO next turn. They fail in opposite directions, which is what makes running
both worth the cost: if the recommended frequencies hold under both, the recommendation
is not an artefact of the objective, and if they diverge the CLI reports that rather than
picking a winner. That disagreement is information about the position, not a defect.

The learned value function of milestone 3 replaces these; it is a drop-in for
:class:`Objective` and the CLI prints whichever it was given.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .position import Position


class Objective(Protocol):
    """A payoff for the row player (side 0), in [0, 1]."""

    #: Short identifier used on the command line.
    name: str
    #: One line naming the quantity, printed with the results.
    formula: str
    #: What the objective is blind to. Printed with the results too.
    blind_to: str

    def __call__(self, pos: Position) -> float: ...

    def batch(self, positions: Sequence[Position]) -> np.ndarray:
        """The same payoff for many positions at once.

        Part of the protocol rather than an optimisation a caller may or may not find.
        A caller holding a whole node's leaves should not be the one deciding whether
        batching is worth it: for a parameter-free objective it costs nothing and this
        loops, and for a learned value function it is 1.86x on a 24x24 matrix over four
        spread classes -- almost none of which is the forward pass itself, it is the
        per-position work around it.
        """
        ...


@dataclass(frozen=True, slots=True)
class _Objective:
    name: str
    formula: str
    blind_to: str
    _value: Callable[[Position], float]
    #: A batch form, when the payoff has one worth using. `None` means loop over `_value`.
    _batch: Callable[[Sequence[Position]], np.ndarray] | None = None
    #: A batch form for leaves that arrive already encoded, or `None`.
    #:
    #: A parameter-free objective has none and never will: it reads the position. A
    #: learned value function's input *is* the encoding, and that is what lets a caller
    #: holding the encoding -- the Rust port, which encodes the leaves it already has --
    #: hand them over as arrays instead of rebuilding positions for this to encode again.
    from_encoded: Callable[[Any], np.ndarray] | None = None

    def __call__(self, pos: Position) -> float:
        return self._value(pos)

    def batch(self, positions: Sequence[Position]) -> np.ndarray:
        """One array for many positions, through `_batch` when one was supplied.

        A parameter-free objective has nothing to gain from batching and loops; a value
        function passes its own batch form in and the loop never runs.
        """
        if self._batch is not None:
            return np.asarray(self._batch(positions), dtype=np.float64)
        return np.fromiter(
            (self._value(p) for p in positions), dtype=np.float64, count=len(positions)
        )


def _decided(pos: Position) -> float | None:
    """1.0 or 0.0 once the battle is over, so a win is never merely a large payoff."""
    if not pos.ended:
        return None
    if pos.winner is None:
        return 0.5
    return 1.0 if pos.winner == pos.sides[0].id else 0.0


def _hp_share(pos: Position) -> float:
    decided = _decided(pos)
    if decided is not None:
        return decided
    remaining = []
    for side in pos.sides:
        live = sum(mon.hp for mon in side.pokemon if not mon.fainted)
        total = sum(mon.maxhp for mon in side.pokemon)
        remaining.append(0.0 if total <= 0 else live / total)
    denominator = remaining[0] + remaining[1]
    # Both sides wiped out on the same turn: the position is a draw, not a division by
    # zero, and `_decided` has already handled the version of that which ends the battle.
    if denominator <= 0:
        return 0.5
    return remaining[0] / denominator


def _faint_share(pos: Position) -> float:
    decided = _decided(pos)
    if decided is not None:
        return decided
    standing = [sum(1 for mon in side.pokemon if not mon.fainted) for side in pos.sides]
    denominator = standing[0] + standing[1]
    if denominator <= 0:
        return 0.5
    return standing[0] / denominator


#: The default. Every point of damage moves it, so it distinguishes positions that
#: `faints` cannot see at all -- which is most positions, most of the time.
HP_SHARE: Objective = _Objective(
    name="hp-share",
    formula="自陣の残 HP / 両陣の残 HP（パーティ4体の合計、決着時は 1/0）",
    blind_to="残り 1 HP と満タンをほぼ同じに扱う。削りを過大評価する",
    _value=_hp_share,
)

#: The cross-check. Runs on the same matrix and is reported beside HP_SHARE.
FAINTS: Objective = _Objective(
    name="faints",
    formula="自陣の生存数 / 両陣の生存数（決着時は 1/0）",
    blind_to="倒れるまでダメージを一切見ない。次ターンの確定数を過小評価する",
    _value=_faint_share,
)

OBJECTIVES: dict[str, Objective] = {HP_SHARE.name: HP_SHARE, FAINTS.name: FAINTS}

DEFAULT_OBJECTIVE = HP_SHARE

__all__ = ["DEFAULT_OBJECTIVE", "FAINTS", "HP_SHARE", "OBJECTIVES", "Objective"]
