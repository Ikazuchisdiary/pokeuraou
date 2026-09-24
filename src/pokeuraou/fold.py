"""A turn's leaves and the fold that turns their values into the turn's value.

Chance is an `Average` and a mid-turn replacement a `BestOf`. Moved out of resolve.py
(IKA-209): the port builds these trees now (`port.turn_leaves`, and the folds it sends
with an encoded node), and resolve.py re-exports them for the callers it still has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .position import Position


@dataclass
class LeafRef:
    """One evaluated position."""

    index: int


@dataclass
class Average:
    """What chance decides: the weighted mean of its parts, normalised by their weight."""

    parts: list[tuple[float, Fold]] = field(default_factory=list)


@dataclass
class BestOf:
    """What a player decides: the option that side likes most.

    Side 0 is the maximiser the payoff matrix is written for, so side 1 choosing means
    taking the value it likes least.
    """

    chooser: int
    options: list[Fold] = field(default_factory=list)


Fold = LeafRef | Average | BestOf


def fold_value(node: Fold, values: Sequence[float]) -> float:
    """Collapses a fold tree against one value per leaf position."""
    if isinstance(node, LeafRef):
        return float(values[node.index])
    if isinstance(node, BestOf):
        scored = [fold_value(option, values) for option in node.options]
        if not scored:
            return 0.0
        return max(scored) if node.chooser == 0 else min(scored)
    total = sum(weight for weight, _ in node.parts)
    if total <= 0:
        return 0.0
    return sum(weight * fold_value(part, values) for weight, part in node.parts) / total


@dataclass
class TurnLeaves:
    """A turn's leaf positions and the fold that turns their values into the turn's value."""

    positions: list[Position]
    root: Fold
    unmodelled: tuple[str, ...] = ()

    def value(self, values: Sequence[float]) -> float:
        return fold_value(self.root, values)

    def shifted(self, offset: int) -> Fold:
        """The fold re-indexed for a caller that appended these positions to a larger batch.

        The search evaluates every leaf in a node in one forward pass, so each cell's
        positions land at an offset in a shared list.
        """
        return _shift(self.root, offset)


def _shift(node: Fold, offset: int) -> Fold:
    """Re-indexes a sub-tree's leaves after its positions were appended to a larger list."""
    if isinstance(node, LeafRef):
        return LeafRef(index=node.index + offset)
    if isinstance(node, BestOf):
        return BestOf(
            chooser=node.chooser, options=[_shift(o, offset) for o in node.options]
        )
    return Average(parts=[(w, _shift(part, offset)) for w, part in node.parts])


def _fold_from_json(node: dict) -> Fold:
    """The fold tree the port describes, as the objects `fold_value` already folds.

    Chance is an `Average` normalised by its own weight and a replacement is a `BestOf`
    taken by whoever chooses it -- the same two shapes `turn_leaves` builds here, so the
    collapse itself is not duplicated.
    """
    if "leaf" in node:
        return LeafRef(index=int(node["leaf"]))
    if "best" in node:
        return BestOf(
            chooser=int(node["best"]),
            options=[_fold_from_json(option) for option in node["options"]],
        )
    return Average(parts=[(float(w), _fold_from_json(part)) for w, part in node["avg"]])


__all__ = ["Average", "BestOf", "Fold", "LeafRef", "TurnLeaves", "fold_value"]
