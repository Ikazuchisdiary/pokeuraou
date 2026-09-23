"""One node solved over several possible benches, resolving each turn once.

The straightforward way to price a turn when the opponent's back two are unknown is to
build a payoff matrix per completion, and that is what :func:`pokeuraou.search.belief_search`
did first. It costs six matrices a turn against one -- 2.84 completions for side 0's
opponent and 3.16 for side 1's, one solve each side -- and measured on the machine that
came to 8.2x, which is more than the answer is worth.

Most of it is not the answer. Timed at the boundary, a matrix is 44.6 ms of resolving and
encoding against 5.2 ms of forward pass, and **an unrevealed benched Pokemon takes no part
in the turn unless it is switched in**. So for the cells where none of them reaches the
field, the turn resolves identically in every completion and the leaves differ only in who
is standing on the bench. Counted over opening nodes, that is **84.3% of cells**.

So this resolves those cells once, on the true position, and builds each completion's
leaves by overwriting the bench rows of the encoding -- every per-Pokemon array is
``(B, 2, M, ...)``, so a slot is one assignment. The true position is only a carrier: the
Pokemon it holds in those slots take no part in the cells being shared, and their rows are
overwritten before anything is scored. Nothing about them reaches a payoff.

The remaining cells are resolved per completion, restricted with ``cells=``:

- the opponent's action switches into a hidden slot;
- our action carries a move that forces their switch;
- the turn suspends for a replacement, whose choices include the hidden slots.

Both sides' games share the one resolution, because their hidden slots are disjoint and a
shared cell's resolution depends on neither. That is what takes 6.0x to a projected 2.2x.

## What has to be true, and is asserted rather than assumed

That a patched encoding equals the encoding of the same completion resolved from scratch.
If it does not -- a feature that reads the bench, a mechanic that counts party members --
the payoffs stay plausible and are wrong, and nothing downstream could tell. `tests/
test_beliefnode.py` compares the two, cell by cell, and the fast path is only taken when
the port is available to be compared against.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .actions import SideAction
from .position import Position
from .regulation import Regulation
from .resolve import Budget, batched_payoff, batched_payoffs, fold_value

#: Moves that put a Pokemon on the field without its owner choosing it, so a cell carrying
#: one can reach a hidden slot whatever the opponent declared. Listed rather than derived
#: because a missing entry is silent: the cell would be shared, the substitution would be
#: wrong, and the payoff would still look like a payoff.
PHAZING_MOVES = frozenset({"dragontail", "circlethrow", "whirlwind", "roar"})


@dataclass
class BeliefNode:
    """Payoff matrices per side, over the completions that side cannot see."""

    #: `matrices[s][k]` is the payoff to side 0 when side `1 - s`'s bench is completion
    #: `k`. Always oriented to side 0, because that is what `batched_payoff` produces and
    #: what every recorded position means by "own"; a caller solving for side 1 transposes.
    matrices: dict[int, list[np.ndarray]]
    unmodelled: set[str] = field(default_factory=set)
    #: Cells resolved once and shared, and cell-resolutions that had to be repeated.
    shared: int = 0
    redone: int = 0


def reaches_bench(
    reg: Regulation,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    hidden: dict[int, tuple[int, ...]],
) -> np.ndarray:
    """(rows, cols) mask: cells where a hidden Pokemon could be put on the field.

    Conservative in the direction that costs time rather than correctness -- a cell wrongly
    marked here is resolved per completion, which is what every cell used to do.
    """
    rows, cols = len(ours), len(theirs)
    mask = np.zeros((rows, cols), dtype=bool)
    ours_hidden = set(hidden.get(0, ()))
    theirs_hidden = set(hidden.get(1, ()))
    row_phazes = [_phazes(reg, action) for action in ours]
    col_phazes = [_phazes(reg, action) for action in theirs]
    # `switch_indices` is 1-based, matching the choice string Showdown reads; `Pokemon.slot`
    # and therefore every hidden slot is 0-based. Comparing them directly marked the wrong
    # column dirty and shared the right one, which the equality test caught as nine cells
    # in one row -- the only reason it was not a silently wrong payoff.
    row_switches = [
        any(index - 1 in ours_hidden for index in action.switch_indices) for action in ours
    ]
    col_switches = [
        any(index - 1 in theirs_hidden for index in action.switch_indices)
        for action in theirs
    ]
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = (
                row_switches[i]
                or col_switches[j]
                # Either side's phazing move can drag the *other* side's bench in, and both
                # sides' benches carry hidden slots here, so one move is enough to spoil
                # the cell whichever side played it.
                or (row_phazes[i] and theirs_hidden)
                or (col_phazes[j] and ours_hidden)
            )
    return mask


def _phazes(reg: Regulation, action: SideAction) -> bool:
    choice = action.to_choice()
    return any(part in PHAZING_MOVES for part in choice.replace(",", " ").split())


def belief_payoffs(
    reg: Regulation,
    position: Position,
    ours: Sequence[SideAction],
    theirs: Sequence[SideAction],
    evaluate: Callable[[list[Position]], np.ndarray],
    *,
    budget: Budget,
    spreads: dict[int, list],
) -> BeliefNode:
    """Every completion's matrix, resolving the turns that do not depend on the bench once.

    `spreads[s]` completes side `s`'s own unseen slots, so side `1 - s`'s game is solved
    over it. A side with nothing hidden contributes a single exact completion and costs
    nothing extra -- which was true of one such side and not of two until IKA-104: with
    both exact the two completions are the same position, and `_per_completion` resolved
    it once per side. It resolves it once now.

    Falls back to a matrix per completion when the port is not available, because the fast
    path is defined as "equal to that" and there is nothing to be equal to without it.
    """
    from . import rustnode

    row, col = list(ours), list(theirs)
    hidden = {
        side: (items[0].slots if items and not items[0].exact else ())
        for side, items in spreads.items()
    }
    node = rustnode.node_for(reg) if rustnode.available() else None
    if node is None or not any(hidden.values()):
        return _per_completion(reg, row, col, evaluate, budget, spreads)

    from .encode import Encoded

    scorer = getattr(getattr(evaluate, "__self__", evaluate), "from_encoded", None)
    if scorer is None:
        return _per_completion(reg, row, col, evaluate, budget, spreads)

    try:
        filled = node.fill_encoded(position, row, col, budget, [], None)
    except Exception:  # noqa: BLE001 - a broken bridge must not fail the run
        return _per_completion(reg, row, col, evaluate, budget, spreads)

    dirty = reaches_bench(reg, row, col, hidden)
    for i, j, _root in filled.folded:
        dirty[i, j] = True
    reference: Encoded = filled.encoded
    unmodelled = set(filled.unmodelled)

    matrices: dict[int, list[np.ndarray]] = {}
    shared = redone = 0
    for side, items in spreads.items():
        other = 1 - side
        slots = hidden.get(side, ())
        per_completion: list[np.ndarray] = []
        for item in items:
            if item.exact or not slots:
                values = np.asarray(scorer(reference), dtype=np.float64)
            else:
                values = np.asarray(
                    scorer(_patched(reference, side, slots, item, reg, position)),
                    dtype=np.float64,
                )
            payoff = np.zeros((len(row), len(col)), dtype=np.float64)
            for i, j, indices, weights in filled.spans:
                if not weights or dirty[i, j]:
                    continue
                payoff[i, j] = float(values[list(indices)] @ np.asarray(weights))
            for i, j, root in filled.folded:
                if not dirty[i, j]:
                    payoff[i, j] = fold_value(root, values)
            shared += int((~dirty).sum())
            wanted = [
                (i, j)
                for i in range(len(row))
                for j in range(len(col))
                if dirty[i, j]
            ]
            if wanted:
                part, notes, _exact = batched_payoffs(
                    reg, item.position, row, col, [evaluate], budget=budget, cells=wanted
                )
                for i, j in wanted:
                    payoff[i, j] = part[0][i, j]
                unmodelled |= notes
                redone += len(wanted)
            per_completion.append(payoff)
        matrices[other] = per_completion
    return BeliefNode(matrices, unmodelled, shared, redone)


def _patched(
    reference,  # noqa: ANN001 - Encoded, imported lazily
    side: int,
    slots: tuple[int, ...],
    item,  # noqa: ANN001 - Completion
    reg: Regulation,
    position: Position,
):  # noqa: ANN202
    """The reference leaves with `side`'s hidden slots replaced by this completion's.

    The rows come from encoding the completion's own root position once. A Pokemon that
    has never been active looks the same in every leaf it was not part of -- full HP, no
    status, no boosts, nothing switched in -- so the root's row is that leaf's row, and
    the test is what says so rather than this comment.
    """
    from .encode import Encoded

    encoder = _encoder_for(reg)
    source = encoder.encode_positions([item.position])
    out = Encoded(
        species=reference.species.copy(),
        ability=reference.ability.copy(),
        item=reference.item.copy(),
        moves=reference.moves.copy(),
        mon=reference.mon.copy(),
        mask=reference.mask.copy(),
        side=reference.side,
        field=reference.field,
        unknown_volatiles=dict(reference.unknown_volatiles),
    )
    for slot in slots:
        out.species[:, side, slot] = source.species[0, side, slot]
        out.ability[:, side, slot] = source.ability[0, side, slot]
        out.item[:, side, slot] = source.item[0, side, slot]
        out.moves[:, side, slot] = source.moves[0, side, slot]
        out.mon[:, side, slot] = source.mon[0, side, slot]
        out.mask[:, side, slot] = source.mask[0, side, slot]
    return out


#: One encoder per regulation. Building one reads the whole vocabulary, and this is called
#: once per completion per node.
_ENCODERS: dict[int, object] = {}


def _encoder_for(reg: Regulation):  # noqa: ANN202
    from .encode import Encoder

    key = id(reg)
    made = _ENCODERS.get(key)
    if made is None:
        made = Encoder(reg)
        _ENCODERS[key] = made
    return made


def _per_completion(
    reg: Regulation,
    ours: list[SideAction],
    theirs: list[SideAction],
    evaluate: Callable[[list[Position]], np.ndarray],
    budget: Budget,
    spreads: dict[int, list],
) -> BeliefNode:
    """A matrix per completion, resolved from scratch. The definition of the answer.

    Per position rather than per entry: a position both sides' lists hold is resolved once,
    and the second side gets a copy of that matrix -- a copy so that the two lists still
    hold two arrays, as they did when each was resolved on its own. Nothing hidden on
    either side is exactly that case, since `completions` then hands each side the
    position itself, and until IKA-104 it was resolved once per side: the port's fill, the
    forward pass and the fold twice over for the same numbers, on 37% of the move
    decisions of IKA-73's width-12 pool (`scratchpad/both_exact.py`).

    Nothing is turned round for side 1. Every matrix here is side 0's payoff, whichever
    list it comes from, and `belief_solve` is what makes side 1's its own game.

    Identity and not equality, because the same object is the only thing known to be the
    same node. A completion with anything in it is a fresh `substitute`, so two of those
    are never one object and nothing else here is shared.
    """
    matrices: dict[int, list[np.ndarray]] = {}
    unmodelled: set[str] = set()
    resolved: dict[int, np.ndarray] = {}
    shared = redone = 0
    for side, items in spreads.items():
        built: list[np.ndarray] = []
        for item in items:
            payoff = resolved.get(id(item.position))
            if payoff is None:
                payoff, notes = batched_payoff(
                    reg, item.position, ours, theirs, evaluate, budget=budget
                )
                resolved[id(item.position)] = payoff
                unmodelled |= notes
                redone += len(ours) * len(theirs)
            else:
                payoff = payoff.copy()
                shared += len(ours) * len(theirs)
            built.append(payoff)
        matrices[1 - side] = built
    return BeliefNode(matrices, unmodelled, shared, redone)


__all__ = ["PHAZING_MOVES", "BeliefNode", "belief_payoffs", "reaches_bench"]
