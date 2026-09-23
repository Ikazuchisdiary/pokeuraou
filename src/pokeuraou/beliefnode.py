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
- the turn suspends for a replacement, whose choices include the hidden slots;
- the port refused the cell (it has no leaves to share), so Python resolves it.

Both sides' games share the one resolution, because their hidden slots are disjoint and a
shared cell's resolution depends on neither. That is what takes 6.0x to a projected 2.2x.

And the whole node is scored in one call of the leaf (IKA-105): every completion's patched
rows, the port's leaves of its dirty cells and the leaves of the cells the port refused
are blocks of one `Encoded`, and each completion reads its values back by where its block
starts. It was a forward pass per completion and another per completion's dirty cells --
15.5 a node over `data/ika73/w12` games 0-209, against 17.74 for a whole `move.hidden`
decision of shipping generation, every pass it makes counted (IKA-98).
The answers are the same to the bit with a leaf whose rows do not depend on the batch;
the learned net's do, in the float32 sums, which moves a value by 1e-9 (IKA-148).

## What has to be true, and is asserted rather than assumed

That a patched encoding equals the encoding of the same completion resolved from scratch.
If it does not -- a feature that reads the bench, a mechanic that counts party members --
the payoffs stay plausible and are wrong, and nothing downstream could tell. `tests/
test_beliefnode.py` compares the two, cell by cell, and the fast path is only taken when
the port is available to be compared against.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from . import timing
from .actions import MoveAction, SideAction
from .position import Position
from .regulation import Regulation
from .resolve import (
    Budget,
    _fold_from_json,
    batched_payoff,
    batched_payoffs,
    fold_value,
)

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
    position: Position | None = None,
) -> np.ndarray:
    """(rows, cols) mask: cells where a hidden Pokemon could be put on the field.

    Conservative in the direction that costs time rather than correctness -- a cell wrongly
    marked here is resolved per completion, which is what every cell used to do.

    With `position`, a held item's path too (IKA-191): a damaging move into a Red Card drags
    the *user's* side in, so a side with a holder on the field spoils every cell where the
    other side attacks, for the attacker's hidden slots. Eject Button and Emergency Exit
    put the holder's own bench in through a mid-turn replacement, and a cell that suspends
    is already dirty (`filled.folded`), so they need no row here.
    """
    rows, cols = len(ours), len(theirs)
    mask = np.zeros((rows, cols), dtype=bool)
    ours_hidden = set(hidden.get(0, ()))
    theirs_hidden = set(hidden.get(1, ()))
    row_phazes = [_phazes(reg, action) for action in ours]
    col_phazes = [_phazes(reg, action) for action in theirs]
    carded = [_red_card_on_field(position, side) for side in (0, 1)]
    row_carded = [carded[1] and _attacks(reg, action) for action in ours]
    col_carded = [carded[0] and _attacks(reg, action) for action in theirs]
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
                or (row_carded[i] and ours_hidden)
                or (col_carded[j] and theirs_hidden)
            )
    return mask


def _phazes(reg: Regulation, action: SideAction) -> bool:
    # By the move's id: the choice string names a move by its index ("move 2 1"), so
    # reading PHAZING_MOVES off it never matched anything (IKA-191).
    return any(isinstance(s, MoveAction) and s.move_id in PHAZING_MOVES for s in action.slots)


def _attacks(reg: Regulation, action: SideAction) -> bool:
    """Whether the action uses a damaging move -- what a Red Card answers."""
    return any(
        isinstance(s, MoveAction)
        and (move := reg.moves.get(s.move_id)) is not None
        and move.category != "Status"
        for s in action.slots
    )


def _red_card_on_field(position: Position | None, side: int) -> bool:
    if position is None:
        return False
    return any(
        mon is not None and not mon.fainted and mon.item == "redcard"
        for mon in position.sides[side].active_pokemon()
    )


# Its own Python -- the copies per completion, the span loop, the folds -- had no stage and
# sat in "the rest" (IKA-98); what it calls is still charged to the stages it calls.
@timing.timed("belief")
@timing.labelled("matrix")
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
    if timing.ON:
        # Per decision, by whether any bench here is hidden (IKA-98's per-decision table).
        timing.refine("hidden" if any(hidden.values()) else "exact")
        timing.count("completions", sum(len(items) for items in spreads.values()))
    node = rustnode.node_for(reg) if rustnode.available() else None
    if node is None or not any(hidden.values()):
        return _per_completion(reg, row, col, evaluate, budget, spreads)

    from .encode import Encoded, rules_of

    owner = getattr(evaluate, "__self__", evaluate)
    scorer = getattr(owner, "from_encoded", None)
    if scorer is None:
        return _per_completion(reg, row, col, evaluate, budget, spreads)
    # The leaf's own encoder and rules, for the port's arrays and for the patched rows alike:
    # a match plays two arms in one process, and each arm is scored the way its leaf says
    # (IKA-141). A leaf without an encoder is scored under the current rules.
    encoder = getattr(owner, "encoder", None) or _encoder_for(reg)
    rules = rules_of(evaluate)

    try:
        filled = node.fill_encoded(position, row, col, budget, [], None, rules=rules)
    except Exception:  # noqa: BLE001 - a broken bridge must not fail the run
        return _per_completion(reg, row, col, evaluate, budget, spreads)
    if rules.mega_from_slots and filled.mega_from_slots is not True:
        # A binary that predates the field encoded with the current rule. The fill cannot
        # be used for this leaf; the per-completion path encodes in Python.
        return _per_completion(reg, row, col, evaluate, budget, spreads)
    from .resolve import _note_port_rule

    _note_port_rule([scorer], filled)

    dirty = reaches_bench(reg, row, col, hidden, position)
    for i, j, _root in filled.folded:
        dirty[i, j] = True
    # A refused cell has no span and no fold, so unless it is dirty nothing ever writes it
    # and it keeps the 0.0 the matrix was made with -- in every completion (IKA-139: 3,886
    # cells of `data/ika73/w12` games 10-209, all Feint). Resolving it per completion is
    # `_gather_dirty`, which fills what the port refuses in Python, as `_per_completion`
    # does.
    for i, j, _why in filled.refused:
        dirty[i, j] = True
    reference: Encoded = filled.encoded
    unmodelled = set(filled.unmodelled)
    wanted = [(i, j) for i in range(len(row)) for j in range(len(col)) if dirty[i, j]]
    # What a leaf resolved here in Python is encoded with: the scorer's own encoder, which
    # is what `evaluate(positions)` would have encoded it with. Without one the Python
    # leaves are scored on their own, as they were before.
    leaf_encoder = getattr(owner, "encoder", None) or getattr(
        getattr(scorer, "__self__", None), "encoder", None
    )

    # Everything the node's leaves need, gathered first and scored once (IKA-105). Until
    # then a hidden decision paid a forward pass per completion for the shared cells, and
    # per completion again for the dirty cells -- the port's leaves in one and the cells
    # it refused, resolved here, in another: 17.74 passes a `move.hidden` decision against
    # 3.02 for `move.exact` (IKA-98). Each part is a block of rows in one `Encoded`; the
    # values go back to the completion that asked for them by the block's start.
    parts: list[tuple[int, Callable[[], Encoded]]] = []

    def block(rows: int, make: Callable[[], Encoded]) -> int:
        parts.append((rows, make))
        return len(parts) - 1

    reference_block: int | None = None
    jobs: list[_Job] = []
    for side, items in spreads.items():
        slots = hidden.get(side, ())
        for item in items:
            job = _Job(side=side, item=item)
            if item.exact or not slots:
                if reference_block is None:
                    reference_block = block(len(reference), lambda: reference)
                job.shared = reference_block
            else:
                job.shared = block(
                    len(reference),
                    lambda item=item, side=side, slots=slots: _patched(
                        reference, side, slots, item, reg, position, encoder, rules
                    ),
                )
            if wanted:
                _gather_dirty(
                    job, reg, position, row, col, budget, wanted, filled, node, rules,
                    scorer, evaluate, leaf_encoder, block,
                )
                if job.dirty_from_reference and reference_block is None:
                    reference_block = block(len(reference), lambda: reference)
            jobs.append(job)

    stacked, starts = _stacked(parts, reference)
    values = (
        np.asarray(scorer(stacked), dtype=np.float64)
        if len(stacked)
        else np.zeros(0, dtype=np.float64)
    )
    del stacked
    # The parts' arrays are in the batch now, and only their sizes, spans and folds are
    # read from here on; a dirty fill is as large as the node, so it is let go.
    parts = [(rows, None) for rows, _make in parts]
    for job in jobs:
        if job.filled is not None:
            job.filled.encoded = None

    def rows_of(index: int) -> np.ndarray:
        start = starts[index]
        return values[start : start + parts[index][0]]

    matrices: dict[int, list[np.ndarray]] = {}
    shared = redone = 0
    for job in jobs:
        shared_values = rows_of(job.shared)
        payoff = np.zeros((len(row), len(col)), dtype=np.float64)
        for i, j, indices, weights in filled.spans:
            if not weights or dirty[i, j]:
                continue
            payoff[i, j] = float(shared_values[list(indices)] @ np.asarray(weights))
        # Every folded cell is dirty (above), so no fold is read off the shared rows.
        shared += int((~dirty).sum())
        if wanted:
            part = np.zeros((len(row), len(col)), dtype=np.float64)
            if job.dirty_from_reference:
                # The exact completion is the true position itself, so its dirty cells are
                # the reference fill's own leaves: what `_per_completion` scores for it.
                ported = rows_of(reference_block)
                for i, j, indices, weights in filled.spans:
                    if weights and dirty[i, j]:
                        part[i, j] = float(ported[indices] @ np.asarray(weights))
                for i, j, root in filled.folded:
                    part[i, j] = fold_value(_fold_from_json(root), ported)
            elif job.dirty is not None:
                ported = rows_of(job.dirty)
                # As `resolve._rust_encoded_payoffs` writes a node the port filled.
                for i, j, indices, weights in job.filled.spans:
                    if not weights:
                        continue
                    part[i, j] = float(ported[indices] @ np.asarray(weights))
                for i, j, root in job.filled.folded:
                    part[i, j] = fold_value(_fold_from_json(root), ported)
            if job.held is not None:
                held_values = (
                    rows_of(job.python)
                    if job.python is not None
                    else np.asarray(evaluate(job.held.leaves), dtype=np.float64)
                    if job.held.leaves
                    else np.zeros(0)
                )
                job.held.write(held_values, part)
            if job.fallback is not None:
                part = job.fallback
            for i, j in wanted:
                payoff[i, j] = part[i, j]
            unmodelled |= job.notes
            redone += len(wanted)
        matrices.setdefault(1 - job.side, []).append(payoff)
    return BeliefNode(matrices, unmodelled, shared, redone)


@dataclass
class _Job:
    """One completion's share of the node: which blocks of the one batch are its own."""

    side: int
    item: object
    #: The block its shared cells read: the reference, or its own patched copy of it.
    shared: int = -1
    #: Its dirty cells: the port's fill of them on this completion, and its block.
    filled: object = None
    dirty: int | None = None
    #: The exact completion is the true position, and reads the reference fill instead.
    dirty_from_reference: bool = False
    #: Cells the port refused, resolved here, and the block their leaves were encoded to
    #: (None: scored through `evaluate` on their own, for a leaf with no encoder).
    held: object = None
    python: int | None = None
    #: The dirty cells from `batched_payoffs`, when the port could not fill them here.
    fallback: np.ndarray | None = None
    notes: set[str] = field(default_factory=set)


def _gather_dirty(  # noqa: PLR0913 - one completion's dirty cells, and where they go
    job: _Job,
    reg: Regulation,
    position: Position,
    row: list[SideAction],
    col: list[SideAction],
    budget: Budget,
    wanted: list[tuple[int, int]],
    filled,  # noqa: ANN001 - EncodedNode
    node,  # noqa: ANN001 - rustnode.RustNode
    rules,  # noqa: ANN001 - EncodingRules
    scorer: Callable,
    evaluate: Callable,
    leaf_encoder,  # noqa: ANN001 - Encoder or None
    block: Callable[[int, Callable], int],
) -> None:
    """Resolve one completion's dirty cells without scoring them, and book their blocks.

    What `batched_payoffs(item.position, cells=wanted)` did, less its forward passes: the
    port fills the cells (`resolve._rust_encoded_payoffs`), and the ones it refuses are
    resolved here (`resolve.HeldLeaves`, which `batched_payoffs` itself uses). Both sets
    of leaves become blocks of the node's one batch.
    """
    from . import rustnode
    from .resolve import HeldLeaves, _note_port_rule

    item = job.item
    with timing.purpose("dirty"):
        if item.exact and item.position is position:
            job.dirty_from_reference = True
            refused = [(i, j) for i, j, _why in filled.refused]
            why = [why for _i, _j, why in filled.refused]
        else:
            try:
                own = node.fill_encoded(
                    item.position, row, col, budget, [], wanted, rules=rules
                )
            except Exception as exc:  # noqa: BLE001 - a broken bridge must not fail the run
                rustnode.disable(str(exc))
                part, notes, _exact = batched_payoffs(
                    reg, item.position, row, col, [evaluate], budget=budget, cells=wanted
                )
                job.fallback, job.notes = part[0], set(notes)
                return
            _note_port_rule([scorer], own)
            timing.count("leaves.node", len(own.encoded))
            job.filled = own
            job.dirty = block(len(own.encoded), lambda own=own: own.encoded)
            job.notes |= set(own.unmodelled)
            refused = [(i, j) for i, j, _why in own.refused]
            why = [why for _i, _j, why in own.refused]
        if not refused:
            return
        started = time.perf_counter()
        held = HeldLeaves()
        exact = np.zeros((len(row), len(col)), dtype=bool)
        for i, j in refused:
            held.resolve_cell(reg, item.position, row, col, i, j, budget, exact, job.notes)
        job.held = held
        timing.count("leaves.refused", len(held.leaves))
        if leaf_encoder is not None and held.leaves:
            job.python = block(
                len(held.leaves), lambda: leaf_encoder.encode_positions(held.leaves)
            )
        timing.add("refused", time.perf_counter() - started, calls=1)
        if timing.ON:
            for reason in why:
                timing.count(f"refused: {reason}")


#: The arrays one batch is made of, in `Encoded`'s order.
_ARRAYS = ("species", "ability", "item", "moves", "mon", "mask", "side", "field")


def _stacked(
    parts: Sequence[tuple[int, Callable]], like  # noqa: ANN001 - Encoded
) -> tuple[object, list[int]]:
    """The parts one after another in one `Encoded`, and where each one starts.

    Written into arrays allocated once, one part made at a time -- a patched copy is the
    size of the whole reference, so holding every completion's before copying them would
    double the node's memory. Each array keeps the reference's dtype: the port's index
    arrays are int32 and the encoder's int64, and an index lookup is the same either way.
    """
    from .encode import Encoded

    total = sum(rows for rows, _make in parts)
    out = {
        name: np.empty((total, *getattr(like, name).shape[1:]), dtype=getattr(like, name).dtype)
        for name in _ARRAYS
    }
    starts: list[int] = []
    at = 0
    for rows, make in parts:
        starts.append(at)
        if rows:
            made = make()
            assert len(made) == rows, (len(made), rows)
            for name in _ARRAYS:
                out[name][at : at + rows] = getattr(made, name)
        at += rows
    return Encoded(**out, unknown_volatiles=dict(like.unknown_volatiles)), starts


def _patched(
    reference,  # noqa: ANN001 - Encoded, imported lazily
    side: int,
    slots: tuple[int, ...],
    item,  # noqa: ANN001 - Completion
    reg: Regulation,
    position: Position,
    encoder=None,  # noqa: ANN001 - Encoder, imported lazily
    rules=None,  # noqa: ANN001 - EncodingRules
):  # noqa: ANN202
    """The reference leaves with `side`'s hidden slots replaced by this completion's.

    The rows come from encoding the completion's own root position once. A Pokemon that
    has never been active looks the same in every leaf it was not part of -- full HP, no
    status, no boosts, nothing switched in -- so the root's row is that leaf's row, and
    the test is what says so rather than this comment.

    Two things are not a row copy (IKA-119). A bench row's `can_mega` is the root's, but the
    side may have mega evolved during the turn, so it is ANDed with the leaf's `mega_used`.
    And the side vector reads the bench -- `mega_available` (a stone anywhere on the side),
    `alive_fraction`, `team_hp_fraction` (the bench's max HP is in the denominator) -- so it
    is rebuilt per leaf from the patched rows. Until then it was the true position's, and
    every completion was scored with the true bench's side features.

    `rules.patch_shares_side` puts that back -- the true position's side vector, and the
    root's `can_mega` copied as it is -- for a match that measures the fix (IKA-141). It is
    the pre-IKA-119 body, line for line, and nothing else should ask for it.
    """
    from .encode import CURRENT_RULES, HP_SCALE, Encoded

    encoder = encoder or _encoder_for(reg)
    rules = rules or CURRENT_RULES
    source = encoder.encode_positions([item.position])
    if hasattr(encoder, "note"):
        encoder.note(
            "patched side=" + ("shared" if rules.patch_shares_side else "rebuilt"),
            len(reference.species),
        )
    if rules.patch_shares_side:
        old = Encoded(
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
            old.species[:, side, slot] = source.species[0, side, slot]
            old.ability[:, side, slot] = source.ability[0, side, slot]
            old.item[:, side, slot] = source.item[0, side, slot]
            old.moves[:, side, slot] = source.moves[0, side, slot]
            old.mon[:, side, slot] = source.mon[0, side, slot]
            old.mask[:, side, slot] = source.mask[0, side, slot]
        return old
    out = Encoded(
        species=reference.species.copy(),
        ability=reference.ability.copy(),
        item=reference.item.copy(),
        moves=reference.moves.copy(),
        mon=reference.mon.copy(),
        mask=reference.mask.copy(),
        side=reference.side.copy(),
        # Nothing in the field vector reads a Pokemon.
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

    mon_k = {name: k for k, name in enumerate(encoder.mon_names)}
    side_k = {name: k for k, name in enumerate(encoder.side_names)}
    spent = out.side[:, side, side_k["mega_used"]] > 0  # (B,)
    rows = out.mon[:, side]  # (B, M, F), a view
    can = mon_k["can_mega"]
    for slot in slots:
        rows[spent, slot, can] = 0.0
    # With the mega unspent no Pokemon of the side is a mega, so "holds its stone" is
    # exactly `can_mega` on every row, patched or not.
    present = out.mask[:, side] > 0  # (B, M)
    holder = ((rows[:, :, can] > 0) & present).any(axis=1)
    out.side[:, side, side_k["mega_available"]] = (holder & ~spent).astype(np.float32)
    # Integers again, so the division is the encoder's own and the float32 is identical.
    maxhp = np.rint(rows[:, :, mon_k["maxhp_scaled"]].astype(np.float64) * HP_SCALE)
    hp = np.rint(rows[:, :, mon_k["hp_fraction"]].astype(np.float64) * maxhp)
    maxhp = np.where(present, maxhp, 0.0)
    hp = np.where(present, hp, 0.0)
    count = present.sum(axis=1)
    alive = (present & ~(rows[:, :, mon_k["fainted"]] > 0)).sum(axis=1)
    out.side[:, side, side_k["alive_fraction"]] = alive / np.maximum(count, 1)
    total = maxhp.sum(axis=1)
    out.side[:, side, side_k["team_hp_fraction"]] = hp.sum(axis=1) / np.where(
        total > 0, total, 1.0
    )
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
