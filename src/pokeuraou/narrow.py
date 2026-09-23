"""Choosing which action combinations get solved exactly.

A doubles turn has a few hundred legal choices per side -- the product of two slots'
options -- and the matrix is the product of both sides', so solving every cell exactly is
not affordable. Narrowing is therefore unavoidable, and the only question is whether what
it throws away is *sayable*.

Two rules keep it sayable.

**Nothing is eliminated, only combinations are.** Ranking by damage and keeping the top N
would silently delete every switch and every status move from the answer -- a Protect or
a Fake Out scores zero damage and would never survive a damage ranking, which is exactly
backwards for the positions where those are the answer. So coverage comes first: the kept
set is chosen to contain every individual slot option that appears anywhere in the
candidates, and only then is the remaining budget spent on the highest-scoring
combinations. If the budget is too small to cover everything, the options left out are
returned in :attr:`Narrowed.uncovered` and printed, never dropped quietly.

**The score orders candidates; it never becomes an output.** It is an average damage
fraction, computed straight from the calculator, and it decides only which cells the
resolver visits. Every number the tool prints -- the equilibrium value, the frequencies,
the EV losses -- comes from exactly resolved turns, so a mediocre ordering costs accuracy
in the *choice of candidates* and cannot bias a printed figure.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from . import timing
from .actions import (
    MoveAction,
    PassAction,
    SideAction,
    SwitchAction,
    TargetNames,
    side_actions,
)
from .battler import Battler
from .damage import calculate, effective_damage
from .position import Position
from .regulation import Regulation
from .resolve import FIRST_TURN_OUT_MOVES

if TYPE_CHECKING:
    from .names import Localiser
from .view import battler, field_state, move_context

#: How many action combinations per side reach the exact solver by default. The matrix is
#: the product of the two sides', so this is 576 resolved turns.
DEFAULT_LIMIT = 24

#: Move targets that hit every adjacent foe, so a single choice lands on two Pokemon.
SPREAD_FOE_TARGETS = frozenset({"allAdjacentFoes", "foeSide"})
#: ...and the ones that hit the user's partner too.
SPREAD_ALL_TARGETS = frozenset({"allAdjacent"})


@dataclass(frozen=True, slots=True)
class Candidate:
    """One side's complete choice, with the score that ranked it."""

    action: SideAction
    #: Sum over slots of expected damage as a fraction of the target's current HP, with
    #: damage to one's own partner subtracted. Ordering only; never printed as a result.
    score: float
    #: Per-slot contributions, so a surprising ranking can be read rather than trusted.
    detail: tuple[str, ...] = ()


@dataclass(slots=True)
class Narrowed:
    """The kept candidates and an account of what narrowing did."""

    kept: list[Candidate]
    considered: int
    #: Slot options the budget could not fit. Empty means the guarantee held.
    uncovered: tuple[str, ...] = ()
    #: The same options as (slot index, slot action), so :meth:`render` can name them in
    #: whatever language the caller asked for. :attr:`uncovered` stays the English label
    #: because it is what the JSON output carries, and a machine consumer should not start
    #: receiving Japanese because a display flag changed.
    uncovered_options: tuple[tuple[int, object], ...] = ()
    #: How many kept candidates were taken for coverage versus for their score.
    for_coverage: int = 0
    for_score: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    @property
    def actions(self) -> list[SideAction]:
        return [c.action for c in self.kept]

    @property
    def complete(self) -> bool:
        """Whether every legal combination survived, making narrowing a no-op."""
        return len(self.kept) == self.considered

    def render(
        self,
        reg: Regulation,
        loc: Localiser | None = None,
        targets: TargetNames | None = None,
    ) -> str:
        head = (
            f"候補 {self.considered} → {len(self.kept)} "
            f"（網羅 {self.for_coverage} / スコア上位 {self.for_score}）"
        )
        kinds = ", ".join(f"{k} {v}" for k, v in sorted(self.by_kind.items()))
        lines = [head, f"  内訳: {kinds}"]
        if self.uncovered:
            if loc is not None and self.uncovered_options:
                dropped = ", ".join(
                    f"slot{index + 1} {action.describe(reg, loc, targets)}"  # type: ignore[union-attr]
                    for index, action in self.uncovered_options
                )
            else:
                dropped = ", ".join(self.uncovered)
            lines.append(f"  予算不足で落ちた選択肢: {dropped}")
        for c in self.kept[:8]:
            lines.append(f"  {c.score:+.3f}  {c.action.describe(reg, loc, targets)}")
        return "\n".join(lines)


def action_kind(reg: Regulation, action: SideAction) -> str:
    """A coarse label for reporting, not for ranking."""
    kinds = set()
    for slot in action.slots:
        if isinstance(slot, SwitchAction):
            kinds.add("switch")
        elif isinstance(slot, PassAction):
            continue
        elif isinstance(slot, MoveAction):
            move = reg.moves.get(slot.move_id)
            if move is None:
                kinds.add("status")
            elif move.raw.get("stallingMove"):
                kinds.add("protect")
            elif move.category == "Status":
                kinds.add("status")
            else:
                kinds.add("attack")
    if action.declares_mega:
        kinds.add("mega")
    return "+".join(sorted(kinds)) or "pass"


def _slot_key(action: SideAction, index: int) -> str:
    """Identity of one slot's option, for the coverage guarantee."""
    slot = action.slots[index]
    if isinstance(slot, MoveAction):
        return f"{index}:move:{slot.move_id}:{slot.target}:{int(slot.mega)}"
    if isinstance(slot, SwitchAction):
        return f"{index}:switch:{slot.party_index}"
    return f"{index}:pass"


def _slot_label(reg: Regulation, action: SideAction, index: int) -> str:
    return f"slot{index + 1} {action.slots[index].describe(reg)}"


def _hit_slots(
    reg: Regulation, move_id: str, target: int | None, slot: int, live_foes: list[int]
) -> list[tuple[int, bool]]:
    """(slot index, is_foe) for every Pokemon a move choice lands on."""
    move = reg.moves.get(move_id)
    if move is None or move.category == "Status":
        return []
    if move.target in SPREAD_FOE_TARGETS:
        return [(i, True) for i in live_foes]
    if move.target in SPREAD_ALL_TARGETS:
        return [(i, True) for i in live_foes] + [(1 - slot, False)]
    if target is None:
        return []
    if target > 0:
        return [(target - 1, True)]
    return [(-target - 1, False)]


def _pin(mon: Battler, weights: np.ndarray | None) -> Battler:
    """The single most likely particle of a belief-carrying Battler.

    Needed when a spread move puts two *hidden* Pokemon on opposite sides of one damage
    calculation -- the opponent's Earthquake hitting its own partner. Their particle axes
    are different lengths and mean different things, so there is no correct way to line
    them up; the calculator vectorises over one belief, not a joint one. Pinning both to
    their modal particle keeps the score defined, and the score only orders candidates.
    """
    if mon.n <= 1:
        return mon
    index = int(np.argmax(weights)) if weights is not None and weights.size == mon.n else 0
    return replace(
        mon,
        stats=mon.stats[index : index + 1],
        hp=mon.hp[index : index + 1],
        maxhp=mon.maxhp[index : index + 1],
    )


def _expected_fraction(
    reg: Regulation,
    attacker: Battler,
    defender: Battler,
    move_id: str,
    field: object,
    *,
    spread: bool,
    defender_side: int,
    weights: np.ndarray | None,
    attacker_weights: np.ndarray | None = None,
    move_ctx: object = None,
) -> tuple[float, bool]:
    """Mean damage as a fraction of the defender's current HP, and whether it is usable.

    The mean is over all 16 rolls and over the belief's particles, weighted. A move the
    calculator cannot evaluate (a base power it does not model, an immunity) returns 0
    and says so, so it is ranked by coverage rather than by a made-up number.
    """
    if attacker.n > 1 and defender.n > 1 and attacker.n != defender.n:
        attacker = _pin(attacker, attacker_weights)
        defender = _pin(defender, weights)
        weights = None
    result = calculate(
        reg, attacker, defender, move_id, field,  # type: ignore[arg-type]
        defender_side=defender_side, spread=spread, move_ctx=move_ctx,  # type: ignore[arg-type]
    )
    if result.immune:
        return 0.0, True
    dealt = effective_damage(result, defender)
    if dealt.size == 0:
        return 0.0, False
    per_particle = dealt.mean(axis=1) / np.maximum(defender.hp, 1)
    if weights is not None and weights.shape[0] == per_particle.shape[0]:
        value = float(np.dot(per_particle, weights) / max(weights.sum(), 1e-12))
    else:
        value = float(per_particle.mean())
    return min(value, 1.0), not result.unmodelled


def _bridged_scores(
    reg: Regulation,
    pos: Position,
    side: int,
    pool: list[SideAction],
) -> list[Candidate] | None:
    """The whole pool scored by the Rust port in one crossing, or None to do it here.

    `score_action` is `damage.calculate` in a loop, and that loop was 13.9% of a bridged
    generation run -- the largest thing left in Python once the resolver crossed over. The
    legality of the pool is 0.2% and stays here; only the arithmetic goes.

    The port carries one particle, so a caller with beliefs is not offered this. It hands
    back which target each move hit and for how much rather than the readable line, so the
    line is still written here and still says the same thing.
    """
    from . import rustnode

    if not rustnode.available():
        return None
    node = rustnode.node_for(reg)
    if node is None:
        return None
    try:
        scored = node.score(pos, side, pool)
    except Exception as exc:  # noqa: BLE001 - a broken bridge must not fail the run
        rustnode.disable(str(exc))
        return None
    if scored is None:
        return None

    out: list[Candidate] = []
    for action, (total, parts) in zip(pool, scored, strict=True):
        detail = tuple(
            f"{action.slots[slot].describe(reg)} -> "
            f"{'foe' if is_foe else 'ally'}{target_slot + 1} {signed:+.3f}{'' if exact else '?'}"
            for slot, target_slot, is_foe, signed, exact in parts
        )
        out.append(Candidate(action=action, score=total, detail=detail))
    return out


def score_action(
    reg: Regulation,
    pos: Position,
    side: int,
    action: SideAction,
    *,
    battlers: Mapping[tuple[int, int], Battler],
    weights: Mapping[tuple[int, int], np.ndarray] | None = None,
) -> Candidate:
    """Damage this choice is expected to do, minus what it does to one's own partner."""
    field = field_state(pos, reg)
    foe = 1 - side
    live_foes = [
        i
        for i in range(len(pos.sides[foe].active))
        if (mon := pos.sides[foe].pokemon[pos.sides[foe].active[i]]) is not None
        and not mon.fainted
    ]

    total = 0.0
    detail: list[str] = []
    for index, slot in enumerate(action.slots):
        if not isinstance(slot, MoveAction):
            continue
        attacker = battlers.get((side, index))
        if attacker is None:
            continue
        hits = _hit_slots(reg, slot.move_id, slot.target, index, live_foes)
        spread = len(hits) > 1
        for target_slot, is_foe in hits:
            target_side = foe if is_foe else side
            defender = battlers.get((target_side, target_slot))
            if defender is None:
                continue
            fraction, exact = _expected_fraction(
                reg, attacker, defender, slot.move_id, field,
                spread=spread, defender_side=target_side,
                # Without this the scorer reads the declared base power and prices Last
                # Respects at 50 for the whole battle -- so a 200-power move can fail to
                # make the candidate list at all.
                move_ctx=move_context(pos, side, index),
                weights=None if weights is None else weights.get((target_side, target_slot)),
                attacker_weights=None if weights is None else weights.get((side, index)),
            )
            signed = fraction if is_foe else -fraction
            total += signed
            flag = "" if exact else "?"
            detail.append(
                f"{slot.describe(reg)} -> "
                f"{'foe' if is_foe else 'ally'}{target_slot + 1} {signed:+.3f}{flag}"
            )
    return Candidate(action=action, score=total, detail=tuple(detail))


def _battlers_from_position(reg: Regulation, pos: Position) -> dict[tuple[int, int], Battler]:
    out: dict[tuple[int, int], Battler] = {}
    for side_index, side in enumerate(pos.sides):
        for slot, mon in enumerate(side.active_pokemon()):
            if mon is None or mon.fainted:
                continue
            out[(side_index, slot)] = battler(reg, mon)
    return out


def drop_dead_actions(
    reg: Regulation, pos: Position, side: int, pool: list[SideAction]
) -> list[SideAction]:
    """Removes actions that cannot do anything, before the candidate budget is spent.

    Fake Out, First Impression and Mat Block fail outright unless their user came in this
    turn. In the champions dex Fake Out and First Impression never get that far: their
    `onDisableMove` takes them off the request once the user has made a move action, so
    `side_actions` does not offer them (IKA-166). What reaches this is a dex without that
    hook -- the base game's Fake Out, and Mat Block everywhere -- where Showdown lets a
    player pick one and fails it at execution, so it is *legal* and `side_actions` is
    right to offer it. It is not worth a candidate slot: a move that provably does
    nothing is dominated by every other move, and the solver was not merely wasting a slot
    on one, it was putting 41.6% of a node's equilibrium weight on it, because a value
    function's noisy cells do not know the move is dead.

    The counter here is the one at decision time. `runMove` bumps it before `onTry` reads
    `activeMoveActions > 1`, so the move is dead from a counter of 1, not 2 -- the test
    was `> 1` until IKA-166 and kept the second turn's Fake Out.

    This is deliberately narrower than the legal set, and only here -- `side_actions`
    still enumerates them, so the differential harness keeps checking that Showdown fails
    them the way we do.

    A combination is dropped when *any* slot's move is dead. In doubles an action is a
    pair, so requiring both to be dead leaves every pairing of a dead Fake Out with a
    live partner in the pool -- which is most of them, and was the first version of this.
    Dropping on `any` is still safe: the same partner action exists alongside every other
    move in the dead slot, so nothing that survives is worse.

    Nothing is dropped when that would empty the pool -- a Pokemon whose only usable move
    is a dead Fake Out must still offer it, because an empty list becomes Struggle and
    that is a different, illegal action.
    """
    if not pool:
        return pool
    dead: list[bool] = []
    for action in pool:
        slots = []
        for slot_action in action.slots:
            move_id = getattr(slot_action, "move_id", None)
            if move_id is None or move_id not in FIRST_TURN_OUT_MOVES:
                slots.append(False)
                continue
            mon = pos.sides[side].active_pokemon()[slot_action.slot]
            slots.append(mon is not None and mon.active_move_actions > 0)
        dead.append(any(slots))
    alive = [action for action, is_dead in zip(pool, dead, strict=True) if not is_dead]
    return alive or pool


@timing.timed("narrow")
def narrow(
    reg: Regulation,
    pos: Position,
    side: int,
    *,
    limit: int = DEFAULT_LIMIT,
    battlers: Mapping[tuple[int, int], Battler] | None = None,
    weights: Mapping[tuple[int, int], np.ndarray] | None = None,
    candidates: list[SideAction] | None = None,
    rank: Callable[[list[SideAction], list[Candidate]], Sequence[float]] | None = None,
) -> Narrowed:
    """Narrows one side's legal choices to at most ``limit``, covering every option.

    ``battlers`` supplies a Battler per active slot, which is how a hidden SP spread gets
    in: the belief layer builds one Battler carrying every particle, and ``weights`` gives
    their probabilities so the score is a belief-weighted mean rather than a guess at the
    opponent's investment.

    ``rank`` replaces the damage score with the caller's own, one value per candidate in
    the order they are handed over. It exists because the damage score and the thing that
    fills the matrix are different functions, and they disagree: measured over 59
    decisions where they disagreed by more than 0.05, playing the action the *leaf*
    preferred instead of the best one on the menu was worth +14.3 points [+7.1, +21.5].
    Coverage, the greedy cover and the tie-break are untouched -- only the ordering
    changes, which is the part that was never claimed to be principled.

    It is handed the damage candidates as well as the pool. They are computed here on every
    call whatever the ordering is, so a ranker that wants them -- a learned one does, as a
    feature -- should not pay for them twice: recomputing them cost a second crossing to
    the port, 0.779 ms a pool, which was 16% of what the learned ordering cost.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    pool = candidates if candidates is not None else side_actions(reg, pos, side)
    pool = drop_dead_actions(reg, pos, side, pool)
    if not pool:
        return Narrowed(kept=[], considered=0)
    scored = None
    if battlers is None and weights is None:
        scored = _bridged_scores(reg, pos, side, pool)
    if scored is None:
        table = dict(battlers) if battlers is not None else _battlers_from_position(reg, pos)
        scored = [
            score_action(reg, pos, side, a, battlers=table, weights=weights) for a in pool
        ]
    if rank is not None:
        # The damage detail is kept beside the new score rather than thrown away: a
        # surprising leaf ranking is exactly when a reader wants to see what the cheap
        # score thought.
        values = rank(pool, scored)
        scored = [
            replace(c, score=float(v), detail=(f"leaf {float(v):+.4f}", *c.detail))
            for c, v in zip(scored, values, strict=True)
        ]
    n_slots = len(pool[0].slots)
    # Highest score first, with a stable tiebreak so the same position gives the same
    # answer twice.
    order = sorted(
        range(len(scored)),
        key=lambda i: (-scored[i].score, pool[i].to_choice()),
    )
    if len(pool) <= limit:
        kept = [scored[i] for i in order]
        return Narrowed(
            kept=kept, considered=len(pool), for_score=len(kept),
            by_kind=_count_kinds(reg, kept),
        )

    needed: dict[str, str] = {}
    options: dict[str, tuple[int, object]] = {}
    for action in pool:
        for index in range(n_slots):
            key = _slot_key(action, index)
            needed.setdefault(key, _slot_label(reg, action, index))
            options.setdefault(key, (index, action.slots[index]))

    # Greedy set cover, taking the best-scoring candidate among those covering the most
    # still-uncovered options. Greedy is not optimal cover, but the guarantee it has to
    # honour is "every option appears", and any cover honours that.
    kept_indices: list[int] = []
    taken: set[int] = set()
    uncovered = set(needed)
    for_coverage = 0
    while uncovered and len(kept_indices) < limit:
        best = -1
        best_gain = 0
        for i in order:
            if i in taken:
                continue
            gain = sum(
                1 for index in range(n_slots) if _slot_key(pool[i], index) in uncovered
            )
            if gain > best_gain:
                best, best_gain = i, gain
            if best_gain == n_slots:
                break
        if best < 0:
            break
        kept_indices.append(best)
        taken.add(best)
        for_coverage += 1
        for index in range(n_slots):
            uncovered.discard(_slot_key(pool[best], index))

    for i in order:
        if len(kept_indices) >= limit:
            break
        if i in taken:
            continue
        kept_indices.append(i)
        taken.add(i)

    kept = sorted(
        (scored[i] for i in kept_indices), key=lambda c: (-c.score, c.action.to_choice())
    )
    return Narrowed(
        kept=kept,
        considered=len(pool),
        uncovered=tuple(sorted(needed[k] for k in uncovered)),
        uncovered_options=tuple(
            options[k] for k in sorted(uncovered, key=lambda k: needed[k])
        ),
        for_coverage=for_coverage,
        for_score=len(kept) - for_coverage,
        by_kind=_count_kinds(reg, kept),
    )


def _count_kinds(reg: Regulation, kept: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in kept:
        key = action_kind(reg, c.action)
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "DEFAULT_LIMIT",
    "Candidate",
    "Narrowed",
    "action_kind",
    "narrow",
    "score_action",
]
