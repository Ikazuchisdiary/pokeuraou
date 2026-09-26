"""Choosing which action combinations get solved exactly.

A doubles turn has a few hundred legal choices per side -- the product of two slots'
options -- and the matrix is the product of both sides', so solving every cell exactly is
not affordable. Narrowing is therefore unavoidable, and the only question is whether what
it throws away is *sayable*.

Two rules keep it sayable.

**Nothing is dropped quietly.** Ranking by damage and keeping the top N would silently
delete every switch and every status move from the answer -- a Protect or a Fake Out
scores zero damage and would never survive a damage ranking, which is exactly backwards
for the positions where those are the answer. So coverage comes first: the kept set is
chosen to contain every individual slot option that appears anywhere in the candidates,
and only then is the remaining budget spent on the highest-scoring combinations. If the
budget is too small to cover everything, the options left out are returned in
:attr:`Narrowed.uncovered` and printed.

The cover is where a menu starts, not a promise that every option stays on it. A search
that has solved the menu knows more than the cover does, and the root's double oracle
that swaps (`deepen`, ``s<W>``; the user's decision of 9/26, IKA-293 / IKA-310) may push
an action out once it carries no weight in the equilibrium -- even when it was the last
one carrying some option. Three conditions keep that sayable: only a weightless action
leaves; the options the menu no longer covers are reported, as :attr:`Narrowed.uncovered`
reports the budget's (`slot_options` is the shared account, `Deepened.uncovered` the
decision's record); and the action is outside the menu again, where the oracle asks it
every round like any other candidate, so it comes back the moment it is a best
response. What is eliminated is always an option somebody can name.

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
from .moveinfo import FIRST_TURN_OUT_MOVES
from .position import Position
from .regulation import Regulation

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


class Candidate:
    """One side's complete choice, with the score that ranked it.

    `action`, `score` and `detail` as the frozen dataclass this was; `detail` may be handed
    over unwritten (IKA-321) and is then written the first time it is read.
    """

    __slots__ = ("_detail", "action", "score")

    def __init__(
        self,
        action: SideAction,
        score: float,
        detail: tuple[str, ...] | _PortDetail | _LeafDetail = (),
    ) -> None:
        self.action = action
        #: Sum over slots of expected damage as a fraction of the target's current HP, with
        #: damage to one's own partner subtracted. Ordering only; never printed as a result.
        self.score = score
        self._detail = detail

    @property
    def detail(self) -> tuple[str, ...]:
        """Per-slot contributions, so a surprising ranking can be read rather than trusted.

        Nothing in a search reads them -- the CLI, a record or a test does -- and writing
        every candidate's lines was 14% of `narrow` (IKA-319), so the port's parts are kept
        and the lines written here, the same text the eager version wrote (IKA-321).
        """
        text = self._detail
        if type(text) is not tuple:
            text = text.write()
            self._detail = text
        return text

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not Candidate:
            return NotImplemented
        return (self.action, self.score, self.detail) == (  # type: ignore[attr-defined]
            other.action, other.score, other.detail,  # type: ignore[attr-defined]
        )

    def __hash__(self) -> int:
        return hash((self.action, self.score, self.detail))

    def __repr__(self) -> str:
        return f"Candidate(action={self.action!r}, score={self.score!r}, detail={self.detail!r})"

    def __reduce__(self) -> tuple[object, ...]:
        # Written out: an unwritten detail holds the regulation.
        return (Candidate, (self.action, self.score, self.detail))


class _PortDetail:
    """A candidate's detail lines from the port's parts, written when read (IKA-321)."""

    __slots__ = ("action", "parts", "reg")

    def __init__(
        self, reg: Regulation, action: SideAction, parts: list[tuple[int, int, bool, float, bool]]
    ) -> None:
        self.reg = reg
        self.action = action
        self.parts = parts

    def write(self) -> tuple[str, ...]:
        reg, slots = self.reg, self.action.slots
        return tuple(
            f"{slots[slot].describe(reg)} -> "
            f"{'foe' if is_foe else 'ally'}{target_slot + 1} {signed:+.3f}{'' if exact else '?'}"
            for slot, target_slot, is_foe, signed, exact in self.parts
        )


class _LeafDetail:
    """The leaf's score in front of the damage detail it replaced, written when read."""

    __slots__ = ("under", "value")

    def __init__(self, value: float, under: Candidate) -> None:
        self.value = value
        self.under = under

    def write(self) -> tuple[str, ...]:
        return (f"leaf {self.value:+.4f}", *self.under.detail)


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


def slot_options(reg: Regulation, actions: Sequence[SideAction]) -> dict[str, str]:
    """Every single slot option `actions` carry, as the cover counts them: key -> label.

    The same identities and the same labels as :attr:`Narrowed.uncovered`, so what a
    later step takes out of a menu's cover is reported in the same words (IKA-293).
    """
    out: dict[str, str] = {}
    for action in actions:
        for index in range(len(action.slots)):
            out.setdefault(_slot_key(action, index), _slot_label(reg, action, index))
    return out


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

    return [
        Candidate(action, total, _PortDetail(reg, action, parts))
        for action, (total, parts) in zip(pool, scored, strict=True)
    ]


#: `narrow`'s "ask the port yourself" (a pool `narrow_many` scored may have been refused,
#: which is None, so None cannot mean "not asked").
_ASK = object()


def _bridged_scores_many(
    reg: Regulation, asks: Sequence[tuple[Position, int, list[SideAction]]]
) -> list[list[Candidate] | None]:
    """`_bridged_scores` of many pools in one crossing (IKA-295): the same candidates each,
    and None where the port declined one (or all of them, where `_bridged_scores` would)."""
    from . import rustnode

    if not asks:
        return []
    if not rustnode.available():
        return [None] * len(asks)
    node = rustnode.node_for(reg)
    if node is None:
        return [None] * len(asks)
    try:
        answers = node.score_many(asks)
    except Exception as exc:  # noqa: BLE001 - a broken bridge must not fail the run
        rustnode.disable(str(exc))
        return [None] * len(asks)
    out: list[list[Candidate] | None] = []
    for (_pos, _side, pool), scored in zip(asks, answers, strict=True):
        if scored is None:
            out.append(None)
            continue
        out.append(
            [
                Candidate(action, total, _PortDetail(reg, action, parts))
                for action, (total, parts) in zip(pool, scored, strict=True)
            ]
        )
    return out


def narrow_many(
    reg: Regulation, asks: Sequence[tuple[Position, int]], *, limit: int = DEFAULT_LIMIT
) -> list[Narrowed | Exception]:
    """`narrow(reg, pos, side, limit=limit)` of every (pos, side), the port's damage scores
    for all their pools asked in one crossing (IKA-295).

    Each answer is `narrow`'s: the pool is built the same way, the same request is asked of
    the port, and the same Python ranks and covers. A pool whose building raised is that
    exception, in its place, for the caller to raise where it would have been met.
    """
    pools: list[list[SideAction] | Exception] = []
    for pos, side in asks:
        try:
            pools.append(drop_dead_actions(reg, pos, side, side_actions(reg, pos, side)))
        except Exception as exc:  # noqa: BLE001 - handed back, raised by the caller
            pools.append(exc)
    wanted = [
        index for index, pool in enumerate(pools) if not isinstance(pool, Exception) and pool
    ]
    scored = _bridged_scores_many(
        reg, [(asks[index][0], asks[index][1], pools[index]) for index in wanted]
    )
    found = dict(zip(wanted, scored, strict=True))
    out: list[Narrowed | Exception] = []
    for index, (pos, side) in enumerate(asks):
        pool = pools[index]
        if isinstance(pool, Exception):
            out.append(pool)
            continue
        try:
            out.append(
                narrow(
                    reg, pos, side, limit=limit, candidates=pool,
                    _prescored=found.get(index, _ASK),
                )
            )
        except Exception as exc:  # noqa: BLE001 - handed back, raised by the caller
            out.append(exc)
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
    _prescored: object = _ASK,
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
        # `narrow_many` asked the port for this pool already, with the others (IKA-295).
        scored = (
            _bridged_scores(reg, pos, side, pool) if _prescored is _ASK else _prescored
        )
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
            Candidate(c.action, value, _LeafDetail(value, c))
            for c, value in ((c, float(v)) for c, v in zip(scored, values, strict=True))
        ]
    # Highest score first, with a stable tiebreak so the same position gives the same
    # answer twice.
    choices = _choices(pool)
    order = sorted(range(len(scored)), key=lambda i: (-scored[i].score, choices[i]))
    if len(pool) <= limit:
        kept = [scored[i] for i in order]
        return Narrowed(
            kept=kept, considered=len(pool), for_score=len(kept),
            by_kind=_count_kinds(reg, kept),
        )
    out = _cover(reg, pool, scored, order, limit)
    out.by_kind = _count_kinds(reg, out.kept)
    return out


def _cover(
    reg: Regulation,
    pool: list[SideAction],
    scored: list[Candidate],
    order: list[int],
    limit: int,
) -> Narrowed:
    """`narrow` past its sort, for a pool larger than `limit`: the cover, then the best
    scores, then the kept set in score order (`by_kind` is left to the caller)."""
    n_slots = len(pool[0].slots)
    keys, options = _option_ids(pool, n_slots)

    # Greedy set cover, taking the best-scoring candidate among those covering the most
    # still-uncovered options. Greedy is not optimal cover, but the guarantee it has to
    # honour is "every option appears", and any cover honours that.
    #
    # Each candidate's options are small integers written once (IKA-321): rebuilding every
    # candidate's `_slot_key` strings on every round was 39% of `narrow` (IKA-319). The
    # rounds, the order they scan and the gains they compare are the same.
    kept_indices: list[int] = []
    taken = bytearray(len(pool))
    uncovered = bytearray(b"\x01") * len(options)
    left = len(options)
    for_coverage = 0
    while left and len(kept_indices) < limit:
        best = -1
        best_gain = 0
        for i in order:
            if taken[i]:
                continue
            if n_slots == 2:
                first, second = keys[i]
                gain = uncovered[first] + uncovered[second]
            else:
                gain = sum(uncovered[k] for k in keys[i])
            if gain > best_gain:
                best, best_gain = i, gain
            if best_gain == n_slots:
                break
        if best < 0:
            break
        kept_indices.append(best)
        taken[best] = 1
        for_coverage += 1
        for k in keys[best]:
            if uncovered[k]:
                uncovered[k] = 0
                left -= 1

    for i in order:
        if len(kept_indices) >= limit:
            break
        if taken[i]:
            continue
        kept_indices.append(i)
        taken[i] = 1

    kept = sorted(
        (scored[i] for i in kept_indices), key=lambda c: (-c.score, c.action.to_choice())
    )
    # Only the options left out are named, so only they are written.
    labels = {
        k: f"slot{options[k][0] + 1} {options[k][1].describe(reg)}"  # as `_slot_label`
        for k in range(len(options))
        if uncovered[k]
    }
    return Narrowed(
        kept=kept,
        considered=len(pool),
        uncovered=tuple(sorted(labels.values())),
        uncovered_options=tuple(options[k] for k in sorted(labels, key=labels.__getitem__)),
        for_coverage=for_coverage,
        for_score=len(kept) - for_coverage,
    )


def _choices(pool: Sequence[SideAction]) -> list[str]:
    """`action.to_choice()` of every action in `pool`, each slot action's written once
    (the combinations share their slot action objects, `_option_ids`)."""
    memo: dict[int, str] = {}
    out = []
    for action in pool:
        parts = []
        for slot in action.slots:
            text = memo.get(id(slot))
            if text is None:
                text = memo[id(slot)] = slot.to_choice()
            parts.append(text)
        out.append(", ".join(parts))
    return out


def _option_ids(
    pool: Sequence[SideAction], n_slots: int
) -> tuple[list[tuple[int, ...]], list[tuple[int, object]]]:
    """Each candidate's slot options as integers, and each integer's (slot index, slot
    action) where it was first met -- the options `_slot_key` tells apart, numbered.

    A slot action object is usually shared by every combination it is in (`side_actions`
    builds the product of the slots' lists), so its key is written once per object; a pool
    whose objects are not shared still gets the same numbers through the key text.
    """
    ids: dict[str, int] = {}
    options: list[tuple[int, object]] = []
    seen: list[dict[int, int]] = [{} for _ in range(n_slots)]
    keys: list[tuple[int, ...]] = []
    for action in pool:
        slots = action.slots
        row = []
        for index in range(n_slots):
            slot = slots[index]
            memo = seen[index]
            k = memo.get(id(slot))
            if k is None:
                name = _slot_key(action, index)
                k = ids.get(name)
                if k is None:
                    k = ids[name] = len(options)
                    options.append((index, slot))
                memo[id(slot)] = k
            row.append(k)
        keys.append(tuple(row))
    return keys, options


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
    "slot_options",
]
