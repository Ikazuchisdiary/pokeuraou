"""Effective Speed, move priority, and the order actions resolve in.

Turn order is where most of the read-ahead value in doubles lives, and it is also the
place where the opponent's hidden SP spread bites hardest: their Speed stat is uncertain,
so different candidate spreads can produce different orders for the *same* pair of
actions. Everything here is therefore vectorised over the belief axis, and
:func:`order_groups` returns the distinct orders together with which particles produce
them, rather than a single order.

Showdown sorts the action queue by, from ``Battle#comparePriority``::

    1. order      ascending   (switch 103, megaEvo 104, move 200, residual 300)
    2. priority   descending
    3. speed      descending
    4. subOrder   ascending
    5. effectOrder ascending

so every switch happens before every Mega Evolution, which happens before every move.
Speed for a move action is ``Pokemon#getActionSpeed``, which is ``getStat('spe')`` negated
under Trick Room -- the champions mod overrides it only to drop the Gen-8 underflow.

``getStat('spe')`` is, in order: the stored stat, the boost ratio, the ``ModifySpe``
modifier chain applied once, and then paralysis, which sits at handler priority -101 and
so runs *after* the chain is finalised and halves the result outside it.

One consequence matters a lot for the resolver. From generation 8 on, Showdown recomputes
every active Pokemon's Speed and re-sorts the remaining queue *after each action*, as long
as the next action is a move::

    if (this.gen >= 8 && this.queue.peek()?.choice === 'move') {
        this.updateSpeed();
        for (const queueAction of this.queue.list) this.getActionSpeed(queueAction);
        this.queue.sort();
    }

So Tailwind going up, a Rock Tomb Speed drop or an item knocked off (waking Unburden)
reorders the actions that have not happened yet. :func:`order_groups` gives the order
implied by one snapshot of the state; a resolver has to call it again whenever a Speed
could have changed. :func:`speed_changing_effects` names the effects that require it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from .actions import PassAction, SideAction, SwitchAction
from .battler import Battler, FieldState
from .fixedpoint import Chain
from .position import Position
from .regulation import Regulation

#: Queue orders from ``BattleQueue#resolveAction``.
ORDER_SWITCH = 103
ORDER_MEGA = 104
ORDER_MOVE = 200
ORDER_RESIDUAL = 300

#: Showdown caps Speed at 10000 unless the format sets ``battle.trunc``; the Champions VGC
#: formats do not, so the cap applies.
SPEED_CAP = 10000

#: Speed-doubling abilities and the weather or terrain they need.
_SPEED_WEATHER_ABILITIES: dict[str, tuple[str, ...]] = {
    "chlorophyll": ("sunnyday", "desolateland"),
    "swiftswim": ("raindance", "primordialsea"),
    "sandrush": ("sandstorm",),
    "slushrush": ("hail", "snowscape", "snow"),
}

#: Items that scale Speed, as (numerator, denominator).
_SPEED_ITEMS: dict[str, tuple[float, float]] = {
    "choicescarf": (1.5, 1.0),
    "ironball": (0.5, 1.0),
    "machobrace": (0.5, 1.0),
    "powerweight": (0.5, 1.0),
    "powerbracer": (0.5, 1.0),
    "powerbelt": (0.5, 1.0),
    "powerlens": (0.5, 1.0),
    "powerband": (0.5, 1.0),
    "poweranklet": (0.5, 1.0),
}

#: Priority added by an ability, given the move.
_PRIORITY_ABILITIES = ("prankster", "galewings", "triage")

#: Fractional priority, which breaks ties within a priority bracket without changing it.
#: Quick Claw and Quick Draw are random; the rest are deterministic.
QUICK_CLAW_CHANCE = 0.2
QUICK_DRAW_CHANCE = 0.3


def effective_speed(
    reg: Regulation,
    mon: Battler,
    field_state: FieldState,
    side_conditions: frozenset[str],
) -> np.ndarray:
    """(N,) Speed as the turn-order comparison sees it, before Trick Room.

    Reproduces ``Pokemon#getStat('spe')``: boosts as an exact integer ratio, then one
    accumulated ``ModifySpe`` modifier, then paralysis on top.
    """
    del reg
    spe = mon.stat("spe")

    chain = Chain()
    # Side conditions.
    if "tailwind" in side_conditions:
        chain.add(2.0, label="tailwind")
    if "grasspledge" in side_conditions:
        chain.add(0.25, label="grasspledge")

    # Abilities.
    weathers = _SPEED_WEATHER_ABILITIES.get(mon.ability)
    if weathers is not None and _effective_weather(field_state, mon) in weathers:
        chain.add(2.0, label=mon.ability)
    if mon.ability == "surgesurfer" and field_state.terrain == "electricterrain":
        chain.add(2.0, label="surgesurfer")
    if mon.ability == "quickfeet" and mon.status is not None:
        chain.add(1.5, label="quickfeet")
    # Unburden only doubles Speed once the item is actually gone: Showdown adds a
    # volatile when the item is lost, and the condition on that volatile is what applies
    # the modifier. Holding no item from the start is not enough.
    if mon.ability == "unburden" and mon.item is None and "unburden" in mon.volatiles:
        chain.add(2.0, label="unburden")
    if mon.ability == "slowstart" and mon.ability_state.get("counter"):
        chain.add(0.5, label="slowstart")

    # Items.
    if mon.item is not None:
        ratio = _SPEED_ITEMS.get(mon.item)
        if ratio is not None:
            chain.add(*ratio, label=mon.item)
        elif mon.item == "quickpowder" and mon.species == "ditto":
            chain.add(2.0, label="quickpowder")

    spe = chain.apply(spe)

    # Paralysis runs at handler priority -101, i.e. after the chain is finalised, and
    # halves with a floor rather than chaining. Quick Feet suppresses it.
    if mon.status == "par" and mon.ability != "quickfeet":
        spe = spe * 50 // 100

    return np.minimum(spe, SPEED_CAP)


def _effective_weather(field_state: FieldState, mon: Battler) -> str | None:
    """Weather as this Pokemon experiences it, for the Speed abilities.

    Cloud Nine and Air Lock suppress weather field-wide, which is why every active
    ability is consulted and not just this one's.
    """
    if field_state.weather is None:
        return None
    suppressed = any(
        ability in ("cloudnine", "airlock")
        for side in field_state.active_abilities
        for ability in side
    )
    if suppressed:
        return None
    if mon.item == "utilityumbrella" and field_state.weather in (
        "sunnyday", "raindance", "desolateland", "primordialsea"
    ):
        return None
    return field_state.weather


def move_priority(
    reg: Regulation, move_id: str, mon: Battler, field_state: FieldState
) -> int:
    """The move's priority after ability and terrain modifiers."""
    move = reg.moves.get(move_id)
    if move is None:
        # The recharge turn's fake move. It has no dex entry and nothing to modify:
        # Showdown queues the action at plain priority 0 and intercepts it before it runs.
        return 0
    priority = move.priority

    # At most one of these applies to any real move, but they are separate abilities and
    # keeping them separate keeps the reason for each visible.
    if mon.ability == "prankster":
        priority += 1 if move.category == "Status" else 0
    elif mon.ability == "galewings":
        priority += 1 if move.type == "Flying" and bool(mon.at_full_hp.all()) else 0
    elif mon.ability == "triage":
        priority += 3 if "heal" in move.flags else 0

    if move_id == "grassyglide" and field_state.terrain == "grassyterrain":
        priority += 1

    return priority


def fractional_priority(
    reg: Regulation, move_id: str, mon: Battler
) -> list[tuple[float, float]]:
    """Fractional priority outcomes as (value, probability).

    Quick Claw and Quick Draw are coin flips, so they produce two branches. Everything
    else is deterministic and produces one.

    The recharge turn's fake move has no dex entry, and is given plain priority with no
    roll. Showdown would consult Quick Claw against an empty move object, but the turn
    does nothing either way and a fractional-priority branch on it would double the
    branch count of every recharge turn to move a no-op earlier or later.

    Showdown runs this as one ``FractionalPriority`` event (``battle-queue.ts``: relay
    value 0), its handlers in ``onFractionalPriorityPriority`` order, each seeing the value
    the previous ones left (IKA-145):

    * priority 0: Stall, Lagging Tail and Full Incense set -0.1 (constants);
    * priority -1: Mycelium Might sets -0.1 on a status move; Quick Draw, on a move that
      is not a status move, rolls 3/10 for 0.1;
    * priority -2: Quick Claw returns at once on a status move under Mycelium Might, and
      otherwise rolls 1/5 for 0.1 **if the value so far is <= 0**.

    That ``priority`` is the relay value, not the move's priority: Quick Claw fires on a
    status move and on a +1 move alike, and only a Quick Draw that already fired stops it.
    """
    move = reg.moves.get(move_id)
    if move is None:
        return [(0.0, 1.0)]
    status = move.category == "Status"
    base = 0.0
    if mon.ability == "stall" or mon.item in ("laggingtail", "fullincense"):
        base = -0.1
    if mon.ability == "myceliummight" and status:
        base = -0.1
    # (value, probability) after the priority -1 handlers.
    outcomes = [(base, 1.0)]
    if mon.ability == "quickdraw" and not status:
        outcomes = [(0.1, QUICK_DRAW_CHANCE), (base, 1.0 - QUICK_DRAW_CHANCE)]
    if mon.item == "quickclaw" and not (status and mon.ability == "myceliummight"):
        rolled: dict[float, float] = {}
        for value, chance in outcomes:
            if value <= 0:
                rolled[0.1] = rolled.get(0.1, 0.0) + chance * QUICK_CLAW_CHANCE
                rolled[value] = rolled.get(value, 0.0) + chance * (1.0 - QUICK_CLAW_CHANCE)
            else:
                rolled[value] = rolled.get(value, 0.0) + chance
        outcomes = list(rolled.items())
    return outcomes


@dataclass(slots=True)
class QueuedAction:
    """One entry in the turn's action queue.

    ``speed`` carries the belief axis: the opponent's Speed follows from their hidden SP
    spread, so this is an array and the resulting order can differ per particle.
    """

    side: int
    slot: int
    kind: str  # 'switch' | 'mega' | 'move'
    order: int
    priority: int
    fractional: float
    speed: np.ndarray
    move_id: str | None = None
    target: int | None = None
    switch_to: int | None = None
    #: Species of the Pokemon being switched to. Party indices move when another switch
    #: resolves first, so the species -- unique within a team under Species Clause -- is
    #: what identifies the target reliably.
    switch_species: str | None = None
    #: Probability of this action's fractional-priority branch.
    branch_probability: float = 1.0

    def label(self, reg: Regulation) -> str:
        if self.kind == "switch":
            return f"p{self.side + 1}{'ab'[self.slot]} switch->{self.switch_to}"
        if self.kind == "mega":
            return f"p{self.side + 1}{'ab'[self.slot]} mega"
        # `.get`: the recharge turn's action names a move with no dex entry, and the
        # label goes into the event log, where a crash would be the worst outcome.
        known = reg.moves.get(self.move_id) if self.move_id else None
        name = known.name if known is not None else (self.move_id or "?")
        return f"p{self.side + 1}{'ab'[self.slot]} {name}"


def sort_keys(actions: list[QueuedAction], trick_room: bool) -> tuple[np.ndarray, ...]:
    """The three comparison columns, each (A, n).

    Kept as separate arrays and combined with :func:`numpy.lexsort` rather than packed
    into one integer: packing needs hand-chosen bit widths, and a Speed or priority
    outside the chosen range would silently reorder the turn.

    Priority is scaled by 10 so the fractional priorities (Quick Claw's +0.1, Stall's
    -0.1) stay integral.
    """
    n = max(a.speed.shape[0] for a in actions)
    order = np.empty((len(actions), n), dtype=np.int64)
    priority = np.empty((len(actions), n), dtype=np.int64)
    speed = np.empty((len(actions), n), dtype=np.int64)
    for i, a in enumerate(actions):
        spe = np.broadcast_to(a.speed, (n,)).astype(np.int64)
        if trick_room:
            # getActionSpeed negates Speed under Trick Room, for every action kind.
            spe = -spe
        order[i] = a.order
        priority[i] = int(round((a.priority + a.fractional) * 10))
        speed[i] = spe
    return order, priority, speed


@dataclass(slots=True)
class OrderGroup:
    """One distinct action order, and the particles that produce it."""

    #: (n,) which particles resolve in this order.
    mask: np.ndarray
    #: Indices into the action list, in resolution order.
    order: tuple[int, ...]
    #: Index groups that compare exactly equal. Showdown shuffles these, so each is a
    #: genuine coin flip the caller must branch on rather than a detail to pick.
    ties: tuple[tuple[int, ...], ...]

    @property
    def count(self) -> int:
        return int(self.mask.sum())


def order_groups(actions: list[QueuedAction], *, trick_room: bool) -> list[OrderGroup]:
    """Distinct action orders, grouped by which particles produce them.

    A single order would be wrong: the opponent's Speed comes from their hidden spread, so
    the same pair of actions can resolve in different orders for different candidate
    spreads, and that difference is often the whole question in a doubles turn.
    """
    if not actions:
        return []
    order_col, prio_col, speed_col = sort_keys(actions, trick_room)
    n = order_col.shape[1]

    # Group by the sequence *and* its tie structure. Grouping by sequence alone would
    # merge a particle whose Speeds tie with one where they merely happen to sort the same
    # way, and the tie -- a coin flip the resolver has to branch on -- would be lost.
    grouped: dict[tuple[tuple[int, ...], tuple[tuple[int, ...], ...]], list[int]] = {}
    for p in range(n):
        # lexsort takes the primary key last.
        seq = tuple(int(i) for i in np.lexsort((-speed_col[:, p], -prio_col[:, p], order_col[:, p])))
        ties: list[tuple[int, ...]] = []
        run = [seq[0]]
        for prev, cur in zip(seq, seq[1:], strict=False):
            same = (
                order_col[cur, p] == order_col[prev, p]
                and prio_col[cur, p] == prio_col[prev, p]
                and speed_col[cur, p] == speed_col[prev, p]
            )
            if same:
                run.append(cur)
            else:
                if len(run) > 1:
                    ties.append(tuple(run))
                run = [cur]
        if len(run) > 1:
            ties.append(tuple(run))
        grouped.setdefault((seq, tuple(ties)), []).append(p)

    out: list[OrderGroup] = []
    for (seq, ties_key), particles in grouped.items():
        mask = np.zeros(n, dtype=bool)
        mask[particles] = True
        out.append(OrderGroup(mask=mask, order=seq, ties=ties_key))
    return out


def build_queue(
    reg: Regulation,
    pos: Position,
    side_actions: Sequence[SideAction],
    battlers: Sequence[Sequence[Battler | None]],
    field_state: FieldState,
) -> list[list[QueuedAction]]:
    """Turns both sides' chosen actions into queue entries.

    Returns one list per fractional-priority branch: Quick Claw and Quick Draw are coin
    flips that change the order, so they multiply out here rather than being averaged
    away. ``branch_probability`` on the first action of each list carries the weight.
    """
    base: list[tuple[QueuedAction, list[tuple[float, float]]]] = []

    for side_index, action in enumerate(side_actions):
        side = pos.sides[side_index]
        for slot_action in action.slots:
            slot = slot_action.slot
            mon = battlers[side_index][slot] if slot < len(battlers[side_index]) else None
            if mon is None:
                continue
            party = side.active[slot]
            if party is None:
                continue
            speed = effective_speed(
                reg, mon, field_state, frozenset(c.id for c in side.side_conditions)
            )

            if isinstance(slot_action, SwitchAction):
                base.append(
                    (
                        QueuedAction(
                            side=side_index,
                            slot=slot,
                            kind="switch",
                            order=ORDER_SWITCH,
                            priority=0,
                            fractional=0.0,
                            speed=speed,
                            switch_to=slot_action.party_index - 1,
                            switch_species=slot_action.species,
                        ),
                        [(0.0, 1.0)],
                    )
                )
                continue
            if isinstance(slot_action, PassAction):
                continue

            if slot_action.mega:
                base.append(
                    (
                        QueuedAction(
                            side=side_index,
                            slot=slot,
                            kind="mega",
                            order=ORDER_MEGA,
                            priority=0,
                            fractional=0.0,
                            speed=speed,
                        ),
                        [(0.0, 1.0)],
                    )
                )
            # A Pokemon part-way through a charging move is locked into it, whatever was
            # chosen: Showdown fires the stored move with `[from] lockedmove`. Honouring
            # the choice instead means resolving a move that never happens and skipping
            # one that does.
            move_id = slot_action.move_id
            charging = next(
                (v for v in pos.sides[side_index].pokemon[party].volatiles if v.id == "twoturnmove"),
                None,
            )
            if charging is not None and charging.move:
                move_id = charging.move

            base.append(
                (
                    QueuedAction(
                        side=side_index,
                        slot=slot,
                        kind="move",
                        order=ORDER_MOVE,
                        priority=move_priority(reg, move_id, mon, field_state),
                        fractional=0.0,
                        speed=speed,
                        move_id=move_id,
                        target=slot_action.target,
                    ),
                    fractional_priority(reg, move_id, mon),
                )
            )

    # Multiply out the fractional-priority branches.
    branches: list[list[QueuedAction]] = [[]]
    weights: list[float] = [1.0]
    for action, options in base:
        if len(options) == 1:
            value, _ = options[0]
            for branch in branches:
                branch.append(replace(action, fractional=value))
            continue
        new_branches: list[list[QueuedAction]] = []
        new_weights: list[float] = []
        for branch, weight in zip(branches, weights, strict=True):
            for value, probability in options:
                new_branches.append([*branch, replace(action, fractional=value)])
                new_weights.append(weight * probability)
        branches, weights = new_branches, new_weights

    for branch, weight in zip(branches, weights, strict=True):
        if branch:
            # A copy, not a write. Every action queued before a split is one object shared
            # by the branches after it, so writing the weight into `branch[0]` left all of
            # them holding the last branch's: a Quick Claw anywhere but the first queued
            # action weighed 0.8 and 0.8, a turn of mass 1.6, and the claw fired half the
            # time. The port clones its queues and never had this (IKA-70).
            branch[0] = replace(branch[0], branch_probability=weight)
    return branches


#: Effects that can change an active Pokemon's Speed mid-turn, and therefore require the
#: remaining queue to be re-sorted. Used by the resolver to decide when to re-sort, and by
#: the order test to tell an unpredictable turn from a wrong prediction.
SPEED_CHANGING_EFFECTS = frozenset(
    {
        # Speed stat changes
        "boost:spe", "unboost:spe",
        # Side conditions
        "tailwind", "grasspledge",
        # Status
        "par",
        # Weather and terrain, through the Speed abilities
        "weather", "terrain",
        # Item gain or loss, through Choice Scarf, Iron Ball, Unburden
        "item",
        # Ability change, through the Speed abilities
        "ability",
    }
)


#: Effects that replace a queued action after the turn has started, which reorders the
#: queue for a reason that is not a Speed change.
ACTION_OVERRIDING_EFFECTS = frozenset({"encore", "instruct", "dancer", "afteryou", "quash"})

#: The subset the resolver models, and which therefore must *not* be excluded from a
#: differential comparison. Encore is the only one of the five that occurs in this format --
#: 109 of the 394 tournament teams carry it, against none for Instruct and Quash -- and it
#: was being skipped, which is precisely how a whole class of bug stays invisible while the
#: divergence rate looks healthy.
MODELLED_ACTION_OVERRIDES = frozenset({"encore"})


def action_overriding_effects(
    protocol_lines: list[str], *, only_unmodelled: bool = False
) -> set[str]:
    """Which action-overriding effects a turn's protocol shows.

    ``only_unmodelled`` is for callers that *skip* on the result: they want what the
    resolver cannot reproduce, not everything that reordered the queue. A caller that
    merely names the causes of a re-sort wants all of them.
    """
    names = (
        ACTION_OVERRIDING_EFFECTS - MODELLED_ACTION_OVERRIDES
        if only_unmodelled
        else ACTION_OVERRIDING_EFFECTS
    )
    found: set[str] = set()
    for line in protocol_lines:
        lowered = line.lower()
        for name in names:
            if f"move: {name}" in lowered or f"|{name}|" in lowered:
                found.add(name)
    return found


def speed_changing_effects(protocol_lines: list[str]) -> set[str]:
    """Which Speed-changing effects a turn's protocol shows.

    A turn that contains any of these cannot be ordered from the pre-turn state alone,
    because Showdown re-sorts the queue partway through.
    """
    found: set[str] = set()
    for line in protocol_lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue
        tag = parts[1]
        if tag in ("-boost", "-unboost") and len(parts) > 3 and parts[3] == "spe":
            found.add(f"{tag[1:]}:spe")
        elif tag in ("-sidestart", "-sideend") and len(parts) > 3:
            name = parts[3].split(": ")[-1].lower().replace(" ", "")
            if name in ("tailwind", "grasspledge"):
                found.add(name)
        elif tag == "-status" and len(parts) > 3 and parts[3] == "par":
            found.add("par")
        elif tag in ("-weather", "-fieldstart", "-fieldend"):
            found.add("weather" if tag == "-weather" else "terrain")
        elif tag in ("-enditem", "-item", "-endability", "-ability"):
            found.add("item" if "item" in tag else "ability")
        elif tag in ("-mega", "detailschange", "-primal"):
            # A Mega Evolution changes the base stats, so the Speed of a Pokemon that has
            # not moved yet changes mid-turn and the queue is re-sorted around it. The
            # resolver handles that because the mega is an action in its own queue; a
            # predictor working from the pre-turn position alone cannot.
            found.add("mega")
    return found


@dataclass(slots=True)
class SpeedReport:
    """Turn-order facts worth showing a human.

    The comparisons, not the raw numbers, are what a player reasons about: whether a
    Pokemon outspeeds, and how much of the belief says so.
    """

    #: (our slot, their slot) -> probability that ours moves first at equal priority.
    outspeed_probability: dict[tuple[int, int], float] = field(default_factory=dict)
    #: (our slot, their slot) -> probability of an exact Speed tie.
    tie_probability: dict[tuple[int, int], float] = field(default_factory=dict)


def speed_comparison_report(
    ours: list[Battler | None],
    theirs: list[Battler | None],
    weights: np.ndarray,
    *,
    trick_room: bool,
) -> SpeedReport:
    """Pairwise Speed comparisons, weighted by the belief.

    ``weights`` are the particle probabilities of the *opponent's* spread; our own Speed is
    known, so a comparison is uncertain only through theirs.
    """
    report = SpeedReport()
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else w
    for i, mine in enumerate(ours):
        if mine is None:
            continue
        my_speed = int(mine.stat("spe")[0])
        for j, other in enumerate(theirs):
            if other is None:
                continue
            their_speed = other.stat("spe")
            faster = my_speed > their_speed
            if trick_room:
                faster = my_speed < their_speed
            tie = my_speed == their_speed
            report.outspeed_probability[(i, j)] = float((w * faster).sum())
            report.tie_probability[(i, j)] = float((w * tie).sum())
    return report
