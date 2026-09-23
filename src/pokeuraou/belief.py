"""What we do not know about the opponent, and how much of it actually matters.

Champions shows open team sheets, and Showdown's implementation reveals nature there but
blanks the spread. So what a SHEET hides is six numbers per opposing Pokemon: its SP
allocation. Species, ability, item, moves and nature are public.

That is this module's part of the hidden state, not all of it (IKA-123, IKA-130). The
opponent brings four of the six, and the two that did not lead stay unknown until they
come in -- `pokeuraou.hidden` carries that belief. And the exact HP inside the displayed
percentage follows from the spread, so it is hidden with it (`pokeuraou.hpdisplay`).
Self-play's search is still shown the opponent's spreads and exact HP; only the bench is
hidden from it today.

The obvious representation is a handful of "archetypes" per species. Measured against the
usage data, that does not work: Incineroar alone appears with 10,154 distinct spreads, and
the eight most common complete stat vectors carry 28.7% of the probability mass. A tool
that reasons over the top eight is reasoning about a quarter of the opponents it will
meet.

So the belief is the full weighted particle set, and it is made affordable by *reduction
rather than truncation*. Two spreads that this position cannot tell apart are merged, and
the merge is lossless: the position resolves identically for every member of a class, so
merging changes no number, only the cost of computing it.

What makes two spreads indistinguishable is a property of the position, not of the
Pokemon:

- **A stat nothing reads is free.** Special Attack does not matter to a Pokemon holding
  four physical moves. Defense does not matter if no opposing Pokemon carries a physical
  move.
- **Speed only matters through order.** Two Speeds that sit between the same pair of
  thresholds produce the same turn, so what a particle needs to carry is not its Speed but
  the sign of its comparison against every Speed that could appear on the field. Equality
  is its own case, because a tie is a coin flip and not a resolution.
- **Unless a move reads the raw number.** Body Press attacks with Defense, Foul Play with
  the *target's* Attack, Electro Ball and Gyro Ball with the Speed ratio. When one of
  those is present anywhere in the position, the stat it reads stops being collapsible
  and is kept exactly. The list is checked against the position, so a mistake here costs
  compute rather than correctness in the common case -- but a missing entry would make the
  reduction lossy, so :data:`STAT_READING_MOVES` is the one table in this module that has
  to be complete.

Milestone 1 stops at the prior: particles conditioned on the revealed nature, reduced for
the position at hand. There is no evidence to condition on yet, so what the CLI prints is
labelled a prior. The Bayesian update from observed damage and turn order is milestone 2
and slots in at :func:`SpreadBelief.reweight`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from .battler import Battler
from .hpdisplay import band, displayed_colour, displayed_percent, uses_floor_display
from .position import Pokemon, Position
from .priors import MetagamePrior

if TYPE_CHECKING:
    from .names import Localiser

from .regulation import STAT_IDS, Regulation
from .speed import effective_speed
from .stats import stats_from_sp
from .view import field_state

#: Stats read by Showdown *code* rather than by a declarative field, so the dump cannot
#: say so and this module has to. Which stat a damaging move attacks and defends with
#: comes from `overrideOffensiveStat` / `overrideDefensiveStat` in the regulation config;
#: what is left here is base powers computed from a stat and the stat-swapping moves.
#:
#: This is the one table in the module that has to be complete: a missing entry makes the
#: reduction merge particles that resolve differently, which is worse than no reduction.
#: :func:`unmodelled_stat_readers` checks it against the regulation's move pool so a
#: regulation that adds one is a test failure rather than a silent wrong answer.
STAT_READING_MOVES: dict[str, tuple[str, ...]] = {
    # basePowerCallback from a Speed ratio.
    "electroball": ("spe",),
    "gyroball": ("spe",),
    # Swap or move stats around, so any of them can end up mattering.
    "powertrick": ("atk", "def"),
    "powerswap": ("atk", "spa"),
    "guardswap": ("def", "spd"),
    "speedswap": ("spe",),
    "heartswap": ("atk", "def", "spa", "spd", "spe"),
    "wonderroom": ("def", "spd"),
    # basePowerCallback over the party's Attack.
    "beatup": ("atk",),
    # Return the damage taken, so the attacker's own stats never enter.
    "metalburst": (),
    "counter": (),
    "mirrorcoat": (),
}

#: Abilities that read a raw stat rather than a comparison. Download compares the targets'
#: Defense and Special Defense on switch-in, which a mid-turn switch can trigger.
STAT_READING_ABILITIES: dict[str, tuple[str, ...]] = {
    "download": ("def", "spd"),
}

#: The offensive stat each damaging category uses.
CATEGORY_STAT = {"Physical": "atk", "Special": "spa"}
#: ...and the defensive one it is measured against.
CATEGORY_DEFENCE = {"Physical": "def", "Special": "spd"}


class BeliefError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SpreadBelief:
    """Weighted particles over one Pokemon's hidden SP spread.

    ``spreads`` is (K, 6) in SP space and ``stats`` the (K, 6) final stats it produces, so
    nothing downstream has to re-derive them. ``weights`` sums to 1.
    """

    species_id: str
    nature: str
    spreads: np.ndarray
    stats: np.ndarray
    weights: np.ndarray
    #: How the particle set was arrived at, for the report. Never a silent fallback.
    provenance: str = "usage prior conditioned on the revealed nature"

    @property
    def size(self) -> int:
        return int(self.weights.shape[0])

    def reweight(self, likelihood: np.ndarray) -> SpreadBelief:
        """Multiplies in a (K,) likelihood and renormalises.

        This is where milestone 2's evidence enters -- an observed damage number, a turn
        order that rules out a Speed range. Nothing calls it yet, and it is here so that
        the update has one place to be and cannot be spread across the resolver.
        """
        likelihood = np.asarray(likelihood, dtype=np.float64).reshape(-1)
        if likelihood.shape != self.weights.shape:
            raise BeliefError(
                f"likelihood has {likelihood.shape[0]} entries for {self.size} particles"
            )
        posterior = self.weights * likelihood
        total = float(posterior.sum())
        if total <= 0:
            raise BeliefError(
                "the evidence has probability zero under every particle; the belief or "
                "the observation is wrong, and guessing which would be worse than failing"
            )
        return replace(self, weights=posterior / total)

    def marginal(self, stat: str) -> list[tuple[int, float]]:
        """(SP value, probability) for one stat, heaviest first."""
        index = STAT_IDS.index(stat)
        out: dict[int, float] = {}
        for value, weight in zip(self.spreads[:, index], self.weights, strict=True):
            out[int(value)] = out.get(int(value), 0.0) + float(weight)
        return sorted(out.items(), key=lambda kv: -kv[1])

    def mean_sp(self) -> dict[str, float]:
        return {
            stat: float(np.dot(self.spreads[:, i], self.weights))
            for i, stat in enumerate(STAT_IDS)
        }


@dataclass(frozen=True, slots=True)
class Reduction:
    """A position-dependent lossless collapse of a particle set."""

    belief: SpreadBelief
    #: Which stats had to be kept exactly, and why.
    live_stats: tuple[str, ...]
    reasons: dict[str, str]
    #: Speed thresholds the comparison was bucketed against.
    speed_thresholds: tuple[int, ...]
    original_size: int
    #: For each class, the indices of the original particles it merged.
    members: tuple[tuple[int, ...], ...]
    #: The unreduced particle set. Kept so the losslessness claim can be *tested* --
    #: resolving two members of a class and comparing needs both members.
    original: SpreadBelief | None = None

    @property
    def size(self) -> int:
        return self.belief.size

    @property
    def factor(self) -> float:
        return self.original_size / max(1, self.size)

    def render(self, reg: Regulation | None = None, loc: Localiser | None = None) -> str:
        """The reduction, with the move names in the reasons resolved at display time.

        The reasons are stored with the move *id* in braces rather than a rendered name,
        because the same reduction is printed in more than one language and baking the
        English name in at construction would make that impossible.
        """

        def fill(reason: str) -> str:
            def resolve(match: re.Match[str]) -> str:
                move_id = match.group(1)
                if loc is not None:
                    return loc.move(move_id)
                found = reg.moves.get(move_id) if reg is not None else None
                return found.name if found is not None else move_id

            return re.sub(r"\{([a-z0-9]+)\}", resolve, reason)

        def stat(name: str) -> str:
            return loc.stat(name) if loc is not None else name

        kept = ", ".join(
            f"{stat(s)}（{fill(self.reasons.get(s, ''))}）" for s in self.live_stats
        )
        lines = [
            f"パーティクル {self.original_size} → 同値類 {self.size}"
            f"（{self.factor:.1f}倍の削減、無損失）",
            f"  この局面で効く能力値: {kept or 'なし'}",
        ]
        if self.speed_thresholds:
            thresholds = ", ".join(str(t) for t in self.speed_thresholds)
            lines.append(f"  素早さの比較境界: {thresholds}")
        return "\n".join(lines)


def build_belief(
    reg: Regulation, prior: MetagamePrior, species_id: str, nature: str
) -> SpreadBelief:
    """Particles for one opposing Pokemon, conditioned on the nature the sheet reveals.

    The open team sheet is evidence, and this is the cheapest place to use it: a revealed
    nature removes every particle that assumed a different one, which is a 20-fold cut
    before any reasoning happens.
    """
    entry = prior.species.get(species_id)
    species = reg.species.get(species_id)
    if species is None:
        raise BeliefError(f"{species_id} is not legal in {reg.meta.format_id}")

    base = np.array(species.base_stats, dtype=np.int64)
    nature_id = nature.lower()

    if entry is not None and entry.n_particles:
        match = np.array(
            [str(n).lower() == nature_id for n in entry.natures], dtype=bool
        )
        if match.any():
            spreads = entry.spreads[match]
            weights = entry.weights[match]
            weights = weights / weights.sum()
            provenance = (
                f"usage prior, {int(match.sum())} of {entry.n_particles} particles match "
                f"nature {nature}"
            )
        else:
            # The sheet says a nature the usage data never saw on this species. The
            # spreads are still informative about *where* points go, so they are kept and
            # re-derived under the revealed nature -- and the report says so, because this
            # belief is weaker than the conditioned one.
            spreads = entry.spreads
            weights = entry.weights / entry.weights.sum()
            provenance = (
                f"usage prior, but nature {nature} is unseen for this species: spreads "
                "reused under the revealed nature"
            )
    else:
        raise BeliefError(
            f"no usage data for {species_id}; a belief has to come from somewhere and "
            "inventing a spread distribution would make every number downstream fiction"
        )

    from .stats import nature_multipliers

    numerators = nature_multipliers(reg, [nature] * spreads.shape[0])
    stats = stats_from_sp(reg, base, spreads, numerators, level=50)
    return SpreadBelief(
        species_id=species_id,
        nature=nature,
        spreads=np.asarray(spreads, dtype=np.int64),
        stats=stats,
        weights=np.asarray(weights, dtype=np.float64),
        provenance=provenance,
    )


def live_stats_for(
    reg: Regulation, pos: Position, key: tuple[int, int]
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Which of one Pokemon's stats this position can distinguish, and why each survived.

    Liveness is a property of the Pokemon, not of the position: its own Attack is read by
    its own physical moves and by a foe's Foul Play, and by nothing else. Asking the
    position-wide question instead -- "does anybody here have a physical move?" -- makes
    every stat live in every ordinary doubles position and the reduction collapses
    nothing.

    Still conservative in the direction that matters. A stat kept unnecessarily costs
    compute; a stat dropped wrongly would merge particles that resolve differently, so
    everything that could read a raw value keeps it.
    """
    side_index, slot = key
    mon = pos.sides[side_index].pokemon[pos.sides[side_index].active[slot]]
    if mon is None:
        raise BeliefError(f"no active Pokemon at {key}")
    live: dict[str, str] = {"hp": "残 HP と割合ダメージに常に効く"}

    # Its own moves: whatever each damaging move attacks with. Foul Play is the exception
    # -- it attacks with the *target's* Attack, so it says nothing about its user's.
    for move_slot in mon.moves:
        move = reg.moves.get(move_slot.id)
        if move is None:
            continue
        if move.category != "Status" and not move.raw.get("overrideOffensivePokemon"):
            offensive = str(move.raw.get("overrideOffensiveStat") or "") or CATEGORY_STAT.get(
                move.category
            )
            if offensive:
                live.setdefault(offensive, f"自分の {{{move.id}}} で使う")
        for stat in STAT_READING_MOVES.get(move_slot.id, ()):
            live.setdefault(stat, f"自分の {move.name} が生の値を読む")

    # Everything on the other side that could hit it, bench included: a switch-in can
    # attack on the same turn.
    for foe in pos.sides[1 - side_index].pokemon:
        if foe.fainted:
            continue
        for move_slot in foe.moves:
            move = reg.moves.get(move_slot.id)
            if move is None or move.category == "Status":
                continue
            defensive = str(move.raw.get("overrideDefensiveStat") or "") or CATEGORY_DEFENCE.get(
                move.category
            )
            if defensive:
                live.setdefault(defensive, f"相手の {{{move.id}}} を受けるのに使う")
            if move.raw.get("overrideOffensivePokemon") == "target":
                # Foul Play: our Attack is what hits us.
                offensive = str(
                    move.raw.get("overrideOffensiveStat") or ""
                ) or CATEGORY_STAT.get(move.category)
                if offensive:
                    live.setdefault(offensive, f"相手の {move.name} が自分の値で殴る")
            for stat in STAT_READING_MOVES.get(move_slot.id, ()):
                live.setdefault(stat, f"相手の {move.name} が生の値を読む")
        for stat in STAT_READING_ABILITIES.get(foe.ability, ()):
            live.setdefault(stat, f"相手の特性 {foe.ability} が生の値を読む")

    order = [s for s in STAT_IDS if s in live and s != "spe"]
    return tuple(order), live


def unmodelled_stat_readers(reg: Regulation) -> tuple[str, ...]:
    """Moves in this regulation that read a stat in a way neither source covers.

    The reduction is only lossless if every stat-reading move is accounted for, either by
    a declarative override in the dump or by :data:`STAT_READING_MOVES`. A move with a
    `basePowerCallback` that nothing here mentions might read one, so it is listed and
    the test asserts the list is empty -- a new regulation that introduces one fails
    loudly instead of quietly merging particles it should not.
    """
    from .moveinfo import HANDLED_VARIABLE_BP, KNOWN_UNIMPLEMENTED_VARIABLE_BP

    suspicious: list[str] = []
    for move in reg.moves.values():
        if move.id in STAT_READING_MOVES:
            continue
        if move.raw.get("overrideOffensiveStat") or move.raw.get("overrideDefensiveStat"):
            continue
        if move.raw.get("overrideOffensivePokemon"):
            continue
        if "basePowerCallback" not in move.custom_hooks:
            continue
        # A variable base power the calculator models, or one it already reports as
        # unmodelled, is not a hidden stat read: in both cases the value is accounted for.
        if move.id in HANDLED_VARIABLE_BP or move.id in KNOWN_UNIMPLEMENTED_VARIABLE_BP:
            continue
        suspicious.append(move.id)
    return tuple(sorted(suspicious))


def speed_thresholds(
    reg: Regulation,
    pos: Position,
    beliefs: dict[tuple[int, int], SpreadBelief],
    *,
    exclude: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    """Every effective Speed that could appear on the field, as comparison boundaries.

    A particle's Speed matters only through the sign of its comparison against these, and
    equality is kept as its own case because a Speed tie is a coin flip rather than an
    order. Collecting the boundaries from *all* particles of the other hidden Pokemon --
    not just its most likely one -- is what makes the bucketing lossless when both
    opposing Pokemon are unknown at once.
    """
    from .view import battler

    field = field_state(pos, reg)
    values: set[int] = set()
    for side_index, side in enumerate(pos.sides):
        conditions = frozenset(c.id for c in side.side_conditions)
        for slot, mon in enumerate(side.active_pokemon()):
            if mon is None or mon.fainted or (side_index, slot) == exclude:
                continue
            key = (side_index, slot)
            belief = beliefs.get(key)
            view = (
                _belief_battler(reg, mon, belief)
                if belief is not None
                else battler(reg, mon)
            )
            values.update(int(v) for v in effective_speed(reg, view, field, conditions))
    return tuple(sorted(values))


def _hp_values(reg: Regulation, mon: Pokemon, maxhp: np.ndarray) -> np.ndarray:
    """Exact HP per particle, consistent with the percentage the game displayed.

    The observation is read back out of the position: its HP was chosen from the band in
    the first place, so re-displaying it recovers the percentage and the bar colour. Doing
    it this way keeps the belief layer dependent on a Position rather than on the scenario
    file the Position came from.
    """
    if mon.maxhp <= 0 or mon.hp <= 0:
        return np.zeros(maxhp.shape[0], dtype=np.int64)
    floor_rule = uses_floor_display(reg)
    percent = displayed_percent(mon.hp, mon.maxhp, floor_rule=floor_rule)
    colour = displayed_colour(mon.hp, mon.maxhp) if floor_rule else None
    out = np.empty(maxhp.shape[0], dtype=np.int64)
    for i, value in enumerate(maxhp):
        out[i] = band(
            percent, int(value), colour=colour, floor_rule=floor_rule
        ).low
    return out


def _belief_battler(
    reg: Regulation, mon: Pokemon, belief: SpreadBelief
) -> Battler:
    """A Battler carrying every particle, each with its own consistent current HP."""
    from .view import battler

    maxhp = belief.stats[:, 0]
    return battler(
        reg, mon, sp=belief.spreads, hp_values=_hp_values(reg, mon, maxhp)
    )


def reduce_for_position(
    reg: Regulation,
    pos: Position,
    key: tuple[int, int],
    beliefs: dict[tuple[int, int], SpreadBelief],
) -> Reduction:
    """Merges the particles this position cannot tell apart.

    Every member of a class produces the same resolved turn, so the merged set is a
    substitute for the full one rather than an approximation of it. The class's weight is
    the sum of its members' and its representative is the heaviest member, which matters
    only for the Speed value carried forward -- and that value is inside a bucket where
    every member compares identically against every Speed on the field.
    """
    belief = beliefs[key]
    side_index, slot = key
    mon = pos.sides[side_index].pokemon[pos.sides[side_index].active[slot]]
    if mon is None:
        raise BeliefError(f"no active Pokemon at {key}")

    live, reasons = live_stats_for(reg, pos, key)
    thresholds = speed_thresholds(reg, pos, beliefs, exclude=key)
    # Speed collapses to comparisons unless something on the field reads the number
    # itself, which Electro Ball and Gyro Ball do.
    spe_is_raw = any(
        "spe" in STAT_READING_MOVES.get(move_slot.id, ())
        for side in pos.sides
        for m in side.pokemon
        if not m.fainted
        for move_slot in m.moves
    )


    field = field_state(pos, reg)
    conditions = frozenset(c.id for c in pos.sides[side_index].side_conditions)
    view = _belief_battler(reg, mon, belief)
    speeds = effective_speed(reg, view, field, conditions).astype(np.int64)

    indices = [STAT_IDS.index(s) for s in live]
    grouped: dict[tuple[int, ...], list[int]] = {}
    for i in range(belief.size):
        stat_key = tuple(int(belief.stats[i, j]) for j in indices)
        if spe_is_raw:
            speed_key: tuple[int, ...] = (int(speeds[i]),)
        else:
            speed_key = tuple(int(np.sign(speeds[i] - t)) for t in thresholds)
        grouped.setdefault(stat_key + speed_key, []).append(i)

    members = tuple(tuple(v) for v in grouped.values())
    representatives = [max(group, key=lambda i: belief.weights[i]) for group in members]
    weights = np.array(
        [float(belief.weights[list(group)].sum()) for group in members], dtype=np.float64
    )
    reduced = SpreadBelief(
        species_id=belief.species_id,
        nature=belief.nature,
        spreads=belief.spreads[representatives],
        stats=belief.stats[representatives],
        weights=weights / weights.sum(),
        provenance=belief.provenance + "; reduced losslessly for this position",
    )
    reasons.setdefault(
        "spe",
        "生の値を読む技がある" if spe_is_raw else "行動順の比較にのみ効く（境界で同値化）",
    )
    return Reduction(
        belief=reduced,
        live_stats=(*live, "spe"),
        reasons=reasons,
        speed_thresholds=() if spe_is_raw else thresholds,
        original_size=belief.size,
        members=members,
        original=belief,
    )


def battler_for(
    reg: Regulation, pos: Position, key: tuple[int, int], belief: SpreadBelief
) -> Battler:
    """A Battler carrying every particle of a belief on its leading axis."""

    side_index, slot = key
    mon = pos.sides[side_index].pokemon[pos.sides[side_index].active[slot]]
    if mon is None:
        raise BeliefError(f"no active Pokemon at {key}")
    return _belief_battler(reg, mon, belief)


def faster_probability(
    reg: Regulation,
    pos: Position,
    key: tuple[int, int],
    belief: SpreadBelief,
    against: tuple[int, int],
) -> tuple[float, float]:
    """(P(faster), P(tie)) for a hidden Pokemon against a known one.

    This is the belief made decision-relevant: the raw marginal over Speed SP is hard to
    act on, whereas "62% chance it moves first" is the question the position asks.
    """
    from .view import battler

    field = field_state(pos, reg)
    side_index, slot = key
    mine = pos.sides[against[0]].pokemon[pos.sides[against[0]].active[against[1]]]
    theirs = pos.sides[side_index].pokemon[pos.sides[side_index].active[slot]]
    if mine is None or theirs is None:
        raise BeliefError("both slots must be occupied")

    my_speed = int(
        effective_speed(
            reg,
            battler(reg, mine),
            field,
            frozenset(c.id for c in pos.sides[against[0]].side_conditions),
        )[0]
    )
    their_speeds = effective_speed(
        reg,
        _belief_battler(reg, theirs, belief),
        field,
        frozenset(c.id for c in pos.sides[side_index].side_conditions),
    )
    faster = float(belief.weights[their_speeds > my_speed].sum())
    tie = float(belief.weights[their_speeds == my_speed].sum())
    return faster, tie


__all__ = [
    "STAT_READING_ABILITIES",
    "STAT_READING_MOVES",
    "BeliefError",
    "Reduction",
    "SpreadBelief",
    "battler_for",
    "build_belief",
    "faster_probability",
    "live_stats_for",
    "reduce_for_position",
    "speed_thresholds",
    "unmodelled_stat_readers",
]
