"""Turning what happened last turn into a posterior over the opponent's spread.

The prior comes from usage statistics and is the same for everyone playing that species.
What makes an analysis of *this* battle worth anything is that the battle keeps handing
over evidence, and the evidence is exact: a damage number is a deterministic function of
the attacker's stats and one of sixteen known rolls, so an observed number does not
suggest a spread, it *rules out* the spreads that could not have produced it.

Four channels, one per hidden stat that matters:

- :class:`DamageTaken` -- our own Pokemon's HP is known exactly, so the damage it took is
  known exactly. Given a particle, the calculator produces sixteen candidate numbers; the
  likelihood is how many of them equal what happened. This is the strongest channel there
  is, and it is the "inverse damage calculation" the whole belief layer was designed
  around.
- :class:`DamageDealt` -- the other direction, seen through the percentage bar. Weaker,
  because the observation is a band rather than a number, but it is the only thing that
  speaks about their defences.
- :class:`MovedFirst` -- turn order at equal priority is a strict inequality on Speed, and
  a Speed tie is a coin flip we know the probability of. Cheap and sharp.
- :class:`FractionalLoss` -- residual damage is a fraction of max HP, so watching a burn
  or a sandstorm tick constrains max HP directly, without any dependence on a spread's
  offensive or defensive numbers.

Every likelihood here is a probability of the *observation*, computed from mechanics that
are already differential-tested against Showdown. Nothing is fitted, and a particle whose
weight goes to zero was genuinely impossible rather than merely unlikely -- which is why
:meth:`SpreadBelief.reweight` raises when everything dies: that means the observation or
the position was entered wrong, and renormalising nothing would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .battler import Battler
from .belief import SpreadBelief, battler_for
from .damage import calculate
from .effects import SURVIVE_AT_ONE_ABILITIES, SURVIVE_AT_ONE_ITEMS
from .hpdisplay import band_bounds, uses_floor_display
from .position import Position
from .regulation import Regulation
from .speed import move_priority
from .view import battler, field_state

#: (side index, active slot), as everywhere else.
SlotKey = tuple[int, int]

#: The widest band the update will enumerate. A 1% bar is consistent with almost any HP,
#: so it carries nearly no information and enumerating it would cost more than it earns.
MAX_BAND_WIDTH = 8


class ObservationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class DamageTaken:
    """Exact damage one of our Pokemon took from a hidden attacker.

    We see our own HP exactly, so this is a number rather than a band -- the sharpest
    evidence the game produces.
    """

    attacker: SlotKey
    defender: SlotKey
    move_id: str
    damage: int
    crit: bool = False
    spread: bool = False
    #: True when the number is a floor rather than the roll: the defender fainted, or a
    #: Sash capped the hit, so the protocol reports the HP that was there and the actual
    #: roll was at least that. Treating a KO as an equation throws away the spreads that
    #: would have overkilled -- which are exactly the heavy-investment ones.
    capped: bool = False

    def describe(self) -> str:
        relation = "以上のダメージ" if self.capped else " ダメージ"
        return (
            f"{self.move_id} が p{self.attacker[0] + 1} から "
            f"p{self.defender[0] + 1} に {self.damage}{relation}"
            + ("（急所）" if self.crit else "")
        )


@dataclass(frozen=True, slots=True)
class DamageDealt:
    """Damage we dealt to a hidden Pokemon, seen only as a change in its percentage."""

    attacker: SlotKey
    defender: SlotKey
    move_id: str
    percent_before: int
    percent_after: int
    crit: bool = False
    spread: bool = False
    colour_before: str | None = None
    colour_after: str | None = None

    def describe(self) -> str:
        return (
            f"{self.move_id} で p{self.defender[0] + 1} が "
            f"{self.percent_before}% → {self.percent_after}%"
        )


@dataclass(frozen=True, slots=True)
class MovedFirst:
    """One Pokemon acted before another. At equal priority this bounds Speed."""

    first: SlotKey
    second: SlotKey
    first_move: str | None = None
    second_move: str | None = None

    def describe(self) -> str:
        return (
            f"p{self.first[0] + 1}[{self.first[1]}] が "
            f"p{self.second[0] + 1}[{self.second[1]}] より先に動いた"
        )


@dataclass(frozen=True, slots=True)
class FractionalLoss:
    """A hidden Pokemon lost a known fraction of its max HP, seen as a percentage change.

    Residual damage -- burn, poison, sandstorm, Leech Seed, Leftovers healing with a
    negative numerator -- depends on nothing but max HP, so this pins the HP investment
    without touching any other stat.
    """

    slot: SlotKey
    numerator: int
    denominator: int
    percent_before: int
    percent_after: int
    reason: str = ""
    colour_before: str | None = None
    colour_after: str | None = None

    def describe(self) -> str:
        share = f"{self.numerator}/{self.denominator}"
        return (
            f"p{self.slot[0] + 1}[{self.slot[1]}] が最大HPの {share}"
            f"（{self.reason or 'residual'}）で "
            f"{self.percent_before}% → {self.percent_after}%"
        )


Observation = DamageTaken | DamageDealt | MovedFirst | FractionalLoss


@dataclass(slots=True)
class UpdateReport:
    """What each observation did to each belief, so the update can be read not trusted."""

    entries: list[tuple[str, SlotKey, int, int, float]]
    #: Observations that carried no information, and why.
    uninformative: list[str]
    #: Observations that could not be applied at all, and why.
    skipped: list[str]

    def render(self, reg: Regulation) -> str:
        del reg
        lines = ["■ 観測による事後更新"]
        if not self.entries:
            lines.append("  情報を持つ観測はありませんでした。")
        for label, key, before, after, bits in self.entries:
            lines.append(
                f"  p{key[0] + 1}[{key[1]}]  候補 {before} → {after}"
                f"（エントロピー {bits:+.2f} bit）  {label}"
            )
        if any(bits < 0 for _l, _k, _b, _a, bits in self.entries):
            lines.append(
                "  ※ エントロピーは減少量。負の値は「事前分布が集中していた側と観測が"
                "食い違った」ことを意味します（更新としては正しい挙動）"
            )
        for note in self.uninformative:
            lines.append(f"  （情報なし）{note}")
        for note in self.skipped:
            lines.append(f"  （適用不能）{note}")
        return "\n".join(lines)


def _entropy(weights: np.ndarray) -> float:
    """Shannon entropy in bits, the honest measure of how much a belief narrowed."""
    w = weights[weights > 0]
    return float(-(w * np.log2(w)).sum())


def _mon(pos: Position, key: SlotKey):  # noqa: ANN202
    side_index, slot = key
    party_index = pos.sides[side_index].active[slot]
    mon = pos.sides[side_index].pokemon[party_index]
    if mon is None:
        raise ObservationError(f"no active Pokemon at {key}")
    return mon


def _known_battler(reg: Regulation, pos: Position, key: SlotKey) -> Battler:
    return battler(reg, _mon(pos, key))


def _rolls(
    reg: Regulation,
    pos: Position,
    attacker: Battler,
    defender: Battler,
    move_id: str,
    *,
    defender_side: int,
    crit: bool,
    spread: bool,
) -> np.ndarray | None:
    """(N, 16) damage rolls, or None when the calculator cannot speak to this hit."""
    result = calculate(
        reg, attacker, defender, move_id, field_state(pos, reg),
        defender_side=defender_side, spread=spread, crit=crit,
    )
    if result.immune or result.unmodelled:
        return None
    rolls = np.asarray(result.rolls, dtype=np.int64)
    return rolls if rolls.size else None


def _survives_at_one(mon: Battler) -> bool:
    return mon.item in SURVIVE_AT_ONE_ITEMS or mon.ability in SURVIVE_AT_ONE_ABILITIES


def damage_taken_likelihood(
    reg: Regulation, pos: Position, belief: SpreadBelief, obs: DamageTaken
) -> np.ndarray | None:
    """P(this exact damage | particle), which is (matching rolls) / 16.

    The defender is ours, so its HP and max HP are known and the cap that Focus Sash and
    Sturdy apply is computable rather than assumed.
    """
    attacker = battler_for(reg, pos, obs.attacker, belief)
    defender = _known_battler(reg, pos, obs.defender)
    rolls = _rolls(
        reg, pos, attacker, defender, obs.move_id,
        defender_side=obs.defender[0], crit=obs.crit, spread=obs.spread,
    )
    if rolls is None:
        return None

    if obs.capped:
        # The observation is "at least this much", so every roll that reaches it counts.
        return (rolls >= obs.damage).mean(axis=1)
    dealt = np.minimum(rolls, defender.hp[0])
    if _survives_at_one(defender) and defender.hp[0] >= defender.maxhp[0]:
        dealt = np.minimum(dealt, max(int(defender.hp[0]) - 1, 0))
    return (dealt == obs.damage).mean(axis=1)


def damage_dealt_likelihood(
    reg: Regulation, pos: Position, belief: SpreadBelief, obs: DamageDealt
) -> np.ndarray | None:
    """P(the bar went from one percentage to the other | particle).

    Weaker than :func:`damage_taken_likelihood` because both ends are bands rather than
    numbers, but it is the only channel that says anything about their defences. The
    starting HP is unknown within its band, and nothing distinguishes the values inside
    one, so they are averaged over.
    """
    attacker = _known_battler(reg, pos, obs.attacker)
    defender = battler_for(reg, pos, obs.defender, belief)
    rolls = _rolls(
        reg, pos, attacker, defender, obs.move_id,
        defender_side=obs.defender[0], crit=obs.crit, spread=obs.spread,
    )
    if rolls is None:
        return None

    maxhp = np.asarray(defender.maxhp, dtype=np.int64)
    floor_rule = uses_floor_display(reg)
    before_low, before_high = band_bounds(
        obs.percent_before, maxhp, colour=obs.colour_before
    )
    width = np.maximum(before_high - before_low + 1, 0)
    if int(width.max(initial=0)) > MAX_BAND_WIDTH:
        return None

    hits = np.zeros(maxhp.shape[0], dtype=np.float64)
    trials = np.zeros(maxhp.shape[0], dtype=np.float64)
    survives = _survives_at_one(defender)
    for offset in range(int(width.max(initial=0))):
        live = offset < width
        if not live.any():
            break
        hp0 = before_low + offset
        dealt = np.minimum(rolls, hp0[:, None])
        if survives:
            at_full = (hp0 >= maxhp)[:, None]
            dealt = np.where(at_full, np.minimum(dealt, (hp0 - 1)[:, None]), dealt)
        remaining = hp0[:, None] - dealt
        if obs.percent_after <= 0:
            consistent = remaining <= 0
        else:
            after_low, after_high = band_bounds(
                obs.percent_after, maxhp, colour=obs.colour_after
            )
            consistent = (remaining >= after_low[:, None]) & (
                remaining <= after_high[:, None]
            )
        hits += live * consistent.sum(axis=1)
        trials += live * rolls.shape[1]
    del floor_rule
    return np.divide(hits, trials, out=np.zeros_like(hits), where=trials > 0)


def moved_first_likelihood(
    reg: Regulation, pos: Position, belief: SpreadBelief, obs: MovedFirst, hidden: SlotKey
) -> np.ndarray | None:
    """P(this order | particle): 1 if faster, 1/2 on a tie, 0 if slower.

    Only the priority-equal case says anything, and Trick Room inverts the comparison.
    Whichever of the two slots is the hidden one decides which side of the inequality the
    particle sits on.
    """
    from .speed import effective_speed

    field = field_state(pos, reg)

    def priority_of(key: SlotKey, move_id: str | None) -> int:
        if move_id is None:
            # A switch, or a move we could not identify: assume the ordinary bracket
            # rather than inventing one. Speed still decided it.
            return 0
        return move_priority(reg, move_id, _known_battler(reg, pos, key), field)

    if priority_of(obs.first, obs.first_move) != priority_of(obs.second, obs.second_move):
        # Different brackets: the order was decided by priority and says nothing at all
        # about Speed. Reweighting on it would manufacture information.
        return None

    def speeds(key: SlotKey) -> np.ndarray:
        mon = _mon(pos, key)
        conditions = frozenset(c.id for c in pos.sides[key[0]].side_conditions)
        view = (
            battler_for(reg, pos, key, belief)
            if key == hidden
            else battler(reg, mon)
        )
        return np.asarray(
            effective_speed(reg, view, field, conditions), dtype=np.int64
        ).reshape(-1)

    first_speed = speeds(obs.first)
    second_speed = speeds(obs.second)
    faster = first_speed > second_speed
    if pos.field.trick_room:
        faster = first_speed < second_speed
    tie = first_speed == second_speed
    return np.where(faster, 1.0, np.where(tie, 0.5, 0.0)).astype(np.float64)


def fractional_loss_likelihood(
    reg: Regulation, belief: SpreadBelief, obs: FractionalLoss
) -> np.ndarray | None:
    """P(both percentages | particle), for a loss that is a known fraction of max HP.

    Zero or one per particle: the amount follows from max HP alone, so either the two
    bars are reproducible or that max HP is ruled out. This is the cheapest strong
    evidence about the HP investment in the game.
    """
    maxhp = np.asarray(belief.stats[:, 0], dtype=np.int64)
    amount = np.maximum(maxhp * obs.numerator // obs.denominator, 1)
    before_low, before_high = band_bounds(
        obs.percent_before, maxhp, colour=obs.colour_before
    )
    width = np.maximum(before_high - before_low + 1, 0)
    if int(width.max(initial=0)) > MAX_BAND_WIDTH:
        return None
    if obs.percent_after > 0:
        after_low, after_high = band_bounds(
            obs.percent_after, maxhp, colour=obs.colour_after
        )
    else:
        after_low = after_high = np.zeros_like(maxhp)

    hits = np.zeros(maxhp.shape[0], dtype=np.float64)
    trials = np.zeros(maxhp.shape[0], dtype=np.float64)
    for offset in range(int(width.max(initial=0))):
        live = offset < width
        if not live.any():
            break
        hp0 = before_low + offset
        remaining = np.maximum(hp0 - amount, 0)
        if obs.percent_after <= 0:
            consistent = remaining <= 0
        else:
            consistent = (remaining >= after_low) & (remaining <= after_high)
        hits += live * consistent
        trials += live
    return np.divide(hits, trials, out=np.zeros_like(hits), where=trials > 0)


def _target_slots(obs: Observation) -> tuple[SlotKey, ...]:
    if isinstance(obs, DamageTaken):
        return (obs.attacker,)
    if isinstance(obs, DamageDealt):
        return (obs.defender,)
    if isinstance(obs, FractionalLoss):
        return (obs.slot,)
    return (obs.first, obs.second)


def likelihood_for(
    reg: Regulation,
    pos: Position,
    beliefs: dict[SlotKey, SpreadBelief],
    obs: Observation,
) -> dict[SlotKey, np.ndarray]:
    """Per-slot likelihoods this observation implies. Empty when it says nothing."""
    hidden = [key for key in _target_slots(obs) if key in beliefs]
    if not hidden:
        return {}
    if isinstance(obs, MovedFirst) and len(hidden) == 2:
        # Two hidden Pokemon compared against each other constrains the *pair*, and the
        # constraint does not factorise into two independent reweightings. Applying it to
        # each separately would double-count it, so it is reported instead.
        raise ObservationError(
            "both Pokemon in this order comparison are hidden; the constraint is joint "
            "and cannot be applied to each belief independently"
        )

    out: dict[SlotKey, np.ndarray] = {}
    for key in hidden:
        belief = beliefs[key]
        if isinstance(obs, DamageTaken):
            values = damage_taken_likelihood(reg, pos, belief, obs)
        elif isinstance(obs, DamageDealt):
            values = damage_dealt_likelihood(reg, pos, belief, obs)
        elif isinstance(obs, FractionalLoss):
            values = fractional_loss_likelihood(reg, belief, obs)
        else:
            values = moved_first_likelihood(reg, pos, belief, obs, key)
        if values is not None:
            out[key] = values
    return out


def update(
    reg: Regulation,
    pos: Position,
    beliefs: dict[SlotKey, SpreadBelief],
    observations: list[Observation],
) -> tuple[dict[SlotKey, SpreadBelief], UpdateReport]:
    """Applies every observation in order, reporting what each one bought.

    Observations are treated as conditionally independent given the spread, which is what
    lets them multiply. That is exact for the channels here: each one is a different
    mechanic reading a different part of the same fixed stat vector, and the randomness
    each consumes (a damage roll, a speed tie) is drawn afresh.
    """
    current = dict(beliefs)
    report = UpdateReport(entries=[], uninformative=[], skipped=[])

    for obs in observations:
        try:
            per_slot = likelihood_for(reg, pos, current, obs)
        except ObservationError as exc:
            report.skipped.append(f"{obs.describe()}: {exc}")
            continue
        if not per_slot:
            report.uninformative.append(
                f"{obs.describe()}: 隠れた個体に触れないか、"
                "計算器が扱えない（優先度が違う・威力不定など）"
            )
            continue
        for key, values in per_slot.items():
            belief = current[key]
            before = int((belief.weights > 0).sum())
            entropy_before = _entropy(belief.weights)
            if float((belief.weights * values).sum()) <= 0:
                report.skipped.append(
                    f"{obs.describe()}: どのパーティクルでも確率 0。"
                    "観測か局面の入力が誤っています（無理に正規化しません）"
                )
                continue
            posterior = belief.reweight(values)
            after = int((posterior.weights > 0).sum())
            gained = entropy_before - _entropy(posterior.weights)
            current[key] = replace(
                posterior,
                provenance=posterior.provenance + f"; {obs.describe()}",
            )
            report.entries.append((obs.describe(), key, before, after, gained))
    return current, report


def _slot(raw: object, field: str) -> SlotKey:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ObservationError(
            f"{field} must be [side, slot] with side 0 = ours, got {raw!r}"
        )
    side, slot = int(raw[0]), int(raw[1])
    if side not in (0, 1) or slot not in (0, 1):
        raise ObservationError(f"{field} {raw!r} is not a doubles slot")
    return side, slot


def _percent(raw: object, field: str) -> tuple[int, str | None]:
    """A displayed percentage, optionally carrying the bar colour ("50%g")."""
    if isinstance(raw, (int, float)):
        return int(raw), None
    if not isinstance(raw, str):
        raise ObservationError(f"{field} must be a percentage, got {raw!r}")
    text = raw.strip()
    colour: str | None = None
    if text and text[-1].isalpha():
        colour, text = text[-1], text[:-1]
    text = text.rstrip("%").strip()
    try:
        return int(round(float(text))), colour
    except ValueError as exc:
        raise ObservationError(f"cannot read {field} {raw!r}") from exc


def parse_observations(reg: Regulation, raw: list[dict]) -> list[Observation]:
    """Reads the scenario file's `observations` list.

    Strict about move names and slots for the same reason the rest of the scenario parser
    is: an observation entered wrong does not fail loudly, it quietly reweights the belief
    towards spreads that were never there.
    """
    from .regulation import to_id

    out: list[Observation] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ObservationError(f"observation {index} is not an object")
        kind = str(entry.get("type", "")).strip()

        def move_of(key: str = "move", *, entry: dict = entry, index: int = index) -> str:
            name = entry.get(key)
            if not name:
                raise ObservationError(f"observation {index} needs a \"{key}\"")
            move_id = to_id(str(name))
            if move_id not in reg.moves:
                raise ObservationError(
                    f"observation {index}: {name!r} is not legal in "
                    f"{reg.meta.format_id}"
                )
            return move_id

        if kind == "damageTaken":
            out.append(
                DamageTaken(
                    attacker=_slot(entry.get("attacker"), "attacker"),
                    defender=_slot(entry.get("defender"), "defender"),
                    move_id=move_of(),
                    damage=int(entry["damage"]),
                    crit=bool(entry.get("crit")),
                    spread=bool(entry.get("spread")),
                    capped=bool(entry.get("capped") or entry.get("fainted")),
                )
            )
        elif kind == "damageDealt":
            before, colour_before = _percent(entry.get("before"), "before")
            after, colour_after = _percent(entry.get("after"), "after")
            out.append(
                DamageDealt(
                    attacker=_slot(entry.get("attacker"), "attacker"),
                    defender=_slot(entry.get("defender"), "defender"),
                    move_id=move_of(),
                    percent_before=before,
                    percent_after=after,
                    crit=bool(entry.get("crit")),
                    spread=bool(entry.get("spread")),
                    colour_before=colour_before,
                    colour_after=colour_after,
                )
            )
        elif kind == "movedFirst":
            out.append(
                MovedFirst(
                    first=_slot(entry.get("first"), "first"),
                    second=_slot(entry.get("second"), "second"),
                    first_move=move_of("firstMove") if entry.get("firstMove") else None,
                    second_move=move_of("secondMove") if entry.get("secondMove") else None,
                )
            )
        elif kind == "fractionalLoss":
            fraction = entry.get("fraction")
            if not isinstance(fraction, (list, tuple)) or len(fraction) != 2:
                raise ObservationError(
                    f"observation {index} needs \"fraction\": [numerator, denominator]"
                )
            before, colour_before = _percent(entry.get("before"), "before")
            after, colour_after = _percent(entry.get("after"), "after")
            out.append(
                FractionalLoss(
                    slot=_slot(entry.get("slot"), "slot"),
                    numerator=int(fraction[0]),
                    denominator=int(fraction[1]),
                    percent_before=before,
                    percent_after=after,
                    reason=str(entry.get("reason", "")),
                    colour_before=colour_before,
                    colour_after=colour_after,
                )
            )
        else:
            raise ObservationError(
                f"observation {index}: unknown type {kind!r}; expected damageTaken, "
                "damageDealt, movedFirst or fractionalLoss"
            )
    return out


__all__ = [
    "MAX_BAND_WIDTH",
    "DamageDealt",
    "DamageTaken",
    "FractionalLoss",
    "MovedFirst",
    "Observation",
    "ObservationError",
    "UpdateReport",
    "damage_dealt_likelihood",
    "damage_taken_likelihood",
    "fractional_loss_likelihood",
    "likelihood_for",
    "moved_first_likelihood",
    "parse_observations",
    "update",
]
