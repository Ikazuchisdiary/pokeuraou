"""Move-specific mechanics: variable base power, type changes, effectiveness overrides.

Showdown implements these as JavaScript callbacks (``basePowerCallback``, ``onModifyType``,
``onEffectiveness``), which the regulation dump cannot serialize -- it only records that
they exist. This module reimplements the ones that occur in the metagame, transcribed from
``vendor/pokemon-showdown/data/moves.ts`` at the pinned commit.

Anything not implemented here is *reported*, not guessed: :func:`base_power` returns
``None`` for a move whose base power it cannot determine, and the caller surfaces that as
an unmodelled effect rather than printing a number with no basis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .battler import Battler
from .effects import item_is_removable
from .regulation import Regulation

#: Weight thresholds shared by Low Kick and Grass Knot, in hectograms as Showdown stores
#: them (``weightkg`` here is kilograms, so the thresholds are divided by 10).
_WEIGHT_BP: tuple[tuple[float, int], ...] = (
    (200.0, 120),
    (100.0, 100),
    (50.0, 80),
    (25.0, 60),
    (10.0, 40),
)

#: Ratio thresholds for Heavy Slam / Heat Crash.
_RELATIVE_WEIGHT_BP: tuple[tuple[float, int], ...] = ((5.0, 120), (4.0, 100), (3.0, 80), (2.0, 60))

#: Weather that halves Solar Beam / Solar Blade.
_SOLAR_WEAK_WEATHER = frozenset(
    {"raindance", "primordialsea", "sandstorm", "hail", "snowscape", "snow"}
)

#: Weather Ball's type and the doubling that comes with it.
_WEATHER_BALL_TYPE: dict[str, str] = {
    "sunnyday": "Fire",
    "desolateland": "Fire",
    "raindance": "Water",
    "primordialsea": "Water",
    "sandstorm": "Rock",
    "hail": "Ice",
    "snowscape": "Ice",
    "snow": "Ice",
}

_TERRAIN_PULSE_TYPE: dict[str, str] = {
    "electricterrain": "Electric",
    "grassyterrain": "Grass",
    "mistyterrain": "Fairy",
    "psychicterrain": "Psychic",
}


@dataclass(slots=True)
class MoveContext:
    """Per-turn facts a variable-base-power move may need.

    These are things the position does not carry as a matter of course, so the resolver
    supplies them explicitly. A field left at its default is stated, not assumed: for
    instance ``side_total_fainted`` defaults to 0, so Last Respects computed without it is
    computed at its floor -- which is why :func:`base_power` reports rather than silently
    using the default when the value matters.
    """

    weather: str | None = None
    terrain: str | None = None
    #: Fainted count on the attacker's side, for Last Respects.
    side_total_fainted: int = 0
    #: Times the attacker has been hit, for Rage Fist.
    times_attacked: int = 0
    #: Whether the target already took damage this turn, for Assurance.
    target_hurt_this_turn: bool = False
    #: Whether the target damaged the attacker earlier this turn, for Avalanche.
    damaged_by_target: bool = False
    #: Whether the attacker's previous move failed, for Stomping Tantrum and Temper Flare.
    previous_move_failed: bool = False
    #: Which hit of a multi-hit move this is, counting from 1. Triple Axel and Triple Kick
    #: scale their power with it.
    hit_index: int = 1
    #: Whether the attacker moves after the target this turn, for Payback.
    moving_last: bool = False
    #: Whether an ally already used the same move this turn, for Round.
    ally_used_same_move: bool = False


#: Moves whose callback returns the declared base power unless a condition holds. The
#: condition comes from :class:`MoveContext`; with the context left at its defaults these
#: fall back to the declared value, which is the common case rather than a guess.
CONDITIONAL_DOUBLING_BP: dict[str, str] = {
    "stompingtantrum": "previous_move_failed",
    "temperflare": "previous_move_failed",
    "payback": "moving_last",
    "round": "ally_used_same_move",
}

#: Hex and friends double against a statused target, which the position always carries.
STATUS_DOUBLING_BP = frozenset({"hex", "infernalparade", "barbbarrage", "venoshock"})

#: Multi-hit moves whose power grows with the hit number: ``base * move.hit``.
ESCALATING_MULTIHIT_BP: dict[str, int] = {
    "tripleaxel": 20,
    "triplekick": 10,
}

#: Moves whose base power this module computes from state the resolver can supply.
IMPLEMENTED_VARIABLE_BP = frozenset(
    {
        "lowkick", "grassknot", "heavyslam", "heatcrash", "electroball", "gyroball",
        "lastrespects", "ragefist", "eruption", "waterspout", "acrobatics", "assurance",
        "avalanche", "storedpower", "powertrip", "weatherball", "terrainpulse",
        "risingvoltage", "expandingforce", "hardpress", "flail", "reversal",
    }
)

#: Every move whose base power this module can produce.
HANDLED_VARIABLE_BP = (
    IMPLEMENTED_VARIABLE_BP
    | frozenset(CONDITIONAL_DOUBLING_BP)
    | STATUS_DOUBLING_BP
    | frozenset(ESCALATING_MULTIHIT_BP)
)

#: Variable-base-power moves this module does NOT compute. Listed explicitly so the
#: coverage report can name them instead of them failing silently.
KNOWN_UNIMPLEMENTED_VARIABLE_BP = frozenset(
    {
        "beatup", "ficklebeam", "lashout", "mistyexplosion", "gravapple", "shellsidearm",
        "boltbeak", "fishiousrend", "punishment", "trumpcard", "wringout", "crushgrip",
        "spitup", "naturalgift", "fling", "brine", "retaliate", "smellingsalts",
        "wakeupslap", "watershuriken",
    }
)

#: Moves that deal a fraction of the target's current HP instead of rolling damage
#: (Showdown's ``damageCallback``). The value is the divisor.
FRACTIONAL_HP_MOVES: dict[str, int] = {
    "superfang": 2,
    "naturesmadness": 2,
    "ruination": 2,
}

#: Moves whose damage equals the user's remaining HP.
USER_HP_DAMAGE_MOVES = frozenset({"finalgambit"})

#: Moves that bring the target down to the user's HP (``damageCallback``).
ENDEAVOR_MOVES = frozenset({"endeavor"})

#: Damage-callback moves that depend on what hit the user this turn, which a single-turn
#: calculation cannot know on its own.
COUNTER_MOVES = frozenset({"counter", "mirrorcoat", "metalburst", "comeuppance"})


def fixed_damage(
    move_id: str, attacker: Battler, defender: Battler
) -> np.ndarray | None:
    """Damage for moves that bypass the damage formula, or None if not such a move.

    Showdown clamps each of these to at least 1.
    """
    divisor = FRACTIONAL_HP_MOVES.get(move_id)
    if divisor is not None:
        return np.maximum(defender.hp // divisor, 1)
    if move_id in USER_HP_DAMAGE_MOVES:
        return np.maximum(attacker.hp, 1)
    if move_id in ENDEAVOR_MOVES:
        return np.maximum(defender.hp - attacker.hp, 0)
    return None


def _weight_bp(weight_kg: float) -> int:
    for threshold, bp in _WEIGHT_BP:
        if weight_kg >= threshold:
            return bp
    return 20


def _relative_weight_bp(attacker_kg: float, target_kg: float) -> int:
    if target_kg <= 0:
        return 120
    for ratio, bp in _RELATIVE_WEIGHT_BP:
        if attacker_kg >= target_kg * ratio:
            return bp
    return 40


def effective_type(
    reg: Regulation, move_id: str, attacker: Battler, ctx: MoveContext
) -> str | None:
    """Move type after a move's own ``onModifyType``, or None if unchanged.

    Ability-driven type changes (the ``-ate`` abilities, Liquid Voice) are handled in
    ``damage.effective_move_type``; this covers the move's own rules.
    """
    del reg
    if move_id == "weatherball":
        return _WEATHER_BALL_TYPE.get(ctx.weather or "")
    if move_id == "terrainpulse":
        return _TERRAIN_PULSE_TYPE.get(ctx.terrain or "")
    if move_id == "aurawheel":
        # Morpeko-Hangry's form makes it Dark; the base form stays Electric.
        return "Dark" if attacker.species == "morpekohangry" else None
    if move_id == "ragingbull":
        # Tauros-Paldea's three breeds give Raging Bull three types.
        return {
            "taurospaldeacombat": "Fighting",
            "taurospaldeablaze": "Fire",
            "taurospaldeaaqua": "Water",
        }.get(attacker.species)
    return None


def effectiveness_override(
    reg: Regulation, move_id: str, move_type: str, defender_types: tuple[str, ...]
) -> tuple[float, int] | None:
    """A move's ``onEffectiveness``, as (multiplier, Showdown's typeMod).

    Freeze-Dry treats Water as super effective; Flying Press adds Flying's effectiveness
    on top of Fighting's. Both are transcribed rather than approximated because they swing
    damage by a factor of four.
    """
    if move_id == "freezedry":
        mult = 1.0
        steps = 0
        row = reg.typechart.get(move_type, {})
        for t in defender_types:
            if t == "Water":
                mult *= 2.0
                steps += 1
                continue
            m = row.get(t, 1.0)
            mult *= m
            steps += 1 if m == 2 else (-1 if m == 0.5 else 0)
        return mult, max(-6, min(6, steps))

    if move_id == "flyingpress":
        mult = 1.0
        steps = 0
        fighting = reg.typechart.get("Fighting", {})
        flying = reg.typechart.get("Flying", {})
        for t in defender_types:
            for row in (fighting, flying):
                m = row.get(t, 1.0)
                mult *= m
                steps += 1 if m == 2 else (-1 if m == 0.5 else 0)
        return mult, max(-6, min(6, steps))

    return None


def base_power(
    reg: Regulation,
    move_id: str,
    attacker: Battler,
    defender: Battler,
    ctx: MoveContext,
) -> np.ndarray | None:
    """Base power for one hit, or None when this module cannot determine it.

    Returns an array because some formulas depend on stats or HP, which vary across
    belief particles (Electro Ball compares Speeds; Eruption scales with the attacker's
    HP fraction).
    """
    move = reg.moves[move_id]
    n = max(attacker.n, defender.n)
    declared = move.base_power

    def const(value: float) -> np.ndarray:
        return np.full(n, int(value), dtype=np.int64)

    if move_id in ("lowkick", "grassknot"):
        species = reg.species[defender.species]
        return const(_weight_bp(species.weightkg))

    if move_id in ("heavyslam", "heatcrash"):
        return const(
            _relative_weight_bp(
                reg.species[attacker.species].weightkg, reg.species[defender.species].weightkg
            )
        )

    if move_id == "electroball":
        atk_spe = attacker.stat("spe")
        def_spe = np.maximum(defender.stat("spe"), 1)
        ratio = np.minimum(atk_spe // def_spe, 4)
        table = np.array([40, 60, 80, 120, 150], dtype=np.int64)
        return table[ratio]

    if move_id == "gyroball":
        atk_spe = np.maximum(attacker.stat("spe"), 1)
        power = (25 * defender.stat("spe")) // atk_spe + 1
        return np.minimum(power, 150)

    if move_id == "lastrespects":
        return const(50 + 50 * ctx.side_total_fainted)

    if move_id == "ragefist":
        return const(min(350, 50 + 50 * ctx.times_attacked))

    if move_id in ("eruption", "waterspout"):
        # trunc(declared * hp / maxhp), and Showdown clamps to at least 1.
        bp = np.trunc(declared * attacker.hp / np.maximum(attacker.maxhp, 1)).astype(np.int64)
        return np.maximum(bp, 1)

    if move_id == "hardpress":
        bp = np.trunc(100 * defender.hp / np.maximum(defender.maxhp, 1)).astype(np.int64)
        return np.maximum(bp, 1)

    if move_id in ("flail", "reversal"):
        ratio = np.trunc(48 * attacker.hp / np.maximum(attacker.maxhp, 1)).astype(np.int64)
        bp = np.select(
            [ratio < 2, ratio < 5, ratio < 10, ratio < 17, ratio < 33],
            [200, 150, 100, 80, 40],
            default=20,
        ).astype(np.int64)
        return bp

    if move_id == "acrobatics":
        return const(declared * 2 if attacker.item is None else declared)

    if move_id == "assurance":
        return const(declared * 2 if ctx.target_hurt_this_turn else declared)

    if move_id == "avalanche":
        return const(declared * 2 if ctx.damaged_by_target else declared)

    if move_id in ("storedpower", "powertrip"):
        positive = sum(v for v in attacker.boosts.values() if v > 0)
        return const(20 + 20 * positive)

    if move_id == "weatherball":
        return const(declared * 2 if ctx.weather in _WEATHER_BALL_TYPE else declared)

    if move_id == "terrainpulse":
        return const(declared * 2 if ctx.terrain in _TERRAIN_PULSE_TYPE else declared)

    if move_id == "risingvoltage":
        return const(declared * 2 if ctx.terrain == "electricterrain" else declared)

    if move_id == "expandingforce":
        return const(declared)

    escalating = ESCALATING_MULTIHIT_BP.get(move_id)
    if escalating is not None:
        return const(escalating * max(1, ctx.hit_index))

    condition = CONDITIONAL_DOUBLING_BP.get(move_id)
    if condition is not None:
        return const(declared * 2 if getattr(ctx, condition) else declared)

    if move_id in STATUS_DOUBLING_BP:
        statused = defender.status is not None
        if move_id == "venoshock":
            statused = defender.status in ("psn", "tox")
        return const(declared * 2 if statused else declared)

    if declared > 0:
        return const(declared)

    return None


def base_power_is_approximate(move_id: str) -> bool:
    """Whether the value :func:`base_power` returns for this move is a fallback.

    A move in :data:`KNOWN_UNIMPLEMENTED_VARIABLE_BP` still has a declared base power, so
    the calculation produces *a* number. Returning it without saying so is the silent
    wrong answer the reporting split exists to catch, so the caller flags it instead.
    """
    return move_id in KNOWN_UNIMPLEMENTED_VARIABLE_BP


def base_power_modifiers(
    reg: Regulation, move_id: str, attacker: Battler, defender: Battler, ctx: MoveContext
) -> tuple[tuple[str, float, float], ...]:
    """A move's own ``onBasePower`` chain entries, as (label, numerator, denominator)."""
    out: list[tuple[str, float, float]] = []

    if move_id == "facade" and attacker.status is not None and attacker.status != "slp":
        out.append(("facade", 2.0, 1.0))

    elif move_id == "knockoff" and item_is_removable(
        reg, defender.species, defender.item
    ):
        out.append(("knockoff", 1.5, 1.0))

    elif move_id in ("solarbeam", "solarblade") and ctx.weather in _SOLAR_WEAK_WEATHER:
        out.append((move_id, 0.5, 1.0))

    elif move_id == "expandingforce" and ctx.terrain == "psychicterrain":
        out.append(("expandingforce", 1.5, 1.0))

    return tuple(out)


#: Grassy Terrain's `weakenedMoves`, halved into a grounded target.
GRASSY_WEAKENED_MOVES = frozenset({"earthquake", "bulldoze", "magnitude"})


def terrain_modifiers(
    reg: Regulation,
    move_id: str,
    move_type: str,
    attacker_grounded: bool,
    terrain: str | None,
    defender_grounded: bool = True,
) -> tuple[tuple[str, float, float], ...]:
    """Terrain's ``onBasePower``: a grounded user's matching type is boosted; Misty Terrain's
    Dragon moves and Grassy Terrain's Earthquake, Bulldoze and Magnitude are halved into a
    grounded *target*, whatever the user stands on (IKA-201)."""
    del reg
    if terrain == "mistyterrain":
        return (("mistyterrain", 2048, 4096),) if move_type == "Dragon" and defender_grounded else ()
    if terrain == "grassyterrain" and move_id in GRASSY_WEAKENED_MOVES:
        return (("grassyterrain", 2048, 4096),) if defender_grounded else ()
    if terrain is None or not attacker_grounded:
        return ()
    if terrain == "electricterrain" and move_type == "Electric":
        return (("electricterrain", 5325, 4096),)
    if terrain == "grassyterrain" and move_type == "Grass":
        return (("grassyterrain", 5325, 4096),)
    if terrain == "psychicterrain" and move_type == "Psychic":
        return (("psychicterrain", 5325, 4096),)
    return ()
