"""Damage calculation, vectorised over belief particles.

Structure of the calculation follows ``modifyDamage`` in
``data/mods/champions/scripts.ts`` step for step, because the modifier order and the
fixed-point rounding at each step both matter: a misplaced modifier shifts damage by 1,
which is enough to move a KO threshold and therefore to change the reported equilibrium.

    baseDamage = trunc(trunc(trunc(trunc(2 * L / 5 + 2) * BP * Atk) / Def) / 50)
    baseDamage += 2
    if spread:       modify(0.75)
    weather modifier
    if crit:         trunc(baseDamage * 1.5)
    random roll:     trunc(trunc(baseDamage * (100 - r)) / 100)   for r in 0..15
    STAB:            modify(1.5), or 2.0 with Adaptability
    type mod:        x2 per step up, trunc(/2) per step down
    burn:            modify(0.5) for physical, unless Guts
    ModifyDamage:    Life Orb, screens, Multiscale, resist berries, ...
    at least 1
    trunc(damage, 16)

The vectorisation split is deliberate: the *structure* (which attacker, which move, which
target) is looped over in Python, while randomness and belief are array axes. All 16
damage rolls come back at once, for every particle, from one call.
"""

from __future__ import annotations

import numpy as np

from .battler import N_ROLLS, Battler, DamageResult, FieldState
from .effects import (
    ABILITY_MODIFIERS,
    AURA_ABILITIES,
    AURA_BREAK_ABILITY,
    AURA_BROKEN_FP,
    AURA_FP,
    GHOST_PIERCING_ABILITIES,
    ITEM_MODIFIERS,
    MOLD_BREAKER_ABILITIES,
    RESIST_BERRIES,
    RETYPING_ABILITIES,
    SCREEN_CONDITIONS,
    SCREEN_FP_DOUBLES,
    SCREEN_FP_SINGLES,
    SURVIVE_AT_ONE_ABILITIES,
    SURVIVE_AT_ONE_ITEMS,
    TYPE_BOOST_ITEMS,
    TYPE_CHANGE_BOOST_FP,
    TYPE_CHANGING_ABILITIES,
    TYPE_IMMUNITY_ABILITIES,
    WEATHER_SUPPRESSING_ABILITIES,
    Ctx,
    Modifier,
    Slot,
    type_boost_item_fp,
)
from .fixedpoint import Chain, modify, trunc, trunc16
from .moveinfo import (
    MoveContext,
    base_power_is_approximate,
    base_power_modifiers,
    effectiveness_override,
    terrain_modifiers,
)
from .moveinfo import base_power as move_base_power
from .moveinfo import effective_type as move_effective_type
from .moveinfo import fixed_damage as move_fixed_damage
from .regulation import Regulation

#: Mega stone ids seen so far, filled in from the regulation on first use so that
#: unmodelled-effect reporting does not flag them.
_MEGA_STONE_IDS: frozenset[str] = frozenset()


def register_mega_stones(reg: Regulation) -> None:
    """Tells the unmodelled-effect reporter which items are mega stones."""
    global _MEGA_STONE_IDS
    _MEGA_STONE_IDS = frozenset(reg.mega_map)


def effective_move_type(reg: Regulation, move_id: str, attacker: Battler) -> tuple[str, int]:
    """Move type after type-changing abilities, plus the boost they chain.

    Returns (type, boostFp) where boostFp is 4096 when nothing changed.
    """
    move = reg.moves[move_id]
    changed = TYPE_CHANGING_ABILITIES.get(attacker.ability)
    if changed is None or move.category == "Status":
        return move.type, 4096
    if attacker.ability == "liquidvoice":
        if "sound" not in move.flags:
            return move.type, 4096
        # Liquid Voice changes the type but adds no boost.
        return changed, 4096
    if attacker.ability == "normalize":
        return changed, TYPE_CHANGE_BOOST_FP
    # The -ate abilities only convert Normal-type moves.
    if move.type != "Normal":
        return move.type, 4096
    return changed, TYPE_CHANGE_BOOST_FP


def type_effectiveness(reg: Regulation, move_type: str, defender: Battler) -> tuple[float, int]:
    """(multiplier, Showdown's clamped typeMod)."""
    mult = reg.type_effectiveness(move_type, defender.types)
    if mult == 0:
        return 0.0, 0
    # Showdown accumulates +1/-1 per type and clamps to -6..6.
    steps = 0
    row = reg.typechart.get(move_type, {})
    for t in defender.types:
        m = row.get(t, 1.0)
        if m == 2:
            steps += 1
        elif m == 0.5:
            steps -= 1
    return mult, max(-6, min(6, steps))


def is_immune(
    reg: Regulation,
    move_id: str,
    move_type: str,
    defender: Battler,
    attacker_ability: str = "",
    *,
    attacker_is_defender: bool = False,
) -> bool:
    """Type immunity, ability immunity and the special cases that override them.

    ``attacker_is_defender`` matters for Soundproof, whose handler is guarded by
    ``target !== source``: it blocks a sound move aimed at its holder by someone else, but
    not one its holder aims at itself. Clangorous Soul is a self-targeting sound move, so
    blocking it would turn a Soundproof Pokemon's own boost into nothing.
    """
    move = reg.moves[move_id]
    if bool(move.raw.get("ignoreImmunity")):
        ignore = move.raw["ignoreImmunity"]
        if ignore is True or (isinstance(ignore, dict) and ignore.get(move_type)):
            return False
    mult = reg.type_effectiveness(move_type, defender.types)
    if mult == 0:
        # Scrappy and Mind's Eye let Normal and Fighting through a Ghost type, and only
        # that immunity.
        pierces_ghost = (
            attacker_ability in GHOST_PIERCING_ABILITIES
            and move_type in ("Normal", "Fighting")
            and "Ghost" in defender.types
        )
        return not pierces_ghost
    if TYPE_IMMUNITY_ABILITIES.get(defender.ability) == move_type:
        return True
    # Air Balloon keeps its holder off the ground until it pops.
    if defender.item == "airballoon" and move_type == "Ground":
        return True
    if defender.ability == "wonderguard" and mult <= 1:
        return True
    if defender.ability == "bulletproof" and "bullet" in move.flags:
        return True
    if defender.ability == "soundproof" and "sound" in move.flags and not attacker_is_defender:
        return True
    if defender.ability == "overcoat" and "powder" in move.flags:
        return True
    return "Grass" in defender.types and "powder" in move.flags


def _collect(
    slot: Slot, ctx: Ctx, attacker: Battler, defender: Battler, field_state: FieldState
) -> Chain:
    """Builds one event's modifier chain, in Showdown's handler order."""
    mods: list[Modifier] = []

    def consider(source_ability: str, from_defender: bool) -> None:
        for mod in ABILITY_MODIFIERS.get(source_ability, ()):
            if mod.slot != slot or mod.from_defender != from_defender:
                continue
            # An aura reaches the whole field, so it is collected once below rather than
            # here. Letting it through as well applied it twice whenever the holder was
            # the attacker.
            if mod.id in AURA_ABILITIES:
                continue
            if mod.when(ctx):
                mods.append(mod)

    consider(attacker.ability, from_defender=False)
    # A defender's damage-reducing ability is ignored by Mold Breaker and friends. Only
    # abilities Showdown flags `breakable` are affected; the ones modelled here that
    # reduce damage all carry that flag.
    if attacker.ability not in MOLD_BREAKER_ABILITIES:
        consider(defender.ability, from_defender=True)

    for item_id, from_defender in ((attacker.item, False), (defender.item, True)):
        if not item_id:
            continue
        for mod in ITEM_MODIFIERS.get(item_id, ()):
            if mod.slot != slot or mod.from_defender != from_defender:
                continue
            if mod.when(ctx):
                mods.append(mod)

    # Aura abilities apply from anywhere on the field, on either side -- but only once,
    # however many holders there are: Showdown's first holder claims the move via
    # `move.auraBooster` and the rest return without modifying.
    aura_fp: int | None = None
    aura_label = ""
    if slot == "base_power":
        broken = AURA_BREAK_ABILITY in ctx.field_abilities
        for ability, move_type in AURA_ABILITIES.items():
            if ability in ctx.field_abilities and ctx.move_type == move_type:
                aura_fp = AURA_BROKEN_FP if broken else AURA_FP
                aura_label = f"{ability}{'+aurabreak' if broken else ''}"
                break

    mods.sort(key=lambda m: -m.priority)

    chain = Chain()
    for mod in mods:
        chain.add(mod.numerator, mod.denominator, label=mod.id)
    if aura_fp is not None:
        # Priority 20 in Showdown, which is ahead of every other base-power handler
        # modelled here, so it goes on first.
        chain.add_fp(aura_fp, label=aura_label)

    # Slot-specific effects that are not ability/item table entries.
    if slot == "base_power":
        if attacker.item and TYPE_BOOST_ITEMS.get(attacker.item) == ctx.move_type:
            chain.add_fp(type_boost_item_fp(), label=attacker.item)
        if "helpinghand" in attacker.volatiles:
            chain.add(1.5, label="helpinghand")
        if "charge" in attacker.volatiles and ctx.move_type == "Electric":
            chain.add(2.0, label="charge")

    elif slot == "damage":
        # Screens. Showdown: chainModify([2732, 4096]) in doubles, 0.5 in singles, and
        # only against non-crit hits of the matching category.
        if not ctx.is_crit and "infiltrates" not in ctx.move_flags:
            screen_fp = (
                SCREEN_FP_DOUBLES if field_state.active_per_half > 1 else SCREEN_FP_SINGLES
            )
            for cond, category in SCREEN_CONDITIONS.items():
                if cond not in ctx.defender_side_conditions:
                    continue
                if category in (ctx.move_category, "both"):
                    chain.add_fp(screen_fp, label=cond)
        # Friend Guard on the defender's ally.
        for ability in ctx.defender_ally_abilities:
            if ability == "friendguard":
                chain.add(0.75, label="friendguard")
        # Resist berries: halve a super-effective hit of the matching type (Chilan halves
        # Normal regardless).
        berry_type = RESIST_BERRIES.get(defender.item or "")
        if berry_type == ctx.move_type and (berry_type == "Normal" or ctx.type_mod > 0):
            chain.add(0.5, label=defender.item or "berry")

    return chain


def _unmodelled(attacker: Battler, defender: Battler) -> tuple[str, ...]:
    from .effects import all_modelled_abilities, all_modelled_items

    known_abilities = all_modelled_abilities()
    # Mega stones carry no damage modifier; the forme change is what matters and the mega
    # machinery owns it, so they are not gaps.
    known_items = all_modelled_items(_MEGA_STONE_IDS)
    out: list[str] = []
    for mon, who in ((attacker, "attacker"), (defender, "defender")):
        if mon.ability and mon.ability not in known_abilities:
            out.append(f"{who}.ability:{mon.ability}")
        if mon.item and mon.item not in known_items:
            out.append(f"{who}.item:{mon.item}")
    return tuple(out)


def calculate(
    reg: Regulation,
    attacker: Battler,
    defender: Battler,
    move_id: str,
    field_state: FieldState,
    *,
    defender_side: int = 1,
    spread: bool = False,
    crit: bool = False,
    move_ctx: MoveContext | None = None,
    base_power_override: int | None = None,
) -> DamageResult:
    """All 16 damage rolls of one hit, for every particle in the batch.

    ``spread`` is whether the move actually hit more than one target this turn, which is
    what Showdown's ``move.spreadHit`` records -- a spread move that ends up hitting a
    single target does not take the 0.75 modifier.

    ``move_ctx`` supplies the per-turn facts a variable-base-power move needs (the fainted
    count for Last Respects, weather for Weather Ball, and so on). Omitting it is safe: the
    defaults are documented on :class:`~pokeuraou.moveinfo.MoveContext`, and a move whose
    base power cannot be determined is reported as unmodelled rather than guessed.
    """
    move = reg.moves[move_id]
    n = max(attacker.n, defender.n)
    ctx_move = move_ctx or MoveContext(
        weather=field_state.weather, terrain=field_state.terrain
    )

    move_type, type_change_fp = effective_move_type(reg, move_id, attacker)
    # A move's own onModifyType (Weather Ball, Terrain Pulse) overrides the declared type.
    own_type = move_effective_type(reg, move_id, attacker, ctx_move)
    if own_type is not None:
        move_type = own_type
    immune = is_immune(
        reg,
        move_id,
        move_type,
        defender,
        attacker.ability,
        attacker_is_defender=attacker is defender,
    )

    override = effectiveness_override(reg, move_id, move_type, defender.types)
    if override is not None:
        eff, type_mod = override
        immune = eff == 0.0
    else:
        eff, type_mod = type_effectiveness(reg, move_type, defender)

    zeros = np.zeros((n, N_ROLLS), dtype=np.int64)
    unmodelled = _unmodelled(attacker, defender)

    if move.category == "Status" or immune:
        return DamageResult(
            rolls=zeros, effectiveness=eff, type_mod=type_mod, immune=immune,
            unmodelled=unmodelled,
        )

    # Disguise and Ice Face absorb one hit completely while intact. Showdown records that
    # the bust has happened by changing the forme, so an intact one is identifiable. They
    # act at the damage step, after the type immunity (IKA-155): a Normal move into an
    # intact Mimikyu is still immune, not absorbed.
    if defender.ability == "disguise" and defender.species == "mimikyu":
        return DamageResult(
            rolls=zeros, effectiveness=eff, type_mod=type_mod, immune=False,
            applied={"absorbed": (("disguise", 0),)}, unmodelled=unmodelled,
        )
    if (
        defender.ability == "iceface"
        and defender.species == "eiscue"
        and move.category == "Physical"
    ):
        return DamageResult(
            rolls=zeros, effectiveness=eff, type_mod=type_mod, immune=False,
            applied={"absorbed": (("iceface", 0),)}, unmodelled=unmodelled,
        )

    fixed = move_fixed_damage(move_id, attacker, defender)
    if fixed is not None:
        return DamageResult(
            rolls=np.repeat(fixed.reshape(-1, 1), N_ROLLS, axis=1),
            effectiveness=eff,
            type_mod=type_mod,
            immune=False,
            applied={"fixed": ((move_id, 0),)},
            unmodelled=unmodelled,
        )

    if base_power_override is not None:
        base_power_vec: np.ndarray | None = np.full(n, base_power_override, dtype=np.int64)
    else:
        base_power_vec = move_base_power(reg, move_id, attacker, defender, ctx_move)
    if base_power_vec is None or not (base_power_vec > 0).any():
        return DamageResult(
            rolls=zeros, effectiveness=eff, type_mod=type_mod, immune=False,
            unmodelled=unmodelled + (f"move.basePowerCallback:{move_id}",),
        )
    if base_power_override is None and base_power_is_approximate(move_id):
        # A declared base power is still a number, but for these moves it is a fallback
        # rather than the real one; say so instead of printing it as if it were right.
        unmodelled = unmodelled + (f"move.basePowerCallback (approximated):{move_id}",)
    base_power = int(base_power_vec.reshape(-1)[0])

    ally_abilities = _ally_abilities(field_state, defender_side, defender)
    ctx = Ctx(
        move_id=move_id,
        move_type=move_type,
        move_category=move.category,
        move_base_power=base_power,
        move_flags=move.flags,
        move_priority=move.priority,
        effectiveness=eff,
        type_mod=type_mod,
        is_crit=crit,
        is_spread=spread,
        has_secondary=bool(move.raw.get("secondaries")),
        has_recoil=bool(move.raw.get("recoil")) or bool(move.raw.get("hasCrashDamage")),
        attacker_species=attacker.species,
        attacker_types=attacker.types,
        attacker_ability=attacker.ability,
        attacker_item=attacker.item,
        attacker_status=attacker.status,
        attacker_volatiles=attacker.volatiles,
        attacker_gender=attacker.gender,
        defender_species=defender.species,
        defender_types=defender.types,
        defender_ability=defender.ability,
        defender_item=defender.item,
        defender_status=defender.status,
        defender_volatiles=defender.volatiles,
        defender_gender=defender.gender,
        defender_at_full_hp=bool(defender.at_full_hp.all()),
        attacker_in_pinch=bool((attacker.hp * 3 <= attacker.maxhp).all()),
        attacker_below_half=bool((attacker.hp * 2 <= attacker.maxhp).all()),
        weather=_effective_weather(field_state, attacker, defender),
        terrain=field_state.terrain,
        defender_side_conditions=field_state.side_conditions[defender_side],
        defender_ally_abilities=ally_abilities,
        field_abilities=_field_abilities(field_state),
        active_per_half=field_state.active_per_half,
    )

    applied: dict[str, tuple[tuple[str, int], ...]] = {}

    # -- base power ---------------------------------------------------------
    bp_chain = _collect("base_power", ctx, attacker, defender, field_state)
    for label, num, den in base_power_modifiers(reg, move_id, attacker, defender, ctx_move):
        bp_chain.add(num, den, label=label)
    for label, num, den in terrain_modifiers(
        reg, move_id, move_type, _is_grounded(attacker), field_state.terrain, _is_grounded(defender)
    ):
        bp_chain.add(num, den, label=label)
    if type_change_fp != 4096:
        bp_chain.add_fp(type_change_fp, label=f"{attacker.ability}(type change)")
    base_power_arr = np.maximum(bp_chain.apply(base_power_vec), 1)
    applied["base_power"] = bp_chain.applied

    # -- attack and defence -------------------------------------------------
    is_physical = move.category == "Physical"
    atk_key = move.raw.get("overrideOffensiveStat") or ("atk" if is_physical else "spa")
    def_key = move.raw.get("overrideDefensiveStat") or ("def" if is_physical else "spd")

    # Foul Play reads the target's Attack, and a few moves read the user's own defence.
    # The ability and item modifiers still come from the real attacker and defender, which
    # is why only the stat source moves.
    stat_owner_atk = defender if move.raw.get("overrideOffensivePokemon") == "target" else attacker
    stat_owner_def = attacker if move.raw.get("overrideDefensivePokemon") == "source" else defender

    # A crit ignores the attacker's negative offensive boosts and the defender's positive
    # defensive ones.
    atk_stage = stat_owner_atk.boosts.get(atk_key, 0)
    def_stage = stat_owner_def.boosts.get(def_key, 0)
    ignore_atk_boost = bool(move.raw.get("ignoreOffensive")) or (crit and atk_stage < 0)
    ignore_def_boost = bool(move.raw.get("ignoreDefensive")) or (crit and def_stage > 0)

    attack = stat_owner_atk.stat(atk_key, ignore_boost=ignore_atk_boost)
    defence = stat_owner_def.stat(def_key, ignore_boost=ignore_def_boost)

    atk_chain = _collect("atk", ctx, attacker, defender, field_state)
    attack = atk_chain.apply(attack)
    applied["atk"] = atk_chain.applied

    def_chain = _collect("def", ctx, defender, attacker, field_state)
    defence = def_chain.apply(defence)
    applied["def"] = def_chain.applied

    # Sandstorm raises Rock-types' SpD; Snow raises Ice-types' Def (gen 9).
    weather = ctx.weather
    if weather == "sandstorm" and def_key == "spd" and "Rock" in defender.types:
        defence = modify(defence, 1.5)
    if weather in ("snow", "snowscape") and def_key == "def" and "Ice" in defender.types:
        defence = modify(defence, 1.5)

    defence = np.maximum(defence, 1)

    # -- the formula --------------------------------------------------------
    level_term = trunc(2 * attacker.level / 5 + 2)
    base = trunc(trunc(trunc(level_term * base_power_arr * attack) / defence) / 50)
    base = base + 2

    if spread:
        base = modify(base, 0.75)

    base = _apply_weather_damage(base, ctx)

    if crit:
        crit_mod = move.raw.get("critModifier", 1.5)
        base = trunc(base * crit_mod)

    # -- the 16 rolls -------------------------------------------------------
    rolls = np.arange(N_ROLLS, dtype=np.int64)[None, :]  # (1, 16)
    if move.raw.get("noDamageVariance"):
        dmg = np.repeat(base[:, None], N_ROLLS, axis=1)
    else:
        dmg = trunc(trunc(base[:, None] * (100 - rolls)) / 100)

    # -- STAB ---------------------------------------------------------------
    if move_type != "???":
        # Protean and Libero retype the user to the move's type just before it lands, but
        # only once per switch-in: Showdown marks that with a volatile of the same name.
        # After it has fired the Pokemon keeps whatever type it was given, which is why
        # `attacker.types` must be the live types and not the species' -- a Meowscarada
        # already retyped to Grass gets no STAB on Knock Off.
        will_retype = attacker.ability in RETYPING_ABILITIES and not attacker.ability_state.get(
            "protean"
        )
        is_stab = (
            bool(move.raw.get("forceSTAB")) or will_retype or move_type in attacker.types
        )
        if is_stab:
            if attacker.ability == "adaptability":
                dmg = modify(dmg, 2.0)
                applied["stab"] = (("adaptability", 8192),)
            else:
                dmg = modify(dmg, 1.5)
                applied["stab"] = (("stab", 6144),)

    # -- type effectiveness -------------------------------------------------
    for _ in range(max(type_mod, 0)):
        dmg = dmg * 2
    for _ in range(max(-type_mod, 0)):
        dmg = trunc(dmg / 2)

    # -- burn ---------------------------------------------------------------
    if (
        attacker.status == "brn"
        and is_physical
        and attacker.ability != "guts"
        and move_id != "facade"
    ):
        dmg = modify(dmg, 0.5)

    # -- final modifiers ----------------------------------------------------
    dmg_chain = _collect("damage", ctx, attacker, defender, field_state)
    applied["damage"] = dmg_chain.applied
    if dmg_chain.is_identity:
        pass
    elif _multiscale_is_mixed(defender, ctx):
        # Multiscale/Shadow Shield depend on the defender being at full HP, which can
        # differ across particles. Apply the chain with and without it and select.
        without = _collect_without(
            "damage", ctx, attacker, defender, field_state, ("multiscale", "shadowshield")
        )
        full = dmg_chain.apply(dmg)
        partial = without.apply(dmg)
        dmg = np.where(defender.at_full_hp[:, None], full, partial)
    else:
        dmg = dmg_chain.apply(dmg)

    # Showdown returns at least 1 damage after the final modifier, then truncates to
    # 16 bits (which can wrap an absurd value to 0).
    dmg = np.maximum(dmg, 1)
    dmg = trunc16(dmg)

    return DamageResult(
        rolls=dmg,
        effectiveness=eff,
        type_mod=type_mod,
        immune=False,
        applied=applied,
        unmodelled=unmodelled,
    )


def _collect_without(
    slot: Slot,
    ctx: Ctx,
    attacker: Battler,
    defender: Battler,
    field_state: FieldState,
    exclude: tuple[str, ...],
) -> Chain:
    full = _collect(slot, ctx, attacker, defender, field_state)
    chain = Chain()
    for label, mod_fp in full.applied:
        if label in exclude:
            continue
        chain.add_fp(mod_fp, label=label)
    return chain


def _multiscale_is_mixed(defender: Battler, ctx: Ctx) -> bool:
    if defender.ability not in ("multiscale", "shadowshield"):
        return False
    full = defender.at_full_hp
    del ctx
    return bool(full.any() and not full.all())


def _ally_abilities(
    field_state: FieldState, defender_side: int, defender: Battler
) -> tuple[str, ...]:
    """Abilities that reach the defender from another slot.

    Friend Guard only applies from the defender's *ally*, so the defender's own ability is
    excluded (it is already consulted directly). Aura abilities apply from anywhere, and
    are picked up from both sides by the caller.
    """
    allies = list(field_state.active_abilities[defender_side])
    if defender.ability in allies:
        allies.remove(defender.ability)
    return tuple(allies)


def _field_abilities(field_state: FieldState) -> tuple[str, ...]:
    """Every active Pokemon's ability, for the field-wide Aura abilities."""
    return tuple(field_state.active_abilities[0]) + tuple(field_state.active_abilities[1])


def _is_grounded(mon: Battler) -> bool:
    """Whether terrain reaches this Pokemon."""
    if "smackdown" in mon.volatiles or "ingrain" in mon.volatiles or mon.item == "ironball":
        return True
    if "magnetrise" in mon.volatiles or "telekinesis" in mon.volatiles:
        return False
    if mon.ability == "levitate" or mon.item == "airballoon":
        return False
    return "Flying" not in mon.types


def _effective_weather(
    field_state: FieldState, attacker: Battler, defender: Battler
) -> str | None:
    """Weather, unless something suppresses it.

    Cloud Nine and Air Lock suppress weather for the entire field, so every active
    Pokemon has to be consulted -- an ally holding one counts, which is exactly the case
    the differential test caught. Utility Umbrella is per-holder, so only the attacker and
    defender matter for it.
    """
    if field_state.weather is None:
        return None
    if any(
        ability in WEATHER_SUPPRESSING_ABILITIES
        for side in field_state.active_abilities
        for ability in side
    ):
        return None
    for mon in (attacker, defender):
        if mon.ability in WEATHER_SUPPRESSING_ABILITIES:
            return None
        if mon.item == "utilityumbrella" and field_state.weather in (
            "sunnyday", "raindance", "desolateland", "primordialsea"
        ):
            return None
    return field_state.weather


def _apply_weather_damage(base: np.ndarray, ctx: Ctx) -> np.ndarray:
    """``WeatherModifyDamage``: Sun/Rain boost or weaken Fire and Water."""
    weather = ctx.weather
    if weather in ("sunnyday", "desolateland"):
        if ctx.move_type == "Fire":
            return modify(base, 1.5)
        if ctx.move_type == "Water":
            if weather == "desolateland":
                return np.zeros_like(base)
            return modify(base, 0.5)
    elif weather in ("raindance", "primordialsea"):
        if ctx.move_type == "Water":
            return modify(base, 1.5)
        if ctx.move_type == "Fire":
            if weather == "primordialsea":
                return np.zeros_like(base)
            return modify(base, 0.5)
    return base


def effective_damage(result: DamageResult, defender: Battler) -> np.ndarray:
    """Damage actually dealt, after capping at the defender's HP and survival effects.

    Focus Sash, Focus Band and Sturdy do not change the damage roll; they stop a full-HP
    Pokemon from being knocked out, leaving it at 1 HP. That is the number Showdown's
    protocol reports, so it is applied here rather than inside the roll pipeline.
    """
    dealt = np.minimum(result.rolls, defender.hp[:, None])
    survives = (
        defender.item in SURVIVE_AT_ONE_ITEMS or defender.ability in SURVIVE_AT_ONE_ABILITIES
    )
    if survives:
        at_full = (defender.hp >= defender.maxhp)[:, None]
        capped = np.minimum(dealt, np.maximum(defender.hp[:, None] - 1, 0))
        dealt = np.where(at_full, capped, dealt)
    return dealt
