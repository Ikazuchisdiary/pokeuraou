//! `modifyDamage`, transcribed from `src/pokeuraou/damage.py` step for step.
//!
//! The Python is itself a step-for-step transcription of Showdown's `scripts.ts`, and the
//! modifier order and the rounding at each step are the whole point, so this follows the
//! Python's structure literally: same slots, same chain order, same truncations.

use crate::battler::{Battler, DamageResult, FieldState, N_ROLLS, V_CHARGE, V_HELPING_HAND};
use crate::battler::{V_INGRAIN, V_MAGNET_RISE, V_SMACK_DOWN, V_TELEKINESIS};
use crate::effects::{
    ability_modifiers, is_mold_breaker, is_retyping, item_modifiers, pierces_ghost, resist_berry,
    suppresses_weather, type_boost_item, type_boost_item_fp, type_changing_ability,
    type_immunity_ability, Ctx, ModDef, Slot, AURA_ABILITIES, AURA_BREAK_ABILITY, AURA_BROKEN_FP,
    AURA_FP, SCREEN_CONDITIONS, SCREEN_FP_DOUBLES, SCREEN_FP_SINGLES, TYPE_CHANGE_BOOST_FP,
};
use crate::fixedpoint::{apply_fp, modify, trunc, trunc16, Chain};
use crate::id::Id;
use crate::moveinfo::{
    base_power, base_power_modifiers, effective_type, effectiveness_override, fixed_damage,
    terrain_modifiers, MoveContext,
};
use crate::position::Types;
use crate::reg::Reg;

/// `damage._unmodelled`: the abilities and items on this hit that the calculator does
/// not account for, named so the caller can report them.
fn unmodelled_effects(attacker: &Battler, defender: &Battler) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (mon, who) in [(attacker, "attacker"), (defender, "defender")] {
        if !mon.ability.is_empty() && !crate::modelled::ability_is_modelled(mon.ability.as_str())
        {
            out.push(format!("{who}.ability:{}", mon.ability));
        }
        if let Some(item) = mon.item {
            if !crate::modelled::item_is_modelled(item.as_str()) {
                out.push(format!("{who}.item:{item}"));
            }
        }
    }
    out
}

fn same(value: Option<Id>, name: &str) -> bool {
    matches!(value, Some(v) if v.as_str() == name)
}

fn effective_move_type(reg: &Reg, move_id: &str, attacker: &Battler) -> (Id, i64) {
    let mv = &reg.moves[move_id];
    let declared = Id::new(&mv.mtype);
    let changed = match type_changing_ability(attacker.ability.as_str()) {
        None => return (declared, 4096),
        Some(t) => t,
    };
    if mv.category == "Status" {
        return (declared, 4096);
    }
    if attacker.ability == "liquidvoice" {
        if !mv.has_flag("sound") {
            return (declared, 4096);
        }
        return (Id::new(changed), 4096);
    }
    if attacker.ability == "normalize" {
        return (Id::new(changed), TYPE_CHANGE_BOOST_FP);
    }
    if mv.mtype != "Normal" {
        return (declared, 4096);
    }
    (Id::new(changed), TYPE_CHANGE_BOOST_FP)
}

fn type_effectiveness(reg: &Reg, move_type: &str, defender: &Battler) -> (f64, i64) {
    let mult = reg.type_effectiveness(move_type, &defender.types);
    if mult == 0.0 {
        return (0.0, 0);
    }
    let mut steps = 0i64;
    if let Some(row) = reg.typechart.get(move_type) {
        for t in defender.types.as_slice() {
            match row.get(t.as_str()).copied().unwrap_or(1.0) {
                v if v == 2.0 => steps += 1,
                v if v == 0.5 => steps -= 1,
                _ => {}
            }
        }
    }
    (mult, steps.clamp(-6, 6))
}

pub fn is_immune(
    reg: &Reg,
    move_id: &str,
    move_type: &str,
    defender: &Battler,
    attacker_ability: &str,
    attacker_is_defender: bool,
) -> bool {
    let mv = &reg.moves[move_id];
    match mv.raw.get("ignoreImmunity") {
        Some(serde_json::Value::Bool(true)) => return false,
        Some(serde_json::Value::Object(o)) => {
            if o.get(move_type).map(|v| v != &serde_json::Value::Bool(false)).unwrap_or(false) {
                return false;
            }
        }
        _ => {}
    }
    let mult = reg.type_effectiveness(move_type, &defender.types);
    if mult == 0.0 {
        let pierced = pierces_ghost(attacker_ability)
            && matches!(move_type, "Normal" | "Fighting")
            && defender.types.contains("Ghost");
        return !pierced;
    }
    if type_immunity_ability(defender.ability.as_str()) == Some(move_type) {
        return true;
    }
    if same(defender.item, "airballoon") && move_type == "Ground" {
        return true;
    }
    if defender.ability == "wonderguard" && mult <= 1.0 {
        return true;
    }
    if defender.ability == "bulletproof" && mv.has_flag("bullet") {
        return true;
    }
    if defender.ability == "soundproof" && mv.has_flag("sound") && !attacker_is_defender {
        return true;
    }
    if defender.ability == "overcoat" && mv.has_flag("powder") {
        return true;
    }
    defender.types.contains("Grass") && mv.has_flag("powder")
}

/// One event's modifier chain, in Showdown's handler order.
fn collect(slot: Slot, ctx: &Ctx, a: &Battler, d: &Battler, field: &FieldState) -> Chain {
    let mut mods: Vec<&'static ModDef> = Vec::new();

    let mut consider = |ability: &str, from_defender: bool, out: &mut Vec<&'static ModDef>| {
        for md in ability_modifiers(ability) {
            if md.slot != slot || md.from_defender != from_defender {
                continue;
            }
            // An aura reaches the whole field; it is collected once below instead.
            if AURA_ABILITIES.iter().any(|(id, _)| *id == md.id) {
                continue;
            }
            if (md.when)(ctx) {
                out.push(md);
            }
        }
    };

    consider(a.ability.as_str(), false, &mut mods);
    if !is_mold_breaker(a.ability.as_str()) {
        consider(d.ability.as_str(), true, &mut mods);
    }

    for (item, from_defender) in [(a.item, false), (d.item, true)] {
        let Some(item) = item else { continue };
        for md in item_modifiers(item.as_str()) {
            if md.slot != slot || md.from_defender != from_defender {
                continue;
            }
            if (md.when)(ctx) {
                mods.push(md);
            }
        }
    }

    let mut aura_fp: Option<i64> = None;
    if slot == Slot::BasePower {
        let broken = ctx.field_abilities.iter().any(|x| x.as_str() == AURA_BREAK_ABILITY);
        for (ability, move_type) in AURA_ABILITIES {
            if ctx.field_abilities.iter().any(|x| x.as_str() == ability)
                && ctx.move_type.as_str() == move_type
            {
                aura_fp = Some(if broken { AURA_BROKEN_FP } else { AURA_FP });
                break;
            }
        }
    }

    mods.sort_by_key(|md| -md.priority);

    let mut chain = Chain::new();
    for md in mods {
        chain.add(md.num, md.den, md.id);
    }
    if let Some(fp) = aura_fp {
        chain.add_fp(fp, "aura");
    }

    if slot == Slot::BasePower {
        if let Some(item) = a.item {
            if type_boost_item(item.as_str()) == Some(ctx.move_type.as_str()) {
                chain.add_fp(type_boost_item_fp(), "typeboostitem");
            }
        }
        if a.volatiles.has(V_HELPING_HAND) {
            chain.add(1.5, 1.0, "helpinghand");
        }
        if a.volatiles.has(V_CHARGE) && ctx.move_type == "Electric" {
            chain.add(2.0, 1.0, "charge");
        }
    } else if slot == Slot::Damage {
        if !ctx.is_crit && !ctx.move_flags.contains("infiltrates") {
            let screen_fp = if field.active_per_half > 1 {
                SCREEN_FP_DOUBLES
            } else {
                SCREEN_FP_SINGLES
            };
            for (cond, category) in SCREEN_CONDITIONS {
                if !ctx.defender_side_conditions.iter().any(|c| c.as_str() == cond) {
                    continue;
                }
                if category == ctx.move_category || category == "both" {
                    chain.add_fp(screen_fp, "screen");
                }
            }
        }
        for ability in ctx.defender_ally_abilities {
            if ability.as_str() == "friendguard" {
                chain.add(0.75, 1.0, "friendguard");
            }
        }
        if let Some(berry_type) = d.item.and_then(|i| resist_berry(i.as_str())) {
            if berry_type == ctx.move_type.as_str()
                && (berry_type == "Normal" || ctx.type_mod > 0)
            {
                chain.add(0.5, 1.0, "resistberry");
            }
        }
    }

    chain
}

pub fn is_grounded(mon: &Battler) -> bool {
    if mon.volatiles.has(V_SMACK_DOWN)
        || mon.volatiles.has(V_INGRAIN)
        || same(mon.item, "ironball")
    {
        return true;
    }
    if mon.volatiles.has(V_MAGNET_RISE) || mon.volatiles.has(V_TELEKINESIS) {
        return false;
    }
    if mon.ability == "levitate" || same(mon.item, "airballoon") {
        return false;
    }
    !mon.types.contains("Flying")
}

pub fn effective_weather(
    field: &FieldState,
    attacker: &Battler,
    defender: &Battler,
) -> Option<Id> {
    let weather = field.weather?;
    for side in &field.active_abilities {
        for ability in side {
            if suppresses_weather(ability.as_str()) {
                return None;
            }
        }
    }
    for mon in [attacker, defender] {
        if suppresses_weather(mon.ability.as_str()) {
            return None;
        }
        if same(mon.item, "utilityumbrella")
            && matches!(
                weather.as_str(),
                "sunnyday" | "raindance" | "desolateland" | "primordialsea"
            )
        {
            return None;
        }
    }
    Some(weather)
}

fn apply_weather_damage(base: i64, weather: Option<Id>, move_type: &str) -> i64 {
    let name = weather.map(|w| w.as_str().to_string());
    match name.as_deref() {
        Some("sunnyday") | Some("desolateland") => {
            if move_type == "Fire" {
                return modify(base, 1.5, 1.0);
            }
            if move_type == "Water" {
                if name.as_deref() == Some("desolateland") {
                    return 0;
                }
                return modify(base, 0.5, 1.0);
            }
        }
        Some("raindance") | Some("primordialsea") => {
            if move_type == "Water" {
                return modify(base, 1.5, 1.0);
            }
            if move_type == "Fire" {
                if name.as_deref() == Some("primordialsea") {
                    return 0;
                }
                return modify(base, 0.5, 1.0);
            }
        }
        _ => {}
    }
    base
}

/// How many damage calculations have been made. Counted for the same reason position
/// clones are: the resolver's cost has to be attributed, not guessed at.
pub static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[allow(clippy::too_many_arguments)]
pub fn calculate(
    reg: &Reg,
    attacker: &Battler,
    defender: &Battler,
    move_id: &str,
    field: &FieldState,
    defender_side: usize,
    spread: bool,
    crit: bool,
    move_ctx: Option<&MoveContext>,
    base_power_override: Option<i64>,
    attacker_is_defender: bool,
) -> DamageResult {
    CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mv = &reg.moves[move_id];
    let owned_ctx;
    let ctx_move = match move_ctx {
        Some(c) => c,
        None => {
            owned_ctx = MoveContext {
                weather: field.weather.map(|w| w.as_str().to_string()),
                terrain: field.terrain.map(|t| t.as_str().to_string()),
                hit_index: 1,
                ..Default::default()
            };
            &owned_ctx
        }
    };

    let (mut move_type, type_change_fp) = effective_move_type(reg, move_id, attacker);
    if let Some(own) = effective_type(move_id, attacker, ctx_move) {
        move_type = Id::new(&own);
    }
    let mut immune = is_immune(
        reg,
        move_id,
        move_type.as_str(),
        defender,
        attacker.ability.as_str(),
        attacker_is_defender,
    );

    let (eff, type_mod) =
        match effectiveness_override(reg, move_id, move_type.as_str(), &defender.types) {
            Some((mult, tm)) => {
                immune = mult == 0.0;
                (mult, tm)
            }
            None => type_effectiveness(reg, move_type.as_str(), defender),
        };

    let unmodelled = unmodelled_effects(attacker, defender);
    let zeros = DamageResult {
        rolls: [0; N_ROLLS],
        effectiveness: eff,
        type_mod,
        immune: false,
        unmodelled: unmodelled.clone(),
    };

    if defender.ability == "disguise" && defender.species == "mimikyu" {
        return zeros;
    }
    if defender.ability == "iceface" && defender.species == "eiscue" && mv.category == "Physical" {
        return zeros;
    }
    if mv.category == "Status" || immune {
        return DamageResult {
            rolls: [0; N_ROLLS],
            effectiveness: eff,
            type_mod,
            immune,
            unmodelled,
        };
    }

    if let Some(fixed) = fixed_damage(move_id, attacker, defender) {
        return DamageResult {
            rolls: [fixed; N_ROLLS],
            effectiveness: eff,
            type_mod,
            immune: false,
            unmodelled,
        };
    }

    let declared = match base_power_override {
        Some(bp) => Some(bp),
        None => base_power(reg, move_id, attacker, defender, ctx_move),
    };
    let mut unmodelled = unmodelled;
    let base_power_value = match declared {
        Some(bp) if bp > 0 => bp,
        _ => {
            let mut out = zeros;
            out.unmodelled.push(format!("move.basePowerCallback:{move_id}"));
            return out;
        }
    };
    if base_power_override.is_none() && crate::moveinfo::is_approximate(move_id) {
        // A declared base power is still a number, but for these moves it is a fallback
        // rather than the real one; say so instead of printing it as if it were right.
        unmodelled.push(format!("move.basePowerCallback (approximated):{move_id}"));
    }

    let mut ally_abilities: Vec<Id> = field.active_abilities[defender_side].clone();
    if let Some(index) = ally_abilities.iter().position(|a| *a == defender.ability) {
        ally_abilities.remove(index);
    }
    let mut field_abilities: Vec<Id> = Vec::with_capacity(4);
    field_abilities.extend_from_slice(&field.active_abilities[0]);
    field_abilities.extend_from_slice(&field.active_abilities[1]);
    let weather = effective_weather(field, attacker, defender);

    let ctx = Ctx {
        move_id,
        move_type,
        move_category: &mv.category,
        move_base_power: base_power_value,
        move_flags: &mv.flags,
        move_priority: mv.priority,
        effectiveness: eff,
        type_mod,
        is_crit: crit,
        is_spread: spread,
        has_secondary: mv.raw_bool("secondaries"),
        has_recoil: mv.raw_bool("recoil") || mv.raw_bool("hasCrashDamage"),
        attacker_species: attacker.species,
        attacker_types: attacker.types,
        attacker_ability: attacker.ability,
        attacker_item: attacker.item,
        attacker_status: attacker.status,
        attacker_volatiles: attacker.volatiles,
        attacker_gender: attacker.gender,
        defender_species: defender.species,
        defender_types: defender.types,
        defender_ability: defender.ability,
        defender_item: defender.item,
        defender_status: defender.status,
        defender_volatiles: defender.volatiles,
        defender_gender: defender.gender,
        defender_at_full_hp: defender.at_full_hp(),
        attacker_in_pinch: attacker.hp * 3 <= attacker.maxhp,
        attacker_below_half: attacker.hp * 2 <= attacker.maxhp,
        weather,
        terrain: field.terrain,
        defender_side_conditions: &field.side_conditions[defender_side],
        defender_ally_abilities: &ally_abilities,
        field_abilities: &field_abilities,
        active_per_half: field.active_per_half,
    };

    // -- base power --------------------------------------------------------
    let mut bp_chain = collect(Slot::BasePower, &ctx, attacker, defender, field);
    for (label, num, den) in base_power_modifiers(reg, move_id, attacker, defender, ctx_move) {
        bp_chain.add(num, den, label);
    }
    for (label, num, den) in terrain_modifiers(
        move_type.as_str(),
        is_grounded(attacker),
        field.terrain.map(|t| t.as_str().to_string()).as_deref(),
    ) {
        bp_chain.add(num, den, label);
    }
    if type_change_fp != 4096 {
        bp_chain.add_fp(type_change_fp, "typechange");
    }
    let base_power_final = bp_chain.apply(base_power_value).max(1);

    // -- attack and defence ------------------------------------------------
    let is_physical = mv.category == "Physical";
    let atk_key = mv
        .raw_str("overrideOffensiveStat")
        .unwrap_or(if is_physical { "atk" } else { "spa" });
    let def_key = mv
        .raw_str("overrideDefensiveStat")
        .unwrap_or(if is_physical { "def" } else { "spd" });

    let stat_owner_atk =
        if mv.raw_str("overrideOffensivePokemon") == Some("target") { defender } else { attacker };
    let stat_owner_def =
        if mv.raw_str("overrideDefensivePokemon") == Some("source") { attacker } else { defender };

    let atk_stage = stat_owner_atk.boost(atk_key);
    let def_stage = stat_owner_def.boost(def_key);
    let ignore_atk_boost = mv.raw_bool("ignoreOffensive") || (crit && atk_stage < 0);
    let ignore_def_boost = mv.raw_bool("ignoreDefensive") || (crit && def_stage > 0);

    let attack = stat_owner_atk.stat(atk_key, ignore_atk_boost);
    let defence = stat_owner_def.stat(def_key, ignore_def_boost);

    let atk_chain = collect(Slot::Atk, &ctx, attacker, defender, field);
    let attack = atk_chain.apply(attack);

    let def_chain = collect(Slot::Def, &ctx, defender, attacker, field);
    let mut defence = def_chain.apply(defence);

    if same(weather, "sandstorm") && def_key == "spd" && defender.types.contains("Rock") {
        defence = modify(defence, 1.5, 1.0);
    }
    if (same(weather, "snow") || same(weather, "snowscape"))
        && def_key == "def"
        && defender.types.contains("Ice")
    {
        defence = modify(defence, 1.5, 1.0);
    }
    let defence = defence.max(1);

    // -- the formula -------------------------------------------------------
    let level_term = trunc(2.0 * attacker.level as f64 / 5.0 + 2.0);
    let inner = trunc((level_term * base_power_final * attack) as f64);
    let base = trunc(trunc(inner as f64 / defence as f64) as f64 / 50.0);
    let mut base = base + 2;

    if spread {
        base = modify(base, 0.75, 1.0);
    }
    base = apply_weather_damage(base, weather, move_type.as_str());
    if crit {
        let crit_mod = mv.raw_f64("critModifier", 1.5);
        base = trunc(base as f64 * crit_mod);
    }

    // -- the 16 rolls ------------------------------------------------------
    let mut dmg = [0i64; N_ROLLS];
    if mv.raw_bool("noDamageVariance") {
        dmg = [base; N_ROLLS];
    } else {
        for (r, slot) in dmg.iter_mut().enumerate() {
            // IKA-99's integer arm. The f64 round trip is exact while the product is below
            // 2^53 (`base` is at most 2^32 after `trunc`, times 100), so wrapping to 32 bits
            // and dividing as integers is the same number; the arm exists to measure that.
            #[cfg(feature = "int-rolls")]
            {
                *slot = crate::fixedpoint::wrap32(base * (100 - r as i64)) / 100;
            }
            #[cfg(not(feature = "int-rolls"))]
            {
                *slot = trunc(trunc((base * (100 - r as i64)) as f64) as f64 / 100.0);
            }
        }
    }

    // -- STAB --------------------------------------------------------------
    if move_type.as_str() != "???" {
        let will_retype = is_retyping(attacker.ability.as_str()) && !attacker.protean_fired;
        let is_stab = mv.raw_bool("forceSTAB")
            || will_retype
            || attacker.types.contains(move_type.as_str());
        if is_stab {
            let fp = if attacker.ability == "adaptability" { 8192 } else { 6144 };
            for value in dmg.iter_mut() {
                *value = apply_fp(*value, fp);
            }
        }
    }

    // -- type effectiveness ------------------------------------------------
    for _ in 0..type_mod.max(0) {
        for value in dmg.iter_mut() {
            *value *= 2;
        }
    }
    for _ in 0..(-type_mod).max(0) {
        for value in dmg.iter_mut() {
            *value = trunc(*value as f64 / 2.0);
        }
    }

    // -- burn --------------------------------------------------------------
    if same(attacker.status, "brn")
        && is_physical
        && attacker.ability != "guts"
        && move_id != "facade"
    {
        for value in dmg.iter_mut() {
            *value = modify(*value, 0.5, 1.0);
        }
    }

    // -- final modifiers ---------------------------------------------------
    let dmg_chain = collect(Slot::Damage, &ctx, attacker, defender, field);
    if !dmg_chain.is_identity() {
        for value in dmg.iter_mut() {
            *value = dmg_chain.apply(*value);
        }
    }

    for value in dmg.iter_mut() {
        *value = trunc16((*value).max(1));
    }

    DamageResult { rolls: dmg, effectiveness: eff, type_mod, immune: false, unmodelled }
}

/// `crit_stage`, before Showdown's clamp to 0..4.
pub fn crit_stage(reg: &Reg, attacker: &Battler, move_id: &str) -> i64 {
    use crate::battler::{V_DRAGON_CHEER, V_FOCUS_ENERGY};
    let mv = &reg.moves[move_id];
    let mut stage = if mv.crit_ratio != 0 { mv.crit_ratio - 1 } else { 0 };
    if mv.raw_bool("willCrit") {
        return 4;
    }
    if attacker.volatiles.has(V_FOCUS_ENERGY) {
        stage += 2;
    }
    if attacker.volatiles.has(V_DRAGON_CHEER) {
        stage += 1;
    }
    if same(attacker.item, "scopelens") {
        stage += 1;
    }
    if same(attacker.item, "leek")
        && (attacker.species.starts_with("farfetchd") || attacker.species.starts_with("sirfetchd"))
    {
        stage += 2;
    }
    if attacker.ability == "superluck" {
        stage += 1;
    }
    stage
}

const CRIT_MULT: [i64; 5] = [0, 24, 8, 2, 1];

pub fn crit_probability(reg: &Reg, attacker: &Battler, defender: &Battler, move_id: &str) -> f64 {
    if defender.ability == "battlearmor" || defender.ability == "shellarmor" {
        return 0.0;
    }
    let ratio = (crit_stage(reg, attacker, move_id) + 1).clamp(0, 4) as usize;
    let denom = CRIT_MULT[ratio];
    if denom == 0 {
        0.0
    } else {
        1.0 / denom as f64
    }
}

/// Damage actually dealt, after capping at the defender's HP and survival effects.
pub fn effective_damage(rolls: &[i64; N_ROLLS], defender: &Battler) -> [i64; N_ROLLS] {
    use crate::effects::{survives_at_one_ability, survives_at_one_item};
    let mut out = [0i64; N_ROLLS];
    let survives = defender.item.map(|i| survives_at_one_item(i.as_str())).unwrap_or(false)
        || survives_at_one_ability(defender.ability.as_str());
    let at_full = defender.hp >= defender.maxhp;
    for (index, roll) in rolls.iter().enumerate() {
        let mut dealt = (*roll).min(defender.hp);
        if survives && at_full {
            dealt = dealt.min((defender.hp - 1).max(0));
        }
        out[index] = dealt;
    }
    out
}

/// The types a Pokemon counts as for a move's effectiveness, given the position's live
/// types or the species' own.
pub fn types_or_species(reg: &Reg, species: &str, live: Types) -> Types {
    if !live.is_empty() {
        return live;
    }
    match reg.species.get(species) {
        None => live,
        Some(entry) => {
            let ids: Vec<Id> = entry.types.iter().map(|t| Id::new(t)).collect();
            Types::from_slice(&ids)
        }
    }
}
