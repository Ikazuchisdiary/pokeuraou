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
use crate::reg::{Category, Move, Reg, TypeSlots, F_BULLET, F_INFILTRATES, F_POWDER, F_SOUND};

/// A short list kept on the stack, spilling to the heap only past `N`. The damage path
/// builds four modifier lists and two ability lists per call, each almost always under
/// four long; as `Vec`s they were an allocation apiece whenever they were not empty
/// (IKA-101).
struct Small<T: Copy, const N: usize> {
    buf: [T; N],
    len: usize,
    spill: Vec<T>,
}

impl<T: Copy, const N: usize> Small<T, N> {
    fn new(fill: T) -> Self {
        Small { buf: [fill; N], len: 0, spill: Vec::new() }
    }

    fn push(&mut self, value: T) {
        if self.spill.is_empty() && self.len < N {
            self.buf[self.len] = value;
            self.len += 1;
            return;
        }
        if self.spill.is_empty() {
            self.spill.extend_from_slice(&self.buf[..self.len]);
        }
        self.spill.push(value);
    }

    fn as_slice(&self) -> &[T] {
        if self.spill.is_empty() {
            &self.buf[..self.len]
        } else {
            &self.spill
        }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        if self.spill.is_empty() {
            &mut self.buf[..self.len]
        } else {
            &mut self.spill
        }
    }
}

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

fn effective_move_type(mv: &Move, attacker: &Battler) -> (Id, i64) {
    let declared = mv.mtype;
    let changed = match type_changing_ability(attacker.ability.as_str()) {
        None => return (declared, 4096),
        Some(t) => t,
    };
    if mv.category == Category::Status {
        return (declared, 4096);
    }
    if attacker.ability == "liquidvoice" {
        if !mv.has_flag(F_SOUND) {
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

/// `mult` is `reg.effectiveness_of(move_slot, defender_slots)`, computed once by the
/// caller; the steps read the same chart row.
fn type_effectiveness(
    reg: &Reg,
    move_slot: usize,
    defender_slots: &TypeSlots,
    mult: f64,
) -> (f64, i64) {
    if mult == 0.0 {
        return (0.0, 0);
    }
    let mut steps = 0i64;
    let row = &reg.type_chart[move_slot];
    for d in defender_slots.as_slice() {
        match row[*d] {
            v if v == 2.0 => steps += 1,
            v if v == 0.5 => steps -= 1,
            _ => {}
        }
    }
    (mult, steps.clamp(-6, 6))
}

/// `is_immune`, with the move already looked up and the effectiveness already computed
/// (`calculate` was its only caller, and computed both again inside it).
fn immune_given(
    mv: &Move,
    move_type: &str,
    mult: f64,
    defender: &Battler,
    attacker_ability: &str,
    attacker_is_defender: bool,
) -> bool {
    if mv.ignore_immunity_all {
        return false;
    }
    if mv.ignore_immunity_types.iter().any(|t| t.as_str() == move_type) {
        return false;
    }
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
    if defender.ability == "bulletproof" && mv.has_flag(F_BULLET) {
        return true;
    }
    if defender.ability == "soundproof" && mv.has_flag(F_SOUND) && !attacker_is_defender {
        return true;
    }
    if defender.ability == "overcoat" && mv.has_flag(F_POWDER) {
        return true;
    }
    defender.types.contains("Grass") && mv.has_flag(F_POWDER)
}

/// Fills `Small`'s unused slots in `collect`; never read, never pushed.
static NO_MOD: ModDef = ModDef {
    id: "",
    slot: Slot::Stab,
    num: 1.0,
    den: 1.0,
    priority: 0,
    from_defender: false,
    when: |_| false,
};

/// One event's modifier chain, in Showdown's handler order.
fn collect(slot: Slot, ctx: &Ctx, a: &Battler, d: &Battler, field: &FieldState) -> Chain {
    let mut mods: Small<&'static ModDef, 8> = Small::new(&NO_MOD);

    let mut consider = |ability: &str, from_defender: bool, out: &mut Small<&'static ModDef, 8>| {
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

    // `sort_by_key` is stable on a slice as it was on the `Vec`, so equal priorities keep
    // the order they were collected in.
    mods.as_mut_slice().sort_by_key(|md| -md.priority);

    let mut chain = Chain::new();
    for md in mods.as_slice() {
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
        if !ctx.is_crit && !ctx.move_flags.has(F_INFILTRATES) {
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
    let name = weather.as_ref().map(|w| w.as_str());
    match name {
        Some("sunnyday") | Some("desolateland") => {
            if move_type == "Fire" {
                return modify(base, 1.5, 1.0);
            }
            if move_type == "Water" {
                if name == Some("desolateland") {
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
                if name == Some("primordialsea") {
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
                weather: field.weather,
                terrain: field.terrain,
                hit_index: 1,
                ..Default::default()
            };
            &owned_ctx
        }
    };

    let (mut move_type, type_change_fp) = effective_move_type(mv, attacker);
    if let Some(own) = effective_type(move_id, attacker, ctx_move) {
        move_type = Id::new(own);
    }
    // The chart is read by slot; the move's and the defender's are looked up once here and
    // shared by the immunity check and the effectiveness, which each computed them anew.
    let move_slot = reg.type_slot_of(move_type.as_str());
    let defender_slots = reg.type_slots(&defender.types);
    let mult = reg.effectiveness_of(move_slot, &defender_slots);
    let mut immune = immune_given(
        mv,
        move_type.as_str(),
        mult,
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
            None => type_effectiveness(reg, move_slot, &defender_slots, mult),
        };

    let unmodelled = unmodelled_effects(attacker, defender);
    let zeros = DamageResult {
        rolls: [0; N_ROLLS],
        effectiveness: eff,
        type_mod,
        immune: false,
        unmodelled: unmodelled.clone(),
    };

    if mv.category == Category::Status || immune {
        return DamageResult {
            rolls: [0; N_ROLLS],
            effectiveness: eff,
            type_mod,
            immune,
            unmodelled,
        };
    }
    // After the immunity, as in damage.py (IKA-155): the forme guards act at the damage
    // step, so a Normal move into an intact Mimikyu is immune, not absorbed.
    // Disguise is `breakable`: a Mold Breaker's move goes through it (IKA-208).
    if defender.ability == "disguise"
        && defender.species == "mimikyu"
        && !is_mold_breaker(attacker.ability.as_str())
    {
        return zeros;
    }
    if defender.ability == "iceface"
        && defender.species == "eiscue"
        && mv.category == Category::Physical
    {
        return zeros;
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

    // The defender's side without the defender's own ability (the first one equal to it,
    // as `Vec::remove(position)` took), and both sides together -- on the stack.
    let mut ally_abilities: Small<Id, 4> = Small::new(Id::EMPTY);
    let mut skipped_self = false;
    for ability in &field.active_abilities[defender_side] {
        if !skipped_self && *ability == defender.ability {
            skipped_self = true;
            continue;
        }
        ally_abilities.push(*ability);
    }
    let mut field_abilities: Small<Id, 4> = Small::new(Id::EMPTY);
    for ability in field.active_abilities[0].iter().chain(&field.active_abilities[1]) {
        field_abilities.push(*ability);
    }
    let weather = effective_weather(field, attacker, defender);

    let ctx = Ctx {
        move_id,
        move_type,
        move_category: mv.category.as_str(),
        move_base_power: base_power_value,
        move_flags: mv.flags,
        move_priority: mv.priority,
        effectiveness: eff,
        type_mod,
        is_crit: crit,
        is_spread: spread,
        has_secondary: mv.has_secondaries,
        has_recoil: mv.has_recoil || mv.has_crash_damage,
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
        defender_ally_abilities: ally_abilities.as_slice(),
        field_abilities: field_abilities.as_slice(),
        active_per_half: field.active_per_half,
    };

    // -- base power --------------------------------------------------------
    let mut bp_chain = collect(Slot::BasePower, &ctx, attacker, defender, field);
    if let Some((label, num, den)) =
        base_power_modifiers(reg, move_id, attacker, defender, ctx_move)
    {
        bp_chain.add(num, den, label);
    }
    if let Some((label, num, den)) = terrain_modifiers(
        move_id,
        move_type.as_str(),
        is_grounded(attacker),
        field.terrain.as_ref().map(|t| t.as_str()),
        is_grounded(defender),
    ) {
        bp_chain.add(num, den, label);
    }
    if type_change_fp != 4096 {
        bp_chain.add_fp(type_change_fp, "typechange");
    }
    let base_power_final = bp_chain.apply(base_power_value).max(1);

    // -- attack and defence ------------------------------------------------
    let is_physical = mv.category == Category::Physical;
    let atk_key = mv
        .override_offensive_stat
        .as_deref()
        .unwrap_or(if is_physical { "atk" } else { "spa" });
    let def_key = mv
        .override_defensive_stat
        .as_deref()
        .unwrap_or(if is_physical { "def" } else { "spd" });

    let stat_owner_atk = if mv.offensive_stat_from_target { defender } else { attacker };
    let stat_owner_def = if mv.defensive_stat_from_source { attacker } else { defender };

    let atk_stage = stat_owner_atk.boost(atk_key);
    let def_stage = stat_owner_def.boost(def_key);
    let ignore_atk_boost = mv.ignore_offensive || (crit && atk_stage < 0);
    let ignore_def_boost = mv.ignore_defensive || (crit && def_stage > 0);

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
        let crit_mod = mv.crit_modifier;
        base = trunc(base as f64 * crit_mod);
    }

    // -- the 16 rolls ------------------------------------------------------
    let mut dmg = [0i64; N_ROLLS];
    if mv.no_damage_variance {
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
        let is_stab = mv.force_stab
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
    if mv.will_crit {
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

/// Crit multiplier denominators by crit stage, gen 7+ (`critMult` in battle-actions.ts).
const CRIT_MULT: [i64; 5] = [0, 24, 8, 2, 1];

/// Exact crit probability, or 0 when the defender cannot be crit. Showdown clamps
/// `critRatio` to 0..4 and looks up `critMult = [0, 24, 8, 2, 1]`: a ratio of 0 never crits,
/// and a ratio of 4 always does.
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
        Some(entry) => entry.type_ids,
    }
}
