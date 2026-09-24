//! Ability and item damage modifiers, transcribed from `src/pokeuraou/effects.py`.
//!
//! Same entries, same event slots, same Showdown handler priorities. The Python file says
//! why each one is where it is; this one keeps the shape so the two can be read side by
//! side -- a table of `ModDef` per ability id, with the predicate as a plain function.

use crate::battler::VolatileFlags;
use crate::id::Id;
use crate::position::Types;
use crate::reg::{MoveFlags, F_BITE, F_CONTACT, F_PULSE, F_PUNCH, F_SLICING, F_SOUND};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    BasePower,
    Atk,
    Def,
    Damage,
    Stab,
    CritRatio,
    Accuracy,
    Speed,
}

pub struct Ctx<'a> {
    pub move_id: &'a str,
    pub move_type: Id,
    pub move_category: &'a str,
    pub move_base_power: i64,
    pub move_flags: MoveFlags,
    pub move_priority: i64,
    pub effectiveness: f64,
    pub type_mod: i64,
    pub is_crit: bool,
    pub is_spread: bool,
    pub has_secondary: bool,
    pub has_recoil: bool,

    pub attacker_species: Id,
    pub attacker_types: Types,
    pub attacker_ability: Id,
    pub attacker_item: Option<Id>,
    pub attacker_status: Option<Id>,
    pub attacker_volatiles: VolatileFlags,
    pub attacker_gender: Id,
    /// Supreme Overlord's count (`Battler::fallen`).
    pub attacker_fallen: i64,

    pub defender_species: Id,
    pub defender_types: Types,
    pub defender_ability: Id,
    pub defender_item: Option<Id>,
    pub defender_status: Option<Id>,
    pub defender_volatiles: VolatileFlags,
    pub defender_gender: Id,

    pub defender_at_full_hp: bool,
    pub attacker_in_pinch: bool,
    pub attacker_below_half: bool,

    pub weather: Option<Id>,
    pub terrain: Option<Id>,
    pub defender_side_conditions: &'a [Id],
    pub defender_ally_abilities: &'a [Id],
    pub field_abilities: &'a [Id],
    pub active_per_half: i64,
}

fn is(value: Option<Id>, name: &str) -> bool {
    matches!(value, Some(v) if v.as_str() == name)
}

pub struct ModDef {
    pub id: &'static str,
    pub slot: Slot,
    pub num: f64,
    pub den: f64,
    pub priority: i32,
    pub from_defender: bool,
    pub when: fn(&Ctx) -> bool,
}

const fn m(
    id: &'static str,
    slot: Slot,
    num: f64,
    den: f64,
    priority: i32,
    from_defender: bool,
    when: fn(&Ctx) -> bool,
) -> ModDef {
    ModDef { id, slot, num, den, priority, from_defender, when }
}

/// A table of modifiers with `'static` lifetime, so the per-ability tables can be looked
/// up and returned by reference rather than rebuilt on every hit.
macro_rules! table {
    ($($entry:expr),* $(,)?) => {{
        const T: &[ModDef] = &[$($entry),*];
        T
    }};
}

// -- predicates ------------------------------------------------------------

fn always(_: &Ctx) -> bool {
    true
}
fn is_physical(c: &Ctx) -> bool {
    c.move_category == "Physical"
}
fn is_special(c: &Ctx) -> bool {
    c.move_category == "Special"
}
fn sun(c: &Ctx) -> bool {
    is(c.weather, "sunnyday") || is(c.weather, "desolateland")
}
fn has_contact(c: &Ctx) -> bool {
    c.move_flags.has(F_CONTACT)
}
fn has_slicing(c: &Ctx) -> bool {
    c.move_flags.has(F_SLICING)
}
fn has_bite(c: &Ctx) -> bool {
    c.move_flags.has(F_BITE)
}
fn has_pulse(c: &Ctx) -> bool {
    c.move_flags.has(F_PULSE)
}
fn has_punch(c: &Ctx) -> bool {
    c.move_flags.has(F_PUNCH)
}
fn has_sound(c: &Ctx) -> bool {
    c.move_flags.has(F_SOUND)
}

fn technician(c: &Ctx) -> bool {
    c.move_base_power > 0 && c.move_base_power <= 60
}
fn has_secondary(c: &Ctx) -> bool {
    c.has_secondary
}
fn has_recoil(c: &Ctx) -> bool {
    c.has_recoil
}
fn type_steel(c: &Ctx) -> bool {
    c.move_type == "Steel"
}
fn type_dragon(c: &Ctx) -> bool {
    c.move_type == "Dragon"
}
fn type_electric(c: &Ctx) -> bool {
    c.move_type == "Electric"
}
fn type_rock(c: &Ctx) -> bool {
    c.move_type == "Rock"
}
fn type_water(c: &Ctx) -> bool {
    c.move_type == "Water"
}
fn type_fire(c: &Ctx) -> bool {
    c.move_type == "Fire"
}
fn type_ghost(c: &Ctx) -> bool {
    c.move_type == "Ghost"
}
fn type_normal(c: &Ctx) -> bool {
    c.move_type == "Normal"
}
fn rivalry_same(c: &Ctx) -> bool {
    (c.attacker_gender == "M" || c.attacker_gender == "F")
        && c.attacker_gender.as_str() == c.defender_gender.as_str()
}
fn rivalry_other(c: &Ctx) -> bool {
    (c.attacker_gender == "M" || c.attacker_gender == "F")
        && (c.defender_gender == "M" || c.defender_gender == "F")
        && c.attacker_gender.as_str() != c.defender_gender.as_str()
}
fn sandforce(c: &Ctx) -> bool {
    is(c.weather, "sandstorm")
        && matches!(c.move_type.as_str(), "Rock" | "Ground" | "Steel")
}
fn guts(c: &Ctx) -> bool {
    is_physical(c) && c.attacker_status.is_some()
}
fn toxicboost(c: &Ctx) -> bool {
    is_physical(c) && (is(c.attacker_status, "psn") || is(c.attacker_status, "tox"))
}
fn flareboost(c: &Ctx) -> bool {
    is_special(c) && is(c.attacker_status, "brn")
}
fn solarpower(c: &Ctx) -> bool {
    is_special(c) && sun(c)
}
fn defeatist(c: &Ctx) -> bool {
    c.attacker_below_half
}
fn pinch_grass(c: &Ctx) -> bool {
    c.attacker_in_pinch && c.move_type == "Grass"
}
fn pinch_fire(c: &Ctx) -> bool {
    c.attacker_in_pinch && c.move_type == "Fire"
}
fn pinch_water(c: &Ctx) -> bool {
    c.attacker_in_pinch && c.move_type == "Water"
}
fn pinch_bug(c: &Ctx) -> bool {
    c.attacker_in_pinch && c.move_type == "Bug"
}
fn flowergift_atk(c: &Ctx) -> bool {
    is_physical(c) && sun(c)
}
fn flowergift_def(c: &Ctx) -> bool {
    is_special(c) && sun(c)
}
fn thickfat(c: &Ctx) -> bool {
    matches!(c.move_type.as_str(), "Fire" | "Ice")
}
fn marvelscale(c: &Ctx) -> bool {
    is_physical(c) && c.defender_status.is_some()
}
fn grasspelt(c: &Ctx) -> bool {
    is_physical(c) && is(c.terrain, "grassyterrain")
}
fn resisted(c: &Ctx) -> bool {
    c.type_mod < 0
}
fn supereffective(c: &Ctx) -> bool {
    c.type_mod > 0
}
fn defender_full_hp(c: &Ctx) -> bool {
    c.defender_at_full_hp
}
fn is_pikachu(c: &Ctx) -> bool {
    c.attacker_species.starts_with("pikachu")
}
fn is_crit(c: &Ctx) -> bool {
    c.is_crit
}
fn fallen_1(c: &Ctx) -> bool {
    c.attacker_fallen == 1
}
fn fallen_2(c: &Ctx) -> bool {
    c.attacker_fallen == 2
}
fn fallen_3(c: &Ctx) -> bool {
    c.attacker_fallen == 3
}
fn fallen_4(c: &Ctx) -> bool {
    c.attacker_fallen == 4
}
fn fallen_5(c: &Ctx) -> bool {
    c.attacker_fallen >= 5
}

// -- tables ----------------------------------------------------------------

pub const AURA_FP: i64 = 5448;
pub const AURA_BROKEN_FP: i64 = 3072;
pub const AURA_BREAK_ABILITY: &str = "aurabreak";
pub const TYPE_CHANGE_BOOST_FP: i64 = 4915;
pub const SCREEN_FP_DOUBLES: i64 = 2732;
pub const SCREEN_FP_SINGLES: i64 = 2048;

/// (ability, move type) -- an aura reaches the whole field and applies once.
pub const AURA_ABILITIES: [(&str, &str); 2] = [("fairyaura", "Fairy"), ("darkaura", "Dark")];

pub fn ability_modifiers(ability: &str) -> &'static [ModDef] {
    match ability {
        // -- BasePower ---------------------------------------------------
        "technician" => table![m("technician", Slot::BasePower, 1.5, 1.0, 30, false, technician)],
        "toughclaws" => {
            table![m("toughclaws", Slot::BasePower, 5325.0, 4096.0, 21, false, has_contact)]
        }
        "sheerforce" => {
            table![m("sheerforce", Slot::BasePower, 5325.0, 4096.0, 21, false, has_secondary)]
        }
        "sharpness" => table![m("sharpness", Slot::BasePower, 1.5, 1.0, 19, false, has_slicing)],
        "strongjaw" => table![m("strongjaw", Slot::BasePower, 1.5, 1.0, 19, false, has_bite)],
        "megalauncher" => table![m("megalauncher", Slot::BasePower, 1.5, 1.0, 19, false, has_pulse)],
        "ironfist" => table![m("ironfist", Slot::BasePower, 1.2, 1.0, 23, false, has_punch)],
        "punkrock" => table![
            m("punkrock", Slot::BasePower, 1.3, 1.0, 7, false, has_sound),
            m("punkrock", Slot::Damage, 0.5, 1.0, 0, true, has_sound),
        ],
        "steelworker" => table![m("steelworker", Slot::Atk, 1.5, 1.0, 5, false, type_steel)],
        "dragonsmaw" => table![m("dragonsmaw", Slot::Atk, 1.5, 1.0, 5, false, type_dragon)],
        "transistor" => {
            table![m("transistor", Slot::Atk, 5325.0, 4096.0, 5, false, type_electric)]
        }
        "rockypayload" => table![m("rockypayload", Slot::Atk, 1.5, 1.0, 5, false, type_rock)],
        "waterbubble" => table![
            m("waterbubble", Slot::Atk, 2.0, 1.0, 5, false, type_water),
            m("waterbubble", Slot::Atk, 0.5, 1.0, 6, true, type_fire),
        ],
        "rivalry" => table![
            m("rivalry", Slot::BasePower, 1.25, 1.0, 8, false, rivalry_same),
            m("rivalry", Slot::BasePower, 0.75, 1.0, 8, false, rivalry_other),
        ],
        "reckless" => table![m("reckless", Slot::BasePower, 4915.0, 4096.0, 23, false, has_recoil)],
        "sandforce" => table![m("sandforce", Slot::BasePower, 5325.0, 4096.0, 21, false, sandforce)],
        // `onBasePowerPriority: 21`, `powMod[this.effectState.fallen]` of
        // `[4096, 4506, 4915, 5325, 5734, 6144]` (IKA-222). The count is taken at switch-in
        // (`resolve::supreme_overlord_start`) and read from `abilityState`.
        "supremeoverlord" => table![
            m("supremeoverlord", Slot::BasePower, 4506.0, 4096.0, 21, false, fallen_1),
            m("supremeoverlord", Slot::BasePower, 4915.0, 4096.0, 21, false, fallen_2),
            m("supremeoverlord", Slot::BasePower, 5325.0, 4096.0, 21, false, fallen_3),
            m("supremeoverlord", Slot::BasePower, 5734.0, 4096.0, 21, false, fallen_4),
            m("supremeoverlord", Slot::BasePower, 6144.0, 4096.0, 21, false, fallen_5),
        ],
        "fairyaura" => table![m("fairyaura", Slot::BasePower, 5448.0, 4096.0, 20, false, always)],
        "darkaura" => table![m("darkaura", Slot::BasePower, 5448.0, 4096.0, 20, false, always)],
        // -- Atk / SpA ---------------------------------------------------
        "hugepower" => table![m("hugepower", Slot::Atk, 2.0, 1.0, 5, false, is_physical)],
        "purepower" => table![m("purepower", Slot::Atk, 2.0, 1.0, 5, false, is_physical)],
        "hustle" => table![
            m("hustle", Slot::Atk, 1.5, 1.0, 5, false, is_physical),
            m("hustle", Slot::Accuracy, 0.8, 1.0, 0, false, is_physical),
        ],
        "guts" => table![m("guts", Slot::Atk, 1.5, 1.0, 5, false, guts)],
        "toxicboost" => table![m("toxicboost", Slot::Atk, 1.5, 1.0, 5, false, toxicboost)],
        "flareboost" => table![m("flareboost", Slot::Atk, 1.5, 1.0, 5, false, flareboost)],
        "solarpower" => table![m("solarpower", Slot::Atk, 1.5, 1.0, 5, false, solarpower)],
        "defeatist" => table![m("defeatist", Slot::Atk, 0.5, 1.0, 5, false, defeatist)],
        "overgrow" => table![m("overgrow", Slot::Atk, 1.5, 1.0, 5, false, pinch_grass)],
        "blaze" => table![m("blaze", Slot::Atk, 1.5, 1.0, 5, false, pinch_fire)],
        "torrent" => table![m("torrent", Slot::Atk, 1.5, 1.0, 5, false, pinch_water)],
        "swarm" => table![m("swarm", Slot::Atk, 1.5, 1.0, 5, false, pinch_bug)],
        "gorillatactics" => {
            table![m("gorillatactics", Slot::Atk, 1.5, 1.0, 5, false, is_physical)]
        }
        "flowergift" => table![
            m("flowergift", Slot::Atk, 1.5, 1.0, 5, false, flowergift_atk),
            m("flowergift", Slot::Def, 1.5, 1.0, 5, false, flowergift_def),
        ],
        "thickfat" => table![m("thickfat", Slot::Atk, 0.5, 1.0, 6, true, thickfat)],
        "heatproof" => table![m("heatproof", Slot::Atk, 0.5, 1.0, 6, true, type_fire)],
        "purifyingsalt" => table![m("purifyingsalt", Slot::Atk, 0.5, 1.0, 6, true, type_ghost)],
        // -- Def / SpD ---------------------------------------------------
        "marvelscale" => table![m("marvelscale", Slot::Def, 1.5, 1.0, 6, false, marvelscale)],
        "furcoat" => table![m("furcoat", Slot::Def, 2.0, 1.0, 6, false, is_physical)],
        "grasspelt" => table![m("grasspelt", Slot::Def, 1.5, 1.0, 6, false, grasspelt)],
        // -- ModifyDamage ------------------------------------------------
        "tintedlens" => table![m("tintedlens", Slot::Damage, 2.0, 1.0, 0, false, resisted)],
        // `onModifyDamage`: 1.5x on a crit (IKA-222). The crit branch is the calculator's
        // `crit = true` call, so the crit's own roll is the one Showdown makes first.
        "sniper" => table![m("sniper", Slot::Damage, 1.5, 1.0, 0, false, is_crit)],
        "neuroforce" => {
            table![m("neuroforce", Slot::Damage, 5120.0, 4096.0, 0, false, supereffective)]
        }
        "solidrock" => table![m("solidrock", Slot::Damage, 0.75, 1.0, 0, true, supereffective)],
        "filter" => table![m("filter", Slot::Damage, 0.75, 1.0, 0, true, supereffective)],
        "prismarmor" => table![m("prismarmor", Slot::Damage, 0.75, 1.0, 0, true, supereffective)],
        "multiscale" => table![m("multiscale", Slot::Damage, 0.5, 1.0, 0, true, defender_full_hp)],
        "shadowshield" => {
            table![m("shadowshield", Slot::Damage, 0.5, 1.0, 0, true, defender_full_hp)]
        }
        "icescales" => table![m("icescales", Slot::Damage, 0.5, 1.0, 0, true, is_special)],
        "fluffy" => table![
            m("fluffy", Slot::Damage, 0.5, 1.0, 0, true, has_contact),
            m("fluffy", Slot::Damage, 2.0, 1.0, 0, true, type_fire),
        ],
        // Python's `auraguard` (IKA-203): Fluffy's contact half, nothing else.
        "auraguard" => table![m("auraguard", Slot::Damage, 0.5, 1.0, 0, true, has_contact)],
        _ => table![],
    }
}

pub fn item_modifiers(item: &str) -> &'static [ModDef] {
    match item {
        "lifeorb" => table![m("lifeorb", Slot::Damage, 5324.0, 4096.0, 0, false, always)],
        "expertbelt" => {
            table![m("expertbelt", Slot::Damage, 4915.0, 4096.0, 0, false, supereffective)]
        }
        "muscleband" => {
            table![m("muscleband", Slot::BasePower, 4505.0, 4096.0, 0, false, is_physical)]
        }
        "wiseglasses" => {
            table![m("wiseglasses", Slot::BasePower, 4505.0, 4096.0, 0, false, is_special)]
        }
        "normalgem" => {
            table![m("normalgem", Slot::BasePower, 5325.0, 4096.0, 0, false, type_normal)]
        }
        "lightball" => table![m("lightball", Slot::Atk, 2.0, 1.0, 5, false, is_pikachu)],
        "choicescarf" => table![m("choicescarf", Slot::Speed, 1.5, 1.0, 0, false, always)],
        "ironball" => table![m("ironball", Slot::Speed, 0.5, 1.0, 0, false, always)],
        "widelens" => table![m("widelens", Slot::Accuracy, 4505.0, 4096.0, 0, false, always)],
        "zoomlens" => table![m("zoomlens", Slot::Accuracy, 1.2, 1.0, 0, false, always)],
        "brightpowder" => table![m("brightpowder", Slot::Accuracy, 0.9, 1.0, 0, true, always)],
        "scopelens" => table![m("scopelens", Slot::CritRatio, 1.0, 1.0, 0, false, always)],
        "leek" => table![m("leek", Slot::CritRatio, 2.0, 1.0, 0, false, always)],
        _ => table![],
    }
}

pub fn type_changing_ability(ability: &str) -> Option<&'static str> {
    match ability {
        "aerilate" => Some("Flying"),
        "pixilate" => Some("Fairy"),
        "galvanize" => Some("Electric"),
        "refrigerate" => Some("Ice"),
        "normalize" => Some("Normal"),
        "liquidvoice" => Some("Water"),
        _ => None,
    }
}

pub fn type_immunity_ability(ability: &str) -> Option<&'static str> {
    match ability {
        "levitate" | "eartheater" => Some("Ground"),
        "flashfire" | "wellbakedbody" => Some("Fire"),
        "waterabsorb" | "dryskin" | "stormdrain" => Some("Water"),
        "voltabsorb" | "lightningrod" | "motordrive" => Some("Electric"),
        "sapsipper" => Some("Grass"),
        "windrider" => Some("Flying"),
        _ => None,
    }
}

/// The Ruin ability an `onAny` stat event meets (IKA-222): Tablets the attacking event of a
/// physical move, Vessel of a special one, Sword the defending event on Def, Beads on SpD.
///
/// The attacking event is the category's whatever stat the move read (`getDamage` resets
/// `attackStat` before `runEvent('Modify' + statTable[attackStat], source, ...)`), so Body
/// Press's Def meets Tablets, and it is the user's, so Foul Play's borrowed Atk does too.
pub fn ruin_of(attacking: bool, stat: &str) -> Option<&'static str> {
    match (attacking, stat) {
        (true, "atk") => Some("tabletsofruin"),
        (true, "spa") => Some("vesselofruin"),
        (false, "def") => Some("swordofruin"),
        (false, "spd") => Some("beadsofruin"),
        _ => None,
    }
}

/// Whether any of the active abilities is a Ruin ability: the one check every hit pays.
#[inline]
pub fn any_ruin(field_abilities: &[Id]) -> bool {
    field_abilities.iter().any(|a| a.as_str().ends_with("ofruin"))
}

/// A Ruin ability's 0.75x lands on the event's Pokemon unless it has the same ability
/// (`if (source.hasAbility('Tablets of Ruin')) return;`); any other active holder, on
/// either side, applies it once (`move.ruinedAtk`).
pub fn ruined(field_abilities: &[Id], owner: Id, ruin: &str) -> bool {
    owner != ruin && field_abilities.iter().any(|a| *a == ruin)
}

pub fn is_mold_breaker(ability: &str) -> bool {
    matches!(ability, "moldbreaker" | "teravolt" | "turboblaze" | "myceliummight")
}

pub fn is_retyping(ability: &str) -> bool {
    matches!(ability, "protean" | "libero")
}

pub fn pierces_ghost(ability: &str) -> bool {
    matches!(ability, "scrappy" | "mindseye")
}

pub fn suppresses_weather(ability: &str) -> bool {
    matches!(ability, "cloudnine" | "airlock")
}

pub fn type_boost_item(item: &str) -> Option<&'static str> {
    match item {
        "blackbelt" => Some("Fighting"),
        "blackglasses" => Some("Dark"),
        "charcoal" => Some("Fire"),
        "dragonfang" => Some("Dragon"),
        "fairyfeather" => Some("Fairy"),
        "hardstone" => Some("Rock"),
        "magnet" => Some("Electric"),
        "metalcoat" => Some("Steel"),
        "miracleseed" => Some("Grass"),
        "mysticwater" => Some("Water"),
        "nevermeltice" => Some("Ice"),
        "poisonbarb" => Some("Poison"),
        "sharpbeak" => Some("Flying"),
        "silkscarf" => Some("Normal"),
        "silverpowder" => Some("Bug"),
        "softsand" => Some("Ground"),
        "spelltag" => Some("Ghost"),
        "twistedspoon" => Some("Psychic"),
        _ => None,
    }
}

pub fn resist_berry(item: &str) -> Option<&'static str> {
    match item {
        "babiriberry" => Some("Steel"),
        "chartiberry" => Some("Rock"),
        "chilanberry" => Some("Normal"),
        "chopleberry" => Some("Fighting"),
        "cobaberry" => Some("Flying"),
        "colburberry" => Some("Dark"),
        "habanberry" => Some("Dragon"),
        "kasibberry" => Some("Ghost"),
        "kebiaberry" => Some("Poison"),
        "occaberry" => Some("Fire"),
        "passhoberry" => Some("Water"),
        "payapaberry" => Some("Psychic"),
        "rindoberry" => Some("Grass"),
        "roseliberry" => Some("Fairy"),
        "shucaberry" => Some("Ground"),
        "tangaberry" => Some("Bug"),
        "wacanberry" => Some("Electric"),
        "yacheberry" => Some("Ice"),
        _ => None,
    }
}

/// Screens, with the move category each one stops.
pub const SCREEN_CONDITIONS: [(&str, &str); 3] =
    [("reflect", "Physical"), ("lightscreen", "Special"), ("auroraveil", "both")];

pub fn survives_at_one_item(item: &str) -> bool {
    item == "focussash"
}

pub fn survives_at_one_ability(ability: &str) -> bool {
    ability == "sturdy"
}

/// Items that give a *chance* to survive a lethal hit, as (numerator, denominator). Focus
/// Band is deliberately not a survive-at-one item like the Sash: it is `randomChance(1, 10)`
/// from any HP and is not consumed, so treating it as a certain full-HP save is wrong in
/// both directions.
pub fn survive_chance_item(item: &str) -> Option<(i64, i64)> {
    if item == "focusband" { Some((1, 10)) } else { None }
}

pub fn type_boost_item_fp() -> i64 {
    4915
}
