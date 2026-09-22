//! The calculator's view of one Pokemon and the field.
//!
//! Scalar, not batched: the resolver works on a fully-known position, and the CLI's matrix
//! path was measured at 100% n=1, so the particle axis the Python carries is pure overhead
//! here. `view.battler` is 10% of a generation run in Python, mostly building dicts and
//! (1,6) arrays; this type is `Copy` and allocates nothing, so building one is a memcpy.
//!
//! The volatiles are a bitmask rather than a set, because the damage and speed layers only
//! ever ask about nine of them by name. Everything else about a Pokemon's volatiles is
//! read from the `Pokemon` itself, where the full list lives.

use crate::id::Id;
use crate::position::{boost_index, Pokemon, Types};
use crate::reg::Reg;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

/// How many Battlers the resolver has built. It rebuilds them whenever the queue is
/// re-sorted and whenever damage is calculated, and the stats come from the SP spread
/// every time, so the count is what says whether that is worth avoiding.
pub static BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Number of damage rolls Showdown uses: 100%, 99%, ..., 85%.
pub const N_ROLLS: usize = 16;

/// The volatiles the damage and speed layers consult by name.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct VolatileFlags(u16);

pub const V_HELPING_HAND: u16 = 1 << 0;
pub const V_CHARGE: u16 = 1 << 1;
pub const V_SMACK_DOWN: u16 = 1 << 2;
pub const V_INGRAIN: u16 = 1 << 3;
pub const V_MAGNET_RISE: u16 = 1 << 4;
pub const V_TELEKINESIS: u16 = 1 << 5;
pub const V_FOCUS_ENERGY: u16 = 1 << 6;
pub const V_DRAGON_CHEER: u16 = 1 << 7;
pub const V_UNBURDEN: u16 = 1 << 8;

impl VolatileFlags {
    pub fn add(&mut self, name: &str) {
        self.0 |= match name {
            "helpinghand" => V_HELPING_HAND,
            "charge" => V_CHARGE,
            "smackdown" => V_SMACK_DOWN,
            "ingrain" => V_INGRAIN,
            "magnetrise" => V_MAGNET_RISE,
            "telekinesis" => V_TELEKINESIS,
            "focusenergy" => V_FOCUS_ENERGY,
            "dragoncheer" => V_DRAGON_CHEER,
            "unburden" => V_UNBURDEN,
            _ => 0,
        };
    }

    #[inline]
    pub fn has(&self, bit: u16) -> bool {
        self.0 & bit != 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Battler {
    pub species: Id,
    pub types: Types,
    pub ability: Id,
    pub item: Option<Id>,
    pub level: i64,
    /// hp, atk, def, spa, spd, spe.
    pub stats: [i64; 6],
    pub hp: i64,
    pub maxhp: i64,
    pub boosts: [i8; 7],
    pub status: Option<Id>,
    pub volatiles: VolatileFlags,
    pub gender: Id,
    /// Protean and Libero record here that they have already retyped their user.
    pub protean_fired: bool,
}

impl Battler {
    /// `view.battler` for a known spread: stats from the SP, HP from the position.
    pub fn from_pokemon(reg: &Reg, mon: &Pokemon) -> Result<Battler, String> {
        BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let species = reg
            .species
            .get(mon.species.as_str())
            .ok_or_else(|| format!("unknown species {}", mon.species))?;
        let stats = if mon.stats_override.is_some() && (mon.transformed || mon.sp.is_none()) {
            mon.stats_override.unwrap()
        } else {
            match mon.sp {
                None => {
                    return Err(format!("{} has a hidden SP spread", mon.species));
                }
                Some(sp) => reg.stats_from_sp(species, mon.nature.as_str(), &sp)?,
            }
        };
        let mut volatiles = VolatileFlags::default();
        for effect in &mon.volatiles {
            volatiles.add(effect.id.as_str());
        }
        Ok(Battler {
            species: mon.species,
            types: if mon.types.is_empty() { species_types(species) } else { mon.types },
            ability: mon.ability,
            item: mon.item,
            level: mon.level,
            stats,
            hp: mon.hp,
            maxhp: mon.maxhp,
            boosts: mon.boosts,
            status: mon.status,
            volatiles,
            gender: mon.gender,
            protean_fired: mon
                .ability_state
                .get("protean")
                .map(|v| v != &serde_json::Value::Bool(false) && !v.is_null())
                .unwrap_or(false),
        })
    }

    pub fn stat(&self, key: &str, ignore_boost: bool) -> i64 {
        let index = crate::position::stat_index(key).unwrap_or(5);
        let raw = self.stats[index];
        if ignore_boost {
            return raw;
        }
        match boost_index(key).map(|i| self.boosts[i]).unwrap_or(0) {
            0 => raw,
            stage => crate::fixedpoint::apply_boost(raw, stage as i64),
        }
    }

    pub fn boost(&self, key: &str) -> i64 {
        boost_index(key).map(|i| self.boosts[i] as i64).unwrap_or(0)
    }

    pub fn at_full_hp(&self) -> bool {
        self.hp >= self.maxhp
    }
}

fn species_types(species: &crate::reg::Species) -> Types {
    let ids: Vec<Id> = species.types.iter().map(|t| Id::new(t)).collect();
    Types::from_slice(&ids)
}

/// The field as the calculator sees it. Owned ids, because the resolver rebuilds it
/// whenever the field changes and the active abilities are at most four.
#[derive(Clone, Debug, Default)]
pub struct FieldState {
    pub weather: Option<Id>,
    pub terrain: Option<Id>,
    pub pseudo_weather: Vec<Id>,
    pub side_conditions: [Vec<Id>; 2],
    pub active_abilities: [Vec<Id>; 2],
    pub active_per_half: i64,
}

impl FieldState {
    pub fn trick_room(&self) -> bool {
        self.pseudo_weather.iter().any(|p| p.as_str() == "trickroom")
    }

    pub fn has_side_condition(&self, side: usize, name: &str) -> bool {
        self.side_conditions[side].iter().any(|c| c.as_str() == name)
    }
}

#[derive(Debug)]
pub struct DamageResult {
    pub rolls: [i64; N_ROLLS],
    pub effectiveness: f64,
    pub type_mod: i64,
    pub immune: bool,
    /// What the calculator could not account for, in Python's own wording. Usually empty,
    /// and an empty `Vec` does not allocate.
    pub unmodelled: Vec<String>,
}

// ---------------------------------------------------------------------------
// Deserialisation, for the damage-case fixtures only
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct BattlerCase {
    pub species: String,
    pub types: Vec<String>,
    pub ability: String,
    pub item: Option<String>,
    pub level: i64,
    pub stats: [i64; 6],
    pub hp: i64,
    pub maxhp: i64,
    #[serde(default)]
    pub boosts: HashMap<String, i64>,
    pub status: Option<String>,
    #[serde(default)]
    pub volatiles: HashSet<String>,
    pub gender: String,
    #[serde(default)]
    pub protean_fired: bool,
}

impl From<&BattlerCase> for Battler {
    fn from(case: &BattlerCase) -> Battler {
        let mut boosts = [0i8; 7];
        for (name, value) in &case.boosts {
            if let Some(index) = boost_index(name) {
                boosts[index] = *value as i8;
            }
        }
        let mut volatiles = VolatileFlags::default();
        for name in &case.volatiles {
            volatiles.add(name);
        }
        let types: Vec<Id> = case.types.iter().map(|t| Id::new(t)).collect();
        Battler {
            species: Id::new(&case.species),
            types: Types::from_slice(&types),
            ability: Id::new(&case.ability),
            item: case.item.as_deref().map(Id::new),
            level: case.level,
            stats: case.stats,
            hp: case.hp,
            maxhp: case.maxhp,
            boosts,
            status: case.status.as_deref().map(Id::new),
            volatiles,
            gender: Id::new(&case.gender),
            protean_fired: case.protean_fired,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct FieldCase {
    pub weather: Option<String>,
    pub terrain: Option<String>,
    #[serde(default)]
    pub pseudo_weather: Vec<String>,
    pub side_conditions: [Vec<String>; 2],
    pub active_abilities: [Vec<String>; 2],
    pub active_per_half: i64,
}

impl From<&FieldCase> for FieldState {
    fn from(case: &FieldCase) -> FieldState {
        let ids = |items: &Vec<String>| -> Vec<Id> { items.iter().map(|s| Id::new(s)).collect() };
        FieldState {
            weather: case.weather.as_deref().map(Id::new),
            terrain: case.terrain.as_deref().map(Id::new),
            pseudo_weather: ids(&case.pseudo_weather),
            side_conditions: [ids(&case.side_conditions[0]), ids(&case.side_conditions[1])],
            active_abilities: [ids(&case.active_abilities[0]), ids(&case.active_abilities[1])],
            active_per_half: case.active_per_half,
        }
    }
}
