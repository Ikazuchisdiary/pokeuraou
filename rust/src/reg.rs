//! The regulation, read from the same `configs/regulations/*.json` the Python loads.
//!
//! This is the part that makes a port cheap: the dex was already extracted into a plain
//! JSON document for the Python side, so nothing here has to touch Showdown's TypeScript.

use crate::id::{FnvBuild, Id, ID_CAPACITY};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Showdown's id normalisation: lowercase, keep only [a-z0-9].
pub fn to_id(name: &str) -> String {
    name.chars().filter(|c| c.is_ascii_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

pub struct Species {
    pub id: String,
    pub types: Vec<String>,
    pub weightkg: f64,
    /// hp, atk, def, spa, spd, spe.
    pub base_stats: [i64; 6],
    pub abilities: Vec<String>,
}

/// A nature's numerators per stat: 110 boosted, 90 hindered, 100 otherwise.
pub struct Nature {
    pub numerators: [i64; 6],
}

pub struct Move {
    pub id: String,
    pub mtype: String,
    pub category: String,
    pub base_power: i64,
    /// None means "never misses", as Showdown's `accuracy: true`.
    pub accuracy: Option<i64>,
    pub priority: i64,
    pub target: String,
    pub crit_ratio: i64,
    pub flags: HashSet<String>,
    pub raw: Value,
}

impl Move {
    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.contains(flag)
    }
    pub fn raw_bool(&self, key: &str) -> bool {
        match self.raw.get(key) {
            None | Some(Value::Null) => false,
            Some(Value::Bool(b)) => *b,
            Some(Value::Array(a)) => !a.is_empty(),
            Some(Value::Object(o)) => !o.is_empty(),
            Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
            Some(Value::String(s)) => !s.is_empty(),
        }
    }
    pub fn raw_str(&self, key: &str) -> Option<&str> {
        self.raw.get(key).and_then(|v| v.as_str()).filter(|s| !s.is_empty())
    }
    pub fn raw_f64(&self, key: &str, default: f64) -> f64 {
        self.raw.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }
}

pub struct Reg {
    pub format_id: String,
    pub stat_ids: Vec<String>,
    pub species: HashMap<String, Species>,
    pub moves: HashMap<String, Move>,
    pub natures: HashMap<String, Nature>,
    pub typechart: HashMap<String, HashMap<String, f64>>,
    /// (speciesId, itemId) pairs that are a mega pairing, for `item_is_removable`.
    pub mega_by_species: HashSet<(String, String)>,
    pub level: i64,
    pub active_per_side: usize,
    pub picked_team_size: usize,
    /// Items that lock their holder into one move, from the dump's `isChoice`.
    pub choice_items: HashSet<String>,
    /// (speciesId, itemId) -> the mega forme it becomes.
    pub mega_targets: HashMap<(String, String), String>,
    /// The same pairs as inline ids, so the encoder can ask of every Pokemon of every leaf
    /// whether it holds its own stone without building two `String`s to look it up.
    pub mega_holders: HashSet<(Id, Id), FnvBuild>,
    /// Effect name -> the types immune to it, from the dump's `effectImmunities`.
    pub effect_immunities: HashMap<String, HashSet<String>>,
    /// Every id in the dump, sorted -- the encoder's vocabulary is an offset into these,
    /// and `encode.build_vocabulary` sorts for exactly the reason this does: a model's
    /// weights are meaningless the moment the same integer means a different Pokemon.
    pub species_ids: Vec<String>,
    pub ability_ids: Vec<String>,
    pub item_ids: Vec<String>,
    pub move_ids: Vec<String>,
    /// The types any species has, sorted. Index into this, not into `types`.
    pub species_types: Vec<String>,
}

impl Reg {
    pub fn load(path: &str) -> Result<Reg, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let doc: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;

        let mut species = HashMap::new();
        for entry in doc["species"].as_array().ok_or("species is not a list")? {
            let id = entry["id"].as_str().unwrap_or_default().to_string();
            let mut base_stats = [0i64; 6];
            if let Some(stats) = entry["baseStats"].as_object() {
                for (name, value) in stats {
                    if let Some(index) = crate::position::stat_index(name) {
                        base_stats[index] = value.as_i64().unwrap_or(0);
                    }
                }
            }
            species.insert(
                id.clone(),
                Species {
                    id,
                    types: entry["types"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|t| t.as_str()).map(String::from).collect())
                        .unwrap_or_default(),
                    weightkg: entry["weightkg"].as_f64().unwrap_or(0.0),
                    base_stats,
                    abilities: entry["abilities"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|t| t.as_str()).map(String::from).collect())
                        .unwrap_or_default(),
                },
            );
        }

        let mut moves = HashMap::new();
        for entry in doc["moves"].as_array().ok_or("moves is not a list")? {
            let id = entry["id"].as_str().unwrap_or_default().to_string();
            let flags = entry["flags"]
                .as_object()
                .map(|o| o.keys().cloned().collect::<HashSet<String>>())
                .unwrap_or_default();
            moves.insert(
                id.clone(),
                Move {
                    id,
                    mtype: entry["type"].as_str().unwrap_or("???").to_string(),
                    category: entry["category"].as_str().unwrap_or("Status").to_string(),
                    base_power: entry["basePower"].as_i64().unwrap_or(0),
                    accuracy: entry["accuracy"].as_i64(),
                    priority: entry["priority"].as_i64().unwrap_or(0),
                    target: entry["target"].as_str().unwrap_or("normal").to_string(),
                    crit_ratio: entry["critRatio"].as_i64().unwrap_or(0),
                    flags,
                    raw: entry.clone(),
                },
            );
        }

        let mut typechart = HashMap::new();
        if let Some(chart) = doc["typechart"].as_object() {
            for (attacking, row) in chart {
                let mut inner = HashMap::new();
                if let Some(obj) = row.as_object() {
                    for (defending, mult) in obj {
                        inner.insert(defending.clone(), mult.as_f64().unwrap_or(1.0));
                    }
                }
                typechart.insert(attacking.clone(), inner);
            }
        }

        let mut mega_by_species = HashSet::new();
        let mut mega_targets = HashMap::new();
        let mut mega_holders: HashSet<(Id, Id), FnvBuild> = HashSet::default();
        if let Some(map) = doc["megaMap"].as_object() {
            for (item_id, table) in map {
                if let Some(obj) = table.as_object() {
                    for (from_species, to_species) in obj {
                        // An id too long for `Id` cannot be on any Pokemon, so it cannot be
                        // asked about either; skipping it keeps `Id::new` from panicking.
                        if from_species.len() <= ID_CAPACITY && item_id.len() <= ID_CAPACITY {
                            mega_holders.insert((Id::new(from_species), Id::new(item_id)));
                        }
                        mega_by_species.insert((from_species.clone(), item_id.clone()));
                        mega_targets.insert(
                            (from_species.clone(), item_id.clone()),
                            to_species.as_str().unwrap_or_default().to_string(),
                        );
                    }
                }
            }
        }

        let mut choice_items = HashSet::new();
        if let Some(list) = doc["items"].as_array() {
            for entry in list {
                if entry["isChoice"].as_bool().unwrap_or(false) {
                    choice_items.insert(entry["id"].as_str().unwrap_or_default().to_string());
                }
            }
        }

        let mut effect_immunities = HashMap::new();
        if let Some(map) = doc["effectImmunities"].as_object() {
            for (effect, types) in map {
                let set: HashSet<String> = types
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str()).map(String::from).collect())
                    .unwrap_or_default();
                effect_immunities.insert(effect.clone(), set);
            }
        }

        let mut natures = HashMap::new();
        if let Some(list) = doc["natures"].as_array() {
            for entry in list {
                let mut numerators = [100i64; 6];
                if let Some(plus) = entry["plus"].as_str() {
                    if let Some(index) = crate::position::stat_index(plus) {
                        numerators[index] = 110;
                    }
                }
                if let Some(minus) = entry["minus"].as_str() {
                    if let Some(index) = crate::position::stat_index(minus) {
                        numerators[index] = 90;
                    }
                }
                natures.insert(
                    entry["name"].as_str().unwrap_or_default().to_string(),
                    Nature { numerators },
                );
            }
        }

        if doc["meta"]["usesLevelClauseMod"].as_bool().unwrap_or(false) {
            return Err("this port implements the level-50 closed form only".into());
        }

        let mut species_ids: Vec<String> = species.keys().cloned().collect();
        species_ids.sort();
        let mut move_ids: Vec<String> = moves.keys().cloned().collect();
        move_ids.sort();
        let mut ability_ids: Vec<String> = doc["abilities"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|e| e["id"].as_str())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();
        ability_ids.sort();
        let mut item_ids: Vec<String> = doc["items"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|e| e["id"].as_str())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();
        item_ids.sort();
        let mut species_types: Vec<String> = {
            let mut seen: HashSet<String> = HashSet::new();
            for entry in species.values() {
                for kind in &entry.types {
                    seen.insert(kind.clone());
                }
            }
            seen.into_iter().collect()
        };
        species_types.sort();

        Ok(Reg {
            species_ids,
            ability_ids,
            item_ids,
            move_ids,
            species_types,
            natures,
            level: doc["meta"]["level"].as_i64().unwrap_or(50),
            active_per_side: doc["meta"]["activePerSide"].as_u64().unwrap_or(2) as usize,
            picked_team_size: doc["meta"]["pickedTeamSize"].as_u64().unwrap_or(4) as usize,
            format_id: doc["meta"]["formatId"].as_str().unwrap_or_default().to_string(),
            stat_ids: doc["statIds"]
                .as_array()
                .map(|a| a.iter().filter_map(|s| s.as_str()).map(String::from).collect())
                .unwrap_or_else(|| {
                    ["hp", "atk", "def", "spa", "spd", "spe"].iter().map(|s| s.to_string()).collect()
                }),
            species,
            moves,
            typechart,
            mega_by_species,
            mega_targets,
            mega_holders,
            choice_items,
            effect_immunities,
        })
    }

    /// `stats.stats_from_sp` for one spread, in the level-50 closed form Champions uses:
    ///
    ///     HP     = base + SP + 75
    ///     others = trunc(trunc16((base + SP + 20) * natureNumerator) / 100)
    pub fn stats_from_sp(
        &self,
        species: &Species,
        nature: &str,
        sp: &[i64; 6],
    ) -> Result<[i64; 6], String> {
        let nature = self
            .natures
            .get(nature)
            .ok_or_else(|| format!("unknown nature {nature}"))?;
        let mut out = [0i64; 6];
        out[0] = species.base_stats[0] + sp[0] + 75;
        for index in 1..6 {
            let raw = (species.base_stats[index] + sp[index] + 20) * nature.numerators[index];
            out[index] = (raw % (1 << 16)) / 100;
        }
        Ok(out)
    }

    pub fn type_effectiveness(&self, attacking: &str, defending: &crate::position::Types) -> f64 {
        match self.typechart.get(attacking) {
            None => 1.0,
            Some(row) => defending
                .as_slice()
                .iter()
                .map(|t| row.get(t.as_str()).copied().unwrap_or(1.0))
                .product(),
        }
    }

    /// `Regulation.mega_target(species, item) is not None`: this Pokemon holds the stone
    /// that megas it. What the encoder's `can_mega` and `mega_available` read, instead of
    /// `Side.mega_capable_slots` -- a party slot number is not an identity, and resolve
    /// renumbers slots on every switch (IKA-121).
    pub fn holds_mega_stone(&self, species: Id, item: Option<Id>) -> bool {
        match item {
            None => false,
            Some(item) => self.mega_holders.contains(&(species, item)),
        }
    }

    pub fn item_is_removable(&self, species_id: &str, item_id: Option<&str>) -> bool {
        match item_id {
            None => false,
            Some(item) => !self
                .mega_by_species
                .contains(&(species_id.to_string(), item.to_string())),
        }
    }
}
