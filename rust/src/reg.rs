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
    /// `types` as inline ids, built once here so a Battler or a damage call that needs a
    /// species' own types copies them instead of collecting a `Vec` (IKA-101).
    pub type_ids: crate::position::Types,
}

/// A nature's numerators per stat: 110 boosted, 90 hindered, 100 otherwise.
pub struct Nature {
    pub numerators: [i64; 6],
}

/// A move's `category`. Showdown has exactly these three; anything else fails the load
/// rather than comparing unequal to all of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    Physical,
    Special,
    Status,
}

impl Category {
    fn parse(name: &str) -> Option<Category> {
        match name {
            "Physical" => Some(Category::Physical),
            "Special" => Some(Category::Special),
            "Status" => Some(Category::Status),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Category::Physical => "Physical",
            Category::Special => "Special",
            Category::Status => "Status",
        }
    }
}

impl PartialEq<&str> for Category {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// The move flags anything in this port asks about, as bits. A flag the dump has and no
/// code reads is not kept; a flag code reads has to be named here, because `has_flag`
/// takes a bit and not a string (IKA-101: this was a SipHash `HashSet<String>`).
pub const F_BITE: u32 = 1 << 0;
pub const F_BULLET: u32 = 1 << 1;
pub const F_CONTACT: u32 = 1 << 2;
pub const F_FAILENCORE: u32 = 1 << 3;
pub const F_HEAL: u32 = 1 << 4;
pub const F_INFILTRATES: u32 = 1 << 5;
pub const F_POWDER: u32 = 1 << 6;
pub const F_PROTECT: u32 = 1 << 7;
pub const F_PULSE: u32 = 1 << 8;
pub const F_PUNCH: u32 = 1 << 9;
pub const F_SLICING: u32 = 1 << 10;
pub const F_SOUND: u32 = 1 << 11;
pub const F_DEFROST: u32 = 1 << 12;
/// Past a Substitute (IKA-180): the sound moves and the rest of Showdown's `bypasssub`.
pub const F_BYPASSSUB: u32 = 1 << 13;

const FLAG_NAMES: [(&str, u32); 14] = [
    ("bite", F_BITE),
    ("bullet", F_BULLET),
    ("contact", F_CONTACT),
    ("failencore", F_FAILENCORE),
    ("heal", F_HEAL),
    ("infiltrates", F_INFILTRATES),
    ("powder", F_POWDER),
    ("protect", F_PROTECT),
    ("pulse", F_PULSE),
    ("punch", F_PUNCH),
    ("slicing", F_SLICING),
    ("sound", F_SOUND),
    ("defrost", F_DEFROST),
    ("bypasssub", F_BYPASSSUB),
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MoveFlags(u32);

impl MoveFlags {
    #[inline]
    pub fn has(&self, bit: u32) -> bool {
        self.0 & bit != 0
    }
}

/// `Move.raw_bool` as it was: Python's truthiness of one key of the dump entry.
fn raw_bool(entry: &Value, key: &str) -> bool {
    match entry.get(key) {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::String(s)) => !s.is_empty(),
    }
}

/// `Move.raw_str` as it was: a non-empty string, or nothing.
fn raw_str(entry: &Value, key: &str) -> Option<String> {
    entry.get(key).and_then(|v| v.as_str()).filter(|s| !s.is_empty()).map(String::from)
}

/// The dump's move entry, with every key the port reads turned into a field once at load.
///
/// IKA-101: these were read out of `raw` (a `serde_json::Value`, whose map is a BTreeMap)
/// by string key -- a dozen times per damage calculation. Each field says which key it is
/// and keeps the reader's own semantics (`raw_bool`'s truthiness, `raw_str`'s "non-empty
/// string"), so a move reads the same whichever way it is asked. `raw` stays for the
/// declarative status-move keys, which are read once per use of a status move.
pub struct Move {
    pub id: String,
    pub mtype: Id,
    pub category: Category,
    pub base_power: i64,
    /// None means "never misses", as Showdown's `accuracy: true`.
    pub accuracy: Option<i64>,
    pub priority: i64,
    pub target: String,
    pub crit_ratio: i64,
    pub flags: MoveFlags,
    pub raw: Value,

    /// `raw_bool("secondaries")`.
    pub has_secondaries: bool,
    /// `raw_bool("recoil")`.
    pub has_recoil: bool,
    /// `raw_bool("hasCrashDamage")`.
    pub has_crash_damage: bool,
    /// `raw_str("overrideOffensiveStat")`.
    pub override_offensive_stat: Option<String>,
    /// `raw_str("overrideDefensiveStat")`.
    pub override_defensive_stat: Option<String>,
    /// `raw_str("overrideOffensivePokemon") == Some("target")`.
    pub offensive_stat_from_target: bool,
    /// `raw_str("overrideDefensivePokemon") == Some("source")`.
    pub defensive_stat_from_source: bool,
    /// `raw_bool("ignoreOffensive")`.
    pub ignore_offensive: bool,
    /// `raw_bool("ignoreDefensive")`.
    pub ignore_defensive: bool,
    /// `raw_f64("critModifier", 1.5)`.
    pub crit_modifier: f64,
    /// `raw_bool("noDamageVariance")`.
    pub no_damage_variance: bool,
    /// `raw_bool("forceSTAB")`.
    pub force_stab: bool,
    /// `raw_bool("willCrit")`.
    pub will_crit: bool,
    /// `ignoreImmunity: true`.
    pub ignore_immunity_all: bool,
    /// `ignoreImmunity: {Type: v}` -- the types whose `v` is anything but `false`.
    pub ignore_immunity_types: Vec<Id>,
    /// `raw_bool("breaksProtect")`.
    pub breaks_protect: bool,
    /// `raw_bool("alwaysHit")`.
    pub always_hit: bool,
    /// `raw_bool("ignoreEvasion")`.
    pub ignore_evasion: bool,
    /// `raw_bool("forceSwitch")`.
    pub force_switch: bool,
    /// `raw_bool("selfSwitch")`.
    pub self_switch: bool,
    /// `raw_bool("stallingMove")`.
    pub stalling_move: bool,
    /// `raw_str("volatileStatus")`.
    pub volatile_status: Option<String>,
    /// `raw_str("status")`.
    pub status: Option<String>,
    /// `raw_str("sideCondition")`.
    pub side_condition: Option<String>,
    /// `raw_str("weather")`, as the dump spells it (not an id: "RainDance").
    pub weather: Option<String>,
    /// `raw_str("terrain")`.
    pub terrain: Option<String>,
    /// `raw_str("pseudoWeather")`.
    pub pseudo_weather: Option<String>,
    /// `raw.get("multihit")`.
    pub multihit: Option<Value>,
    /// `raw.get("secondaries")` when it is a list, else empty.
    pub secondaries: Vec<Value>,
    /// `raw.get("drain")`.
    pub drain: Option<Value>,
    /// `raw.get("recoil")`.
    pub recoil: Option<Value>,
    /// `raw.get("self")`.
    pub self_effect: Option<Value>,
    /// `raw.get("selfBoost").get("boosts")` when it is an object.
    pub self_boost_boosts: Option<serde_json::Map<String, Value>>,
    /// The first of `resolve`'s unhandled move fields this entry has with a non-null value.
    pub unhandled_field: Option<&'static str>,
}

impl Move {
    #[inline]
    pub fn has_flag(&self, bit: u32) -> bool {
        self.flags.has(bit)
    }
}

/// Type-chart slots: every type the regulation names gets one, and the last is "a type
/// the chart does not list", whose row and column are all 1.0 -- exactly what the nested
/// map's two `unwrap_or(1.0)` gave.
pub const TYPE_SLOTS: usize = 32;
pub const UNKNOWN_TYPE: usize = TYPE_SLOTS - 1;

/// A Pokemon's types as `type_chart` slots, in its own order.
#[derive(Clone, Copy, Debug)]
pub struct TypeSlots {
    slots: [usize; 3],
    len: usize,
}

impl TypeSlots {
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        &self.slots[..self.len]
    }
}

pub struct Reg {
    pub format_id: String,
    pub stat_ids: Vec<String>,
    /// FNV rather than SipHash, keyed by inline id; looked up by `&str` all the same.
    pub species: HashMap<Id, Species, FnvBuild>,
    pub moves: HashMap<Id, Move, FnvBuild>,
    pub natures: HashMap<String, Nature>,
    /// `typechart[attacking][defending]`, dense: a type's slot comes from `type_slot`, and
    /// a type the chart does not name is `UNKNOWN_TYPE` (IKA-101: this was two nested
    /// SipHash maps of `String`, walked twice per damage calculation).
    pub type_chart: Box<[[f64; TYPE_SLOTS]; TYPE_SLOTS]>,
    pub type_slot: HashMap<Id, u8, FnvBuild>,
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
    /// Every id of the vocabulary, in index order -- the encoder's vocabulary is an offset
    /// into these. The order is the committed, append-only one in `configs/vocab/`
    /// (IKA-82, `vocab_order`), read exactly as `encode.build_vocabulary` reads it: a
    /// model's weights are meaningless the moment the same integer means a different
    /// Pokemon. A retired id keeps its slot, so these can list ids the dump no longer has.
    pub species_ids: Vec<String>,
    pub ability_ids: Vec<String>,
    pub item_ids: Vec<String>,
    pub move_ids: Vec<String>,
    /// The types any species has, sorted. Index into this, not into `types`.
    pub species_types: Vec<String>,
}

fn format_id_of(doc: &Value) -> String {
    doc["meta"]["formatId"].as_str().unwrap_or_default().to_string()
}

/// `configs/vocab/<name>` for `configs/regulations/<name>`: `encode.vocab_order_path`.
/// None when there is no such file, and then the ids stay sorted, as they were before
/// the order was committed -- which is also what the Python does.
fn vocab_order(path: &str, format_id: &str) -> Result<Option<Value>, String> {
    let dump = std::path::Path::new(path);
    let (Some(dir), Some(name)) = (dump.parent().and_then(|d| d.parent()), dump.file_name())
    else {
        return Ok(None);
    };
    let order_path = dir.join("vocab").join(name);
    let text = match std::fs::read_to_string(&order_path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("{}: {e}", order_path.display())),
    };
    let doc: Value = serde_json::from_str(&text)
        .map_err(|e| format!("{}: {e}", order_path.display()))?;
    if doc["formatId"].as_str() != Some(format_id) {
        return Err(format!("{} is not the order for {format_id}", order_path.display()));
    }
    Ok(Some(doc))
}

/// The committed order of one table, refusing a dump id it does not list: an append made
/// here would be sorted among that day's other new ids and could move the next time, so
/// `tools/vocab_order.py --append` is the only way an id joins (as in `build_vocabulary`).
fn ordered(table: &str, sorted_ids: Vec<String>, order: &Value) -> Result<Vec<String>, String> {
    let listed: Vec<String> = order[table]
        .as_array()
        .ok_or_else(|| format!("vocabulary order has no {table} list"))?
        .iter()
        .map(|v| v.as_str().map(String::from).ok_or_else(|| format!("{table}: not an id")))
        .collect::<Result<_, _>>()?;
    let known: HashSet<&str> = listed.iter().map(String::as_str).collect();
    if known.len() != listed.len() {
        return Err(format!("vocabulary order lists a {table} id twice"));
    }
    let missing: Vec<&str> =
        sorted_ids.iter().map(String::as_str).filter(|id| !known.contains(id)).collect();
    if !missing.is_empty() {
        return Err(format!(
            "the dump has {table} the committed vocabulary order does not ({}); run \
             `python tools/vocab_order.py --append`",
            missing.join(", ")
        ));
    }
    Ok(listed)
}

impl Reg {
    pub fn load(path: &str) -> Result<Reg, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let doc: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;

        // An id longer than `Id` holds cannot be on a Pokemon or in an action, so it could
        // never be looked up; refusing the load says so instead of panicking in `Id::new`.
        let checked_id = |text: &str| -> Result<Id, String> {
            if text.len() > ID_CAPACITY {
                return Err(format!("id {text:?} is over the {ID_CAPACITY} bytes `Id` holds"));
            }
            Ok(Id::new(text))
        };

        let mut species: HashMap<Id, Species, FnvBuild> = HashMap::default();
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
            let types: Vec<String> = entry["types"]
                .as_array()
                .map(|a| a.iter().filter_map(|t| t.as_str()).map(String::from).collect())
                .unwrap_or_default();
            let type_ids: Vec<Id> =
                types.iter().map(|t| checked_id(t)).collect::<Result<_, _>>()?;
            if type_ids.len() > 3 {
                return Err(format!("species {id} has more than three types"));
            }
            species.insert(
                checked_id(&id)?,
                Species {
                    id,
                    types,
                    weightkg: entry["weightkg"].as_f64().unwrap_or(0.0),
                    base_stats,
                    abilities: entry["abilities"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|t| t.as_str()).map(String::from).collect())
                        .unwrap_or_default(),
                    type_ids: crate::position::Types::from_slice(&type_ids),
                },
            );
        }

        let mut moves: HashMap<Id, Move, FnvBuild> = HashMap::default();
        for entry in doc["moves"].as_array().ok_or("moves is not a list")? {
            let id = entry["id"].as_str().unwrap_or_default().to_string();
            let mut flags = 0u32;
            if let Some(obj) = entry["flags"].as_object() {
                for name in obj.keys() {
                    if let Some((_, bit)) = FLAG_NAMES.iter().find(|(n, _)| n == name) {
                        flags |= bit;
                    }
                }
            }
            let category_name = entry["category"].as_str().unwrap_or("Status");
            let category = Category::parse(category_name)
                .ok_or_else(|| format!("move {id}: unknown category {category_name:?}"))?;
            let (ignore_immunity_all, ignore_immunity_types) = match entry.get("ignoreImmunity") {
                Some(Value::Bool(true)) => (true, Vec::new()),
                Some(Value::Object(o)) => (
                    false,
                    o.iter()
                        .filter(|(_, v)| **v != Value::Bool(false))
                        .map(|(t, _)| checked_id(t))
                        .collect::<Result<Vec<Id>, String>>()?,
                ),
                _ => (false, Vec::new()),
            };
            let unhandled_field = crate::resolve::UNHANDLED_MOVE_FIELDS
                .iter()
                .copied()
                .find(|field| entry.get(*field).map(|v| !v.is_null()).unwrap_or(false));
            moves.insert(
                checked_id(&id)?,
                Move {
                    mtype: checked_id(entry["type"].as_str().unwrap_or("???"))?,
                    category,
                    base_power: entry["basePower"].as_i64().unwrap_or(0),
                    accuracy: entry["accuracy"].as_i64(),
                    priority: entry["priority"].as_i64().unwrap_or(0),
                    target: entry["target"].as_str().unwrap_or("normal").to_string(),
                    crit_ratio: entry["critRatio"].as_i64().unwrap_or(0),
                    flags: MoveFlags(flags),
                    has_secondaries: raw_bool(entry, "secondaries"),
                    has_recoil: raw_bool(entry, "recoil"),
                    has_crash_damage: raw_bool(entry, "hasCrashDamage"),
                    override_offensive_stat: raw_str(entry, "overrideOffensiveStat"),
                    override_defensive_stat: raw_str(entry, "overrideDefensiveStat"),
                    offensive_stat_from_target: raw_str(entry, "overrideOffensivePokemon").as_deref()
                        == Some("target"),
                    defensive_stat_from_source: raw_str(entry, "overrideDefensivePokemon").as_deref()
                        == Some("source"),
                    ignore_offensive: raw_bool(entry, "ignoreOffensive"),
                    ignore_defensive: raw_bool(entry, "ignoreDefensive"),
                    crit_modifier: entry.get("critModifier").and_then(|v| v.as_f64()).unwrap_or(1.5),
                    no_damage_variance: raw_bool(entry, "noDamageVariance"),
                    force_stab: raw_bool(entry, "forceSTAB"),
                    will_crit: raw_bool(entry, "willCrit"),
                    ignore_immunity_all,
                    ignore_immunity_types,
                    breaks_protect: raw_bool(entry, "breaksProtect"),
                    always_hit: raw_bool(entry, "alwaysHit"),
                    ignore_evasion: raw_bool(entry, "ignoreEvasion"),
                    force_switch: raw_bool(entry, "forceSwitch"),
                    self_switch: raw_bool(entry, "selfSwitch"),
                    stalling_move: raw_bool(entry, "stallingMove"),
                    volatile_status: raw_str(entry, "volatileStatus"),
                    status: raw_str(entry, "status"),
                    side_condition: raw_str(entry, "sideCondition"),
                    weather: raw_str(entry, "weather"),
                    terrain: raw_str(entry, "terrain"),
                    pseudo_weather: raw_str(entry, "pseudoWeather"),
                    multihit: entry.get("multihit").cloned(),
                    secondaries: entry
                        .get("secondaries")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default(),
                    drain: entry.get("drain").cloned(),
                    recoil: entry.get("recoil").cloned(),
                    self_effect: entry.get("self").cloned(),
                    self_boost_boosts: entry
                        .get("selfBoost")
                        .and_then(|s| s.get("boosts"))
                        .and_then(Value::as_object)
                        .cloned(),
                    unhandled_field,
                    id,
                    raw: entry.clone(),
                },
            );
        }

        // A slot for every type the chart names, as attacker or defender. Any other type
        // (a move's "???") is `UNKNOWN_TYPE`, whose row and column stay 1.0.
        let mut type_slot: HashMap<Id, u8, FnvBuild> = HashMap::default();
        let mut add_type = |name: &str| -> Result<(), String> {
            let id = checked_id(name)?;
            if !type_slot.contains_key(&id) {
                if type_slot.len() >= UNKNOWN_TYPE {
                    return Err(format!("more than {UNKNOWN_TYPE} types"));
                }
                let slot = type_slot.len() as u8;
                type_slot.insert(id, slot);
            }
            Ok(())
        };
        if let Some(chart) = doc["typechart"].as_object() {
            for (attacking, row) in chart {
                add_type(attacking)?;
                if let Some(obj) = row.as_object() {
                    for defending in obj.keys() {
                        add_type(defending)?;
                    }
                }
            }
        }
        let mut type_chart = Box::new([[1.0f64; TYPE_SLOTS]; TYPE_SLOTS]);
        if let Some(chart) = doc["typechart"].as_object() {
            for (attacking, row) in chart {
                let a = type_slot[attacking.as_str()] as usize;
                if let Some(obj) = row.as_object() {
                    for (defending, mult) in obj {
                        let d = type_slot[defending.as_str()] as usize;
                        type_chart[a][d] = mult.as_f64().unwrap_or(1.0);
                    }
                }
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

        let mut species_ids: Vec<String> = species.keys().map(|k| k.as_str().to_string()).collect();
        species_ids.sort();
        let mut move_ids: Vec<String> = moves.keys().map(|k| k.as_str().to_string()).collect();
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

        if let Some(order) = vocab_order(path, &format_id_of(&doc))? {
            species_ids = ordered("species", species_ids, &order)?;
            ability_ids = ordered("abilities", ability_ids, &order)?;
            item_ids = ordered("items", item_ids, &order)?;
            move_ids = ordered("moves", move_ids, &order)?;
        }

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
            type_chart,
            type_slot,
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

    /// A type's slot in `type_chart`, or `UNKNOWN_TYPE` for one the chart does not name.
    #[inline]
    pub fn type_slot_of(&self, name: &str) -> usize {
        self.type_slot.get(name).map(|s| *s as usize).unwrap_or(UNKNOWN_TYPE)
    }

    /// The slots of a Pokemon's types, in its own order, looked up once.
    #[inline]
    pub fn type_slots(&self, types: &crate::position::Types) -> TypeSlots {
        let names = types.as_slice();
        let mut slots = [UNKNOWN_TYPE; 3];
        for (slot, name) in slots.iter_mut().zip(names) {
            *slot = self.type_slot_of(name.as_str());
        }
        TypeSlots { slots, len: names.len() }
    }

    /// `type_effectiveness` on slots already looked up. The product runs over the
    /// defender's types in order, from 1.0, as `Iterator::product` did.
    #[inline]
    pub fn effectiveness_of(&self, attacking: usize, defending: &TypeSlots) -> f64 {
        let row = &self.type_chart[attacking];
        defending.as_slice().iter().map(|d| row[*d]).product()
    }

    pub fn type_effectiveness(&self, attacking: &str, defending: &crate::position::Types) -> f64 {
        self.effectiveness_of(self.type_slot_of(attacking), &self.type_slots(defending))
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
