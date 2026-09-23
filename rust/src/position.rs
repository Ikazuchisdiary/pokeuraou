//! The canonical position, mirroring `src/pokeuraou/position.py` field for field.
//!
//! The JSON on the wire is the same document Python reads and writes, which is what makes
//! the differential harness possible: Python dumps a position, Rust reads it, resolves a
//! turn, writes the successors back out, and the two documents are compared key by key.
//!
//! The representation is chosen for the copy. A turn clones the position once per branch
//! and per action, and in Python that is 16% of a generation run. Here every id is inline
//! (`Id`), boosts are a fixed array, moves are a fixed four, and the two free-form JSON
//! blobs the resolver only ever reads (`abilityState`, an effect's `extra`) are behind an
//! `Rc` -- so a clone is a memcpy plus a refcount bump, with an allocation only for the
//! volatiles a Pokemon actually carries.

use crate::id::Id;
use serde_json::{json, Map, Value};
use std::rc::Rc;

pub const STAT_IDS: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];
pub const BOOST_IDS: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

pub fn boost_index(name: &str) -> Option<usize> {
    BOOST_IDS.iter().position(|b| *b == name)
}

pub fn stat_index(name: &str) -> Option<usize> {
    STAT_IDS.iter().position(|s| *s == name)
}

/// A JSON object the resolver only reads. Shared rather than copied, as Python shares it.
pub type Blob = Rc<Map<String, Value>>;

fn empty_blob() -> Blob {
    thread_local! {
        static EMPTY: Blob = Rc::new(Map::new());
    }
    EMPTY.with(|e| e.clone())
}

fn blob_from(value: Option<&Value>) -> Blob {
    match value.and_then(|v| v.as_object()) {
        None => empty_blob(),
        Some(map) if map.is_empty() => empty_blob(),
        Some(map) => Rc::new(map.clone()),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Effect {
    pub id: Id,
    pub duration: Option<i64>,
    pub layers: Option<i64>,
    pub counter: Option<i64>,
    pub source_slot: Option<Id>,
    pub move_id: Option<Id>,
    pub extra: Blob,
}

impl Effect {
    pub fn new(id: Id) -> Effect {
        Effect {
            id,
            duration: None,
            layers: None,
            counter: None,
            source_slot: None,
            move_id: None,
            extra: empty_blob(),
        }
    }

    pub fn from_json(value: &Value) -> Effect {
        Effect {
            id: Id::new(value["id"].as_str().unwrap_or_default()),
            duration: value.get("duration").and_then(Value::as_i64),
            layers: value.get("layers").and_then(Value::as_i64),
            counter: value.get("counter").and_then(Value::as_i64),
            source_slot: value.get("sourceSlot").and_then(Value::as_str).map(Id::new),
            move_id: value.get("move").and_then(Value::as_str).map(Id::new),
            extra: blob_from(value.get("extra")),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.id.as_str()));
        if let Some(v) = self.duration {
            out.insert("duration".into(), json!(v));
        }
        if let Some(v) = self.layers {
            out.insert("layers".into(), json!(v));
        }
        if let Some(v) = self.counter {
            out.insert("counter".into(), json!(v));
        }
        if let Some(v) = self.source_slot {
            out.insert("sourceSlot".into(), json!(v.as_str()));
        }
        if let Some(v) = self.move_id {
            out.insert("move".into(), json!(v.as_str()));
        }
        if !self.extra.is_empty() {
            out.insert("extra".into(), Value::Object((*self.extra).clone()));
        }
        Value::Object(out)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveSlot {
    pub id: Id,
    pub pp: i64,
    pub maxpp: i64,
    pub disabled: bool,
    pub used: bool,
}

impl MoveSlot {
    pub fn from_json(value: &Value) -> MoveSlot {
        MoveSlot {
            id: Id::new(value["id"].as_str().unwrap_or_default()),
            pp: value["pp"].as_i64().unwrap_or(0),
            maxpp: value["maxpp"].as_i64().unwrap_or(0),
            disabled: value.get("disabled").and_then(Value::as_bool).unwrap_or(false),
            used: value.get("used").and_then(Value::as_bool).unwrap_or(false),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id.as_str(),
            "pp": self.pp,
            "maxpp": self.maxpp,
            "disabled": self.disabled,
            "used": self.used,
        })
    }

    pub fn usable(&self) -> bool {
        self.pp > 0 && !self.disabled
    }
}

/// Up to four move slots, inline.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Moves {
    slots: [Option<MoveSlot>; 4],
}

impl Moves {
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }
    pub fn iter(&self) -> impl Iterator<Item = &MoveSlot> {
        self.slots.iter().flatten()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut MoveSlot> {
        self.slots.iter_mut().flatten()
    }
    pub fn get(&self, move_id: Id) -> Option<&MoveSlot> {
        self.iter().find(|m| m.id == move_id)
    }
    pub fn get_mut(&mut self, move_id: Id) -> Option<&mut MoveSlot> {
        self.iter_mut().find(|m| m.id == move_id)
    }
    pub fn push(&mut self, slot: MoveSlot) {
        for place in self.slots.iter_mut() {
            if place.is_none() {
                *place = Some(slot);
                return;
            }
        }
        panic!("a Pokemon with more than four moves");
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pokemon {
    pub slot: usize,
    pub species: Id,
    pub base_species: Id,
    /// Empty means "use the species' types".
    pub types: Types,
    pub ability: Id,
    pub nature: Id,
    pub moves: Moves,
    pub hp: i64,
    pub maxhp: i64,
    pub level: i64,
    pub gender: Id,
    pub item: Option<Id>,
    pub base_item: Option<Id>,
    pub sp: Option<[i64; 6]>,
    pub fainted: bool,
    pub status: Option<Id>,
    pub status_duration: Option<i64>,
    pub status_counter: Option<i64>,
    pub boosts: [i8; 7],
    pub volatiles: Vec<Effect>,
    pub unmodelled_volatiles: Vec<Id>,
    pub is_mega: bool,
    pub active_index: Option<usize>,
    pub trapped: bool,
    pub newly_switched: bool,
    pub last_move: Option<Id>,
    pub locked_move: Option<Id>,
    pub move_last_turn_failed: bool,
    pub times_attacked: i64,
    pub active_move_actions: i64,
    pub ability_state: Blob,
    pub stats_override: Option<[i64; 6]>,
    pub transformed: bool,
}

/// A Pokemon's live types: at most three, inline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Types {
    len: u8,
    names: [Id; 3],
}

impl Types {
    pub fn from_slice(items: &[Id]) -> Types {
        assert!(items.len() <= 3, "a Pokemon with more than three types");
        let mut names = [Id::EMPTY; 3];
        names[..items.len()].copy_from_slice(items);
        Types { len: items.len() as u8, names }
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[Id] {
        &self.names[..self.len as usize]
    }
    pub fn contains(&self, name: &str) -> bool {
        self.as_slice().iter().any(|t| t.as_str() == name)
    }
}

impl Pokemon {
    pub fn from_json(value: &Value) -> Pokemon {
        let mut moves = Moves::default();
        if let Some(list) = value["moves"].as_array() {
            for entry in list {
                moves.push(MoveSlot::from_json(entry));
            }
        }
        let mut boosts = [0i8; 7];
        if let Some(map) = value.get("boosts").and_then(Value::as_object) {
            for (name, amount) in map {
                if let Some(index) = boost_index(name) {
                    boosts[index] = amount.as_i64().unwrap_or(0) as i8;
                }
            }
        }
        let types: Vec<Id> = value
            .get("types")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).map(Id::new).collect())
            .unwrap_or_default();
        let species = Id::new(value["species"].as_str().unwrap_or_default());
        Pokemon {
            slot: value["slot"].as_u64().unwrap_or(0) as usize,
            species,
            base_species: value
                .get("baseSpecies")
                .and_then(Value::as_str)
                .map(Id::new)
                .unwrap_or(species),
            types: Types::from_slice(&types),
            ability: Id::new(value["ability"].as_str().unwrap_or_default()),
            nature: value
                .get("nature")
                .and_then(Value::as_str)
                .map(Id::new)
                .unwrap_or_else(|| Id::new("Serious")),
            moves,
            hp: value["hp"].as_i64().unwrap_or(0),
            maxhp: value["maxhp"].as_i64().unwrap_or(0),
            level: value.get("level").and_then(Value::as_i64).unwrap_or(50),
            gender: value
                .get("gender")
                .and_then(Value::as_str)
                .map(Id::new)
                .unwrap_or_else(|| Id::new("N")),
            item: value.get("item").and_then(Value::as_str).map(Id::new),
            base_item: value.get("baseItem").and_then(Value::as_str).map(Id::new),
            sp: read_stats(value.get("sp")),
            fainted: value.get("fainted").and_then(Value::as_bool).unwrap_or(false),
            status: value.get("status").and_then(Value::as_str).map(Id::new),
            status_duration: value.get("statusDuration").and_then(Value::as_i64),
            status_counter: value.get("statusCounter").and_then(Value::as_i64),
            boosts,
            volatiles: value
                .get("volatiles")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(Effect::from_json).collect())
                .unwrap_or_default(),
            unmodelled_volatiles: value
                .get("unmodelledVolatiles")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).map(Id::new).collect())
                .unwrap_or_default(),
            is_mega: value.get("isMega").and_then(Value::as_bool).unwrap_or(false),
            active_index: value.get("activeIndex").and_then(Value::as_u64).map(|v| v as usize),
            trapped: value.get("trapped").and_then(Value::as_bool).unwrap_or(false),
            newly_switched: value.get("newlySwitched").and_then(Value::as_bool).unwrap_or(false),
            last_move: value.get("lastMove").and_then(Value::as_str).map(Id::new),
            locked_move: value.get("lockedMove").and_then(Value::as_str).map(Id::new),
            move_last_turn_failed: value
                .get("moveLastTurnFailed")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            times_attacked: value.get("timesAttacked").and_then(Value::as_i64).unwrap_or(0),
            active_move_actions: value
                .get("activeMoveActions")
                .and_then(Value::as_i64)
                .unwrap_or(0),
            ability_state: blob_from(value.get("abilityState")),
            stats_override: read_stats(value.get("statsOverride")),
            transformed: value.get("transformed").and_then(Value::as_bool).unwrap_or(false),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut boosts = Map::new();
        for (index, name) in BOOST_IDS.iter().enumerate() {
            if self.boosts[index] != 0 {
                boosts.insert((*name).into(), json!(self.boosts[index]));
            }
        }
        let mut out = Map::new();
        out.insert("slot".into(), json!(self.slot));
        out.insert("species".into(), json!(self.species.as_str()));
        out.insert("baseSpecies".into(), json!(self.base_species.as_str()));
        out.insert(
            "types".into(),
            Value::Array(self.types.as_slice().iter().map(|t| json!(t.as_str())).collect()),
        );
        out.insert("level".into(), json!(self.level));
        out.insert("gender".into(), json!(self.gender.as_str()));
        out.insert("ability".into(), json!(self.ability.as_str()));
        out.insert("item".into(), id_json(self.item));
        out.insert("baseItem".into(), id_json(self.base_item));
        out.insert("nature".into(), json!(self.nature.as_str()));
        out.insert("sp".into(), stats_json(self.sp, &STAT_IDS));
        out.insert(
            "moves".into(),
            Value::Array(self.moves.iter().map(MoveSlot::to_json).collect()),
        );
        out.insert("hp".into(), json!(self.hp));
        out.insert("maxhp".into(), json!(self.maxhp));
        out.insert("fainted".into(), json!(self.fainted));
        out.insert("status".into(), id_json(self.status));
        out.insert("boosts".into(), Value::Object(boosts));
        out.insert(
            "volatiles".into(),
            Value::Array(self.volatiles.iter().map(Effect::to_json).collect()),
        );
        out.insert(
            "unmodelledVolatiles".into(),
            Value::Array(self.unmodelled_volatiles.iter().map(|v| json!(v.as_str())).collect()),
        );
        out.insert("isMega".into(), json!(self.is_mega));
        out.insert("activeIndex".into(), match self.active_index {
            None => Value::Null,
            Some(v) => json!(v),
        });
        out.insert("trapped".into(), json!(self.trapped));
        out.insert("newlySwitched".into(), json!(self.newly_switched));
        out.insert("lastMove".into(), id_json(self.last_move));
        out.insert("lockedMove".into(), id_json(self.locked_move));
        out.insert("moveLastTurnFailed".into(), json!(self.move_last_turn_failed));
        out.insert("timesAttacked".into(), json!(self.times_attacked));
        out.insert("activeMoveActions".into(), json!(self.active_move_actions));
        out.insert("abilityState".into(), Value::Object((*self.ability_state).clone()));
        out.insert("transformed".into(), json!(self.transformed));
        if self.stats_override.is_some() {
            out.insert("statsOverride".into(), stats_json(self.stats_override, &STAT_IDS));
        }
        if let Some(v) = self.status_duration {
            out.insert("statusDuration".into(), json!(v));
        }
        if let Some(v) = self.status_counter {
            out.insert("statusCounter".into(), json!(v));
        }
        Value::Object(out)
    }

    pub fn is_active(&self) -> bool {
        self.active_index.is_some()
    }

    pub fn volatile(&self, vid: &str) -> Option<&Effect> {
        self.volatiles.iter().find(|v| v.id.as_str() == vid)
    }

    pub fn volatile_mut(&mut self, vid: &str) -> Option<&mut Effect> {
        self.volatiles.iter_mut().find(|v| v.id.as_str() == vid)
    }

    pub fn has_volatile(&self, vid: &str) -> bool {
        self.volatile(vid).is_some()
    }

    pub fn boost(&self, stat: &str) -> i8 {
        boost_index(stat).map(|i| self.boosts[i]).unwrap_or(0)
    }
}

fn id_json(value: Option<Id>) -> Value {
    match value {
        None => Value::Null,
        Some(v) => json!(v.as_str()),
    }
}

fn read_stats(value: Option<&Value>) -> Option<[i64; 6]> {
    let map = value?.as_object()?;
    let mut out = [0i64; 6];
    for (name, amount) in map {
        if let Some(index) = stat_index(name) {
            out[index] = amount.as_i64().unwrap_or(0);
        }
    }
    Some(out)
}

fn stats_json(values: Option<[i64; 6]>, names: &[&str; 6]) -> Value {
    match values {
        None => Value::Null,
        Some(values) => {
            let mut out = Map::new();
            for (index, name) in names.iter().enumerate() {
                out.insert((*name).into(), json!(values[index]));
            }
            Value::Object(out)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Side {
    pub id: Rc<str>,
    pub name: Rc<str>,
    pub active: Vec<Option<usize>>,
    /// Shared, not owned. A `Pokemon` is 928 bytes and a side holds six, so cloning a
    /// position used to copy eleven kilobytes -- and the resolver clones one 44 times a
    /// turn, which measured at 65% of its whole cost. A turn touches the two Pokemon that
    /// are out; the rest are on the bench and identical in every branch. `Rc::make_mut`
    /// copies one only when something writes to it while it is still shared.
    pub pokemon: Vec<Rc<Pokemon>>,
    pub side_conditions: Vec<Effect>,
    pub slot_conditions: Vec<Vec<Effect>>,
    pub mega_used: bool,
    /// Party slots holding their own mega stone *as numbered when the side was built*.
    /// Resolve renumbers `Pokemon.slot` on a switch and does not touch this, so it is not
    /// an identity; `Reg::holds_mega_stone` is what the encoder asks (IKA-121).
    pub mega_capable_slots: Vec<usize>,
}

impl Side {
    pub fn from_json(value: &Value) -> Side {
        Side {
            id: Rc::from(value["id"].as_str().unwrap_or_default()),
            name: Rc::from(
                value
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| value["id"].as_str().unwrap_or_default()),
            ),
            active: value["active"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().map(|n| n as usize)).collect())
                .unwrap_or_default(),
            pokemon: value["pokemon"]
                .as_array()
                .map(|a| a.iter().map(|v| Rc::new(Pokemon::from_json(v))).collect())
                .unwrap_or_default(),
            side_conditions: value
                .get("sideConditions")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(Effect::from_json).collect())
                .unwrap_or_default(),
            slot_conditions: value
                .get("slotConditions")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|g| {
                            g.as_array()
                                .map(|inner| inner.iter().map(Effect::from_json).collect())
                                .unwrap_or_default()
                        })
                        .collect()
                })
                .unwrap_or_default(),
            mega_used: value.get("megaUsed").and_then(Value::as_bool).unwrap_or(false),
            mega_capable_slots: value
                .get("megaCapableSlots")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
                .unwrap_or_default(),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "id": &*self.id,
            "name": &*self.name,
            "active": self.active.iter().map(|s| match s {
                None => Value::Null,
                Some(v) => json!(v),
            }).collect::<Vec<_>>(),
            "pokemon": self.pokemon.iter().map(|m| m.to_json()).collect::<Vec<_>>(),
            "sideConditions": self.side_conditions.iter().map(Effect::to_json).collect::<Vec<_>>(),
            "slotConditions": self.slot_conditions.iter()
                .map(|g| g.iter().map(Effect::to_json).collect::<Vec<_>>())
                .collect::<Vec<_>>(),
            "megaUsed": self.mega_used,
            "megaCapableSlots": self.mega_capable_slots,
        })
    }

    pub fn side_condition(&self, cid: &str) -> Option<&Effect> {
        self.side_conditions.iter().find(|c| c.id.as_str() == cid)
    }

    pub fn has_side_condition(&self, cid: &str) -> bool {
        self.side_condition(cid).is_some()
    }

    pub fn active_pokemon(&self, slot: usize) -> Option<&Pokemon> {
        self.active.get(slot).copied().flatten().map(|index| &*self.pokemon[index])
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Field {
    pub weather: Option<Id>,
    pub weather_duration: Option<i64>,
    pub terrain: Option<Id>,
    pub terrain_duration: Option<i64>,
    pub pseudo_weather: Vec<Effect>,
}

impl Field {
    pub fn from_json(value: &Value) -> Field {
        Field {
            weather: value.get("weather").and_then(Value::as_str).map(Id::new),
            weather_duration: value.get("weatherDuration").and_then(Value::as_i64),
            terrain: value.get("terrain").and_then(Value::as_str).map(Id::new),
            terrain_duration: value.get("terrainDuration").and_then(Value::as_i64),
            pseudo_weather: value
                .get("pseudoWeather")
                .and_then(Value::as_array)
                .map(|a| a.iter().map(Effect::from_json).collect())
                .unwrap_or_default(),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("weather".into(), id_json(self.weather));
        out.insert("terrain".into(), id_json(self.terrain));
        out.insert(
            "pseudoWeather".into(),
            Value::Array(self.pseudo_weather.iter().map(Effect::to_json).collect()),
        );
        if let Some(v) = self.weather_duration {
            out.insert("weatherDuration".into(), json!(v));
        }
        if let Some(v) = self.terrain_duration {
            out.insert("terrainDuration".into(), json!(v));
        }
        Value::Object(out)
    }

    pub fn has_pseudo_weather(&self, pid: &str) -> bool {
        self.pseudo_weather.iter().any(|p| p.id.as_str() == pid)
    }

    pub fn trick_room(&self) -> bool {
        self.has_pseudo_weather("trickroom")
    }
}

/// How many positions have been cloned. The resolver's cost is largely this, and a
/// counter is the only honest way to say how largely: an atomic increment beside an
/// allocation is noise, and the alternative is guessing.
pub static CLONES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many times a write to a shared Pokemon forced a copy of it. The counterpart to
/// `CLONES`: sharing only pays if this stays well under twelve per clone.
pub static UNSHARED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

impl Clone for Position {
    fn clone(&self) -> Position {
        CLONES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Position {
            format: self.format.clone(),
            sides: self.sides.clone(),
            turn: self.turn,
            field: self.field.clone(),
            request_state: self.request_state.clone(),
            ended: self.ended,
            winner: self.winner.clone(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Position {
    /// `Rc<str>` rather than `String` throughout: none of these ever changes during a turn,
    /// and cloning one used to be an allocation apiece. With the Pokemon shared the clone
    /// became allocation-bound rather than copy-bound, and this is where the allocations
    /// were. A refcount bump has no length limit, which `Id` would impose.
    pub format: Rc<str>,
    /// Exactly two, which is what a battle has. A `Vec` here was one more allocation per
    /// clone for a length that is never anything else.
    pub sides: [Side; 2],
    pub turn: i64,
    pub field: Field,
    pub request_state: Rc<str>,
    pub ended: bool,
    pub winner: Option<Rc<str>>,
}

impl Position {
    pub fn from_json(value: &Value) -> Position {
        Position {
            format: Rc::from(value["format"].as_str().unwrap_or_default()),
            sides: {
                let listed = value["sides"].as_array().map(Vec::as_slice).unwrap_or(&[]);
                if listed.len() != 2 {
                    panic!("a position has two sides, not {}", listed.len());
                }
                [Side::from_json(&listed[0]), Side::from_json(&listed[1])]
            },
            turn: value.get("turn").and_then(Value::as_i64).unwrap_or(1),
            field: value.get("field").map(Field::from_json).unwrap_or_default(),
            request_state: Rc::from(
                value.get("requestState").and_then(Value::as_str).unwrap_or("move"),
            ),
            ended: value.get("ended").and_then(Value::as_bool).unwrap_or(false),
            winner: value.get("winner").and_then(Value::as_str).map(Rc::from),
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "format": &*self.format,
            "turn": self.turn,
            "field": self.field.to_json(),
            "sides": self.sides.iter().map(Side::to_json).collect::<Vec<_>>(),
            "requestState": &*self.request_state,
            "ended": self.ended,
            "winner": match &self.winner {
                None => Value::Null,
                Some(w) => json!(&**w),
            },
        })
    }

    pub fn mon_at(&self, side: usize, slot: usize) -> Option<&Pokemon> {
        self.sides.get(side)?.active_pokemon(slot)
    }

    pub fn mon_at_mut(&mut self, side: usize, slot: usize) -> Option<&mut Pokemon> {
        let index = (*self.sides.get(side)?).active.get(slot).copied().flatten()?;
        let shared = &mut self.sides[side].pokemon[index];
        if Rc::strong_count(shared) > 1 {
            UNSHARED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Some(Rc::make_mut(shared))
    }
}
