//! Position -> numbers, transcribed from `src/pokeuraou/encode.py`.
//!
//! This is the load-bearing half of the value function, so it is the half a port has to be
//! most careful with: the arrays produced here are the network's input, and a feature
//! written one slot along, or rounded differently, is a plausible-looking win probability
//! that means nothing.
//!
//! Two things are copied rather than re-derived:
//!
//! - **the layout**, offset by offset, in the order `_encode_mon`, `_encode_side` and
//!   `_encode_field` write it;
//! - **the arithmetic**, including where Python computes in float64 and lands in a float32
//!   array (most features) and where it divides in float32 throughout (the SP block). The
//!   two round differently and the differential compares the arrays bit for bit.
//!
//! The vocabulary is an offset into the regulation's committed id order (`configs/vocab/`,
//! read by `Reg::load`), with 0 reserved for "absent or unknown", exactly as
//! `build_vocabulary` does.

use crate::position::{Pokemon, Position, Side, BOOST_IDS};
use crate::reg::Reg;
use std::collections::HashMap;

pub const VOLATILES: [&str; 28] = [
    "stall",
    "partiallytrapped",
    "choicelock",
    "mustrecharge",
    "twoturnmove",
    "roost",
    "encore",
    "yawn",
    "perishsong",
    "imprison",
    "leechseed",
    "curse",
    "substitute",
    "stockpile",
    "taunt",
    "endure",
    "healblock",
    "disable",
    "confusion",
    "flashfire",
    "helpinghand",
    "followme",
    "protect",
    "pendingselfswitch",
    "pendingforceswitch",
    "unburden",
    "lockedmove",
    "throatchop",
];

pub const STATUSES: [&str; 6] = ["brn", "par", "slp", "frz", "psn", "tox"];

pub const WEATHERS: [&str; 8] = [
    "sunnyday",
    "raindance",
    "sandstorm",
    "snowscape",
    "hail",
    "desolateland",
    "primordialsea",
    "deltastream",
];
pub const TERRAINS: [&str; 4] =
    ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"];
pub const PSEUDO_WEATHERS: [&str; 6] =
    ["trickroom", "gravity", "magicroom", "wonderroom", "mudsport", "watersport"];

pub const SIDE_CONDITIONS: [&str; 11] = [
    "tailwind",
    "reflect",
    "lightscreen",
    "auroraveil",
    "safeguard",
    "mist",
    "luckychant",
    "stealthrock",
    "spikes",
    "toxicspikes",
    "stickyweb",
];
pub const SLOT_CONDITIONS: [&str; 6] =
    ["wideguard", "quickguard", "craftyshield", "matblock", "healingwish", "lunardance"];

/// IKA-425's state columns, appended to each block (`encode.py`'s `STATE_MON_FEATURES`
/// and the lists after it): sleep turns, toxic stage, Perish Song, the locked move's id,
/// the last move's id per Pokemon; turns left of these per side; of these on the field.
pub const STATE_MON_FEATURES: usize = 5;
pub const STATE_SIDE_CONDITIONS: [&str; 4] = ["tailwind", "reflect", "lightscreen", "auroraveil"];
pub const STATE_PSEUDO_WEATHERS: [&str; 1] = ["trickroom"];
/// `encode.LOCK_VOLATILES`: where the locked move is read, first found wins.
pub const LOCK_VOLATILES: [&str; 4] = ["twoturnmove", "lockedmove", "choicelock", "encore"];
/// IKA-429's bind columns (`encode.py`'s `BIND_MON_FEATURES` and `BIND_SIDE_FEATURES`),
/// appended after the state columns: four per Pokemon, seven per side (`bind.rs`).
pub const BIND_MON_FEATURES: usize = crate::bind::BIND_FEATURES;
pub const BIND_SIDE_FEATURES: usize = crate::bind::BIND_SIDE_FEATURES;

const STAT_SCALE: f64 = 200.0;
const HP_SCALE: f64 = 250.0;
const TURN_CLIP: f64 = 40.0;

pub struct Vocabulary {
    pub species: HashMap<String, i32>,
    pub abilities: HashMap<String, i32>,
    pub items: HashMap<String, i32>,
    pub moves: HashMap<String, i32>,
    pub types: Vec<String>,
    type_index: HashMap<String, usize>,
    volatile_index: HashMap<String, usize>,
}

fn numbered(ids: &[String]) -> HashMap<String, i32> {
    ids.iter().enumerate().map(|(i, id)| (id.clone(), i as i32 + 1)).collect()
}

impl Vocabulary {
    pub fn build(reg: &Reg) -> Vocabulary {
        Vocabulary {
            species: numbered(&reg.species_ids),
            abilities: numbered(&reg.ability_ids),
            items: numbered(&reg.item_ids),
            moves: numbered(&reg.move_ids),
            types: reg.species_types.clone(),
            type_index: reg
                .species_types
                .iter()
                .enumerate()
                .map(|(i, t)| (t.clone(), i))
                .collect(),
            volatile_index: VOLATILES
                .iter()
                .enumerate()
                .map(|(i, v)| ((*v).to_string(), i))
                .collect(),
        }
    }
}

pub struct Widths {
    pub mon: usize,
    pub side: usize,
    pub field: usize,
    pub mons_per_side: usize,
}

/// One batch of positions as flat buffers, in the shapes `Encoded` declares.
#[derive(Default)]
pub struct Encoded {
    pub species: Vec<i32>,
    pub ability: Vec<i32>,
    pub item: Vec<i32>,
    pub moves: Vec<i32>,
    pub mon: Vec<f32>,
    pub mask: Vec<f32>,
    pub side: Vec<f32>,
    pub field: Vec<f32>,
    pub unknown_volatiles: HashMap<String, usize>,
}

/// `EncodingRules` in `encode.py`, the half of it the port applies: which `can_mega` rule.
///
/// Per request and not per process, because one worker plays both arms of a match through
/// one node process (IKA-141). `mega_from_slots` is revision 1 -- `mon.slot in
/// side.mega_capable_slots`, and "that list is non-empty" for `mega_available` -- kept only
/// to be played against the fix. The default is the current rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncodeRules {
    pub mega_from_slots: bool,
}

pub struct Encoder<'a> {
    reg: &'a Reg,
    pub vocab: Vocabulary,
    pub widths: Widths,
    /// `_StatCache`: live stats from SP, memoised. The key is the same triple.
    stats: std::cell::RefCell<HashMap<(String, String, [i64; 6]), [f64; 6]>>,
    /// IKA-429: the bind columns' pair rolls, under exact keys (`bind::BindCache`).
    pub bind_cache: std::cell::RefCell<crate::bind::BindCache>,
    /// Microseconds spent on the bind columns since the last `take_bind_us`.
    bind_us: std::cell::Cell<f64>,
}

impl<'a> Encoder<'a> {
    pub fn new(reg: &'a Reg) -> Encoder<'a> {
        let vocab = Vocabulary::build(reg);
        let widths = Widths {
            mon: 2 + 6 + 6 + BOOST_IDS.len() + 1 + STATUSES.len() + 8 + vocab.types.len() + 8
                + VOLATILES.len()
                + 1
                + STATE_MON_FEATURES
                + BIND_MON_FEATURES,
            side: SIDE_CONDITIONS.len() + 4 + 2 * SLOT_CONDITIONS.len() + STATE_SIDE_CONDITIONS.len()
                + BIND_SIDE_FEATURES,
            field: 2 + WEATHERS.len() + 2 + TERRAINS.len() + PSEUDO_WEATHERS.len() + 2
                + STATE_PSEUDO_WEATHERS.len(),
            mons_per_side: reg.picked_team_size,
        };
        Encoder {
            reg,
            vocab,
            widths,
            stats: Default::default(),
            bind_cache: Default::default(),
            bind_us: Default::default(),
        }
    }

    /// The bind columns' microseconds since the last call (IKA-429), for a fill's header.
    pub fn take_bind_us(&self) -> f64 {
        self.bind_us.replace(0.0)
    }

    /// IKA-429: `bind::bind` through this encoder's cache (and, built with `bind-verify`,
    /// afresh as well, stopping where the two differ).
    pub fn bind_of(&self, position: &Position) -> crate::bind::Bound {
        let m = self.widths.mons_per_side;
        let bound = crate::bind::bind(self.reg, position, m, Some(&mut self.bind_cache.borrow_mut()));
        #[cfg(feature = "bind-verify")]
        {
            let fresh = crate::bind::bind(self.reg, position, m, None);
            if fresh.rows != bound.rows || fresh.sides != bound.sides {
                panic!(
                    "bind-verify: the cached bind columns differ from the fresh ones at turn {}: \
                     cached {:?} {:?}, fresh {:?} {:?}",
                    position.turn, bound.rows, bound.sides, fresh.rows, fresh.sides
                );
            }
        }
        bound
    }

    fn stats_of(&self, mon: &Pokemon) -> [f64; 6] {
        let sp = mon.sp.unwrap_or([0; 6]);
        let nature = if mon.nature.is_empty() { "Serious" } else { mon.nature.as_str() };
        let key = (mon.species.as_str().to_string(), nature.to_string(), sp);
        if let Some(found) = self.stats.borrow().get(&key) {
            return *found;
        }
        let computed = match self.reg.species.get(mon.species.as_str()) {
            None => [0.0; 6],
            Some(species) => match self.reg.stats_from_sp(species, nature, &sp) {
                Err(_) => [0.0; 6],
                Ok(values) => {
                    let mut out = [0.0f64; 6];
                    for index in 0..6 {
                        out[index] = values[index] as f64;
                    }
                    out
                }
            },
        };
        self.stats.borrow_mut().insert(key, computed);
        computed
    }

    pub fn encode_positions(&self, positions: &[&Position]) -> Encoded {
        self.encode_positions_with(positions, EncodeRules::default())
    }

    pub fn encode_positions_with(&self, positions: &[&Position], rules: EncodeRules) -> Encoded {
        let n = positions.len();
        let m = self.widths.mons_per_side;
        let mut out = Encoded {
            species: vec![0; n * 2 * m],
            ability: vec![0; n * 2 * m],
            item: vec![0; n * 2 * m],
            moves: vec![0; n * 2 * m * 4],
            mon: vec![0.0; n * 2 * m * self.widths.mon],
            mask: vec![0.0; n * 2 * m],
            side: vec![0.0; n * 2 * self.widths.side],
            field: vec![0.0; n * self.widths.field],
            unknown_volatiles: HashMap::new(),
        };

        for (b, position) in positions.iter().enumerate() {
            let field_at = b * self.widths.field;
            self.encode_field(&mut out.field[field_at..field_at + self.widths.field], position);
            for (s, one_side) in position.sides.iter().enumerate() {
                let side_at = (b * 2 + s) * self.widths.side;
                self.encode_side(
                    &mut out.side[side_at..side_at + self.widths.side],
                    one_side,
                    rules,
                );
                for (p, one_mon) in one_side.pokemon.iter().take(m).enumerate() {
                    let flat = (b * 2 + s) * m + p;
                    out.mask[flat] = 1.0;
                    out.species[flat] = self
                        .vocab
                        .species
                        .get(one_mon.species.as_str())
                        .copied()
                        .unwrap_or(0);
                    out.ability[flat] = self
                        .vocab
                        .abilities
                        .get(one_mon.ability.as_str())
                        .copied()
                        .unwrap_or(0);
                    out.item[flat] = one_mon
                        .item
                        .and_then(|i| self.vocab.items.get(i.as_str()).copied())
                        .unwrap_or(0);
                    for (k, slot) in one_mon.moves.iter().take(4).enumerate() {
                        out.moves[flat * 4 + k] =
                            self.vocab.moves.get(slot.id.as_str()).copied().unwrap_or(0);
                    }
                    let mon_at = flat * self.widths.mon;
                    self.encode_mon(
                        &mut out.mon[mon_at..mon_at + self.widths.mon],
                        one_mon,
                        one_side,
                        rules,
                        &mut out.unknown_volatiles,
                    );
                }
            }
        }
        // IKA-429: the bind columns, last in each Pokemon's row and each side's; timed apart
        // (`bind_us`), so a fill's header can say what they cost.
        let bind_started = std::time::Instant::now();
        for (b, position) in positions.iter().enumerate() {
            let bound = self.bind_of(position);
            for s in 0..2 {
                let side_end = (b * 2 + s + 1) * self.widths.side;
                for (k, value) in bound.sides[s].iter().enumerate() {
                    #[cfg(feature = "ika429-control")]
                    let k = (k + 1) % BIND_SIDE_FEATURES;
                    out.side[side_end - BIND_SIDE_FEATURES + k] = *value;
                }
                for p in 0..m {
                    let mon_end = ((b * 2 + s) * m + p + 1) * self.widths.mon;
                    out.mon[mon_end - BIND_MON_FEATURES..mon_end].copy_from_slice(&bound.rows[s * m + p]);
                }
            }
        }
        self.bind_us.set(self.bind_us.get() + bind_started.elapsed().as_secs_f64() * 1e6);
        out
    }

    fn encode_field(&self, out: &mut [f32], position: &Position) {
        let mut base = 0;
        let weather = position.field.weather;
        out[base] = if weather.is_none() { 1.0 } else { 0.0 };
        for (i, name) in WEATHERS.iter().enumerate() {
            out[base + 1 + i] =
                if matches!(weather, Some(w) if w.as_str() == *name) { 1.0 } else { 0.0 };
        }
        out[base + 1 + WEATHERS.len()] =
            (position.field.weather_duration.unwrap_or(0) as f64 / 8.0) as f32;
        base += 2 + WEATHERS.len();

        let terrain = position.field.terrain;
        out[base] = if terrain.is_none() { 1.0 } else { 0.0 };
        for (i, name) in TERRAINS.iter().enumerate() {
            out[base + 1 + i] =
                if matches!(terrain, Some(t) if t.as_str() == *name) { 1.0 } else { 0.0 };
        }
        out[base + 1 + TERRAINS.len()] =
            (position.field.terrain_duration.unwrap_or(0) as f64 / 8.0) as f32;
        base += 2 + TERRAINS.len();

        for (i, name) in PSEUDO_WEATHERS.iter().enumerate() {
            out[base + i] = if position.field.has_pseudo_weather(name) { 1.0 } else { 0.0 };
        }
        base += PSEUDO_WEATHERS.len();

        let turn = position.turn as f64;
        out[base] = (turn.min(TURN_CLIP) / TURN_CLIP) as f32;
        out[base + 1] = if turn <= 1.0 { 1.0 } else { 0.0 };
        base += 2;

        // IKA-425: turns left, from the effect's `duration` (absent reads 0).
        for (i, name) in STATE_PSEUDO_WEATHERS.iter().enumerate() {
            let duration = position
                .field
                .pseudo_weather
                .iter()
                .find(|p| p.id.as_str() == *name)
                .and_then(|p| p.duration)
                .unwrap_or(0);
            out[base + i] = (duration as f64 / 8.0) as f32;
        }
    }

    fn encode_side(&self, out: &mut [f32], side: &Side, rules: EncodeRules) {
        let mut base = 0;
        for (i, name) in SIDE_CONDITIONS.iter().enumerate() {
            out[base + i] = if side.has_side_condition(name) { 1.0 } else { 0.0 };
        }
        base += SIDE_CONDITIONS.len();

        let mons = &side.pokemon;
        let alive = mons.iter().filter(|p| !p.fainted).count();
        out[base] = if side.mega_used { 1.0 } else { 0.0 };
        // Read off the Pokemon, as `can_mega` is, not off `mega_capable_slots` (IKA-121),
        // unless the request asked for revision 1 (IKA-141).
        let holder = if rules.mega_from_slots {
            !side.mega_capable_slots.is_empty()
        } else {
            mons.iter().any(|p| self.reg.holds_mega_stone(p.species, p.item))
        };
        out[base + 1] = if !side.mega_used && holder { 1.0 } else { 0.0 };
        out[base + 2] = (alive as f64 / mons.len().max(1) as f64) as f32;
        let total: i64 = mons.iter().map(|p| p.maxhp).sum();
        let total = if total == 0 { 1 } else { total };
        let live: i64 = mons.iter().map(|p| p.hp).sum();
        out[base + 3] = (live as f64 / total as f64) as f32;
        base += 4;

        for slot in 0..2 {
            for (i, name) in SLOT_CONDITIONS.iter().enumerate() {
                let present = side
                    .slot_conditions
                    .get(slot)
                    .map(|group| group.iter().any(|c| c.id.as_str() == *name))
                    .unwrap_or(false);
                out[base + i] = if present { 1.0 } else { 0.0 };
            }
            base += SLOT_CONDITIONS.len();
        }

        // IKA-425: turns left, from the condition's `duration` (absent reads 0).
        for (i, name) in STATE_SIDE_CONDITIONS.iter().enumerate() {
            let duration = side.side_condition(name).and_then(|c| c.duration).unwrap_or(0);
            #[cfg(feature = "ika425-control")]
            let i = (i + 1) % STATE_SIDE_CONDITIONS.len();
            out[base + i] = (duration as f64 / 8.0) as f32;
        }
    }

    fn encode_mon(
        &self,
        out: &mut [f32],
        mon: &Pokemon,
        side: &Side,
        rules: EncodeRules,
        unknown: &mut HashMap<String, usize>,
    ) {
        let maxhp = if mon.maxhp == 0 { 1.0 } else { mon.maxhp as f64 };
        let mut base = 0;
        out[base] = (mon.hp as f64 / maxhp) as f32;
        out[base + 1] = (maxhp / HP_SCALE) as f32;
        base += 2;

        let sp = mon.sp.unwrap_or([0; 6]);
        let stats = self.stats_of(mon);
        for index in 0..6 {
            out[base + index] = (stats[index] / STAT_SCALE) as f32;
        }
        base += 6;
        // Python divides this block in float32, not float64: `np.array(sp, float32) / 32.0`.
        for index in 0..6 {
            out[base + index] = (sp[index] as f32) / 32.0f32;
        }
        base += 6;

        for (i, _name) in BOOST_IDS.iter().enumerate() {
            out[base + i] = (mon.boosts[i] as f64 / 6.0) as f32;
        }
        base += BOOST_IDS.len();

        let status = mon.status;
        out[base] = if status.is_none() { 1.0 } else { 0.0 };
        for (i, name) in STATUSES.iter().enumerate() {
            out[base + 1 + i] =
                if matches!(status, Some(s) if s.as_str() == *name) { 1.0 } else { 0.0 };
        }
        base += 1 + STATUSES.len();

        out[base] = if mon.active_index.is_some() { 1.0 } else { 0.0 };
        out[base + 1] = if mon.active_index == Some(0) { 1.0 } else { 0.0 };
        out[base + 2] = if mon.active_index == Some(1) { 1.0 } else { 0.0 };
        out[base + 3] = if mon.fainted { 1.0 } else { 0.0 };
        out[base + 4] = if mon.is_mega { 1.0 } else { 0.0 };
        // The Pokemon's own species and stone, as `_holds_mega_stone` asks. Its party slot
        // is not an identity: resolve renumbers it on every switch, and
        // `side.mega_capable_slots` keeps the numbers from the start of the game (IKA-121).
        let holds = if rules.mega_from_slots {
            side.mega_capable_slots.contains(&mon.slot) // revision 1 (IKA-141)
        } else {
            self.reg.holds_mega_stone(mon.species, mon.item)
        };
        out[base + 5] = if holds
            && !side.mega_used
            && !mon.is_mega
        {
            1.0
        } else {
            0.0
        };
        out[base + 6] = if mon.trapped { 1.0 } else { 0.0 };
        out[base + 7] = if mon.newly_switched { 1.0 } else { 0.0 };
        base += 8;

        // The *live* types, as stored. A Pokemon whose position leaves them empty encodes
        // as no types at all, which is what Python does -- `_encode_mon` reads `mon.types`
        // and does not fall back to the species.
        for kind in mon.types.as_slice() {
            if let Some(index) = self.vocab.type_index.get(kind.as_str()) {
                out[base + index] = 1.0;
            }
        }
        base += self.vocab.types.len();

        for (i, slot) in mon.moves.iter().take(4).enumerate() {
            out[base + i] = (slot.pp as f64 / (slot.maxpp as f64).max(1.0)) as f32;
            out[base + 4 + i] = if slot.disabled { 1.0 } else { 0.0 };
        }
        base += 8;

        for volatile in &mon.volatiles {
            match self.vocab.volatile_index.get(volatile.id.as_str()) {
                Some(index) => out[base + index] = 1.0,
                None => {
                    out[base + VOLATILES.len()] = 1.0;
                    *unknown.entry(volatile.id.as_str().to_string()).or_insert(0) += 1;
                }
            }
        }
        for vid in &mon.unmodelled_volatiles {
            out[base + VOLATILES.len()] = 1.0;
            *unknown.entry(vid.as_str().to_string()).or_insert(0) += 1;
        }
        base += VOLATILES.len() + 1;

        // IKA-425's state columns, as `_encode_mon` writes them.
        let counter = mon.status_counter.unwrap_or(0) as f64;
        out[base] =
            if matches!(status, Some(s) if s.as_str() == "slp") { (counter / 3.0) as f32 } else { 0.0 };
        out[base + 1] =
            if matches!(status, Some(s) if s.as_str() == "tox") { (counter / 8.0) as f32 } else { 0.0 };
        out[base + 2] = match mon.volatile("perishsong") {
            Some(perish) => (perish.duration.unwrap_or(0) as f64 / 4.0) as f32,
            None => 0.0,
        };
        out[base + 3] = self.move_id_of(locked_move_of(mon));
        out[base + 4] = self.move_id_of(mon.last_move);
    }

    /// A move's vocabulary integer as a float32 column (0 for none or unknown).
    fn move_id_of(&self, id: Option<crate::id::Id>) -> f32 {
        match id {
            Some(id) if !id.as_str().is_empty() => {
                self.vocab.moves.get(id.as_str()).copied().unwrap_or(0) as f32
            }
            _ => 0.0,
        }
    }
}

/// `encode.locked_move_of`: the move the first of `LOCK_VOLATILES` present names.
pub fn locked_move_of(mon: &Pokemon) -> Option<crate::id::Id> {
    for vid in LOCK_VOLATILES {
        if let Some(effect) = mon.volatile(vid) {
            if let Some(m) = effect.move_id {
                if !m.as_str().is_empty() {
                    return Some(m);
                }
            }
        }
    }
    None
}
