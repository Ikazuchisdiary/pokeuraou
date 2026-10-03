//! 縛り (IKA-429): which active Pokemon can move first and knock an opposing one out.
//!
//! The user's term: a Pokemon "binds" a foe when it moves before that foe and one of its
//! attacks takes the foe's current HP. The value function reads Speed, stats and moves
//! separately and had to rediscover this from them; here it is computed from the port's own
//! damage calculator and handed over as numbers per Pokemon row and per side
//! (`ENCODING_REVISION` 4).
//!
//! Only the active Pokemon are paired (two against two). A bench row reads 0 in all four
//! of its columns. That keeps every row a function of the active Pokemon and the field,
//! which is what `beliefnode._patched` relies on when it copies a hidden bench row from
//! another position's encoding.
//!
//! For an attacker X and a defender Y on the other side, over X's usable damaging moves m:
//!
//! * `ko(X, Y, m)`: the share of the 16 damage rolls that take Y's current HP. No critical
//!   hit (a `willCrit` move is taken as critical, since it always is), accuracy not read,
//!   a spread move at 0.75 when it reaches two or more Pokemon, Focus Sash and Sturdy at
//!   full HP and Disguise as the calculator applies them, a multi-hit move as its minimum
//!   count (Skill Link: the maximum) of the same roll.
//! * `first(X, Y, m)`: 1 if m acts before Y would act with a priority-0 move, 0 if after,
//!   0.5 on a Speed tie: the move's priority (Prankster, Gale Wings, Grassy Glide), then
//!   effective Speed (Tailwind, paralysis, Choice Scarf, weather abilities) with Trick Room
//!   reversing it. Y's own move is not known, so its priority is taken as 0. A priority move
//!   into Psychic Terrain on a grounded Y, or into a side with Dazzling, Queenly Majesty or
//!   Armor Tail, fails and counts as no knock-out.
//! * `bind(X, Y, m) = first * ko`, `bind(X, Y) = max_m bind(X, Y, m)`,
//!   `ko(X, Y) = max_m ko(X, Y, m)`.
//!
//! Row of X: `[sum_Y bind(X, Y), sum_Y bind(Y, X), sum_Y ko(X, Y), sum_Y ko(Y, X)] / 2`.
//! The side columns are `BIND_SIDE_FEATURES`. Every value is a multiple of 1/2048, exact
//! in float32, so the port and Python (which asks the port) agree to the bit.
//!
//! Usable moves: PP left and not disabled; a Pokemon locked into a move (`LOCK_VOLATILES`)
//! uses that move only.
//!
//! Cost. A leaf is encoded at every node the search fills, so `BindCache` keeps, under
//! exact keys: each Pokemon as the calculator sees it (its `Battler` and usable moves,
//! keyed on the fields they are built from, the HP only through the flags the calculator
//! reads), the field, and each (attacker, defender) pair's damage rolls. The knock-out share
//! is then taken against each leaf's own HP. Without a cache everything is computed afresh;
//! the build with `bind-verify` holds the two to the same bits on every leaf it encodes.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::rc::Rc;

use crate::battler::{Battler, FieldState, VolatileFlags, N_ROLLS};
use crate::damage::{calculate, effective_damage, is_grounded};
use crate::id::Id;
use crate::moveinfo::MoveContext;
use crate::position::{Pokemon, Position};
use crate::reg::{Move, Reg};
use crate::resolve::field_state;
use crate::speed::{effective_speed, move_priority};

pub const BIND_FEATURES: usize = 4;

/// Per side, from the matrix `B[i][j] = bind(i, j)` of its active Pokemon i on the foes j
/// and the foes' `C[k][i] = bind(k, i)` (IKA-429; the user's examples, 10/4):
///
/// * `foes_bound = sum_j max_i B[i][j] / 2`: how many of the foes are bound (two is 両縛り)
/// * `double_bind = min_j max_i B[i][j]` with two foes on the field, else 0: both bound
/// * `secure_bound = sum_j max_i B[i][j] (1 - max_k C[k][i]) / 2`: a bind counts only as far
///   as the binder is not itself bound by a foe, which is the release the user named (the
///   bound Pokemon protects and its partner knocks the binder out first)
/// * `spread_double = max_{i, m spread} min_j bind(i, j, m)`: one Pokemon's one move that
///   reaches both foes (`allAdjacentFoes`, `allAdjacent`, at 0.75) binds both at once -- a
///   両縛り that Protect on one foe does not release (the user's Heat Wave and Dazzling
///   Gleam, 10/4)
/// * `spread_ally_ko`: the share of rolls with which that move knocks out the user's own
///   partner (Earthquake, Surf), the least among the moves that give `spread_double`
/// * `spread_wide_guard`: 1 when a foe can use Wide Guard against that spread move
/// * `spread_fake_out`: 1 when a foe's Fake Out (its first move out) stops the spread move's
///   user this turn (`fake_out_stops`)
///
/// The last two are 0 when no spread move binds both foes; a 両縛り by two Pokemon's
/// single-target moves is released differently (Protect on the bound one, the partner
/// knocking a binder out first), which `secure_bound` reads. A side's own "bound" count is
/// the other side's `foes_bound`, which the head reads beside it.
pub const BIND_SIDE_FEATURES: usize = 7;

/// The moves that release a bind (the user's list, 10/4), in one place. The bind columns
/// read Wide Guard and Fake Out against a spread 両縛り; redirection and the Protect family
/// are listed for the record and read by the network through its move ids only.
pub const RELEASE_WIDE_GUARD: &str = "wideguard";
pub const RELEASE_FAKE_OUT: &str = "fakeout";
#[allow(dead_code)]
pub const RELEASE_REDIRECTION: [&str; 2] = ["followme", "ragepowder"];
#[allow(dead_code)]
pub const RELEASE_PROTECTION: [&str; 8] = [
    "protect",
    "detect",
    "spikyshield",
    "kingsshield",
    "banefulbunker",
    "silktrap",
    "burningbulwark",
    "obstruct",
];

/// Moves whose damage reads an HP beyond the pinch / half / full flags (`moveinfo`'s
/// `base_power` and `fixed_damage`, `damage_callback`'s Endeavor). A pair with one of these
/// keys on both HPs exactly.
const HP_MOVES: [&str; 13] = [
    "eruption",
    "waterspout",
    "dragonenergy",
    "flail",
    "reversal",
    "finalgambit",
    "endeavor",
    "hardpress",
    "crushgrip",
    "wringout",
    "superfang",
    "naturesmadness",
    "ruination",
];

/// FxHash over whole words: the keys here are arrays of u64.
#[derive(Default, Clone, Copy)]
pub struct WordHasher(u64);

impl Hasher for WordHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.write_u64(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            self.write_u64(u64::from_le_bytes(word));
        }
    }
    #[inline]
    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    #[inline]
    fn write_usize(&mut self, word: usize) {
        self.write_u64(word as u64);
    }
}

type WordBuild = BuildHasherDefault<WordHasher>;

/// Words of a key, written field by field.
#[derive(Default)]
struct Words(Vec<u64>);

impl Words {
    #[inline]
    fn id(&mut self, id: Id) {
        self.0.extend_from_slice(&id.words());
    }
    #[inline]
    fn opt(&mut self, id: Option<Id>) {
        match id {
            None => self.0.push(u64::MAX),
            Some(id) => self.id(id),
        }
    }
    #[inline]
    fn int(&mut self, value: i64) {
        self.0.push(value as u64);
    }
}

/// One usable damaging move against one defender: what the knock-out share needs.
#[derive(Clone, Copy)]
struct MoveEval {
    /// The move's place in the attacker's usable moves, so its hits on two Pokemon pair up.
    index: u8,
    /// Whether it reaches both foes in one use (`allAdjacentFoes`, `allAdjacent`).
    spread_kind: bool,
    /// 2 first, 0 after, 1 decided by Speed.
    first: u8,
    hits: i64,
    rolls: [i64; N_ROLLS],
}

#[derive(Default)]
struct PairValue {
    moves: Vec<MoveEval>,
}

/// One Pokemon as the calculator sees it: what the key words are built from, kept.
struct MonEntry {
    id: u64,
    /// Its HP is that of the first Pokemon seen under the key; each use sets its own.
    battler: Battler,
    moves: Vec<Id>,
    /// A usable move reads both HPs exactly (`HP_MOVES`), or its side's fainted count
    /// (Last Respects): either joins the pair's key.
    hp_exact: bool,
    reads_fainted: bool,
}

impl MonEntry {
    fn new(id: u64, battler: Battler, mon: &Pokemon) -> MonEntry {
        let moves = usable_moves(mon);
        let hp_exact = moves.iter().any(|id| HP_MOVES.contains(&id.as_str()));
        let reads_fainted = moves.iter().any(|id| id.as_str() == "lastrespects");
        MonEntry { id, battler, moves, hp_exact, reads_fainted }
    }
}

/// Exact-keyed memory of Pokemon, fields and pairs. Bounded: emptied when full.
#[derive(Default)]
pub struct BindCache {
    fields: HashMap<Box<[u64]>, (u64, Rc<FieldState>), WordBuild>,
    mons: HashMap<Box<[u64]>, Rc<MonEntry>, WordBuild>,
    pairs: HashMap<[u64; 7], Rc<PairValue>, WordBuild>,
    words: Words,
    /// The numbers given to Pokemon and fields, never reused (not even after emptying),
    /// so a pair's key can never name a Pokemon it was not made for.
    next: u64,
    pub hits: usize,
    pub misses: usize,
}

/// Entries of each map before all three are emptied (about 30 MB at most per process).
const CACHE_LIMIT: usize = 1 << 17;

impl BindCache {
    fn check_size(&mut self) {
        if self.pairs.len() > CACHE_LIMIT || self.mons.len() > CACHE_LIMIT || self.fields.len() > CACHE_LIMIT {
            self.fields.clear();
            self.mons.clear();
            self.pairs.clear();
        }
    }

    fn number(&mut self) -> u64 {
        self.next += 1;
        self.next
    }
}

fn move_context(pos: &Position, side: usize, mon: &Pokemon) -> MoveContext {
    MoveContext {
        weather: pos.field.weather,
        terrain: pos.field.terrain,
        side_total_fainted: pos.sides[side].pokemon.iter().filter(|m| m.fainted).count() as i64,
        times_attacked: mon.times_attacked,
        previous_move_failed: mon.move_last_turn_failed,
        hit_index: 1,
        ..Default::default()
    }
}

/// The hit count taken for a multi-hit move: its minimum, Skill Link's maximum.
fn hit_count(mv: &Move, ability: &str) -> i64 {
    let Some(multihit) = mv.multihit.as_ref() else { return 1 };
    if let Some(fixed) = multihit.as_u64() {
        return fixed as i64;
    }
    let Some(list) = multihit.as_array() else { return 1 };
    let low = list.first().and_then(|v| v.as_u64()).unwrap_or(1) as i64;
    let high = list.last().and_then(|v| v.as_u64()).unwrap_or(low as u64) as i64;
    if ability == "skilllink" {
        high
    } else {
        low
    }
}

/// Whether `mv` reaches a foe from this slot, and if so whether as a spread hit (0.75).
fn reach(mv: &Move, live_foes: usize, ally_alive: bool) -> Option<bool> {
    match mv.target.as_str() {
        "normal" | "any" | "adjacentFoe" | "randomNormal" => Some(false),
        "allAdjacentFoes" => Some(live_foes > 1),
        "allAdjacent" => Some(live_foes + usize::from(ally_alive) > 1),
        _ => None,
    }
}

fn priority_blocked(priority: i64, defender: &Battler, field: &FieldState, defender_side: usize) -> bool {
    if priority <= 0 {
        return false;
    }
    if matches!(field.terrain, Some(t) if t.as_str() == "psychicterrain") && is_grounded(defender) {
        return true;
    }
    field.active_abilities[defender_side]
        .iter()
        .any(|a| matches!(a.as_str(), "dazzling" | "queenlymajesty" | "armortail"))
}

/// The moves this Pokemon can use: PP left, not disabled, the locked one only if locked.
fn usable_moves(mon: &Pokemon) -> Vec<Id> {
    let locked = crate::encode::locked_move_of(mon);
    mon.moves
        .iter()
        .filter(|slot| slot.pp > 0 && !slot.disabled)
        .filter(|slot| !matches!(locked, Some(l) if l != slot.id))
        .map(|slot| slot.id)
        .collect()
}

/// Every usable move of `attacker` against `defender`, with rolls (the slow part). With
/// `on_ally` the defender is the attacker's partner, `defender_side` the attacker's own, and
/// only the moves that hit every adjacent Pokemon (`allAdjacent`: Earthquake, Surf) count.
#[allow(clippy::too_many_arguments)]
fn evaluate_pair(
    reg: &Reg,
    field: &FieldState,
    attacker: &Battler,
    defender: &Battler,
    moves: &[Id],
    defender_side: usize,
    live_foes: usize,
    ally_alive: bool,
    ctx: &MoveContext,
    on_ally: bool,
) -> PairValue {
    let mut out = PairValue::default();
    for (index, move_id) in moves.iter().enumerate() {
        let move_id = move_id.as_str();
        let Some(mv) = reg.moves.get(move_id) else { continue };
        if mv.category == "Status" {
            continue;
        }
        if on_ally && mv.target.as_str() != "allAdjacent" {
            continue;
        }
        let Some(spread) = reach(mv, live_foes, ally_alive) else { continue };
        let spread_kind = matches!(mv.target.as_str(), "allAdjacentFoes" | "allAdjacent");
        let priority = move_priority(reg, move_id, attacker, field);
        if !on_ally && priority_blocked(priority, defender, field, defender_side) {
            continue;
        }
        let result = calculate(
            reg, attacker, defender, move_id, field, defender_side, spread, mv.will_crit, Some(ctx),
            None, false,
        );
        if result.immune {
            continue;
        }
        let first = if priority > 0 {
            2
        } else if priority < 0 {
            0
        } else {
            1
        };
        out.moves.push(MoveEval {
            index: index as u8,
            spread_kind,
            first,
            hits: hit_count(mv, attacker.ability.as_str()),
            rolls: result.rolls,
        });
    }
    out
}

/// Rolls (out of 16) of `eval` that take `defender`'s current HP.
#[allow(dead_code)]
fn ko_rolls(eval: &MoveEval, defender: &Battler) -> u32 {
    let hp = defender.hp;
    if hp <= 0 {
        return 0;
    }
    if eval.hits > 1 {
        // A second hit goes through Focus Sash and Sturdy; the same roll each hit.
        return eval.rolls.iter().filter(|d| **d * eval.hits >= hp).count() as u32;
    }
    effective_damage(&eval.rolls, defender).iter().filter(|d| **d >= hp).count() as u32
}

/// `ko_rolls`, with `effective_damage` read through: a single hit is capped at the HP, and
/// with Focus Sash or Sturdy at full HP at one below it, so it takes the HP exactly when
/// its roll reaches the HP and no such survival applies (`holds` is that survival, worked
/// out once per defender).
#[inline]
fn ko_count(eval: &MoveEval, hp: i64, holds: bool) -> u32 {
    if hp <= 0 {
        return 0;
    }
    if eval.hits > 1 {
        return eval.rolls.iter().filter(|d| **d * eval.hits >= hp).count() as u32;
    }
    if holds {
        return 0;
    }
    eval.rolls.iter().filter(|d| **d >= hp).count() as u32
}

/// Whether `effective_damage` leaves this defender at 1 HP from any single hit.
fn survives_one_hit(defender: &Battler) -> bool {
    use crate::effects::{survives_at_one_ability, survives_at_one_item};
    let survives = defender.item.map(|i| survives_at_one_item(i.as_str())).unwrap_or(false)
        || survives_at_one_ability(defender.ability.as_str());
    survives && defender.hp >= defender.maxhp
}

/// The key words of one Pokemon: every field `Battler::from_pokemon` and `usable_moves`
/// read, the HP only through the flags the calculator and the priority read, and the move
/// context only where a move reads it.
fn mon_words(words: &mut Words, mon: &Pokemon) {
    words.0.clear();
    words.id(mon.species);
    let types = mon.types.as_slice();
    words.int(types.len() as i64);
    for kind in types {
        words.id(*kind);
    }
    words.id(mon.ability);
    words.opt(mon.item);
    words.int(mon.level);
    match mon.sp {
        None => words.int(-1),
        Some(sp) => {
            for value in sp {
                words.int(value);
            }
        }
    }
    words.id(mon.nature);
    match mon.stats_override {
        None => words.int(-1),
        Some(stats) => {
            for value in stats {
                words.int(value);
            }
        }
    }
    words.int(i64::from(mon.transformed));
    words.int(mon.maxhp);
    // The HP flags (`effects.rs`: Blaze and the like, Defeatist, Multiscale and Shadow
    // Shield; `speed.rs`: Gale Wings), each only for the ability that reads it. Without
    // that, a Pokemon's key would change with every hit.
    let (hp, max) = (mon.hp, mon.maxhp);
    let ability = mon.ability.as_str();
    let pinch = matches!(ability, "overgrow" | "blaze" | "torrent" | "swarm") && hp * 3 <= max;
    let half = ability == "defeatist" && hp * 2 <= max;
    let full = matches!(ability, "multiscale" | "shadowshield" | "galewings") && hp >= max;
    let flags = i64::from(pinch) | i64::from(half) << 1 | i64::from(full) << 2;
    #[cfg(feature = "ika429-cache-control")]
    let flags = { let _ = flags; 0 };
    words.int(flags);
    let mut boosts = 0u64;
    for (i, b) in mon.boosts.iter().enumerate() {
        boosts |= (*b as u8 as u64) << (8 * i);
    }
    words.0.push(boosts);
    words.opt(mon.status);
    let mut volatiles = VolatileFlags::default();
    let mut roost = false;
    for effect in &mon.volatiles {
        volatiles.add(effect.id.as_str());
        roost |= effect.id.as_str() == "roost";
    }
    words.int(i64::from(volatiles.bits()) | i64::from(roost) << 16);
    words.id(mon.gender);
    if matches!(ability, "protean" | "libero") {
        let fired = mon
            .ability_state
            .get("protean")
            .map(|v| v != &serde_json::Value::Bool(false) && !v.is_null())
            .unwrap_or(false);
        words.int(i64::from(fired));
    }
    if ability == "supremeoverlord" {
        words.int(mon.ability_state.get("fallen").and_then(|v| v.as_i64()).unwrap_or(0));
    }
    // The moves and what decides which are usable.
    let mut reads = [false; 3];
    for slot in mon.moves.iter() {
        words.id(slot.id);
        words.int(i64::from(slot.pp > 0) | i64::from(slot.disabled) << 1);
        match slot.id.as_str() {
            "lastrespects" => reads[0] = true,
            "ragefist" => reads[1] = true,
            "stompingtantrum" | "temperflare" => reads[2] = true,
            _ => {}
        }
    }
    words.opt(crate::encode::locked_move_of(mon));
    // `side_total_fainted` is the caller's: it is not the Pokemon's own.
    words.int(if reads[1] { mon.times_attacked } else { -1 });
    words.int(i64::from(reads[2] && mon.move_last_turn_failed));
    words.int(i64::from(reads[0]));
}

/// The field's key words, read off the position as `field_state` builds the field.
fn field_words(words: &mut Words, pos: &Position) {
    words.0.clear();
    words.opt(pos.field.weather);
    words.opt(pos.field.terrain);
    words.int(pos.field.pseudo_weather.len() as i64);
    for effect in &pos.field.pseudo_weather {
        words.id(effect.id);
    }
    for side in &pos.sides {
        words.int(side.side_conditions.len() as i64);
        for effect in &side.side_conditions {
            words.id(effect.id);
        }
        let mut count = 0i64;
        for slot in 0..side.active.len() {
            if let Some(mon) = side.active_pokemon(slot) {
                if !mon.fainted {
                    words.id(mon.ability);
                    let shielded = matches!(mon.item, Some(i) if i.as_str() == "abilityshield");
                    words.int(i64::from(shielded));
                    count += 1;
                }
            }
        }
        words.int(count);
    }
    words.int(pos.sides[0].active.len() as i64);
}

/// One live active Pokemon of a position.
struct Active {
    index: usize,
    slot: usize,
    battler: Battler,
    entry: Rc<MonEntry>,
    ctx: MoveContext,
    speed: i64,
    /// `survives_one_hit`.
    holds: bool,
}

/// What `bind` gives the encoder: four columns for every Pokemon row (`2 * m` rows,
/// side-major, party order, as `mon` is laid out: an active Pokemon's row at its party
/// index, every other row 0), and `BIND_SIDE_FEATURES` for each side.
pub struct Bound {
    pub rows: Vec<[f32; BIND_FEATURES]>,
    pub sides: [[f32; BIND_SIDE_FEATURES]; 2],
}

/// `bind` without the side columns.
#[allow(dead_code)]
pub fn bind_rows(reg: &Reg, pos: &Position, m: usize, cache: Option<&mut BindCache>) -> Vec<[f32; BIND_FEATURES]> {
    bind(reg, pos, m, cache).rows
}

/// The pair's value from the cache (made by `make` and kept, on a miss), or made afresh.
fn lookup(cache: Option<&mut BindCache>, key: [u64; 7], make: impl FnOnce() -> PairValue) -> Rc<PairValue> {
    match cache {
        None => Rc::new(make()),
        Some(cache) => {
            if let Some(found) = cache.pairs.get(&key) {
                cache.hits += 1;
                return found.clone();
            }
            cache.misses += 1;
            let made = Rc::new(make());
            cache.pairs.insert(key, made.clone());
            made
        }
    }
}

pub fn bind(reg: &Reg, pos: &Position, m: usize, cache: Option<&mut BindCache>) -> Bound {
    let mut rows = vec![[0.0f32; BIND_FEATURES]; 2 * m];
    let mut sides = [[0.0f32; BIND_SIDE_FEATURES]; 2];
    let mut cache = cache;
    if let Some(cache) = cache.as_deref_mut() {
        // Before anything is looked up, so nothing found here is emptied under this call.
        cache.check_size();
    }
    let mut actives: [Vec<Active>; 2] = [Vec::with_capacity(2), Vec::with_capacity(2)];
    let mut next_id = 0u64;
    for (s, side) in pos.sides.iter().enumerate().take(2) {
        for (slot, index) in side.active.iter().enumerate().take(2) {
            let Some(index) = *index else { continue };
            let Some(mon) = side.pokemon.get(index) else { continue };
            if mon.fainted || mon.hp <= 0 || index >= m {
                continue;
            }
            let entry = match cache.as_deref_mut() {
                Some(cache) => {
                    let mut words = std::mem::take(&mut cache.words);
                    mon_words(&mut words, mon);
                    let found = cache.mons.get(&words.0[..]).cloned();
                    let entry = match found {
                        Some(entry) => Some(entry),
                        None => match Battler::from_pokemon(reg, mon) {
                            Err(_) => None,
                            Ok(battler) => {
                                let made = Rc::new(MonEntry::new(cache.number(), battler, mon));
                                cache.mons.insert(words.0.clone().into_boxed_slice(), made.clone());
                                Some(made)
                            }
                        },
                    };
                    cache.words = words;
                    entry
                }
                None => Battler::from_pokemon(reg, mon).ok().map(|battler| {
                    next_id += 1;
                    Rc::new(MonEntry::new(next_id, battler, mon))
                }),
            };
            let Some(entry) = entry else { continue };
            let mut battler = entry.battler;
            battler.hp = mon.hp;
            let holds = survives_one_hit(&battler);
            actives[s].push(Active {
                index,
                slot,
                battler,
                entry,
                ctx: move_context(pos, s, mon),
                speed: 0,
                holds,
            });
        }
    }
    if actives[0].is_empty() || actives[1].is_empty() {
        return Bound { rows, sides };
    }
    let (field_key, field): (u64, Rc<FieldState>) = match cache.as_deref_mut() {
        Some(cache) => {
            let mut words = std::mem::take(&mut cache.words);
            field_words(&mut words, pos);
            let found = cache.fields.get(&words.0[..]).cloned();
            let got = match found {
                Some(found) => found,
                None => {
                    let made = (cache.number(), Rc::new(field_state(pos)));
                    cache.fields.insert(words.0.clone().into_boxed_slice(), made.clone());
                    made
                }
            };
            cache.words = words;
            got
        }
        None => (0, Rc::new(field_state(pos))),
    };
    let field: &FieldState = &field;
    let trick_room = field.trick_room();
    for (s, list) in actives.iter_mut().enumerate() {
        for active in list.iter_mut() {
            active.speed = effective_speed(&active.battler, field, &pos.sides[s].side_conditions);
        }
    }
    let hp_key = |exact: bool, hp: i64| if exact { hp as u64 } else { u64::MAX };
    let fainted_key = |attacker: &Active| {
        if attacker.entry.reads_fainted {
            attacker.ctx.side_total_fainted as u64
        } else {
            u64::MAX
        }
    };

    // bind in 1/32 units (first 0..2 times rolls 0..16), ko in 1/16, summed per Pokemon.
    let mut bind_out = [[0u32; 2]; 2];
    let mut bind_in = [[0u32; 2]; 2];
    let mut ko_out = [[0u32; 2]; 2];
    let mut ko_in = [[0u32; 2]; 2];
    // matrix[s][a][d]: bind of side s's a-th active on its d-th foe, in 1/32.
    let mut matrix = [[[0u32; 2]; 2]; 2];
    // spread[s][a][k][d]: the same for each of a's moves that reach both foes (k its place).
    let mut spread = [[[[0u32; 2]; 4]; 2]; 2];
    for s in 0..2 {
        let foe_side = 1 - s;
        let live_foes = actives[foe_side].len();
        let ally_alive = actives[s].len() > 1;
        for (a, attacker) in actives[s].iter().enumerate() {
            let hp_exact = attacker.entry.hp_exact;
            #[cfg(feature = "ika429-cache-control")]
            let hp_exact = { let _ = hp_exact; false };
            for (d, defender) in actives[foe_side].iter().enumerate() {
                let (mine, theirs) = (attacker.speed, defender.speed);
                let (mine, theirs) = if trick_room { (theirs, mine) } else { (mine, theirs) };
                let by_speed: u32 = if mine > theirs {
                    2
                } else if mine < theirs {
                    0
                } else {
                    1
                };
                let key = [
                    field_key,
                    (foe_side as u64) | (live_foes as u64) << 8 | u64::from(ally_alive) << 16,
                    attacker.entry.id,
                    defender.entry.id,
                    hp_key(hp_exact, attacker.battler.hp),
                    hp_key(hp_exact, defender.battler.hp),
                    fainted_key(attacker),
                ];
                let value = lookup(cache.as_deref_mut(), key, || {
                    evaluate_pair(
                        reg, field, &attacker.battler, &defender.battler, &attacker.entry.moves, foe_side,
                        live_foes, ally_alive, &attacker.ctx, false,
                    )
                });
                let mut best_bind = 0u32;
                let mut best_ko = 0u32;
                for eval in &value.moves {
                    let rolls = ko_count(eval, defender.battler.hp, defender.holds);
                    #[cfg(feature = "bind-verify")]
                    assert_eq!(rolls, ko_rolls(eval, &defender.battler), "bind-verify: ko_count");
                    if rolls == 0 {
                        continue;
                    }
                    let first = if eval.first == 1 { by_speed } else { u32::from(eval.first) };
                    best_ko = best_ko.max(rolls);
                    best_bind = best_bind.max(first * rolls);
                    if eval.spread_kind {
                        spread[s][a][eval.index as usize][d] = first * rolls;
                    }
                }
                matrix[s][a][d] = best_bind;
                bind_out[s][a] += best_bind;
                bind_in[foe_side][d] += best_bind;
                ko_out[s][a] += best_ko;
                ko_in[foe_side][d] += best_ko;
            }
        }
    }
    for s in 0..2 {
        for (a, active) in actives[s].iter().enumerate() {
            // Sums over at most two foes, halved: bind is in 1/32, so / 64; ko in 1/16, / 32.
            rows[s * m + active.index] = [
                bind_out[s][a] as f32 / 64.0,
                bind_in[s][a] as f32 / 64.0,
                ko_out[s][a] as f32 / 32.0,
                ko_in[s][a] as f32 / 32.0,
            ];
        }
    }
    // The side columns, in integers: bind in 1/32, so a product of two in 1/1024.
    for s in 0..2 {
        let foe_side = 1 - s;
        let mine = actives[s].len();
        let foes = actives[foe_side].len();
        let mut bound = 0u32;
        let mut secure = 0u32;
        let mut least = u32::MAX;
        for j in 0..foes {
            let held = (0..mine).map(|i| matrix[s][i][j]).max().unwrap_or(0);
            bound += held;
            least = least.min(held);
            // The best bind on j whose binder no foe binds back.
            let kept = (0..mine)
                .map(|i| {
                    let back = (0..foes).map(|k| matrix[foe_side][k][i]).max().unwrap_or(0);
                    matrix[s][i][j] * (32 - back)
                })
                .max()
                .unwrap_or(0);
            secure += kept;
        }
        // One Pokemon's one spread move binding both foes, and what that move does to its
        // partner: the best such bind, and of the moves that give it the least partner damage.
        let mut spread_best = 0u32;
        let mut spread_ally = 0u32;
        let mut spread_by: Option<(usize, usize)> = None;
        if foes == 2 {
            for a in 0..mine {
                for k in 0..4 {
                    let both = spread[s][a][k][0].min(spread[s][a][k][1]);
                    if both == 0 || both < spread_best {
                        continue;
                    }
                    let ally_rolls = if mine == 2 {
                        let attacker = &actives[s][a];
                        let partner = &actives[s][1 - a];
                        let hp_exact = attacker.entry.hp_exact;
                        // The partner as the defender, on the attacker's own side (bit 24).
                        let key = [
                            field_key,
                            (s as u64) | (foes as u64) << 8 | 1 << 16 | 1 << 24,
                            attacker.entry.id,
                            partner.entry.id,
                            hp_key(hp_exact, attacker.battler.hp),
                            hp_key(hp_exact, partner.battler.hp),
                            fainted_key(attacker),
                        ];
                        let value = lookup(cache.as_deref_mut(), key, || {
                            evaluate_pair(
                                reg, field, &attacker.battler, &partner.battler, &attacker.entry.moves, s,
                                foes, true, &attacker.ctx, true,
                            )
                        });
                        value
                            .moves
                            .iter()
                            .find(|eval| eval.index as usize == k)
                            .map(|eval| ko_count(eval, partner.battler.hp, partner.holds))
                            .unwrap_or(0)
                    } else {
                        0
                    };
                    if both > spread_best || ally_rolls < spread_ally {
                        spread_best = both;
                        spread_ally = ally_rolls;
                        spread_by = Some((a, k));
                    }
                }
            }
        }
        // What the bound side has against that one Pokemon's spread move.
        let (mut wide_guard, mut fake_out) = (0.0f32, 0.0f32);
        if let Some((a, k)) = spread_by {
            let binder = &actives[s][a];
            let binder_priority = move_priority(reg, binder.entry.moves[k].as_str(), &binder.battler, field);
            for foe in &actives[foe_side] {
                if foe.entry.moves.iter().any(|id| id.as_str() == RELEASE_WIDE_GUARD) {
                    wide_guard = 1.0;
                }
                let fresh = pos.mon_at(foe_side, foe.slot).map(|mon| mon.active_move_actions == 0);
                if fresh == Some(true)
                    && foe.entry.moves.iter().any(|id| id.as_str() == RELEASE_FAKE_OUT)
                    && fake_out_stops(reg, field, foe, binder, binder_priority, s, trick_room)
                {
                    fake_out = 1.0;
                }
            }
        }
        sides[s] = [
            bound as f32 / 64.0,
            if foes == 2 { least as f32 / 32.0 } else { 0.0 },
            secure as f32 / 2048.0,
            spread_best as f32 / 32.0,
            spread_ally as f32 / 16.0,
            wide_guard,
            fake_out,
        ];
    }
    let _ = next_id;
    Bound { rows, sides }
}

/// Whether a Fake Out from `user` stops `binder`'s move this turn: it goes first (its
/// priority over the binder's move's, or the same priority and the higher Speed, Trick Room
/// reversing it; a tie does not count), it is not blocked (Psychic Terrain on a grounded
/// binder, Dazzling, Queenly Majesty or Armor Tail beside it), and the binder flinches (not
/// a Ghost, no Inner Focus or Shield Dust, no Covert Cloak). The user's own Fake Out being
/// its first move out is the caller's to check.
fn fake_out_stops(
    reg: &Reg,
    field: &FieldState,
    user: &Active,
    binder: &Active,
    binder_priority: i64,
    binder_side: usize,
    trick_room: bool,
) -> bool {
    let priority = move_priority(reg, RELEASE_FAKE_OUT, &user.battler, field);
    let first = if priority != binder_priority {
        priority > binder_priority
    } else if trick_room {
        user.speed < binder.speed
    } else {
        user.speed > binder.speed
    };
    let target = &binder.battler;
    if !first || priority_blocked(priority, target, field, binder_side) {
        return false;
    }
    if target.types.contains("Ghost") {
        return false;
    }
    if matches!(target.ability.as_str(), "innerfocus" | "shielddust") {
        return false;
    }
    !matches!(target.item, Some(i) if i.as_str() == "covertcloak")
}
