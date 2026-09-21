//! Single-turn resolver, transcribed from `src/pokeuraou/resolve.py`.
//!
//! Generation spends 96.7% of its wall clock in `resolve_turn` (measured with counters;
//! cProfile misattributes a sixth of the run to scipy), so this is the whole target of the
//! port.
//!
//! **It refuses rather than guesses.** Every entry point returns `Err(reason)` when it
//! meets something this port does not model -- an ability outside the handled set, a move
//! field it does not implement, a mid-turn replacement. The caller falls back to Python for
//! that turn. That is what makes a partial port useful: a turn Rust resolves is exactly
//! Python's answer, a turn it refuses costs nothing but Python's own time, and the refusal
//! reasons are counted so the next thing to implement is chosen by impact.
//!
//! Two things Python produces are deliberately absent: the event log and the per-action
//! `acts` offsets. They are for display, they allocate on every action, and nothing in the
//! differential compares them.

use crate::battler::{Battler, FieldState};
use crate::damage::{self, crit_probability};
use crate::effects::survive_chance_item;
use crate::id::Id;
use crate::moveinfo::MoveContext;
use crate::position::{boost_index, Effect, Pokemon, Position, Types, BOOST_IDS};
use crate::reg::{Move, Reg};
use std::rc::Rc;

use crate::speed::{
    effective_speed, fractional_priority, move_priority, order_actions, ActionKind, QueuedAction,
    ORDER_MEGA, ORDER_MOVE, ORDER_SWITCH,
};
use crate::moves::{do_move, residuals};
use serde_json::Value;

pub type Slot = (usize, usize);

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Budget {
    pub damage_rolls: i64,
    pub enumerate_crit: bool,
    pub enumerate_accuracy: bool,
    pub enumerate_status_checks: bool,
    pub enumerate_secondary: bool,
    pub enumerate_speed_ties: bool,
    pub pinned_policy: bool,
    pub max_branches: usize,
    /// Fold branches that reached the same state into one. Lossless, so it is on; see
    /// `Budget.merge_duplicates` in Python, which this mirrors field for field.
    pub merge_duplicates: bool,
}

impl Budget {
    /// The setting a matrix is filled with: the median roll, no crit branch, everything
    /// else enumerated. `Budget.matrix()` in Python.
    pub fn matrix() -> Budget {
        Budget {
            damage_rolls: -1 - 8,
            enumerate_crit: false,
            enumerate_accuracy: true,
            enumerate_status_checks: true,
            enumerate_secondary: true,
            enumerate_speed_ties: true,
            pinned_policy: false,
            max_branches: 16,
            merge_duplicates: true,
        }
    }

    /// The budget the fixture recorded for this call. A generated game uses two -- the
    /// matrix budget for the search and an exact one to advance the game -- and reading
    /// the recorded fields is what keeps the comparison an equality test.
    pub fn from_json(value: &Value) -> Budget {
        Budget {
            damage_rolls: value["damageRolls"].as_i64().unwrap_or(16),
            enumerate_crit: value["enumerateCrit"].as_bool().unwrap_or(true),
            enumerate_accuracy: value["enumerateAccuracy"].as_bool().unwrap_or(true),
            enumerate_status_checks: value["enumerateStatusChecks"].as_bool().unwrap_or(true),
            enumerate_secondary: value["enumerateSecondary"].as_bool().unwrap_or(true),
            enumerate_speed_ties: value["enumerateSpeedTies"].as_bool().unwrap_or(true),
            pinned_policy: value["pinnedPolicy"].as_bool().unwrap_or(false),
            max_branches: value["maxBranches"].as_u64().unwrap_or(512) as usize,
            // Defaulted to on, as Python's field is: a fixture recorded before the merge
            // existed replays with it rather than against it.
            merge_duplicates: value["mergeDuplicates"].as_bool().unwrap_or(true),
        }
    }

    pub fn fixed_roll(&self) -> Option<usize> {
        if self.damage_rolls >= 0 {
            None
        } else {
            Some((-self.damage_rolls - 1) as usize)
        }
    }

    pub fn per_action_branches(&self) -> usize {
        let rolls = if self.fixed_roll().is_some() {
            1
        } else {
            self.damage_rolls.clamp(1, 16) as usize
        };
        let mut factor = rolls;
        if self.enumerate_crit {
            factor *= 2;
        }
        if self.enumerate_accuracy {
            factor *= 2;
        }
        if self.enumerate_status_checks {
            factor *= 2;
        }
        factor
    }

    /// This budget, reduced until one action produces at most `room` outcomes.
    pub fn narrowed(&self, room: usize) -> Budget {
        if room >= self.per_action_branches() || self.fixed_roll().is_some() {
            return *self;
        }
        let mut narrowed = *self;
        while narrowed.damage_rolls > 1 && narrowed.per_action_branches() > room {
            narrowed.damage_rolls = (narrowed.damage_rolls / 2).max(1);
        }
        for field in 0..3 {
            if narrowed.per_action_branches() <= room {
                break;
            }
            match field {
                0 if narrowed.enumerate_crit => narrowed.enumerate_crit = false,
                1 if narrowed.enumerate_status_checks => {
                    narrowed.enumerate_status_checks = false
                }
                2 if narrowed.enumerate_secondary => narrowed.enumerate_secondary = false,
                _ => {}
            }
        }
        if narrowed.per_action_branches() > room && narrowed.enumerate_accuracy {
            narrowed.enumerate_accuracy = false;
        }
        narrowed
    }
}

pub fn stratified_rolls(budget: &Budget) -> Vec<(usize, f64)> {
    if let Some(fixed) = budget.fixed_roll() {
        return vec![(fixed, 1.0)];
    }
    let count = budget.damage_rolls.clamp(1, 16) as usize;
    let mut out = Vec::with_capacity(count);
    let mut edges = Vec::with_capacity(count + 1);
    for index in 0..=count {
        // numpy.linspace(0, 16, count + 1).round()
        let raw = 16.0 * index as f64 / count as f64;
        edges.push(round_half_even(raw) as usize);
    }
    for window in edges.windows(2) {
        let (lo, hi) = (window[0], window[1]);
        if hi <= lo {
            continue;
        }
        out.push(((lo + hi - 1) / 2, (hi - lo) as f64 / 16.0));
    }
    out
}

/// numpy rounds halves to even, and the roll edges land on halves.
fn round_half_even(value: f64) -> f64 {
    let floor = value.floor();
    let diff = value - floor;
    if (diff - 0.5).abs() < 1e-9 {
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    } else {
        value.round()
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

pub struct Branch {
    pub probability: f64,
    pub position: Position,
}

/// A turn Showdown stopped half-way through to ask for a replacement.
///
/// The pause is the honest information state: whoever owes the replacement chooses it
/// without seeing how the rest of the turn goes, so the choice is made once, here, rather
/// than inside each outcome that follows it.
pub struct Suspended<'a> {
    pub probability: f64,
    pub turn: Turn<'a>,
    pub remaining: Vec<QueuedAction>,
}

pub struct TurnResult<'a> {
    pub branches: Vec<Branch>,
    pub exact: bool,
    /// Outcomes that stopped at a mid-turn replacement. `branches` and these together
    /// carry the turn's probability; a caller that ignores them drops that mass.
    pub suspended: Vec<Suspended<'a>>,
    /// The union over every branch, as Python's `TurnResult.unmodelled` is.
    pub unmodelled: std::collections::BTreeSet<String>,
}

impl TurnResult<'_> {
    pub fn is_suspended(&self) -> bool {
        !self.suspended.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Working state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Turn<'a> {
    pub reg: &'a Reg,
    pub pos: Position,
    pub(crate) budget: Budget,
    /// Which active slots are about to use a damaging move, for Sucker Punch.
    pub(crate) attacks: [[bool; 2]; 2],
    pub(crate) hurt_this_turn: [[bool; 2]; 2],
    pub(crate) move_failed: [[bool; 2]; 2],
    pub(crate) move_damage_total: i64,
    pub(crate) move_connected: bool,
    pub(crate) acted: [[bool; 2]; 2],
    pub(crate) actions_remaining: usize,
    pub(crate) self_switch_pending: bool,
    pub(crate) pending_secondaries: Vec<(f64, Value, Slot)>,
    pub(crate) current_actor: Option<Slot>,
    pub(crate) wipe_order: Vec<usize>,
    /// Effects met that this port models only approximately -- the same strings Python
    /// reports, because a caller that prints them must not see the set shrink just
    /// because the turn was resolved in Rust.
    pub(crate) unmodelled: std::collections::BTreeSet<String>,
}

impl<'a> Turn<'a> {
    fn new(reg: &'a Reg, pos: Position, budget: Budget, attacks: [[bool; 2]; 2]) -> Turn<'a> {
        Turn {
            reg,
            pos,
            budget,
            attacks,
            hurt_this_turn: [[false; 2]; 2],
            move_failed: [[false; 2]; 2],
            move_damage_total: 0,
            move_connected: false,
            acted: [[false; 2]; 2],
            actions_remaining: 0,
            self_switch_pending: false,
            pending_secondaries: Vec::new(),
            current_actor: None,
            wipe_order: Vec::new(),
            unmodelled: Default::default(),
        }
    }

    pub(crate) fn report(&mut self, note: impl Into<String>) {
        self.unmodelled.insert(note.into());
    }

    pub fn mon_at(&self, side: usize, slot: usize) -> Option<&Pokemon> {
        self.pos.mon_at(side, slot)
    }

    pub(crate) fn mon_at_mut(&mut self, side: usize, slot: usize) -> Option<&mut Pokemon> {
        self.pos.mon_at_mut(side, slot)
    }

    pub(crate) fn battler_at(&self, side: usize, slot: usize) -> Result<Option<Battler>, String> {
        match self.mon_at(side, slot) {
            None => Ok(None),
            Some(mon) if mon.fainted => Ok(None),
            Some(mon) => Ok(Some(Battler::from_pokemon(self.reg, mon)?)),
        }
    }

    pub(crate) fn types_of(&self, mon: &Pokemon) -> Types {
        damage::types_or_species(self.reg, mon.species.as_str(), mon.types)
    }

    pub(crate) fn field(&self) -> FieldState {
        field_state(&self.pos)
    }

    pub(crate) fn fraction_of_max(&self, side: usize, slot: usize, ratio: (i64, i64)) -> i64 {
        match self.mon_at(side, slot) {
            None => 0,
            Some(mon) => (mon.maxhp * ratio.0 / ratio.1).max(1),
        }
    }

    /// Applies damage, honouring survival effects, and returns what was dealt.
    pub(crate) fn deal_damage(
        &mut self,
        side: usize,
        slot: usize,
        amount: i64,
        from_move: bool,
    ) -> Result<i64, String> {
        let (mut dealt, uses_sash) = {
            let Some(mon) = self.mon_at(side, slot) else { return Ok(0) };
            if mon.fainted || amount <= 0 {
                return Ok(0);
            }
            let mut dealt = amount.min(mon.hp);
            let mut uses_sash = false;
            if from_move && dealt >= mon.hp {
                if mon.has_volatile("endure") {
                    dealt = mon.hp - 1;
                } else if mon.hp >= mon.maxhp
                    && (mon.item.map(|i| i.as_str() == "focussash").unwrap_or(false)
                        || mon.ability == "sturdy")
                {
                    dealt = mon.hp - 1;
                    uses_sash = mon.item.map(|i| i.as_str() == "focussash").unwrap_or(false);
                } else if mon.item.map(|i| survive_chance_item(i.as_str()).is_some()).unwrap_or(false)
                {
                    return Err("survival chance item (focusband) is not branched".into());
                }
            }
            (dealt, uses_sash)
        };
        if uses_sash {
            self.consume_item(side, slot);
        }
        if dealt < 0 {
            dealt = 0;
        }
        let fainted = {
            let mon = self.mon_at_mut(side, slot).unwrap();
            mon.hp -= dealt;
            mon.hp <= 0
        };
        self.hurt_this_turn[side][slot] = true;
        if fainted {
            self.faint(side, slot);
        } else {
            self.check_berry(side, slot);
        }
        Ok(dealt)
    }

    pub(crate) fn heal(&mut self, side: usize, slot: usize, amount: i64) -> i64 {
        let Some(mon) = self.mon_at_mut(side, slot) else { return 0 };
        if mon.fainted || amount <= 0 {
            return 0;
        }
        let healed = amount.min(mon.maxhp - mon.hp);
        mon.hp += healed;
        healed
    }

    pub(crate) fn faint(&mut self, side: usize, slot: usize) {
        let Some(mon) = self.mon_at_mut(side, slot) else { return };
        if mon.fainted {
            return;
        }
        mon.hp = 0;
        mon.fainted = true;
        mon.boosts = [0; 7];
        mon.volatiles.clear();
        mon.status = Some(Id::new("fnt"));
        mon.status_counter = None;
        let wiped = self.pos.sides[side].pokemon.iter().all(|m| m.fainted);
        if wiped && !self.wipe_order.contains(&side) {
            self.wipe_order.push(side);
        }
    }

    pub(crate) fn consume_item(&mut self, side: usize, slot: usize) {
        let Some(mon) = self.mon_at_mut(side, slot) else { return };
        if mon.item.is_none() {
            return;
        }
        if mon.ability == "unburden" && !mon.has_volatile("unburden") {
            mon.volatiles.push(Effect::new(Id::new("unburden")));
        }
        mon.item = None;
    }

    pub(crate) fn berries_blocked(&self, side: usize) -> bool {
        let foe_side = 1 - side;
        (0..self.pos.sides[foe_side].active.len()).any(|slot| {
            matches!(self.mon_at(foe_side, slot), Some(foe)
                if !foe.fainted
                    && matches!(
                        foe.ability.as_str(),
                        "unnerve" | "asonechillingneigh" | "asonegrimneigh"
                    ))
        })
    }

    pub(crate) fn check_berry(&mut self, side: usize, slot: usize) {
        let Some(mon) = self.mon_at(side, slot) else { return };
        let Some(item) = mon.item else { return };
        if mon.fainted {
            return;
        }
        let is_pinch_berry = matches!(item.as_str(), "sitrusberry" | "oranberry");
        if !is_pinch_berry || mon.hp * 2 > mon.maxhp {
            return;
        }
        if self.berries_blocked(side) {
            return;
        }
        let amount = if item.as_str() == "sitrusberry" {
            (mon.maxhp / 4).max(1)
        } else {
            10
        };
        self.consume_item(side, slot);
        self.heal(side, slot, amount);
    }

    /// Applies a boost table, and says whether any stat actually moved.
    pub(crate) fn apply_boosts(
        &mut self,
        side: usize,
        slot: usize,
        boosts: &[(&str, i64)],
        from_foe: bool,
    ) -> bool {
        let Some(mon) = self.mon_at(side, slot) else { return false };
        if mon.fainted {
            return false;
        }
        let contrary = mon.ability == "contrary";
        let blocks_drops = matches!(
            mon.ability.as_str(),
            "clearbody" | "whitesmoke" | "fullmetalbody"
        );
        let mut changed = false;
        for (stat, raw_delta) in boosts {
            let delta = if contrary { -raw_delta } else { *raw_delta };
            let Some(index) = boost_index(stat) else {
                self.report(format!("boost:{stat}"));
                continue;
            };
            if delta < 0 && from_foe && blocks_drops {
                continue;
            }
            let mon = self.mon_at_mut(side, slot).unwrap();
            let before = mon.boosts[index] as i64;
            let after = (before + delta).clamp(-6, 6);
            if after == before {
                continue;
            }
            mon.boosts[index] = after as i8;
            changed = true;
            if delta < 0 && from_foe {
                self.on_stat_lowered_by_foe(side, slot);
            }
        }
        changed
    }

    fn on_stat_lowered_by_foe(&mut self, side: usize, slot: usize) {
        let ability = match self.mon_at(side, slot) {
            None => return,
            Some(mon) => mon.ability,
        };
        if ability == "defiant" {
            self.apply_boosts(side, slot, &[("atk", 2)], false);
        } else if ability == "competitive" {
            self.apply_boosts(side, slot, &[("spa", 2)], false);
        }
    }

    pub(crate) fn apply_status(&mut self, side: usize, slot: usize, status: &str) -> Result<bool, String> {
        let (types, ability, grounded) = {
            let Some(mon) = self.mon_at(side, slot) else { return Ok(false) };
            if mon.fainted || mon.status.is_some() {
                return Ok(false);
            }
            (self.types_of(mon), mon.ability, grounded(self, mon))
        };
        let immune_by_type: &[&str] = match status {
            "psn" | "tox" => &["Poison", "Steel"],
            "brn" => &["Fire"],
            "par" => &["Electric"],
            "frz" => &["Ice"],
            _ => &[],
        };
        if immune_by_type.iter().any(|t| types.contains(t)) {
            return Ok(false);
        }
        if matches!(
            ability.as_str(),
            "immunity"
                | "limber"
                | "waterveil"
                | "insomnia"
                | "vitalspirit"
                | "comatose"
                | "purifyingsalt"
                | "thermalexchange"
        ) {
            return Ok(false);
        }
        let terrain = self.pos.field.terrain;
        if matches!(terrain, Some(t) if t.as_str() == "mistyterrain") && grounded {
            return Ok(false);
        }
        if status == "slp"
            && matches!(terrain, Some(t) if t.as_str() == "electricterrain")
            && grounded
        {
            return Ok(false);
        }
        let sleep_counter = if self.budget.pinned_policy {
            SLEEP_COUNTER_PINNED
        } else {
            SLEEP_COUNTER_MODAL
        };
        if status == "slp" && !self.budget.pinned_policy {
            self.report(format!(
                "sleep duration ({} turns, the modal outcome of Champions' 1-or-2; not branched)",
                SLEEP_COUNTER_MODAL - 1
            ));
        }
        let lum = {
            let mon = self.mon_at_mut(side, slot).unwrap();
            mon.status = Some(Id::new(status));
            mon.status_counter = match status {
                "tox" => Some(0),
                "frz" => Some(FREEZE_COUNTER),
                // Champions' sleep is `sample([2, 3, 3])`. Python uses the modal
                // outcome and reports it; under the pinned policy the oracle's `sample`
                // returns the first element. Both are deterministic, so both are matched.
                "slp" => Some(sleep_counter),
                _ => None,
            };
            mon.item.map(|i| i.as_str() == "lumberry").unwrap_or(false)
        };
        if lum {
            self.consume_item(side, slot);
            let mon = self.mon_at_mut(side, slot).unwrap();
            mon.status = None;
            mon.status_counter = None;
        }
        Ok(true)
    }

    pub(crate) fn add_volatile(&mut self, side: usize, slot: usize, vid: &str, duration: Option<i64>) {
        let Some(mon) = self.mon_at_mut(side, slot) else { return };
        if mon.fainted || mon.has_volatile(vid) {
            return;
        }
        let mut effect = Effect::new(Id::new(vid));
        effect.duration = duration;
        mon.volatiles.push(effect);
    }

    pub(crate) fn add_side_condition(&mut self, side: usize, cid: &str, duration: Option<i64>) {
        if let Some(existing) = self.pos.sides[side]
            .side_conditions
            .iter_mut()
            .find(|c| c.id.as_str() == cid)
        {
            if cid == "spikes" || cid == "toxicspikes" {
                let cap = if cid == "spikes" { 3 } else { 2 };
                existing.layers = Some((existing.layers.unwrap_or(1) + 1).min(cap));
            }
            return;
        }
        let mut effect = Effect::new(Id::new(cid));
        effect.duration = duration;
        effect.layers = Some(1);
        self.pos.sides[side].side_conditions.push(effect);
    }
}

pub const FREEZE_COUNTER: i64 = 3;
pub const SLEEP_COUNTER_PINNED: i64 = 2;
pub const SLEEP_COUNTER_MODAL: i64 = 3;
pub const FULL_PARALYSIS_CHANCE: f64 = 1.0 / 8.0;
pub const CONFUSION_SELF_HIT_CHANCE: f64 = 1.0 / 3.0;
pub const THAW_CHANCE: f64 = 0.25;

const BURN_DAMAGE: (i64, i64) = (1, 16);
const POISON_DAMAGE: (i64, i64) = (1, 8);
const SANDSTORM_DAMAGE: (i64, i64) = (1, 16);
const LEECH_SEED_DRAIN: (i64, i64) = (1, 8);
const LEFTOVERS_HEAL: (i64, i64) = (1, 16);
const PARTIAL_TRAP_DAMAGE: (i64, i64) = (1, 8);
const SALT_CURE_DAMAGE: (i64, i64) = (1, 8);
const SALT_CURE_DAMAGE_WEAK: (i64, i64) = (1, 4);

pub(crate) fn grounded(turn: &Turn, mon: &Pokemon) -> bool {
    if mon.has_volatile("smackdown")
        || mon.has_volatile("ingrain")
        || matches!(mon.item, Some(i) if i.as_str() == "ironball")
    {
        return true;
    }
    if mon.has_volatile("magnetrise") || mon.has_volatile("telekinesis") {
        return false;
    }
    if mon.ability == "levitate" || matches!(mon.item, Some(i) if i.as_str() == "airballoon") {
        return false;
    }
    !turn.types_of(mon).contains("Flying")
}

pub fn field_state(pos: &Position) -> FieldState {
    let mut abilities: [Vec<Id>; 2] = [Vec::new(), Vec::new()];
    let mut conditions: [Vec<Id>; 2] = [Vec::new(), Vec::new()];
    for (index, side) in pos.sides.iter().enumerate() {
        for slot in 0..side.active.len() {
            if let Some(mon) = side.active_pokemon(slot) {
                if !mon.fainted {
                    abilities[index].push(mon.ability);
                }
            }
        }
        conditions[index] = side.side_conditions.iter().map(|c| c.id).collect();
    }
    FieldState {
        weather: pos.field.weather,
        terrain: pos.field.terrain,
        pseudo_weather: pos.field.pseudo_weather.iter().map(|p| p.id).collect(),
        side_conditions: conditions,
        active_abilities: abilities,
        active_per_half: pos.sides[0].active.len() as i64,
    }
}

// ---------------------------------------------------------------------------
// What this port claims to handle
// ---------------------------------------------------------------------------

/// Abilities whose effect on a turn this port reproduces -- either because it implements
/// them or because they provably do nothing a turn can observe. Anything else is refused.
fn ability_handled(ability: &str) -> bool {
    matches!(
        ability,
        // Implemented here or in the damage layer.
        "intimidate" | "defiant" | "competitive" | "contrary" | "clearbody" | "whitesmoke"
            | "fullmetalbody" | "unburden" | "regenerator" | "hospitality" | "magicguard"
            | "sturdy" | "roughskin" | "ironbarbs" | "levitate" | "prankster" | "galewings"
            | "triage" | "quickfeet" | "chlorophyll" | "swiftswim" | "sandrush" | "slushrush"
            | "surgesurfer" | "speedboost" | "stamina" | "weakarmor" | "justified" | "rattled"
            | "steamengine" | "watercompaction" | "toxicdebris" | "spicyspray" | "earlybird"
            | "armortail" | "queenlymajesty" | "dazzling" | "unnerve" | "friendguard"
            | "moldbreaker" | "teravolt" | "turboblaze" | "myceliummight" | "scrappy"
            | "mindseye" | "cloudnine" | "airlock" | "overcoat" | "soundproof" | "bulletproof"
            | "wonderguard" | "battlearmor" | "shellarmor" | "superluck" | "noguard"
            | "compoundeyes" | "hustle" | "adaptability" | "technician" | "toughclaws"
            | "sheerforce" | "sharpness" | "strongjaw" | "megalauncher" | "ironfist"
            | "punkrock" | "steelworker" | "dragonsmaw" | "transistor" | "rockypayload"
            | "waterbubble" | "rivalry" | "reckless" | "sandforce" | "fairyaura" | "darkaura"
            | "aurabreak" | "hugepower" | "purepower" | "guts" | "toxicboost" | "flareboost"
            | "solarpower" | "defeatist" | "overgrow" | "blaze" | "torrent" | "swarm"
            | "gorillatactics" | "flowergift" | "thickfat" | "heatproof" | "purifyingsalt"
            | "marvelscale" | "furcoat" | "grasspelt" | "tintedlens" | "neuroforce"
            | "solidrock" | "filter" | "prismarmor" | "multiscale" | "shadowshield"
            | "icescales" | "fluffy" | "immunity" | "limber" | "waterveil" | "insomnia"
            | "vitalspirit" | "thermalexchange" | "innerfocus" | "oblivious" | "owntempo"
            | "guarddog" | "hypercutter" | "bigpecks" | "keeneye" | "rockhead"
            // Weather setters, applied on switch-in and mega.
            | "drought" | "drizzle" | "sandstream" | "snowwarning"
            // Type immunities and absorbers, applied by `absorb`.
            | "flashfire" | "waterabsorb" | "dryskin" | "voltabsorb" | "lightningrod"
            | "motordrive" | "stormdrain" | "sapsipper" | "eartheater" | "wellbakedbody"
            | "windrider"
            // No effect a turn can observe.
            | "pressure" | "shadowtag" | "arenatrap" | "magnetpull" | "runaway" | "telepathy"
            | "healer" | "symbiosis" | "sweetveil" | "flowerveil" | "aromaveil" | "damp"
            | "lightmetal" | "heavymetal" | "sandveil" | "snowcloak" | "stall"
            // Weather setters this port applies on switch-in and mega.
            | "desolateland" | "primordialsea" | "deltastream"
            // Type changers and retypers: the damage layer owns them, and the resolver
            // never rewrites the Pokemon's types for them -- nor does Python.
            | "aerilate" | "pixilate" | "galvanize" | "refrigerate" | "normalize"
            | "liquidvoice" | "protean" | "libero"
            // Fractional priority and the weather HP abilities, both implemented here.
            | "quickdraw" | "icebody" | "raindish" | "comatose" | "mirrorarmor"
            // With secondaries enumerated this one branches a Disable, which is why it
            // took implementing the `disable` condition before it could be listed here.
            | "cursedbody"
            // Python reports these and changes nothing, so ignoring them agrees with it.
            | "static" | "flamebody" | "effectspore" | "poisonpoint" | "cutecharm"
            | "poisontouch" | "angerpoint" | "berserk" | "angershell" | "truant" | "dancer"
            // The forme change this drives is in `use_move`, which is why it is here and
            // no longer in `check_position_supported`'s pair.
            | "stancechange"
    ) || crate::inert::ability_is_inert(ability)
}

/// Items likewise. A held item this port does not know is refused rather than ignored.
fn item_handled(item: &str) -> bool {
    if crate::effects::type_boost_item(item).is_some()
        || crate::effects::resist_berry(item).is_some()
    {
        return true;
    }
    matches!(
        item,
        "lifeorb" | "expertbelt" | "muscleband" | "wiseglasses" | "normalgem" | "lightball"
            | "choicescarf" | "choiceband" | "choicespecs" | "ironball" | "widelens"
            | "zoomlens" | "brightpowder" | "scopelens" | "leek" | "focussash" | "sitrusberry"
            | "oranberry" | "leftovers" | "lumberry" | "rockyhelmet" | "airballoon"
            | "safetygoggles" | "utilityumbrella" | "shellbell" | "lightclay"
            | "terrainextender" | "damprock" | "heatrock" | "icyrock" | "smoothrock"
            | "assaultvest" | "clearamulet" | "covertcloak" | "loadeddice" | "protectivepads"
            | "ejectpack" | "boosterenergy" | "abilityshield" | "mirrorherb" | "punchingglove"
            // `check_white_herb` and its four call sites landed with the port and this
            // list was never told, so every position a holder could be involved in was
            // refused for an effect that was already here -- 63% of the cells generation
            // refused and 76% of a match's, measured 2026-09-20 (IKA-29).
            | "whiteherb"
    ) || crate::inert::item_is_inert(item)
        // Mega stones carry no turn effect of their own; the mega action owns the forme
        // change, and `reg.mega_targets` is what says which stone belongs to whom.
        || item.ends_with("ite")
        || item.ends_with("itex")
        || item.ends_with("itey")
}

/// Move fields this port does not implement. A move carrying one is refused.
const UNHANDLED_MOVE_FIELDS: [&str; 10] = [
    "damageCallback",
    "multiaccuracy",
    "selfdestruct",
    "struggleRecoil",
    "mindBlownRecoil",
    "smartTarget",
    "stealsBoosts",
    "sleepUsable",
    "onHitField",
    "willCrit",
];

/// The status moves this port resolves.
///
/// Narrower than Python's own `STATUS_MOVES_FULLY_MODELLED`, which answers a different
/// question -- which ones it does not *report* -- and is generated into `modelled.rs`
/// rather than retyped. One list was briefly doing both jobs.
pub(crate) fn status_move_handled(move_id: &str) -> bool {
    matches!(
        move_id,
        "protect" | "detect" | "banefulbunker" | "burningbulwark" | "spikyshield"
            | "kingsshield" | "obstruct" | "silktrap" | "maxguard" | "wideguard" | "quickguard"
            | "followme" | "ragepowder" | "spotlight" | "helpinghand" | "tailwind" | "trickroom"
            | "lightscreen" | "reflect" | "auroraveil" | "sunnyday" | "raindance" | "sandstorm"
            | "snowscape" | "electricterrain" | "grassyterrain" | "mistyterrain"
            | "psychicterrain" | "spikes" | "toxicspikes" | "stealthrock" | "stickyweb"
            | "swordsdance" | "nastyplot" | "calmmind" | "irondefense" | "amnesia" | "agility"
            | "bulkup" | "howl" | "growl" | "leer" | "tailwhip" | "screech" | "charm"
            | "faketears" | "metalsound" | "willowisp" | "thunderwave" | "toxic" | "taunt"
            | "leechseed" | "partingshot" | "lifedew" | "recover" | "softboiled" | "slackoff"
            | "milkdrink" | "confuseray" | "yawn" | "hypnosis" | "spore" | "sleeppowder"
            | "encore" | "shellsmash" | "roost" | "moonlight" | "synthesis" | "morningsun"
            | "perishsong" | "knockoff" | "thief" | "covet"
    )
}

fn check_move_supported(reg: &Reg, move_id: &str) -> Result<(), String> {
    // The recharge turn is a fake move with no dex entry: Showdown builds the request
    // entry by hand and intercepts the action before anything is executed, so there is
    // nothing to look up and nothing to check.
    if move_id == "recharge" {
        return Ok(());
    }
    let Some(mv) = reg.moves.get(move_id) else {
        return Err(format!("move not in the regulation: {move_id}"));
    };
    for field in UNHANDLED_MOVE_FIELDS {
        if mv.raw.get(field).map(|v| !v.is_null()).unwrap_or(false) {
            return Err(format!("move field {field}: {move_id}"));
        }
    }
    // A status move Python does not fully model gets the declarative fields and a report,
    // and nothing else -- which is exactly what this port does with one too. So refusing is
    // right only where Python has custom code it *does* model and this port has not
    // implemented it; refusing the rest was refusing to do the same nothing.
    if mv.category == "Status"
        && !status_move_handled(move_id)
        && crate::modelled::status_move_is_fully_modelled(move_id)
    {
        return Err(format!("status move: {move_id}"));
    }
    // Trick and Switcheroo swap items, which this port does not implement; Last Resort
    // reads which of its user's other moves have been used.
    if matches!(move_id, "lastresort" | "trick" | "switcheroo") {
        return Err(format!("item-swapping or history move: {move_id}"));
    }
    Ok(())
}

pub(crate) const TWO_TURN_MOVES: [(&str, &[&str]); 12] = [
    ("solarbeam", &["sunnyday", "desolateland"]),
    ("solarblade", &["sunnyday", "desolateland"]),
    ("electroshot", &["raindance", "primordialsea"]),
    ("fly", &[]),
    ("dig", &[]),
    ("dive", &[]),
    ("bounce", &[]),
    ("phantomforce", &[]),
    ("shadowforce", &[]),
    ("skyattack", &[]),
    ("meteorbeam", &[]),
    ("geomancy", &[]),
];

/// The Pokemon this turn can involve: the four on the field, plus anything a chosen
/// switch brings in. A benched Pokemon with an ability this port does not model cannot
/// affect the turn, and refusing for it was throwing away half the coverage.
fn involved<'a>(pos: &'a Position, side_actions: &[Vec<SlotAction>; 2]) -> Vec<&'a Pokemon> {
    let mut out: Vec<&Pokemon> = Vec::new();
    for (side_index, side) in pos.sides.iter().enumerate() {
        for slot in 0..side.active.len() {
            if let Some(mon) = side.active_pokemon(slot) {
                out.push(mon);
            }
        }
        for action in &side_actions[side_index] {
            if let SlotAction::Switch { party_index, species, .. } = action {
                let found = side
                    .pokemon
                    .iter()
                    .find(|mon| mon.species == *species || mon.base_species == *species)
                    .or_else(|| side.pokemon.get(party_index.saturating_sub(1)));
                if let Some(mon) = found {
                    out.push(mon);
                }
            }
        }
    }
    out
}

fn check_position_supported(
    pos: &Position,
    side_actions: &[Vec<SlotAction>; 2],
) -> Result<(), String> {
    {
        for mon in involved(pos, side_actions) {
            if !ability_handled(mon.ability.as_str()) {
                return Err(format!("ability: {}", mon.ability));
            }
            if let Some(item) = mon.item {
                if !item_handled(item.as_str()) {
                    return Err(format!("item: {}", item));
                }
            }
            if !mon.unmodelled_volatiles.is_empty() {
                return Err("position carries unmodelled volatiles".into());
            }
            for volatile in &mon.volatiles {
                if !volatile_handled(volatile.id.as_str()) {
                    return Err(format!("volatile: {}", volatile.id));
                }
            }
            if mon.transformed || mon.stats_override.is_some() {
                return Err("transformed Pokemon".into());
            }
            // A position the game could not reach is not a thing to hold two
            // implementations to: Python deals negative damage on one, and whatever this
            // port did instead would be a different wrong answer rather than a bug. These
            // are `position.validate_position`'s own checks, on the Pokemon a turn can
            // involve.
            if mon.hp < 0 || mon.hp > mon.maxhp {
                return Err(format!("hp {} outside 0..{}", mon.hp, mon.maxhp));
            }
            if mon.fainted != (mon.hp == 0) {
                return Err(format!("fainted={} disagrees with hp={}", mon.fainted, mon.hp));
            }
            // Slow Start changes the position in a way this port does not implement,
            // and is refused here rather than at the point of use so a turn can never get
            // half-way through it. Stance Change was the other one until `change_forme`.
            if mon.ability.as_str() == "slowstart" {
                return Err(format!("ability: {}", mon.ability));
            }
        }
    }
    Ok(())
}

pub(crate) fn volatile_is_handled(vid: &str) -> bool {
    volatile_handled(vid)
}

fn volatile_handled(vid: &str) -> bool {
    // Inverted on purpose, and now empty. A whitelist of volatiles refused whatever it had
    // not been told about -- Hyper Beam's `mustrecharge` cost 45% of the cells of a node,
    // for an effect this port implements -- so what belongs here is the set the Python
    // resolver acts on by name and this port does not. `disable` was the last of them and
    // is implemented; the flag on the move slot and its clearing in the residual were the
    // work.
    //
    // Everything else is either implemented here or carried generically: added, counted
    // down, and removed at zero, which is what Python does with a volatile it does not
    // name. The names Python *does* name are, in full: choicelock, confusion, disable,
    // encore, endure, flinch, followme, glaiverush, helpinghand, ingrain, leechseed,
    // magnetrise, mustrecharge, partiallytrapped, pendingselfswitch, perishsong,
    // ragepowder, saltcure, smackdown, spotlight, stall, taunt, telekinesis, throatchop,
    // torment, twoturnmove, unburden, yawn -- and all of them are handled.
    let _ = vid;
    true
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// The actions one side chose, as the case fixture writes them.
#[derive(Clone, Debug)]
pub enum SlotAction {
    Move { slot: usize, move_id: Id, target: Option<i64>, mega: bool },
    Switch { slot: usize, party_index: usize, species: Id },
    Pass { slot: usize },
}

/// One side's choice for a turn: one slot action per active slot.
pub fn parse_actions_list(value: &Value) -> Vec<SlotAction> {
    value
        .as_array()
        .map(|list| {
            list.iter()
                .map(|entry| {
                    let slot = entry["slot"].as_u64().unwrap_or(0) as usize;
                    match entry["kind"].as_str().unwrap_or("pass") {
                        "move" => SlotAction::Move {
                            slot,
                            move_id: Id::new(entry["moveId"].as_str().unwrap_or_default()),
                            target: entry["target"].as_i64(),
                            mega: entry["mega"].as_bool().unwrap_or(false),
                        },
                        "switch" => SlotAction::Switch {
                            slot,
                            party_index: entry["partyIndex"].as_u64().unwrap_or(1) as usize,
                            species: Id::new(entry["species"].as_str().unwrap_or_default()),
                        },
                        _ => SlotAction::Pass { slot },
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn parse_actions(case: &Value) -> [Vec<SlotAction>; 2] {
    [parse_actions_list(&case["ours"]), parse_actions_list(&case["theirs"])]
}

pub fn resolve_turn<'a>(
    reg: &'a Reg,
    pos: &Position,
    side_actions: &[Vec<SlotAction>; 2],
    budget: Budget,
) -> Result<TurnResult<'a>, String> {
    let started = phase_start();
    check_position_supported(pos, side_actions)?;
    phase_end(0, started);
    if pos.sides.len() != 2 || pos.sides.iter().any(|s| s.active.len() != 2) {
        return Err("this port handles two sides of two active slots".into());
    }

    let mut attacks = [[false; 2]; 2];
    for (side_index, actions) in side_actions.iter().enumerate() {
        for action in actions {
            match action {
                SlotAction::Move { slot, move_id, .. } => {
                    check_move_supported(reg, move_id.as_str())?;
                    attacks[side_index][*slot] = reg
                        .moves
                        .get(move_id.as_str())
                        .map(|m| m.category != "Status")
                        .unwrap_or(false);
                }
                SlotAction::Switch { slot, .. } | SlotAction::Pass { slot } => {
                    attacks[side_index][*slot] = false;
                }
            }
        }
    }

    let started = phase_start();
    let field = field_state(pos);
    let queues = build_queue(reg, pos, side_actions, &field)?;
    phase_end(1, started);

    let mut branches: Vec<Branch> = Vec::new();
    let mut suspended: Vec<Suspended<'a>> = Vec::new();
    let mut exact = true;
    let mut unmodelled: std::collections::BTreeSet<String> = Default::default();
    for queue in queues {
        if queue.is_empty() {
            continue;
        }
        let weight = queue[0].branch_probability;
        let (order, ties) = order_actions(&queue, pos.field.trick_room());
        for (permutation, tie_weight) in tie_permutations(&order, &ties, &budget) {
            let sequence: Vec<QueuedAction> =
                permutation.iter().map(|index| queue[*index].clone()).collect();
            let start = Live {
                weight: 1.0,
                turn: Turn::new(reg, pos.clone(), budget, attacks),
                remaining: sequence,
            };
            let started = phase_start();
            let sub = run_queue(reg, vec![start], budget)?;
            phase_end(2, started);
            exact = exact && sub.exact;
            unmodelled.extend(sub.unmodelled);
            for mut branch in sub.branches {
                branch.probability *= weight * tie_weight;
                branches.push(branch);
            }
            for mut pause in sub.suspended {
                pause.probability *= weight * tie_weight;
                suspended.push(pause);
            }
        }
    }
    // Across the Speed-tie orders and the queue's fractional-priority branches, which are
    // separate resolutions and so cannot have met inside `run_queue`.
    if budget.merge_duplicates {
        branches = merge_branches(branches);
        suspended = merge_suspended(suspended);
    }
    Ok(TurnResult { branches, exact, suspended, unmodelled })
}

fn tie_permutations(
    order: &[usize],
    ties: &[Vec<usize>],
    budget: &Budget,
) -> Vec<(Vec<usize>, f64)> {
    if ties.is_empty() || !budget.enumerate_speed_ties {
        return vec![(order.to_vec(), 1.0)];
    }
    let mut result = vec![(order.to_vec(), 1.0)];
    for group in ties {
        if group.len() != 2 {
            continue;
        }
        let (a, b) = (group[0], group[1]);
        let mut expanded = Vec::with_capacity(result.len() * 2);
        for (sequence, weight) in result {
            let mut swapped = sequence.clone();
            let i = swapped.iter().position(|v| *v == a).unwrap();
            let j = swapped.iter().position(|v| *v == b).unwrap();
            swapped.swap(i, j);
            expanded.push((sequence, weight / 2.0));
            expanded.push((swapped, weight / 2.0));
        }
        result = expanded;
    }
    result
}

fn build_queue(
    reg: &Reg,
    pos: &Position,
    side_actions: &[Vec<SlotAction>; 2],
    field: &FieldState,
) -> Result<Vec<Vec<QueuedAction>>, String> {
    let mut base: Vec<(QueuedAction, Vec<(f64, f64)>)> = Vec::new();

    for (side_index, actions) in side_actions.iter().enumerate() {
        let side = &pos.sides[side_index];
        for action in actions {
            let slot = match action {
                SlotAction::Move { slot, .. }
                | SlotAction::Switch { slot, .. }
                | SlotAction::Pass { slot } => *slot,
            };
            let Some(mon) = side.active_pokemon(slot) else { continue };
            if mon.fainted {
                continue;
            }
            let battler = Battler::from_pokemon(reg, mon)?;
            let speed = effective_speed(&battler, field, &side.side_conditions);

            match action {
                SlotAction::Switch { party_index, species, .. } => {
                    base.push((
                        QueuedAction {
                            side: side_index,
                            slot,
                            kind: ActionKind::Switch,
                            order: ORDER_SWITCH,
                            priority: 0,
                            fractional: 0.0,
                            speed,
                            move_id: None,
                            target: None,
                            switch_to: Some(party_index - 1),
                            switch_species: Some(*species),
                            branch_probability: 1.0,
                        },
                        vec![(0.0, 1.0)],
                    ));
                }
                SlotAction::Pass { .. } => continue,
                SlotAction::Move { move_id, target, mega, .. } => {
                    if *mega {
                        base.push((
                            QueuedAction {
                                side: side_index,
                                slot,
                                kind: ActionKind::Mega,
                                order: ORDER_MEGA,
                                priority: 0,
                                fractional: 0.0,
                                speed,
                                move_id: None,
                                target: None,
                                switch_to: None,
                                switch_species: None,
                                branch_probability: 1.0,
                            },
                            vec![(0.0, 1.0)],
                        ));
                    }
                    let mut chosen = *move_id;
                    if let Some(charging) = mon.volatile("twoturnmove") {
                        if let Some(stored) = charging.move_id {
                            // A Pokemon part-way through a charging move is locked into
                            // it, whatever was chosen -- so the move that will actually
                            // resolve is this one, and it is the one that has to be
                            // supported.
                            check_move_supported(reg, stored.as_str())?;
                            chosen = stored;
                        }
                    }
                    base.push((
                        QueuedAction {
                            side: side_index,
                            slot,
                            kind: ActionKind::Move,
                            order: ORDER_MOVE,
                            priority: move_priority(reg, chosen.as_str(), &battler, field),
                            fractional: 0.0,
                            speed,
                            move_id: Some(chosen),
                            target: *target,
                            switch_to: None,
                            switch_species: None,
                            branch_probability: 1.0,
                        },
                        fractional_priority(reg, chosen.as_str(), &battler),
                    ));
                }
            }
        }
    }

    let mut branches: Vec<Vec<QueuedAction>> = vec![Vec::new()];
    let mut weights: Vec<f64> = vec![1.0];
    for (action, options) in base {
        if options.len() == 1 {
            let value = options[0].0;
            for branch in branches.iter_mut() {
                let mut copy = action.clone();
                copy.fractional = value;
                branch.push(copy);
            }
            continue;
        }
        let mut fresh_branches = Vec::new();
        let mut fresh_weights = Vec::new();
        for (branch, weight) in branches.iter().zip(weights.iter()) {
            for (value, probability) in &options {
                let mut extended = branch.clone();
                let mut copy = action.clone();
                copy.fractional = *value;
                extended.push(copy);
                fresh_branches.push(extended);
                fresh_weights.push(weight * probability);
            }
        }
        branches = fresh_branches;
        weights = fresh_weights;
    }
    for (branch, weight) in branches.iter_mut().zip(weights.iter()) {
        if let Some(first) = branch.first_mut() {
            first.branch_probability = *weight;
        }
    }
    Ok(branches)
}

// ---------------------------------------------------------------------------
// The queue loop
// ---------------------------------------------------------------------------

struct Live<'a> {
    weight: f64,
    turn: Turn<'a>,
    remaining: Vec<QueuedAction>,
}

// ---------------------------------------------------------------------------
// Lossless branch merge
// ---------------------------------------------------------------------------
//
// Transcribed from `_merge_live` / `_merge_branches` / `_merge_suspended` in Python, and
// it has to stay transcribed: the differential is an equality test on the branch list, so
// a merge that folded a different pair, or kept a different representative, or produced
// the same branches in a different order, would show up there as a port bug.

/// The cheap half of "is this the same branch": a hash over the fields that move.
///
/// Necessary, not sufficient. Equal states always hash the same; two that collide are
/// still compared in full by `same_turn`, which is what decides.
fn state_fingerprint(pos: &Position, remaining: &[QueuedAction], self_switch_pending: bool) -> u64 {
    // FNV-1a rather than the default hasher, and the same trade `encoded_node::leaf_key`
    // makes: this is a bucket key, not a checksum. SipHash over the thirty-odd values
    // below was worth measuring against the clone it is here to save.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    eat(pos.turn as u64);
    eat(pos.ended as u64);
    eat(self_switch_pending as u64);
    eat(pos.field.weather.map(|w| w.as_str().len() as u64 + 3).unwrap_or(0));
    eat(pos.field.terrain.map(|t| t.as_str().len() as u64 + 5).unwrap_or(0));
    // The four on the field only. Any function of the state is a valid fingerprint --
    // equal states hash equal whatever is left out -- so this reads the part a turn
    // changes and pays for four Pokemon rather than twelve.
    for side in &pos.sides {
        for slot in 0..side.active.len() {
            eat(side.active[slot].map(|index| index as u64 + 1).unwrap_or(0));
            if let Some(mon) = side.active_pokemon(slot) {
                eat(mon.hp as u64);
                eat(mon.fainted as u64);
                eat(mon.status.map(|s| s.as_str().len() as u64 + 7).unwrap_or(0));
                eat(mon.volatiles.len() as u64);
            }
        }
    }
    eat(remaining.len() as u64);
    for action in remaining {
        eat(action.side as u64);
        eat(action.slot as u64);
        eat(action.kind as u64);
        eat(action.move_id.map(|m| m.as_str().len() as u64 + 11).unwrap_or(0));
        eat(action.target.unwrap_or(-1) as u64);
        eat(action.switch_to.map(|index| index as u64 + 1).unwrap_or(0));
    }
    hash
}

/// Whether two parties are the same party, taking the sharing into account.
///
/// A side holds its six behind `Rc`, and a turn writes to the two that are out: two
/// branches of one turn therefore share the allocation of everything they did not touch.
/// `Rc::ptr_eq` settles those for the price of a pointer compare, and the derived `==` --
/// 928 bytes with move slots, boosts and volatiles inside -- runs only on the two that
/// moved. Measured at 45.1 us a turn against 41.4 without the merge at all before this,
/// and the gap is what it closes.
fn same_party(a: &[Rc<Pokemon>], b: &[Rc<Pokemon>]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| same_mon(x, y))
}

/// Whether two Pokemon are the same Pokemon.
///
/// The derived `==` compares in declaration order, which puts the four move slots ahead of
/// HP and the volatiles behind the boosts -- so two branches that differ only in the
/// damage dealt, or only in whether a secondary landed, are the expensive comparisons
/// rather than the cheap ones. The conditions in front are the ones branches actually
/// differ in; each is part of `==` as well, so this rejects sooner and never accepts more.
fn same_mon(x: &Rc<Pokemon>, y: &Rc<Pokemon>) -> bool {
    if Rc::ptr_eq(x, y) {
        return true;
    }
    x.hp == y.hp
        && x.fainted == y.fainted
        && x.status == y.status
        && x.item == y.item
        && x.boosts == y.boosts
        && x.volatiles.len() == y.volatiles.len()
        && x == y
}

/// Whether two sides are equal.
///
/// Destructured rather than written field by field from memory: a field added to `Side`
/// and not compared here would let two branches that differ in it merge, and the `let`
/// with no `..` makes that a compile error rather than a silent one.
fn same_side(a: &crate::position::Side, b: &crate::position::Side) -> bool {
    let crate::position::Side {
        id,
        name,
        active,
        pokemon,
        side_conditions,
        slot_conditions,
        mega_used,
        mega_capable_slots,
    } = a;
    *active == b.active
        && *mega_used == b.mega_used
        && *side_conditions == b.side_conditions
        && *slot_conditions == b.slot_conditions
        && *mega_capable_slots == b.mega_capable_slots
        && *id == b.id
        && *name == b.name
        && same_party(pokemon, &b.pokemon)
}

fn same_position(a: &Position, b: &Position) -> bool {
    let Position { format, sides, turn, field, request_state, ended, winner } = a;
    *turn == b.turn
        && *ended == b.ended
        && *winner == b.winner
        && *request_state == b.request_state
        && *format == b.format
        && *field == b.field
        && same_side(&sides[0], &b.sides[0])
        && same_side(&sides[1], &b.sides[1])
}

/// Whether two working states are the same state, by everything that can differ.
///
/// The field list mirrors `_MERGE_COMPARED_STATE`; Python's `events` and `acts` have no
/// counterpart here, and `unmodelled` is unioned rather than compared because it reports
/// on the resolver's coverage rather than on the state. Destructured for the same reason
/// as `same_side`: a new field in `Turn` has to be dealt with here or it will not build.
fn same_turn(a: &Turn, b: &Turn) -> bool {
    let Turn {
        reg,
        pos,
        budget,
        attacks,
        hurt_this_turn,
        move_failed,
        move_damage_total,
        move_connected,
        acted,
        actions_remaining,
        self_switch_pending,
        pending_secondaries,
        current_actor,
        wipe_order,
        unmodelled: _,
    } = a;
    std::ptr::eq(*reg, b.reg)
        && *self_switch_pending == b.self_switch_pending
        && *actions_remaining == b.actions_remaining
        && *move_damage_total == b.move_damage_total
        && *move_connected == b.move_connected
        && *current_actor == b.current_actor
        && *wipe_order == b.wipe_order
        && *acted == b.acted
        && *hurt_this_turn == b.hurt_this_turn
        && *move_failed == b.move_failed
        && *attacks == b.attacks
        && *budget == b.budget
        && *pending_secondaries == b.pending_secondaries
        && same_position(pos, &b.pos)
}

/// Folds equal items into their first occurrence, in one pass, order preserved.
fn fold_equal<T>(
    items: Vec<T>,
    fingerprint: impl Fn(&T) -> u64,
    same: impl Fn(&T, &T) -> bool,
    absorb: impl Fn(&mut T, T),
) -> (Vec<T>, usize) {
    let mut kept: Vec<T> = Vec::with_capacity(items.len());
    let mut buckets: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();
    let mut folded = 0usize;
    for item in items {
        let key = fingerprint(&item);
        let group = buckets.entry(key).or_default();
        let mut hit: Option<usize> = None;
        for index in group.iter() {
            if same(&kept[*index], &item) {
                hit = Some(*index);
                break;
            }
        }
        match hit {
            Some(index) => {
                absorb(&mut kept[index], item);
                folded += 1;
            }
            None => {
                group.push(kept.len());
                kept.push(item);
            }
        }
    }
    (kept, folded)
}

fn merge_live<'a>(items: Vec<Live<'a>>) -> Vec<Live<'a>> {
    if items.len() < 2 {
        return items;
    }
    fold_equal(
        items,
        |item| {
            state_fingerprint(&item.turn.pos, &item.remaining, item.turn.self_switch_pending)
        },
        |x, y| x.remaining == y.remaining && same_turn(&x.turn, &y.turn),
        |into, other| {
            into.weight += other.weight;
            into.turn.unmodelled.extend(other.turn.unmodelled);
        },
    )
    .0
}

/// Finished outcomes that are the same position.
///
/// Needed as well as `merge_live` because Speed ties are resolved above the queue: each
/// order is its own `run_queue`, so two orders that end the same way never meet until the
/// results are collected.
fn merge_branches(items: Vec<Branch>) -> Vec<Branch> {
    if items.len() < 2 {
        return items;
    }
    fold_equal(
        items,
        |branch| state_fingerprint(&branch.position, &[], false),
        |x, y| same_position(&x.position, &y.position),
        |into, other| into.probability += other.probability,
    )
    .0
}

fn merge_suspended<'a>(items: Vec<Suspended<'a>>) -> Vec<Suspended<'a>> {
    if items.len() < 2 {
        return items;
    }
    fold_equal(
        items,
        |pause| state_fingerprint(&pause.turn.pos, &pause.remaining, true),
        |x, y| x.remaining == y.remaining && same_turn(&x.turn, &y.turn),
        |into, other| into.probability += other.probability,
    )
    .0
}

fn run_queue<'a>(
    reg: &'a Reg,
    start: Vec<Live<'a>>,
    budget: Budget,
) -> Result<TurnResult<'a>, String> {
    let mut live = start;
    let mut finished: Vec<Live<'a>> = Vec::new();
    let mut paused: Vec<Live<'a>> = Vec::new();
    let mut exact = true;
    let mut dropped = 0usize;

    let prune = |items: &mut Vec<Live<'a>>, dropped: &mut usize| {
        if items.len() <= budget.max_branches {
            return;
        }
        items.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        *dropped += items.len() - budget.max_branches;
        items.truncate(budget.max_branches);
    };

    while !live.is_empty() {
        let pending = live.iter().filter(|item| !item.remaining.is_empty()).count();
        let room = (budget.max_branches / pending.max(1)).max(1);
        let step_budget = budget.narrowed(room);
        if step_budget != budget {
            exact = false;
        }

        let mut next: Vec<Live<'a>> = Vec::new();
        let (finished_before, paused_before) = (finished.len(), paused.len());
        for mut item in live.into_iter() {
            if item.remaining.is_empty() {
                finished.push(item);
                continue;
            }
            let started = phase_start();
            let ordered = resort(reg, &item.turn, &item.remaining)?;
            phase_end(5, started);
            let action = ordered[0].clone();
            let rest: Vec<QueuedAction> = ordered[1..].to_vec();
            item.turn.actions_remaining = rest.len();
            item.turn.budget = step_budget;
            // An Encore that landed earlier this turn rewrites the action, and Showdown
            // re-picks its target at random -- so this is a list, not a single action.
            let started = phase_start();
            let variants = encore_override(reg, &item.turn, &action);
            phase_end(4, started);
            let overridden = variants
                .first()
                .map(|(_, variant)| variant.move_id != action.move_id)
                .unwrap_or(false);
            let mut outcomes: Vec<(f64, Turn<'a>)> = Vec::new();
            // One variant is the ordinary case -- Encore is what makes it more than one --
            // and a turn is an eleven-kilobyte position, so the ordinary case hands the
            // state over rather than copying it. This one clone was 28% of all of them.
            if variants.len() == 1 {
                let (variant_weight, variant) = &variants[0];
                let mut base = item.turn;
                if overridden {
                    base.report(
                        "encore override changed the move Sucker Punch was read against",
                    );
                }
                let started = phase_start();
                let produced = execute(reg, base, variant, step_budget)?;
                phase_end(6, started);
                for (weight, turn) in produced {
                    outcomes.push((variant_weight * weight, turn));
                }
            } else {
                for (variant_weight, variant) in &variants {
                    let mut base = item.turn.clone();
                    if overridden {
                        // Sucker Punch was read against the move the side chose, and Encore
                        // has just replaced it; Python says so and so does this.
                        base.report(
                            "encore override changed the move Sucker Punch was read against",
                        );
                    }
                    for (weight, turn) in execute(reg, base, variant, step_budget)? {
                        outcomes.push((variant_weight * weight, turn));
                    }
                }
            }
            for (weight, turn) in outcomes {
                let wiped = turn.pos.sides.iter().any(|s| s.pokemon.iter().all(|m| m.fainted));
                let remaining_actions = if wiped { Vec::new() } else { rest.clone() };
                let child =
                    Live { weight: item.weight * weight, turn, remaining: remaining_actions };
                // A self-switching move ends `runAction` with `switchFlag` set, and
                // Showdown answers that with a fresh switch request -- so the turn stops
                // here, before the rest of the queue and before the residual phase.
                if !wiped && child.turn.self_switch_pending {
                    paused.push(child);
                } else {
                    next.push(child);
                }
            }
            if next.len() > 2 * budget.max_branches {
                if budget.merge_duplicates {
                    next = merge_live(next);
                }
                prune(&mut next, &mut dropped);
            }
        }
        // Merging comes before the cap, always: two halves of one state that were each
        // about to be dropped are worth keeping once they are one branch carrying both
        // weights, and a shorter live list gives the next generation more resolution.
        if budget.merge_duplicates {
            next = merge_live(next);
            // Only the lists this generation added to are worth walking again.
            if finished.len() != finished_before {
                finished = merge_live(finished);
            }
            if paused.len() != paused_before {
                paused = merge_live(paused);
            }
        }
        prune(&mut next, &mut dropped);
        prune(&mut finished, &mut dropped);
        prune(&mut paused, &mut dropped);
        if dropped > 0 {
            exact = false;
        }
        live = next;
    }

    // Renormalise once, at the end: dropping low-probability branches leaves the rest
    // summing to less than one, and rescaling after every generation would compound the
    // rounding.
    let total: f64 = finished.iter().map(|item| item.weight).sum::<f64>()
        + paused.iter().map(|item| item.weight).sum::<f64>();
    if total > 0.0 && (total - 1.0).abs() > 1e-12 {
        for item in finished.iter_mut().chain(paused.iter_mut()) {
            item.weight /= total;
        }
    }

    let mut branches = Vec::with_capacity(finished.len());
    let mut unmodelled: std::collections::BTreeSet<String> = Default::default();
    for mut item in finished {
        let started = phase_start();
        residuals(reg, &mut item.turn)?;
        phase_end(3, started);
        item.turn.pos.turn += 1;
        unmodelled.extend(item.turn.unmodelled.iter().cloned());
        branches.push(Branch { probability: item.weight, position: item.turn.pos });
    }
    // A paused branch gets no residuals and no turn increment: the residual phase is
    // behind the interrupt, so it belongs to whatever the resume produces.
    let mut suspended = Vec::with_capacity(paused.len());
    for item in paused {
        unmodelled.extend(item.turn.unmodelled.iter().cloned());
        suspended.push(Suspended {
            probability: item.weight,
            turn: item.turn,
            remaining: item.remaining,
        });
    }
    Ok(TurnResult { branches, exact, suspended, unmodelled })
}

/// The action Encore forces its user to take, with its target's odds.
///
/// Showdown keeps the original move's priority -- the queue was already ordered on it and
/// only the move changes -- and re-picks the target with `getRandomTarget`, which is
/// uniform over adjacent foes. A single-target encored move against two standing foes is
/// therefore a genuine coin flip, and returning one of them would teach the search a
/// preference the game does not have.
fn encore_override(
    reg: &Reg,
    turn: &Turn,
    action: &QueuedAction,
) -> Vec<(f64, QueuedAction)> {
    let unchanged = vec![(1.0, action.clone())];
    if action.kind != ActionKind::Move {
        return unchanged;
    }
    let Some(mon) = turn.mon_at(action.side, action.slot) else { return unchanged };
    if mon.fainted {
        return unchanged;
    }
    let Some(encore) = mon.volatile("encore") else { return unchanged };
    let Some(forced) = encore.move_id else { return unchanged };
    if Some(forced) == action.move_id {
        return unchanged;
    }
    let Some(replacement) = reg.moves.get(forced.as_str()) else { return unchanged };

    let targets: Vec<Option<i64>> = match replacement.target.as_str() {
        "normal" | "any" | "adjacentFoe" => {
            let foe_side = 1 - action.side;
            let live: Vec<Option<i64>> = (0..turn.pos.sides[foe_side].active.len())
                .filter(|slot| {
                    matches!(turn.mon_at(foe_side, *slot), Some(m) if !m.fainted)
                })
                .map(|slot| Some(slot as i64 + 1))
                .collect();
            if live.is_empty() {
                return unchanged;
            }
            live
        }
        "self" | "all" | "allAdjacent" | "allAdjacentFoes" | "allies" | "allySide"
        | "allyTeam" | "foeSide" | "randomNormal" | "scripted" | "adjacentAllyOrSelf" => {
            vec![None]
        }
        _ => vec![action.target],
    };

    let weight = 1.0 / targets.len() as f64;
    targets
        .into_iter()
        .map(|target| {
            let mut copy = action.clone();
            copy.move_id = Some(forced);
            copy.target = target;
            (weight, copy)
        })
        .collect()
}

/// Wall clock inside the phases of a turn, in nanoseconds. Coarse on purpose: a timer
/// costs about 25 ns here, so it goes around things called a handful of times a turn and
/// never around anything in the innermost loop, where it would measure itself.
pub static PHASE_NS: [std::sync::atomic::AtomicU64; 13] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 13];
pub const PHASE_NAMES: [&str; 13] = [
    "supported",
    "build_queue",
    "run_queue",
    "residuals",
    "encore_override",
    "resort",
    "execute",
    "  can_act",
    "  use_move",
    "  hit_target",
    "    calculate",
    "    after_hit",
    "    clone+ctx",
];

/// Zeroes every counter. The differential pass resolves each fixture once before the
/// speed loop does, and counting both made a per-turn figure a third too large.
pub fn reset_counters() {
    let relaxed = std::sync::atomic::Ordering::Relaxed;
    for slot in PHASE_NS.iter() {
        slot.store(0, relaxed);
    }
    for slot in PHASE_ALLOCS.iter() {
        slot.store(0, relaxed);
    }
    RESORTS.store(0, relaxed);
    RESIDUALS.store(0, relaxed);
    crate::position::CLONES.store(0, relaxed);
    crate::position::UNSHARED.store(0, relaxed);
    crate::damage::CALLS.store(0, relaxed);
    crate::battler::BUILDS.store(0, relaxed);
}

/// Allocations made inside each phase, when the counting allocator is built in. The
/// phases nest, so an inner one's allocations are counted by the outer one too.
pub static PHASE_ALLOCS: [std::sync::atomic::AtomicU64; 13] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 13];

#[cfg(feature = "count-allocations")]
fn allocations_now() -> usize {
    crate::counting_alloc::ALLOCATIONS.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(not(feature = "count-allocations"))]
fn allocations_now() -> usize {
    0
}

/// What a phase measurement carries, which with `profile` off is nothing at all.
///
/// A clock read costs about 25 ns here and the innermost of these is entered ten times a
/// turn, so the timers are behind a feature rather than shipped: a build that is not being
/// profiled should not be paying to be profiled.
#[cfg(feature = "profile")]
pub(crate) type PhaseStart = (std::time::Instant, usize);
#[cfg(not(feature = "profile"))]
pub(crate) type PhaseStart = ();

#[cfg(feature = "profile")]
#[inline]
pub(crate) fn phase_start() -> PhaseStart {
    (std::time::Instant::now(), allocations_now())
}

#[cfg(not(feature = "profile"))]
#[inline]
pub(crate) fn phase_start() -> PhaseStart {}

#[cfg(feature = "profile")]
#[inline]
pub(crate) fn phase_end(index: usize, start: PhaseStart) {
    let relaxed = std::sync::atomic::Ordering::Relaxed;
    PHASE_NS[index].fetch_add(start.0.elapsed().as_nanos() as u64, relaxed);
    PHASE_ALLOCS[index].fetch_add((allocations_now() - start.1) as u64, relaxed);
}

#[cfg(not(feature = "profile"))]
#[inline]
pub(crate) fn phase_end(_index: usize, _start: PhaseStart) {}

/// How often the queue is re-sorted, and how often a residual phase runs. Both rebuild
/// Battlers, which is the expensive part of each, so the counts are what say whether the
/// rebuilding is worth avoiding.
pub static RESORTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
pub static RESIDUALS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn resort(
    reg: &Reg,
    turn: &Turn,
    remaining: &[QueuedAction],
) -> Result<Vec<QueuedAction>, String> {
    if remaining.len() < 2 {
        return Ok(remaining.to_vec());
    }
    RESORTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let field = turn.field();
    let mut refreshed: Vec<QueuedAction> = Vec::with_capacity(remaining.len());
    for action in remaining {
        let mut copy = action.clone();
        if let Some(battler) = turn.battler_at(action.side, action.slot)? {
            let conditions = &turn.pos.sides[action.side].side_conditions;
            copy.speed = effective_speed(&battler, &field, conditions);
        }
        refreshed.push(copy);
    }
    let (order, _ties) = order_actions(&refreshed, turn.pos.field.trick_room());
    Ok(order.iter().map(|index| remaining[*index].clone()).collect())
}

pub(crate) type Outcome<'a> = (f64, Turn<'a>);

fn execute<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    budget: Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    match action.kind {
        ActionKind::Switch => {
            do_switch(reg, &mut turn, action)?;
            Ok(vec![(1.0, turn)])
        }
        ActionKind::Mega => {
            do_mega(reg, &mut turn, action)?;
            Ok(vec![(1.0, turn)])
        }
        ActionKind::Move => {
            let mut outcomes = do_move(reg, turn, action, budget)?;
            for (_weight, state) in outcomes.iter_mut() {
                state.acted[action.side][action.slot] = true;
            }
            Ok(outcomes)
        }
    }
}

fn do_switch(reg: &Reg, turn: &mut Turn, action: &QueuedAction) -> Result<(), String> {
    do_switch_with(reg, turn, action, true)
}

/// `run_switch_in` is false for a mid-turn replacement, where every incoming Pokemon is
/// placed before any of them sees a hazard or an Intimidate.
fn do_switch_with(
    reg: &Reg,
    turn: &mut Turn,
    action: &QueuedAction,
    run_switch_in: bool,
) -> Result<(), String> {
    let Some(_) = action.switch_to else { return Ok(()) };
    let incoming_index = {
        let side = &turn.pos.sides[action.side];
        let by_species = action.switch_species.and_then(|species| {
            side.pokemon
                .iter()
                .position(|mon| mon.species == species || mon.base_species == species)
        });
        match by_species.or(action.switch_to.filter(|i| *i < side.pokemon.len())) {
            None => return Ok(()),
            Some(index) => index,
        }
    };
    {
        let incoming = &turn.pos.sides[action.side].pokemon[incoming_index];
        if incoming.fainted || incoming.is_active() {
            return Ok(());
        }
    }

    let leaving_index = turn.pos.sides[action.side].active[action.slot];
    if let Some(leaving_index) = leaving_index {
        let (regenerates, fainted) = {
            let leaving = &turn.pos.sides[action.side].pokemon[leaving_index];
            (!leaving.fainted && leaving.ability == "regenerator", leaving.fainted)
        };
        if regenerates {
            let amount = {
                let leaving = &turn.pos.sides[action.side].pokemon[leaving_index];
                (leaving.maxhp / 3).max(1)
            };
            turn.heal(action.side, action.slot, amount);
        }
        {
            let leaving =
                Rc::make_mut(&mut turn.pos.sides[action.side].pokemon[leaving_index]);
            leaving.active_index = None;
            leaving.boosts = [0; 7];
            leaving.volatiles.clear();
            leaving.last_move = None;
            leaving.locked_move = None;
            for move_slot in leaving.moves.iter_mut() {
                move_slot.disabled = false;
            }
            leaving.newly_switched = false;
            if fainted {
                leaving.status = None;
                leaving.status_counter = None;
            }
        }
        // Swap party positions, as `BattleActions#switchIn` does.
        let vacated = turn.pos.sides[action.side].pokemon[incoming_index].slot;
        turn.pos.sides[action.side].pokemon.swap(incoming_index, leaving_index);
        Rc::make_mut(&mut turn.pos.sides[action.side].pokemon[action.slot]).slot = action.slot;
        Rc::make_mut(&mut turn.pos.sides[action.side].pokemon[vacated]).slot = vacated;
    }
    turn.pos.sides[action.side].active[action.slot] = Some(action.slot);

    {
        let incoming = Rc::make_mut(&mut turn.pos.sides[action.side].pokemon[action.slot]);
        incoming.active_index = Some(action.slot);
        incoming.newly_switched = true;
        incoming.active_move_actions = 0;
        for move_slot in incoming.moves.iter_mut() {
            move_slot.used = false;
        }
        if matches!(incoming.status, Some(s) if s.as_str() == "tox") {
            incoming.status_counter = Some(0);
        }
    }
    if run_switch_in {
        on_switch_in(reg, turn, action.side, action.slot)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Mid-turn replacements
// ---------------------------------------------------------------------------

/// Per side, per active slot, whether a self-switching move is waiting on a choice.
pub fn self_switches_needed(pos: &Position) -> [[bool; 2]; 2] {
    let mut out = [[false; 2]; 2];
    for (side_index, side) in pos.sides.iter().enumerate() {
        let bench = side
            .pokemon
            .iter()
            .filter(|mon| !mon.fainted && !mon.is_active())
            .count();
        for slot in 0..side.active.len().min(2) {
            out[side_index][slot] = bench > 0
                && matches!(side.active_pokemon(slot), Some(mon)
                    if !mon.fainted && mon.has_volatile("pendingselfswitch"));
        }
    }
    out
}

/// The replacements the interrupted side could send in, as one action list per option.
///
/// `switch_actions_after_faint`: fill as many slots as there are Pokemon to fill them
/// with, and which slot the last one takes is the player's choice.
fn replacement_options(
    pos: &Position,
    side_index: usize,
    owed: [bool; 2],
) -> Vec<Vec<SlotAction>> {
    let side = &pos.sides[side_index];
    let bench: Vec<&Pokemon> = side
        .pokemon
        .iter()
        .map(|mon| &**mon)
        .filter(|mon| !mon.fainted && !mon.is_active())
        .collect();
    let owed_count = owed.iter().filter(|needed| **needed).count();
    let fillable = bench.len().min(owed_count);

    let mut per_slot: Vec<Vec<SlotAction>> = Vec::new();
    for (slot, needed) in owed.iter().enumerate().take(side.active.len()) {
        if !needed {
            per_slot.push(vec![SlotAction::Pass { slot }]);
            continue;
        }
        let mut options: Vec<SlotAction> = bench
            .iter()
            .map(|mon| SlotAction::Switch {
                slot,
                party_index: mon.slot + 1,
                species: mon.species,
            })
            .collect();
        options.push(SlotAction::Pass { slot });
        per_slot.push(options);
    }

    let mut out: Vec<Vec<SlotAction>> = Vec::new();
    let mut combination: Vec<SlotAction> = Vec::new();
    build_combinations(&per_slot, 0, &mut combination, fillable, &mut out);
    out
}

fn build_combinations(
    per_slot: &[Vec<SlotAction>],
    index: usize,
    current: &mut Vec<SlotAction>,
    fillable: usize,
    out: &mut Vec<Vec<SlotAction>>,
) {
    if index == per_slot.len() {
        let switches: Vec<usize> = current
            .iter()
            .filter_map(|a| match a {
                SlotAction::Switch { party_index, .. } => Some(*party_index),
                _ => None,
            })
            .collect();
        let mut unique = switches.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() == switches.len() && switches.len() == fillable {
            out.push(current.clone());
        }
        return;
    }
    for option in &per_slot[index] {
        current.push(option.clone());
        build_combinations(per_slot, index + 1, current, fillable, out);
        current.pop();
    }
}

/// Finishes a turn that stopped at a mid-turn replacement request.
pub fn resume_turn<'a>(
    reg: &'a Reg,
    paused: &Suspended<'a>,
    choices: &[Vec<SlotAction>; 2],
) -> Result<TurnResult<'a>, String> {
    let mut turn = paused.turn.clone();
    let owed = self_switches_needed(&turn.pos);
    let mut placed: Vec<(i64, usize, usize)> = Vec::new();

    for (side_index, actions) in choices.iter().enumerate() {
        for action in actions {
            let SlotAction::Switch { slot, party_index, species } = action else { continue };
            if !owed[side_index][*slot] {
                continue;
            }
            if let Some(mon) = turn.mon_at_mut(side_index, *slot) {
                mon.volatiles.retain(|v| v.id.as_str() != "pendingselfswitch");
            }
            let queued = QueuedAction {
                side: side_index,
                slot: *slot,
                kind: ActionKind::Switch,
                order: ORDER_SWITCH,
                priority: 0,
                fractional: 0.0,
                speed: 0,
                move_id: None,
                target: None,
                switch_to: Some(party_index - 1),
                switch_species: Some(*species),
                branch_probability: 1.0,
            };
            do_switch_with(reg, &mut turn, &queued, false)?;
            let speed = match turn.battler_at(side_index, *slot)? {
                None => 0,
                Some(incoming) => {
                    let field = turn.field();
                    let conditions = &turn.pos.sides[side_index].side_conditions;
                    effective_speed(&incoming, &field, conditions)
                }
            };
            placed.push((speed, side_index, *slot));
        }
    }

    // `runSwitch` is order 101 sorted on speed, fastest first, so a fast replacement takes
    // the hazards and fires its ability before a slow one.
    placed.sort_by_key(|(speed, side, slot)| (-speed, *side, *slot));
    for (_speed, side_index, slot) in placed {
        on_switch_in(reg, &mut turn, side_index, slot)?;
    }

    turn.self_switch_pending = self_switches_needed(&turn.pos)
        .iter()
        .any(|side| side.iter().any(|flag| *flag));
    let budget = turn.budget;
    let remaining = paused.remaining.clone();
    let mut result = run_queue(reg, vec![Live { weight: 1.0, turn, remaining }], budget)?;
    for branch in result.branches.iter_mut() {
        branch.probability *= paused.probability;
    }
    for pause in result.suspended.iter_mut() {
        pause.probability *= paused.probability;
    }
    Ok(result)
}

/// Every replacement the interrupted side could send in, and the turn each produces.
pub fn resume_alternatives<'a>(
    reg: &'a Reg,
    paused: &Suspended<'a>,
) -> Result<(Option<usize>, Vec<TurnResult<'a>>), String> {
    let owed = self_switches_needed(&paused.turn.pos);
    let sides: Vec<usize> = (0..2).filter(|i| owed[*i].iter().any(|f| *f)).collect();
    let Some(&chooser) = sides.first() else { return Ok((None, Vec::new())) };
    let other = 1 - chooser;
    let passes: Vec<SlotAction> = (0..paused.turn.pos.sides[other].active.len())
        .map(|slot| SlotAction::Pass { slot })
        .collect();

    let mut out = Vec::new();
    for option in replacement_options(&paused.turn.pos, chooser, owed[chooser]) {
        let choices = if chooser == 0 {
            [option, passes.clone()]
        } else {
            [passes.clone(), option]
        };
        let mut resumed = resume_turn(reg, paused, &choices)?;
        if sides.len() > 1 {
            resumed.unmodelled.insert("simultaneous mid-turn replacements".into());
        }
        out.push(resumed);
    }
    Ok((Some(chooser), out))
}

/// The turn's value under one objective, folding through the replacement choice.
///
/// Python builds a tree of leaf positions and folds values over it, because its leaves are
/// scored elsewhere. Here the objective is in hand, so the fold is the recursion itself --
/// the same arithmetic: chance is a weighted mean normalised by its own weight, and a
/// replacement is the option its chooser likes most.
pub fn turn_value(
    reg: &Reg,
    result: &TurnResult,
    score: fn(&Position) -> f64,
    depth: usize,
    notes: &mut std::collections::BTreeSet<String>,
) -> Result<f64, String> {
    notes.extend(result.unmodelled.iter().cloned());
    let total: f64 = result.branches.iter().map(|b| b.probability).sum::<f64>()
        + result.suspended.iter().map(|s| s.probability).sum::<f64>();
    if total <= 0.0 {
        return Ok(0.0);
    }
    // Python has two code paths here and they round differently, so this has two as
    // well. An ordinary cell is `values @ (probabilities / total)` -- divided first, then
    // summed -- and a cell that folds through a replacement is
    // `sum(weight * value) / total`. Matching the arithmetic is what makes the node
    // differential an equality test rather than a tolerance.
    if result.suspended.is_empty() {
        let mut dotted = 0.0;
        for branch in &result.branches {
            dotted += (branch.probability / total) * score(&branch.position);
        }
        return Ok(dotted);
    }
    let mut accumulated = 0.0;
    for branch in &result.branches {
        accumulated += branch.probability * score(&branch.position);
    }
    for pause in &result.suspended {
        // Four Pokemon a side means a turn cannot interrupt itself indefinitely; the guard
        // is against a bug becoming unbounded recursion, and it reports where it stopped.
        if depth >= 4 {
            notes.insert("more than four mid-turn replacements in one turn".into());
            accumulated += pause.probability * score(&pause.turn.pos);
            continue;
        }
        let (chooser, alternatives) = resume_alternatives(reg, pause)?;
        let Some(chooser) = chooser else {
            notes.insert("a suspended turn offered no replacement".into());
            accumulated += pause.probability * score(&pause.turn.pos);
            continue;
        };
        if alternatives.is_empty() {
            notes.insert("a suspended turn offered no replacement".into());
            accumulated += pause.probability * score(&pause.turn.pos);
            continue;
        }
        let mut scored: Vec<f64> = Vec::with_capacity(alternatives.len());
        for resumed in &alternatives {
            scored.push(turn_value(reg, resumed, score, depth + 1, notes)?);
        }
        // Side 0 is the maximiser the payoff matrix is written for.
        let best = if chooser == 0 {
            scored.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        } else {
            scored.iter().cloned().fold(f64::INFINITY, f64::min)
        };
        accumulated += pause.probability * best;
    }
    Ok(accumulated / total)
}

fn on_switch_in(reg: &Reg, turn: &mut Turn, side: usize, slot: usize) -> Result<(), String> {
    let (types, is_grounded) = {
        let Some(mon) = turn.mon_at(side, slot) else { return Ok(()) };
        (turn.types_of(mon), grounded(turn, mon))
    };

    let conditions: Vec<(Id, Option<i64>)> = turn.pos.sides[side]
        .side_conditions
        .iter()
        .map(|c| (c.id, c.layers))
        .collect();
    for (id, layers) in conditions {
        match id.as_str() {
            "stealthrock" => {
                let mult = reg.type_effectiveness("Rock", &types);
                let maxhp = turn.mon_at(side, slot).map(|m| m.maxhp).unwrap_or(0);
                let amount = ((maxhp as f64 * mult / 8.0) as i64).max(1);
                turn.deal_damage(side, slot, amount, false)?;
            }
            "spikes" if is_grounded => {
                let denominator = match layers.unwrap_or(1).min(3) {
                    1 => 8,
                    2 => 6,
                    _ => 4,
                };
                let maxhp = turn.mon_at(side, slot).map(|m| m.maxhp).unwrap_or(0);
                turn.deal_damage(side, slot, (maxhp / denominator).max(1), false)?;
            }
            "toxicspikes" if is_grounded => {
                if types.contains("Poison") {
                    turn.pos.sides[side]
                        .side_conditions
                        .retain(|c| c.id.as_str() != "toxicspikes");
                } else if !types.contains("Steel") {
                    let status = if layers.unwrap_or(1) >= 2 { "tox" } else { "psn" };
                    turn.apply_status(side, slot, status)?;
                }
            }
            "stickyweb" if is_grounded => {
                turn.apply_boosts(side, slot, &[("spe", -1)], true);
            }
            _ => {}
        }
    }

    if turn.mon_at(side, slot).map(|m| m.fainted).unwrap_or(true) {
        return Ok(());
    }
    switch_in_ability(turn, side, slot);
    check_white_herb(turn);
    Ok(())
}

fn switch_in_ability(turn: &mut Turn, side: usize, slot: usize) {
    let (ability, item, maxhp) = {
        let Some(mon) = turn.mon_at(side, slot) else { return };
        if mon.fainted {
            return;
        }
        (mon.ability, mon.item, mon.maxhp)
    };
    let _ = maxhp;

    let weather = match ability.as_str() {
        "drought" => Some("sunnyday"),
        "drizzle" => Some("raindance"),
        "sandstream" => Some("sandstorm"),
        "snowwarning" => Some("snowscape"),
        "desolateland" => Some("desolateland"),
        "primordialsea" => Some("primordialsea"),
        "deltastream" => Some("deltastream"),
        _ => None,
    };
    if let Some(weather) = weather {
        let current = turn.pos.field.weather.map(|w| w.as_str().to_string());
        if current.as_deref() != Some(weather) {
            turn.pos.field.weather = Some(Id::new(weather));
            let rock = match weather {
                "sunnyday" => Some("heatrock"),
                "raindance" => Some("damprock"),
                "sandstorm" => Some("smoothrock"),
                "snowscape" => Some("icyrock"),
                _ => None,
            };
            let extended = matches!((rock, item), (Some(r), Some(i)) if i.as_str() == r);
            turn.pos.field.weather_duration = Some(if extended { 8 } else { 5 });
        }
    }

    if ability == "intimidate" {
        for foe_slot in 0..turn.pos.sides[1 - side].active.len() {
            let skip = match turn.mon_at(1 - side, foe_slot) {
                None => true,
                Some(foe) => {
                    foe.fainted
                        || matches!(
                            foe.ability.as_str(),
                            "innerfocus"
                                | "oblivious"
                                | "owntempo"
                                | "scrappy"
                                | "guarddog"
                                | "clearbody"
                                | "whitesmoke"
                                | "fullmetalbody"
                                | "hypercutter"
                        )
                }
            };
            if skip {
                continue;
            }
            turn.apply_boosts(1 - side, foe_slot, &[("atk", -1)], true);
        }
    }

    if ability == "hospitality" {
        let ally_slot = 1 - slot;
        let amount = match turn.mon_at(side, ally_slot) {
            Some(ally) if !ally.fainted => (ally.maxhp / 4).max(1),
            _ => 0,
        };
        if amount > 0 {
            turn.heal(side, ally_slot, amount);
        }
    }
}

pub(crate) fn check_white_herb(turn: &mut Turn) {
    for side in 0..turn.pos.sides.len() {
        for slot in 0..turn.pos.sides[side].active.len() {
            let skip = match turn.mon_at(side, slot) {
                None => continue,
                Some(mon) => {
                    mon.fainted
                        || !matches!(mon.item, Some(i) if i.as_str() == "whiteherb")
                        || !mon.boosts.iter().any(|v| *v < 0)
                }
            };
            if skip {
                continue;
            }
            if let Some(mon) = turn.mon_at_mut(side, slot) {
                for value in mon.boosts.iter_mut() {
                    if *value < 0 {
                        *value = 0;
                    }
                }
            }
            turn.consume_item(side, slot);
        }
    }
}

/// A Pokemon moved to another forme: the species and the types, and nothing else.
///
/// It reads as though it resizes the Pokemon, because Python's `_change_forme` reads that
/// way -- recompute the battler, take `maxhp` from it, carry the HP across the difference.
/// It does not. `view.battler` hands `maxhp` back as *the position's own* whenever the
/// spread is known (`src/pokeuraou/view.py:132`), and a known spread is the only kind this
/// port resolves, so over there the maximum is written back unchanged and the difference
/// carried across is zero. Only the floor of 1 survives, and it is kept.
///
/// Writing the recomputed HP stat here instead -- which is what `do_mega` below does, and
/// gets away with because no mega changes its HP base -- moved every cell of a node whose
/// forme has a different one: a Kingambit given Stance Change became a 167-HP Aegislash
/// here and stayed a 207-HP one over there, for 6.4e-03 across 24 of 36 cells. Aegislash's
/// own two formes share their HP base, so only a holder that is not Aegislash shows it.
///
/// The stats the damage layer reads still change, because they are computed from
/// `mon.species` every time `Battler::from_pokemon` is called. It is the *stored* maximum
/// that stays.
///
/// The ability is deliberately left alone. A mega gets its target's ability and sets it
/// itself; Aegislash keeps Stance Change across both of its formes.
pub(crate) fn change_forme(
    reg: &Reg,
    turn: &mut Turn,
    side: usize,
    slot: usize,
    species_id: &str,
) -> Result<(), String> {
    let Some(entry) = reg.species.get(species_id) else { return Ok(()) };
    let types: Vec<Id> = entry.types.iter().map(|t| Id::new(t)).collect();
    let maxhp_before = {
        let Some(mon) = turn.mon_at_mut(side, slot) else { return Ok(()) };
        let before = mon.maxhp;
        mon.species = Id::new(species_id);
        mon.types = Types::from_slice(&types);
        before
    };
    {
        // Built because Python builds one here, so a species this cannot construct fails
        // at the forme change rather than somewhere later. Its stats are not read.
        let mon = turn.mon_at(side, slot).unwrap();
        let _refreshed = Battler::from_pokemon(reg, mon)?;
    }
    let mon = turn.mon_at_mut(side, slot).unwrap();
    // `mon.maxhp` is deliberately not assigned: see above. Written as the whole expression
    // anyway so the two implementations can be read against each other.
    mon.hp = (mon.hp + (mon.maxhp - maxhp_before)).max(1).min(mon.maxhp);
    Ok(())
}

fn do_mega(reg: &Reg, turn: &mut Turn, action: &QueuedAction) -> Result<(), String> {
    let (species, item) = {
        let Some(mon) = turn.mon_at(action.side, action.slot) else { return Ok(()) };
        if mon.fainted {
            return Ok(());
        }
        (mon.species, mon.item)
    };
    let Some(item) = item else { return Ok(()) };
    let Some(target) = reg
        .mega_targets
        .get(&(species.as_str().to_string(), item.as_str().to_string()))
        .cloned()
    else {
        return Ok(());
    };
    let target_id = crate::reg::to_id(&target);
    let Some(entry) = reg.species.get(&target_id) else {
        return Err(format!("mega target not in the regulation: {target}"));
    };
    let ability = crate::reg::to_id(&entry.abilities[0]);
    if !ability_handled(&ability) {
        return Err(format!("ability: {ability}"));
    }
    let types: Vec<Id> = entry.types.iter().map(|t| Id::new(t)).collect();
    let maxhp_before;
    {
        let mon = turn.mon_at_mut(action.side, action.slot).unwrap();
        maxhp_before = mon.maxhp;
        mon.species = Id::new(&target_id);
        mon.types = Types::from_slice(&types);
        mon.is_mega = true;
        mon.ability = Id::new(&ability);
    }
    let refreshed = {
        let mon = turn.mon_at(action.side, action.slot).unwrap();
        Battler::from_pokemon(reg, mon)?
    };
    {
        // HP keeps its absolute value, adjusted by any change in maximum, as the Python
        // does: `mon.hp = min(mon.maxhp, mon.hp + (mon.maxhp - maxhp_before))`.
        let mon = turn.mon_at_mut(action.side, action.slot).unwrap();
        mon.maxhp = refreshed.stats[0];
        mon.hp = (mon.hp + (mon.maxhp - maxhp_before)).min(mon.maxhp);
    }
    turn.pos.sides[action.side].mega_used = true;
    switch_in_ability(turn, action.side, action.slot);
    check_white_herb(turn);
    Ok(())
}
