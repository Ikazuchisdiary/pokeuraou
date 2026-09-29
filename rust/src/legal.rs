//! IKA-389: one side's legal actions -- `actions.side_actions` less
//! `narrow.drop_dead_actions` (`qhead.legal_pool`) -- and a candidate's rows for a Q
//! (`qhead.encode_actions`), in the port.
//!
//! The Python is the reference: `tests/test_actions.py` holds it to Showdown's own
//! `probe_choices`, and this is held to the Python, action for action and in the same order,
//! on those positions and on thousands of generated ones (`tests/test_port_menus.py`,
//! records/IKA-389.md). The order is part of the answer: a Q's game is solved over the pool
//! as listed, and an LP's answer moves with the order of its rows.
//!
//! Every rule below names the Python function it is.

use crate::encode::Vocabulary;
use crate::id::Id;
use crate::position::{Pokemon, Position, Side};
use crate::reg::{Category, Reg, F_SOUND};
use crate::resolve::SlotAction;

/// `actions.STRUGGLE`, `actions.RECHARGE`.
const STRUGGLE: &str = "struggle";
const RECHARGE: &str = "recharge";
/// `actions.TRAPPING_CONDITIONS`.
const TRAPPING_CONDITIONS: [&str; 2] = ["partiallytrapped", "trapped"];
/// `actions.LOCKING_VOLATILES`.
const LOCKING_VOLATILES: [&str; 2] = ["twoturnmove", "lockedmove"];
/// `actions.TRAPPING_ABILITIES`.
const TRAPPING_ABILITIES: [&str; 3] = ["shadowtag", "arenatrap", "magnetpull"];
/// `actions.DISABLED_ONCE_MOVED`.
const DISABLED_ONCE_MOVED: [&str; 2] = ["fakeout", "firstimpression"];
/// `moveinfo.FIRST_TURN_OUT_MOVES`.
const FIRST_TURN_OUT_MOVES: [&str; 3] = ["fakeout", "firstimpression", "matblock"];
/// `regulation.TARGETS_REQUIRING_FOE` / `_ALLY` / `_WITHOUT_CHOICE`.
const TARGETS_REQUIRING_FOE: [&str; 3] = ["normal", "any", "adjacentFoe"];
const TARGETS_REQUIRING_ALLY: [&str; 2] = ["adjacentAlly", "adjacentAllyOrSelf"];
const TARGETS_WITHOUT_CHOICE: [&str; 10] = [
    "self", "all", "allAdjacent", "allAdjacentFoes", "allies", "allySide", "allyTeam", "foeSide",
    "randomNormal", "scripted",
];

/// One slot's choice, as `actions.MoveAction` / `SwitchAction` / `PassAction` hold it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    Move { slot: usize, index: usize, move_id: Id, target: Option<i64>, mega: bool },
    Switch { slot: usize, party: usize, species: Id },
    Pass { slot: usize },
}

impl Act {
    /// `to_choice`.
    pub fn write_choice(&self, out: &mut String) {
        use std::fmt::Write;
        match self {
            Act::Move { index, target, mega, .. } => {
                let _ = write!(out, "move {index}");
                if let Some(t) = target {
                    let _ = write!(out, " {t}");
                }
                if *mega {
                    out.push_str(" mega");
                }
            }
            Act::Switch { party, .. } => {
                let _ = write!(out, "switch {party}");
            }
            Act::Pass { .. } => out.push_str("pass"),
        }
    }

    /// The resolver's action (`rustnode.dump_action` read back by `parse_slot_action`).
    pub fn slot_action(&self) -> SlotAction {
        match *self {
            Act::Move { slot, move_id, target, mega, .. } => SlotAction::Move { slot, move_id, target, mega },
            Act::Switch { slot, party, species } => SlotAction::Switch { slot, party_index: party, species },
            Act::Pass { slot } => SlotAction::Pass { slot },
        }
    }

    /// The compact form the caller rebuilds its `SlotAction` from (`portmenus._slot`).
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            Act::Move { slot, index, move_id, target, mega } => {
                json!(["m", slot, index, move_id.as_str(), target, mega])
            }
            Act::Switch { slot, party, species } => json!(["s", slot, party, species.as_str()]),
            Act::Pass { slot } => json!(["p", slot]),
        }
    }
}

/// One side's whole choice: an `Act` per active slot.
pub type SideAct = Vec<Act>;

/// `SideAction.to_choice`: the slots' choices joined by ", ".
pub fn choice(action: &SideAct) -> String {
    let mut out = String::with_capacity(24);
    for (k, act) in action.iter().enumerate() {
        if k > 0 {
            out.push_str(", ");
        }
        act.write_choice(&mut out);
    }
    out
}

fn text(id: &Option<Id>) -> Option<&str> {
    id.as_ref().map(Id::as_str).filter(|s| !s.is_empty())
}

fn move_flag(reg: &Reg, move_id: &str, flag: &str) -> bool {
    reg.moves
        .get(move_id)
        .is_some_and(|m| m.raw["flags"].as_object().is_some_and(|f| f.contains_key(flag)))
}

fn species_types<'a>(reg: &'a Reg, mon: &'a Pokemon) -> Vec<&'a str> {
    if !mon.types.is_empty() {
        return mon.types.as_slice().iter().map(Id::as_str).collect();
    }
    reg.species.get(mon.species.as_str()).map(|s| s.types.iter().map(String::as_str).collect()).unwrap_or_default()
}

/// `imprisoned_moves`.
fn imprisoned_moves(pos: &Position, side: usize) -> Vec<Id> {
    let foe = &pos.sides[1 - side];
    let mut out = Vec::new();
    for party in foe.active.iter().flatten() {
        let mon = &foe.pokemon[*party];
        if mon.fainted || !mon.has_volatile("imprison") {
            continue;
        }
        for m in mon.moves.iter() {
            if m.id.as_str() != STRUGGLE && !out.contains(&m.id) {
                out.push(m.id);
            }
        }
    }
    out
}

/// `_disabled_after_itself`.
fn disabled_after_itself(reg: &Reg, move_id: Id, mon: &Pokemon) -> bool {
    reg.moves.contains_key(move_id.as_str())
        && move_flag(reg, move_id.as_str(), "cantusetwice")
        && mon.last_move == Some(move_id)
}

/// `_disabled_once_moved`.
fn disabled_once_moved(reg: &Reg, move_id: Id) -> bool {
    DISABLED_ONCE_MOVED.contains(&move_id.as_str())
        && reg.moves.get(move_id.as_str()).is_some_and(|m| {
            m.raw["customHooks"].as_array().is_some_and(|h| h.iter().any(|v| v.as_str() == Some("onDisableMove")))
        })
}

/// `_usable_move_slots`: (1-based index, move id).
fn usable_move_slots(reg: &Reg, mon: &Pokemon, imprisoned: &[Id]) -> Vec<(usize, Id)> {
    if let Some(locked) = text(&mon.locked_move) {
        for (i, m) in mon.moves.iter().enumerate() {
            if m.id.as_str() == locked && m.usable() {
                if disabled_after_itself(reg, m.id, mon) {
                    return Vec::new();
                }
                if imprisoned.contains(&m.id) && !mon.has_volatile("twoturnmove") {
                    return Vec::new();
                }
                return vec![(i + 1, m.id)];
            }
        }
    }
    let taunted = mon.has_volatile("taunt");
    let tormented = mon.has_volatile("torment");
    let throat_chopped = mon.has_volatile("throatchop");
    let disabled_move = mon.volatile("disable").and_then(|e| e.move_id);
    let encored_move = mon.volatile("encore").and_then(|e| e.move_id);
    let choice_move = mon.volatile("choicelock").and_then(|choice| {
        let wanted = text(&choice.move_id)?;
        let item = mon.item?;
        (reg.choice_items.contains(item.as_str()) && mon.moves.iter().any(|m| m.id.as_str() == wanted))
            .then_some(choice.move_id.unwrap())
    });
    let mut out = Vec::new();
    for (i, m) in mon.moves.iter().enumerate() {
        if !m.usable() {
            continue;
        }
        let entry = reg.moves.get(m.id.as_str());
        if taunted && entry.is_some_and(|e| e.category == Category::Status) {
            continue;
        }
        if throat_chopped && entry.is_some_and(|e| e.flags.has(F_SOUND)) {
            continue;
        }
        if disabled_move.is_some_and(|d| d == m.id) {
            continue;
        }
        if encored_move.is_some_and(|e| e != m.id) {
            continue;
        }
        if choice_move.is_some_and(|c| c != m.id) {
            continue;
        }
        if tormented && mon.last_move == Some(m.id) {
            continue;
        }
        if mon.active_move_actions != 0 && disabled_once_moved(reg, m.id) {
            continue;
        }
        if disabled_after_itself(reg, m.id, mon) {
            continue;
        }
        if imprisoned.contains(&m.id) {
            continue;
        }
        out.push((i + 1, m.id));
    }
    out
}

/// `_targets_for`: the targets of one move from one slot (`user_types` for Curse).
fn targets_for(
    reg: &Reg,
    move_id: &str,
    slot: usize,
    foe: &Side,
    active_per_side: usize,
    user_types: &[&str],
) -> Vec<Option<i64>> {
    let Some(entry) = reg.moves.get(move_id) else {
        return vec![None];
    };
    let mut target = entry.target.as_str();
    if move_id == "curse" && !user_types.contains(&"Ghost") {
        target = "self";
    }
    if TARGETS_WITHOUT_CHOICE.contains(&target) {
        return vec![None];
    }
    if TARGETS_REQUIRING_FOE.contains(&target) {
        let mut options = Vec::new();
        for i in 0..active_per_side {
            let mon = foe.active.get(i).copied().flatten().map(|p| &foe.pokemon[p]);
            if mon.is_some_and(|m| !m.fainted) {
                options.push(Some(i as i64 + 1));
            }
        }
        if target == "any" {
            for i in 0..active_per_side {
                if i != slot {
                    options.push(Some(-(i as i64 + 1)));
                }
            }
        }
        return options;
    }
    if TARGETS_REQUIRING_ALLY.contains(&target) {
        let mut options = Vec::new();
        for i in 0..active_per_side {
            if target == "adjacentAlly" && i == slot {
                continue;
            }
            options.push(Some(-(i as i64 + 1)));
        }
        return options;
    }
    vec![None]
}

/// `locked_move` (the function, not the field).
fn locked_move(reg: &Reg, mon: &Pokemon) -> Option<Id> {
    for vid in LOCKING_VOLATILES {
        let Some(effect) = mon.volatile(vid) else { continue };
        if text(&effect.move_id).is_some() {
            return effect.move_id;
        }
        if vid == "twoturnmove" {
            if let Some(last) = text(&mon.last_move) {
                if reg.moves.contains_key(last) && move_flag(reg, last, "charge") {
                    return mon.last_move;
                }
            }
        }
    }
    None
}

/// `charge_target`.
fn charge_target(mon: &Pokemon) -> Option<i64> {
    let charging = mon.volatile("twoturnmove")?;
    text(&charging.move_id)?;
    // An int that is not a bool and not 0 (`isinstance(loc, int) and not isinstance(loc, bool)`).
    charging.extra.get("targetLoc").and_then(|v| v.as_i64()).filter(|v| *v != 0)
}

/// `_grounded`.
fn grounded(pos: &Position, mon: &Pokemon, types: &[&str]) -> bool {
    if pos.field.has_pseudo_weather("gravity") {
        return true;
    }
    if mon.has_volatile("ingrain") || mon.has_volatile("smackdown") || mon.item.is_some_and(|i| i.as_str() == "ironball") {
        return true;
    }
    if types.contains(&"Flying") || mon.ability.as_str() == "levitate" {
        return false;
    }
    if mon.has_volatile("magnetrise") || mon.has_volatile("telekinesis") {
        return false;
    }
    mon.item.is_none_or(|i| i.as_str() != "airballoon")
}

/// `_adjacent_foes`.
fn adjacent_foes(active_per_side: usize, mine: usize, theirs: usize) -> bool {
    if active_per_side <= 2 {
        return true;
    }
    (mine as i64 + theirs as i64 + 1 - active_per_side as i64).abs() <= 1
}

/// `_trapped_by_foe_ability`.
fn trapped_by_foe_ability(reg: &Reg, pos: &Position, side: usize, mon: &Pokemon) -> bool {
    let own = &pos.sides[side];
    let foe = &pos.sides[1 - side];
    let Some(mine) = mon.active_index else { return false };
    for (position, party) in foe.active.iter().enumerate() {
        let Some(party) = party else { continue };
        let holder = &foe.pokemon[*party];
        if holder.fainted || !TRAPPING_ABILITIES.contains(&holder.ability.as_str()) {
            continue;
        }
        if !adjacent_foes(own.active.len(), mine, position) {
            continue;
        }
        match holder.ability.as_str() {
            "shadowtag" => {
                if mon.ability.as_str() != "shadowtag" {
                    return true;
                }
                continue;
            }
            ability => {
                let types = species_types(reg, mon);
                if ability == "magnetpull" && types.contains(&"Steel") {
                    return true;
                }
                if ability == "arenatrap" && grounded(pos, mon, &types) {
                    return true;
                }
            }
        }
    }
    false
}

/// `_escapes_traps`.
fn escapes_traps(reg: &Reg, mon: &Pokemon) -> bool {
    let types = species_types(reg, mon);
    if reg.effect_immunities.get("trapped").is_some_and(|immune| types.iter().any(|t| immune.contains(*t))) {
        return true;
    }
    if mon.item.is_some_and(|i| i.as_str() == "shedshell") && reg.items_freeing.contains("shedshell") {
        return true;
    }
    mon.ability.as_str() == "runaway" && reg.abilities_freeing.contains("runaway")
}

/// `_is_trapped`.
fn is_trapped(reg: &Reg, pos: &Position, side: usize, mon: &Pokemon) -> bool {
    if mon.trapped {
        return true;
    }
    if locked_move(reg, mon).is_some() {
        return true;
    }
    if escapes_traps(reg, mon) {
        return false;
    }
    if mon.volatiles.iter().any(|v| {
        TRAPPING_CONDITIONS.contains(&v.id.as_str()) || reg.trapping_volatiles.contains(v.id.as_str())
    }) {
        return true;
    }
    if pos.field.pseudo_weather.iter().any(|e| reg.trapping_pseudo_weather.contains(e.id.as_str())) {
        return true;
    }
    trapped_by_foe_ability(reg, pos, side, mon)
}

/// `slot_actions` (mega and switches allowed).
fn slot_actions(reg: &Reg, pos: &Position, side_index: usize, slot: usize) -> Vec<Act> {
    let side = &pos.sides[side_index];
    let foe = &pos.sides[1 - side_index];
    let active_per_side = side.active.len();
    let Some(party_slot) = side.active.get(slot).copied().flatten() else {
        return vec![Act::Pass { slot }];
    };
    let mon = &side.pokemon[party_slot];
    if mon.fainted {
        return vec![Act::Pass { slot }];
    }
    if mon.has_volatile("mustrecharge") {
        return vec![Act::Move { slot, index: 1, move_id: Id::new(RECHARGE), target: None, mega: false }];
    }
    if let Some(locked) = locked_move(reg, mon) {
        if let Some(i) = mon.moves.iter().position(|m| m.id == locked) {
            let index = i + 1;
            let charging = mon.volatile("twoturnmove");
            if charging.is_some_and(|c| c.move_id == Some(locked)) && charge_target(mon).is_some() {
                return vec![Act::Move { slot, index, move_id: locked, target: None, mega: false }];
            }
            return targets_for(reg, locked.as_str(), slot, foe, active_per_side, &[])
                .into_iter()
                .map(|target| Act::Move { slot, index, move_id: locked, target, mega: false })
                .collect();
        }
    }
    let mut out = Vec::new();
    let usable = usable_move_slots(reg, mon, &imprisoned_moves(pos, side_index));
    let mega = !side.mega_used
        && !mon.is_mega
        && mon.item.is_some_and(|item| {
            !item.is_empty()
                && reg.mega_targets.contains_key(&(mon.species.as_str().to_string(), item.as_str().to_string()))
        });
    if !usable.is_empty() {
        let user_types = species_types(reg, mon);
        for (index, move_id) in usable {
            for target in targets_for(reg, move_id.as_str(), slot, foe, active_per_side, &user_types) {
                out.push(Act::Move { slot, index, move_id, target, mega: false });
                if mega {
                    out.push(Act::Move { slot, index, move_id, target, mega: true });
                }
            }
        }
    } else {
        for target in targets_for(reg, STRUGGLE, slot, foe, active_per_side, &[]) {
            out.push(Act::Move { slot, index: 1, move_id: Id::new(STRUGGLE), target, mega: false });
        }
    }
    if !is_trapped(reg, pos, side_index, mon) {
        for candidate in &side.pokemon {
            if candidate.fainted || candidate.is_active() {
                continue;
            }
            out.push(Act::Switch { slot, party: candidate.slot + 1, species: candidate.species });
        }
    }
    out
}

/// `side_actions`: the product of the slots' lists in order, less two Megas or two switches
/// to one Pokemon.
pub fn side_actions(reg: &Reg, pos: &Position, side_index: usize) -> Vec<SideAct> {
    let side = &pos.sides[side_index];
    let per_slot: Vec<Vec<Act>> = (0..side.active.len()).map(|slot| slot_actions(reg, pos, side_index, slot)).collect();
    let mut out: Vec<SideAct> = vec![Vec::new()];
    // `itertools.product`: the first list outermost.
    for options in &per_slot {
        let mut next = Vec::with_capacity(out.len() * options.len());
        for prefix in &out {
            for act in options {
                let mut combo = prefix.clone();
                combo.push(*act);
                next.push(combo);
            }
        }
        out = next;
    }
    out.retain(|combo| {
        let megas = combo.iter().filter(|a| matches!(a, Act::Move { mega: true, .. })).count();
        if megas > 1 {
            return false;
        }
        let switches: Vec<usize> =
            combo.iter().filter_map(|a| if let Act::Switch { party, .. } = a { Some(*party) } else { None }).collect();
        let mut distinct = switches.clone();
        distinct.sort_unstable();
        distinct.dedup();
        distinct.len() == switches.len()
    });
    out
}

/// `drop_dead_actions`: a combination with a move that fails for a user already in (Fake Out
/// and the like) goes, unless that empties the pool.
pub fn drop_dead_actions(pos: &Position, side: usize, pool: Vec<SideAct>) -> Vec<SideAct> {
    if pool.is_empty() {
        return pool;
    }
    let own = &pos.sides[side];
    let dead = |action: &SideAct| {
        action.iter().any(|act| match act {
            Act::Move { slot, move_id, .. } if FIRST_TURN_OUT_MOVES.contains(&move_id.as_str()) => {
                own.active_pokemon(*slot).is_some_and(|mon| mon.active_move_actions > 0)
            }
            _ => false,
        })
    };
    let alive: Vec<SideAct> = pool.iter().filter(|a| !dead(a)).cloned().collect();
    if alive.is_empty() {
        pool
    } else {
        alive
    }
}

/// `qhead.legal_pool`.
pub fn legal_pool(reg: &Reg, pos: &Position, side: usize) -> Vec<SideAct> {
    drop_dead_actions(pos, side, side_actions(reg, pos, side))
}

/// `qhead` field order and codes.
const KIND_PASS: i32 = 0;
const KIND_MOVE: i32 = 1;
const KIND_SWITCH: i32 = 2;
const TARGET_NONE: i32 = 0;
const TARGET_FOE: i32 = 1;
const TARGET_ALLY: i32 = 2;
/// Numbers per slot (`qhead.ACTION_FIELDS`) and slots per action (`qhead.SLOTS`).
pub const ACTION_FIELDS: usize = 7;
pub const SLOTS: usize = 2;

/// `qhead.encode_actions`: (N, 2, 7) int32, row-major, or the Python's `ValueError` (a switch
/// to a party index no Pokemon has).
pub fn encode_actions(vocab: &Vocabulary, pos: &Position, side: usize, actions: &[SideAct]) -> Result<Vec<i32>, String> {
    let own = &pos.sides[side];
    let foe = &pos.sides[1 - side];
    let mut out = vec![-1i32; actions.len() * SLOTS * ACTION_FIELDS];
    for (n, action) in actions.iter().enumerate() {
        for (t, act) in action.iter().take(SLOTS).enumerate() {
            let row = &mut out[(n * SLOTS + t) * ACTION_FIELDS..(n * SLOTS + t + 1) * ACTION_FIELDS];
            let slot = match act {
                Act::Move { slot, .. } | Act::Switch { slot, .. } | Act::Pass { slot } => *slot,
            };
            let actor = own.active.get(slot).copied().flatten();
            row[1] = actor.map_or(-1, |a| a as i32);
            row[2] = 0;
            row[3] = 0;
            row[4] = TARGET_NONE;
            match act {
                Act::Move { move_id, target, mega, .. } => {
                    row[0] = KIND_MOVE;
                    row[2] = vocab.moves.get(move_id.as_str()).copied().unwrap_or(0);
                    row[3] = i32::from(*mega);
                    match target {
                        Some(t) if *t > 0 => {
                            row[4] = TARGET_FOE;
                            let at = foe.active.get((*t - 1) as usize).copied().flatten();
                            row[5] = at.map_or(-1, |a| a as i32);
                        }
                        Some(t) if *t < 0 => {
                            row[4] = TARGET_ALLY;
                            let at = own.active.get((-*t - 1) as usize).copied().flatten();
                            row[5] = at.map_or(-1, |a| a as i32);
                        }
                        _ => {}
                    }
                }
                Act::Switch { party, .. } => {
                    row[0] = KIND_SWITCH;
                    let found = own.pokemon.iter().position(|mon| mon.slot + 1 == *party);
                    row[6] = match found {
                        Some(r) => r as i32,
                        None => return Err(format!("no Pokemon in party slot {party}")),
                    };
                }
                Act::Pass { .. } => row[0] = KIND_PASS,
            }
        }
    }
    Ok(out)
}
