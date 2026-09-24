//! The moves whose damage is Showdown's `damageCallback` (IKA-213).
//!
//! A `damageCallback` is a hook, not a field, so the regulation dump lists it under
//! `customHooks` and `UNHANDLED_MOVE_FIELDS` never saw it: Counter, Mirror Coat, Metal Burst
//! and Comeuppance reached the calculator with a base power of 0, hit the first foe for
//! nothing and noted `move.basePowerCallback`. Final Gambit, Super Fang and Endeavor were
//! already `moveinfo::fixed_damage`; Endeavor's `onTryImmunity` is here.
//!
//! Showdown a5df827, data/moves.ts (the champions mod overrides none of them):
//!
//! ```text
//! counter: damageCallback(pokemon) { if (!pokemon.volatiles['counter']) return 0;
//!                                     return pokemon.volatiles['counter'].damage || 1; }
//!     priority -5, beforeTurnCallback: addVolatile('counter'),
//!     onTry(source): false without the volatile or while its `slot` is null,
//!     condition.onDamagingHit(damage, target, source, move):
//!         if (!source.isAlly(target) && this.getCategory(move) === 'Physical')
//!             { effectState.slot = source.getSlot(); effectState.damage = 2 * damage; }
//!     condition.onRedirectTarget (priority -1): move.id === 'counter' -> getAtSlot(slot)
//! mirrorcoat: the same with 'Special'.
//! metalburst / comeuppance:
//!     damageCallback(pokemon) { const d = pokemon.getLastDamagedBy(true);
//!                               if (d !== undefined) return (d.damage * 1.5) || 1; return 0; }
//!     onTry(source): false unless getLastDamagedBy(true)?.thisTurn
//!     onModifyTarget: target = getAtSlot(getLastDamagedBy(true).slot)
//! endeavor: damageCallback = target.hp - pokemon.hp; onTryImmunity: pokemon.hp < target.hp
//! ```
//!
//! `getLastDamagedBy(true)` is the last `attackedBy` entry from a foe whose damage is a number:
//! `hitStepMoveHitLoop` pushes one per target when a damaging move ends, with the last hit's
//! damage, and a Substitute that took the hit leaves none. Entries from earlier turns have
//! `thisTurn` false, so for these moves the record of one turn is the whole record, and it
//! starts empty: `Turn::damaged_by`. `spreadDamage` floors Metal Burst's fraction
//! (`clampIntRange(damage, 1)`). A switch-in clears `attackedBy` in Showdown and not here:
//! a Pokemon that comes in mid-turn has no move queued that turn, so its slot's record is
//! never read. The record is keyed by slot because `getAtSlot` is: a reply lands on whoever
//! stands where the attacker stood (a U-turn's replacement), and it crosses a pause as JSON.

use crate::battler::Battler;
use crate::resolve::{Slot, Turn};
use crate::speed::QueuedAction;
use serde_json::{json, Value};

/// One active slot's record for the turn: `attackedBy`'s last foe entry, and the last
/// physical and special ones (Counter's and Mirror Coat's `effectState`), each as the
/// attacker's slot and the damage its last hit dealt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DamagedBy {
    pub last: Option<(Slot, i64)>,
    pub physical: Option<(Slot, i64)>,
    pub special: Option<(Slot, i64)>,
}

/// The moves whose `damageCallback` this port implements. Their damage skips `modifyDamage`,
/// so a resist berry (`onSourceModifyDamage`) is not eaten (`moves::after_hit`). Consulted by
/// no gate: a damaging move with custom code the port never names is reported
/// (`modelled::damaging_move_is_unmodelled`), and `tests/test_damage_callback_oracle.py` holds
/// this list to every move with the hook in the regulation dumps, so a new one is a failing
/// test before it is a note.
pub(crate) fn ported(move_id: &str) -> bool {
    matches!(
        move_id,
        "counter" | "mirrorcoat" | "metalburst" | "comeuppance" | "endeavor" | "superfang" | "finalgambit"
    )
}

/// A foe's damaging hit landed on `target` (a number, 0 included; not a doll).
pub(crate) fn record(turn: &mut Turn, source: Slot, target: Slot, dealt: i64, category: &str) {
    if source.0 == target.0 {
        return;
    }
    let entry = &mut turn.damaged_by[target.0][target.1];
    entry.last = Some((source, dealt));
    match category {
        "Physical" => entry.physical = Some((source, dealt)),
        "Special" => entry.special = Some((source, dealt)),
        _ => {}
    }
}

fn entry(turn: &Turn, action: &QueuedAction, move_id: &str) -> Option<(Slot, i64)> {
    let record = turn.damaged_by.get(action.side)?.get(action.slot)?;
    match move_id {
        "counter" => record.physical,
        "mirrorcoat" => record.special,
        "metalburst" | "comeuppance" => record.last,
        _ => None,
    }
}

fn replies(move_id: &str) -> bool {
    matches!(move_id, "counter" | "mirrorcoat" | "metalburst" | "comeuppance")
}

/// The move's `onTry`: a reply with nothing this turn to reply to fails.
pub(crate) fn fails_on_try(turn: &Turn, action: &QueuedAction, move_id: &str) -> bool {
    replies(move_id) && entry(turn, action, move_id).is_none()
}

/// The slot a reply is aimed at: Counter's and Mirror Coat's `onRedirectTarget`, Metal
/// Burst's and Comeuppance's `onModifyTarget`. None for any other move.
pub(crate) fn target(turn: &Turn, action: &QueuedAction, move_id: &str) -> Option<Slot> {
    entry(turn, action, move_id).map(|(slot, _)| slot)
}

/// Counter and Mirror Coat redirect to the recorded slot *after* `getMoveTargets` retargeted a
/// fainted one, so they go back to it (and fail); Metal Burst's target was set before.
pub(crate) fn redirects_back(move_id: &str) -> bool {
    matches!(move_id, "counter" | "mirrorcoat")
}

/// What `damageCallback` returns for a reply, or 0 for any other move.
pub(crate) fn damage(turn: &Turn, action: &QueuedAction, move_id: &str) -> i64 {
    match (move_id, entry(turn, action, move_id)) {
        ("counter" | "mirrorcoat", Some((_, dealt))) => (2 * dealt).max(1),
        ("metalburst" | "comeuppance", Some((_, dealt))) => (dealt * 3 / 2).max(1),
        _ => 0,
    }
}

/// Endeavor's `onTryImmunity`: `pokemon.hp < target.hp`, or the target is immune.
pub(crate) fn immune_on_try(move_id: &str, attacker: &Battler, defender: &Battler) -> bool {
    move_id == "endeavor" && attacker.hp >= defender.hp
}

fn entry_json(entry: Option<(Slot, i64)>) -> Value {
    match entry {
        None => Value::Null,
        Some(((side, slot), dealt)) => json!([side, slot, dealt]),
    }
}

fn entry_from(value: &Value) -> Result<Option<(Slot, i64)>, String> {
    if value.is_null() {
        return Ok(None);
    }
    let part = |i: usize| value[i].as_i64().ok_or("a damagedBy entry does not parse");
    Ok(Some(((part(0)? as usize, part(1)? as usize), part(2)?)))
}

/// A pause's copy of `Turn::damaged_by` (commands.rs), side by side and slot by slot.
pub(crate) fn to_json(records: &[[DamagedBy; 2]; 2]) -> Value {
    json!(records
        .iter()
        .map(|side| side
            .iter()
            .map(|r| json!([entry_json(r.last), entry_json(r.physical), entry_json(r.special)]))
            .collect::<Vec<_>>())
        .collect::<Vec<_>>())
}

/// The inverse of `to_json`. A pause written before IKA-213 has none: nothing was recorded.
pub(crate) fn from_json(value: &Value) -> Result<[[DamagedBy; 2]; 2], String> {
    let mut out: [[DamagedBy; 2]; 2] = Default::default();
    if value.is_null() {
        return Ok(out);
    }
    for (side, row) in out.iter_mut().enumerate() {
        for (slot, record) in row.iter_mut().enumerate() {
            let cell = &value[side][slot];
            *record = DamagedBy {
                last: entry_from(&cell[0])?,
                physical: entry_from(&cell[1])?,
                special: entry_from(&cell[2])?,
            };
        }
    }
    Ok(out)
}
