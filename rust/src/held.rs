//! Positions this process holds for the caller, named by number (IKA-302).
//!
//! A decision sends the port the same positions again and again -- the node's root to
//! `needed`, both sides' `score` and every fill; a depth-2 pass each sub-game's position,
//! which this process wrote out as a turn's branch a moment before, to `score` and then to
//! its `fills` -- and each one crossed as 10 to 15 KB of JSON, written by the caller and read
//! here every time. Now a position crosses once: the caller defines it (`hold`, a line that
//! is not answered) or this process names a branch it wrote (`refs` on a `turn`), and every
//! request after that says `{"held": id}` where the position would have been.
//!
//! Numbers are never reused: the caller's are even and this process's are odd, from two
//! counters that only go up. The store is emptied by `forget` (a line that is not answered),
//! which the caller sends when a decision is over -- the span over which it promises not to
//! change a position in place (`rustnode.hold_positions`). A number this process does not
//! hold is a caller's bug and stops the request loudly rather than answering another
//! position's question.

use crate::position::Position;
use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

thread_local! {
    static STORE: RefCell<HashMap<u64, Position>> = RefCell::new(HashMap::new());
    static NEXT: Cell<u64> = const { Cell::new(1) };
    /// The kept (odd) numbers by `leaf_key`, to find an equal position already kept.
    static KEPT: RefCell<HashMap<u64, Vec<u64>>> = RefCell::new(HashMap::new());
}

/// The number a `{"held": id}` value names, if it is one.
pub fn held_id(value: &Value) -> Option<u64> {
    let object = value.as_object()?;
    if object.len() != 1 {
        return None;
    }
    object.get("held")?.as_u64()
}

/// A request's position: the JSON itself, or the one held under the number it names.
pub fn position(value: &Value) -> Position {
    match held_id(value) {
        None => Position::from_json(value),
        Some(id) => STORE.with(|store| match store.borrow().get(&id) {
            Some(position) => position.clone(),
            None => panic!("no position is held under {id}; the caller and the port disagree"),
        }),
    }
}

/// The position's JSON, a held one written out: for a cell thread, which cannot see this
/// thread's store and parses its own copy (IKA-32).
pub fn json(value: &Value) -> Value {
    match held_id(value) {
        None => value.clone(),
        Some(_) => position(value).to_json(),
    }
}

/// `hold`: the caller's position under the caller's (even) number.
pub fn define(id: u64, value: &Value) {
    let position = Position::from_json(value);
    STORE.with(|store| store.borrow_mut().insert(id, position));
}

/// A `hold` line: `position` whole, or (IKA-350) a completion of a held one -- `base` with
/// the Pokemon at the listed party indices of `side` replaced and that side's
/// `megaCapableSlots` set, everything else the base's. That is what the caller's
/// `hidden.substitute` builds, so the position is the one its whole JSON would have read
/// into: `from_json` reads each field on its own.
pub fn define_line(value: &Value) {
    let Some(id) = value["id"].as_u64() else { return };
    let Some(base) = value.get("base").and_then(Value::as_u64) else {
        define(id, &value["position"]);
        return;
    };
    let mut position = STORE.with(|store| match store.borrow().get(&base) {
        Some(position) => position.clone(),
        None => panic!("no position is held under {base}; the caller and the port disagree"),
    });
    let side_index = value["side"].as_u64().unwrap_or(0) as usize;
    let side = &mut position.sides[side_index];
    for pair in value["pokemon"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        let index = pair[0].as_u64().expect("a party index") as usize;
        side.pokemon[index] = std::rc::Rc::new(crate::position::Pokemon::from_json(&pair[1]));
    }
    side.mega_capable_slots = value["megaCapableSlots"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
        .unwrap_or_default();
    STORE.with(|store| store.borrow_mut().insert(id, position));
}

/// A position this process wrote, kept under a fresh (odd) number for the caller to name --
/// or under the number of an equal one it already kept this decision. The caller memoises
/// answers on its request's bytes, which once held the position's text: two equal branches
/// of two turns wrote the same text and hit the same answer, and under one number they
/// still do.
pub fn keep(position: &Position) -> u64 {
    let key = crate::encoded_node::leaf_key(position);
    let found = KEPT.with(|kept| {
        STORE.with(|store| {
            let store = store.borrow();
            kept.borrow().get(&key).and_then(|ids| {
                ids.iter().copied().find(|id| store.get(id).is_some_and(|held| held == position))
            })
        })
    });
    if let Some(id) = found {
        return id;
    }
    let id = NEXT.with(|next| {
        let id = next.get();
        next.set(id + 2);
        id
    });
    STORE.with(|store| store.borrow_mut().insert(id, position.clone()));
    KEPT.with(|kept| kept.borrow_mut().entry(key).or_default().push(id));
    id
}

/// `forget`: the decision is over.
pub fn forget() {
    STORE.with(|store| store.borrow_mut().clear());
    KEPT.with(|kept| kept.borrow_mut().clear());
}

// -- Slot actions by number (IKA-350).
//
// A candidate list crossed as one JSON object per slot action -- `score`, `qfeatures`, every
// fill's two lists, a turn's and a phase's choices, over half a megabyte a game of M-C
// generation -- and the same few hundred actions were written and read again every time.
// The caller now numbers each action text once (`acts`, a line that is not answered) and a
// list names them: `[[3, 7], [3, 9]]` where `[[{…}, {…}], …]` was. For the life of the
// process: an action's meaning never changes, so nothing is forgotten, and a number is
// never given to another text. A cell thread (IKA-32) reads the same table, so it is shared
// rather than per thread.

static ACTS: std::sync::RwLock<Option<HashMap<u64, crate::resolve::SlotAction>>> =
    std::sync::RwLock::new(None);

/// `acts`: `[[id, {…action…}], …]`, each number the caller's for that action text.
pub fn define_actions(listed: &Value) {
    let Some(pairs) = listed.as_array() else { return };
    let mut table = ACTS.write().expect("the action table's lock");
    let table = table.get_or_insert_with(HashMap::new);
    for pair in pairs {
        if let (Some(id), Some(action)) = (pair[0].as_u64(), pair.get(1)) {
            let parsed = crate::resolve::parse_slot_action(action);
            table.insert(id, parsed);
        }
    }
}

/// The action the caller numbered `id`. A number never defined is the caller's bug and
/// stops the request loudly, as an unknown held position does.
pub fn action(id: u64) -> crate::resolve::SlotAction {
    let table = ACTS.read().expect("the action table's lock");
    match table.as_ref().and_then(|t| t.get(&id)) {
        Some(action) => action.clone(),
        None => panic!("no action is numbered {id}; the caller and the port disagree"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_lone_held_key_is_a_number() {
        assert_eq!(held_id(&serde_json::json!({ "held": 4 })), Some(4));
        assert_eq!(held_id(&serde_json::json!({ "held": 4, "format": "x" })), None);
        assert_eq!(held_id(&serde_json::json!({ "format": "x" })), None);
    }

    #[test]
    fn kept_numbers_are_odd_and_never_reused() {
        let a = keep(&test_position());
        assert_eq!(keep(&test_position()), a, "an equal position is kept once");
        forget();
        let b = keep(&test_position());
        assert!(a % 2 == 1 && b % 2 == 1 && b > a);
    }

    fn test_position() -> Position {
        Position::from_json(&serde_json::json!({
            "format": "t",
            "sides": [{ "id": "p1", "active": [], "pokemon": [] }, { "id": "p2", "active": [], "pokemon": [] }],
        }))
    }
}
