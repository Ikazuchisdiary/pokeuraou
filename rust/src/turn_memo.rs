//! A cell's turn, kept for a later request of this process that asks the same cell (IKA-375).
//!
//! The ladder (`src/pokeuraou/ladder.py`) reads a root cell again at each stage whose kind
//! differs -- more branches kept, wider children's menus, the knock-out fork -- and each time
//! its children's sub-games are filled anew. A child's sub-game at eight actions a side is
//! the top-left of the same child's at twelve, so most of its cells are a turn this process
//! resolved a stage before: 40% of the sub-game cells an 8 s read of the open game asked were
//! asked before in the same read, 49% with the bench hidden. Resolving turns is 88% of what
//! a worker's port spends on its fills, and the port is the largest share of the machine's
//! CPU in an all-core read.
//!
//! Only for a request that asks (`"memo": true` on a `fills` node, which only the ladder's
//! workers send), so every other road's requests and answers are the bytes they were.
//!
//! Exact, not approximate: `resolve_turn` is a function of the position, the two sides'
//! actions and the budget, and an entry is found by all three compared whole (`==`, derived
//! over every field), with a cheap hash only to find the candidates. What is kept is what
//! `encoded_node`'s collector takes from a turn that did not pause -- its branches'
//! probabilities and positions in order, `exact`, the notes -- or the refusal's reason. A
//! turn that paused for a replacement is not kept (its pauses hold the turn's own state,
//! resumed later); it is resolved every time, as before.
//!
//! Bounded: two generations of `capacity()` entries each (`POKEURAOU_PORT_TURN_MEMO`, 0 turns
//! it off). When the newer is full the older is dropped and the newer becomes the older, so
//! at most twice that many turns are held, the most recently kept or found.

use crate::position::Position;
use crate::resolve::{Branch, Budget, SlotAction, TurnResult};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::hash::{Hash, Hasher};

/// Entries per generation, from `POKEURAOU_PORT_TURN_MEMO` (default 2048; 0: off).
///
/// An entry is its branches' positions, and a branch copies the Pokemon the turn touched
/// (928 bytes each): about 10 KB an entry. 16384 a generation took fifteen workers' ports to
/// 12.5 GB and the host to under 1 GB free in an 8 s read.
fn capacity() -> usize {
    static CAPACITY: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAPACITY.get_or_init(|| {
        std::env::var("POKEURAOU_PORT_TURN_MEMO")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048)
    })
}

/// A kept turn: the non-paused result's parts, or the refusal.
enum Kept {
    Turn { exact: bool, unmodelled: BTreeSet<String>, branches: Vec<(f64, Position)> },
    Refused(String),
}

struct Entry {
    position: Position,
    actions: [Vec<SlotAction>; 2],
    budget: Budget,
    kept: Kept,
}

#[derive(Default)]
struct Memo {
    new: HashMap<u64, Vec<Entry>>,
    old: HashMap<u64, Vec<Entry>>,
    count: usize,
}

/// What the memo did in this process: cells found, cells resolved and kept, generations
/// dropped. For the wire report and a node's header.
#[derive(Default, Clone, Copy)]
pub struct Counts {
    pub hits: u64,
    pub misses: u64,
    pub dropped: u64,
}

thread_local! {
    static MEMO: RefCell<Memo> = RefCell::new(Memo::default());
    static COUNTS: std::cell::Cell<Counts> = const { std::cell::Cell::new(Counts { hits: 0, misses: 0, dropped: 0 }) };
}

pub fn counts() -> Counts {
    COUNTS.with(std::cell::Cell::get)
}

fn hash_action(action: &SlotAction, hasher: &mut impl Hasher) {
    match action {
        SlotAction::Move { slot, move_id, target, mega } => {
            0u8.hash(hasher);
            slot.hash(hasher);
            move_id.hash(hasher);
            target.hash(hasher);
            mega.hash(hasher);
        }
        SlotAction::Switch { slot, party_index, species } => {
            1u8.hash(hasher);
            slot.hash(hasher);
            party_index.hash(hasher);
            species.hash(hasher);
        }
        SlotAction::Pass { slot } => {
            2u8.hash(hasher);
            slot.hash(hasher);
        }
    }
}

fn key(position: &Position, actions: &[Vec<SlotAction>; 2], budget: &Budget) -> u64 {
    let mut hasher = crate::id::FnvHasher::default();
    crate::encoded_node::leaf_key(position).hash(&mut hasher);
    for side in actions {
        side.len().hash(&mut hasher);
        for action in side {
            hash_action(action, &mut hasher);
        }
    }
    budget.damage_rolls.hash(&mut hasher);
    budget.max_branches.hash(&mut hasher);
    budget.enumerate_knockouts.hash(&mut hasher);
    hasher.finish()
}

fn rebuilt<'a>(kept: &Kept) -> Result<TurnResult<'a>, String> {
    match kept {
        Kept::Refused(reason) => Err(reason.clone()),
        Kept::Turn { exact, unmodelled, branches } => Ok(TurnResult {
            branches: branches
                .iter()
                .map(|(probability, position)| Branch {
                    probability: *probability,
                    position: position.clone(),
                    log: None,
                })
                .collect(),
            exact: *exact,
            suspended: Vec::new(),
            unmodelled: unmodelled.clone(),
        }),
    }
}

/// The cell's turn: the kept one if this process resolved the same cell before (and still
/// holds it), else `resolve`'s, kept for next time unless it paused.
pub fn turn<'a>(
    position: &Position,
    actions: &[Vec<SlotAction>; 2],
    budget: Budget,
    resolve: impl FnOnce() -> Result<TurnResult<'a>, String>,
) -> Result<TurnResult<'a>, String> {
    let cap = capacity();
    if cap == 0 {
        return resolve();
    }
    let k = key(position, actions, &budget);
    let found = MEMO.with(|memo| {
        let memo = memo.borrow();
        let mut got = None;
        for generation in [&memo.new, &memo.old] {
            if let Some(entry) = generation.get(&k).and_then(|entries| {
                entries
                    .iter()
                    .find(|e| e.budget == budget && e.actions == *actions && e.position == *position)
            }) {
                got = Some(rebuilt(&entry.kept));
                break;
            }
        }
        got
    });
    let mut counts = COUNTS.with(std::cell::Cell::get);
    if let Some(result) = found {
        counts.hits += 1;
        COUNTS.with(|c| c.set(counts));
        return result;
    }
    counts.misses += 1;
    let result = resolve();
    let kept = match &result {
        Err(reason) => Some(Kept::Refused(reason.clone())),
        Ok(turn) if turn.is_suspended() => None,
        Ok(turn) => Some(Kept::Turn {
            exact: turn.exact,
            unmodelled: turn.unmodelled.clone(),
            branches: turn.branches.iter().map(|b| (b.probability, b.position.clone())).collect(),
        }),
    };
    if let Some(kept) = kept {
        MEMO.with(|memo| {
            let mut memo = memo.borrow_mut();
            if memo.count >= cap {
                memo.old = std::mem::take(&mut memo.new);
                memo.count = 0;
                counts.dropped += 1;
            }
            memo.count += 1;
            memo.new.entry(k).or_default().push(Entry {
                position: position.clone(),
                actions: actions.clone(),
                budget,
                kept,
            });
        });
    }
    COUNTS.with(|c| c.set(counts));
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_kept_refusal_and_a_miss_are_counted() {
        let position = Position::from_json(&serde_json::json!({
            "format": "t",
            "sides": [{ "id": "p1", "active": [], "pokemon": [] }, { "id": "p2", "active": [], "pokemon": [] }],
        }));
        let actions = [vec![SlotAction::Pass { slot: 0 }], vec![SlotAction::Pass { slot: 0 }]];
        let before = counts();
        let mut calls = 0;
        for _ in 0..3 {
            let got: Result<TurnResult<'static>, String> =
                turn(&position, &actions, Budget::matrix(), || {
                    calls += 1;
                    Err("no".into())
                });
            assert_eq!(got.err().as_deref(), Some("no"));
        }
        assert_eq!(calls, 1, "resolved once, found twice");
        let after = counts();
        assert_eq!((after.hits - before.hits, after.misses - before.misses), (2, 1));
        let other = [vec![SlotAction::Pass { slot: 1 }], vec![SlotAction::Pass { slot: 0 }]];
        let _ = turn(&position, &other, Budget::matrix(), || -> Result<TurnResult<'static>, String> {
            calls += 1;
            Err("other".into())
        });
        assert_eq!(calls, 2, "another action is another cell");
    }
}
