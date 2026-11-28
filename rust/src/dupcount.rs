//! IKA-462 step 1: a count of the turns resolved more than once, by the input they were given.
//!
//! Only built with the `count-resolves` feature. Every `resolve_turn_logged` call appends one
//! 32-byte record to `<POKEURAOU_RESOLVE_LOG>/<pid>.bin`: the game (the worker's own number),
//! the decision inside that game, the fill inside that decision (0 outside any fill: the
//! turn the game plays), and two hashes -- the position, and the actions with the budget. The reader is a scratch tool, not committed. Nothing here changes an answer.

use crate::position::Position;
use crate::resolve::{Budget, SlotAction};
use std::cell::Cell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::sync::Mutex;

thread_local! {
    static GAME: Cell<u64> = const { Cell::new(0) };
    static DECISION: Cell<u32> = const { Cell::new(0) };
    static FILL: Cell<u32> = const { Cell::new(0) };
}

static OUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());

pub fn set_game(game: u64) {
    GAME.with(|g| g.set(game));
    DECISION.with(|d| d.set(0));
    FILL.with(|f| f.set(0));
}

pub fn next_decision() {
    DECISION.with(|d| d.set(d.get() + 1));
    FILL.with(|f| f.set(0));
}

pub fn end_fills() {
    FILL.with(|f| f.set(0));
}

pub fn next_fill() {
    FILL.with(|f| f.set(f.get() + 1));
}

fn hash_of(text: &str) -> u64 {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

pub fn flush() {
    let Ok(dir) = std::env::var("POKEURAOU_RESOLVE_LOG") else { return };
    let mut buf = OUT.lock().unwrap();
    if buf.is_empty() {
        return;
    }
    let path = format!("{dir}/{}.bin", std::process::id());
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(&buf);
    }
    buf.clear();
}

pub fn record(pos: &Position, actions: &[Vec<SlotAction>; 2], budget: &Budget) {
    if std::env::var_os("POKEURAOU_RESOLVE_LOG").is_none() {
        return;
    }
    let position = format!("{pos:?}");
    let rest = format!("{actions:?}{budget:?}");
    let game = GAME.with(Cell::get);
    let decision = DECISION.with(Cell::get);
    let fill = FILL.with(Cell::get);
    let p = hash_of(&position);
    let a = hash_of(&rest);
    let mut buf = OUT.lock().unwrap();
    buf.extend_from_slice(&game.to_le_bytes());
    buf.extend_from_slice(&decision.to_le_bytes());
    buf.extend_from_slice(&fill.to_le_bytes());
    buf.extend_from_slice(&p.to_le_bytes());
    buf.extend_from_slice(&a.to_le_bytes());
    if buf.len() > 4 << 20 {
        drop(buf);
        flush();
    }
}
