//! The parameter-free payoffs, transcribed from `src/pokeuraou/payoff.py`.
//!
//! Only these two. A learned value function cannot cross this boundary as a position --
//! the leaves of one node are tens of megabytes of JSON -- so a node scored by a value net
//! stays in Python until the encoder is ported too. `hp-share` is what generation runs
//! today, because this regulation has no trained model yet.

use crate::position::Position;

/// 1.0 or 0.0 once the battle is over, so a win is never merely a large payoff.
fn decided(pos: &Position) -> Option<f64> {
    if !pos.ended {
        return None;
    }
    match &pos.winner {
        None => Some(0.5),
        Some(winner) => Some(if *winner == pos.sides[0].id { 1.0 } else { 0.0 }),
    }
}

pub fn hp_share(pos: &Position) -> f64 {
    if let Some(value) = decided(pos) {
        return value;
    }
    let mut remaining = [0.0f64; 2];
    for (index, side) in pos.sides.iter().enumerate() {
        let live: i64 = side.pokemon.iter().filter(|m| !m.fainted).map(|m| m.hp).sum();
        let total: i64 = side.pokemon.iter().map(|m| m.maxhp).sum();
        remaining[index] = if total <= 0 { 0.0 } else { live as f64 / total as f64 };
    }
    let denominator = remaining[0] + remaining[1];
    if denominator <= 0.0 {
        return 0.5;
    }
    remaining[0] / denominator
}

pub fn faint_share(pos: &Position) -> f64 {
    if let Some(value) = decided(pos) {
        return value;
    }
    let standing: Vec<f64> = pos
        .sides
        .iter()
        .map(|side| side.pokemon.iter().filter(|m| !m.fainted).count() as f64)
        .collect();
    let denominator = standing[0] + standing[1];
    if denominator <= 0.0 {
        return 0.5;
    }
    standing[0] / denominator
}

pub fn by_name(name: &str) -> Option<fn(&Position) -> f64> {
    match name {
        "hp-share" => Some(hp_share),
        "faints" => Some(faint_share),
        _ => None,
    }
}
