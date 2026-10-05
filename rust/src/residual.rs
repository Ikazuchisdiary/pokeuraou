//! Candidate features for the value net's next inputs (IKA-434, measurement only).
//!
//! `residual-features <regulation.json> <positions.jsonl> <out.bin>` reads one position per
//! line and writes, for each, `2 * FEATURES` little-endian float32: side 0's features, then
//! side 1's, each from that side's view ("mine" against "the foe's"). Nothing here is read by
//! the encoder or the search; the numbers go to a residual table in Python.
//!
//! The pairs use `bind.rs`'s own calculation (`evaluate_pair`: usable moves, spread at 0.75,
//! priority, multi-hit, Focus Sash and Sturdy), so the field-to-field counts here reproduce the
//! revision-4 columns, and the bench pairs extend them:
//!
//! * field pairs: an active Pokemon against an opposing active one (what `bind` already pairs);
//! * bench defender: an active Pokemon against an opposing bench Pokemon at its current HP, as
//!   if it stood in the field now (spread moves at 0.75 when the foe side has two actives);
//! * bench attacker: a bench Pokemon against an opposing active one, as if it stood in the field.
//!
//! "Sure" is the revision-4 meaning: all 16 rolls take the HP; for a bind, also first for
//! certain (priority, or the higher effective Speed with Trick Room reversing it).

use crate::battler::{Battler, FieldState};
use crate::bind::{evaluate_pair, ko_count, move_context, survives_one_hit, usable_moves, PairValue};
use crate::id::Id;
use crate::moveinfo::MoveContext;
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::field_state;
use crate::speed::effective_speed;

pub const FEATURES: usize = 16;

/// The names of the columns, in order (Python reads them from the header line on stderr).
pub const NAMES: [&str; FEATURES] = [
    "field_bind_sure",
    "field_ko_sure",
    "bench_bind_sure",
    "bench_ko_sure",
    "benchatk_ko_sure",
    "benchatk_bind_sure",
    "two_hit_sure",
    "focus_ko_sure",
    "frac_noko",
    "frac_all",
    "outspeed",
    "fastest",
    "good_switch",
    "absorb_best",
    "bench_alive",
    "active_alive",
];

struct Mon {
    battler: Battler,
    moves: Vec<Id>,
    ctx: MoveContext,
    speed: i64,
    holds: bool,
}

/// What one directed pair gives: every quantity is over the attacker's usable damaging moves.
#[derive(Clone, Copy, Default)]
struct Pair {
    /// Some move takes the HP on all 16 rolls.
    ko_sure: bool,
    /// Some move takes it on all 16 rolls and moves first for certain.
    bind_sure: bool,
    /// Some move takes it on at least one roll.
    ko_any: bool,
    /// Two uses of one move take it on all 16 rolls (the same roll each time).
    two_sure: bool,
    /// The best mean roll (times the hit count) over the defender's current HP, at most 1.
    frac: f64,
    /// The best lowest roll (times the hit count), in HP.
    least: i64,
}

fn pair_of(value: &PairValue, defender: &Mon, by_speed: u32) -> Pair {
    let hp = defender.battler.hp;
    let mut out = Pair::default();
    if hp <= 0 {
        return out;
    }
    for eval in &value.moves {
        let rolls = ko_count(eval, hp, defender.holds);
        let first = if eval.first == 1 { by_speed } else { u32::from(eval.first) };
        if rolls == 16 {
            out.ko_sure = true;
            if first == 2 {
                out.bind_sure = true;
            }
        }
        out.ko_any |= rolls > 0;
        let low = eval.rolls.iter().copied().min().unwrap_or(0) * eval.hits;
        let mean = eval.rolls.iter().sum::<i64>() as f64 / eval.rolls.len() as f64 * eval.hits as f64;
        out.two_sure |= low * 2 >= hp;
        out.least = out.least.max(low);
        out.frac = out.frac.max((mean / hp as f64).min(1.0));
    }
    out
}

fn by_speed(mine: i64, theirs: i64, trick_room: bool) -> u32 {
    let (mine, theirs) = if trick_room { (theirs, mine) } else { (mine, theirs) };
    if mine > theirs {
        2
    } else if mine < theirs {
        0
    } else {
        1
    }
}

/// Calculator calls made by each group of pairs, for the cost table.
#[derive(Default)]
pub struct Calls {
    pub field: usize,
    pub bench_defender: usize,
    pub bench_attacker: usize,
}

fn count_calls() -> usize {
    crate::damage::CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn features(reg: &Reg, pos: &Position, m: usize, calls: &mut Calls) -> [[f32; FEATURES]; 2] {
    let mut out = [[0.0f32; FEATURES]; 2];
    let field: FieldState = field_state(pos);
    let trick_room = field.trick_room();
    // actives[s], bench[s]: the live Pokemon of each side, built once.
    let mut actives: [Vec<Mon>; 2] = [Vec::new(), Vec::new()];
    let mut bench: [Vec<Mon>; 2] = [Vec::new(), Vec::new()];
    for (s, side) in pos.sides.iter().enumerate().take(2) {
        let active_indices: Vec<usize> = side.active.iter().take(2).filter_map(|i| *i).collect();
        for (index, mon) in side.pokemon.iter().enumerate().take(m) {
            if mon.fainted || mon.hp <= 0 {
                continue;
            }
            let Ok(mut battler) = Battler::from_pokemon(reg, mon) else { continue };
            battler.hp = mon.hp;
            let speed = effective_speed(&battler, &field, &side.side_conditions);
            let holds = survives_one_hit(&battler);
            let built = Mon { battler, moves: usable_moves(mon), ctx: move_context(pos, s, mon), speed, holds };
            if active_indices.contains(&index) {
                actives[s].push(built);
            } else {
                bench[s].push(built);
            }
        }
    }
    for s in 0..2 {
        out[s][14] = bench[s].len() as f32;
        out[s][15] = actives[s].len() as f32;
    }
    if actives[0].is_empty() || actives[1].is_empty() {
        for s in 0..2 {
            out[s][13] = -1.0;
        }
        return out;
    }
    let eval = |attacker: &Mon, defender: &Mon, defender_side: usize, live_foes: usize, ally_alive: bool| {
        let value = evaluate_pair(
            reg, &field, &attacker.battler, &defender.battler, &attacker.moves, defender_side, live_foes,
            ally_alive, &attacker.ctx, false,
        );
        pair_of(&value, defender, by_speed(attacker.speed, defender.speed, trick_room))
    };
    // field[s][a][d], bench_def[s][a][b], bench_atk[s][b][d], switch_in[s][d][b]: the last is
    // the foe's active d against side s's bench b (bench_def from the other side).
    let mut field_pairs = [[[Pair::default(); 2]; 2]; 2];
    let mut bench_def: [Vec<Vec<Pair>>; 2] = [Vec::new(), Vec::new()];
    let mut bench_atk: [Vec<Vec<Pair>>; 2] = [Vec::new(), Vec::new()];
    for s in 0..2 {
        let foe = 1 - s;
        let live_foes = actives[foe].len();
        let ally_alive = actives[s].len() > 1;
        let before = count_calls();
        for (a, attacker) in actives[s].iter().enumerate() {
            for (d, defender) in actives[foe].iter().enumerate() {
                field_pairs[s][a][d] = eval(attacker, defender, foe, live_foes, ally_alive);
            }
        }
        let after_field = count_calls();
        bench_def[s] = actives[s]
            .iter()
            .map(|attacker| bench[foe].iter().map(|b| eval(attacker, b, foe, live_foes, ally_alive)).collect())
            .collect();
        let after_def = count_calls();
        // A bench Pokemon coming in has a partner whenever its side keeps one active.
        bench_atk[s] = bench[s]
            .iter()
            .map(|attacker| actives[foe].iter().map(|d| eval(attacker, d, foe, live_foes, true)).collect())
            .collect();
        let after_atk = count_calls();
        calls.field += after_field - before;
        calls.bench_defender += after_def - after_field;
        calls.bench_attacker += after_atk - after_def;
    }
    for s in 0..2 {
        let foe = 1 - s;
        let (na, nf) = (actives[s].len(), actives[foe].len());
        let row = &mut out[s];
        for d in 0..nf {
            let pairs: Vec<Pair> = (0..na).map(|a| field_pairs[s][a][d]).collect();
            let one_sure = pairs.iter().any(|p| p.ko_sure);
            row[0] += f32::from(pairs.iter().any(|p| p.bind_sure));
            row[1] += f32::from(one_sure);
            if !one_sure {
                row[6] += f32::from(pairs.iter().any(|p| p.two_sure));
                let together: i64 = pairs.iter().map(|p| p.least).sum();
                row[7] += f32::from(na == 2 && together >= actives[foe][d].battler.hp);
            }
            let best = pairs.iter().map(|p| p.frac).fold(0.0, f64::max);
            if !pairs.iter().any(|p| p.ko_any) {
                row[8] += best as f32;
            }
            row[9] += best as f32;
            for a in 0..na {
                row[10] += by_speed(actives[s][a].speed, actives[foe][d].speed, trick_room) as f32 / 2.0;
            }
        }
        for b in 0..bench[foe].len() {
            row[2] += f32::from((0..na).any(|a| bench_def[s][a][b].bind_sure));
            row[3] += f32::from((0..na).any(|a| bench_def[s][a][b].ko_sure));
        }
        for d in 0..nf {
            row[4] += f32::from((0..bench[s].len()).any(|b| bench_atk[s][b][d].ko_sure));
            row[5] += f32::from((0..bench[s].len()).any(|b| bench_atk[s][b][d].bind_sure));
        }
        // My bench against the foe's actives: the foe's bench_def.
        let mut absorb_best = -1.0f64;
        for b in 0..bench[s].len() {
            let worst = (0..nf).map(|d| bench_def[foe][d][b].frac).fold(0.0, f64::max);
            row[12] += f32::from(worst < 0.5);
            absorb_best = if absorb_best < 0.0 { worst } else { absorb_best.min(worst) };
        }
        row[13] = absorb_best as f32;
    }
    // The fastest of the four (Trick Room reversing): 1 mine, 0.5 a tie across sides.
    let key = |m: &Mon| if trick_room { -m.speed } else { m.speed };
    let top = [0, 1].map(|s| actives[s].iter().map(key).max().unwrap_or(i64::MIN));
    let fastest0 = match top[0].cmp(&top[1]) {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => 0.0,
        std::cmp::Ordering::Equal => 0.5,
    };
    out[0][11] = fastest0;
    out[1][11] = 1.0 - fastest0;
    out
}

/// `residual-features <regulation.json> <positions.jsonl> <out.bin>`.
pub fn main(args: &[String]) {
    let reg = Reg::load(&args[0]).unwrap_or_else(|e| {
        eprintln!("regulation: {e}");
        std::process::exit(1);
    });
    let encoder = crate::encode::Encoder::new(&reg);
    let m = encoder.widths.mons_per_side;
    let text = std::fs::read_to_string(&args[1]).expect("positions");
    let mut bytes: Vec<u8> = Vec::new();
    let mut calls = Calls::default();
    let mut n = 0usize;
    let started = std::time::Instant::now();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).expect("position json");
        let pos = Position::from_json(&value);
        let rows = features(&reg, &pos, m, &mut calls);
        for row in rows.iter() {
            for v in row.iter() {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        n += 1;
    }
    std::fs::write(&args[2], &bytes).expect("write");
    let per = |c: usize| c as f64 / n.max(1) as f64;
    eprintln!(
        "{{\"positions\": {n}, \"names\": {:?}, \"calls_field\": {:.3}, \"calls_bench_defender\": {:.3}, \
         \"calls_bench_attacker\": {:.3}, \"seconds\": {:.3}}}",
        NAMES,
        per(calls.field),
        per(calls.bench_defender),
        per(calls.bench_attacker),
        started.elapsed().as_secs_f64()
    );
}
