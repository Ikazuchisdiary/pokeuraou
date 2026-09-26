//! Per-candidate features for the candidate model Q(s, a, b) (IKA-274).
//!
//! The model learns the depth-1 matrix of every legal pair, and its input names each
//! candidate by the Pokemon and moves it involves. What the dex and the calculator already
//! know about a candidate -- how hard it hits each opposing Pokemon, whether it can knock
//! one out, whether it moves first -- is handed over as numbers here rather than left for
//! the model to rediscover from identities. One crossing per position and both sides'
//! pools, so the search can ask it at every node without a Python loop per candidate.
//!
//! Everything is read off `damage::calculate`, the calculator `score.rs` (the ranking's
//! damage score) already uses; nothing here changes what any other command answers.
//!
//! Per candidate, `WIDTH` numbers, side-relative ("foe k" is the opposing active position
//! k, as the choice's `target k + 1` names it):
//!
//! ```text
//!   per slot t (2), per foe position k (2):  eff_log2, fraction, ko, first, hits   (20)
//!   per slot t:                               ally_fraction                        (2)
//!   per slot t, per foe position k:           switch_threat, switch_reach           (8)
//!   per foe position k:                       combined_ko, both_hit                 (4)
//! ```
//!
//! * `eff_log2`: log2 of the move's type effectiveness on that foe (-3 for an immunity),
//!   0 for a status move or a slot that does not move
//! * `fraction`: the mean of the 16 rolls as a share of the foe's current HP (capped at 1),
//!   non-critical -- `score.rs`'s number
//! * `ko`: the chance the hit alone takes the foe's current HP: the 16 rolls, critical hits
//!   at the port's own `damage::crit_probability` mixed in. Accuracy is not in it (the
//!   model sees the move's accuracy in the dex table)
//! * `first`: 1 if this move acts before that foe's Pokemon would with a priority-0 move,
//!   0 if after, 0.5 on a speed tie; priority, Trick Room and the speed modifiers included.
//!   The foe's actual move is not known here, so its priority is taken as 0 (approximate)
//! * `hits`: 1 if the move lands on that foe (a spread move on every live foe)
//! * `ally_fraction`: the share of the partner's HP a spread move takes from it
//! * `switch_threat` / `switch_reach`: for a switch, the largest fraction of the incoming
//!   Pokemon's HP any of foe k's four moves takes, and the largest fraction of foe k's HP
//!   any of the incoming Pokemon's moves takes (both non-critical means)
//! * `combined_ko`: when both slots hit foe k, the chance the two hits together take its
//!   current HP -- the two roll distributions (critical hits mixed in) convolved. The order
//!   of the hits and what it changes (Focus Sash, Multiscale, Disguise, one of them fainting
//!   first) is not modelled: an approximation, and `both_hit` says where it applies
//!
//! A mega declared with a move is scored in the Pokemon's current form (as `score.rs` does).

use std::collections::HashMap;

use crate::battler::{Battler, FieldState, N_ROLLS};
use crate::damage::{calculate, crit_probability, effective_damage};
use crate::moveinfo::MoveContext;
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::{field_state, SlotAction};
use crate::speed::{effective_speed, move_priority};

pub const WIDTH: usize = 34;
const PER_SLOT: usize = 10;

/// One damaging hit on one defender: the rolls as dealt, with and without a critical hit.
#[derive(Clone)]
struct Hit {
    eff_log2: f64,
    fraction: f64,
    ko: f64,
    first: f64,
    crit: f64,
    normal: [i64; N_ROLLS],
    critical: [i64; N_ROLLS],
    hp: i64,
}

fn move_context(pos: &Position, side: usize, slot: usize) -> MoveContext {
    let mon = pos.mon_at(side, slot);
    MoveContext {
        weather: pos.field.weather,
        terrain: pos.field.terrain,
        side_total_fainted: pos.sides[side].pokemon.iter().filter(|m| m.fainted).count() as i64,
        times_attacked: mon.map(|m| m.times_attacked).unwrap_or(0),
        previous_move_failed: mon.map(|m| m.move_last_turn_failed).unwrap_or(false),
        hit_index: 1,
        ..Default::default()
    }
}

#[allow(clippy::too_many_arguments)]
fn hit(
    reg: &Reg,
    attacker: &Battler,
    defender: &Battler,
    move_id: &str,
    field: &FieldState,
    defender_side: usize,
    spread: bool,
    ctx: &MoveContext,
    first: f64,
) -> Hit {
    let normal = calculate(
        reg, attacker, defender, move_id, field, defender_side, spread, false, Some(ctx), None,
        false,
    );
    let eff_log2 = if normal.immune {
        -3.0
    } else if normal.effectiveness > 0.0 {
        normal.effectiveness.log2()
    } else {
        -3.0
    };
    let critical_result = calculate(
        reg, attacker, defender, move_id, field, defender_side, spread, true, Some(ctx), None,
        false,
    );
    let (normal_rolls, critical_rolls) = if normal.immune {
        ([0i64; N_ROLLS], [0i64; N_ROLLS])
    } else {
        (
            effective_damage(&normal.rolls, defender),
            effective_damage(&critical_result.rolls, defender),
        )
    };
    let hp = defender.hp.max(1);
    let mean: f64 = normal_rolls.iter().map(|d| *d as f64).sum::<f64>() / N_ROLLS as f64;
    let c = crit_probability(reg, attacker, defender, move_id);
    let share = |rolls: &[i64; N_ROLLS]| {
        rolls.iter().filter(|d| **d >= hp).count() as f64 / N_ROLLS as f64
    };
    let ko = (1.0 - c) * share(&normal_rolls) + c * share(&critical_rolls);
    Hit {
        eff_log2,
        fraction: (mean / hp as f64).min(1.0),
        ko,
        first,
        crit: c,
        // Uncapped, for the convolution: a second hit lands on a Pokemon no longer at full.
        normal: if normal.immune { [0; N_ROLLS] } else { normal.rolls },
        critical: if normal.immune { [0; N_ROLLS] } else { critical_result.rolls },
        hp,
    }
}

/// P(first + second >= hp), each hit's rolls uniform with its critical hits mixed in.
fn combined_ko(a: &Hit, b: &Hit) -> f64 {
    let hp = a.hp;
    let mut total = 0.0;
    for (wa, rolls_a) in [(1.0 - a.crit, &a.normal), (a.crit, &a.critical)] {
        if wa == 0.0 {
            continue;
        }
        for (wb, rolls_b) in [(1.0 - b.crit, &b.normal), (b.crit, &b.critical)] {
            if wb == 0.0 {
                continue;
            }
            let mut count = 0usize;
            for x in rolls_a.iter() {
                for y in rolls_b.iter() {
                    if x + y >= hp {
                        count += 1;
                    }
                }
            }
            total += wa * wb * count as f64 / (N_ROLLS * N_ROLLS) as f64;
        }
    }
    total
}

fn first_against(
    reg: &Reg,
    pos: &Position,
    field: &FieldState,
    side: usize,
    attacker: &Battler,
    move_id: &str,
    foe: &Battler,
) -> f64 {
    let priority = move_priority(reg, move_id, attacker, field);
    if priority > 0 {
        return 1.0;
    }
    if priority < 0 {
        return 0.0;
    }
    let mine = effective_speed(attacker, field, &pos.sides[side].side_conditions);
    let theirs = effective_speed(foe, field, &pos.sides[1 - side].side_conditions);
    let (mine, theirs) = if field.trick_room() { (theirs, mine) } else { (mine, theirs) };
    if mine > theirs {
        1.0
    } else if mine < theirs {
        0.0
    } else {
        0.5
    }
}

/// Every candidate's features for one side, `WIDTH` numbers each.
pub fn side_features(
    reg: &Reg,
    pos: &Position,
    side: usize,
    candidates: &[Vec<SlotAction>],
) -> Result<Vec<[f64; WIDTH]>, String> {
    crate::score::check_scorable(pos)?;
    let field = field_state(pos);
    let foe_side = 1 - side;
    let actives = pos.sides[side].active.len().min(2);
    let foes = pos.sides[foe_side].active.len().min(2);
    let mut battlers: Vec<[Option<Battler>; 2]> = vec![[None, None], [None, None]];
    for (s, one_side) in pos.sides.iter().enumerate() {
        for slot in 0..one_side.active.len().min(2) {
            if let Some(mon) = pos.mon_at(s, slot) {
                if !mon.fainted {
                    battlers[s][slot] = Some(Battler::from_pokemon(reg, mon)?);
                }
            }
        }
    }
    let live_foes: Vec<usize> = (0..foes).filter(|k| battlers[foe_side][*k].is_some()).collect();
    let contexts: Vec<MoveContext> = (0..actives).map(|s| move_context(pos, side, s)).collect();

    // A slot's move choice recurs across many candidates; each is priced once.
    let mut moves: HashMap<(usize, String, Option<i64>), ([Option<Hit>; 2], f64)> = HashMap::new();
    let mut switches: HashMap<usize, [f64; 4]> = HashMap::new();

    let mut out = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut row = [0.0f64; WIDTH];
        let mut hits_on: [[Option<Hit>; 2]; 2] = [[None, None], [None, None]];
        for (t, slot_action) in candidate.iter().enumerate().take(2) {
            match slot_action {
                SlotAction::Move { move_id, target, .. } => {
                    let Some(attacker) = battlers[side][t].as_ref() else { continue };
                    let key = (t, move_id.as_str().to_string(), *target);
                    if !moves.contains_key(&key) {
                        moves.insert(
                            key.clone(),
                            price_move(
                                reg, pos, &field, side, t, attacker, move_id.as_str(), *target,
                                &battlers, &live_foes, &contexts[t],
                            ),
                        );
                    }
                    let (per_foe, ally) = &moves[&key];
                    for k in 0..2 {
                        if let Some(h) = &per_foe[k] {
                            let at = t * PER_SLOT + k * 5;
                            row[at] = h.eff_log2;
                            row[at + 1] = h.fraction;
                            row[at + 2] = h.ko;
                            row[at + 3] = h.first;
                            row[at + 4] = 1.0;
                            hits_on[t][k] = Some(h.clone());
                        }
                    }
                    row[20 + t] = *ally;
                }
                SlotAction::Switch { party_index, .. } => {
                    let index = *party_index;
                    if !switches.contains_key(&index) {
                        switches.insert(
                            index,
                            price_switch(reg, pos, &field, side, index, &battlers, &contexts)?,
                        );
                    }
                    let got = switches[&index];
                    row[22 + t * 4..22 + t * 4 + 4].copy_from_slice(&got);
                }
                SlotAction::Pass { .. } => {}
            }
        }
        for k in 0..2 {
            if let (Some(a), Some(b)) = (&hits_on[0][k], &hits_on[1][k]) {
                row[30 + k * 2] = combined_ko(a, b);
                row[30 + k * 2 + 1] = 1.0;
            }
        }
        out.push(row);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn price_move(
    reg: &Reg,
    pos: &Position,
    field: &FieldState,
    side: usize,
    slot: usize,
    attacker: &Battler,
    move_id: &str,
    target: Option<i64>,
    battlers: &[[Option<Battler>; 2]],
    live_foes: &[usize],
    ctx: &MoveContext,
) -> ([Option<Hit>; 2], f64) {
    let mut per_foe: [Option<Hit>; 2] = [None, None];
    let Some(mv) = reg.moves.get(move_id) else { return (per_foe, 0.0) };
    if mv.category == "Status" {
        return (per_foe, 0.0);
    }
    let foe_side = 1 - side;
    let (foe_targets, ally_hit): (Vec<usize>, bool) = match mv.target.as_str() {
        "allAdjacentFoes" | "foeSide" => (live_foes.to_vec(), false),
        "allAdjacent" => (live_foes.to_vec(), true),
        _ => match target {
            Some(t) if t > 0 => (vec![(t - 1) as usize], false),
            _ => (Vec::new(), false),
        },
    };
    let ally = if ally_hit { battlers[side].get(1 - slot).and_then(|b| b.as_ref()) } else { None };
    let spread = foe_targets.len() + usize::from(ally.is_some()) > 1;
    for k in foe_targets {
        let Some(defender) = battlers[foe_side].get(k).and_then(|b| b.as_ref()) else { continue };
        let first = first_against(reg, pos, field, side, attacker, move_id, defender);
        per_foe[k] = Some(hit(reg, attacker, defender, move_id, field, foe_side, spread, ctx, first));
    }
    let ally_fraction = match ally {
        Some(partner) => hit(reg, attacker, partner, move_id, field, side, spread, ctx, 0.0).fraction,
        None => 0.0,
    };
    (per_foe, ally_fraction)
}

fn price_switch(
    reg: &Reg,
    pos: &Position,
    field: &FieldState,
    side: usize,
    party_index: usize,
    battlers: &[[Option<Battler>; 2]],
    contexts: &[MoveContext],
) -> Result<[f64; 4], String> {
    let mut out = [0.0f64; 4];
    let Some(mon) = pos.sides[side].pokemon.iter().find(|m| m.slot + 1 == party_index) else {
        return Ok(out);
    };
    let incoming = Battler::from_pokemon(reg, mon)?;
    let foe_side = 1 - side;
    let blank = MoveContext::default();
    for k in 0..2 {
        let Some(foe) = battlers[foe_side].get(k).and_then(|b| b.as_ref()) else { continue };
        let Some(foe_mon) = pos.mon_at(foe_side, k) else { continue };
        let foe_ctx = move_context(pos, foe_side, k);
        let mut threat = 0.0f64;
        for slot in foe_mon.moves.iter() {
            let id = slot.id.as_str();
            if reg.moves.get(id).map(|m| m.category == "Status").unwrap_or(true) {
                continue;
            }
            threat = threat.max(hit(reg, foe, &incoming, id, field, side, false, &foe_ctx, 0.0).fraction);
        }
        let mut reach = 0.0f64;
        for slot in mon.moves.iter() {
            let id = slot.id.as_str();
            if reg.moves.get(id).map(|m| m.category == "Status").unwrap_or(true) {
                continue;
            }
            let ctx = contexts.first().unwrap_or(&blank);
            reach = reach.max(hit(reg, &incoming, foe, id, field, foe_side, false, ctx, 0.0).fraction);
        }
        out[k * 2] = threat;
        out[k * 2 + 1] = reach;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(value: i64, crit: f64, critical: i64, hp: i64) -> Hit {
        Hit {
            eff_log2: 0.0,
            fraction: 0.0,
            ko: 0.0,
            first: 0.0,
            crit,
            normal: [value; N_ROLLS],
            critical: [critical; N_ROLLS],
            hp,
        }
    }

    #[test]
    fn two_hits_that_each_fall_short_knock_out_together() {
        let a = flat(60, 0.0, 90, 100);
        let b = flat(50, 0.0, 75, 100);
        assert_eq!(combined_ko(&a, &b), 1.0);
        let c = flat(30, 0.0, 45, 100);
        assert_eq!(combined_ko(&a, &c), 0.0);
    }

    #[test]
    fn a_critical_hit_counts_at_its_chance() {
        // 60 + 30 falls short; a critical first hit (90) makes it.
        let a = flat(60, 0.25, 90, 100);
        let c = flat(30, 0.0, 45, 100);
        assert!((combined_ko(&a, &c) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn rolls_split_the_chance() {
        let mut a = flat(0, 0.0, 0, 100);
        for (i, r) in a.normal.iter_mut().enumerate() {
            *r = 40 + i as i64; // 40..55
        }
        let b = flat(50, 0.0, 50, 100);
        // 40 + i + 50 >= 100 when i >= 10: six of sixteen rolls.
        assert!((combined_ko(&a, &b) - 6.0 / 16.0).abs() < 1e-12);
    }
}
