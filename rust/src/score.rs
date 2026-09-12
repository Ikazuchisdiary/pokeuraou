//! The damage score that decides which candidates reach the matrix at all.
//!
//! `narrow.score_action`, transcribed. It was 13.9% of a bridged generation run -- the
//! largest thing left in Python once the resolver crossed over -- and nearly all of that
//! is `damage.calculate`, which is already here. What stays in Python is the part that is
//! not arithmetic: enumerating the legal choices (`actions.py`, 500 lines of rules about
//! PP, Disable, Encore, Choice locks and traps) costs 0.2% and is not worth the risk of a
//! second implementation. So the crossing is narrow: candidates go out, one score and its
//! per-target detail comes back, and the covering, ordering and tie-breaks stay where they
//! are.
//!
//! One particle only. The belief layer hands `narrow` Battlers carrying a whole spread
//! with weights, and the calculator vectorises over it; this does not, so a node with
//! beliefs is refused and scored in Python. Generation does not use beliefs, and
//! generation is what this is for.

use crate::battler::{Battler, FieldState};
use crate::damage::{calculate, effective_damage};
use crate::moveinfo::MoveContext;
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::{field_state, SlotAction};

/// What one move contributed: which target it hit, how much, and whether the calculator
/// had everything it needed. Python turns this back into the readable line it prints.
pub struct Contribution {
    pub slot: usize,
    pub target_slot: usize,
    pub is_foe: bool,
    pub signed: f64,
    pub exact: bool,
}

pub struct Scored {
    pub score: f64,
    pub detail: Vec<Contribution>,
}

/// (slot index, is_foe) for every Pokemon a move choice lands on.
fn hit_slots(
    reg: &Reg,
    move_id: &str,
    target: Option<i64>,
    slot: usize,
    live_foes: &[usize],
) -> Vec<(usize, bool)> {
    let Some(mv) = reg.moves.get(move_id) else { return Vec::new() };
    if mv.category == "Status" {
        return Vec::new();
    }
    match mv.target.as_str() {
        "allAdjacentFoes" | "foeSide" => live_foes.iter().map(|i| (*i, true)).collect(),
        "allAdjacent" => {
            let mut out: Vec<(usize, bool)> = live_foes.iter().map(|i| (*i, true)).collect();
            out.push((1 - slot, false));
            out
        }
        _ => match target {
            None => Vec::new(),
            Some(t) if t > 0 => vec![((t - 1) as usize, true)],
            Some(t) => vec![((-t - 1) as usize, false)],
        },
    }
}

/// Mean damage as a fraction of the defender's current HP, and whether it is usable.
///
/// The mean is over all 16 rolls. Every roll is an integer well under 2^53, so a float64
/// sum of sixteen of them is exact whatever order it is taken in and the divide is by a
/// power of two -- which is why this can be a plain loop and still equal numpy's answer
/// bit for bit, where a weighted payoff cannot.
fn expected_fraction(
    reg: &Reg,
    attacker: &Battler,
    defender: &Battler,
    move_id: &str,
    field: &FieldState,
    spread: bool,
    defender_side: usize,
    move_ctx: &MoveContext,
    attacker_is_defender: bool,
) -> (f64, bool) {
    let result = calculate(
        reg,
        attacker,
        defender,
        move_id,
        field,
        defender_side,
        spread,
        false,
        Some(move_ctx),
        None,
        attacker_is_defender,
    );
    if result.immune {
        return (0.0, true);
    }
    let dealt = effective_damage(&result.rolls, defender);
    let total: f64 = dealt.iter().map(|d| *d as f64).sum();
    let mean = total / dealt.len() as f64;
    let value = mean / defender.hp.max(1) as f64;
    (value.min(1.0), result.unmodelled.is_empty())
}

/// The position-derived half of a move's context: everything a *scorer* can know.
///
/// What is missing is only what a turn in progress knows -- whether the target has already
/// been hurt, whether it damaged the attacker first -- and those are false for a scorer
/// looking at a position before anyone has moved.
fn move_context(pos: &Position, side: usize, slot: usize) -> MoveContext {
    let mon = pos.mon_at(side, slot);
    MoveContext {
        weather: pos.field.weather.map(|w| w.as_str().to_string()),
        terrain: pos.field.terrain.map(|t| t.as_str().to_string()),
        side_total_fainted: pos.sides[side].pokemon.iter().filter(|m| m.fainted).count() as i64,
        times_attacked: mon.map(|m| m.times_attacked).unwrap_or(0),
        previous_move_failed: mon.map(|m| m.move_last_turn_failed).unwrap_or(false),
        hit_index: 1,
        ..Default::default()
    }
}

/// Damage each choice is expected to do, minus what it does to one's own partner.
pub fn score_candidates(
    reg: &Reg,
    pos: &Position,
    side: usize,
    candidates: &[Vec<SlotAction>],
) -> Result<Vec<Scored>, String> {
    let field = field_state(pos);
    let foe = 1 - side;
    let live_foes: Vec<usize> = (0..pos.sides[foe].active.len())
        .filter(|i| matches!(pos.mon_at(foe, *i), Some(mon) if !mon.fainted))
        .collect();

    // The same guard the resolver keeps: a position the game could not reach is not a
    // thing to hold two implementations to, and a volatile this port dropped on the way in
    // could be one the calculator reads.
    for one_side in pos.sides.iter() {
        for slot in 0..one_side.active.len() {
            let Some(mon) = one_side.active_pokemon(slot) else { continue };
            if mon.hp < 0 || mon.hp > mon.maxhp {
                return Err(format!("hp {} outside 0..{}", mon.hp, mon.maxhp));
            }
            if mon.fainted != (mon.hp == 0) {
                return Err(format!("fainted={} disagrees with hp={}", mon.fainted, mon.hp));
            }
            if !mon.unmodelled_volatiles.is_empty() {
                return Err("position carries unmodelled volatiles".into());
            }
            if mon.transformed || mon.stats_override.is_some() {
                return Err("transformed Pokemon".into());
            }
        }
    }

    // One Battler per active slot, built once for the whole pool rather than per candidate.
    let mut battlers: Vec<[Option<Battler>; 4]> = vec![[None, None, None, None]; 2];
    for (side_index, one_side) in pos.sides.iter().enumerate() {
        for slot in 0..one_side.active.len().min(4) {
            if let Some(mon) = pos.mon_at(side_index, slot) {
                if !mon.fainted {
                    battlers[side_index][slot] = Some(Battler::from_pokemon(reg, mon)?);
                }
            }
        }
    }
    let contexts: Vec<Option<MoveContext>> = (0..pos.sides[side].active.len().min(4))
        .map(|slot| Some(move_context(pos, side, slot)))
        .collect();

    let mut out: Vec<Scored> = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut total = 0.0f64;
        let mut detail: Vec<Contribution> = Vec::new();
        for (index, slot_action) in candidate.iter().enumerate() {
            let SlotAction::Move { move_id, target, .. } = slot_action else { continue };
            let Some(attacker) = battlers[side].get(index).and_then(|b| b.as_ref()) else {
                continue;
            };
            let hits = hit_slots(reg, move_id.as_str(), *target, index, &live_foes);
            let spread = hits.len() > 1;
            for (target_slot, is_foe) in hits {
                let target_side = if is_foe { foe } else { side };
                let Some(defender) = battlers[target_side].get(target_slot).and_then(|b| b.as_ref())
                else {
                    continue;
                };
                let ctx = contexts
                    .get(index)
                    .and_then(|c| c.as_ref())
                    .ok_or_else(|| format!("no move context for slot {index}"))?;
                let (fraction, exact) = expected_fraction(
                    reg,
                    attacker,
                    defender,
                    move_id.as_str(),
                    &field,
                    spread,
                    target_side,
                    ctx,
                    target_side == side && target_slot == index,
                );
                let signed = if is_foe { fraction } else { -fraction };
                total += signed;
                detail.push(Contribution {
                    slot: index,
                    target_slot,
                    is_foe,
                    signed,
                    exact,
                });
            }
        }
        out.push(Scored { score: total, detail });
    }
    Ok(out)
}
