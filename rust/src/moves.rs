//! Move execution and the end-of-turn phase, transcribed from `src/pokeuraou/resolve.py`.
//!
//! Split from `resolve.rs` only for length; the state it works on is `resolve::Turn`.
//! Every path that meets something this port does not model returns `Err(reason)` so the
//! caller can fall back to Python rather than take a wrong answer.

use crate::battler::Battler;
use crate::damage::{calculate, crit_probability};
use crate::effects::resist_berry;
use crate::id::Id;
use crate::moveinfo::MoveContext;
use crate::position::{Effect, Position};
use crate::reg::{Move, Reg};
use crate::resolve::{
    check_white_herb, grounded, stratified_rolls, Budget, Outcome, Slot, Turn,
    CONFUSION_SELF_HIT_CHANCE, FREEZE_COUNTER, FULL_PARALYSIS_CHANCE, THAW_CHANCE,
    TWO_TURN_MOVES,
};
use crate::speed::QueuedAction;
use serde_json::{json, Value};

const RECHARGE: &str = "recharge";

/// Protect-family volatiles, and what each blocks.
const PROTECT_VOLATILES: [(&str, &str); 9] = [
    ("protect", "all"),
    ("detect", "all"),
    ("banefulbunker", "all"),
    ("burningbulwark", "all"),
    ("spikyshield", "all"),
    ("kingsshield", "damaging"),
    ("obstruct", "damaging"),
    ("silktrap", "damaging"),
    ("maxguard", "all"),
];

const FIRST_TURN_OUT_MOVES: [&str; 3] = ["fakeout", "firstimpression", "matblock"];
const STALL_BUMPING_MOVES: [&str; 2] = ["wideguard", "quickguard"];
const MAX_BRANCHED_SECONDARIES: usize = 2;

const CHARGE_SPA: &[(&str, i64)] = &[("spa", 1)];
const CHARGE_GEOMANCY: &[(&str, i64)] = &[("spa", 2), ("spd", 2), ("spe", 2)];

/// Stat changes a charge turn brings with it. Electro Shot and Meteor Beam raise Special
/// Attack while winding up, which is most of the reason to use them.
fn charge_turn_boosts(move_id: &str) -> Option<&'static [(&'static str, i64)]> {
    match move_id {
        "electroshot" | "meteorbeam" => Some(CHARGE_SPA),
        "geomancy" => Some(CHARGE_GEOMANCY),
        _ => None,
    }
}

fn is(value: Option<Id>, name: &str) -> bool {
    matches!(value, Some(v) if v.as_str() == name)
}

// ---------------------------------------------------------------------------
// One move
// ---------------------------------------------------------------------------

pub(crate) fn do_move<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    budget: Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let move_id = action.move_id.ok_or("a move action with no move")?;
    if move_id.as_str() == RECHARGE {
        if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
            mon.volatiles.retain(|v| v.id.as_str() != "mustrecharge");
        }
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }
    let mv = reg.moves.get(move_id.as_str()).ok_or("move not in the regulation")?;

    let started = crate::resolve::phase_start();
    let checks = can_act(&mut turn, action, &budget)?;
    crate::resolve::phase_end(7, started);
    let mut outcomes: Vec<Outcome<'a>> = Vec::new();
    // The last check cannot hand `turn` over instead of copying it, however tempting: the
    // fallback below needs it if every branch turns out to weigh nothing, and a move that
    // produced no outcome at all is a thing Python answers rather than refuses.
    for (probability, blocked) in checks.iter() {
        if *probability <= 0.0 {
            continue;
        }
        let mut state = turn.clone();
        match blocked {
            Some(reason) => {
                if reason.as_str() == "flinch" {
                    if let Some(mon) = state.mon_at_mut(action.side, action.slot) {
                        mon.volatiles.retain(|v| v.id.as_str() != "flinch");
                    }
                }
                if reason.as_str() == "confusion" {
                    let amount = confusion_damage(&state, action.side, action.slot)?;
                    state.deal_damage(action.side, action.slot, amount, false)?;
                }
                state.move_failed[action.side][action.slot] = true;
                outcomes.push((*probability, state));
            }
            None => {
                let started = crate::resolve::phase_start();
                let produced = use_move(reg, state, action, mv, budget)?;
                crate::resolve::phase_end(8, started);
                for (weight, sub_state) in produced {
                    outcomes.push((probability * weight, sub_state));
                }
            }
        }
    }
    if outcomes.is_empty() {
        outcomes.push((1.0, turn));
    }
    Ok(outcomes)
}

fn confusion_damage(turn: &Turn, side: usize, slot: usize) -> Result<i64, String> {
    let Some(mon) = turn.battler_at(side, slot)? else { return Ok(0) };
    let attack = mon.stat("atk", false);
    let defence = mon.stat("def", false).max(1);
    let level_term = (2.0 * mon.level as f64 / 5.0 + 2.0).trunc() as i64;
    let inner = (level_term * 40 * attack) as f64;
    let base = ((inner.trunc() / defence as f64).trunc() / 50.0).trunc() as i64;
    Ok(base + 2)
}

/// (probability, reason it could not act) for the pre-move checks.
fn can_act(
    turn: &mut Turn,
    action: &QueuedAction,
    budget: &Budget,
) -> Result<Vec<(f64, Option<String>)>, String> {
    let (fainted, has_flinch, status, has_confusion) = {
        let Some(mon) = turn.mon_at(action.side, action.slot) else {
            return Ok(vec![(1.0, Some("fainted".into()))]);
        };
        (
            mon.fainted,
            mon.has_volatile("flinch"),
            mon.status,
            mon.has_volatile("confusion"),
        )
    };
    if fainted {
        return Ok(vec![(1.0, Some("fainted".into()))]);
    }
    if has_flinch {
        return Ok(vec![(1.0, Some("flinch".into()))]);
    }

    if is(status, "slp") {
        let woke = {
            let mon = turn.mon_at_mut(action.side, action.slot).unwrap();
            let mut counter = mon.status_counter.unwrap_or(0) - 1;
            if mon.ability == "earlybird" {
                counter -= 1;
            }
            mon.status_counter = Some(counter);
            if counter <= 0 {
                mon.status = None;
                mon.status_counter = None;
                true
            } else {
                false
            }
        };
        return Ok(if woke {
            vec![(1.0, None)]
        } else {
            vec![(1.0, Some("slp".into()))]
        });
    }

    if is(status, "frz") {
        let thawed = {
            let mon = turn.mon_at_mut(action.side, action.slot).unwrap();
            let counter = mon.status_counter.unwrap_or(FREEZE_COUNTER) - 1;
            mon.status_counter = Some(counter);
            if counter <= 0 {
                mon.status = None;
                mon.status_counter = None;
                true
            } else {
                false
            }
        };
        if thawed {
            return Ok(vec![(1.0, None)]);
        }
        if !budget.enumerate_status_checks {
            return Ok(vec![(1.0, Some("frz".into()))]);
        }
        // The two outcomes differ in more than "did it act": one of them is no longer
        // frozen next turn, and this returns weights rather than states, so it cannot
        // express that. The weights are right and the cured state is reported as missing.
        turn.report("thaw roll (1 in 4; the cured state is not branched)");
        return Ok(vec![(THAW_CHANCE, None), (1.0 - THAW_CHANCE, Some("frz".into()))]);
    }

    // Priority-blocking abilities and Psychic Terrain.
    let move_id = action.move_id.ok_or("a move action with no move")?;
    let aimed_at_foes = turn
        .reg
        .moves
        .get(move_id.as_str())
        .map(|mv| {
            !matches!(
                mv.target.as_str(),
                "self"
                    | "allySide"
                    | "allyTeam"
                    | "allies"
                    | "adjacentAlly"
                    | "adjacentAllyOrSelf"
                    | "all"
            )
        })
        .unwrap_or(false);
    if action.priority > 0 && aimed_at_foes {
        let foe_side = 1 - action.side;
        for slot in 0..turn.pos.sides[foe_side].active.len() {
            if let Some(foe) = turn.mon_at(foe_side, slot) {
                if !foe.fainted
                    && matches!(
                        foe.ability.as_str(),
                        "armortail" | "queenlymajesty" | "dazzling"
                    )
                {
                    return Ok(vec![(1.0, Some(format!("ability: {}", foe.ability)))]);
                }
            }
        }
        if is(turn.pos.field.terrain, "psychicterrain") {
            for slot in 0..turn.pos.sides[foe_side].active.len() {
                let blocked = match turn.mon_at(foe_side, slot) {
                    None => false,
                    Some(foe) => !foe.fainted && grounded(turn, foe),
                };
                if blocked {
                    return Ok(vec![(1.0, Some("psychicterrain".into()))]);
                }
            }
        }
    }

    let mut outcomes: Vec<(f64, Option<String>)> = vec![(1.0, None)];
    if is(status, "par") && budget.enumerate_status_checks {
        outcomes = vec![
            (1.0 - FULL_PARALYSIS_CHANCE, None),
            (FULL_PARALYSIS_CHANCE, Some("par".into())),
        ];
    }
    if has_confusion && budget.enumerate_status_checks {
        let mut expanded = Vec::new();
        for (weight, reason) in outcomes {
            match reason {
                Some(reason) => expanded.push((weight, Some(reason))),
                None => {
                    expanded.push((weight * (1.0 - CONFUSION_SELF_HIT_CHANCE), None));
                    expanded
                        .push((weight * CONFUSION_SELF_HIT_CHANCE, Some("confusion".into())));
                }
            }
        }
        outcomes = expanded;
    }
    Ok(outcomes)
}

fn use_move<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    budget: Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let move_id = action.move_id.unwrap();
    spend_pp(&mut turn, action);
    turn.current_actor = Some((action.side, action.slot));

    {
        let choice_items = &reg.choice_items;
        let locks = match turn.mon_at(action.side, action.slot) {
            None => false,
            Some(mon) => mon
                .item
                .map(|i| choice_items.contains(i.as_str()))
                .unwrap_or(false),
        };
        if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
            mon.last_move = Some(move_id);
        }
        if locks {
            turn.add_volatile(action.side, action.slot, "choicelock", None);
            if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
                if let Some(locked) = mon.volatile_mut("choicelock") {
                    locked.move_id = Some(move_id);
                }
            }
        }
    }

    // `runMove` increments this before `onTry` runs.
    let first_turn_failure = {
        let Some(mon) = turn.mon_at_mut(action.side, action.slot) else {
            return Ok(vec![(1.0, turn)]);
        };
        mon.active_move_actions += 1;
        FIRST_TURN_OUT_MOVES.contains(&move_id.as_str()) && mon.active_move_actions > 1
    };
    if first_turn_failure {
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    if move_id.as_str() == "suckerpunch" {
        let candidates = resolve_targets(reg, &mut turn, action, mv)?;
        let pending = candidates.iter().any(|slot| {
            slot.0 != action.side
                && turn.attacks[slot.0][slot.1]
                && !turn.acted[slot.0][slot.1]
        });
        if !pending {
            turn.move_failed[action.side][action.slot] = true;
            return Ok(vec![(1.0, turn)]);
        }
    }

    // A move that spends a turn winding up. Python models exactly this much of them and
    // no more -- there is no semi-invulnerability anywhere in the engine -- so Fly and Dig
    // have the same shape here as Solar Beam, and having the same shape is the whole
    // requirement: the oracle for this port is Python, not Showdown.
    if let Some((_, skip_weather)) =
        TWO_TURN_MOVES.iter().find(|(id, _)| *id == move_id.as_str())
    {
        let charged = turn
            .mon_at(action.side, action.slot)
            .map(|mon| mon.has_volatile("twoturnmove"))
            .unwrap_or(false);
        if charged {
            // The charge already happened last turn: drop the marker and attack. Showdown's
            // `onTryMove` starts with `if (attacker.removeVolatile(move.id)) return;`, so
            // there is no second boost.
            if let Some(mon) = turn.mon_at_mut(action.side, action.slot) {
                mon.volatiles.retain(|v| v.id.as_str() != "twoturnmove");
            }
        } else {
            // The charge-turn boost applies whether or not the charge is skipped: Showdown
            // boosts first and only then checks the weather. Electro Shot in rain both
            // raises Special Attack and attacks in the same turn, so leaving the boost out
            // understates its damage by a whole stage.
            if let Some(boosts) = charge_turn_boosts(move_id.as_str()) {
                turn.apply_boosts(action.side, action.slot, boosts, false);
            }
            let skipped = turn
                .pos
                .field
                .weather
                .as_ref()
                .map(|w| skip_weather.contains(&w.as_str()))
                .unwrap_or(false);
            if !skipped {
                turn.add_volatile(action.side, action.slot, "twoturnmove", None);
                return Ok(vec![(1.0, turn)]);
            }
        }
    }

    let targets = resolve_targets(reg, &mut turn, action, mv)?;
    let no_target_needed = matches!(
        mv.target.as_str(),
        "self" | "allySide" | "allyTeam" | "all" | "foeSide"
    );
    if targets.is_empty() && !no_target_needed {
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    if mv.raw_bool("breaksProtect") {
        return Err("breaksProtect move".into());
    }

    if mv.category == "Status" {
        return do_status_move(reg, turn, action, mv, &targets, budget);
    }

    let spread = move_hits_multiple(reg, move_id.as_str(), targets.len());
    turn.move_damage_total = 0;
    turn.move_connected = false;
    let mut branches: Vec<Outcome<'a>> = vec![(1.0, turn)];
    for target in &targets {
        let mut expanded: Vec<Outcome<'a>> = Vec::new();
        for (weight, state) in branches.into_iter() {
            let started = crate::resolve::phase_start();
            let hit = hit_target(reg, state, action, mv, *target, spread, budget)?;
            crate::resolve::phase_end(9, started);
            for (inner_weight, inner_state) in hit {
                expanded.push((weight * inner_weight, inner_state));
            }
        }
        branches = expanded;
    }
    for (_weight, state) in branches.iter_mut() {
        after_move(state, action, mv)?;
    }
    Ok(branches)
}

fn spend_pp(turn: &mut Turn, action: &QueuedAction) {
    let Some(move_id) = action.move_id else { return };
    let Some(mon) = turn.mon_at_mut(action.side, action.slot) else { return };
    if let Some(slot) = mon.moves.get_mut(move_id) {
        if slot.pp > 0 {
            slot.pp -= 1;
        }
        slot.used = true;
    }
}

fn move_hits_multiple(reg: &Reg, move_id: &str, live_targets: usize) -> bool {
    match reg.moves.get(move_id).map(|m| m.target.as_str()) {
        Some("allAdjacentFoes") => live_targets > 1,
        Some("allAdjacent") => live_targets >= 1,
        _ => false,
    }
}

fn resolve_targets(
    reg: &Reg,
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
) -> Result<Vec<Slot>, String> {
    let me = (action.side, action.slot);
    let foe_side = 1 - action.side;
    let slots = 0..turn.pos.sides[foe_side].active.len();

    let live = |turn: &Turn, side: usize, slot: usize| -> bool {
        matches!(turn.mon_at(side, slot), Some(mon) if !mon.fainted)
    };

    match mv.target.as_str() {
        "self" | "allySide" | "allyTeam" | "all" | "foeSide" => return Ok(vec![me]),
        "allAdjacentFoes" => {
            return Ok(slots.filter(|s| live(turn, foe_side, *s)).map(|s| (foe_side, s)).collect())
        }
        "allAdjacent" => {
            let mut out: Vec<Slot> =
                slots.filter(|s| live(turn, foe_side, *s)).map(|s| (foe_side, s)).collect();
            let ally = 1 - action.slot;
            if live(turn, action.side, ally) {
                out.push((action.side, ally));
            }
            return Ok(out);
        }
        "allies" => {
            return Ok((0..turn.pos.sides[action.side].active.len())
                .filter(|s| live(turn, action.side, *s))
                .map(|s| (action.side, s))
                .collect())
        }
        "adjacentAlly" | "adjacentAllyOrSelf" => {
            let Some(target) = action.target else { return Ok(vec![me]) };
            let slot = (-target - 1) as usize;
            return Ok(if live(turn, action.side, slot) {
                vec![(action.side, slot)]
            } else {
                Vec::new()
            });
        }
        "randomNormal" => {
            return Ok(slots
                .filter(|s| live(turn, foe_side, *s))
                .map(|s| (foe_side, s))
                .take(1)
                .collect())
        }
        _ => {}
    }

    let mut chosen = match action.target {
        None => {
            match (0..turn.pos.sides[foe_side].active.len())
                .find(|s| live(turn, foe_side, *s))
            {
                None => return Ok(Vec::new()),
                Some(slot) => (foe_side, slot),
            }
        }
        Some(target) if target > 0 => (foe_side, (target - 1) as usize),
        Some(target) => (action.side, (-target - 1) as usize),
    };

    if !live(turn, chosen.0, chosen.1) && chosen.0 != action.side {
        if let Some(slot) = (0..turn.pos.sides[foe_side].active.len())
            .find(|s| live(turn, foe_side, *s))
        {
            chosen = (foe_side, slot);
        }
    }

    if let Some(redirected) = redirection_target(turn, action, mv, chosen) {
        chosen = redirected;
    }
    let _ = reg;
    Ok(if live(turn, chosen.0, chosen.1) { vec![chosen] } else { Vec::new() })
}

fn redirection_target(
    turn: &Turn,
    action: &QueuedAction,
    mv: &Move,
    chosen: Slot,
) -> Option<Slot> {
    if chosen.0 == action.side {
        return None;
    }
    let foe_side = chosen.0;
    let powder_immune = match turn.mon_at(action.side, action.slot) {
        None => false,
        Some(user) => {
            turn.types_of(user).contains("Grass")
                || is(user.item, "safetygoggles")
                || user.ability == "overcoat"
        }
    };
    for slot in 0..turn.pos.sides[foe_side].active.len() {
        let Some(mon) = turn.mon_at(foe_side, slot) else { continue };
        if mon.fainted {
            continue;
        }
        let drawing: Vec<&str> = ["followme", "ragepowder", "spotlight"]
            .into_iter()
            .filter(|v| mon.has_volatile(v))
            .collect();
        if drawing.is_empty() {
            continue;
        }
        if drawing == ["ragepowder"] && powder_immune {
            continue;
        }
        return Some((foe_side, slot));
    }
    for slot in 0..turn.pos.sides[foe_side].active.len() {
        let Some(mon) = turn.mon_at(foe_side, slot) else { continue };
        if mon.fainted || (foe_side, slot) == chosen {
            continue;
        }
        let draws = match mon.ability.as_str() {
            "lightningrod" => Some("Electric"),
            "stormdrain" => Some("Water"),
            _ => None,
        };
        if draws == Some(mv.mtype.as_str()) {
            return Some((foe_side, slot));
        }
    }
    None
}

fn blocked_by_protect(
    turn: &Turn,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
) -> Option<String> {
    if !mv.has_flag("protect") || mv.raw_bool("breaksProtect") {
        return None;
    }
    let side = &turn.pos.sides[target.0];
    let from_foe = target.0 != action.side;
    if from_foe
        && matches!(mv.target.as_str(), "allAdjacentFoes" | "allAdjacent")
        && side.has_side_condition("wideguard")
    {
        return Some("wideguard".into());
    }
    if from_foe && action.priority > 0 && side.has_side_condition("quickguard") {
        return Some("quickguard".into());
    }
    let mon = turn.mon_at(target.0, target.1)?;
    for (vid, blocks) in PROTECT_VOLATILES {
        if !mon.has_volatile(vid) {
            continue;
        }
        if blocks == "damaging" && mv.category == "Status" {
            continue;
        }
        return Some(vid.to_string());
    }
    None
}

fn multihit_counts(mv: &Move, budget: &Budget) -> Vec<(usize, f64)> {
    let Some(multihit) = mv.raw.get("multihit") else { return vec![(1, 1.0)] };
    if let Some(fixed) = multihit.as_u64() {
        return vec![(fixed as usize, 1.0)];
    }
    let Some(list) = multihit.as_array() else { return vec![(1, 1.0)] };
    let low = list.first().and_then(Value::as_u64).unwrap_or(1) as usize;
    let high = list.last().and_then(Value::as_u64).unwrap_or(low as u64) as usize;
    if !budget.enumerate_secondary {
        return vec![(low, 1.0)];
    }
    if (low, high) == (2, 5) {
        return vec![
            (2, 1.0 / 3.0),
            (3, 1.0 / 3.0),
            (4, 1.0 / 6.0),
            (5, 1.0 / 6.0),
        ];
    }
    let span = high - low + 1;
    (0..span).map(|i| (low + i, 1.0 / span as f64)).collect()
}

#[allow(clippy::too_many_arguments)]
fn hit_target<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
    spread: bool,
    budget: Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let move_id = action.move_id.unwrap();
    if let Some(blocked) = blocked_by_protect(&turn, action, mv, target) {
        protect_punish(&mut turn, action, mv, &blocked)?;
        return Ok(vec![(1.0, turn)]);
    }

    let Some(attacker) = turn.battler_at(action.side, action.slot)? else {
        return Ok(vec![(1.0, turn)]);
    };
    let Some(defender) = turn.battler_at(target.0, target.1)? else {
        return Ok(vec![(1.0, turn)]);
    };
    if matches!(defender.ability.as_str(), "disguise" | "iceface") {
        return Err(format!("forme guard: {}", defender.ability));
    }

    let accuracy = accuracy_of(&turn, mv, &attacker, &defender);
    let crit_p = crit_probability(reg, &attacker, &defender, move_id.as_str());

    let accuracy_branches: Vec<(f64, bool)> =
        if budget.enumerate_accuracy && accuracy > 0.0 && accuracy < 1.0 {
            vec![(accuracy, true), (1.0 - accuracy, false)]
        } else {
            vec![(1.0, accuracy > 0.0)]
        };
    let crit_branches: Vec<(f64, bool)> =
        if budget.enumerate_crit && crit_p > 0.0 && crit_p < 1.0 {
            vec![(crit_p, true), (1.0 - crit_p, false)]
        } else {
            vec![(1.0, crit_p >= 1.0)]
        };
    let rolls = stratified_rolls(&budget);
    // Does not depend on the roll, and the exact budget enumerates sixteen of them -- so
    // building this inside the loop was fifteen wasted allocations per hit on the path that
    // advances a game. Under the matrix budget the roll is fixed and it costs nothing,
    // which is why the allocation count barely moved and the clock did.
    let hit_counts = multihit_counts(mv, &budget);

    let ctx_started = crate::resolve::phase_start();
    let move_ctx = MoveContext {
        weather: turn.pos.field.weather.map(|w| w.as_str().to_string()),
        terrain: turn.pos.field.terrain.map(|t| t.as_str().to_string()),
        side_total_fainted: turn.pos.sides[action.side]
            .pokemon
            .iter()
            .filter(|m| m.fainted)
            .count() as i64,
        times_attacked: turn
            .mon_at(action.side, action.slot)
            .map(|m| m.times_attacked)
            .unwrap_or(0),
        target_hurt_this_turn: turn.hurt_this_turn[target.0][target.1],
        damaged_by_target: false,
        previous_move_failed: turn
            .mon_at(action.side, action.slot)
            .map(|m| m.move_last_turn_failed)
            .unwrap_or(false),
        hit_index: 1,
        moving_last: false,
        ally_used_same_move: false,
    };

    crate::resolve::phase_end(12, ctx_started);

    let mut outcomes: Vec<Outcome<'a>> = Vec::new();
    for (acc_weight, hit) in accuracy_branches {
        if acc_weight <= 0.0 {
            continue;
        }
        if !hit {
            let mut state = turn.clone();
            state.move_failed[action.side][action.slot] = true;
            outcomes.push((acc_weight, state));
            continue;
        }
        let field = turn.field();
        for (crit_weight, crit) in crit_branches.iter().copied() {
        if crit_weight <= 0.0 {
            continue;
        }
        let started = crate::resolve::phase_start();
        let result = calculate(
            reg,
            &attacker,
            &defender,
            move_id.as_str(),
            &field,
            target.0,
            spread,
            crit,
            Some(&move_ctx),
            None,
            false,
        );
        crate::resolve::phase_end(10, started);
        for note in &result.unmodelled {
            turn.report(note.clone());
        }
        if result.immune {
            let mut state = turn.clone();
            state.move_failed[action.side][action.slot] = true;
            absorb(&mut state, mv, target);
            outcomes.push((acc_weight * crit_weight, state));
            continue;
        }
        for (roll, roll_weight) in &rolls {
            for (hits, hit_weight) in hit_counts.iter().copied() {
                let mut state = turn.clone();
                for hit_index in 0..hits {
                    let gone = match state.mon_at(target.0, target.1) {
                        None => true,
                        Some(mon) => mon.fainted,
                    };
                    if gone {
                        break;
                    }
                    let amount = if hit_index == 0 {
                        result.rolls[*roll]
                    } else {
                        let Some(live_attacker) =
                            state.battler_at(action.side, action.slot)?
                        else {
                            break;
                        };
                        let Some(live_defender) = state.battler_at(target.0, target.1)? else {
                            break;
                        };
                        let mut again_ctx = move_ctx.clone();
                        again_ctx.hit_index = hit_index as i64 + 1;
                        let field = state.field();
                        let again = calculate(
                            reg,
                            &live_attacker,
                            &live_defender,
                            move_id.as_str(),
                            &field,
                            target.0,
                            spread,
                            crit,
                            Some(&again_ctx),
                            None,
                            false,
                        );
                        if again.immune {
                            break;
                        }
                        again.rolls[*roll]
                    };
                    let dealt = state.deal_damage(target.0, target.1, amount, true)?;
                    let after_started = crate::resolve::phase_start();
                    after_hit(&mut state, action, mv, target, dealt, &budget, result.type_mod)?;
                    crate::resolve::phase_end(11, after_started);
                }
                let weight = acc_weight * crit_weight * roll_weight * hit_weight;
                for (extra, expanded) in spread_secondaries(state, action, hits > 1)? {
                    outcomes.push((weight * extra, expanded));
                }
            }
        }
        }
    }
    if outcomes.is_empty() {
        outcomes.push((1.0, turn));
    }
    Ok(outcomes)
}

fn accuracy_of(turn: &Turn, mv: &Move, attacker: &Battler, defender: &Battler) -> f64 {
    if mv.accuracy.is_none() || mv.raw_bool("alwaysHit") {
        return 1.0;
    }
    if attacker.ability == "noguard" || defender.ability == "noguard" {
        return 1.0;
    }
    if mv.id == "toxic" && attacker.types.contains("Poison") {
        return 1.0;
    }
    let mut accuracy = mv.accuracy.unwrap() as f64;
    let weather = turn.pos.field.weather.map(|w| w.as_str().to_string());
    if mv.id == "blizzard" && matches!(weather.as_deref(), Some("hail" | "snowscape" | "snow")) {
        return 1.0;
    }
    if mv.id == "thunder" || mv.id == "hurricane" {
        if matches!(weather.as_deref(), Some("raindance" | "primordialsea")) {
            return 1.0;
        }
        if matches!(weather.as_deref(), Some("sunnyday" | "desolateland")) {
            accuracy = 50.0;
        }
    }
    if attacker.ability == "compoundeyes" {
        accuracy *= 1.3;
    }
    if attacker.ability == "hustle" && mv.category == "Physical" {
        accuracy *= 0.8;
    }
    if is(attacker.item, "widelens") {
        accuracy *= 1.1;
    }
    if is(defender.item, "brightpowder") {
        accuracy *= 0.9;
    }
    if !mv.raw_bool("ignoreEvasion") {
        let stages = (attacker.boost("accuracy") - defender.boost("evasion")).clamp(-6, 6);
        let ratio = if stages >= 0 {
            (3.0 + stages as f64) / 3.0
        } else {
            3.0 / (3.0 - stages as f64)
        };
        accuracy *= ratio;
    }
    (accuracy / 100.0).clamp(0.0, 1.0)
}

fn protect_punish(
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    blocked: &str,
) -> Result<(), String> {
    if !mv.has_flag("contact") {
        return Ok(());
    }
    let me = (action.side, action.slot);
    let maxhp = match turn.mon_at(me.0, me.1) {
        None => return Ok(()),
        Some(mon) => mon.maxhp,
    };
    match blocked {
        "spikyshield" => {
            turn.deal_damage(me.0, me.1, (maxhp / 8).max(1), false)?;
        }
        "banefulbunker" => {
            turn.apply_status(me.0, me.1, "psn")?;
        }
        "burningbulwark" => {
            turn.apply_status(me.0, me.1, "brn")?;
        }
        "kingsshield" => {
            turn.apply_boosts(me.0, me.1, &[("atk", -1)], true);
        }
        "obstruct" => {
            turn.apply_boosts(me.0, me.1, &[("def", -2)], true);
        }
        "silktrap" => {
            turn.apply_boosts(me.0, me.1, &[("spe", -1)], true);
        }
        _ => {}
    }
    Ok(())
}

fn absorb(turn: &mut Turn, mv: &Move, target: Slot) {
    let (ability, maxhp) = match turn.mon_at(target.0, target.1) {
        None => return,
        Some(mon) if mon.fainted => return,
        Some(mon) => (mon.ability, mon.maxhp),
    };
    let heals = match ability.as_str() {
        "waterabsorb" | "dryskin" => Some("Water"),
        "voltabsorb" => Some("Electric"),
        "eartheater" => Some("Ground"),
        _ => None,
    };
    if heals == Some(mv.mtype.as_str()) {
        turn.heal(target.0, target.1, (maxhp / 4).max(1));
        return;
    }
    let boosts: Option<(&str, &[(&str, i64)])> = match ability.as_str() {
        "lightningrod" => Some(("Electric", &[("spa", 1)])),
        "stormdrain" => Some(("Water", &[("spa", 1)])),
        "motordrive" => Some(("Electric", &[("spe", 1)])),
        "sapsipper" => Some(("Grass", &[("atk", 1)])),
        "windrider" => Some(("Flying", &[("atk", 1)])),
        "wellbakedbody" => Some(("Fire", &[("def", 2)])),
        "steamengine" => Some(("Fire", &[("spe", 6)])),
        _ => None,
    };
    if let Some((wanted, table)) = boosts {
        if wanted == mv.mtype.as_str() {
            turn.apply_boosts(target.0, target.1, table, false);
            return;
        }
    }
    if ability == "flashfire" && mv.mtype == "Fire" {
        turn.add_volatile(target.0, target.1, "flashfire", None);
    }
}

fn after_hit(
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
    dealt: i64,
    budget: &Budget,
    type_mod: i64,
) -> Result<(), String> {
    let me = (action.side, action.slot);
    turn.move_damage_total += dealt;
    turn.move_connected = true;

    let defender_alive = matches!(turn.mon_at(target.0, target.1), Some(m) if !m.fainted);

    if defender_alive {
        if let Some(vid) = mv.raw_str("volatileStatus") {
            let vid = vid.to_string();
            if !crate::resolve::volatile_is_handled(&vid) {
                return Err(format!("move volatile: {vid}"));
            }
            let duration = effect_duration(turn, mv, &vid, action.side, action.slot);
            if duration_is_rolled(mv, &vid) {
                turn.report(format!(
                    "{vid} duration (Showdown rolls it; pinned to the low end)"
                ));
            }
            turn.add_volatile(target.0, target.1, &vid, duration);
            if vid == "partiallytrapped" {
                if let Some(mon) = turn.mon_at_mut(target.0, target.1) {
                    if let Some(applied) = mon.volatile_mut("partiallytrapped") {
                        applied.source_slot = Some(Id::new(&format!("{}{}", me.0, me.1)));
                    }
                }
            }
        }
        if let Some(status) = mv.raw_str("status") {
            let status = status.to_string();
            turn.apply_status(target.0, target.1, &status)?;
        }
    }

    let defender_ability = turn.mon_at(target.0, target.1).map(|m| m.ability);
    if dealt > 0 && matches!(defender_ability, Some(a) if a.as_str() == "spicyspray") {
        turn.apply_status(me.0, me.1, "brn")?;
    }

    // Throat Chop adds its own condition from a 100%-chance `secondary.onHit`, so there is
    // nothing declarative in the dump to drive it.
    if mv.id == "throatchop" && dealt > 0 && defender_alive {
        let duration = effect_duration(turn, mv, "throatchop", action.side, action.slot);
        turn.add_volatile(target.0, target.1, "throatchop", duration);
    }

    // Cursed Body: `onDamagingHit` with `randomChance(3, 10)`, gated on neither contact
    // nor the target surviving -- the handler runs from `damage()`, before the faint is
    // processed. 43 of the 394 tournament teams carry it.
    if dealt > 0 && matches!(defender_ability, Some(a) if a.as_str() == "cursedbody") {
        let attacker_free = turn
            .mon_at(me.0, me.1)
            .map(|mon| !mon.has_volatile("disable"))
            .unwrap_or(false);
        if attacker_free {
            if budget.enumerate_secondary {
                turn.pending_secondaries.push((0.3, json!({ "disable": true }), me));
            } else if !budget.pinned_policy {
                // Same reasoning as a secondary: under the pinned policy `randomChance`
                // is answered with no, so not applying it is exact rather than approximate.
                turn.report("cursedbody (30% disable, not branched)");
            }
        }
    }

    if mv.has_flag("contact") && dealt > 0 {
        let (ability, item) = match turn.mon_at(target.0, target.1) {
            None => (None, None),
            Some(mon) => (Some(mon.ability), mon.item),
        };
        let attacker_maxhp = turn.mon_at(me.0, me.1).map(|m| m.maxhp).unwrap_or(0);
        if matches!(ability, Some(a) if matches!(a.as_str(), "roughskin" | "ironbarbs")) {
            turn.deal_damage(me.0, me.1, (attacker_maxhp / 8).max(1), false)?;
        }
        if is(item, "rockyhelmet") {
            turn.deal_damage(me.0, me.1, (attacker_maxhp / 6).max(1), false)?;
        }
        if let Some(ability) = ability {
            if matches!(
                ability.as_str(),
                "static" | "flamebody" | "effectspore" | "poisonpoint" | "cutecharm"
            ) {
                turn.report(format!("contact ability: {ability}"));
            }
        }
    }

    if matches!(attacker_ability_of(turn, me), Some(a) if a.as_str() == "poisontouch")
        && mv.has_flag("contact")
    {
        turn.report("ability: poisontouch (30% poison not branched)");
    }

    // A resist berry is eaten only by a hit it actually weakened.
    let eats_berry = match turn.mon_at(target.0, target.1) {
        None => false,
        Some(mon) if mon.fainted => false,
        Some(mon) => match mon.item.and_then(|i| resist_berry(i.as_str())) {
            None => false,
            Some(berry_type) => {
                berry_type == mv.mtype.as_str() && (berry_type == "Normal" || type_mod > 0)
            }
        },
    };
    if eats_berry && !turn.berries_blocked(target.0) {
        turn.consume_item(target.0, target.1);
    }

    // Knock Off removes what it hit; Thief and Covet take it when the attacker has
    // nothing. A mega stone refuses to leave the species it belongs to, and only that
    // species, which `item_is_removable` reads from the regulation's own mega map.
    if matches!(mv.id.as_str(), "knockoff" | "thief" | "covet") {
        let target_item = match turn.mon_at(target.0, target.1) {
            None => None,
            Some(mon) if mon.fainted => None,
            Some(mon) => mon.item,
        };
        if let Some(item) = target_item {
            let species = turn.mon_at(target.0, target.1).unwrap().species;
            let removable = turn
                .reg
                .item_is_removable(species.as_str(), Some(item.as_str()));
            let attacker_empty =
                matches!(turn.mon_at(me.0, me.1), Some(mon) if mon.item.is_none());
            if removable && mv.id == "knockoff" {
                turn.consume_item(target.0, target.1);
            } else if removable && attacker_empty {
                turn.consume_item(target.0, target.1);
                if let Some(mon) = turn.mon_at_mut(me.0, me.1) {
                    mon.item = Some(item);
                }
            }
        }
    }

    let attacker_ability = turn.mon_at(me.0, me.1).map(|m| m.ability);
    if let Some(list) = mv.raw.get("secondaries").and_then(Value::as_array) {
        for secondary in list {
            if matches!(attacker_ability, Some(a) if a.as_str() == "sheerforce") {
                continue;
            }
            let chance = secondary
                .get("chance")
                .and_then(Value::as_f64)
                .unwrap_or(100.0)
                / 100.0;
            check_secondary_supported(secondary)?;
            if chance >= 1.0 {
                apply_secondary(turn, action, secondary, target)?;
                continue;
            }
            if !budget.enumerate_secondary {
                if !budget.pinned_policy {
                    turn.report(format!(
                        "secondary {}%: {} (not branched)",
                        (chance * 100.0) as i64,
                        mv.id
                    ));
                }
                continue;
            }
            turn.pending_secondaries.push((chance, secondary.clone(), target));
        }
    }

    if mv.raw_bool("forceSwitch") {
        return Err("forceSwitch move".into());
    }

    on_being_hit(turn, mv, target, action.side)
}

fn check_secondary_supported(secondary: &Value) -> Result<(), String> {
    for key in secondary.as_object().map(|o| o.keys().collect::<Vec<_>>()).unwrap_or_default() {
        if !matches!(key.as_str(), "chance" | "status" | "volatileStatus" | "boosts" | "self")
        {
            return Err(format!("secondary field: {key}"));
        }
    }
    Ok(())
}

fn apply_secondary(
    turn: &mut Turn,
    action: &QueuedAction,
    secondary: &Value,
    target: Slot,
) -> Result<(), String> {
    if secondary.get("disable").and_then(Value::as_bool).unwrap_or(false) {
        // Cursed Body, pushed through the secondary fan-out because it is the same shape:
        // a chance whose consequence outlives the turn.
        apply_disable(turn, target.0, target.1, None);
        return Ok(());
    }
    if let Some(status) = secondary.get("status").and_then(Value::as_str) {
        let status = status.to_string();
        turn.apply_status(target.0, target.1, &status)?;
    }
    if let Some(vid) = secondary.get("volatileStatus").and_then(Value::as_str) {
        let vid = vid.to_string();
        if !crate::resolve::volatile_is_handled(&vid) {
            return Err(format!("secondary volatile: {vid}"));
        }
        turn.add_volatile(target.0, target.1, &vid, None);
    }
    if let Some(boosts) = secondary.get("boosts").and_then(Value::as_object) {
        let table: Vec<(&str, i64)> = boosts
            .iter()
            .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
            .collect();
        turn.apply_boosts(target.0, target.1, &table, true);
    }
    if let Some(self_boosts) = secondary
        .get("self")
        .and_then(|s| s.get("boosts"))
        .and_then(Value::as_object)
    {
        let table: Vec<(&str, i64)> = self_boosts
            .iter()
            .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
            .collect();
        turn.apply_boosts(action.side, action.slot, &table, false);
    }
    Ok(())
}

fn spread_secondaries<'a>(
    mut state: Turn<'a>,
    action: &QueuedAction,
    multihit: bool,
) -> Result<Vec<(f64, Turn<'a>)>, String> {
    if state.pending_secondaries.is_empty() {
        return Ok(vec![(1.0, state)]);
    }
    let pending = std::mem::take(&mut state.pending_secondaries);
    if pending.len() > MAX_BRANCHED_SECONDARIES {
        return Err("more secondaries on one hit than this port branches".into());
    }
    if multihit {
        return Err("secondary on a multi-hit move".into());
    }
    let mut out: Vec<(f64, Turn<'a>)> = vec![(1.0, state)];
    for (chance, secondary, target) in pending {
        let mut expanded: Vec<(f64, Turn<'a>)> = Vec::new();
        for (weight, current) in out.into_iter() {
            let mut fired = current.clone();
            fired.pending_secondaries.clear();
            apply_secondary(&mut fired, action, &secondary, target)?;
            expanded.push((weight * chance, fired));
            expanded.push((weight * (1.0 - chance), current));
        }
        out = expanded;
    }
    Ok(out)
}

fn on_being_hit(
    turn: &mut Turn,
    mv: &Move,
    target: Slot,
    attacker_side: usize,
) -> Result<(), String> {
    let ability = match turn.mon_at(target.0, target.1) {
        None => return Ok(()),
        Some(mon) if mon.fainted => return Ok(()),
        Some(mon) => mon.ability,
    };
    if ability == "toxicdebris" && mv.category == "Physical" && target.0 != attacker_side {
        let layers = turn.pos.sides[attacker_side]
            .side_condition("toxicspikes")
            .and_then(|c| c.layers)
            .unwrap_or(0);
        if layers < 2 {
            turn.add_side_condition(attacker_side, "toxicspikes", None);
        }
    }
    let entry: Option<(&[(&str, i64)], &[&str])> = match ability.as_str() {
        "stamina" => Some((&[("def", 1)], &[])),
        "weakarmor" => Some((&[("def", -1), ("spe", 2)], &[])),
        "justified" => Some((&[("atk", 1)], &["Dark"])),
        "rattled" => Some((&[("spe", 1)], &["Bug", "Dark", "Ghost"])),
        "steamengine" => Some((&[("spe", 6)], &["Fire", "Water"])),
        "watercompaction" => Some((&[("def", 2)], &["Water"])),
        _ => None,
    };
    if let Some((boosts, types)) = entry {
        if types.is_empty() || types.contains(&mv.mtype.as_str()) {
            turn.apply_boosts(target.0, target.1, boosts, false);
        }
    }
    // Cursed Body stays on this list even though the Disable is branched: Python reports
    // it here regardless, and the report is part of what a turn returns.
    if matches!(ability.as_str(), "angerpoint" | "berserk" | "angershell" | "cursedbody") {
        turn.report(format!("on-hit ability: {ability}"));
    }
    Ok(())
}

fn attacker_ability_of(turn: &Turn, me: Slot) -> Option<Id> {
    turn.mon_at(me.0, me.1).map(|mon| mon.ability)
}

/// Records that the user of a self-switching move has to be replaced.
///
/// Which Pokemon comes in is the player's choice, so nothing is picked here. A Pokemon with
/// an empty bench is not marked at all -- Showdown's `switchFlag` has nothing to answer it
/// with, and the move simply leaves it in place.
fn mark_self_switch(turn: &mut Turn, action: &QueuedAction) {
    let alive = matches!(turn.mon_at(action.side, action.slot), Some(mon) if !mon.fainted);
    if !alive {
        return;
    }
    let bench = turn.pos.sides[action.side]
        .pokemon
        .iter()
        .filter(|mon| !mon.fainted && !mon.is_active())
        .count();
    if bench == 0 {
        return;
    }
    turn.add_volatile(action.side, action.slot, "pendingselfswitch", None);
    turn.self_switch_pending = true;
}

fn round_fraction(amount: i64, ratio: &Value) -> i64 {
    let list = ratio.as_array();
    let (numerator, denominator) = match list {
        Some(values) if values.len() >= 2 => (
            values[0].as_i64().unwrap_or(1),
            values[1].as_i64().unwrap_or(1),
        ),
        _ => (1, 1),
    };
    let scaled = amount * numerator;
    ((scaled * 2 + denominator) / (2 * denominator)).max(1)
}

fn after_move(turn: &mut Turn, action: &QueuedAction, mv: &Move) -> Result<(), String> {
    let me = (action.side, action.slot);
    let total = turn.move_damage_total;

    if let Some(drain) = mv.raw.get("drain") {
        if total > 0 {
            let amount = round_fraction(total, drain);
            turn.heal(me.0, me.1, amount);
        }
    }
    let rockhead = matches!(turn.mon_at(me.0, me.1), Some(m) if m.ability == "rockhead");
    if let Some(recoil) = mv.raw.get("recoil") {
        if total > 0 && !rockhead {
            let amount = round_fraction(total, recoil);
            turn.deal_damage(me.0, me.1, amount, false)?;
        }
    }
    let (item, maxhp) = match turn.mon_at(me.0, me.1) {
        None => (None, 0),
        Some(mon) => (mon.item, mon.maxhp),
    };
    if is(item, "lifeorb") && total > 0 {
        turn.deal_damage(me.0, me.1, (maxhp / 10).max(1), false)?;
    }
    if is(item, "shellbell") && total >= 8 {
        turn.heal(me.0, me.1, total / 8);
    }

    let connected = turn.move_connected;
    let alive = matches!(turn.mon_at(me.0, me.1), Some(m) if !m.fainted);
    if alive && connected {
        if let Some(self_effect) = mv.raw.get("self") {
            if let Some(boosts) = self_effect.get("boosts").and_then(Value::as_object) {
                let table: Vec<(&str, i64)> = boosts
                    .iter()
                    .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
                    .collect();
                turn.apply_boosts(me.0, me.1, &table, false);
            }
            if let Some(vid) = self_effect.get("volatileStatus").and_then(Value::as_str) {
                let vid = vid.to_string();
                if !crate::resolve::volatile_is_handled(&vid) {
                    return Err(format!("self volatile: {vid}"));
                }
                turn.add_volatile(me.0, me.1, &vid, None);
            }
        }
        if let Some(boosts) = mv
            .raw
            .get("selfBoost")
            .and_then(|s| s.get("boosts"))
            .and_then(Value::as_object)
        {
            let table: Vec<(&str, i64)> = boosts
                .iter()
                .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
                .collect();
            turn.apply_boosts(me.0, me.1, &table, false);
        }
    }

    if mv.raw_bool("selfSwitch") && turn.move_connected {
        mark_self_switch(turn, action);
    }

    check_white_herb(turn);
    turn.move_damage_total = 0;
    turn.move_connected = false;
    Ok(())
}

// ---------------------------------------------------------------------------
// Status moves
// ---------------------------------------------------------------------------

fn do_status_move<'a>(
    reg: &'a Reg,
    mut turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    targets: &[Slot],
    budget: Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let Some(attacker) = turn.battler_at(action.side, action.slot)? else {
        return Ok(vec![(1.0, turn)]);
    };

    let mut reachable: Vec<Slot> = Vec::new();
    for target in targets {
        if *target != (action.side, action.slot)
            && blocked_by_protect(&turn, action, mv, *target).is_some()
        {
            continue;
        }
        if immune_to_move(reg, &turn, action, mv, *target).is_some() {
            continue;
        }
        reachable.push(*target);
    }

    if reachable.is_empty() && !targets.is_empty() {
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    let mut accuracy = 1.0f64;
    for target in &reachable {
        if *target == (action.side, action.slot) {
            continue;
        }
        if let Some(defender) = turn.battler_at(target.0, target.1)? {
            accuracy = accuracy.min(accuracy_of(&turn, mv, &attacker, &defender));
        }
    }

    if mv.raw_bool("stallingMove") {
        return do_protect(turn, action, mv, &budget);
    }

    if STALL_BUMPING_MOVES.contains(&mv.id.as_str()) && turn.actions_remaining == 0 {
        turn.move_failed[action.side][action.slot] = true;
        return Ok(vec![(1.0, turn)]);
    }

    if !budget.enumerate_accuracy || accuracy >= 1.0 || accuracy <= 0.0 {
        if accuracy <= 0.0 {
            turn.move_failed[action.side][action.slot] = true;
            return Ok(vec![(1.0, turn)]);
        }
        apply_status_move(turn.reg, &mut turn, action, mv, &reachable)?;
        return Ok(vec![(1.0, turn)]);
    }

    let mut hit_state = turn.clone();
    apply_status_move(reg, &mut hit_state, action, mv, &reachable)?;
    turn.move_failed[action.side][action.slot] = true;
    Ok(vec![(accuracy, hit_state), (1.0 - accuracy, turn)])
}

fn immune_to_move(
    reg: &Reg,
    turn: &Turn,
    action: &QueuedAction,
    mv: &Move,
    target: Slot,
) -> Option<String> {
    let defender = turn.mon_at(target.0, target.1)?;
    if defender.fainted {
        return None;
    }
    let self_targeted = target == (action.side, action.slot);
    let types = turn.types_of(defender);

    if mv.has_flag("powder") && !self_targeted {
        let immune = reg
            .effect_immunities
            .get("powder")
            .map(|set| types.as_slice().iter().any(|t| set.contains(t.as_str())))
            .unwrap_or(false);
        if immune {
            return Some("powder".into());
        }
        if defender.ability == "overcoat" {
            return Some("overcoat".into());
        }
        if is(defender.item, "safetygoggles") {
            return Some("safetygoggles".into());
        }
    }

    if mv.category == "Status" && target.0 != action.side {
        let attacker_is_prankster = turn
            .mon_at(action.side, action.slot)
            .map(|m| m.ability == "prankster")
            .unwrap_or(false);
        if attacker_is_prankster {
            let immune = reg
                .effect_immunities
                .get("prankster")
                .map(|set| types.as_slice().iter().any(|t| set.contains(t.as_str())))
                .unwrap_or(false);
            if immune {
                return Some("prankster".into());
            }
        }
    }
    None
}

/// Disables the target's last move, as Showdown's `disable` condition does.
///
/// Fails -- with no volatile at all -- when the target has not moved yet, or when the move
/// it last used has no PP left; both are `return false` in `onStart`, which means no
/// volatile rather than a volatile that disables nothing.
///
/// The duration is decremented immediately when the target still owes an action this turn
/// or when it is the Pokemon whose own move triggered this, which covers Cursed Body and
/// the ordinary Disable; five turns is left for disabling something that has already acted.
fn apply_disable(turn: &mut Turn, side: usize, slot: usize, mv: Option<&Move>) -> bool {
    let (last_move, already) = match turn.mon_at(side, slot) {
        None => return false,
        Some(mon) if mon.fainted => return false,
        Some(mon) => (mon.last_move, mon.has_volatile("disable")),
    };
    let Some(last_move) = last_move else { return false };
    if already {
        return false;
    }
    let has_pp = match turn.mon_at(side, slot) {
        None => false,
        Some(mon) => mon.moves.get(last_move).map(|m| m.pp > 0).unwrap_or(false),
    };
    if !has_pp {
        return false;
    }
    let mut duration = match mv {
        Some(mv) => effect_duration(turn, mv, "disable", side, slot).unwrap_or(5),
        None => 5,
    };
    if !turn.acted[side][slot] || turn.current_actor == Some((side, slot)) {
        duration -= 1;
    }
    if let Some(mon) = turn.mon_at_mut(side, slot) {
        let mut effect = Effect::new(Id::new("disable"));
        effect.duration = Some(duration);
        effect.move_id = Some(last_move);
        mon.volatiles.push(effect);
        // The flag lives on the move slot, which is what the legality half reads.
        if let Some(move_slot) = mon.moves.get_mut(last_move) {
            move_slot.disabled = true;
        }
    }
    true
}

/// Locks the target into the move it last used. Fails -- with no volatile at all --
/// when the target has not moved, when that move cannot be encored, or when it is out of
/// PP, all three of which are `return false` in Showdown's `onStart`.
fn apply_encore(turn: &mut Turn, side: usize, slot: usize, mv: &Move) -> bool {
    let (last_move, already) = match turn.mon_at(side, slot) {
        None => return false,
        Some(mon) if mon.fainted => return false,
        Some(mon) => (mon.last_move, mon.has_volatile("encore")),
    };
    let Some(last_move) = last_move else { return false };
    if already {
        return false;
    }
    match turn.reg.moves.get(last_move.as_str()) {
        None => return false,
        Some(last) if last.has_flag("failencore") => return false,
        Some(_) => {}
    }
    let has_pp = match turn.mon_at(side, slot) {
        None => false,
        Some(mon) => mon.moves.get(last_move).map(|m| m.pp > 0).unwrap_or(false),
    };
    if !has_pp {
        return false;
    }
    let mut duration = effect_duration(turn, mv, "encore", side, slot).unwrap_or(3);
    if turn.acted[side][slot] {
        duration += 1;
    }
    if let Some(mon) = turn.mon_at_mut(side, slot) {
        let mut effect = Effect::new(Id::new("encore"));
        effect.duration = Some(duration);
        effect.move_id = Some(last_move);
        mon.volatiles.push(effect);
    }
    true
}

fn stall_success_chance(counter: i64) -> f64 {
    1.0 / counter.max(1) as f64
}

fn bump_stall(turn: &mut Turn, side: usize, slot: usize) {
    let Some(mon) = turn.mon_at_mut(side, slot) else { return };
    match mon.volatile_mut("stall") {
        None => {
            let mut effect = Effect::new(Id::new("stall"));
            effect.duration = Some(2);
            effect.counter = Some(3);
            mon.volatiles.push(effect);
        }
        Some(existing) => {
            existing.counter = Some((existing.counter.unwrap_or(1) * 3).min(729));
            existing.duration = Some(2);
        }
    }
}

fn do_protect<'a>(
    mut turn: Turn<'a>,
    action: &QueuedAction,
    mv: &Move,
    budget: &Budget,
) -> Result<Vec<Outcome<'a>>, String> {
    let me = (action.side, action.slot);
    let Some(mon) = turn.mon_at(me.0, me.1) else { return Ok(vec![(1.0, turn)]) };
    if turn.actions_remaining == 0 {
        turn.move_failed[me.0][me.1] = true;
        return Ok(vec![(1.0, turn)]);
    }
    let counter = mon.volatile("stall").and_then(|s| s.counter).unwrap_or(1);
    let chance = stall_success_chance(counter);

    let succeed = |state: &mut Turn, move_id: &str| {
        state.add_volatile(me.0, me.1, move_id, Some(1));
        bump_stall(state, me.0, me.1);
    };
    let fail = |state: &mut Turn| {
        state.move_failed[me.0][me.1] = true;
        if let Some(mon) = state.mon_at_mut(me.0, me.1) {
            mon.volatiles.retain(|v| v.id.as_str() != "stall");
        }
    };

    if chance >= 1.0 {
        succeed(&mut turn, &mv.id);
        return Ok(vec![(1.0, turn)]);
    }
    if !budget.enumerate_status_checks {
        fail(&mut turn);
        return Ok(vec![(1.0, turn)]);
    }
    let mut hit_state = turn.clone();
    succeed(&mut hit_state, &mv.id);
    fail(&mut turn);
    Ok(vec![(chance, hit_state), (1.0 - chance, turn)])
}

/// Whether Showdown really *rolls* this duration. A `durationCallback` alone does not
/// mean that -- most of them apply an item or ability extension -- so the dumper records
/// whether calling it consumed randomness, and that is what this reads.
fn duration_is_rolled(mv: &Move, effect_id: &str) -> bool {
    mv.raw
        .get("durations")
        .and_then(Value::as_object)
        .and_then(|durations| durations.get(effect_id))
        .and_then(Value::as_object)
        .and_then(|entry| entry.get("rolled"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn effect_duration(
    turn: &Turn,
    mv: &Move,
    effect_id: &str,
    side: usize,
    slot: usize,
) -> Option<i64> {
    let durations = mv.raw.get("durations")?.as_object()?;
    let entry = durations.get(effect_id)?.as_object()?;
    if let Some(mon) = turn.mon_at(side, slot) {
        if let (Some(by_item), Some(item)) = (entry.get("byItem").and_then(Value::as_object), mon.item)
        {
            if let Some(value) = by_item.get(item.as_str()).and_then(Value::as_i64) {
                return Some(value);
            }
        }
        if let Some(by_ability) = entry.get("byAbility").and_then(Value::as_object) {
            if let Some(value) = by_ability.get(mon.ability.as_str()).and_then(Value::as_i64) {
                return Some(value);
            }
        }
    }
    entry
        .get("base")
        .and_then(Value::as_i64)
        .or_else(|| entry.get("duration").and_then(Value::as_i64))
}

fn apply_status_move(
    reg: &Reg,
    turn: &mut Turn,
    action: &QueuedAction,
    mv: &Move,
    targets: &[Slot],
) -> Result<(), String> {
    let me = (action.side, action.slot);
    let mut suppress_self_switch = false;

    if let Some(condition) = mv.raw_str("sideCondition") {
        let condition = condition.to_string();
        let duration = effect_duration(turn, mv, &condition, action.side, action.slot);
        if duration_is_rolled(mv, &condition) {
            turn.report(format!(
                "{condition} duration (Showdown rolls it; pinned to the low end)"
            ));
        }
        turn.add_side_condition(action.side, &condition, duration);
    }
    if let Some(weather) = mv.raw_str("weather") {
        let weather = weather.to_lowercase().replace(' ', "");
        turn.pos.field.weather = Some(Id::new(&weather));
        turn.pos.field.weather_duration =
            Some(effect_duration(turn, mv, &weather, action.side, action.slot).unwrap_or(5));
    }
    if let Some(terrain) = mv.raw_str("terrain") {
        let terrain = terrain.to_lowercase().replace(' ', "");
        turn.pos.field.terrain = Some(Id::new(&terrain));
        turn.pos.field.terrain_duration =
            Some(effect_duration(turn, mv, &terrain, action.side, action.slot).unwrap_or(5));
    }
    if let Some(pseudo) = mv.raw_str("pseudoWeather") {
        let pid = pseudo.to_lowercase().replace(' ', "");
        let already = turn.pos.field.has_pseudo_weather(&pid);
        let toggling = matches!(pid.as_str(), "trickroom" | "magicroom" | "wonderroom");
        if already && toggling {
            turn.pos.field.pseudo_weather.retain(|p| p.id.as_str() != pid);
        } else if !already {
            let duration =
                effect_duration(turn, mv, &pid, action.side, action.slot).unwrap_or(5);
            let mut effect = Effect::new(Id::new(&pid));
            effect.duration = Some(duration);
            turn.pos.field.pseudo_weather.push(effect);
        }
    }

    if let Some(self_effect) = mv.raw.get("self") {
        if let Some(boosts) = self_effect.get("boosts").and_then(Value::as_object) {
            let table: Vec<(&str, i64)> = boosts
                .iter()
                .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
                .collect();
            turn.apply_boosts(me.0, me.1, &table, false);
        }
        if let Some(vid) = self_effect.get("volatileStatus").and_then(Value::as_str) {
            let vid = vid.to_string();
            if !crate::resolve::volatile_is_handled(&vid) {
                return Err(format!("self volatile: {vid}"));
            }
            turn.add_volatile(me.0, me.1, &vid, None);
        }
        if let Some(condition) = self_effect.get("sideCondition").and_then(Value::as_str) {
            let condition = condition.to_string();
            turn.add_side_condition(action.side, &condition, None);
        }
    }

    for target in targets {
        let own_side = target.0 == action.side;
        if let Some(boosts) = mv.raw.get("boosts").and_then(Value::as_object) {
            let table: Vec<(&str, i64)> = boosts
                .iter()
                .map(|(stat, value)| (stat.as_str(), value.as_i64().unwrap_or(0)))
                .collect();
            turn.apply_boosts(target.0, target.1, &table, !own_side);
        }
        if let Some(status) = mv.raw_str("status") {
            let status = status.to_string();
            turn.apply_status(target.0, target.1, &status)?;
        }
        if let Some(vid) = mv.raw_str("volatileStatus") {
            let vid = vid.to_string();
            if vid == "encore" {
                if !apply_encore(turn, target.0, target.1, mv) {
                    turn.move_failed[action.side][action.slot] = true;
                }
                continue;
            }
            if vid == "disable" {
                // Which move is the whole effect, and it fails outright when the target has
                // not moved, so the generic path cannot express it.
                if !apply_disable(turn, target.0, target.1, Some(mv)) {
                    turn.move_failed[action.side][action.slot] = true;
                }
                continue;
            }
            if !crate::resolve::volatile_is_handled(&vid) {
                return Err(format!("status move volatile: {vid}"));
            }
            let duration = effect_duration(turn, mv, &vid, action.side, action.slot);
            turn.add_volatile(target.0, target.1, &vid, duration);
            if vid == "leechseed" {
                if let Some(mon) = turn.mon_at_mut(target.0, target.1) {
                    if let Some(applied) = mon.volatile_mut("leechseed") {
                        applied.source_slot = Some(Id::new(&format!("{}{}", me.0, me.1)));
                    }
                }
            }
        }
        if let Some(heal) = mv.raw.get("heal") {
            let maxhp = turn.mon_at(target.0, target.1).map(|m| m.maxhp).unwrap_or(0);
            if maxhp > 0 {
                let amount = round_fraction(maxhp, heal);
                turn.heal(target.0, target.1, amount);
            }
        }
    }

    if mv.id == "partingshot" {
        let mut landed = false;
        for target in targets {
            if turn.apply_boosts(target.0, target.1, &[("atk", -1), ("spa", -1)], true) {
                landed = true;
            }
            if matches!(turn.mon_at(target.0, target.1), Some(m) if m.ability == "mirrorarmor") {
                landed = true;
            }
        }
        if !landed {
            suppress_self_switch = true;
        }
    }

    if matches!(mv.id.as_str(), "followme" | "ragepowder" | "spotlight") {
        turn.add_volatile(me.0, me.1, &mv.id, None);
    }
    if mv.id == "helpinghand" {
        for target in targets {
            turn.add_volatile(target.0, target.1, "helpinghand", None);
        }
    }
    if STALL_BUMPING_MOVES.contains(&mv.id.as_str()) {
        turn.add_side_condition(action.side, &mv.id, Some(1));
        bump_stall(turn, action.side, action.slot);
    }

    if mv.raw.get("hasCustomCode").and_then(Value::as_bool).unwrap_or(false)
        && !crate::modelled::status_move_is_fully_modelled(&mv.id)
    {
        turn.report(format!("status move: {}", mv.id));
    }
    if mv.raw_bool("selfSwitch") && !suppress_self_switch {
        mark_self_switch(turn, action);
    }
    if mv.raw_bool("forceSwitch") {
        return Err("forceSwitch status move".into());
    }
    let _ = reg;
    Ok(())
}

// ---------------------------------------------------------------------------
// End of turn
// ---------------------------------------------------------------------------

fn slot_of(turn: &Turn, source_slot: Option<Id>) -> Option<Slot> {
    let source = source_slot?;
    let text = source.as_str();
    let bytes = text.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let side = (bytes[0] as char).to_digit(10)? as usize;
    let slot = (bytes[1] as char).to_digit(10)? as usize;
    if side > 1 || slot >= turn.pos.sides[side].active.len() {
        return None;
    }
    Some((side, slot))
}

fn trapper_gone(turn: &Turn, source_slot: Option<Id>) -> bool {
    let Some((side, slot)) = slot_of(turn, source_slot) else { return false };
    match turn.mon_at(side, slot) {
        None => true,
        Some(mon) => mon.fainted,
    }
}

/// Active slots in Showdown's residual order: by Speed, fastest first.
fn residual_order(turn: &Turn) -> Result<(Vec<Slot>, bool), String> {
    let trick_room = turn.pos.field.trick_room();
    let field = turn.field();
    let mut entries: Vec<(i64, i64, String, usize, usize)> = Vec::new();
    let mut speeds: Vec<i64> = Vec::new();
    for side in 0..2 {
        let conditions = turn.pos.sides[side].side_conditions.clone();
        for slot in 0..turn.pos.sides[side].active.len() {
            let mon = turn.mon_at(side, slot);
            let fighter = turn.battler_at(side, slot)?;
            match (mon, fighter) {
                (Some(mon), Some(fighter)) => {
                    let mut speed = crate::speed::effective_speed(&fighter, &field, &conditions);
                    if trick_room {
                        speed = 10000 - speed;
                    }
                    speeds.push(speed);
                    entries.push((0, -speed, mon.species.as_str().to_string(), slot, side));
                }
                _ => entries.push((1, 0, String::new(), slot, side)),
            }
        }
    }
    let mut unique = speeds.clone();
    unique.sort_unstable();
    unique.dedup();
    let tied = unique.len() != speeds.len();
    // A Speed tie here is reported by Python rather than branched, and broken by
    // (-speed, species, slot, side) -- deliberately not by side, so the residual phase
    // cannot become seat-dependent. The same sort gives the same order, so the tie is not
    // a reason to refuse; it is a reason to sort on exactly the same key.
    entries.sort();
    Ok((
        entries.into_iter().map(|(_, _, _, slot, side)| (side, slot)).collect(),
        tied,
    ))
}

pub(crate) fn residuals(reg: &Reg, turn: &mut Turn) -> Result<(), String> {
    crate::resolve::RESIDUALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let _ = reg;
    let (order, tied) = residual_order(turn)?;
    if tied {
        turn.report("residual speed tie (Showdown breaks it at random)");
    }

    // Residual order 1: weather.
    let mut weather_expired = false;
    if turn.pos.field.weather.is_some() {
        if let Some(duration) = turn.pos.field.weather_duration {
            let left = duration - 1;
            turn.pos.field.weather_duration = Some(left);
            if left <= 0 {
                turn.pos.field.weather = None;
                turn.pos.field.weather_duration = None;
                weather_expired = true;
            }
        }
    }

    if matches!(turn.pos.field.weather, Some(w) if w.as_str() == "sandstorm") && !weather_expired
    {
        for (side, slot) in order.iter().copied() {
            let skip = match turn.mon_at(side, slot) {
                None => true,
                Some(mon) => {
                    mon.fainted
                        || turn
                            .types_of(mon)
                            .as_slice()
                            .iter()
                            .any(|t| matches!(t.as_str(), "Rock" | "Ground" | "Steel"))
                        || matches!(
                            mon.ability.as_str(),
                            "sandveil" | "sandrush" | "sandforce" | "overcoat" | "magicguard"
                        )
                        || is(mon.item, "safetygoggles")
                }
            };
            if skip {
                continue;
            }
            let amount = turn.fraction_of_max(side, slot, SANDSTORM_DAMAGE);
            turn.deal_damage(side, slot, amount, false)?;
        }
    }

    if turn.pos.field.weather.is_some() && !weather_expired {
        let weather = turn.pos.field.weather.unwrap();
        for (side, slot) in order.iter().copied() {
            let ability = match turn.mon_at(side, slot) {
                None => continue,
                Some(mon) if mon.fainted => continue,
                Some(mon) => mon.ability,
            };
            let ratio: Option<(i64, i64)> = match (ability.as_str(), weather.as_str()) {
                ("solarpower", "sunnyday") | ("solarpower", "desolateland") => Some((-1, 8)),
                ("dryskin", "sunnyday") | ("dryskin", "desolateland") => Some((-1, 8)),
                ("dryskin", "raindance") | ("dryskin", "primordialsea") => Some((1, 8)),
                ("raindish", "raindance") | ("raindish", "primordialsea") => Some((1, 16)),
                ("icebody", "hail") | ("icebody", "snowscape") => Some((1, 16)),
                _ => None,
            };
            let Some((numerator, denominator)) = ratio else { continue };
            let amount = turn.fraction_of_max(side, slot, (numerator.abs(), denominator));
            if numerator < 0 {
                turn.deal_damage(side, slot, amount, false)?;
            } else {
                turn.heal(side, slot, amount);
            }
        }
    }

    for (side, slot) in order.iter().copied() {
        let leftovers = matches!(turn.mon_at(side, slot), Some(mon)
            if !mon.fainted && is(mon.item, "leftovers"));
        if leftovers {
            let amount = turn.fraction_of_max(side, slot, LEFTOVERS_HEAL);
            turn.heal(side, slot, amount);
        }
    }

    for (side, slot) in order.iter().copied() {
        let seed_source = match turn.mon_at(side, slot) {
            None => continue,
            Some(mon) if mon.fainted || mon.ability == "magicguard" => continue,
            Some(mon) => mon.volatile("leechseed").map(|s| s.source_slot),
        };
        let Some(source_slot) = seed_source else { continue };
        let amount = turn.fraction_of_max(side, slot, LEECH_SEED_DRAIN);
        let drained = turn.deal_damage(side, slot, amount, false)?;
        if drained > 0 {
            if let Some(planter) = slot_of(turn, source_slot) {
                let alive =
                    matches!(turn.mon_at(planter.0, planter.1), Some(m) if !m.fainted);
                if alive {
                    turn.heal(planter.0, planter.1, drained);
                }
            }
        }
    }

    for (side, slot) in order.iter().copied() {
        let (status, has_trap, trap_source, has_saltcure, maxhp, counter) =
            match turn.mon_at(side, slot) {
                None => continue,
                Some(mon) if mon.fainted || mon.ability == "magicguard" => continue,
                Some(mon) => (
                    mon.status,
                    mon.has_volatile("partiallytrapped"),
                    mon.volatile("partiallytrapped").and_then(|t| t.source_slot),
                    mon.has_volatile("saltcure"),
                    mon.maxhp,
                    mon.status_counter,
                ),
            };
        if is(status, "brn") {
            let amount = turn.fraction_of_max(side, slot, BURN_DAMAGE);
            turn.deal_damage(side, slot, amount, false)?;
        } else if is(status, "psn") {
            let amount = turn.fraction_of_max(side, slot, POISON_DAMAGE);
            turn.deal_damage(side, slot, amount, false)?;
        } else if is(status, "tox") {
            let stage = (counter.unwrap_or(0) + 1).min(15);
            if let Some(mon) = turn.mon_at_mut(side, slot) {
                mon.status_counter = Some(stage);
            }
            let per_stage = (maxhp / 16).max(1);
            turn.deal_damage(side, slot, per_stage * stage, false)?;
        }
        if has_trap {
            if trapper_gone(turn, trap_source) {
                if let Some(mon) = turn.mon_at_mut(side, slot) {
                    mon.volatiles.retain(|v| v.id.as_str() != "partiallytrapped");
                }
            } else {
                let amount = turn.fraction_of_max(side, slot, PARTIAL_TRAP_DAMAGE);
                turn.deal_damage(side, slot, amount, false)?;
            }
        }
        if has_saltcure {
            let weak = match turn.mon_at(side, slot) {
                None => false,
                Some(mon) => {
                    let types = turn.types_of(mon);
                    types.contains("Water") || types.contains("Steel")
                }
            };
            let ratio = if weak { SALT_CURE_DAMAGE_WEAK } else { SALT_CURE_DAMAGE };
            let amount = turn.fraction_of_max(side, slot, ratio);
            turn.deal_damage(side, slot, amount, false)?;
        }
    }

    // Residual order 24: Perish Song.
    for (side, slot) in order.iter().copied() {
        let left = {
            let Some(mon) = turn.mon_at_mut(side, slot) else { continue };
            if mon.fainted {
                continue;
            }
            match mon.volatile_mut("perishsong") {
                None => continue,
                Some(perish) => match perish.duration {
                    None => continue,
                    Some(duration) => {
                        perish.duration = Some(duration - 1);
                        duration - 1
                    }
                },
            }
        };
        if left <= 0 {
            turn.faint(side, slot);
        }
    }

    // Residual order 28: Speed Boost.
    for (side, slot) in order.iter().copied() {
        let boosts = matches!(turn.mon_at(side, slot), Some(mon)
            if !mon.fainted && mon.ability == "speedboost" && !mon.newly_switched);
        if boosts {
            turn.apply_boosts(side, slot, &[("spe", 1)], false);
        }
    }

    check_white_herb(turn);

    for side in 0..2 {
        let mut kept: Vec<Effect> = Vec::new();
        for mut condition in std::mem::take(&mut turn.pos.sides[side].side_conditions) {
            if let Some(duration) = condition.duration {
                condition.duration = Some(duration - 1);
                if duration - 1 <= 0 {
                    continue;
                }
            }
            kept.push(condition);
        }
        turn.pos.sides[side].side_conditions = kept;
    }

    if let Some(duration) = turn.pos.field.terrain_duration {
        turn.pos.field.terrain_duration = Some(duration - 1);
        if duration - 1 <= 0 {
            turn.pos.field.terrain = None;
            turn.pos.field.terrain_duration = None;
        }
    }
    let mut kept_pseudo: Vec<Effect> = Vec::new();
    for mut pseudo in std::mem::take(&mut turn.pos.field.pseudo_weather) {
        if let Some(duration) = pseudo.duration {
            pseudo.duration = Some(duration - 1);
            if duration - 1 <= 0 {
                continue;
            }
        }
        kept_pseudo.push(pseudo);
    }
    turn.pos.field.pseudo_weather = kept_pseudo;

    for (side, slot) in order.iter().copied() {
        let failed = turn.move_failed[side][slot];
        let Some(mon) = turn.mon_at_mut(side, slot) else { continue };
        let mut kept: Vec<Effect> = Vec::new();
        let mut yawn_expired = false;
        for mut volatile in std::mem::take(&mut mon.volatiles) {
            let single_turn = PROTECT_VOLATILES.iter().any(|(v, _)| *v == volatile.id.as_str())
                || matches!(
                    volatile.id.as_str(),
                    "flinch" | "helpinghand" | "followme" | "ragepowder" | "spotlight"
                        | "glaiverush"
                );
            if single_turn {
                continue;
            }
            if volatile.id.as_str() == "perishsong" {
                kept.push(volatile);
                continue;
            }
            if let Some(duration) = volatile.duration {
                volatile.duration = Some(duration - 1);
                if duration - 1 <= 0 {
                    if volatile.id.as_str() == "yawn" {
                        yawn_expired = true;
                    }
                    if volatile.id.as_str() == "disable" {
                        // The flag lives on the move slot, so it has to be cleared here or
                        // the move stays unusable for the rest of the battle.
                        if let Some(move_id) = volatile.move_id {
                            if let Some(move_slot) = mon.moves.get_mut(move_id) {
                                move_slot.disabled = false;
                            }
                        }
                    }
                    continue;
                }
            }
            if volatile.id.as_str() == "encore" {
                // `onResidual`: an Encore whose move has run out of PP ends early.
                let spent = volatile
                    .move_id
                    .and_then(|id| mon.moves.get(id))
                    .map(|slot| slot.pp > 0)
                    .unwrap_or(false);
                if !spent {
                    continue;
                }
            }
            kept.push(volatile);
        }
        mon.volatiles = kept;
        mon.newly_switched = false;
        mon.move_last_turn_failed = failed;
        if yawn_expired {
            turn.apply_status(side, slot, "slp")?;
        }
    }

    settle_outcome(&mut turn.pos, &turn.wipe_order);
    Ok(())
}

pub fn settle_outcome(pos: &mut Position, wipe_order: &[usize]) {
    let wiped: Vec<usize> = (0..pos.sides.len())
        .filter(|index| pos.sides[*index].pokemon.iter().all(|m| m.fainted))
        .collect();
    if wiped.is_empty() {
        return;
    }
    pos.ended = true;
    if wiped.len() == 1 {
        pos.winner = Some(pos.sides[1 - wiped[0]].id.clone());
        return;
    }
    let ordered: Vec<usize> = wipe_order
        .iter()
        .copied()
        .filter(|index| wiped.contains(index))
        .collect();
    pos.winner = ordered.last().map(|index| pos.sides[*index].id.clone());
}

const BURN_DAMAGE: (i64, i64) = (1, 16);
const POISON_DAMAGE: (i64, i64) = (1, 8);
const SANDSTORM_DAMAGE: (i64, i64) = (1, 16);
const LEECH_SEED_DRAIN: (i64, i64) = (1, 8);
const LEFTOVERS_HEAL: (i64, i64) = (1, 16);
const PARTIAL_TRAP_DAMAGE: (i64, i64) = (1, 8);
const SALT_CURE_DAMAGE: (i64, i64) = (1, 8);
const SALT_CURE_DAMAGE_WEAK: (i64, i64) = (1, 4);
