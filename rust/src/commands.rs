//! What only Python's resolver used to answer, as node commands (IKA-211).
//!
//! `resolve` hands back a turn's weights and one chosen position. Four things a game needs
//! were still Python's alone, and each is a command here:
//!
//! * `turn` -- a turn with every branch (`full`), or one chosen outcome (`select`), which
//!   may be a *pause*: a turn a self-switching move stopped for a replacement. It also
//!   resumes a pause with both sides' choices, so the rest of the turn is the port's too.
//! * `alternatives` -- every replacement the interrupted side could send in, and the turn
//!   each produces (`resume_alternatives`), optionally in another completion of the
//!   position (`paused_in`, for a hidden bench).
//! * `replacements` -- the replacement phase after a turn (`resolve_replacements`).
//! * `leads` -- the leads' switch-ins before turn 1 (`apply_lead_abilities`).
//!
//! A pause crosses the pipe as data, not as an id into this process. The node holds
//! nothing between requests -- `rustnode.disable` answers a failure by starting a fresh
//! one, and that loses nothing only while there is nothing to lose -- so the continuation
//! (`Turn` and the rest of the queue) is written out whole and read back. Every `f64` in it
//! goes as its bit pattern, so a resumed turn is the same turn to the last bit, and the
//! writers destructure their structs exhaustively: a field added to `Turn` or
//! `QueuedAction` is a compile error here, never a field silently dropped at a pause.
//!
//! The draws keep Python's order: the caller is handed weights and samples with its own
//! generator, then asks again with the choice made. A replacement phase's Trace is the same
//! -- the caller's `presets` are replayed and the first draw past them comes back as
//! weights -- so a game played through these commands draws the same numbers from the same
//! generator as one played in Python.

use super::*;
use crate::moves::settle_outcome;
use serde_json::json;

/// The kinds this module answers.
pub fn handles(kind: Option<&str>) -> bool {
    matches!(kind, Some("turn" | "alternatives" | "replacements" | "leads" | "needed"))
}

pub fn answer(reg: &Reg, value: &Value) -> Value {
    let outcome = match value["kind"].as_str() {
        Some("turn") => turn_command(reg, value),
        Some("alternatives") => alternatives_command(reg, value),
        Some("replacements") => phase_command(reg, value, Phase::Replacements),
        Some("leads") => phase_command(reg, value, Phase::Leads),
        Some("needed") => needed_command(reg, value),
        _ => Err("unknown command".to_string()),
    };
    outcome.unwrap_or_else(|reason| json!({ "refused": reason }))
}

// ---------------------------------------------------------------------------
// A pause, written out and read back
// ---------------------------------------------------------------------------

fn bits(x: f64) -> Value {
    json!(format!("{:016x}", x.to_bits()))
}

fn from_bits(value: &Value) -> Result<f64, String> {
    let text = value.as_str().ok_or("a pause float is not a bit string")?;
    u64::from_str_radix(text, 16)
        .map(f64::from_bits)
        .map_err(|e| format!("a pause float does not parse: {e}"))
}

fn flags_json(flags: &[[bool; 2]; 2]) -> Value {
    json!([[flags[0][0], flags[0][1]], [flags[1][0], flags[1][1]]])
}

fn flags_from(value: &Value) -> Result<[[bool; 2]; 2], String> {
    let mut out = [[false; 2]; 2];
    for (side, row) in out.iter_mut().enumerate() {
        for (slot, flag) in row.iter_mut().enumerate() {
            *flag = value[side][slot].as_bool().ok_or("a pause flag is missing")?;
        }
    }
    Ok(out)
}

fn slot_json(slot: &Slot) -> Value {
    json!([slot.0, slot.1])
}

fn slot_from(value: &Value) -> Result<Slot, String> {
    let side = value[0].as_u64().ok_or("a pause slot is missing")? as usize;
    let slot = value[1].as_u64().ok_or("a pause slot is missing")? as usize;
    Ok((side, slot))
}

fn budget_json(budget: &Budget) -> Value {
    let Budget {
        damage_rolls,
        enumerate_crit,
        enumerate_accuracy,
        enumerate_status_checks,
        enumerate_secondary,
        enumerate_speed_ties,
        pinned_policy,
        max_branches,
        merge_duplicates,
    } = budget;
    json!({
        "damageRolls": damage_rolls,
        "enumerateCrit": enumerate_crit,
        "enumerateAccuracy": enumerate_accuracy,
        "enumerateStatusChecks": enumerate_status_checks,
        "enumerateSecondary": enumerate_secondary,
        "enumerateSpeedTies": enumerate_speed_ties,
        "pinnedPolicy": pinned_policy,
        "maxBranches": max_branches,
        "mergeDuplicates": merge_duplicates,
    })
}

fn kind_name(kind: ActionKind) -> &'static str {
    match kind {
        ActionKind::Switch => "switch",
        ActionKind::Mega => "mega",
        ActionKind::Move => "move",
    }
}

fn queued_json(queued: &QueuedAction) -> Value {
    let QueuedAction {
        side,
        slot,
        kind,
        order,
        priority,
        fractional,
        speed,
        move_id,
        target,
        switch_to,
        switch_species,
        branch_probability,
    } = queued;
    json!({
        "side": side,
        "slot": slot,
        "kind": kind_name(*kind),
        "order": order,
        "priority": priority,
        "fractional": bits(*fractional),
        "speed": speed,
        "moveId": move_id.map(|m| m.as_str().to_string()),
        "target": target,
        "switchTo": switch_to,
        "switchSpecies": switch_species.map(|s| s.as_str().to_string()),
        "branchProbability": bits(*branch_probability),
    })
}

fn queued_from(value: &Value) -> Result<QueuedAction, String> {
    let int = |key: &str| value[key].as_i64().ok_or(format!("a queued action has no {key}"));
    let id = |key: &str| value[key].as_str().map(Id::new);
    Ok(QueuedAction {
        side: int("side")? as usize,
        slot: int("slot")? as usize,
        kind: match value["kind"].as_str() {
            Some("switch") => ActionKind::Switch,
            Some("mega") => ActionKind::Mega,
            Some("move") => ActionKind::Move,
            other => return Err(format!("a queued action of kind {other:?}")),
        },
        order: int("order")?,
        priority: int("priority")?,
        fractional: from_bits(&value["fractional"])?,
        speed: int("speed")?,
        move_id: id("moveId"),
        target: value["target"].as_i64(),
        switch_to: value["switchTo"].as_u64().map(|v| v as usize),
        switch_species: id("switchSpecies"),
        branch_probability: from_bits(&value["branchProbability"])?,
    })
}

/// The continuation of one pause, without its position (which travels beside it).
fn turn_state_json(turn: &Turn) -> Result<Value, String> {
    let Turn {
        reg: _,
        pos: _,
        budget,
        attacks,
        hurt_this_turn,
        move_failed,
        move_damage_total,
        move_connected,
        acted,
        actions_remaining,
        self_switch_pending,
        pending_secondaries,
        current_actor,
        wipe_order,
        unmodelled,
        rolls_stratified,
        move_start_hp,
        move_hit,
        draws,
        log,
        damaged_by,
    } = turn;
    // Both are set and cleared inside one action, and a pause is taken between actions.
    // Either one here would mean a pause taken somewhere this module does not expect.
    if draws.is_some() {
        return Err("a pause taken inside a switch-in's draws".into());
    }
    if *rolls_stratified {
        return Err("a pause taken before its stratified roll was booked".into());
    }
    Ok(json!({
        "budget": budget_json(budget),
        "attacks": flags_json(attacks),
        "hurtThisTurn": flags_json(hurt_this_turn),
        "moveFailed": flags_json(move_failed),
        "moveDamageTotal": move_damage_total,
        "moveConnected": move_connected,
        "acted": flags_json(acted),
        "actionsRemaining": actions_remaining,
        "selfSwitchPending": self_switch_pending,
        "pendingSecondaries": pending_secondaries
            .iter()
            .map(|(chance, secondary, target)| json!([bits(*chance), secondary, slot_json(target)]))
            .collect::<Vec<_>>(),
        "currentActor": current_actor.as_ref().map(slot_json),
        "wipeOrder": wipe_order,
        "unmodelled": unmodelled.iter().cloned().collect::<Vec<_>>(),
        "moveStartHp": move_start_hp.map(|hp| json!([[hp[0][0], hp[0][1]], [hp[1][0], hp[1][1]]])),
        "moveHit": flags_json(move_hit),
        "log": log.as_deref().map(log_json),
        "damagedBy": crate::damage_callback::to_json(damaged_by),
    }))
}

/// A trace as the caller reads it: Python's `events` and `acts` (IKA-215).
fn log_json(log: &EventLog) -> Value {
    json!({
        "events": log.events,
        "acts": log.acts.iter().map(|(start, label)| json!([start, label])).collect::<Vec<_>>(),
    })
}

fn log_from(value: &Value) -> Result<Option<Box<EventLog>>, String> {
    if value.is_null() {
        return Ok(None);
    }
    let events = value["events"]
        .as_array()
        .ok_or("a pause log has no events")?
        .iter()
        .map(|line| line.as_str().map(String::from).ok_or("a pause event is not a string"))
        .collect::<Result<Vec<_>, _>>()?;
    let acts = value["acts"]
        .as_array()
        .ok_or("a pause log has no acts")?
        .iter()
        .map(|entry| match (entry[0].as_u64(), entry[1].as_str()) {
            (Some(start), Some(label)) => Ok((start as usize, label.to_string())),
            _ => Err("a pause act does not parse"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(Box::new(EventLog { events, acts })))
}

/// Whether a request asks for the trace.
fn wants_events(value: &Value) -> bool {
    value["events"].as_bool().unwrap_or(false)
}

/// Puts `log` into the answer for one outcome, when there is one.
fn with_log(mut out: Value, log: Option<&EventLog>) -> Value {
    if let Some(log) = log {
        out["events"] = json!(log.events);
        out["acts"] = log_json(log)["acts"].clone();
    }
    out
}

fn turn_from<'a>(reg: &'a Reg, pos: Position, state: &Value) -> Result<Turn<'a>, String> {
    let mut turn = Turn::new(reg, pos, Budget::from_json(&state["budget"]), flags_from(&state["attacks"])?);
    turn.hurt_this_turn = flags_from(&state["hurtThisTurn"])?;
    turn.damaged_by = crate::damage_callback::from_json(&state["damagedBy"])?;
    turn.move_failed = flags_from(&state["moveFailed"])?;
    turn.move_damage_total = state["moveDamageTotal"].as_i64().ok_or("moveDamageTotal")?;
    turn.move_connected = state["moveConnected"].as_bool().ok_or("moveConnected")?;
    turn.acted = flags_from(&state["acted"])?;
    turn.actions_remaining = state["actionsRemaining"].as_u64().ok_or("actionsRemaining")? as usize;
    turn.self_switch_pending = state["selfSwitchPending"].as_bool().ok_or("selfSwitchPending")?;
    for entry in state["pendingSecondaries"].as_array().ok_or("pendingSecondaries")? {
        turn.pending_secondaries.push((from_bits(&entry[0])?, entry[1].clone(), slot_from(&entry[2])?));
    }
    turn.current_actor = match &state["currentActor"] {
        Value::Null => None,
        slot => Some(slot_from(slot)?),
    };
    turn.wipe_order = state["wipeOrder"]
        .as_array()
        .ok_or("wipeOrder")?
        .iter()
        .filter_map(Value::as_u64)
        .map(|v| v as usize)
        .collect();
    turn.unmodelled = state["unmodelled"]
        .as_array()
        .ok_or("unmodelled")?
        .iter()
        .filter_map(Value::as_str)
        .map(String::from)
        .collect();
    turn.move_start_hp = match &state["moveStartHp"] {
        Value::Null => None,
        hp => {
            let mut out = [[0i64; 2]; 2];
            for (side, row) in out.iter_mut().enumerate() {
                for (slot, cell) in row.iter_mut().enumerate() {
                    *cell = hp[side][slot].as_i64().ok_or("moveStartHp")?;
                }
            }
            Some(out)
        }
    };
    turn.move_hit = flags_from(&state["moveHit"])?;
    turn.log = log_from(&state["log"])?;
    Ok(turn)
}

/// A pause as the caller holds it: its weight and position to read, and the state to hand
/// back. `position` is the pause's own and is not repeated inside `state`.
fn pause_json(pause: &Suspended) -> Result<Value, String> {
    Ok(with_log(
        json!({
            "probability": pause.probability,
            "position": pause.turn.pos.to_json(),
            "state": {
                "format": reg_format(pause.turn.reg),
                "probability": bits(pause.probability),
                "turn": turn_state_json(&pause.turn)?,
                "remaining": pause.remaining.iter().map(queued_json).collect::<Vec<_>>(),
            },
        }),
        pause.turn.log.as_deref(),
    ))
}

fn reg_format(reg: &Reg) -> &str {
    reg.format_id.as_str()
}

fn pause_from<'a>(reg: &'a Reg, entry: &Value) -> Result<Suspended<'a>, String> {
    let state = &entry["state"];
    if state["format"].as_str() != Some(reg.format_id.as_str()) {
        return Err("the pause is for another regulation".into());
    }
    let position = Position::from_json(&entry["position"]);
    let turn = turn_from(reg, position, &state["turn"])?;
    let remaining = state["remaining"]
        .as_array()
        .ok_or("remaining")?
        .iter()
        .map(queued_from)
        .collect::<Result<Vec<_>, _>>()?;
    // The positive control (`--features ika211-control`): the rest of the queue loses its
    // last action, which is the continuation this module exists to carry.
    #[cfg(feature = "ika211-control")]
    let remaining = {
        let mut remaining = remaining;
        remaining.pop();
        remaining
    };
    Ok(Suspended { probability: from_bits(&state["probability"])?, turn, remaining })
}

/// Python's `_find_switch_target`: by species identity, then by index.
fn switch_target<'p>(side: &'p crate::position::Side, queued: &QueuedAction) -> Option<&'p Pokemon> {
    if let Some(species) = queued.switch_species {
        if let Some(mon) = side
            .pokemon
            .iter()
            .find(|mon| crate::transform::switch_names(mon, species))
        {
            return Some(mon);
        }
    }
    queued.switch_to.and_then(|index| side.pokemon.get(index)).map(|mon| &**mon)
}

/// Python's `paused_in`: the same pause with `position` in place of its own, a queued
/// switch of `side`'s re-aimed by slot at whoever stands there in `position`.
///
/// For the self-switch node under a hidden bench (IKA-120): `position` is a completion of
/// the pause's own position -- `side`'s unseen slots rebuilt from the sheet, everything else
/// the pause's -- and resuming the result is resuming the turn in that world. An unseen
/// Pokemon took no part in the turn up to the interrupt, so nothing else in the continuation
/// refers to it except a queued switch into its slot, which names its target by species
/// first -- and the true species would find nobody.
fn paused_in<'a>(paused: Suspended<'a>, position: Position, side: usize) -> Suspended<'a> {
    let Suspended { probability, mut turn, remaining } = paused;
    let before = turn.pos.sides[side].clone();
    let after = position.sides[side].clone();
    turn.pos = position;
    let remaining = remaining
        .into_iter()
        .map(|mut queued| {
            if queued.side == side && queued.kind == ActionKind::Switch {
                if let Some(target) = switch_target(&before, &queued) {
                    if let Some(standing) = after.pokemon.iter().find(|mon| mon.slot == target.slot) {
                        if standing.species != target.species {
                            queued.switch_species = Some(standing.species);
                        }
                    }
                }
            }
            queued
        })
        .collect();
    Suspended { probability, turn, remaining }
}

/// The pause a request names, rebuilt in another completion when it says `in`. It keeps
/// its trace when the request asks for events, and starts one if it carried none.
fn requested_pause<'a>(reg: &'a Reg, value: &Value) -> Result<Suspended<'a>, String> {
    let mut pause = pause_from(reg, &value["pause"])?;
    if !wants_events(value) {
        pause.turn.log = None;
    } else if pause.turn.log.is_none() {
        pause.turn.log = Some(Box::default());
    }
    match value.get("in") {
        None | Some(Value::Null) => Ok(pause),
        Some(world) => {
            let position = Position::from_json(&world["position"]);
            let side = world["side"].as_u64().ok_or("`in` names no side")? as usize;
            Ok(paused_in(pause, position, side))
        }
    }
}

// ---------------------------------------------------------------------------
// Turns
// ---------------------------------------------------------------------------

fn result_json(result: &TurnResult, full: bool) -> Result<Value, String> {
    let unmodelled: Vec<String> = result.unmodelled.iter().cloned().collect();
    if !full {
        return Ok(json!({
            "branches": result.branches.iter().map(|b| b.probability).collect::<Vec<_>>(),
            "suspended": result.suspended.iter().map(|s| s.probability).collect::<Vec<_>>(),
            "exact": result.exact,
            "unmodelled": unmodelled,
        }));
    }
    Ok(json!({
        "branches": result
            .branches
            .iter()
            .map(|b| {
                with_log(
                    json!({ "probability": b.probability, "position": b.position.to_json() }),
                    b.log.as_deref(),
                )
            })
            .collect::<Vec<_>>(),
        "suspended": result.suspended.iter().map(pause_json).collect::<Result<Vec<_>, _>>()?,
        "exact": result.exact,
        "unmodelled": unmodelled,
    }))
}

fn two_sides(value: &Value) -> [Vec<SlotAction>; 2] {
    [parse_actions_list(&value[0]), parse_actions_list(&value[1])]
}

/// A turn from a position and both sides' actions, or the rest of a paused one.
///
/// `select` is an index into the branches followed by the pauses, as the caller sampled
/// it from the weights: a branch comes back as `position`, a pause as `pause`.
fn turn_command(reg: &Reg, value: &Value) -> Result<Value, String> {
    let result = if value.get("pause").is_some_and(|p| !p.is_null()) {
        let pause = requested_pause(reg, value)?;
        resume_turn(reg, &pause, &two_sides(&value["choices"]))?
    } else {
        let position = Position::from_json(&value["position"]);
        if &*position.format != reg.format_id.as_str() {
            return Err("position is for another regulation".into());
        }
        let budget = Budget::from_json(&value["budget"]);
        resolve_turn_logged(reg, &position, &two_sides(&value["actions"]), budget, wants_events(value))?
    };
    let full = value["full"].as_bool().unwrap_or(false);
    let mut out = result_json(&result, full)?;
    if let Some(index) = value.get("select").and_then(Value::as_u64).map(|k| k as usize) {
        let count = result.branches.len();
        if index < count {
            out["position"] = result.branches[index].position.to_json();
            out = with_log(out, result.branches[index].log.as_deref());
        } else {
            match result.suspended.get(index - count) {
                None => return Err("branch index out of range".into()),
                Some(pause) => out["pause"] = pause_json(pause)?,
            }
        }
    }
    Ok(out)
}

fn slot_action_json(action: &SlotAction) -> Value {
    match action {
        SlotAction::Move { slot, move_id, target, mega } => json!({
            "kind": "move", "slot": slot, "moveId": move_id.as_str(), "target": target, "mega": mega,
        }),
        SlotAction::Switch { slot, party_index, species } => json!({
            "kind": "switch", "slot": slot, "partyIndex": party_index, "species": species.as_str(),
        }),
        SlotAction::Pass { slot } => json!({ "kind": "pass", "slot": slot }),
    }
}

/// `resume_alternatives`, with the options it resumed: the interrupted side's every
/// replacement and the turn each produces, whole.
fn alternatives_command(reg: &Reg, value: &Value) -> Result<Value, String> {
    let paused = requested_pause(reg, value)?;
    let owed = self_switches_needed(&paused.turn.pos);
    let sides: Vec<usize> = (0..2).filter(|i| owed[*i].iter().any(|f| *f)).collect();
    let Some(&chooser) = sides.first() else {
        return Ok(json!({ "chooser": null, "options": [], "results": [] }));
    };
    let other = 1 - chooser;
    let passes: Vec<SlotAction> = (0..paused.turn.pos.sides[other].active.len())
        .map(|slot| SlotAction::Pass { slot })
        .collect();
    let full = value["full"].as_bool().unwrap_or(true);
    let mut options = Vec::new();
    let mut results = Vec::new();
    for option in replacement_options(&paused.turn.pos, chooser, owed[chooser]) {
        let choices = if chooser == 0 {
            [option.clone(), passes.clone()]
        } else {
            [passes.clone(), option.clone()]
        };
        let mut resumed = resume_turn(reg, &paused, &choices)?;
        if sides.len() > 1 {
            resumed.unmodelled.insert("simultaneous mid-turn replacements".into());
        }
        options.push(option.iter().map(slot_action_json).collect::<Vec<_>>());
        results.push(result_json(&resumed, full)?);
    }
    Ok(json!({ "chooser": chooser, "options": options, "results": results }))
}

// ---------------------------------------------------------------------------
// The replacement phase and the leads
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Replacements,
    Leads,
}

/// `PENDING_REPLACEMENT_VOLATILES`.
const PENDING_REPLACEMENT: [&str; 2] = ["pendingselfswitch", "pendingforceswitch"];

/// Python's `replacements_needed`: per side, per active slot, whether the player owes a
/// replacement -- a faint, a self-switch or a forced switch, and something on the bench.
pub fn replacements_needed(pos: &Position) -> Vec<Vec<bool>> {
    pos.sides
        .iter()
        .map(|side| {
            let bench = side.pokemon.iter().filter(|m| !m.fainted && !m.is_active()).count();
            (0..side.active.len())
                .map(|slot| match side.active_pokemon(slot) {
                    None => bench > 0,
                    Some(mon) => {
                        let pending = PENDING_REPLACEMENT.iter().any(|v| mon.has_volatile(v));
                        bench > 0 && (mon.fainted || pending)
                    }
                })
                .collect()
        })
        .collect()
}

/// Showdown's `forcedPassesLeft` (`side.ts` `clearChoice`): of the slots `side` owes, how
/// many it may leave empty, `canSwitchOut - min(canSwitchOut, canSwitchIn)`. Two faints
/// with one Pokemon left is `forceSwitch: [true, true]` and one pass (`battle.ts` keeps both
/// `switchFlag`s while `canSwitch` is not 0), so `replacements_needed` stays per slot and the
/// shortfall is counted here (IKA-257).
fn forced_passes(pos: &Position, side: usize, needed: &[bool]) -> usize {
    let bench = pos.sides[side].pokemon.iter().filter(|m| !m.fainted && !m.is_active()).count();
    needed.iter().filter(|owed| **owed).count().saturating_sub(bench)
}

fn needed_command(reg: &Reg, value: &Value) -> Result<Value, String> {
    let position = Position::from_json(&value["position"]);
    if &*position.format != reg.format_id.as_str() {
        return Err("position is for another regulation".into());
    }
    Ok(json!({ "needed": replacements_needed(&position) }))
}

/// `Budget.deterministic(0)`, which both phases run under in Python.
fn deterministic() -> Budget {
    Budget {
        damage_rolls: -1,
        enumerate_crit: false,
        enumerate_accuracy: false,
        enumerate_status_checks: false,
        enumerate_secondary: false,
        enumerate_speed_ties: false,
        pinned_policy: true,
        max_branches: 1,
        merge_duplicates: true,
    }
}

/// `resolve_replacements` and `apply_lead_abilities`: one position, from Python's same
/// steps -- place every incoming Pokemon, then run the switch-ins fastest first.
///
/// `presets` absent: a draw takes its first option and is noted, as Python's with no
/// generator. `presets` a list: those are the choices already drawn, and the first draw
/// past them comes back as `draw` (its weights) for the caller to sample and ask again.
///
/// The replacement phase: a slot that owes nothing carries a pass, and a slot that owes a
/// replacement and is given a pass is left alone and reported, because silently choosing
/// for the player is the thing the resolver exists not to do -- unless the bench is too short
/// to fill it, which is Showdown's own pass (`forced_passes`). `runSwitch` carries order 101
/// and is sorted on speed, fastest first, so a fast replacement eats the hazards and fires
/// its ability before a slow one.
///
/// The leads: a freshly built turn-1 position has had nothing applied to it -- no
/// Intimidate, no Defiant answering it, no weather from a lead's ability -- and Showdown has
/// done all of that before the first request goes out (`|turn|1`), so a position without it
/// is not the position the game starts from. Speed-ordered like the replacement phase and
/// through the same switch-in, so hazards and White Herb behave identically should a caller
/// hand it a position that has them.
fn phase_command(reg: &Reg, value: &Value, phase: Phase) -> Result<Value, String> {
    let position = Position::from_json(&value["position"]);
    if &*position.format != reg.format_id.as_str() {
        return Err("position is for another regulation".into());
    }
    let choices = if phase == Phase::Replacements { two_sides(&value["choices"]) } else { [Vec::new(), Vec::new()] };
    check_position_supported(&position, &choices)?;
    let mut state = Turn::new(reg, position, deterministic(), [[false; 2]; 2]);
    if wants_events(value) {
        state.log = Some(Box::default());
    }
    state.draws = Some(match value.get("presets").and_then(Value::as_array) {
        None => Draws { report: true, ..Default::default() },
        Some(listed) => Draws {
            replay: true,
            // The positive control answers every draw with its first option.
            presets: if cfg!(feature = "ika211-control") {
                vec![0; listed.len()]
            } else {
                listed.iter().filter_map(Value::as_u64).map(|v| v as usize).collect()
            },
            report: true,
            ..Default::default()
        },
    });
    let mut notes: std::collections::BTreeSet<String> = Default::default();
    let mut placed: Vec<(i64, usize, usize)> = Vec::new();
    match phase {
        Phase::Replacements => {
            let needed = replacements_needed(&state.pos);
            for (side_index, actions) in choices.iter().enumerate() {
                let mut passes_left = forced_passes(&state.pos, side_index, &needed[side_index]);
                for action in actions {
                    match action {
                        SlotAction::Pass { slot } => {
                            if needed[side_index].get(*slot).copied().unwrap_or(false) {
                                // A pass the bench cannot fill is Showdown's own (IKA-257).
                                if passes_left > 0 {
                                    passes_left -= 1;
                                    continue;
                                }
                                notes.insert(format!(
                                    "replacement owed at p{}[{}] but none was chosen",
                                    side_index + 1,
                                    slot
                                ));
                            }
                        }
                        SlotAction::Move { .. } => {
                            return Err("a replacement phase only takes switches".into());
                        }
                        SlotAction::Switch { slot, party_index, species } => {
                            if let Some(outgoing) = state.mon_at_mut(side_index, *slot) {
                                outgoing
                                    .volatiles
                                    .retain(|v| !PENDING_REPLACEMENT.contains(&v.id.as_str()));
                            }
                            let queued = QueuedAction {
                                side: side_index,
                                slot: *slot,
                                kind: ActionKind::Switch,
                                order: ORDER_SWITCH,
                                priority: 0,
                                fractional: 0.0,
                                speed: 0,
                                move_id: None,
                                target: None,
                                switch_to: Some(party_index - 1),
                                switch_species: Some(*species),
                                branch_probability: 1.0,
                            };
                            // The positive control runs each switch-in as it is placed.
                            if cfg!(feature = "ika211-control") {
                                do_switch_with(reg, &mut state, &queued, true)?;
                                continue;
                            }
                            do_switch_with(reg, &mut state, &queued, false)?;
                            placed.push((phase_speed(&state, side_index, *slot)?, side_index, *slot));
                        }
                    }
                }
            }
        }
        Phase::Leads => {
            for side_index in 0..state.pos.sides.len() {
                for slot in 0..state.pos.sides[side_index].active.len() {
                    if state.pos.sides[side_index].active[slot].is_none() {
                        continue;
                    }
                    placed.push((phase_speed(&state, side_index, slot)?, side_index, slot));
                }
            }
        }
    }
    // The positive control leaves the switch-ins in the order they were placed.
    #[cfg(not(feature = "ika211-control"))]
    placed.sort_by_key(|(speed, side, slot)| (-speed, *side, *slot));
    for (_speed, side_index, slot) in &placed {
        on_switch_in(reg, &mut state, *side_index, *slot)?;
    }
    let opened = state.draws.take().map(|d| d.opened).unwrap_or_default();
    if phase == Phase::Replacements {
        let wipe_order = state.wipe_order.clone();
        settle_outcome(&mut state.pos, &wipe_order);
        for side in state.pos.sides.iter_mut() {
            for mon in side.pokemon.iter_mut() {
                if mon.trapped {
                    std::rc::Rc::make_mut(mon).trapped = false;
                }
            }
        }
    }
    notes.extend(state.unmodelled.iter().cloned());
    let mut out = json!({
        "position": state.pos.to_json(),
        "unmodelled": notes.into_iter().collect::<Vec<_>>(),
        "draw": opened.first(),
    });
    if let Some(log) = state.log.as_deref() {
        out["events"] = json!(log.events);
    }
    Ok(out)
}

fn phase_speed(state: &Turn, side: usize, slot: usize) -> Result<i64, String> {
    Ok(match state.battler_at(side, slot)? {
        None => 0,
        Some(incoming) => {
            let field = state.field();
            effective_speed(&incoming, &field, &state.pos.sides[side].side_conditions)
        }
    })
}

// ---------------------------------------------------------------------------
// A pause's replacements, their turns flattened and encoded (IKA-209)
// ---------------------------------------------------------------------------

/// `alternatives_command`'s options and resumed turns, without the JSON.
fn alternatives_of<'a>(
    reg: &'a Reg,
    paused: &Suspended<'a>,
) -> Result<Option<(usize, Vec<(Vec<SlotAction>, TurnResult<'a>)>)>, String> {
    let owed = self_switches_needed(&paused.turn.pos);
    let sides: Vec<usize> = (0..2).filter(|i| owed[*i].iter().any(|f| *f)).collect();
    let Some(&chooser) = sides.first() else {
        return Ok(None);
    };
    let other = 1 - chooser;
    let passes: Vec<SlotAction> = (0..paused.turn.pos.sides[other].active.len())
        .map(|slot| SlotAction::Pass { slot })
        .collect();
    let mut out = Vec::new();
    for option in replacement_options(&paused.turn.pos, chooser, owed[chooser]) {
        let choices = if chooser == 0 {
            [option.clone(), passes.clone()]
        } else {
            [passes.clone(), option.clone()]
        };
        let mut resumed = resume_turn(reg, paused, &choices)?;
        if sides.len() > 1 {
            resumed.unmodelled.insert("simultaneous mid-turn replacements".into());
        }
        out.push((option, resumed));
    }
    Ok(Some((chooser, out)))
}

/// Python's `port.turn_leaves` over the port's own turn: every leaf in the same order (no
/// sharing -- the caller scores them in one batch beside other plans, and the rows must be
/// the rows it would have had), the fold with indices local to `leaves`.
fn flatten<'a>(
    reg: &'a Reg,
    result: &TurnResult<'a>,
    depth: usize,
    leaves: &mut Vec<Position>,
    notes: &mut std::collections::BTreeSet<String>,
) -> Result<Value, String> {
    notes.extend(result.unmodelled.iter().cloned());
    let mut parts: Vec<Value> = Vec::new();
    for branch in &result.branches {
        leaves.push(branch.position.clone());
        parts.push(json!([branch.probability, { "leaf": leaves.len() - 1 }]));
    }
    for pause in &result.suspended {
        if depth >= 4 {
            notes.insert("more than four mid-turn replacements in one turn".into());
            leaves.push(pause.turn.pos.clone());
            parts.push(json!([pause.probability, { "leaf": leaves.len() - 1 }]));
            continue;
        }
        match alternatives_of(reg, pause)? {
            Some((chooser, resumed)) if !resumed.is_empty() => {
                let mut options = Vec::new();
                for (_option, one) in &resumed {
                    options.push(flatten(reg, one, depth + 1, leaves, notes)?);
                }
                parts.push(json!([pause.probability, { "best": chooser, "options": options }]));
            }
            _ => {
                notes.insert("a suspended turn offered no replacement".into());
                leaves.push(pause.turn.pos.clone());
                parts.push(json!([pause.probability, { "leaf": leaves.len() - 1 }]));
            }
        }
    }
    // The positive control (`--features ika209-control`): the first part of every fold
    // weighs half, which is a plan valued wrong while every leaf is right.
    #[cfg(feature = "ika209-control")]
    if let Some(first) = parts.first_mut() {
        first[0] = json!(first[0].as_f64().unwrap_or(0.0) * 0.5);
    }
    Ok(json!({ "avg": parts }))
}

/// `alternativesEncoded`: the `alternatives` of a pause (in `in`'s completion when given),
/// each wanted option's turn flattened as `turn_leaves` flattens it and every leaf encoded
/// -- the self-switch node's leaves as arrays instead of positions (IKA-209).
///
/// `want` lists the options to flatten (default all). `shared: {side, slots}` also answers,
/// per option, what `_shared_self_switch_plans` asks of it: that its rest of the turn does
/// not pause again and leaves `side`'s Pokemon at `slots` as the pause held them, off the
/// field -- or false for every option when `side` still has a switch queued.
pub fn alternatives_encoded(
    reg: &Reg,
    encoder: &crate::encode::Encoder,
    value: &Value,
) -> Result<(Value, crate::encode::Encoded, Vec<f64>), String> {
    let paused = requested_pause(reg, value)?;
    let rules = crate::encode::EncodeRules {
        mega_from_slots: value
            .get("encoding")
            .and_then(|e| e.get("megaFromSlots"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    let found = alternatives_of(reg, &paused)?;
    let (chooser, resumed) = match found {
        None => (None, Vec::new()),
        Some((chooser, resumed)) => (Some(chooser), resumed),
    };
    let wanted: Option<Vec<usize>> = value.get("want").and_then(Value::as_array).map(|list| {
        list.iter().filter_map(Value::as_u64).map(|k| k as usize).collect()
    });
    let shared = value.get("shared").filter(|s| !s.is_null()).map(|s| {
        let side = s["side"].as_u64().unwrap_or(0) as usize;
        let slots: Vec<usize> = s["slots"]
            .as_array()
            .map(|l| l.iter().filter_map(Value::as_u64).map(|k| k as usize).collect())
            .unwrap_or_default();
        (side, slots)
    });
    let queued_switch = shared.as_ref().is_some_and(|(side, _)| {
        paused
            .remaining
            .iter()
            .any(|queued| queued.side == *side && queued.kind == ActionKind::Switch)
    });

    let mut leaves: Vec<Position> = Vec::new();
    let mut options = Vec::new();
    let mut plans = Vec::new();
    for (k, (option, one)) in resumed.iter().enumerate() {
        options.push(option.iter().map(slot_action_json).collect::<Vec<_>>());
        let untouched = shared.as_ref().map(|(side, slots)| {
            if queued_switch || !one.suspended.is_empty() {
                return false;
            }
            let before = &paused.turn.pos.sides[*side].pokemon;
            one.branches.iter().all(|branch| {
                before.iter().enumerate().filter(|(_, mon)| slots.contains(&mon.slot)).all(
                    |(index, mon)| {
                        branch.position.sides[*side].pokemon.get(index).is_some_and(|leaf| {
                            **leaf == **mon && leaf.active_index.is_none()
                        })
                    },
                )
            })
        });
        if wanted.as_ref().is_some_and(|w| !w.contains(&k)) {
            plans.push(json!({ "untouched": untouched }));
            continue;
        }
        let start = leaves.len();
        let mut notes = std::collections::BTreeSet::new();
        let root = flatten(reg, one, 0, &mut leaves, &mut notes)?;
        // Local to the plan, as `turn_leaves` numbers its own positions.
        let root = shift_fold(root, start);
        plans.push(json!({
            "start": start,
            "count": leaves.len() - start,
            "fold": root,
            "unmodelled": notes.into_iter().collect::<Vec<_>>(),
            "suspended": !one.suspended.is_empty(),
            "untouched": untouched,
        }));
    }
    // A parameter-free objective is scored here per leaf, and with `encode: false` the
    // arrays are not built at all: an hp-share node needs the values and nothing else.
    let objectives: Vec<String> = value["objectives"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect())
        .unwrap_or_default();
    let mut leaf_values: Vec<f64> = Vec::new();
    for name in &objectives {
        let score = crate::objective::by_name(name)
            .ok_or_else(|| format!("objective not available in the port: {name}"))?;
        leaf_values.extend(leaves.iter().map(score));
    }
    let encode = value.get("encode").and_then(Value::as_bool).unwrap_or(true);
    let borrowed: Vec<&Position> =
        if encode { leaves.iter().collect() } else { Vec::new() };
    let encoded = encoder.encode_positions_with(&borrowed, rules);
    let body_bytes = leaf_values.len() * 8
        + (encoded.species.len()
        + encoded.ability.len()
        + encoded.item.len()
        + encoded.moves.len()
        + encoded.mon.len()
        + encoded.mask.len()
        + encoded.side.len()
        + encoded.field.len())
        * 4;
    let header = json!({
        "kind": "alternativesEncoded",
        "chooser": chooser,
        "options": options,
        "plans": plans,
        "leaves": borrowed.len(),
        "valueRows": leaves.len(),
        "spans": [],
        "folded": [],
        "exact": [],
        "refused": [],
        "unmodelled": [],
        "unknownVolatiles": encoded.unknown_volatiles,
        "leafObjectives": objectives,
        "encoding": { "megaFromSlots": rules.mega_from_slots },
        "monsPerSide": encoder.widths.mons_per_side,
        "monWidth": encoder.widths.mon,
        "sideWidth": encoder.widths.side,
        "fieldWidth": encoder.widths.field,
        "bytes": body_bytes,
    });
    Ok((header, encoded, leaf_values))
}

/// A fold's leaf indices less `start`: the plan's leaves are numbered from its own first.
fn shift_fold(node: Value, start: usize) -> Value {
    if let Some(index) = node.get("leaf").and_then(Value::as_u64) {
        return json!({ "leaf": index as usize - start });
    }
    if let Some(options) = node.get("options").and_then(Value::as_array) {
        return json!({
            "best": node["best"],
            "options": options.iter().cloned().map(|o| shift_fold(o, start)).collect::<Vec<_>>(),
        });
    }
    let parts = node["avg"].as_array().cloned().unwrap_or_default();
    json!({
        "avg": parts
            .into_iter()
            .map(|part| json!([part[0], shift_fold(part[1].clone(), start)]))
            .collect::<Vec<_>>()
    })
}
