//! IKA-458 stage 3.6b: a game's turns played in one request.
//!
//! `selfplay.play_game`'s loop, for the shape generation plays (a hidden bench, one leaf for both
//! sides, the Q's menus, depth 1, the LP's strategy): each step the loop took -- what each side
//! has shown and the positions its unseen slots could hold (`hiddenStep`), the replacement phase
//! if one is owed (`replacementPhase`), else the move node (`moveNode`: both menus and the node
//! over them), the two draws, and the turn played on to its end (`advanceTurn`, a drawn pause
//! carried on node by node) -- is made here, one after another, from one generator whose state
//! the caller handed over. The pieces are the ones the stages below this made; what the loop
//! adds is that a step's answer feeds the next step in this process: the completions `hiddenStep`
//! made are the ones the move node and the replacement phase read (one decision's completions are
//! made once), and the menus' and the node's inputs are typed (`MenusArgs`, `menus_core`) where
//! they were a request Python wrote and the port read.
//!
//! ```text
//! { "kind": "playGame", "position": P, "rng": {state, inc, hasUint32, uinteger},
//!   "context": { servers, leaf, hidden: { game, seen, leads } },   // as `advanceTurn`'s `selfSwitch`
//!   "node": { widths, properties, megaFromSlots, qServer, blas,    // `rootMenus`'
//!             budget, server, netScoresEnds, patch, encoding? },   // `beliefNode`'s
//!   "exactBudget": B, "steps": n }
//! ```
//!
//! What comes back is the steps taken, each as the facts the caller records (its `Decision`'s
//! fields, the notes, the server's work), the position and the generator's state after them, and
//! why it stopped: the game ended, the steps ran out, the turn had no outcome (`unresolved`), or a
//! step is one the port does not do the way Python does (`handoff`, with the reason). A handoff
//! is made before the step draws anything or records anything, so the caller plays that step
//! itself from the same state, its generator exactly where the steps before left it.

use crate::advance::{advance_core, choice, Pcg64};
use crate::encode::Encoder;
use crate::hiddenbench::{step_of, Item};
use crate::legal;
use crate::lp::{b64_encode, f64s, numpy_sum};
use crate::position::Position;
use crate::qmenus::{menus_core, move_node_over, MenusArgs};
use crate::reg::Reg;
use crate::resolve::commands::replacements_needed;
use crate::resolve::{resolve_turn, Budget, SlotAction};
use crate::selfswitch::Context;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The move nodes this process played in a game's loop, the positive control of the stage.
pub static MOVES: AtomicU64 = AtomicU64::new(0);

pub fn play_game(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    match run(reg, encoder, value) {
        Ok(answer) | Err(answer) => answer,
    }
}

/// Why a step is left to the caller, and the answer that says so.
struct Handoff(String);

impl Handoff {
    fn of(why: impl Into<String>) -> Handoff {
        Handoff(why.into())
    }
}

fn red_card(pos: &Position, side: usize) -> bool {
    (0..pos.sides[side].active.len()).any(|slot| {
        pos.sides[side]
            .active_pokemon(slot)
            .is_some_and(|mon| !mon.fainted && mon.item.as_ref().is_some_and(|item| item.as_str() == "redcard"))
    })
}

/// The first of the heaviest completions (`selfplay.ranking_view`'s `max`).
fn heaviest(items: &[Item]) -> usize {
    let mut best = 0;
    for (k, item) in items.iter().enumerate().skip(1) {
        if item.weight > items[best].weight {
            best = k;
        }
    }
    best
}

/// How many completions the step made (`hidden.completions`' count).
fn completions(step: &crate::hiddenbench::Step) -> usize {
    step.sides.iter().flatten().filter_map(|side| side.as_ref().ok()).map(Vec::len).sum()
}

fn choices(menu: &[legal::SideAct]) -> Vec<String> {
    menu.iter().map(legal::choice).collect()
}

fn strings(side: &[String]) -> Value {
    json!(side)
}

/// What one move node needs of a step: the completions as typed items, who each side ranks from,
/// and the request `beliefNode` reads over them.
struct Node {
    views: [Position; 2],
    same: bool,
    used: [Option<(usize, Vec<String>)>; 2],
    asked: Value,
}

fn node_of(base: &Value, pos: &Position, spreads: [&Vec<Item>; 2]) -> Node {
    let root = crate::held::keep(pos);
    let mut asked = base.clone();
    asked["position"] = json!({ "held": root });
    let mut jobs = Vec::new();
    let mut hidden: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
    for side in 0..2 {
        let items = spreads[side];
        let slots: Vec<usize> = match items.first() {
            Some(first) if !first.exact => first.slots.clone(),
            _ => Vec::new(),
        };
        for item in items {
            let at = jobs.len();
            let held = if item.exact {
                root
            } else {
                crate::held::keep(item.position.as_ref().expect("a completion's position"))
            };
            asked[format!("p{at}")] = json!({ "held": held });
            let pk: Value = if item.exact || item.pk.is_empty() {
                json!(pos.sides[side]
                    .pokemon
                    .iter()
                    .map(|m| json!([crate::reg::to_id(m.species.as_str()), crate::reg::to_id(m.base_species.as_str()), m.slot]))
                    .collect::<Vec<_>>())
            } else {
                json!(item.pk)
            };
            jobs.push(json!({
                "side": side,
                "slots": slots,
                "exact": item.exact,
                "root": item.exact,
                "pk": pk,
            }));
        }
        hidden[side] = slots;
    }
    asked["hidden"] = json!([hidden[0], hidden[1]]);
    asked["carded"] = json!([red_card(pos, 0), red_card(pos, 1)]);
    asked["jobs"] = Value::Array(jobs);
    let mut solve = Vec::new();
    for side in 0..2 {
        let items = spreads[1 - side];
        if !items.is_empty() {
            let weights: Vec<u8> = items.iter().flat_map(|i| i.weight.to_le_bytes()).collect();
            solve.push(json!({ "side": side, "weights": b64_encode(&weights) }));
        }
    }
    asked["solve"] = Value::Array(solve);

    // `ranking_view`: the heaviest completion of the other side's bench is where a side ranks from.
    let mut views: Vec<Position> = Vec::new();
    let mut used: [Option<(usize, Vec<String>)>; 2] = [None, None];
    let mut roots = [false; 2];
    for side in 0..2 {
        let items = spreads[1 - side];
        if items.is_empty() {
            views.push(pos.clone());
            roots[side] = true;
            continue;
        }
        let at = heaviest(items);
        used[side] = Some((at, if items[at].exact { Vec::new() } else { items[at].species.clone() }));
        roots[side] = items[at].exact;
        views.push(if items[at].exact { pos.clone() } else { items[at].position.clone().expect("a completion's position") });
    }
    let same = roots[0] && roots[1];
    let second = views.pop().expect("two views");
    let first = views.pop().expect("two views");
    Node { views: [first, second], same, used, asked }
}

/// A side's strategy and the game's value, from `beliefNode`'s answer for the side.
fn solved_side(answer: &Value, side: usize) -> Result<(Vec<f64>, f64), String> {
    let one = answer["answers"]
        .as_array()
        .and_then(|list| list.iter().find(|a| a["side"].as_u64() == Some(side as u64)))
        .ok_or("the node's answer has no game for a side")?;
    if one.get("x").is_none() {
        return Err(format!("a side's game was not solved: {one}"));
    }
    let x = f64s(&one["x"])?;
    let row = one["valueRow"].as_f64().ok_or("no value")?;
    let col = one["valueCol"].as_f64().ok_or("no value")?;
    Ok((x, 0.5 * (row + col)))
}

fn run(reg: &Reg, encoder: &Encoder, value: &Value) -> Result<Value, Value> {
    let fault = |e: String| json!({ "fault": e });
    let mut rng = Pcg64::from_json(&value["rng"]).ok_or_else(|| fault("playGame without a PCG64 state".into()))?;
    let mut context = Context::from_json(&value["context"]).map_err(fault)?;
    let hidden = context.hidden.as_ref().ok_or_else(|| fault("playGame without a hidden bench".into()))?;
    let args = MenusArgs::from_json(&value["node"])?;
    let exact = Budget::from_json(&value["exactBudget"]);
    let steps = value["steps"].as_u64().ok_or_else(|| fault("playGame without steps".into()))? as usize;
    let mut pos = crate::held::position(&value["position"]);
    if &*pos.format != reg.format_id.as_str() {
        return Err(fault("position is for another regulation".into()));
    }
    let game = hidden.game;
    let leads = hidden.leads.clone();
    let mut seen: [Vec<String>; 2] = hidden.seen.clone();

    let mut events: Vec<Value> = Vec::new();
    let mut stop = "steps".to_string();
    let mut why = String::new();
    let mut taken = 0usize;
    let mut turns: Option<i64> = None;
    for _ in 0..steps {
        if pos.ended {
            stop = "ended".into();
            break;
        }
        crate::held::forget_kept();
        match one_step(reg, encoder, &mut context, &args, &exact, game, &leads, &mut pos, &mut seen, &mut rng, &value["node"]) {
            Ok(Done::Event(event, advanced)) => {
                if advanced {
                    turns = Some(pos.turn);
                }
                events.push(event);
                taken += 1;
            }
            Ok(Done::Unresolved(event)) => {
                events.push(event);
                taken += 1;
                stop = "unresolved".into();
                break;
            }
            Err(Handoff(reason)) => {
                stop = "handoff".into();
                why = reason;
                break;
            }
        }
    }
    crate::held::forget_kept();
    Ok(json!({
        "kind": "playGame",
        "events": events,
        "steps": taken,
        "stop": stop,
        "why": why,
        "turns": turns,
        "position": pos.to_json(),
        "seen": seen,
        "rng": rng.to_json_full(),
    }))
}

enum Done {
    /// The step's event; whether it was a move that advanced the turn.
    Event(Value, bool),
    /// A move whose turn had no outcome to go on from (the game ends there).
    Unresolved(Value),
}

/// A replacement phase's `Decision` as the record writes it, from `phase_core`'s answer.
fn replacement_decision(pos: &Position, step: &crate::hiddenbench::Step, answer: &Value) -> Result<Value, String> {
    let side_of = |key: &str, side: usize| -> Result<&Value, String> {
        answer.get(key).and_then(|v| v.get(side)).ok_or_else(|| format!("the answer has no {key}[{side}]"))
    };
    let chosen = |side: usize| -> Result<usize, String> {
        side_of("chosen", side)?.as_u64().map(|i| i as usize).ok_or_else(|| "chosen is not an index".to_string())
    };
    let picked = |side: usize| -> Result<Value, String> {
        side_of("options", side)?.get(chosen(side)?).cloned().ok_or_else(|| "chosen is past the options".to_string())
    };
    Ok(json!({
        "turn": pos.turn,
        "kind": "replacement",
        "position": pos.to_json(),
        "ownActions": side_of("options", 0)?,
        "ownPolicy": side_of("policy", 0)?,
        "foeActions": side_of("options", 1)?,
        "foePolicy": side_of("policy", 1)?,
        "searchValue": answer.get("value").ok_or("the answer has no value")?,
        "ownChosen": picked(0)?,
        "foeChosen": picked(1)?,
        "shownIdentities": step.seen,
        "foeSearchValue": answer.get("foeValue").ok_or("the answer has no foeValue")?,
    }))
}

#[allow(clippy::too_many_arguments)]
fn one_step(
    reg: &Reg,
    encoder: &Encoder,
    context: &mut Context,
    args: &MenusArgs,
    exact: &Budget,
    game: u64,
    leads: &[Option<Vec<String>>; 2],
    pos: &mut Position,
    seen: &mut [Vec<String>; 2],
    rng: &mut Pcg64,
    base: &Value,
) -> Result<Done, Handoff> {
    let step = step_of(reg, game, pos, seen, leads, None).ok_or_else(|| Handoff::of(format!("no game {game} is defined")))?;
    // What a mid-turn node (a self-switch, a replacement) reads as seen is this step's.
    if let Some(h) = context.hidden.as_mut() {
        h.seen = [step.seen[0].clone(), step.seen[1].clone()];
    }
    let owed = replacements_needed(pos);
    if owed.iter().flatten().any(|o| *o) {
        let mut trial = rng.clone();
        let (answer, after) = crate::replacephase::phase_core(reg, encoder, context, pos, &mut trial, &step)
            .map_err(|failed| Handoff::of(format!("replacement: {failed}")))?;
        let decision = replacement_decision(pos, &step, &answer)
            .map_err(|why| Handoff::of(format!("replacement record: {why}")))?;
        let event = json!({
            "kind": "replacement",
            "stepNotes": step.notes,
            "completions": completions(&step),
            "answer": answer,
            "decision": decision,
        });
        *rng = trial;
        *seen = [step.seen[0].clone(), step.seen[1].clone()];
        *pos = after;
        return Ok(Done::Event(event, false));
    }

    let mut lists: Vec<&Vec<Item>> = Vec::new();
    for side in 0..2 {
        match step.sides.get(side).and_then(|s| s.as_ref()) {
            Some(Ok(items)) => lists.push(items),
            Some(Err(why)) => return Err(Handoff::of(format!("hidden bench: {why}"))),
            None => return Err(Handoff::of("the completions of a side are missing")),
        }
    }
    let started = Instant::now();
    let node = node_of(base, pos, [lists[0], lists[1]]);
    let got = menus_core(reg, encoder, args, pos, [&node.views[0], &node.views[1]], node.same)
        .map_err(|failed| Handoff::of(format!("menus: {failed}")))?;
    let kept = got.1.clone();
    let mut answer = move_node_over(reg, encoder, got, node.asked.clone());
    for key in ["fault", "serverError", "refused"] {
        if answer.get(key).is_some() {
            return Err(Handoff::of(format!("node: {key}")));
        }
    }
    if args.rules.mega_from_slots && answer["mega"] != json!(true) {
        return Err(Handoff::of("node: the binary encoded with another can_mega rule"));
    }
    let (own_strategy, own_value) = solved_side(&answer, 0).map_err(Handoff::of)?;
    let (foe_strategy, foe_value) = solved_side(&answer, 1).map_err(Handoff::of)?;
    let seconds = started.elapsed().as_secs_f64();
    if kept[0].is_empty() || kept[1].is_empty() {
        return Err(Handoff::of("a menu is empty"));
    }
    if own_strategy.len() != kept[0].len() || foe_strategy.len() != kept[1].len() {
        return Err(Handoff::of("a strategy is not as long as its menu"));
    }

    // `_sample_index` of side 0's strategy, then side 1's; a draw numpy would not make is Python's.
    let mut trial = rng.clone();
    if numpy_sum(&own_strategy) <= 0.0 || numpy_sum(&foe_strategy) <= 0.0 {
        return Err(Handoff::of("a strategy with no mass"));
    }
    let own_at = choice(&own_strategy, &mut trial).map_err(Handoff::of)?;
    #[allow(unused_mut)]
    let mut foe_at = choice(&foe_strategy, &mut trial).map_err(Handoff::of)?;
    if cfg!(feature = "ika458-play-control") {
        foe_at = (foe_at + 1) % kept[1].len();
    }
    let chosen: [Vec<SlotAction>; 2] = [
        kept[0][own_at].iter().map(legal::Act::slot_action).collect(),
        kept[1][foe_at].iter().map(legal::Act::slot_action).collect(),
    ];
    let result = resolve_turn(reg, pos, &chosen, *exact).map_err(|why| Handoff::of(format!("resolve: {why}")))?;
    let advanced = advance_core(reg, encoder, result, Some(context), &mut trial).map_err(|failed| Handoff::of(format!("advance: {failed}")))?;
    MOVES.fetch_add(1, Ordering::Relaxed);

    // What the caller records of the node; the matrices and the games' texts stay here.
    let mut kept_answer = serde_json::Map::new();
    for key in ["fills", "served", "patched", "rootMenus", "shared", "redone", "mega", "lps"] {
        if let Some(part) = answer.get_mut(key) {
            kept_answer.insert(key.to_string(), part.take());
        }
    }
    let rank_views: Vec<Value> = node.used.iter().map(|u| u.as_ref().map_or(Value::Null, |(at, species)| json!([at, species]))).collect();
    let ours = choices(&kept[0]);
    let theirs = choices(&kept[1]);
    // The `Decision` as the record writes it (`GameRecord.to_json`), made here and only written by Python.
    let mut decision = json!({
        "turn": pos.turn,
        "kind": "move",
        "position": pos.to_json(),
        "ownActions": ours,
        "ownPolicy": own_strategy,
        "foeActions": theirs,
        "foePolicy": foe_strategy,
        "searchValue": own_value,
        "ownChosen": ours[own_at],
        "foeChosen": theirs[foe_at],
        "shownIdentities": step.seen,
        "foeSearchValue": -foe_value,
    });
    if rank_views.iter().any(|v| !v.is_null()) {
        decision["rankViews"] = Value::Array(rank_views);
    }
    let mut event = json!({
        "kind": "move",
        "stepNotes": step.notes,
        "completions": completions(&step),
        "hiddenSlots": node.asked["hidden"].clone(),
        "decision": decision,
        "answer": Value::Object(kept_answer),
        "nodeNotes": answer["unmodelled"].clone(),
        "advanceNotes": advanced.unmodelled,
        "index": advanced.index,
        "seconds": seconds,
    });
    if let Some(chain) = &advanced.chain {
        event["chain"] = chain.to_json();
    }
    *rng = trial;
    *seen = [step.seen[0].clone(), step.seen[1].clone()];
    match advanced.position {
        None => Ok(Done::Unresolved(event)),
        Some(landed) => {
            *pos = landed;
            Ok(Done::Event(event, true))
        }
    }
}
