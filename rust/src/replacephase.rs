//! IKA-458 stage 3.6a: a game's replacement phase (the node after a faint) chosen and played
//! in one request.
//!
//! `selfplay._do_replacement_node` made its options (`switch_actions_after_faint`), asked for the
//! node (`replacementNode`), drew each side's replacement from its strategy with numpy's
//! `choice`, and asked for the phase once more for each draw it opened
//! (`resolve_replacements`). Here one request does all of it, in the same order and so with the
//! same requests to the inference server and the same draws from the generator: the options, the
//! completions of both benches (`hiddenbench::step_of`, as `hiddenStep` answers them), every
//! completion's matrix (`replacenode::score_job`), the two Bayesian games, the two draws
//! (`advance::choice`), and the phase replayed with each draw it opens answered from the same
//! generator (`commands::replacement_phase_drawn`, whose answer is `Generator.choice(n, p=w / sum(w))`).
//!
//! ```text
//! { "kind": "replacementPhase", "position": P, "rng": {state, inc, hasUint32, uinteger},
//!   "context": { servers, leaf, hidden: { game, seen, leads } } }     // as `advanceTurn`'s `selfSwitch`
//! ```
//!
//! What Python reads back: each side's options as the record writes them, the strategies and
//! values, the indices drawn, the position the phase ended in with the replacements it owes, the
//! phase's notes, the generator's state and the server's work. Anything the port cannot answer
//! the way the Python road would -- a side with no completions, a game the port does not hold, a
//! game the LP did not solve, a draw numpy would refuse -- is a `fault`: Python then does the node
//! itself, its generator untouched.

use crate::advance::{choice, choose_p, Pcg64};
use crate::beliefnode::solve_asks;
use crate::encode::Encoder;
use crate::hiddenbench::step_of;
use crate::lp::{sum_f64, sum_of, LPS};
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::commands::{replacement_phase_drawn, replacements_needed};
use crate::resolve::{replacement_options, SlotAction};
use crate::selfswitch::{choice_string, Context};
use crate::served::Stats;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

/// The phases this process played, the positive control of the stage.
pub static PHASES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn replacement_phase(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    match phase(reg, encoder, value) {
        Ok(answer) | Err(answer) => answer,
    }
}

/// `switch_actions_after_faint` of the slots a side owes, or its passes where it owes none.
fn options_of(pos: &Position, side: usize, owed: &[bool]) -> Vec<Vec<SlotAction>> {
    let found = if owed.iter().any(|o| *o) {
        let both = [owed.first().copied().unwrap_or(false), owed.get(1).copied().unwrap_or(false)];
        replacement_options(pos, side, both)
    } else {
        Vec::new()
    };
    if found.is_empty() {
        vec![(0..pos.sides[side].active.len()).map(|slot| SlotAction::Pass { slot }).collect()]
    } else {
        found
    }
}

fn phase(reg: &Reg, encoder: &Encoder, value: &Value) -> Result<Value, Value> {
    let fault = |e: String| json!({ "fault": e });
    let mut rng = Pcg64::from_json(&value["rng"]).ok_or_else(|| fault("replacementPhase without a PCG64 state".into()))?;
    let context = Context::from_json(&value["context"]).map_err(fault)?;
    let hidden = context.hidden.as_ref().ok_or_else(|| fault("replacementPhase without a hidden bench".into()))?;
    let pos = crate::held::position(&value["position"]);
    if &*pos.format != reg.format_id.as_str() {
        return Err(fault("position is for another regulation".into()));
    }

    // The completions of both benches, as `hiddenStep` gives them.
    let step = step_of(reg, hidden.game, &pos, &hidden.seen, &hidden.leads, None)
        .ok_or_else(|| fault(format!("replacementPhase: no game {} is defined", hidden.game)))?;
    let (answer, _after) = phase_core(reg, encoder, &context, &pos, &mut rng, &step)?;
    Ok(answer)
}

/// The phase at `pos` over `step`'s completions, the draws made from `rng`; the answer
/// `replacementPhase` gives, without the generator's state (the caller's to add). The generator
/// is moved on only as far as the draws that were made: a fault leaves the caller to clone it.
pub(crate) fn phase_core(
    reg: &Reg,
    encoder: &Encoder,
    context: &Context,
    pos: &Position,
    rng: &mut Pcg64,
    step: &crate::hiddenbench::Step,
) -> Result<(Value, Position), Value> {
    let fault = |e: String| json!({ "fault": e });
    let needed = replacements_needed(pos);
    if !needed.iter().flatten().any(|o| *o) {
        return Err(fault("no replacement is owed".into()));
    }
    let options = [options_of(pos, 0, &needed[0]), options_of(pos, 1, &needed[1])];
    let mut spreads: Vec<(Vec<Position>, Vec<f64>)> = Vec::new();
    for side in 0..2 {
        match step.sides.get(side).and_then(|s| s.as_ref()) {
            Some(Ok(items)) => spreads.push((
                items.iter().map(|item| item.position.clone().unwrap_or_else(|| pos.clone())).collect(),
                items.iter().map(|item| item.weight).collect(),
            )),
            Some(Err(why)) => return Err(fault(format!("hidden bench at a replacement: {why}"))),
            None => return Err(fault("the completions of a side are missing".into())),
        }
    }

    let (m, n) = (options[0].len(), options[1].len());
    let pairs: Vec<[Vec<SlotAction>; 2]> =
        options[0].iter().flat_map(|a| options[1].iter().map(move |b| [a.clone(), b.clone()])).collect();
    let before = LPS.load(Ordering::Relaxed);
    let mut stats: Vec<Stats> = context.servers.iter().map(|_| Stats::default()).collect();
    let mut ended: Vec<u64> = vec![0; context.servers.len()];
    let mut matrices: Vec<Vec<f64>> = Vec::new();
    let mut job_sides: Vec<i64> = Vec::new();
    let mut fills: Vec<Value> = Vec::new();
    // Side 0's leaf scores the completions of side 1's bench, then side 1's those of side 0's.
    for side in 0..2 {
        let leaf = context.leaf[side];
        for position in &spreads[1 - side].0 {
            let (cells, mut fill) = crate::replacenode::score_job(
                reg,
                encoder,
                position,
                &pairs,
                context.rules[leaf],
                &context.servers[leaf],
                context.net_scores_ends[leaf],
                &mut stats[leaf],
                &mut ended[leaf],
                m,
                n,
            )?;
            fill["leaf"] = json!(leaf);
            matrices.push(cells);
            job_sides.push((1 - side) as i64);
            fills.push(fill);
        }
    }
    let asks: Vec<(i64, Vec<f64>)> = (0..2).map(|side| (side as i64, spreads[1 - side].1.clone())).collect();
    let mut strategies: Vec<Vec<f64>> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    #[cfg(feature = "lp-dump")]
    let _tag = crate::lpbench::tag("replacement");
    for (_side, solved) in solve_asks(&asks, &matrices, &job_sides, m, n) {
        match solved {
            Ok(s) => {
                values.push(s.value());
                strategies.push(s.x);
            }
            Err(why) => return Err(fault(format!("the replacement node's game was not solved: {why:?}"))),
        }
    }
    let lps = LPS.load(Ordering::Relaxed) - before;

    // `_sample_index` of each side's strategy, side 0's first.
    let mut picked = [0usize; 2];
    for side in 0..2 {
        if sum_f64(&strategies[side]) <= 0.0 {
            return Err(fault("a strategy with no mass".into()));
        }
        picked[side] = choice(&strategies[side], rng).map_err(fault)?;
    }
    if cfg!(feature = "ika458-phase-control") {
        for side in 0..2 {
            picked[side] = (picked[side] + 1) % options[side].len();
        }
    }
    let chosen = [options[0][picked[0]].clone(), options[1][picked[1]].clone()];
    // `RustNode._phase`: `rng.choice(len(weights), p=[w / total for w in weights])`, the total
    // Python's own `sum`.
    let (after, notes) = replacement_phase_drawn(reg, pos, chosen, |weights| {
        let total = sum_of(weights.iter().copied());
        let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
        choose_p(&p, &mut *rng)
    })
    .map_err(fault)?;
    PHASES.fetch_add(1, Ordering::Relaxed);

    let strings = |side: usize| -> Result<Vec<String>, Value> {
        options[side].iter().map(|o| choice_string(o)).collect::<Result<_, _>>().map_err(fault)
    };
    let answer = json!({
        "kind": "replacementPhase",
        "options": [strings(0)?, strings(1)?],
        "policy": strategies,
        "value": values[0],
        "foeValue": -values[1],
        "chosen": picked,
        "position": after.to_json(),
        "needed": replacements_needed(&after),
        "notes": notes.into_iter().collect::<Vec<_>>(),
        "rng": rng.to_json_full(),
        "fills": fills,
        "lps": lps,
        "served": stats
            .iter()
            .zip(&ended)
            .map(|(s, e)| json!({
                "requests": s.requests,
                "rows": s.rows,
                "ended": e,
                "waitUs": s.wait_ns as f64 / 1e3,
                "copyUs": s.copy_ns as f64 / 1e3,
            }))
            .collect::<Vec<_>>(),
    });
    Ok((answer, after))
}
