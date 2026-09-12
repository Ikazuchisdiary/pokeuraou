//! One whole node -- a matrix of cells -- resolved and scored in one crossing.
//!
//! `batched_payoffs` is already the project's node-level boundary, so it is the right
//! place to put a process boundary too: a position and two lists of actions go in, a
//! payoff matrix comes out, and the leaves never cross. A cell this port refuses is named
//! in `refused` and the caller fills it in Python, which is what makes a partial port
//! usable rather than merely measurable.
//!
//! The protocol is JSONL over stdio, the same shape the sim-bridge already uses: one
//! request per line, one response per line, and the process stays warm.

use crate::objective;
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::{parse_actions_list, resolve_turn, turn_value, Budget, SlotAction};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::{BufRead, Write};

pub struct Request {
    pub position: Position,
    pub ours: Vec<Vec<SlotAction>>,
    pub theirs: Vec<Vec<SlotAction>>,
    pub budget: Budget,
    pub objectives: Vec<String>,
    /// Return the encoder's arrays and a fold instead of payoffs, for a learned leaf.
    pub encode: bool,
}

pub fn parse_request(value: &Value) -> Result<Request, String> {
    let position = Position::from_json(&value["position"]);
    let read = |key: &str| -> Vec<Vec<SlotAction>> {
        value[key]
            .as_array()
            .map(|list| list.iter().map(parse_actions_list).collect())
            .unwrap_or_default()
    };
    let objectives: Vec<String> = value["objectives"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect())
        .unwrap_or_default();
    let encode = value.get("encode").and_then(Value::as_bool).unwrap_or(false);
    // An encoded node may still be asked for named objectives beside the learned leaf --
    // the analyser runs one as a cross-check -- so the names are checked either way.
    for name in &objectives {
        if objective::by_name(name).is_none() {
            return Err(format!("objective not available in the port: {name}"));
        }
    }
    Ok(Request {
        encode,
        position,
        ours: read("ours"),
        theirs: read("theirs"),
        budget: Budget::from_json(&value["budget"]),
        objectives,
    })
}

/// Fills the matrix, leaving the cells this port refuses for the caller.
pub fn fill(reg: &Reg, request: &Request) -> Value {
    let rows = request.ours.len();
    let cols = request.theirs.len();
    let scorers: Vec<fn(&Position) -> f64> = request
        .objectives
        .iter()
        .map(|name| objective::by_name(name).unwrap())
        .collect();

    let mut payoffs: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![0.0; cols]; rows]; scorers.len().max(1)];
    let mut exact = vec![vec![false; cols]; rows];
    let mut refused: Vec<Value> = Vec::new();
    let mut unmodelled: BTreeSet<String> = BTreeSet::new();

    for (i, ours) in request.ours.iter().enumerate() {
        for (j, theirs) in request.theirs.iter().enumerate() {
            let actions = [ours.clone(), theirs.clone()];
            match resolve_turn(reg, &request.position, &actions, request.budget) {
                Err(reason) => refused.push(json!([i, j, reason])),
                Ok(result) => {
                    exact[i][j] = result.exact;
                    let mut failed: Option<String> = None;
                    for (index, score) in scorers.iter().enumerate() {
                        match turn_value(reg, &result, *score, 0, &mut unmodelled) {
                            Ok(value) => payoffs[index][i][j] = value,
                            Err(reason) => {
                                failed = Some(reason);
                                break;
                            }
                        }
                    }
                    if let Some(reason) = failed {
                        refused.push(json!([i, j, reason]));
                    }
                }
            }
        }
    }

    json!({
        "payoffs": payoffs,
        "exact": exact,
        "refused": refused,
        "unmodelled": unmodelled.into_iter().collect::<Vec<_>>(),
    })
}

/// One turn, for advancing a game rather than filling a matrix.
///
/// The weights come back first and the caller samples an index with its own generator --
/// which is what keeps a generated game bit-identical to one played without this bridge,
/// since `numpy.random.Generator.choice` is the stream. Then it asks for that one
/// position. Returning every branch instead would be right and useless: an exact budget
/// produces a hundred-odd of them, and a hundred positions is two megabytes of JSON per
/// turn against the thirty kilobytes this costs.
pub fn resolve_one(reg: &Reg, value: &Value) -> Value {
    let position = Position::from_json(&value["position"]);
    if &*position.format != reg.format_id.as_str() {
        return json!({ "refused": "position is for another regulation" });
    }
    let actions_value = &value["actions"];
    let actions = [
        parse_actions_list(&actions_value[0]),
        parse_actions_list(&actions_value[1]),
    ];
    let budget = Budget::from_json(&value["budget"]);
    let result = match resolve_turn(reg, &position, &actions, budget) {
        Err(reason) => return json!({ "refused": reason }),
        Ok(result) => result,
    };
    // A suspension's continuation state cannot cross a process boundary, but its weight
    // can -- and the caller only has to resolve the turn itself when it actually draws
    // one, which is rarer than merely having one.
    let weights: Vec<f64> = result.branches.iter().map(|b| b.probability).collect();
    let paused: Vec<f64> = result.suspended.iter().map(|s| s.probability).collect();
    let selected = value.get("select").and_then(Value::as_u64).map(|k| k as usize);
    let chosen = match selected {
        None => Value::Null,
        Some(index) => match result.branches.get(index) {
            None => return json!({ "refused": "branch index out of range" }),
            Some(branch) => branch.position.to_json(),
        },
    };
    json!({
        "branches": weights,
        "suspended": paused,
        "exact": result.exact,
        "unmodelled": result.unmodelled.iter().cloned().collect::<Vec<_>>(),
        "position": chosen,
    })
}

/// Scores a pool of candidates, which is what decides the menu the matrix is filled from.
///
/// Separate from `fill` because it is a different question with a different shape: no
/// turns are resolved, only damage calculated, and what comes back is one number per
/// candidate rather than a matrix.
fn score_pool(reg: &Reg, value: &Value) -> Value {
    let position = Position::from_json(&value["position"]);
    if &*position.format != reg.format_id.as_str() {
        return json!({
            "error": format!(
                "position is {} but the regulation is {}", position.format, reg.format_id
            )
        });
    }
    let side = value["side"].as_u64().unwrap_or(0) as usize;
    let candidates: Vec<Vec<crate::resolve::SlotAction>> = value["candidates"]
        .as_array()
        .map(|list| list.iter().map(crate::resolve::parse_actions_list).collect())
        .unwrap_or_default();
    match crate::score::score_candidates(reg, &position, side, &candidates) {
        Err(reason) => json!({ "refused": reason }),
        Ok(scored) => json!({
            "scores": scored.iter().map(|s| s.score).collect::<Vec<_>>(),
            "detail": scored
                .iter()
                .map(|s| {
                    s.detail
                        .iter()
                        .map(|c| json!([c.slot, c.target_slot, c.is_foe, c.signed, c.exact]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        }),
    }
}

/// JSONL over stdio: one request per line, one response per line.
pub fn serve(reg: &Reg) {
    let encoder = crate::encode::Encoder::new(reg);
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        // A panic answering one node must not cost the caller the rest of the run. The
        // node it panicked on is refused, which the caller already knows how to fill in
        // Python, and the process stays up. Two generation workers lost a thousand games
        // each to a failure that took the bridge down for good.
        let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            answer(reg, &encoder, &line, &mut stdout)
        }));
        match answered {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                // Loudly. Every one of these used to be a bare `break`, so the process
                // exited cleanly with nothing on stderr and the caller was left holding
                // "the Rust node sent 0 of 64219920 bytes:" with no reason after the colon.
                eprintln!("node: the pipe failed ({error}); stopping");
                break;
            }
            Err(_) => {
                // `catch_unwind` has already printed the panic and where it came from.
                let refusal = json!({ "refused": "the port panicked on this node" });
                if writeln!(stdout, "{refusal}").and_then(|()| stdout.flush()).is_err() {
                    break;
                }
            }
        }
    }
}

/// Answers one request onto `stdout`. `Err` means the pipe is gone and serving is over.
fn answer<W: Write>(
    reg: &Reg,
    encoder: &crate::encode::Encoder,
    line: &str,
    stdout: &mut W,
) -> std::io::Result<()> {
    let response = match serde_json::from_str::<Value>(line) {
        Err(error) => json!({ "error": error.to_string() }),
        Ok(value) if value["kind"].as_str() == Some("resolve") => resolve_one(reg, &value),
        Ok(value) if value["kind"].as_str() == Some("score") => score_pool(reg, &value),
        Ok(value) => match parse_request(&value) {
            Err(reason) => json!({ "error": reason }),
            Ok(request) => {
                if &*request.position.format != reg.format_id.as_str() {
                    json!({
                        "error": format!(
                            "position is {} but the regulation is {}",
                            request.position.format, reg.format_id
                        )
                    })
                } else if request.encode {
                    // A header line, then the raw buffers on the same pipe.
                    let (header, encoded, leaf_values) =
                        crate::encoded_node::fill(reg, encoder, &request);
                    writeln!(stdout, "{header}")?;
                    crate::encoded_node::write_body(stdout, &encoded, &leaf_values)?;
                    return stdout.flush();
                } else {
                    fill(reg, &request)
                }
            }
        },
    };
    writeln!(stdout, "{response}")?;
    stdout.flush()
}
