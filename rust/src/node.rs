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
    for name in &objectives {
        if objective::by_name(name).is_none() {
            return Err(format!("objective not available in the port: {name}"));
        }
    }
    Ok(Request {
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

/// JSONL over stdio: one request per line, one response per line.
pub fn serve(reg: &Reg) {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Err(error) => json!({ "error": error.to_string() }),
            Ok(value) => match parse_request(&value) {
                Err(reason) => json!({ "error": reason }),
                Ok(request) => {
                    if request.position.format != reg.format_id {
                        json!({
                            "error": format!(
                                "position is {} but the regulation is {}",
                                request.position.format, reg.format_id
                            )
                        })
                    } else {
                        fill(reg, &request)
                    }
                }
            },
        };
        if writeln!(stdout, "{response}").is_err() {
            break;
        }
        if stdout.flush().is_err() {
            break;
        }
    }
}
