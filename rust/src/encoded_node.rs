//! A node whose leaves are scored by a learned value function.
//!
//! The parameter-free objectives are computed in Rust and only the matrix crosses. A
//! learned leaf cannot work that way -- its input *is* the leaves -- so this crossing
//! carries the encoder's arrays instead: the same buffers `Encoder.encode_positions`
//! produces, as raw little-endian bytes, plus the structure that turns one value per leaf
//! back into one number per cell.
//!
//! That structure is `batched_payoffs`' own: a span and a weight vector per ordinary cell,
//! and for a cell whose turn stopped for a replacement, the fold tree `turn_leaves` builds
//! -- chance averaged, and the replacement taken by whoever chooses it. Python keeps
//! `fold_value`; this only has to describe the tree.
//!
//! Sending positions instead was measured and rejected: a node's leaves are about 2,000
//! and a position is 15 KB of JSON, against 3.7 KB of encoding each and no parsing.

use crate::encode::{Encoded, Encoder};
use crate::node::Request;
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::{resolve_turn, resume_alternatives, Suspended, TurnResult};
use serde_json::{json, Value};
use std::collections::BTreeSet;

/// What one cell contributes: leaves, and how to fold their values back into a number.
enum Cell {
    /// (start, weights) -- `values[start..start + weights.len()] @ weights`.
    Span(usize, Vec<f64>),
    /// A fold tree over leaf indices.
    Folded(Value),
    /// Nothing to score; the payoff stays zero, as it does in Python.
    Empty,
}

struct Collector<'a> {
    reg: &'a Reg,
    leaves: Vec<Position>,
    notes: BTreeSet<String>,
}

impl<'a> Collector<'a> {
    fn add_leaf(&mut self, position: Position) -> usize {
        self.leaves.push(position);
        self.leaves.len() - 1
    }

    /// `turn_leaves`: flattens a turn -- suspensions included -- into leaves plus a fold.
    fn turn_leaves(&mut self, result: &TurnResult<'a>, depth: usize) -> Value {
        self.notes.extend(result.unmodelled.iter().cloned());
        let mut parts: Vec<Value> = Vec::new();
        for branch in &result.branches {
            let index = self.add_leaf(branch.position.clone());
            parts.push(json!([branch.probability, { "leaf": index }]));
        }
        for pause in &result.suspended {
            // Four Pokemon a side means a turn cannot interrupt itself indefinitely; the
            // guard is against a bug becoming unbounded recursion.
            if depth >= 4 {
                self.notes.insert("more than four mid-turn replacements in one turn".into());
                let index = self.add_leaf(pause.turn.pos.clone());
                parts.push(json!([pause.probability, { "leaf": index }]));
                continue;
            }
            match self.alternatives(pause) {
                None => {
                    self.notes.insert("a suspended turn offered no replacement".into());
                    let index = self.add_leaf(pause.turn.pos.clone());
                    parts.push(json!([pause.probability, { "leaf": index }]));
                }
                Some((chooser, resumed)) => {
                    let options: Vec<Value> = resumed
                        .iter()
                        .map(|one| self.turn_leaves(one, depth + 1))
                        .collect();
                    parts.push(json!([
                        pause.probability,
                        { "best": chooser, "options": options }
                    ]));
                }
            }
        }
        json!({ "avg": parts })
    }

    fn alternatives(&mut self, pause: &Suspended<'a>) -> Option<(usize, Vec<TurnResult<'a>>)> {
        match resume_alternatives(self.reg, pause) {
            Err(_) => None,
            Ok((None, _)) => None,
            Ok((Some(chooser), options)) if !options.is_empty() => Some((chooser, options)),
            Ok(_) => None,
        }
    }
}

/// Fills a node as leaves plus a fold, and encodes the leaves.
pub fn fill(reg: &Reg, encoder: &Encoder, request: &Request) -> (Value, Vec<u8>) {
    let rows = request.ours.len();
    let cols = request.theirs.len();
    let mut exact = vec![vec![false; cols]; rows];
    let mut refused: Vec<Value> = Vec::new();
    let mut cells: Vec<(usize, usize, Cell)> = Vec::new();
    let mut collector = Collector { reg, leaves: Vec::new(), notes: BTreeSet::new() };

    for (i, ours) in request.ours.iter().enumerate() {
        for (j, theirs) in request.theirs.iter().enumerate() {
            let actions = [ours.clone(), theirs.clone()];
            let result = match resolve_turn(reg, &request.position, &actions, request.budget) {
                Err(reason) => {
                    refused.push(json!([i, j, reason]));
                    continue;
                }
                Ok(result) => result,
            };
            exact[i][j] = result.exact;
            if result.is_suspended() {
                let before = collector.leaves.len();
                let root = collector.turn_leaves(&result, 0);
                if collector.leaves.len() == before {
                    cells.push((i, j, Cell::Empty));
                } else {
                    cells.push((i, j, Cell::Folded(root)));
                }
                continue;
            }
            collector.notes.extend(result.unmodelled.iter().cloned());
            let total: f64 = result.branches.iter().map(|b| b.probability).sum();
            if result.branches.is_empty() || total <= 0.0 {
                cells.push((i, j, Cell::Empty));
                continue;
            }
            let start = collector.leaves.len();
            let weights: Vec<f64> =
                result.branches.iter().map(|b| b.probability / total).collect();
            for branch in result.branches {
                collector.leaves.push(branch.position);
            }
            cells.push((i, j, Cell::Span(start, weights)));
        }
    }

    let borrowed: Vec<&Position> = collector.leaves.iter().collect();
    let encoded = encoder.encode_positions(&borrowed);
    let mut bytes = pack(&encoded);

    // A node can want both: the learned leaf, and a parameter-free objective beside it as
    // a cross-check. The leaves are here and these are cheap, so they go back per leaf and
    // fold through the same spans rather than sending the whole node back for the sake of
    // the second column.
    for name in &request.objectives {
        let score = crate::objective::by_name(name).expect("the request was checked");
        for position in &collector.leaves {
            bytes.extend_from_slice(&score(position).to_le_bytes());
        }
    }

    let mut spans: Vec<Value> = Vec::new();
    let mut folded: Vec<Value> = Vec::new();
    for (i, j, cell) in cells {
        match cell {
            Cell::Empty => {}
            Cell::Span(start, weights) => spans.push(json!([i, j, start, weights])),
            Cell::Folded(root) => folded.push(json!([i, j, root])),
        }
    }

    let header = json!({
        "kind": "encoded",
        "leaves": collector.leaves.len(),
        "spans": spans,
        "folded": folded,
        "exact": exact,
        "refused": refused,
        "unmodelled": collector.notes.into_iter().collect::<Vec<_>>(),
        "unknownVolatiles": encoded.unknown_volatiles,
        "leafObjectives": request.objectives,
        "monsPerSide": encoder.widths.mons_per_side,
        "monWidth": encoder.widths.mon,
        "sideWidth": encoder.widths.side,
        "fieldWidth": encoder.widths.field,
        "bytes": bytes.len(),
    });
    (header, bytes)
}

/// The buffers in the order the reader expects, little-endian.
fn pack(encoded: &Encoded) -> Vec<u8> {
    let total = (encoded.species.len()
        + encoded.ability.len()
        + encoded.item.len()
        + encoded.moves.len())
        * 4
        + (encoded.mon.len() + encoded.mask.len() + encoded.side.len() + encoded.field.len())
            * 4;
    let mut out = Vec::with_capacity(total);
    for value in &encoded.species {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in &encoded.ability {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in &encoded.item {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in &encoded.moves {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in &encoded.mon {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in &encoded.mask {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in &encoded.side {
        out.extend_from_slice(&value.to_le_bytes());
    }
    for value in &encoded.field {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}
