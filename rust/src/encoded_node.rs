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
use std::collections::{BTreeSet, HashMap};

/// What one cell contributes: leaves, and how to fold their values back into a number.
enum Cell {
    /// (leaf indices, weights) -- `values[indices] @ weights`.
    Span(Vec<usize>, Vec<f64>),
    /// A fold tree over leaf indices.
    Folded(Value),
    /// Nothing to score; the payoff stays zero, as it does in Python.
    Empty,
}

struct Collector<'a> {
    reg: &'a Reg,
    leaves: Vec<Position>,
    /// Leaf indices grouped by a cheap key, for finding a position already collected.
    seen: HashMap<u64, Vec<usize>>,
    notes: BTreeSet<String>,
}

/// A cheap, partial hash of a position.
///
/// Partial on purpose: what makes deduplication *correct* is the `==` below, which is
/// derived over every field. A key that misses something only costs a comparison that
/// fails, never a leaf that is wrongly shared. So this hashes what varies between the
/// leaves of one node -- HP, status, boosts, which volatiles are on, who is out -- and
/// skips the rest.
fn leaf_key(pos: &Position) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    eat(pos.turn as u64);
    eat(pos.ended as u64);
    for side in pos.sides.iter() {
        for slot in side.active.iter() {
            eat(slot.map(|index| index as u64 + 1).unwrap_or(0));
        }
        eat(side.side_conditions.len() as u64);
        for mon in side.pokemon.iter() {
            eat(mon.hp as u64);
            eat(mon.fainted as u64);
            eat(mon.status.map(|s| s.as_str().len() as u64 + 7).unwrap_or(0));
            for boost in mon.boosts {
                eat(boost as i64 as u64);
            }
            eat(mon.volatiles.len() as u64);
            for volatile in &mon.volatiles {
                eat(volatile.id.as_str().len() as u64);
                eat(volatile.duration.unwrap_or(-1) as u64);
            }
        }
    }
    eat(pos.field.weather.map(|w| w.as_str().len() as u64 + 3).unwrap_or(0));
    eat(pos.field.terrain.map(|t| t.as_str().len() as u64 + 5).unwrap_or(0));
    hash
}

impl<'a> Collector<'a> {
    /// Collects a leaf, reusing one already collected when it is the same position.
    ///
    /// A third to two fifths of a node's leaves are a position the node already has --
    /// both sides' moves failing, two Protects, a switch meeting the same reply -- and
    /// each duplicate was an encoding, 3.7 KB across the pipe and a row of the forward
    /// pass. Sharing them is exact rather than approximate: the network sees the same
    /// input, and a row's output does not depend on how many rows are beside it (checked
    /// directly, on both CPU and GPU, over 19,333 leaves).
    fn add_leaf(&mut self, position: Position) -> usize {
        let key = leaf_key(&position);
        if let Some(candidates) = self.seen.get(&key) {
            for index in candidates {
                if self.leaves[*index] == position {
                    return *index;
                }
            }
        }
        self.leaves.push(position);
        let index = self.leaves.len() - 1;
        self.seen.entry(key).or_default().push(index);
        index
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
    let mut collector = Collector {
        reg,
        leaves: Vec::new(),
        seen: HashMap::new(),
        notes: BTreeSet::new(),
    };
    // Resolving and encoding are different costs with different fixes, and the split was
    // being guessed at from two benchmarks that did not add up. It goes back in the header.
    let resolve_started = std::time::Instant::now();

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
                let root = collector.turn_leaves(&result, 0);
                // Whether the tree has anything in it, rather than whether collecting it
                // grew the leaf list: with leaves shared, a cell can reference only leaves
                // the node already had, and counting would call such a cell empty and
                // leave its payoff at zero.
                let empty = root
                    .get("avg")
                    .and_then(Value::as_array)
                    .map(|parts| parts.is_empty())
                    .unwrap_or(true);
                cells.push((i, j, if empty { Cell::Empty } else { Cell::Folded(root) }));
                continue;
            }
            collector.notes.extend(result.unmodelled.iter().cloned());
            let total: f64 = result.branches.iter().map(|b| b.probability).sum();
            if result.branches.is_empty() || total <= 0.0 {
                cells.push((i, j, Cell::Empty));
                continue;
            }
            let weights: Vec<f64> =
                result.branches.iter().map(|b| b.probability / total).collect();
            let indices: Vec<usize> = result
                .branches
                .into_iter()
                .map(|branch| collector.add_leaf(branch.position))
                .collect();
            cells.push((i, j, Cell::Span(indices, weights)));
        }
    }

    let resolve_us = resolve_started.elapsed().as_secs_f64() * 1e6;
    let encode_started = std::time::Instant::now();
    let borrowed: Vec<&Position> = collector.leaves.iter().collect();
    let encoded = encoder.encode_positions(&borrowed);
    let encode_us = encode_started.elapsed().as_secs_f64() * 1e6;
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
            Cell::Span(indices, weights) => spans.push(json!([i, j, indices, weights])),
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
        "resolveUs": resolve_us,
        "encodeUs": encode_us,
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
