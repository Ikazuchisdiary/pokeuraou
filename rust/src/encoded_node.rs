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
use std::io::Write;

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
    /// How many leaves were handed to `add_leaf`, against how many `leaves` ended up
    /// holding. The sharing below is the reason a node crossing this way is smaller than
    /// the same node resolved in Python, and that reason has been read off the code and
    /// never counted -- so it is counted here, where it happens.
    offered: usize,
}

/// A cheap, partial hash of a position.
///
/// Partial on purpose: what makes deduplication *correct* is the `==` below, which is
/// derived over every field. A key that misses something only costs a comparison that
/// fails, never a leaf that is wrongly shared. So this hashes what varies between the
/// leaves of one node -- HP, status, boosts, which volatiles are on, who is out -- and
/// skips the rest.
pub fn leaf_key(pos: &Position) -> u64 {
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
        self.offered += 1;
        let key = leaf_key(&position);
        // IKA-62's positive control: `POKEURAOU_RUST_LEAF_SHARING=0` stores every leaf it
        // is offered, so what sharing takes off a node can be put back and seen to come
        // back. Read once per process; unset or any other value is the default, sharing on.
        static SHARING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let sharing = *SHARING.get_or_init(|| {
            std::env::var("POKEURAOU_RUST_LEAF_SHARING").map_or(true, |value| value != "0")
        });
        if let Some(candidates) = self.seen.get(&key).filter(|_| sharing) {
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
            let shared = crate::resolve::sharing_budget(pause, result.suspended.len());
            match self.alternatives(shared.as_ref().unwrap_or(pause)) {
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
        alternatives(self.reg, pause)
    }

    /// `turn_leaves` over pauses already resumed by `resume_all`: the same leaves handed to
    /// `add_leaf` in the same order, and the same notes, because resuming never touched the
    /// collector -- only where the resuming ran has moved (IKA-32).
    fn turn_leaves_resumed(
        &mut self,
        result: &TurnResult<'a>,
        resumed: &Resumed<'a>,
        depth: usize,
    ) -> Value {
        self.notes.extend(result.unmodelled.iter().cloned());
        let mut parts: Vec<Value> = Vec::new();
        for branch in &result.branches {
            let index = self.add_leaf(branch.position.clone());
            parts.push(json!([branch.probability, { "leaf": index }]));
        }
        for (pause, walk) in result.suspended.iter().zip(&resumed.pauses) {
            match walk {
                Walk::TooDeep => {
                    self.notes.insert("more than four mid-turn replacements in one turn".into());
                    let index = self.add_leaf(pause.turn.pos.clone());
                    parts.push(json!([pause.probability, { "leaf": index }]));
                }
                Walk::Nothing => {
                    self.notes.insert("a suspended turn offered no replacement".into());
                    let index = self.add_leaf(pause.turn.pos.clone());
                    parts.push(json!([pause.probability, { "leaf": index }]));
                }
                Walk::Options(chooser, options) => {
                    let options: Vec<Value> = options
                        .iter()
                        .map(|(one, below)| self.turn_leaves_resumed(one, below, depth + 1))
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
}

fn alternatives<'a>(reg: &'a Reg, pause: &Suspended<'a>) -> Option<(usize, Vec<TurnResult<'a>>)> {
    match resume_alternatives(reg, pause) {
        Err(_) => None,
        Ok((None, _)) => None,
        Ok((Some(chooser), options)) if !options.is_empty() => Some((chooser, options)),
        Ok(_) => None,
    }
}

/// A turn's pauses resumed ahead of collecting it, on the cell thread that resolved it
/// (IKA-32), one `Walk` per pause in `suspended`'s order.
struct Resumed<'a> {
    pauses: Vec<Walk<'a>>,
}

/// What `turn_leaves` does at one pause.
enum Walk<'a> {
    /// Four replacements deep already: the paused position is the leaf.
    TooDeep,
    /// No replacement to choose: the paused position is the leaf.
    Nothing,
    /// The chooser, and each option's resumed turn with its own pauses resumed.
    Options(usize, Vec<(TurnResult<'a>, Resumed<'a>)>),
}

/// `turn_leaves`' resuming, without the collecting: the same `sharing_budget` and
/// `resume_alternatives` calls in the same order, recursively.
fn resume_all<'a>(reg: &'a Reg, result: &TurnResult<'a>, depth: usize) -> Resumed<'a> {
    let pauses = result
        .suspended
        .iter()
        .map(|pause| {
            if depth >= 4 {
                return Walk::TooDeep;
            }
            let shared = crate::resolve::sharing_budget(pause, result.suspended.len());
            match alternatives(reg, shared.as_ref().unwrap_or(pause)) {
                None => Walk::Nothing,
                Some((chooser, resumed)) => Walk::Options(
                    chooser,
                    resumed
                        .into_iter()
                        .map(|one| {
                            let below = resume_all(reg, &one, depth + 1);
                            (one, below)
                        })
                        .collect(),
                ),
            }
        })
        .collect();
    Resumed { pauses }
}

/// Fills a node as leaves plus a fold, and encodes the leaves.
///
/// Returns the header, the encoded arrays and the per-leaf objective values; `write_body`
/// puts the last two on the wire. They are not packed into one buffer here because that
/// buffer is the largest allocation this process makes and it is a copy of what it is
/// built from.
pub fn fill(
    reg: &Reg,
    encoder: &Encoder,
    request: &Request,
) -> (Value, Encoded, Vec<f64>, Vec<u8>) {
    let (header, encoded, leaf_values, spans, _kept) =
        fill_shared(reg, encoder, request, None, false);
    (header, encoded, leaf_values, spans)
}

/// A node's resolved turns, kept for a later node of the same `fills` crossing (IKA-295).
pub struct Kept {
    position: Position,
    /// The request's actions as it sent them, to find a cell by its two actions.
    ours: Vec<Value>,
    theirs: Vec<Value>,
    /// Per (i, j), a turn that did not pause: its weights, notes and branches.
    cells: HashMap<(usize, usize), KeptCell>,
}

struct KeptCell {
    exact: bool,
    unmodelled: BTreeSet<String>,
    branches: Vec<(f64, Position)>,
}

/// How a node may read its cells' turns off an earlier node's (IKA-295): the earlier one
/// is `kept`, and the two positions are two completions of one position -- the same but
/// for `side`'s hidden `slots` (checked when this is made, not assumed).
pub struct Like<'k> {
    kept: &'k Kept,
    side: usize,
    slots: Vec<usize>,
    ours: Vec<Value>,
    theirs: Vec<Value>,
}

impl<'k> Like<'k> {
    /// `None` unless `position` is `kept`'s position with other Pokemon in `side`'s
    /// `slots` (and that side's `mega_capable_slots`, which are read off the party) and
    /// in nothing else.
    pub fn new(
        kept: &'k Kept,
        position: &Position,
        side: usize,
        slots: Vec<usize>,
        ours: Vec<Value>,
        theirs: Vec<Value>,
    ) -> Option<Like<'k>> {
        let before = &kept.position.sides.get(side)?.pokemon;
        let after = &position.sides.get(side)?.pokemon;
        if slots.is_empty()
            || before.len() != after.len()
            || slots.iter().any(|&s| s >= before.len() || before[s].slot != s || after[s].slot != s)
        {
            return None;
        }
        let mut probe = position.clone();
        for &s in &slots {
            probe.sides[side].pokemon[s] = before[s].clone();
        }
        probe.sides[side].mega_capable_slots = kept.position.sides[side].mega_capable_slots.clone();
        if probe != kept.position {
            return None;
        }
        Some(Like { kept, side, slots, ours, theirs })
    }

    /// Cell (i, j)'s turn in `position`, read off the kept node's cell with the same two
    /// actions: its branches with `position`'s Pokemon in the hidden slots. `None` -- and
    /// the cell is resolved -- when there is no such cell, when it paused, or when any
    /// branch left a hidden slot other than it found it (a Pokemon brought in, dragged in
    /// or touched), which is a turn the bench took part in.
    fn turn<'a>(&self, i: usize, j: usize, position: &Position) -> Option<TurnResult<'a>> {
        let ri = self.kept.ours.iter().position(|a| *a == self.ours[i])?;
        let rj = self.kept.theirs.iter().position(|a| *a == self.theirs[j])?;
        let cell = self.kept.cells.get(&(ri, rj))?;
        let before = &self.kept.position.sides[self.side];
        let after = &position.sides[self.side];
        let mut branches = Vec::with_capacity(cell.branches.len());
        for (probability, branch) in &cell.branches {
            let mine = &branch.sides[self.side];
            if mine.mega_capable_slots != before.mega_capable_slots
                || self.slots.iter().any(|&s| {
                    !std::rc::Rc::ptr_eq(&mine.pokemon[s], &before.pokemon[s])
                        && *mine.pokemon[s] != *before.pokemon[s]
                })
            {
                return None;
            }
            let mut made = branch.clone();
            for &s in &self.slots {
                made.sides[self.side].pokemon[s] = after.pokemon[s].clone();
            }
            made.sides[self.side].mega_capable_slots = after.mega_capable_slots.clone();
            branches.push(crate::resolve::Branch { probability: *probability, position: made, log: None });
        }
        Some(TurnResult {
            branches,
            exact: cell.exact,
            suspended: Vec::new(),
            unmodelled: cell.unmodelled.clone(),
        })
    }
}

/// `fill`, reading cells' turns off an earlier node's where `like` allows and keeping this
/// node's for a later one when `keep` (IKA-295). The leaves, their order, the sharing
/// between them and the encoding are `fill`'s: a turn read off is the branches the port
/// would have resolved, handed to the same collector.
pub fn fill_shared(
    reg: &Reg,
    encoder: &Encoder,
    request: &Request,
    like: Option<&Like>,
    keep: bool,
) -> (Value, Encoded, Vec<f64>, Vec<u8>, Option<Kept>) {
    fill_shared_on(reg, encoder, request, like, keep, crate::par::Pool::global())
}

/// Cells per thread in one of the pool's chunks (IKA-32).
const CELLS_PER_THREAD: usize = 32;

/// `fill_shared`, with its cells resolved on `pool` when there is one (IKA-32).
pub fn fill_shared_on<'r>(
    reg: &'r Reg,
    encoder: &Encoder,
    request: &Request,
    like: Option<&Like>,
    keep: bool,
    pool: Option<&crate::par::Pool>,
) -> (Value, Encoded, Vec<f64>, Vec<u8>, Option<Kept>) {
    let mut kept = keep.then(|| Kept {
        position: request.position.clone(),
        ours: request.ours_json.clone(),
        theirs: request.theirs_json.clone(),
        cells: HashMap::new(),
    });
    let mut read_off = 0usize;
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
        offered: 0,
    };
    // Resolving and encoding are different costs with different fixes, and the split was
    // being guessed at from two benchmarks that did not add up. It goes back in the header.
    let resolve_started = std::time::Instant::now();

    // What the loop below does with a cell once its turn is in hand. On one thread the turn
    // is resolved right before this and the pauses resumed inside `turn_leaves`; on the pool
    // (IKA-32) both happened on a cell thread and arrive as `resumed`. Either way the cells
    // come through here in `wanted_cells()`'s order, so the leaves, their sharing and the
    // spans are the same.
    let mut absorb = |i: usize, j: usize, result: TurnResult<'r>, resumed: Option<&Resumed<'r>>| {
            if let Some(kept) = kept.as_mut() {
                if !result.is_suspended() {
                    kept.cells.insert(
                        (i, j),
                        KeptCell {
                            exact: result.exact,
                            unmodelled: result.unmodelled.clone(),
                            branches: result
                                .branches
                                .iter()
                                .map(|b| (b.probability, b.position.clone()))
                                .collect(),
                        },
                    );
                }
            }
            exact[i][j] = result.exact;
            if result.is_suspended() {
                let root = match resumed {
                    Some(resumed) => collector.turn_leaves_resumed(&result, resumed, 0),
                    None => collector.turn_leaves(&result, 0),
                };
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
                return;
            }
            collector.notes.extend(result.unmodelled.iter().cloned());
            let total: f64 = result.branches.iter().map(|b| b.probability).sum();
            if result.branches.is_empty() || total <= 0.0 {
                cells.push((i, j, Cell::Empty));
                return;
            }
            let weights: Vec<f64> =
                result.branches.iter().map(|b| b.probability / total).collect();
            let indices: Vec<usize> = result
                .branches
                .into_iter()
                .map(|branch| collector.add_leaf(branch.position))
                .collect();
            cells.push((i, j, Cell::Span(indices, weights)));
    };

    let wanted = request.wanted_cells();
    match pool.filter(|_| !request.position_json.is_null()) {
        None => {
            for &(i, j) in &wanted {
                let reused = like.and_then(|like| like.turn(i, j, &request.position));
                if reused.is_some() {
                    read_off += 1;
                }
                let result = match reused {
                    Some(result) => result,
                    None => {
                        let actions = [request.ours[i].clone(), request.theirs[j].clone()];
                        match resolve_turn(reg, &request.position, &actions, request.budget) {
                            Err(reason) => {
                                refused.push(json!([i, j, reason]));
                                continue;
                            }
                            Ok(result) => result,
                        }
                    }
                };
                absorb(i, j, result, None);
            }
        }
        Some(pool) => {
            // A chunk at a time, so a width-48 node does not hold every cell's turn at once
            // before the first is collected; a chunk is enough items per thread that the
            // last one to finish does not keep the rest waiting long.
            let chunk_len = (pool.threads() * CELLS_PER_THREAD).max(1);
            let (ours, theirs, budget) = (&request.ours, &request.theirs, request.budget);
            let position_json = &request.position_json;
            for chunk in wanted.chunks(chunk_len) {
                // Reading a turn off an earlier node touches that node's positions, which
                // are this thread's: done here, before the pool is asked for the rest.
                let mut reused: Vec<Option<TurnResult<'r>>> = chunk
                    .iter()
                    .map(|&(i, j)| like.and_then(|like| like.turn(i, j, &request.position)))
                    .collect();
                let todo: Vec<usize> =
                    (0..chunk.len()).filter(|&k| reused[k].is_none()).collect();
                let resolved = pool.map_with(
                    todo.len(),
                    || Position::from_json(position_json),
                    |position, k| {
                        let (i, j) = chunk[todo[k]];
                        let actions = [ours[i].clone(), theirs[j].clone()];
                        resolve_turn(reg, position, &actions, budget).map(|result| {
                            let resumed = resume_all(reg, &result, 0);
                            (result, resumed)
                        })
                    },
                );
                let mut resolved = resolved.into_iter();
                for (k, &(i, j)) in chunk.iter().enumerate() {
                    match reused[k].take() {
                        Some(result) => {
                            read_off += 1;
                            absorb(i, j, result, None);
                        }
                        None => match resolved.next().expect("one answer per cell asked") {
                            Err(reason) => refused.push(json!([i, j, reason])),
                            Ok((result, resumed)) => absorb(i, j, result, Some(&resumed)),
                        },
                    }
                }
            }
        }
    }

    let resolve_us = resolve_started.elapsed().as_secs_f64() * 1e6;
    let encode_started = std::time::Instant::now();
    let borrowed: Vec<&Position> = collector.leaves.iter().collect();
    let encoded = encoder.encode_positions_with(&borrowed, request.encoding);
    let encode_us = encode_started.elapsed().as_secs_f64() * 1e6;

    // A node can want both: the learned leaf, and a parameter-free objective beside it as
    // a cross-check. The leaves are here and these are cheap, so they go back per leaf and
    // fold through the same spans rather than sending the whole node back for the sake of
    // the second column.
    let mut leaf_values: Vec<f64> = Vec::new();
    for name in &request.objectives {
        let score = crate::objective::by_name(name).expect("the request was checked");
        leaf_values.extend(collector.leaves.iter().map(score));
    }

    // The fold was the header's large part -- a cell's leaf indices and weights, one entry
    // per filled cell -- and it went as JSON until IKA-302 measured it. The spans now go
    // in the body behind the arrays and the leaf values (`span_block`); a fold tree, rare
    // (a cell whose turn paused), stays in the header.
    let fold_started = std::time::Instant::now();
    let mut spans: Vec<(usize, usize, Vec<usize>, Vec<f64>)> = Vec::new();
    let mut folded: Vec<Value> = Vec::new();
    for (i, j, cell) in cells {
        match cell {
            Cell::Empty => {}
            Cell::Span(indices, weights) => spans.push((i, j, indices, weights)),
            Cell::Folded(root) => folded.push(json!([i, j, root])),
        }
    }
    let span_at = packed_len(&encoded) + leaf_values.len() * 8;
    // The JSON road is kept for the test that holds the two to the same lists
    // (`jsonSpans` on the request); nothing on the production roads asks for it.
    let (span_bytes, span_leaves) = if request.json_spans {
        (Vec::new(), 0)
    } else {
        span_block(&spans)
    };
    let body_bytes = span_at + span_bytes.len();
    let fold_us = fold_started.elapsed().as_secs_f64() * 1e6;

    let mut header = json!({
        "kind": "encoded",
        "leaves": collector.leaves.len(),
        // Leaves handed to `add_leaf`, against the `leaves` above that were kept. The
        // ratio is the sharing's own account of itself, for a caller that would otherwise
        // have to infer it from how large the node was.
        "offered": collector.offered,
        // The spans' block in the body (`span_block`): where, how many, how many leaves.
        "spanAt": span_at,
        "spanCount": spans.len(),
        "spanLeaves": span_leaves,
        "folded": folded,
        "exact": exact,
        "refused": refused,
        "unmodelled": collector.notes.into_iter().collect::<Vec<_>>(),
        "unknownVolatiles": encoded.unknown_volatiles,
        "decided": crate::objective::decided_leaves(&collector.leaves),
        "leafObjectives": request.objectives,
        // The rule these arrays were encoded under, echoed so the caller can tell a binary
        // that applied it from one that never heard of it (IKA-141).
        "encoding": { "megaFromSlots": request.encoding.mega_from_slots },
        "monsPerSide": encoder.widths.mons_per_side,
        "monWidth": encoder.widths.mon,
        "sideWidth": encoder.widths.side,
        "fieldWidth": encoder.widths.field,
        "bytes": body_bytes,
        "resolveUs": resolve_us,
        "encodeUs": encode_us,
        "foldUs": fold_us,
    });
    if request.json_spans {
        let listed: Vec<Value> =
            spans.iter().map(|(i, j, indices, weights)| json!([i, j, indices, weights])).collect();
        let object = header.as_object_mut().expect("the header is an object");
        object.remove("spanAt");
        object.remove("spanCount");
        object.remove("spanLeaves");
        header["spans"] = Value::Array(listed);
    }
    if like.is_some() {
        // The cells whose turn was read off the earlier node rather than resolved.
        header["readOff"] = json!(read_off);
    }
    (header, encoded, leaf_values, span_bytes, kept)
}

/// A node's spans as one binary block (IKA-302), little-endian: every span's weights
/// (f64, bit for bit), then the spans' rows, columns and lengths (u32 each), then every
/// span's leaf indices (u32). Python's `rustnode._binary_spans` reads it back into the
/// lists the JSON header gave. Returns the block and the number of (index, weight) pairs.
pub fn span_block(spans: &[(usize, usize, Vec<usize>, Vec<f64>)]) -> (Vec<u8>, usize) {
    let leaves: usize = spans.iter().map(|(_, _, indices, _)| indices.len()).sum();
    let mut out = Vec::with_capacity(leaves * 12 + spans.len() * 12);
    for (_, _, _, weights) in spans {
        for w in weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    for (i, _, _, _) in spans {
        out.extend_from_slice(&(*i as u32).to_le_bytes());
    }
    for (_, j, _, _) in spans {
        out.extend_from_slice(&(*j as u32).to_le_bytes());
    }
    for (_, _, indices, weights) in spans {
        debug_assert_eq!(indices.len(), weights.len());
        out.extend_from_slice(&(indices.len() as u32).to_le_bytes());
    }
    for (_, _, indices, _) in spans {
        for index in indices {
            out.extend_from_slice(&(*index as u32).to_le_bytes());
        }
    }
    (out, leaves)
}

/// How many bytes `write_body` will write for the arrays, without building them.
fn packed_len(encoded: &Encoded) -> usize {
    (encoded.species.len() + encoded.ability.len() + encoded.item.len() + encoded.moves.len())
        * 4
        + (encoded.mon.len() + encoded.mask.len() + encoded.side.len() + encoded.field.len())
            * 4
}

/// Writes the buffers straight out, in the order the reader expects, little-endian.
///
/// Streamed rather than packed into a `Vec` first. A width-48 node is 40 to 60 MB, and
/// building that beside the arrays it is copied from doubled the peak for the largest
/// thing this process ever holds -- on a machine running fourteen of these, next to
/// fourteen Python workers holding a gigabyte each. The reusable window is 64 KB.
pub fn write_body<W: Write + ?Sized>(
    out: &mut W,
    encoded: &Encoded,
    leaf_values: &[f64],
    tail: &[u8],
) -> std::io::Result<()> {
    const WINDOW: usize = 1 << 16;
    let mut buffer: Vec<u8> = Vec::with_capacity(WINDOW + 8);

    macro_rules! stream {
        ($values:expr) => {
            for value in $values {
                buffer.extend_from_slice(&value.to_le_bytes());
                if buffer.len() >= WINDOW {
                    out.write_all(&buffer)?;
                    buffer.clear();
                }
            }
        };
    }
    stream!(&encoded.species);
    stream!(&encoded.ability);
    stream!(&encoded.item);
    stream!(&encoded.moves);
    stream!(&encoded.mon);
    stream!(&encoded.mask);
    stream!(&encoded.side);
    stream!(&encoded.field);
    stream!(leaf_values);
    if !buffer.is_empty() {
        out.write_all(&buffer)?;
    }
    // IKA-302: the spans' block, already bytes.
    if !tail.is_empty() {
        out.write_all(tail)?;
    }
    Ok(())
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
