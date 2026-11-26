//! IKA-455 stage 1.2: a belief node's matrices folded here, and its Bayesian games solved.
//!
//! `beliefnode.belief_payoffs` scores every completion's leaves in one batch and then, in
//! Python, folds each completion's matrix out of the values: its shared cells from the
//! reference fill's spans, read at the completion's own block of values; its dirty cells from
//! the reference's, its own fill's, or cells borrowed from another completion's block. That
//! loop was 2.5 ms of Python a `move.hidden` decision (29% of the node's Python), and the
//! 2 Bayesian games of `belief_solve` came after it.
//!
//! `beliefFold` does it from what the port itself wrote: the span blocks of the fills (the
//! bytes `fills` sent), the values, and for each completion where its blocks sit. The
//! arithmetic is `port._folded`'s: numpy's dot in the order `lp::numpy_dot` copies, a replaced
//! cell's tree by `lp::fold_value`. The matrices come back whole (`beliefnode.BeliefNode`
//! keeps them), and the games Python names are solved with `lp::solve_bayesian` in the same
//! crossing: side 1's game is the transpose of the negation, as `search.belief_solve` builds it.
//!
//! Request (all f64 arrays base64 of little-endian bytes, as `lp.rs` reads them):
//!
//! ```text
//! { "kind": "beliefFold", "rows": m, "cols": n, "values": f64s, "dirty": u8 m*n,
//!   "ref": { "spans", "count", "total", "folded": text, "at": [start, len] | null },
//!   "jobs": [ { "side": s, "shared": [start, len], "refDirty": bool,
//!               "own": null | { "spans", "count", "total", "folded", "at": [start, len] },
//!               "borrowed": [ { "at": [start, len],
//!                               "cells": null | { "spans": [[i, j]], "folds": [[i, j]] },
//!                               "packed": null | { "spans", "count", "total" },
//!                               "folded": text } ] } ],
//!   "solve": [ { "side": s, "weights": f64s } ] }
//! ```
//!
//! A `borrowed` entry with `cells` reads those cells of the reference's spans and folds (the
//! cells that do not reach the side's own bench); one with `packed` carries its own spans.

use crate::lp::{
    b64_decode, f64s, f64s_text, fold_tree_from_value, fold_value, folded_cells, numpy_dot, solve_bayesian,
    unsolved_json, FoldTree,
    GAMES, LPS,
};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

/// A node's span block, read once: every span's cell, leaves and weights.
pub(crate) struct Spans {
    pub(crate) rows: Vec<usize>,
    pub(crate) cols: Vec<usize>,
    pub(crate) starts: Vec<usize>,
    pub(crate) lengths: Vec<usize>,
    pub(crate) weights: Vec<f64>,
    pub(crate) indices: Vec<usize>,
}

impl Spans {
    /// The block `encoded_node::write_body` writes: weights (f64), then rows, columns and
    /// lengths (u32 each), then the leaf indices (u32), little-endian.
    pub(crate) fn read(block: &[u8], count: usize, total: usize) -> Result<Spans, String> {
        let need = total * 8 + count * 12 + total * 4;
        if block.len() < need {
            return Err(format!("a span block of {} bytes for {count} spans of {total} leaves", block.len()));
        }
        let u32_at = |at: usize| u32::from_le_bytes(block[at..at + 4].try_into().unwrap()) as usize;
        let rows_at = total * 8;
        let cols_at = rows_at + count * 4;
        let lengths_at = cols_at + count * 4;
        let indices_at = lengths_at + count * 4;
        let mut spans = Spans {
            rows: (0..count).map(|s| u32_at(rows_at + 4 * s)).collect(),
            cols: (0..count).map(|s| u32_at(cols_at + 4 * s)).collect(),
            starts: Vec::with_capacity(count),
            lengths: (0..count).map(|s| u32_at(lengths_at + 4 * s)).collect(),
            weights: (0..total).map(|t| f64::from_le_bytes(block[8 * t..8 * t + 8].try_into().unwrap())).collect(),
            indices: (0..total).map(|t| u32_at(indices_at + 4 * t)).collect(),
        };
        let mut at = 0;
        for length in &spans.lengths {
            spans.starts.push(at);
            at += length;
        }
        if at > total {
            return Err("span lengths past the span block".into());
        }
        Ok(spans)
    }

    fn of(value: &Value) -> Result<Spans, String> {
        let count = value["count"].as_u64().ok_or("a span block without count")? as usize;
        let total = value["total"].as_u64().ok_or("a span block without total")? as usize;
        let text = value["spans"].as_str().ok_or("a span block without spans")?;
        Spans::read(&b64_decode(text)?, count, total)
    }

    /// Span `s` against `values` (the block it reads): `values[indices] @ weights`, as numpy.
    fn dot(&self, s: usize, values: &[f64]) -> Result<f64, String> {
        let (start, end) = (self.starts[s], self.starts[s] + self.lengths[s]);
        let mut xs = Vec::with_capacity(end - start);
        for &leaf in &self.indices[start..end] {
            xs.push(*values.get(leaf).ok_or("a span's leaf past its block of values")?);
        }
        Ok(numpy_dot(&xs, &self.weights[start..end]))
    }
}

/// `[start, len]` of a block of the values, the slice it names.
fn window<'a>(values: &'a [f64], at: &Value) -> Result<&'a [f64], String> {
    let pair = at.as_array().filter(|p| p.len() == 2).ok_or("a block without [start, len]")?;
    let start = pair[0].as_u64().ok_or("a block's start")? as usize;
    let len = pair[1].as_u64().ok_or("a block's length")? as usize;
    values.get(start..start + len).ok_or_else(|| "a block past the values".to_string())
}

fn cells(value: &Value) -> Result<Vec<(usize, usize)>, String> {
    value
        .as_array()
        .ok_or("a cell list is not a list")?
        .iter()
        .map(|c| {
            let p = c.as_array().filter(|p| p.len() == 2).ok_or("a cell is not [row, column]")?;
            Ok((p[0].as_u64().ok_or("a cell's row")? as usize, p[1].as_u64().ok_or("a cell's column")? as usize))
        })
        .collect()
}

/// A fill's replacement cells: the trees by cell (the last one stands, as a dict's would).
fn trees(text: Option<&str>) -> Result<Vec<(usize, usize, FoldTree)>, String> {
    match text {
        None | Some("") => Ok(Vec::new()),
        Some(text) => folded_cells(text),
    }
}

/// `trees` of a request's `folded`: the JSON text Python wrote, or (stage 1.4, where the port
/// built the node itself) the header's own list of `[row, column, tree]`.
fn trees_v(v: &Value) -> Result<Vec<(usize, usize, FoldTree)>, String> {
    match v {
        Value::String(text) => trees(Some(text)),
        Value::Array(cells) => cells
            .iter()
            .map(|cell| {
                let p = cell.as_array().filter(|p| p.len() == 3).ok_or("a folded cell is not [row, column, tree]")?;
                Ok((
                    p[0].as_u64().ok_or("a folded cell's row")? as usize,
                    p[1].as_u64().ok_or("a folded cell's column")? as usize,
                    fold_tree_from_value(&p[2])?,
                ))
            })
            .collect(),
        _ => Ok(Vec::new()),
    }
}

fn put(part: &mut [f64], m: usize, n: usize, i: usize, j: usize, v: f64) -> Result<(), String> {
    if i >= m || j >= n {
        return Err("a cell outside its node".into());
    }
    part[i * n + j] = v;
    Ok(())
}

/// `beliefFold`: see the module's text.
pub fn belief_fold(value: &Value) -> Value {
    match fold_node(value) {
        Ok(answer) => answer,
        Err(why) => json!({ "fault": why }),
    }
}

fn fold_node(value: &Value) -> Result<Value, String> {
    fold_with(value, f64s(&value["values"])?)
}

fn fold_with(value: &Value, values: Vec<f64>) -> Result<Value, String> {
    let m = value["rows"].as_u64().ok_or("beliefFold without rows")? as usize;
    let n = value["cols"].as_u64().ok_or("beliefFold without cols")? as usize;
    let dirty: Vec<bool> = b64_decode(value["dirty"].as_str().ok_or("beliefFold without dirty")?)?
        .iter()
        .map(|b| *b != 0)
        .collect();
    if dirty.len() != m * n {
        return Err("the dirty mask is not rows x cols".into());
    }
    let wanted = dirty.iter().any(|d| *d);
    let reference = &value["ref"];
    let ref_spans = Spans::of(reference)?;
    let ref_folds = trees_v(&reference["folded"])?;
    // A cell's span and tree in the reference, as Python's dicts keep them (the last wins).
    let mut ref_span_at: Vec<Option<usize>> = vec![None; m * n];
    for s in 0..ref_spans.rows.len() {
        let (i, j) = (ref_spans.rows[s], ref_spans.cols[s]);
        if i >= m || j >= n {
            return Err("a span outside its node".into());
        }
        ref_span_at[i * n + j] = Some(s);
    }
    let mut ref_fold_at: Vec<Option<usize>> = vec![None; m * n];
    for (k, (i, j, _)) in ref_folds.iter().enumerate() {
        if *i >= m || *j >= n {
            return Err("a folded cell outside its node".into());
        }
        ref_fold_at[i * n + j] = Some(k);
    }
    let jobs = value["jobs"].as_array().ok_or("beliefFold without jobs")?;
    let mut matrices: Vec<Vec<f64>> = Vec::with_capacity(jobs.len());
    let mut sides: Vec<i64> = Vec::with_capacity(jobs.len());
    for job in jobs {
        sides.push(job["side"].as_i64().ok_or("a job without side")?);
        let shared = window(&values, &job["shared"])?;
        let mut payoff = vec![0.0; m * n];
        // The cells no bench reaches: the reference's turn, read at this completion's rows.
        for s in 0..ref_spans.rows.len() {
            let (i, j) = (ref_spans.rows[s], ref_spans.cols[s]);
            if ref_spans.lengths[s] > 0 && !dirty[i * n + j] {
                payoff[i * n + j] = ref_spans.dot(s, shared)?;
            }
        }
        if wanted {
            let mut part = vec![0.0; m * n];
            if job["refDirty"].as_bool().unwrap_or(false) {
                let at = window(&values, &reference["at"])?;
                for s in 0..ref_spans.rows.len() {
                    let (i, j) = (ref_spans.rows[s], ref_spans.cols[s]);
                    if ref_spans.lengths[s] > 0 && dirty[i * n + j] {
                        part[i * n + j] = ref_spans.dot(s, at)?;
                    }
                }
                for (i, j, tree) in &ref_folds {
                    put(&mut part, m, n, *i, *j, fold_value(tree, at)?)?;
                }
            } else if let Some(own) = job.get("own").filter(|o| !o.is_null()) {
                let spans = Spans::of(own)?;
                let at = window(&values, &own["at"])?;
                for s in 0..spans.rows.len() {
                    if spans.lengths[s] > 0 {
                        put(&mut part, m, n, spans.rows[s], spans.cols[s], spans.dot(s, at)?)?;
                    }
                }
                for (i, j, tree) in trees_v(&own["folded"])? {
                    put(&mut part, m, n, i, j, fold_value(&tree, at)?)?;
                }
            }
            for entry in job["borrowed"].as_array().map(Vec::as_slice).unwrap_or_default() {
                let at = window(&values, &entry["at"])?;
                if let Some(c) = entry.get("cells").filter(|c| !c.is_null()) {
                    for (i, j) in cells(&c["spans"])? {
                        if let Some(Some(s)) = ref_span_at.get(i * n + j) {
                            if ref_spans.lengths[*s] > 0 {
                                put(&mut part, m, n, i, j, ref_spans.dot(*s, at)?)?;
                            }
                        }
                    }
                    for (i, j) in cells(&c["folds"])? {
                        if let Some(Some(k)) = ref_fold_at.get(i * n + j) {
                            put(&mut part, m, n, i, j, fold_value(&ref_folds[*k].2, at)?)?;
                        }
                    }
                }
                if let Some(p) = entry.get("packed").filter(|p| !p.is_null()) {
                    let spans = Spans::of(p)?;
                    for s in 0..spans.rows.len() {
                        if spans.lengths[s] > 0 {
                            put(&mut part, m, n, spans.rows[s], spans.cols[s], spans.dot(s, at)?)?;
                        }
                    }
                }
                for (i, j, tree) in trees_v(&entry["folded"])? {
                    put(&mut part, m, n, i, j, fold_value(&tree, at)?)?;
                }
            }
            for (cell, is_dirty) in dirty.iter().enumerate() {
                if *is_dirty {
                    payoff[cell] = part[cell];
                }
            }
        }
        matrices.push(payoff);
    }

    solve_games(value, &matrices, &sides, m, n)
}

/// The games `value["solve"]` names, over `matrices` (job k's, `sides[k]` its side), and the
/// answer `beliefFold` gives: the matrices and the solved games. IKA-457 stage 2.2's replacement
/// node uses it too.
pub(crate) fn solve_games(
    value: &Value,
    matrices: &[Vec<f64>],
    sides: &[i64],
    m: usize,
    n: usize,
) -> Result<Value, String> {
    // The games Python names: side `s`'s is over the matrices of the jobs of side `1 - s`, side
    // 1's as the transpose of the negation (`search.belief_solve`).
    let before = LPS.load(Ordering::Relaxed);
    let mut answers = Vec::new();
    for ask in value["solve"].as_array().map(Vec::as_slice).unwrap_or_default() {
        let side = ask["side"].as_i64().ok_or("a solve without side")?;
        let weights = f64s(&ask["weights"])?;
        let mut mats: Vec<(Vec<f64>, usize, usize)> = Vec::new();
        for (payoff, job_side) in matrices.iter().zip(sides) {
            if 1 - job_side != side {
                continue;
            }
            if side == 0 {
                mats.push((payoff.clone(), m, n));
            } else {
                let mut t = vec![0.0; m * n];
                for i in 0..m {
                    for j in 0..n {
                        t[j * m + i] = -payoff[i * n + j];
                    }
                }
                mats.push((t, n, m));
            }
        }
        GAMES.fetch_add(1, Ordering::Relaxed);
        answers.push(match solve_bayesian(&mats, &weights, 1e-9) {
            Ok(s) => json!({
                "side": side,
                "valueRow": s.value_row,
                "valueCol": s.value_col,
                "x": f64s_text(&s.x),
                "ys": s.ys.iter().map(|y| f64s_text(y)).collect::<Vec<_>>(),
            }),
            Err(why) => {
                let mut one = unsolved_json(why);
                one["side"] = json!(side);
                one
            }
        });
    }
    let flat: Vec<f64> = matrices.iter().flatten().copied().collect();
    Ok(json!({
        "kind": "beliefFold",
        "matrices": f64s_text(&flat),
        "answers": answers,
        "lps": LPS.load(Ordering::Relaxed) - before,
    }))
}

// ---------------------------------------------------------------------------
// Stage 1.3: the node's leaves held here, stacked and scored here
// ---------------------------------------------------------------------------

use crate::encode::{Encoded, Widths};
use std::cell::RefCell;
use std::collections::HashMap;

/// A fill kept in the port for the request that stacks it: its arrays and which of its leaves
/// the battle ended in (NaN where it did not, as `Encoded.decided`).
struct Held {
    encoded: Encoded,
    decided: Vec<f64>,
}

thread_local! {
    static HELD: RefCell<HashMap<u64, Held>> = RefCell::new(HashMap::new());
}

fn copy_of(e: &Encoded) -> Encoded {
    Encoded {
        species: e.species.clone(),
        ability: e.ability.clone(),
        item: e.item.clone(),
        moves: e.moves.clone(),
        mon: e.mon.clone(),
        mask: e.mask.clone(),
        side: e.side.clone(),
        field: e.field.clone(),
        unknown_volatiles: e.unknown_volatiles.clone(),
    }
}

/// Keep a fill under `ticket` (`clear` first drops every one held: a node's reference fill
/// starts the node). `header` is the fill's own header, whose `decided` says which leaves ended.
pub fn hold(ticket: u64, clear: bool, encoded: &Encoded, header: &Value) {
    let rows = header["leaves"].as_u64().unwrap_or(0) as usize;
    let mut decided = vec![f64::NAN; rows];
    for pair in header["decided"].as_array().map(Vec::as_slice).unwrap_or_default() {
        if let (Some(i), Some(v)) = (pair.get(0).and_then(Value::as_u64), pair.get(1).and_then(Value::as_f64)) {
            if (i as usize) < rows {
                decided[i as usize] = v;
            }
        }
    }
    HELD.with(|h| {
        let mut h = h.borrow_mut();
        if clear {
            h.clear();
        }
        h.insert(ticket, Held { encoded: copy_of(encoded), decided });
    });
}

/// `hold` for a fill the port made itself (stage 1.4): the arrays moved in, not copied.
pub(crate) fn hold_owned(ticket: u64, clear: bool, encoded: Encoded, header: &Value) {
    let rows = header["leaves"].as_u64().unwrap_or(0) as usize;
    let mut decided = vec![f64::NAN; rows];
    for pair in header["decided"].as_array().map(Vec::as_slice).unwrap_or_default() {
        if let (Some(i), Some(v)) = (pair.get(0).and_then(Value::as_u64), pair.get(1).and_then(Value::as_f64)) {
            if (i as usize) < rows {
                decided[i as usize] = v;
            }
        }
    }
    HELD.with(|h| {
        let mut h = h.borrow_mut();
        if clear {
            h.clear();
        }
        h.insert(ticket, Held { encoded, decided });
    });
}

/// Rows `rows` of `e`, in that order (`beliefnode._subset`).
fn subset(e: &Encoded, w: &Widths, rows: &[usize]) -> Result<Encoded, String> {
    let m = w.mons_per_side;
    let per = (2 * m, 2 * m, 2 * m, 2 * m * 4, 2 * m * w.mon, 2 * m, 2 * w.side, w.field);
    let n = e.species.len() / per.0.max(1);
    if let Some(bad) = rows.iter().find(|r| **r >= n) {
        return Err(format!("a subset row {bad} of {n}"));
    }
    fn pick<T: Copy>(v: &[T], per: usize, rows: &[usize]) -> Vec<T> {
        let mut out = Vec::with_capacity(rows.len() * per);
        for r in rows {
            out.extend_from_slice(&v[r * per..(r + 1) * per]);
        }
        out
    }
    Ok(Encoded {
        species: pick(&e.species, per.0, rows),
        ability: pick(&e.ability, per.1, rows),
        item: pick(&e.item, per.2, rows),
        moves: pick(&e.moves, per.3, rows),
        mon: pick(&e.mon, per.4, rows),
        mask: pick(&e.mask, per.5, rows),
        side: pick(&e.side, per.6, rows),
        field: pick(&e.field, per.7, rows),
        unknown_volatiles: e.unknown_volatiles.clone(),
    })
}

/// The names' places in a row, and `HP_SCALE`, as `encode.py` has them (`beliefnode._patched`).
struct PatchAt {
    can_mega: usize,
    maxhp_scaled: usize,
    hp_fraction: usize,
    fainted: usize,
    mega_used: usize,
    mega_available: usize,
    alive_fraction: usize,
    team_hp_fraction: usize,
    hp_scale: f64,
}

impl PatchAt {
    fn of(v: &Value) -> Result<PatchAt, String> {
        let at = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("a patch without `{k}`"));
        Ok(PatchAt {
            can_mega: at("can_mega")?,
            maxhp_scaled: at("maxhp_scaled")?,
            hp_fraction: at("hp_fraction")?,
            fainted: at("fainted")?,
            mega_used: at("mega_used")?,
            mega_available: at("mega_available")?,
            alive_fraction: at("alive_fraction")?,
            team_hp_fraction: at("team_hp_fraction")?,
            hp_scale: v["hp_scale"].as_f64().ok_or("a patch without `hp_scale`")?,
        })
    }
}

/// A list of whole numbers; anything else in it, or no list, is an error (no element is dropped).
fn usizes(v: &Value, what: &str) -> Result<Vec<usize>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not a list"))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as usize).ok_or_else(|| format!("{what} holds a non-number: {x}")))
        .collect()
}

fn i32s(v: &Value) -> Result<Vec<i32>, String> {
    let bytes = b64_decode(v.as_str().ok_or("an i32 array is not base64 text")?)?;
    Ok(bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect())
}

fn f32s(v: &Value) -> Result<Vec<f32>, String> {
    let bytes = b64_decode(v.as_str().ok_or("an f32 array is not base64 text")?)?;
    Ok(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
}

/// `beliefnode._patched` (the rebuilt-side-vector rule): `e`'s rows with side `side`'s hidden
/// `slots` replaced by the completion's (`source`: one row per slot, in `slots`' order), the
/// side's `can_mega` cleared where it has mega evolved, and its `mega_available`,
/// `alive_fraction` and `team_hp_fraction` rebuilt from the patched rows. numpy's float32
/// reads and float64 arithmetic are kept as they are (the HPs are whole numbers there, so the
/// sums are exact in any order).
fn patch(e: &mut Encoded, w: &Widths, at: &PatchAt, side: usize, slots: &[usize], source: &Value) -> Result<(), String> {
    let (m, f, s) = (w.mons_per_side, w.mon, w.side);
    let n = e.species.len() / (2 * m).max(1);
    let (species, ability, item) = (i32s(&source["species"])?, i32s(&source["ability"])?, i32s(&source["item"])?);
    let (moves, mon, mask) = (i32s(&source["moves"])?, f32s(&source["mon"])?, f32s(&source["mask"])?);
    let k = slots.len();
    if species.len() != k || ability.len() != k || item.len() != k || moves.len() != 4 * k || mon.len() != f * k || mask.len() != k {
        return Err("a patch source of the wrong shape".into());
    }
    if slots.iter().any(|slot| *slot >= m) {
        return Err("a patch slot past the party".into());
    }
    for r in 0..n {
        let base = (r * 2 + side) * m;
        for (t, &slot) in slots.iter().enumerate() {
            e.species[base + slot] = species[t];
            e.ability[base + slot] = ability[t];
            e.item[base + slot] = item[t];
            e.moves[(base + slot) * 4..(base + slot) * 4 + 4].copy_from_slice(&moves[4 * t..4 * t + 4]);
            e.mon[(base + slot) * f..(base + slot) * f + f].copy_from_slice(&mon[f * t..f * t + f]);
            e.mask[base + slot] = mask[t];
        }
        let sbase = (r * 2 + side) * s;
        let spent = e.side[sbase + at.mega_used] > 0.0;
        if spent {
            for &slot in slots {
                e.mon[(base + slot) * f + at.can_mega] = 0.0;
            }
        }
        let mut holder = false;
        let (mut count, mut alive) = (0i64, 0i64);
        let (mut maxhp_sum, mut hp_sum) = (0.0f64, 0.0f64);
        for j in 0..m {
            let present = e.mask[base + j] > 0.0;
            let row = (base + j) * f;
            if present && e.mon[row + at.can_mega] > 0.0 {
                holder = true;
            }
            if present {
                count += 1;
                if !(e.mon[row + at.fainted] > 0.0) {
                    alive += 1;
                }
                let maxhp = (e.mon[row + at.maxhp_scaled] as f64 * at.hp_scale).round_ties_even();
                let hp = (e.mon[row + at.hp_fraction] as f64 * maxhp).round_ties_even();
                maxhp_sum += maxhp;
                hp_sum += hp;
            }
        }
        e.side[sbase + at.mega_available] = if holder && !spent { 1.0 } else { 0.0 };
        e.side[sbase + at.alive_fraction] = (alive as f64 / count.max(1) as f64) as f32;
        let total = if maxhp_sum > 0.0 { maxhp_sum } else { 1.0 };
        e.side[sbase + at.team_hp_fraction] = (hp_sum / total) as f32;
    }
    Ok(())
}

pub(crate) fn server_failed(failed: crate::served::Failed) -> Value {
    match failed {
        crate::served::Failed::Refused { error, oom, cap_gb } => json!({ "serverError": error, "oom": oom, "capGb": cap_gb }),
        crate::served::Failed::Broken(error) => json!({ "serverError": error }),
    }
}

/// `beliefServed`: `beliefFold`'s request with the values not given but asked for. Its `parts`
/// are the node's blocks of leaves in the order Python stacks them (`beliefnode._stacked`);
/// each is a held fill (`ticket`), optionally cut to `rows` and patched
/// (`{side, slots, source}`). They go to the inference server as `RemoteValue.from_encoded`
/// sends a stacked batch (the same requests, cut where it cuts), the ended leaves are settled as
/// `encode.settle` does, and the values are folded as `beliefFold` folds them.
pub fn belief_served(widths: &Widths, value: &Value) -> Value {
    match served_node(widths, value) {
        Ok(answer) => answer,
        Err(failed) => failed,
    }
}

pub(crate) fn served_node(widths: &Widths, value: &Value) -> Result<Value, Value> {
    let error = |e: String| json!({ "fault": e });
    let server = crate::served::Server::from_json(&value["server"]).map_err(error)?;
    let net_scores_ends = value.get("netScoresEnds").and_then(Value::as_bool).unwrap_or(false);
    let at = PatchAt::of(&value["patch"]).map_err(error)?;
    let parts = value["parts"].as_array().ok_or_else(|| error("beliefServed without parts".into()))?;
    let mut stats = crate::served::Stats::default();
    let (mut values, decided) = HELD.with(|h| -> Result<(Vec<f64>, Vec<f64>), Value> {
        let h = h.borrow();
        enum Src {
            Held(u64),
            Owned(usize),
        }
        let mut owned: Vec<Encoded> = Vec::new();
        let mut sources: Vec<Src> = Vec::new();
        let mut rows: Vec<usize> = Vec::new();
        let mut decided: Vec<f64> = Vec::new();
        for part in parts {
            let ticket = part["ticket"].as_u64().ok_or_else(|| error("a part without a ticket".into()))?;
            let held = h.get(&ticket).ok_or_else(|| error(format!("no fill held as {ticket}")))?;
            let cut: Option<Vec<usize>> = match part.get("rows").filter(|r| !r.is_null()) {
                None => None,
                Some(list) => Some(usizes(list, "a part's rows").map_err(error)?),
            };
            let patching = part.get("patch").is_some_and(|p| !p.is_null());
            let mut part_decided: Vec<f64> = match &cut {
                Some(rs) => {
                    let mut got = Vec::with_capacity(rs.len());
                    for r in rs {
                        got.push(*held.decided.get(*r).ok_or_else(|| error(format!("a part's row {r} past the fill")))?);
                    }
                    got
                }
                None => held.decided.clone(),
            };
            let per_row = (2 * widths.mons_per_side).max(1);
            if cut.is_some() || patching {
                let mut block = match &cut {
                    Some(rs) => subset(&held.encoded, widths, rs).map_err(error)?,
                    None => copy_of(&held.encoded),
                };
                if patching {
                    let p = &part["patch"];
                    let side = p["side"].as_u64().ok_or_else(|| error("a patch without side".into()))? as usize;
                    let slots = usizes(&p["slots"], "a patch's slots").map_err(error)?;
                    patch(&mut block, widths, &at, side, &slots, &p["source"]).map_err(error)?;
                }
                rows.push(block.species.len() / per_row);
                sources.push(Src::Owned(owned.len()));
                owned.push(block);
            } else {
                rows.push(held.encoded.species.len() / per_row);
                sources.push(Src::Held(ticket));
            }
            if let Some(want) = part.get("count").and_then(Value::as_u64) {
                if *rows.last().unwrap() != want as usize {
                    return Err(error(format!("a part of {} rows where {want} were named", rows.last().unwrap())));
                }
            }
            decided.append(&mut part_decided);
        }
        let blocks: Vec<&Encoded> = sources
            .iter()
            .map(|s| match s {
                Src::Held(t) => &h[t].encoded,
                Src::Owned(i) => &owned[*i],
            })
            .collect();
        let batch = crate::served::begin(&server, widths, &blocks, &rows, &mut stats).map_err(server_failed)?;
        let values = crate::served::finish(&server, batch, &mut stats).map_err(server_failed)?;
        Ok((values, decided))
    })?;
    // `encode.settle`: an ended leaf is its result, after the pass.
    if decided.len() != values.len() {
        return Err(json!({ "fault": format!(
            "the server answered {} values for {} leaves", values.len(), decided.len()) }));
    }
    let mut ended = 0u64;
    for (i, d) in decided.iter().enumerate() {
        if !d.is_nan() {
            ended += 1;
            if !net_scores_ends {
                values[i] = *d;
            }
        }
    }
    let mut answer = fold_with(value, values).map_err(error)?;
    answer["served"] = json!({
        "requests": stats.requests,
        "rows": stats.rows,
        "ended": ended,
        "waitUs": stats.wait_ns as f64 / 1e3,
        "copyUs": stats.copy_ns as f64 / 1e3,
    });
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::{b64_encode, fold};

    fn block(spans: &[(u32, u32, Vec<u32>, Vec<f64>)]) -> (String, usize, usize) {
        let total: usize = spans.iter().map(|s| s.2.len()).sum();
        let mut bytes = Vec::new();
        for s in spans {
            for w in &s.3 {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
        }
        for s in spans {
            bytes.extend_from_slice(&s.0.to_le_bytes());
        }
        for s in spans {
            bytes.extend_from_slice(&s.1.to_le_bytes());
        }
        for s in spans {
            bytes.extend_from_slice(&(s.2.len() as u32).to_le_bytes());
        }
        for s in spans {
            for k in &s.2 {
                bytes.extend_from_slice(&k.to_le_bytes());
            }
        }
        (b64_encode(&bytes), spans.len(), total)
    }

    fn span_json(spans: &[(u32, u32, Vec<u32>, Vec<f64>)], folded: &str, at: [usize; 2]) -> Value {
        let (text, count, total) = block(spans);
        json!({ "spans": text, "count": count, "total": total, "folded": folded, "at": at })
    }

    #[test]
    fn a_shared_cell_reads_its_completions_block_and_a_dirty_one_its_own() {
        // 2 x 2. Reference: cell (0,0) over leaves [0, 1], cell (1,1) over [1] (dirty), (0,1)
        // empty; its fill has 2 leaves. Two completions, blocks at 2 and 4 of the 6 values.
        let reference = vec![
            (0u32, 0u32, vec![0u32, 1], vec![0.25, 0.75]),
            (0, 1, vec![], vec![]),
            (1, 1, vec![1], vec![1.0]),
        ];
        let own = vec![(1u32, 1u32, vec![0u32], vec![1.0])];
        let values = [0.1, 0.2, 0.3, 0.9, 0.5, 0.6];
        let mut dirty = [0u8; 4];
        dirty[3] = 1;
        let request = json!({
            "rows": 2, "cols": 2, "values": f64s_text(&values), "dirty": b64_encode(&dirty),
            "ref": span_json(&reference, "", [0, 2]),
            "jobs": [
                { "side": 0, "shared": [2, 2], "refDirty": true, "borrowed": [] },
                { "side": 0, "shared": [4, 2], "refDirty": false,
                  "own": span_json(&own, "", [4, 1]), "borrowed": [] },
            ],
            "solve": [],
        });
        let got = fold_node(&request).unwrap();
        let flat = f64s(&got["matrices"]).unwrap();
        // Job 0: (0,0) from its shared block [0.3, 0.9]; (1,1) from the reference's own [0.1, 0.2].
        let (rb, rc, rt) = {
            let (t, c, total) = block(&reference);
            (b64_decode(&t).unwrap(), c, total)
        };
        let whole = fold(&rb, rc, rt, &[0.3, 0.9], 2, 2).unwrap();
        assert_eq!(flat[0].to_bits(), whole[0].to_bits());
        assert_eq!(flat[1], 0.0);
        assert_eq!(flat[2], 0.0);
        assert_eq!(flat[3].to_bits(), 0.2f64.to_bits());
        // Job 1: (0,0) from [0.5, 0.6]; (1,1) from its own fill's one leaf at 4 (0.5).
        let second = fold(&rb, rc, rt, &[0.5, 0.6], 2, 2).unwrap();
        assert_eq!(flat[4].to_bits(), second[0].to_bits());
        assert_eq!(flat[7].to_bits(), 0.5f64.to_bits());
    }

    #[test]
    fn a_borrowed_cell_reads_the_references_span_at_the_borrowers_block() {
        // The dirty cell (1,1) does not reach this completion's bench: its span is the
        // reference's, read at the completion's own rows.
        let reference = vec![(0u32, 0u32, vec![0u32], vec![1.0]), (1, 1, vec![1], vec![1.0])];
        let values = [0.1, 0.2, 0.7, 0.8];
        let mut dirty = [0u8; 4];
        dirty[3] = 1;
        let request = json!({
            "rows": 2, "cols": 2, "values": f64s_text(&values), "dirty": b64_encode(&dirty),
            "ref": span_json(&reference, "", [0, 2]),
            "jobs": [ { "side": 1, "shared": [2, 2], "refDirty": false,
                        "borrowed": [ { "at": [2, 2], "cells": { "spans": [[1, 1]], "folds": [] } } ] } ],
            "solve": [],
        });
        let flat = f64s(&fold_node(&request).unwrap()["matrices"]).unwrap();
        assert_eq!(flat[0].to_bits(), 0.7f64.to_bits());
        assert_eq!(flat[3].to_bits(), 0.8f64.to_bits());
    }

    #[test]
    fn side_one_is_solved_as_the_transpose_of_the_negation() {
        // One class each side; the game of side 1 over a 2 x 3 matrix is its -A^T, 3 x 2.
        let reference = vec![
            (0u32, 0u32, vec![0u32], vec![1.0]),
            (0, 1, vec![1], vec![1.0]),
            (0, 2, vec![2], vec![1.0]),
            (1, 0, vec![3], vec![1.0]),
            (1, 1, vec![4], vec![1.0]),
            (1, 2, vec![5], vec![1.0]),
        ];
        let values = [0.3, 0.8, 0.6, 0.2, 0.5, 0.4];
        let request = json!({
            "rows": 2, "cols": 3, "values": f64s_text(&values),
            "dirty": b64_encode(&[0u8; 6]),
            "ref": span_json(&reference, "", [0, 6]),
            "jobs": [ { "side": 0, "shared": [0, 6], "refDirty": false, "borrowed": [] } ],
            "solve": [ { "side": 1, "weights": f64s_text(&[1.0]) } ],
        });
        let got = fold_node(&request).unwrap();
        assert_eq!(f64s(&got["matrices"]).unwrap(), values.to_vec());
        let negated: Vec<f64> = (0..3).flat_map(|j| (0..2).map(move |i| -values[i * 3 + j])).collect();
        let want = solve_bayesian(&[(negated, 3, 2)], &[1.0], 1e-9).unwrap();
        assert_eq!(got["answers"][0]["valueRow"].as_f64().unwrap().to_bits(), want.value_row.to_bits());
        assert_eq!(got["answers"][0]["x"].as_str().unwrap(), f64s_text(&want.x));
    }

    fn tiny() -> (Widths, PatchAt, Encoded) {
        // One row, two Pokemon a side, four mon columns, four side columns.
        let w = Widths { mon: 4, side: 4, field: 1, mons_per_side: 2 };
        let at = PatchAt {
            can_mega: 0,
            maxhp_scaled: 1,
            hp_fraction: 2,
            fainted: 3,
            mega_used: 0,
            mega_available: 1,
            alive_fraction: 2,
            team_hp_fraction: 3,
            hp_scale: 250.0,
        };
        let mut e = Encoded::default();
        e.species = vec![1, 2, 3, 4];
        e.ability = vec![0; 4];
        e.item = vec![0; 4];
        e.moves = vec![0; 16];
        // Side 0: slot 0 present (max HP 100, half HP), slot 1 hidden (a placeholder).
        e.mon = vec![0.0; 16];
        e.mon[1] = 0.4;
        e.mon[2] = 0.5;
        e.mask = vec![1.0, 1.0, 1.0, 1.0];
        e.side = vec![0.0; 8];
        e.field = vec![0.0];
        (w, at, e)
    }

    fn source() -> Value {
        // Slot 1's own row: holds a stone, max HP 50, full HP, standing.
        json!({
            "species": f32_free_i32s(&[9]),
            "ability": f32_free_i32s(&[8]),
            "item": f32_free_i32s(&[7]),
            "moves": f32_free_i32s(&[1, 2, 3, 4]),
            "mon": f32s_text(&[1.0, 0.2, 1.0, 0.0]),
            "mask": f32s_text(&[1.0]),
        })
    }

    fn f32_free_i32s(v: &[i32]) -> String {
        b64_encode(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
    }

    fn f32s_text(v: &[f32]) -> String {
        b64_encode(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
    }

    #[test]
    fn a_patched_row_takes_the_completions_bench_and_rebuilds_the_side() {
        let (w, at, mut e) = tiny();
        patch(&mut e, &w, &at, 0, &[1], &source()).unwrap();
        // The slot's row is the completion's; the other side and the other slot are not touched.
        assert_eq!((e.species[0], e.species[1], e.species[2]), (1, 9, 3));
        assert_eq!(&e.moves[4..8], &[1, 2, 3, 4]);
        assert_eq!(&e.mon[4..8], &[1.0, 0.2, 1.0, 0.0]);
        // A stone is held and the Mega is unspent; both stand; HP 50 + 50 of 100 + 50.
        assert_eq!(e.side[1], 1.0);
        assert_eq!(e.side[2], 1.0);
        assert_eq!(e.side[3], (100.0f64 / 150.0) as f32);
        // The other side's vector is as it was.
        assert_eq!(&e.side[4..8], &[0.0; 4]);
    }

    #[test]
    fn a_spent_mega_takes_the_stone_off_the_patched_row() {
        let (w, at, mut e) = tiny();
        e.side[0] = 1.0; // mega_used
        patch(&mut e, &w, &at, 0, &[1], &source()).unwrap();
        assert_eq!(e.mon[4], 0.0, "can_mega is cleared where the side has mega evolved");
        assert_eq!(e.side[1], 0.0);
    }

    #[test]
    fn a_part_read_wrongly_is_an_error_and_not_a_shorter_part() {
        let (w, _at, e) = tiny();
        hold(7, true, &e, &json!({ "leaves": 1, "decided": [] }));
        let request = |rows: Value, slots: Value| {
            json!({
                "server": { "address": "127.0.0.1:1", "model": "m", "shm": "s", "bytes": 1 },
                "patch": { "can_mega": 0, "maxhp_scaled": 1, "hp_fraction": 2, "fainted": 3, "mega_used": 0,
                           "mega_available": 1, "alive_fraction": 2, "team_hp_fraction": 3, "hp_scale": 250.0 },
                "parts": [ { "ticket": 7, "rows": rows,
                             "patch": { "side": 0, "slots": slots, "source": {} } } ],
            })
        };
        // A row that is not a number (it used to be dropped, leaving a shorter part).
        let got = belief_served(&w, &request(json!([0, "x"]), json!([1])));
        assert!(got["fault"].as_str().unwrap().contains("a part's rows holds a non-number"), "{got}");
        // A slot that is not a number.
        let got = belief_served(&w, &request(json!([0]), json!(["x"])));
        assert!(got["fault"].as_str().unwrap().contains("a patch's slots holds a non-number"), "{got}");
        // A row past the fill.
        let got = belief_served(&w, &request(json!([5]), json!([1])));
        assert!(got["fault"].as_str().is_some(), "{got}");
    }

    #[test]
    fn a_subset_keeps_its_rows_in_the_order_asked() {
        let (w, _at, mut e) = tiny();
        // Two rows: the second a copy with other species.
        e.species.extend_from_slice(&[5, 6, 7, 8]);
        e.ability.extend_from_slice(&[0; 4]);
        e.item.extend_from_slice(&[0; 4]);
        e.moves.extend_from_slice(&[0; 16]);
        e.mon.extend_from_slice(&[0.0; 16]);
        e.mask.extend_from_slice(&[1.0; 4]);
        e.side.extend_from_slice(&[0.0; 8]);
        e.field.push(0.5);
        let cut = subset(&e, &w, &[1, 0, 1]).unwrap();
        assert_eq!(cut.species, vec![5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(cut.field, vec![0.5, 0.0, 0.5]);
        assert!(subset(&e, &w, &[2]).is_err());
    }
}
