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
    b64_decode, f64s, f64s_text, fold_value, folded_cells, numpy_dot, solve_bayesian, unsolved_json, FoldTree,
    GAMES, LPS,
};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

/// A node's span block, read once: every span's cell, leaves and weights.
struct Spans {
    rows: Vec<usize>,
    cols: Vec<usize>,
    starts: Vec<usize>,
    lengths: Vec<usize>,
    weights: Vec<f64>,
    indices: Vec<usize>,
}

impl Spans {
    /// The block `encoded_node::write_body` writes: weights (f64), then rows, columns and
    /// lengths (u32 each), then the leaf indices (u32), little-endian.
    fn read(block: &[u8], count: usize, total: usize) -> Result<Spans, String> {
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
        Err(why) => json!({ "error": why }),
    }
}

fn fold_node(value: &Value) -> Result<Value, String> {
    let m = value["rows"].as_u64().ok_or("beliefFold without rows")? as usize;
    let n = value["cols"].as_u64().ok_or("beliefFold without cols")? as usize;
    let values = f64s(&value["values"])?;
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
    let ref_folds = trees(reference["folded"].as_str())?;
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
                for (i, j, tree) in trees(own["folded"].as_str())? {
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
                for (i, j, tree) in trees(entry["folded"].as_str())? {
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

    // The games Python names: side `s`'s is over the matrices of the jobs of side `1 - s`, side
    // 1's as the transpose of the negation (`search.belief_solve`).
    let before = LPS.load(Ordering::Relaxed);
    let mut answers = Vec::new();
    for ask in value["solve"].as_array().map(Vec::as_slice).unwrap_or_default() {
        let side = ask["side"].as_i64().ok_or("a solve without side")?;
        let weights = f64s(&ask["weights"])?;
        let mut mats: Vec<(Vec<f64>, usize, usize)> = Vec::new();
        for (payoff, job_side) in matrices.iter().zip(&sides) {
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
}
