//! IKA-455 stage 1.4: a hidden bench's whole node in one request.
//!
//! `beliefnode.belief_payoffs` fills the reference node, decides from the actions which cells
//! can put a hidden Pokemon on the field (`reaches_bench`), which of those only the side's own
//! switches reach (`_reaching`), fills each completion's own cells while reading the rest off
//! another completion's fill (`_borrow`, `_entering`), stacks, scores and folds. Stages 1.2 and
//! 1.3 moved the last three into the port; this moves the rest, so the node costs one crossing
//! instead of about seven: Python sends the positions (held by number), the two action lists,
//! what Python alone reads off the actions (`rowFeat`, `colFeat`: whether a move forces a
//! switch or attacks, and which party members a switch brings in) and off each completion
//! (`jobs[].pk`: each Pokemon's species, base species and slot), and gets back the matrices,
//! the solved games and the notes.
//!
//! The plan is `beliefnode.py`'s, line for line: the same order of cells, the same keys, the
//! same order in which blocks are laid out (which is what makes the server's requests, and so
//! its values, the same). The pieces after the plan are stage 1.3's `served_node`: the plan
//! builds the request it takes and the port runs it, so the fold and the solve are the same
//! code. Fills are made on the thread that reads the request (`HELD` is thread-local).
//!
//! Request (positions and action lists as `fills`/`fill_encoded` send them; a position is a
//! `{"held": n}` or the whole JSON):
//!
//! ```text
//! { "kind": "beliefNode", "position": P, "p0": P, "p1": P, ...   // the root; completion k
//!   "ours": A, "theirs": A, "budget": B, "encoding": E?, "rows": m, "cols": n,
//!   "hidden": [[slots of side 0], [slots of side 1]], "carded": [bool, bool],
//!   "rowFeat": [F], "colFeat": [F],              // F = {"p": phazes, "a": attacks, "s": [[party index, species id]]}
//!   "jobs": [ { "side": s, "slots": [..], "exact": bool, "root": bool,
//!               "pk": [[species id, base species id, slot], ..] } ],  // in `spreads` order
//!   "solve", "server", "netScoresEnds", "patch" }                    // as `beliefServed`
//! ```

use crate::beliefnode::{hold_owned, served_node, Spans};
use crate::encode::{EncodeRules, Encoded, Encoder};
use crate::lp::b64_encode;
use crate::reg::Reg;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};

/// What Python reads off one action: whether it forces a switch (`_phazes`), attacks
/// (`_attacks`), and the party members (1-based) its switches bring in, with their species ids.
struct Feat {
    phazes: bool,
    attacks: bool,
    switches: Vec<(usize, String)>,
}

fn feats(v: &Value, what: &str) -> Result<Vec<Feat>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not a list"))?
        .iter()
        .map(|f| {
            let switches = f["s"]
                .as_array()
                .ok_or_else(|| format!("{what}: an action without `s`"))?
                .iter()
                .map(|s| {
                    let p = s.as_array().filter(|p| p.len() == 2).ok_or("a switch is not [index, species]")?;
                    Ok((
                        p[0].as_u64().ok_or("a switch's party index")? as usize,
                        p[1].as_str().ok_or("a switch's species")?.to_string(),
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Feat {
                phazes: f["p"].as_bool().ok_or_else(|| format!("{what}: an action without `p`"))?,
                attacks: f["a"].as_bool().ok_or_else(|| format!("{what}: an action without `a`"))?,
                switches,
            })
        })
        .collect()
}

fn slot_list(v: &Value, what: &str) -> Result<Vec<usize>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not a list"))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as usize).ok_or_else(|| format!("{what} holds a non-number")))
        .collect()
}

/// `reaches_bench`: the (rows x cols) mask of cells where a hidden Pokemon could be put on the
/// field. `hidden[s]` is side `s`'s hidden slots; `carded[s]` whether side `s` has a Red Card on
/// the field.
fn reaches(rows: &[Feat], cols: &[Feat], hidden: [&[usize]; 2], carded: [bool; 2]) -> Vec<bool> {
    let ours: HashSet<usize> = hidden[0].iter().copied().collect();
    let theirs: HashSet<usize> = hidden[1].iter().copied().collect();
    // The choice string numbers party members from 1, `Pokemon.slot` from 0.
    let switches = |f: &Feat, hidden: &HashSet<usize>| f.switches.iter().any(|(index, _)| index.checked_sub(1).is_some_and(|slot| hidden.contains(&slot)));
    let row_switches: Vec<bool> = rows.iter().map(|f| switches(f, &ours)).collect();
    let col_switches: Vec<bool> = cols.iter().map(|f| switches(f, &theirs)).collect();
    let mut mask = vec![false; rows.len() * cols.len()];
    for (i, r) in rows.iter().enumerate() {
        for (j, c) in cols.iter().enumerate() {
            mask[i * cols.len() + j] = row_switches[i]
                || col_switches[j]
                || (r.phazes && !theirs.is_empty())
                || (c.phazes && !ours.is_empty())
                || (carded[1] && r.attacks && !ours.is_empty())
                || (carded[0] && c.attacks && !theirs.is_empty());
        }
    }
    mask
}

/// `_choosers`: the sides that choose a replacement anywhere in a fold tree.
fn choosers(tree: &Value, found: &mut BTreeSet<i64>) {
    let mut stack = vec![tree];
    while let Some(node) = stack.pop() {
        if let Some(side) = node.get("best") {
            if let Some(s) = side.as_i64() {
                found.insert(s);
            }
            if let Some(options) = node["options"].as_array() {
                stack.extend(options.iter());
            }
        } else if let Some(parts) = node.get("avg").and_then(Value::as_array) {
            stack.extend(parts.iter().filter_map(|p| p.get(1)));
        }
    }
}

/// How a dirty cell reaches `side`'s own bench: through which of its switches
/// (`Some(party index, species)`), or by anything (`None`: phazing, a Red Card, a replacement).
type Reach = HashMap<(usize, usize), Option<Vec<(usize, String)>>>;

/// `_reaching`.
#[allow(clippy::too_many_arguments)]
fn reaching(
    rows: &[Feat],
    cols: &[Feat],
    side: usize,
    slots: &[usize],
    carded: [bool; 2],
    folded: &[(usize, usize, Value)],
    dirty: &[bool],
) -> Result<Reach, String> {
    let n = cols.len();
    let mine = if side == 0 {
        reaches(rows, cols, [slots, &[]], carded)
    } else {
        reaches(rows, cols, [&[], slots], carded)
    };
    let against = carded[1 - side];
    let mut out = Reach::new();
    for i in 0..rows.len() {
        for j in 0..n {
            if !mine[i * n + j] {
                continue;
            }
            let (own, other) = if side == 0 { (&rows[i], &cols[j]) } else { (&cols[j], &rows[i]) };
            let switches: Vec<(usize, String)> =
                own.switches.iter().filter(|(index, _)| index.checked_sub(1).is_some_and(|slot| slots.contains(&slot))).cloned().collect();
            let anywhere = other.phazes || (against && own.attacks);
            out.insert((i, j), if anywhere || switches.is_empty() { None } else { Some(switches) });
        }
    }
    for (i, j, root) in folded {
        let mut who = BTreeSet::new();
        choosers(root, &mut who);
        if who.contains(&(side as i64)) {
            out.insert((*i, *j), None);
        }
    }
    if out.keys().any(|(i, j)| !dirty[i * n + j]) {
        return Err("a cell reaching a bench was not dirty".into());
    }
    Ok(out)
}

/// One completion's party, as `_entering` reads it.
struct Party {
    /// (species id, base species id, slot) of each Pokemon of the side, in party order.
    mons: Vec<(String, String, i64)>,
}

/// `_entering`: which of this completion's slots the switches bring in, and who stands there.
fn entering(party: &Party, hidden: &HashSet<usize>, switches: &[(usize, String)]) -> Option<Vec<(usize, String)>> {
    let mut out: Vec<(usize, String)> = Vec::new();
    for (party_index, wanted) in switches {
        let found = party
            .mons
            .iter()
            .position(|(species, base, _)| species == wanted || base == wanted)
            .map(|q| q as i64)
            .unwrap_or(*party_index as i64 - 1);
        if found < 0 || found as usize >= party.mons.len() {
            return None;
        }
        let q = found as usize;
        if party.mons[q].2 != found || !hidden.contains(&q) {
            return None;
        }
        out.push((q, party.mons[q].0.clone()));
    }
    let distinct: HashSet<usize> = out.iter().map(|(q, _)| *q).collect();
    if distinct.len() != out.len() {
        return None;
    }
    Some(out)
}

/// A fill this node made and kept: its spans by cell, once something borrows from them.
struct OwnFill {
    ticket: u64,
    leaves: usize,
    bytes: Vec<u8>,
    count: usize,
    total: usize,
    folded: Value,
    spans: Spans,
    by_cell: Option<HashMap<(usize, usize), (Vec<usize>, Vec<f64>)>>,
    fold_cells: HashSet<(usize, usize)>,
}

enum Borrowed {
    /// The reference's own cells, read at this completion's rows.
    Shared { spans: Vec<(usize, usize)>, folds: Vec<(usize, usize)> },
    /// Another completion's fill cut to `part`, its cells' spans re-indexed (as `_packed`).
    Lent { part: usize, packed: Option<(Vec<u8>, usize, usize)> },
}

struct Job {
    side: usize,
    slots: Vec<usize>,
    exact: bool,
    root: bool,
    party: Party,
    shared: usize,
    ref_dirty: bool,
    dirty_part: Option<usize>,
    borrowed: Vec<Borrowed>,
    resolved: Option<usize>,
    filled: Option<OwnFill>,
}

/// `_packed`: spans `(row, column, leaf indices, weights)` as a span block, the empty ones left out.
fn packed(spans: &[(usize, usize, Vec<usize>, Vec<f64>)]) -> Option<(Vec<u8>, usize, usize)> {
    let kept: Vec<_> = spans.iter().filter(|s| !s.3.is_empty()).collect();
    if kept.is_empty() {
        return None;
    }
    let total: usize = kept.iter().map(|s| s.2.len()).sum();
    let mut bytes = Vec::with_capacity(total * 12 + kept.len() * 12);
    for s in &kept {
        for w in &s.3 {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
    }
    for s in &kept {
        bytes.extend_from_slice(&(s.0 as u32).to_le_bytes());
    }
    for s in &kept {
        bytes.extend_from_slice(&(s.1 as u32).to_le_bytes());
    }
    for s in &kept {
        bytes.extend_from_slice(&(s.2.len() as u32).to_le_bytes());
    }
    for s in &kept {
        for k in &s.2 {
            bytes.extend_from_slice(&(*k as u32).to_le_bytes());
        }
    }
    Some((bytes, kept.len(), total))
}

/// The completion's own row for each slot, as `_Patcher.desc` sends it.
fn source_rows(encoded: &Encoded, m: usize, f: usize, side: usize, slots: &[usize]) -> Value {
    let at = |slot: usize| side * m + slot;
    let ints = |v: &[i32], per: usize| -> String {
        let mut bytes = Vec::new();
        for &slot in slots {
            for x in &v[at(slot) * per..at(slot) * per + per] {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        b64_encode(&bytes)
    };
    let floats = |v: &[f32], per: usize| -> String {
        let mut bytes = Vec::new();
        for &slot in slots {
            for x in &v[at(slot) * per..at(slot) * per + per] {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        b64_encode(&bytes)
    };
    json!({
        "species": ints(&encoded.species, 1),
        "ability": ints(&encoded.ability, 1),
        "item": ints(&encoded.item, 1),
        "moves": ints(&encoded.moves, 4),
        "mon": floats(&encoded.mon, f),
        "mask": floats(&encoded.mask, 1),
    })
}

/// `beliefNode`: see the module's text.
pub fn belief_node(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    match node(reg, encoder, value) {
        Ok(answer) => answer,
        Err(failed) => failed,
    }
}

struct Plan<'a, 'b> {
    reg: &'a Reg,
    encoder: &'a Encoder<'b>,
    value: &'a Value,
    rules: EncodeRules,
    fills: Vec<Value>,
    notes: BTreeSet<String>,
    next_ticket: u64,
}

/// `_gather_dirty`: this completion's own fill of `cells`, kept in the port and booked as a block.
/// The completion that is the true position reads the reference's leaves instead. A refused
/// cell stops the node: its answer is returned for Python to raise.
fn gather(
    plan: &mut Plan,
    job: &mut Job,
    jidx: usize,
    cells: &[(usize, usize)],
    parts: &mut Vec<(usize, Value)>,
) -> Result<Option<Value>, Value> {
    if job.exact && job.root {
        job.ref_dirty = true;
        return Ok(None);
    }
    let ticket = plan.next_ticket;
    plan.next_ticket += 1;
    let (header, bytes, encoded) = plan.fill(&format!("p{jidx}"), Some(cells), ticket, "dirty")?;
    if header["refused"].as_array().is_some_and(|r| !r.is_empty()) {
        return Ok(Some(json!({ "refused": header["refused"], "fills": plan.fills })));
    }
    let leaves = header["leaves"].as_u64().unwrap_or(0) as usize;
    let count = header["spanCount"].as_u64().unwrap_or(0) as usize;
    let total = header["spanLeaves"].as_u64().unwrap_or(0) as usize;
    let spans = Spans::read(&bytes, count, total).map_err(|e| json!({ "fault": e }))?;
    let folded = header["folded"].clone();
    let fold_cells: HashSet<(usize, usize)> = folded
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|c| Some((c.get(0)?.as_u64()? as usize, c.get(1)?.as_u64()? as usize)))
                .collect()
        })
        .unwrap_or_default();
    hold_owned(ticket, false, encoded, &header);
    job.filled = Some(OwnFill { ticket, leaves, bytes, count, total, folded, spans, by_cell: None, fold_cells });
    parts.push((leaves, json!({ "ticket": ticket })));
    job.dirty_part = Some(parts.len() - 1);
    Ok(None)
}

impl Plan<'_, '_> {
    /// One fill of `position_key`'s position over `cells` (all when None), kept under `ticket`.
    fn fill(&mut self, position_key: &str, cells: Option<&[(usize, usize)]>, ticket: u64, purpose: &str)
        -> Result<(Value, Vec<u8>, Encoded), Value>
    {
        let error = |e: String| json!({ "fault": e });
        let v = self.value;
        let mut request = json!({
            "position": v[position_key],
            "ours": v["ours"],
            "theirs": v["theirs"],
            "budget": v["budget"],
            "objectives": [],
            "encode": true,
        });
        if let Some(e) = v.get("encoding").filter(|e| !e.is_null()) {
            request["encoding"] = e.clone();
        }
        if let Some(cells) = cells {
            request["cells"] = json!(cells.iter().map(|(i, j)| json!([i, j])).collect::<Vec<_>>());
        }
        #[cfg(feature = "dump-fills")]
        crate::resolvebench::dump_fill(&request);
        let parsed = crate::node::parse_request(&request).map_err(error)?;
        if &*parsed.position.format != self.reg.format_id.as_str() {
            return Err(error(format!(
                "position is {} but the regulation is {}",
                parsed.position.format, self.reg.format_id
            )));
        }
        let (header, encoded, _leaf_values, spans) = crate::encoded_node::fill(self.reg, self.encoder, &parsed);
        let cell_count = match cells {
            Some(c) => c.len(),
            None => v["rows"].as_u64().unwrap_or(0) as usize * v["cols"].as_u64().unwrap_or(0) as usize,
        };
        self.fills.push(json!({
            "purpose": purpose,
            "leaves": header["leaves"],
            "offered": header["offered"],
            "cells": cell_count,
            "resolveUs": header["resolveUs"],
            "encodeUs": header["encodeUs"],
            "bindUs": header["bindUs"],
            "mega": header["encoding"]["megaFromSlots"],
            "ticket": ticket,
        }));
        if let Some(listed) = header["unmodelled"].as_array() {
            for note in listed.iter().filter_map(Value::as_str) {
                self.notes.insert(note.to_string());
            }
        }
        Ok((header, spans, encoded))
    }
}

fn node(reg: &Reg, encoder: &Encoder, value: &Value) -> Result<Value, Value> {
    let error = |e: String| json!({ "fault": e });
    let m = value["rows"].as_u64().ok_or_else(|| error("beliefNode without rows".into()))? as usize;
    let n = value["cols"].as_u64().ok_or_else(|| error("beliefNode without cols".into()))? as usize;
    let row_feat = feats(&value["rowFeat"], "rowFeat").map_err(error)?;
    let col_feat = feats(&value["colFeat"], "colFeat").map_err(error)?;
    if row_feat.len() != m || col_feat.len() != n {
        return Err(error("the action features are not rows and columns long".into()));
    }
    let hidden_lists = value["hidden"].as_array().filter(|h| h.len() == 2).ok_or_else(|| error("beliefNode without hidden".into()))?;
    let hidden: [Vec<usize>; 2] = [
        slot_list(&hidden_lists[0], "hidden[0]").map_err(error)?,
        slot_list(&hidden_lists[1], "hidden[1]").map_err(error)?,
    ];
    let carded_list = value["carded"].as_array().filter(|c| c.len() == 2).ok_or_else(|| error("beliefNode without carded".into()))?;
    let carded = [carded_list[0].as_bool().unwrap_or(false), carded_list[1].as_bool().unwrap_or(false)];
    let widths = &encoder.widths;
    let rules = EncodeRules {
        mega_from_slots: value
            .get("encoding")
            .and_then(|e| e.get("megaFromSlots"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    let mut plan = Plan { reg, encoder, value, rules, fills: Vec::new(), notes: BTreeSet::new(), next_ticket: 1 };

    // The reference fill, kept under ticket 0 (which clears whatever the last node held).
    let (header, ref_bytes, ref_encoded) = plan.fill("position", None, 0, "matrix")?;
    let echo = header["encoding"]["megaFromSlots"].clone();
    let refused_of = |h: &Value| h["refused"].as_array().is_some_and(|r| !r.is_empty());
    if refused_of(&header) {
        return Ok(json!({ "refused": header["refused"], "fills": plan.fills, "mega": echo }));
    }
    let ref_leaves = header["leaves"].as_u64().unwrap_or(0) as usize;
    let ref_count = header["spanCount"].as_u64().unwrap_or(0) as usize;
    let ref_total = header["spanLeaves"].as_u64().unwrap_or(0) as usize;
    let ref_spans = Spans::read(&ref_bytes, ref_count, ref_total).map_err(error)?;
    let ref_folded = header["folded"].as_array().cloned().unwrap_or_default();
    let mut folded_cells: Vec<(usize, usize, Value)> = Vec::with_capacity(ref_folded.len());
    for cell in &ref_folded {
        let p = cell.as_array().filter(|p| p.len() == 3).ok_or_else(|| error("a folded cell is not [row, column, tree]".into()))?;
        folded_cells.push((
            p[0].as_u64().ok_or_else(|| error("a folded cell's row".into()))? as usize,
            p[1].as_u64().ok_or_else(|| error("a folded cell's column".into()))? as usize,
            p[2].clone(),
        ));
    }
    hold_owned(0, true, ref_encoded, &header);

    // Which cells a hidden Pokemon could be put on the field in.
    let mut dirty = reaches(&row_feat, &col_feat, [&hidden[0], &hidden[1]], carded);
    for (i, j, _root) in &folded_cells {
        if *i >= m || *j >= n {
            return Err(error("a folded cell outside its node".into()));
        }
        dirty[i * n + j] = true;
    }
    let wanted: Vec<(usize, usize)> =
        (0..m).flat_map(|i| (0..n).map(move |j| (i, j))).filter(|(i, j)| dirty[i * n + j]).collect();
    // A cell's span and fold in the reference, as Python's dicts keep them (the last wins).
    let mut ref_span_at: HashMap<(usize, usize), usize> = HashMap::new();
    for s in 0..ref_spans.rows.len() {
        ref_span_at.insert((ref_spans.rows[s], ref_spans.cols[s]), s);
    }
    let ref_fold_cells: HashSet<(usize, usize)> = folded_cells.iter().map(|(i, j, _)| (*i, *j)).collect();

    // Which dirty cells reach each side's own bench (`rules.patch_shares_side` is Python's).
    let mut reach: [Option<Reach>; 2] = [None, None];
    for side in 0..2 {
        if !hidden[side].is_empty() {
            reach[side] = Some(reaching(&row_feat, &col_feat, side, &hidden[side], carded, &folded_cells, &dirty).map_err(error)?);
        }
    }
    let mut claimed: [HashMap<((usize, usize), Vec<(usize, String)>), usize>; 2] = [HashMap::new(), HashMap::new()];

    // The jobs, and the blocks of leaves in the order Python lays them out.
    let jobs_in = value["jobs"].as_array().ok_or_else(|| error("beliefNode without jobs".into()))?;
    let mut parts: Vec<(usize, Value)> = Vec::new();
    let mut roots: HashMap<usize, Encoded> = HashMap::new();
    let mut reference_block: Option<usize> = None;
    let mut jobs: Vec<Job> = Vec::with_capacity(jobs_in.len());
    let (f, mons) = (widths.mon, widths.mons_per_side);
    for (jidx, jd) in jobs_in.iter().enumerate() {
        let side = jd["side"].as_u64().ok_or_else(|| error("a job without side".into()))? as usize;
        let slots = slot_list(&jd["slots"], "a job's slots").map_err(error)?;
        let pk = jd["pk"].as_array().ok_or_else(|| error("a job without pk".into()))?;
        let party = Party {
            mons: pk
                .iter()
                .map(|p| {
                    let a = p.as_array().filter(|a| a.len() == 3).ok_or("a pk entry is not [species, base, slot]")?;
                    Ok((
                        a[0].as_str().ok_or("a pk species")?.to_string(),
                        a[1].as_str().ok_or("a pk base species")?.to_string(),
                        a[2].as_i64().ok_or("a pk slot")?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
                .map_err(error)?,
        };
        let mut job = Job {
            side,
            slots: slots.clone(),
            exact: jd["exact"].as_bool().unwrap_or(false),
            root: jd["root"].as_bool().unwrap_or(false),
            party,
            shared: 0,
            ref_dirty: false,
            dirty_part: None,
            borrowed: Vec::new(),
            resolved: None,
            filled: None,
        };
        // This completion's own rows for some slots (its root encoded once, as `_Patcher` does).
        let mut patch_of = |plan: &Plan, slots: &[usize]| -> Result<Value, Value> {
            if !roots.contains_key(&jidx) {
                let position = crate::held::position(&value[format!("p{jidx}")]);
                roots.insert(jidx, plan.encoder.encode_positions_with(&[&position], plan.rules));
            }
            Ok(json!({ "side": side, "slots": slots, "source": source_rows(&roots[&jidx], mons, f, side, slots) }))
        };
        if job.exact || slots.is_empty() {
            if reference_block.is_none() {
                parts.push((ref_leaves, json!({ "ticket": 0 })));
                reference_block = Some(parts.len() - 1);
            }
            job.shared = reference_block.unwrap();
        } else {
            let patch = patch_of(&plan, &slots)?;
            parts.push((ref_leaves, json!({ "ticket": 0, "rows": null, "patch": patch })));
            job.shared = parts.len() - 1;
        }
        let owns_bench = reach[side].is_some() && !job.exact;
        if !wanted.is_empty() && owns_bench {
            // `_borrow`.
            let reach_of = reach[side].as_ref().unwrap();
            let everything: HashSet<usize> = slots.iter().copied().collect();
            let mut shared_spans: Vec<(usize, usize)> = Vec::new();
            let mut shared_folds: Vec<(usize, usize)> = Vec::new();
            let mut lent_keys: Vec<(usize, Vec<usize>)> = Vec::new();
            let mut lent: HashMap<(usize, Vec<usize>), Vec<((usize, usize), (Vec<usize>, Vec<f64>))>> = HashMap::new();
            let mut own_cells: Vec<(usize, usize)> = Vec::new();
            for &cell in &wanted {
                let Some(how) = reach_of.get(&cell) else {
                    if ref_span_at.contains_key(&cell) {
                        shared_spans.push(cell);
                    } else if ref_fold_cells.contains(&cell) {
                        shared_folds.push(cell);
                    }
                    continue;
                };
                let party_ref = &job.party;
                let hidden_set: HashSet<usize> = slots.iter().copied().collect();
                let entering_here = how.as_ref().and_then(|sw| entering(party_ref, &hidden_set, sw));
                let Some(entering_here) = entering_here else {
                    own_cells.push(cell);
                    continue;
                };
                let key = (cell, entering_here.clone());
                let owner_of = claimed[side].get(&key).copied();
                if let Some(owner) = owner_of {
                    let owner_fill: &mut OwnFill = jobs[owner].filled.as_mut().ok_or_else(|| error("a claim by a job with no fill".into()))?;
                    if owner_fill.by_cell.is_none() {
                        let mut map = HashMap::new();
                        for s in 0..owner_fill.spans.rows.len() {
                            let (a, b) = (owner_fill.spans.starts[s], owner_fill.spans.starts[s] + owner_fill.spans.lengths[s]);
                            map.insert(
                                (owner_fill.spans.rows[s], owner_fill.spans.cols[s]),
                                (owner_fill.spans.indices[a..b].to_vec(), owner_fill.spans.weights[a..b].to_vec()),
                            );
                        }
                        owner_fill.by_cell = Some(map);
                    }
                    if let Some(span) = owner_fill.by_cell.as_ref().unwrap().get(&cell) {
                        let taken: HashSet<usize> = entering_here.iter().map(|(q, _)| *q).collect();
                        let mut rest: Vec<usize> = everything.difference(&taken).copied().collect();
                        rest.sort_unstable();
                        let k = (owner, rest);
                        if !lent.contains_key(&k) {
                            lent_keys.push(k.clone());
                        }
                        lent.entry(k).or_default().push((cell, span.clone()));
                        continue;
                    }
                    if !owner_fill.fold_cells.contains(&cell) {
                        continue; // an empty cell there: empty here, as it would be filled
                    }
                } else {
                    claimed[side].insert(key, jidx);
                }
                own_cells.push(cell);
            }
            if !shared_spans.is_empty() || !shared_folds.is_empty() {
                job.borrowed.push(Borrowed::Shared { spans: shared_spans, folds: shared_folds });
            }
            for k in lent_keys {
                let (owner, rest) = &k;
                let cells = &lent[&k];
                let mut rows: Vec<usize> = cells.iter().flat_map(|(_, (ix, _))| ix.iter().copied()).collect();
                rows.sort_unstable();
                rows.dedup();
                let at: HashMap<usize, usize> = rows.iter().enumerate().map(|(n, k)| (*k, n)).collect();
                let renumbered: Vec<(usize, usize, Vec<usize>, Vec<f64>)> = cells
                    .iter()
                    .map(|((i, j), (ix, w))| (*i, *j, ix.iter().map(|k| at[k]).collect(), w.clone()))
                    .collect();
                let owner_ticket = jobs[*owner].filled.as_ref().map(|o| o.ticket).unwrap_or(0);
                let patch = patch_of(&plan, rest)?;
                parts.push((rows.len(), json!({ "ticket": owner_ticket, "rows": rows, "patch": patch })));
                job.borrowed.push(Borrowed::Lent { part: parts.len() - 1, packed: packed(&renumbered) });
            }
            job.resolved = Some(own_cells.len());
            if !own_cells.is_empty() {
                if let Some(refused) = gather(&mut plan, &mut job, jidx, &own_cells, &mut parts)? {
                    return Ok(refused);
                }
            }
        } else if !wanted.is_empty() {
            if let Some(refused) = gather(&mut plan, &mut job, jidx, &wanted, &mut parts)? {
                return Ok(refused);
            }
            if job.ref_dirty && reference_block.is_none() {
                parts.push((ref_leaves, json!({ "ticket": 0 })));
                reference_block = Some(parts.len() - 1);
            }
        }
        jobs.push(job);
    }
    // Everything is planned. What `beliefServed` is asked is what Python asked it of (stage 1.3):
    // the blocks of leaves in their order, and each completion's windows on them.
    let mut starts = Vec::with_capacity(parts.len());
    let mut at_row = 0usize;
    for (rows, _desc) in &parts {
        starts.push(at_row);
        at_row += rows;
    }
    let at = |index: usize| json!([starts[index], parts[index].0]);
    let block = |bytes: &[u8], count: usize, total: usize| json!({ "spans": b64_encode(bytes), "count": count, "total": total });
    let jobs_json: Vec<Value> = jobs
        .iter()
        .map(|job| {
            let mut one = json!({
                "side": job.side,
                "shared": at(job.shared),
                "refDirty": job.ref_dirty,
                "borrowed": job.borrowed.iter().map(|b| match b {
                    Borrowed::Shared { spans, folds } => json!({
                        "at": at(job.shared),
                        "cells": {
                            "spans": spans.iter().map(|(i, j)| json!([i, j])).collect::<Vec<_>>(),
                            "folds": folds.iter().map(|(i, j)| json!([i, j])).collect::<Vec<_>>(),
                        },
                    }),
                    Borrowed::Lent { part, packed } => json!({
                        "at": at(*part),
                        "packed": packed.as_ref().map(|(bytes, count, total)| block(bytes, *count, *total)),
                        "folded": "",
                    }),
                }).collect::<Vec<_>>(),
            });
            if let (false, Some(own), Some(part)) = (job.ref_dirty, job.filled.as_ref(), job.dirty_part) {
                let mut b = block(&own.bytes, own.count, own.total);
                b["folded"] = own.folded.clone();
                b["at"] = at(part);
                one["own"] = b;
            }
            one
        })
        .collect();
    let mut reference = block(&ref_bytes, ref_count, ref_total);
    reference["folded"] = Value::Array(ref_folded);
    reference["at"] = reference_block.map(at).unwrap_or(Value::Null);
    let dirty_bytes: Vec<u8> = dirty.iter().map(|d| *d as u8).collect();
    let parts_json: Vec<Value> = parts
        .iter()
        .map(|(rows, desc)| {
            let mut d = desc.clone();
            d["count"] = json!(rows);
            d
        })
        .collect();
    let built = json!({
        "rows": m,
        "cols": n,
        "dirty": b64_encode(&dirty_bytes),
        "ref": reference,
        "jobs": jobs_json,
        "parts": parts_json,
        "solve": value["solve"],
        "server": value["server"],
        "netScoresEnds": value["netScoresEnds"],
        "patch": value["patch"],
    });
    let mut answer = served_node(widths, &built)?;
    let clean = dirty.iter().filter(|d| !**d).count();
    let redone: usize = if wanted.is_empty() {
        0
    } else {
        jobs.iter().map(|j| j.resolved.unwrap_or(wanted.len())).sum()
    };
    answer["shared"] = json!(clean * jobs.len());
    answer["redone"] = json!(redone);
    answer["unmodelled"] = json!(plan.notes.iter().collect::<Vec<_>>());
    answer["fills"] = json!(plan.fills);
    answer["mega"] = echo;
    // The rows `_patched` would have counted (`encoder.note`), one entry a patched block.
    answer["patched"] = json!(parts.iter().filter(|(_, d)| !d["patch"].is_null()).map(|(rows, _)| *rows).collect::<Vec<_>>());
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(phazes: bool, attacks: bool, switches: &[(usize, &str)]) -> Feat {
        Feat { phazes, attacks, switches: switches.iter().map(|(i, s)| (*i, s.to_string())).collect() }
    }

    #[test]
    fn a_cell_reaches_a_bench_by_a_switch_a_phazing_move_or_a_red_card() {
        // Rows: a switch to party member 3 (slot 2), a phazing move, an attack, a quiet move.
        let rows = [feat(false, false, &[(3, "a")]), feat(true, false, &[]), feat(false, true, &[]), feat(false, false, &[])];
        let cols = [feat(false, false, &[]), feat(false, false, &[(4, "b")])];
        // Side 0 hides slot 2; side 1 hides slot 3; nobody holds a Red Card.
        let mask = reaches(&rows, &cols, [&[2], &[3]], [false, false]);
        let at = |i: usize, j: usize| mask[i * 2 + j];
        assert!(at(0, 0) && at(0, 1), "own switch into a hidden slot");
        assert!(at(1, 0), "a phazing move drags the other side's bench in");
        assert!(!at(2, 0) && !at(3, 0), "nothing reaches a bench");
        assert!(at(2, 1) && at(3, 1), "the other side's switch into its hidden slot");
        // A Red Card on side 1's field spoils every cell where side 0 attacks, for side 0's slots.
        let carded = reaches(&rows, &cols, [&[2], &[3]], [false, true]);
        assert!(carded[2 * 2] && !carded[3 * 2]);
        // With side 0 hiding nothing the same attack reaches no bench.
        let none = reaches(&rows, &cols, [&[], &[3]], [false, true]);
        assert!(!none[2 * 2]);
    }

    #[test]
    fn a_switch_numbered_zero_reaches_no_bench_as_python_minus_one_is_in_no_set() {
        // A party index is 1-based; 0 would be slot -1 in Python, which no hidden set holds.
        // It must not wrap (release) or panic (debug).
        let rows = [feat(false, false, &[(0, "a")])];
        let cols = [feat(false, false, &[(0, "b")])];
        assert_eq!(reaches(&rows, &cols, [&[0, 1], &[0, 1]], [false, false]), vec![false]);
        let found = reaching(&rows, &cols, 0, &[0, 1], [false, false], &[], &[false]).unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn a_switch_brings_in_the_same_species_wherever_it_stands() {
        let party = Party {
            mons: vec![
                ("garchomp".into(), "garchomp".into(), 0),
                ("sylveon".into(), "sylveon".into(), 1),
                ("floette".into(), "floette".into(), 2),
            ],
        };
        let hidden: HashSet<usize> = [1, 2].into_iter().collect();
        // The true bench's Sylveon is looked up by species, wherever it sits in this completion.
        let got = entering(&party, &hidden, &[(3, "sylveon".into())]).unwrap();
        assert_eq!(got, vec![(1, "sylveon".to_string())]);
        // A species this completion lacks falls back to the party index (3 -> slot 2).
        let got = entering(&party, &hidden, &[(3, "tyranitar".into())]).unwrap();
        assert_eq!(got, vec![(2, "floette".to_string())]);
        // Two switches to one Pokemon, a slot that is not hidden, and one renumbered: none.
        assert!(entering(&party, &hidden, &[(2, "sylveon".into()), (3, "sylveon".into())]).is_none());
        assert!(entering(&party, &hidden, &[(1, "garchomp".into())]).is_none());
        let renumbered = Party { mons: vec![("a".into(), "a".into(), 5)] };
        assert!(entering(&renumbered, &[0].into_iter().collect(), &[(1, "a".into())]).is_none());
    }

    #[test]
    fn the_sides_a_fold_tree_lets_choose_are_found() {
        let tree = json!({ "avg": [[0.5, { "best": 1, "options": [{ "leaf": 0 }, { "leaf": 1 }] }], [0.5, { "leaf": 2 }]] });
        let mut who = BTreeSet::new();
        choosers(&tree, &mut who);
        assert_eq!(who.into_iter().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn spans_are_packed_in_the_blocks_layout_and_the_empty_ones_dropped() {
        let spans = vec![(0, 1, vec![2usize, 0], vec![0.25, 0.75]), (1, 1, vec![], vec![])];
        let (bytes, count, total) = packed(&spans).unwrap();
        assert_eq!((count, total), (1, 2));
        let read = Spans::read(&bytes, count, total).unwrap();
        assert_eq!((read.rows[0], read.cols[0], read.indices.clone(), read.weights.clone()), (0, 1, vec![2, 0], vec![0.25, 0.75]));
        assert!(packed(&[(0, 0, vec![], vec![])]).is_none());
    }
}
