//! IKA-464: a move's best-first deepening, in the port (`deepen.py`'s ``m...sallh`` reading).
//!
//! A person's move (`humanplay.solve_move`) and the analysis mode read the depth-1 node and
//! then spend the rest of the budget one cell at a time, best first, re-solving the tree after
//! every step, so that wherever the budget stops the last equilibrium is an answer. Here the
//! whole read is the port's: the root's matrices (`beliefplan::belief_node`), the tree, the
//! priorities, the turns of the refined cells, the children's menus by the Q
//! (`qmenus::menus_of`), the children's matrices on the inference server, every LP (`lp.rs`)
//! and the root's swap oracle. The caller starts a read, runs it in slices of wall time or of
//! counted cells, and ends it:
//!
//! ```text
//! { "kind": "deepenStart",
//!   "node": B,                       // a `beliefNode` request (positions, jobs, server, ...)
//!   "side": s,                       // whose game is read
//!   "menus": [A0, A1],               // the root's menus, side 0's and side 1's (`Act` codes)
//!   "weights": [w..],                // the other side's completions (its jobs, in order)
//!   "outside": [A0, A1] | null,      // the swap oracle's candidates (`sall`: every legal action)
//!   "levels": L, "discount": d | null, "branches": 3, "childWidth": 8,
//!   "qServer": S, "properties": bool, "megaFromSlots": bool }
//! { "kind": "deepenRun", "id": n, "ms": t?, "cells": c?, "steps": k?, "grow": [A0, A1]? }
//! { "kind": "deepenEnd", "id": n }
//! ```
//!
//! `deepenStart` and `deepenRun` answer the read as it stands (`snapshot`). `cells` is the
//! read's whole budget in counted cells (a refined cell's turn, each child matrix's cells and
//! each cell the oracle fills), so a read on counted cells is the same read on any machine;
//! `ms` is this slice's wall time; `steps` caps the steps of this slice.
//!
//! **The root** is the side's Bayesian game over the other side's completions (an open root is
//! one completion of weight 1): one row set, a matrix per completion in this side's own
//! orientation (side 1's is side 0's negated transpose). Its cells are (completion, row,
//! column); refining one resolves the turn in that completion's position and reads the
//! branches as if the bench were known (IKA-111's determinization).
//!
//! **One step** (`deepen._expand`): the turn resolved with every branch, the `branches`
//! likeliest kept and renormalised, each live branch a child node: each side's `childWidth`
//! best by the Q (the root's `q-nocover` without the cover, IKA-307's ``c<k>``; `deepen.py`'s
//! default was `narrow`'s damage order, which is not in the port), the child's matrix on the
//! inference server and its zero-sum LP. A finished branch is its result. The cell's value is
//! its branches' values, weighted, and every node up to the root is solved again.
//!
//! **The priority** is `deepen.py`'s ``bern/gap`` (d(1 - d) over the two actions' regrets plus
//! `GAP_FLOOR`), times the completion's weight at the root; below it a node's cells inherit
//! the parent cell's priority times the branch's weight (times `discount`), scaled by their
//! own signal over the node's best. A node `levels` plies down opens no cell.
//!
//! **The swap oracle** (`deepen._BeliefOracle` with ``swap``): before every step the outside
//! actions are asked against the other side's support; the one that gains most joins the root
//! (its line filled in every completion) and a weightless action of the same side, with no
//! refined cell, leaves.

use crate::legal::{Act, SideAct};
use crate::lp::{self, f64s, f64s_text};
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::{resolve_turn, Budget, SlotAction};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::time::Instant;

/// `deepen.GAP_FLOOR`.
const GAP_FLOOR: f64 = 1e-3;
/// `deepen.WEIGHTLESS`.
const WEIGHTLESS: f64 = 1e-12;
/// `deepen.ORACLE_TOLERANCE`.
const ORACLE_TOLERANCE: f64 = 1e-6;
/// The strategies' cleaning threshold (`equilibrium.solve`'s).
const EPS: f64 = 1e-9;
/// A support's threshold (`deepen._Oracle._rows`).
const SUPPORT: f64 = 1e-9;
/// Children's Q menus per server request.
const Q_CHUNK: usize = 16;

/// An equilibrium as the tree reads it: the value, the row strategy, each completion's reply,
/// and the regrets the priority divides by.
#[derive(Clone)]
struct Eq {
    value: f64,
    x: Vec<f64>,
    ys: Vec<Vec<f64>>,
    row_loss: Vec<f64>,
    col_loss: Vec<Vec<f64>>,
}

/// `equilibrium.solve` of an m x n matrix (row-major), or None where the LP fails.
fn solve_zero(p: &[f64], m: usize, n: usize) -> Option<Eq> {
    let s = lp::solve(p, m, n, EPS).ok()?;
    let value = s.value();
    let y = s.ys[0].clone();
    let row_ev: Vec<f64> = (0..m).map(|i| (0..n).map(|j| p[i * n + j] * y[j]).sum()).collect();
    let col_ev: Vec<f64> = (0..n).map(|j| (0..m).map(|i| s.x[i] * p[i * n + j]).sum()).collect();
    Some(Eq {
        value,
        row_loss: row_ev.iter().map(|r| (value - r).max(0.0)).collect(),
        col_loss: vec![col_ev.iter().map(|c| (c - value).max(0.0)).collect()],
        x: s.x,
        ys: vec![y],
    })
}

/// `equilibrium.solve_bayesian` over the completions' m x n matrices, or None.
fn solve_bayes(prices: &[Vec<f64>], m: usize, n: usize, w: &[f64]) -> Option<Eq> {
    let mats: Vec<(Vec<f64>, usize, usize)> = prices.iter().map(|p| (p.clone(), m, n)).collect();
    let s = lp::solve_bayesian(&mats, w, EPS).ok()?;
    let value = s.value();
    let mut row_ev = vec![0.0; m];
    for (k, p) in prices.iter().enumerate() {
        for (i, r) in row_ev.iter_mut().enumerate() {
            *r += w[k] * (0..n).map(|j| p[i * n + j] * s.ys[k][j]).sum::<f64>();
        }
    }
    let col_loss = prices
        .iter()
        .map(|p| {
            let ev: Vec<f64> = (0..n).map(|j| (0..m).map(|i| s.x[i] * p[i * n + j]).sum()).collect();
            let low = ev.iter().cloned().fold(f64::INFINITY, f64::min);
            ev.iter().map(|c| (c - low).max(0.0)).collect()
        })
        .collect();
    Some(Eq { value, row_loss: row_ev.iter().map(|r| (value - r).max(0.0)).collect(), col_loss, x: s.x, ys: s.ys })
}

/// A branch of a refined cell: a child node (its index), or a finished game's value.
#[derive(Clone, Copy)]
enum Kid {
    Node(usize),
    Done(f64),
}

type Branches = Vec<(f64, Kid)>;

/// Where a child node hangs: a root cell, or a cell of another node.
#[derive(Clone, Copy)]
enum Up {
    Root((usize, usize, usize)),
    Node(usize, (usize, usize)),
}

/// A node below the root: an open position's matrix in side 0's orientation.
struct Child {
    pos: Position,
    rows: Vec<SideAct>,
    cols: Vec<SideAct>,
    payoff: Vec<f64>,
    eq: Eq,
    level: usize,
    up: Up,
    children: BTreeMap<(usize, usize), Branches>,
    refused: BTreeSet<(usize, usize)>,
}

impl Child {
    fn n(&self) -> usize {
        self.cols.len()
    }

    /// ``bern/gap`` of every cell.
    fn signal(&self) -> Vec<f64> {
        let n = self.n();
        self.payoff
            .iter()
            .enumerate()
            .map(|(c, v)| v * (1.0 - v) / (self.eq.row_loss[c / n] + self.eq.col_loss[0][c % n] + GAP_FLOOR))
            .collect()
    }
}

/// The root: this side's Bayesian game over the other side's completions.
struct Root {
    side: usize,
    own: Vec<SideAct>,
    other: Vec<SideAct>,
    comps: Vec<Position>,
    w: Vec<f64>,
    /// Completion k's matrix, own.len() x other.len(), in this side's orientation.
    prices: Vec<Vec<f64>>,
    eq: Eq,
    children: BTreeMap<(usize, usize, usize), Branches>,
    refused: BTreeSet<(usize, usize, usize)>,
}

impl Root {
    fn m(&self) -> usize {
        self.own.len()
    }
    fn n(&self) -> usize {
        self.other.len()
    }

    /// Row `i` and column `j` as the turn's two actions, side 0's first.
    fn pair(&self, i: usize, j: usize) -> [SideAct; 2] {
        if self.side == 0 {
            [self.own[i].clone(), self.other[j].clone()]
        } else {
            [self.other[j].clone(), self.own[i].clone()]
        }
    }

    /// ``bern/gap`` of every cell (k, i, j), times w_k, flat in that order.
    fn signal(&self) -> Vec<f64> {
        let (m, n) = (self.m(), self.n());
        let mut out = Vec::with_capacity(self.prices.len() * m * n);
        for (k, p) in self.prices.iter().enumerate() {
            for i in 0..m {
                for j in 0..n {
                    let won = if self.side == 0 { p[i * n + j] } else { -p[i * n + j] };
                    let gap = self.eq.row_loss[i] + self.eq.col_loss[k][j] + GAP_FLOOR;
                    out.push(won * (1.0 - won) / gap * self.w[k]);
                }
            }
        }
        out
    }

    /// Re-solve at today's prices; the last answer stands where the LP fails.
    fn solve(&mut self) -> bool {
        match solve_bayes(&self.prices, self.m(), self.n(), &self.w) {
            Some(eq) => {
                self.eq = eq;
                true
            }
            None => false,
        }
    }
}

/// The swap oracle's state (`deepen._BeliefOracle`).
struct Oracle {
    candidates: [Vec<SideAct>; 2],
    known: HashMap<(String, String), Vec<f64>>,
    dropped: [HashSet<String>; 2],
    left: HashSet<String>,
    cap: usize,
    stalled: bool,
    probes: u64,
    probed: u64,
    widened: u64,
    swapped: u64,
}

/// Where the best cell is.
#[derive(Clone, Copy)]
enum Target {
    Root((usize, usize, usize)),
    Node(usize, (usize, usize)),
}

/// A node met while looking for the best cell: its bound, the order it was met, what it
/// inherited (None at the root).
struct Open {
    bound: f64,
    order: u64,
    at: Option<usize>,
    inherited: Option<f64>,
}

impl PartialEq for Open {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl std::cmp::Eq for Open {}
impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Open {
    /// The largest bound first, then the first met.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.bound.total_cmp(&other.bound).then_with(|| other.order.cmp(&self.order))
    }
}

/// One read: the root, the tree, the oracle, what the caller asked for and what was spent.
struct Read {
    base: Value,
    child_base: Value,
    budget: Budget,
    root: Root,
    nodes: Vec<Child>,
    oracle: Option<Oracle>,
    levels: usize,
    discount: Option<f64>,
    branches: usize,
    child_width: usize,
    q_server: Value,
    properties: bool,
    rules: crate::encode::EncodeRules,
    notes: BTreeSet<String>,
    spent: u64,
    refines: u64,
    fills: u64,
    expanded: u64,
    deepest: usize,
    refused_count: u64,
    guarded: u64,
    steps: u64,
    exhausted: bool,
    child_passes: u64,
    child_ranked: u64,
    wait_us: f64,
    grown: u64,
    grown_cells: u64,
    elapsed: std::time::Duration,
}

thread_local! {
    static READS: RefCell<HashMap<u64, Read>> = RefCell::new(HashMap::new());
    static NEXT: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

/// The process's reads started and steps taken: the positive control a test reads.
pub static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn codes(list: &[SideAct]) -> Value {
    Value::Array(list.iter().map(|a| Value::Array(a.iter().map(Act::to_json).collect())).collect())
}

fn read_codes(v: &Value, what: &str) -> Result<Vec<SideAct>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not a list"))?
        .iter()
        .map(|a| {
            a.as_array()
                .ok_or_else(|| format!("{what}: an action is not a list"))?
                .iter()
                .map(|c| Act::from_json(c).ok_or_else(|| format!("{what}: an unreadable slot action {c}")))
                .collect()
        })
        .collect()
}

fn choice(a: &SideAct) -> String {
    crate::legal::choice(a)
}

fn request_actions(list: &[SideAct]) -> Value {
    Value::Array(list.iter().map(|a| Value::Array(a.iter().map(crate::qmenus::act_json).collect())).collect())
}

fn slot_actions(a: &SideAct) -> Vec<SlotAction> {
    a.iter().map(Act::slot_action).collect()
}

/// The fault answer of a `beliefNode` answer that is not matrices, if it is one.
fn failed(answer: &Value) -> Option<Value> {
    for key in ["fault", "serverError", "tooBig", "invalid"] {
        if answer.get(key).is_some() {
            return Some(answer.clone());
        }
    }
    None
}

impl Read {
    /// `beliefNode` on `request` with the two lists (side 0's, side 1's): the flat matrices.
    fn node_fill(
        &mut self,
        reg: &Reg,
        encoder: &crate::encode::Encoder,
        mut request: Value,
        ours: &[SideAct],
        theirs: &[SideAct],
    ) -> Result<Result<Vec<f64>, String>, Value> {
        request["ours"] = request_actions(ours);
        request["theirs"] = request_actions(theirs);
        request["rows"] = json!(ours.len());
        request["cols"] = json!(theirs.len());
        request["rowFeat"] = Value::Array(ours.iter().map(|a| crate::qmenus::action_feature(reg, a)).collect());
        request["colFeat"] = Value::Array(theirs.iter().map(|a| crate::qmenus::action_feature(reg, a)).collect());
        request["solve"] = json!([]);
        let answer = crate::beliefplan::belief_node(reg, encoder, &request);
        if let Some(bad) = failed(&answer) {
            return Err(bad);
        }
        for note in answer["unmodelled"].as_array().into_iter().flatten() {
            if let Some(text) = note.as_str() {
                self.notes.insert(text.to_string());
            }
        }
        self.wait_us += answer["served"]["waitUs"].as_f64().unwrap_or(0.0);
        if let Some(refused) = answer.get("refused") {
            return Ok(Err(format!("the port refused a cell: {refused}")));
        }
        f64s(&answer["matrices"]).map(Ok).map_err(|e| json!({ "fault": e }))
    }

    /// This side's prices of `rows` x `cols` in every completion, in its orientation.
    fn root_fill(
        &mut self,
        reg: &Reg,
        encoder: &crate::encode::Encoder,
        rows: &[SideAct],
        cols: &[SideAct],
    ) -> Result<Vec<Vec<f64>>, Value> {
        let side = self.root.side;
        let (ours, theirs) = if side == 0 { (rows, cols) } else { (cols, rows) };
        let mut request = self.base.clone();
        for (k, pos) in self.root.comps.iter().enumerate() {
            request[format!("p{k}")] = json!({ "held": crate::held::keep(pos) });
        }
        let flat = match self.node_fill(reg, encoder, request, ours, theirs)? {
            Ok(flat) => flat,
            Err(why) => return Err(json!({ "fault": format!("the root's fill: {why}") })),
        };
        let (a, b) = (ours.len(), theirs.len());
        if flat.len() != a * b * self.root.comps.len() {
            return Err(json!({ "fault": "the root's fill answered another size" }));
        }
        Ok((0..self.root.comps.len())
            .map(|k| {
                let mat = &flat[k * a * b..(k + 1) * a * b];
                if side == 0 {
                    mat.to_vec()
                } else {
                    (0..b).flat_map(|j| (0..a).map(move |i| -mat[i * b + j])).collect()
                }
            })
            .collect())
    }

    /// The best unrefined cell with a positive priority, or None (`deepen._best`).
    fn best(&mut self) -> Option<Target> {
        let mut best: Option<(f64, Target)> = None;
        let mut blocked: f64 = 0.0;
        let mut met = 0u64;
        let mut heap = BinaryHeap::new();
        heap.push(Open { bound: f64::INFINITY, order: 0, at: None, inherited: None });
        while let Some(open) = heap.pop() {
            if let Some((score, _)) = best {
                if open.bound <= score {
                    break;
                }
            }
            match open.at {
                None => {
                    let root = &self.root;
                    let (m, n) = (root.m(), root.n());
                    let scores = root.signal();
                    let mut top: Option<(f64, usize)> = None;
                    for (c, s) in scores.iter().enumerate() {
                        let cell = (c / (m * n), (c / n) % m, c % n);
                        if root.children.contains_key(&cell) || root.refused.contains(&cell) {
                            continue;
                        }
                        if top.is_none_or(|(t, _)| *s > t) {
                            top = Some((*s, c));
                        }
                    }
                    if let Some((s, c)) = top {
                        if s > 0.0 {
                            best = Some((s, Target::Root((c / (m * n), (c / n) % m, c % n))));
                        }
                    }
                    for (cell, kids) in &root.children {
                        let s = scores[cell.0 * m * n + cell.1 * n + cell.2];
                        for (weight, kid) in kids {
                            if let Kid::Node(at) = kid {
                                let passed = s * weight * self.discount.unwrap_or(1.0);
                                if passed > 0.0 {
                                    met += 1;
                                    heap.push(Open { bound: passed, order: met, at: Some(*at), inherited: Some(passed) });
                                }
                            }
                        }
                    }
                }
                Some(at) => {
                    let node = &self.nodes[at];
                    let n = node.n();
                    let signal = node.signal();
                    let live_top = signal
                        .iter()
                        .enumerate()
                        .filter(|(c, _)| !node.refused.contains(&(c / n, c % n)))
                        .map(|(_, s)| *s)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let inherited = open.inherited.unwrap_or(0.0);
                    let scores: Vec<f64> = if live_top > 0.0 {
                        signal.iter().map(|s| inherited * s / live_top).collect()
                    } else {
                        vec![0.0; signal.len()]
                    };
                    let mut top: Option<(f64, usize)> = None;
                    for (c, s) in scores.iter().enumerate() {
                        let cell = (c / n, c % n);
                        if node.children.contains_key(&cell) || node.refused.contains(&cell) {
                            continue;
                        }
                        if top.is_none_or(|(t, _)| *s > t) {
                            top = Some((*s, c));
                        }
                    }
                    if node.level < self.levels {
                        if let Some((s, c)) = top {
                            if s > 0.0 && best.is_none_or(|(b, _)| s > b) {
                                best = Some((s, Target::Node(at, (c / n, c % n))));
                            }
                        }
                    } else if let Some((s, _)) = top {
                        blocked = blocked.max(s);
                    }
                    for (cell, kids) in &node.children {
                        let s = scores[cell.0 * n + cell.1];
                        for (weight, kid) in kids {
                            if let Kid::Node(child) = kid {
                                let passed = s * weight * self.discount.unwrap_or(1.0);
                                if passed > 0.0 {
                                    met += 1;
                                    heap.push(Open { bound: passed, order: met, at: Some(*child), inherited: Some(passed) });
                                }
                            }
                        }
                    }
                }
            }
        }
        if blocked > best.map_or(0.0, |(s, _)| s) {
            self.guarded += 1;
        }
        best.map(|(_, t)| t)
    }

    /// Refine one cell (`deepen._expand`): Ok(true) when it took a deeper value.
    fn expand(&mut self, reg: &Reg, encoder: &crate::encode::Encoder, target: Target) -> Result<bool, Value> {
        let (pos, pair, level) = match target {
            Target::Root((k, i, j)) => (self.root.comps[k].clone(), self.root.pair(i, j), 0),
            Target::Node(at, (i, j)) => {
                let node = &self.nodes[at];
                (node.pos.clone(), [node.rows[i].clone(), node.cols[j].clone()], node.level)
            }
        };
        self.spent += 1;
        self.refines += 1;
        let actions = [slot_actions(&pair[0]), slot_actions(&pair[1])];
        let result = match resolve_turn(reg, &pos, &actions, self.budget) {
            Ok(result) => result,
            Err(reason) => {
                self.notes.insert(format!("the port refused a deepened cell's turn: {reason}"));
                return Ok(false);
            }
        };
        self.notes.extend(result.unmodelled.iter().cloned());
        if !result.suspended.is_empty() || result.branches.is_empty() {
            return Ok(false);
        }
        let mut branches: Vec<(f64, Position)> = result.branches.into_iter().map(|b| (b.probability, b.position)).collect();
        branches.sort_by(|a, b| b.0.total_cmp(&a.0));
        if branches.len() > self.branches {
            branches.truncate(self.branches);
            self.notes.insert(format!("depth-2 kept the {} likeliest branches of a refined cell", self.branches));
        }
        let total: f64 = branches.iter().map(|b| b.0).sum();
        if total <= 0.0 {
            return Ok(false);
        }
        let live: Vec<usize> = (0..branches.len()).filter(|&b| !branches[b].1.ended).collect();
        let positions: Vec<Position> = live.iter().map(|&b| branches[b].1.clone()).collect();
        let menus = if positions.is_empty() {
            Vec::new()
        } else {
            let server = crate::served::Server::from_json(&self.q_server).map_err(|e| json!({ "fault": e }))?;
            let got = crate::qmenus::menus_of(
                reg, encoder, &positions, self.child_width, &server, self.rules, self.properties, EPS, Q_CHUNK,
            )?;
            self.wait_us += got.stats.wait_ns as f64 / 1e3;
            self.child_passes += 1;
            self.child_ranked += positions.len() as u64;
            got.menus
        };
        if menus.iter().any(|[a, b]| a.is_empty() || b.is_empty()) {
            return Ok(false);
        }
        let mut made: Vec<(f64, Result<Child, f64>)> = Vec::with_capacity(branches.len());
        let mut menus = menus.into_iter();
        for (probability, child_pos) in branches {
            let weight = probability / total;
            if child_pos.ended {
                let Some(value) = crate::objective::decided(&child_pos) else {
                    return Err(json!({ "fault": "a finished branch has no result" }));
                };
                made.push((weight, Err(value)));
                continue;
            }
            let [rows, cols] = menus.next().expect("a menu per live branch");
            let mut request = self.child_base.clone();
            let id = json!({ "held": crate::held::keep(&child_pos) });
            request["position"] = id.clone();
            request["p0"] = id;
            request["jobs"][0]["pk"] = json!(child_pos.sides[1]
                .pokemon
                .iter()
                .map(|m| json!([crate::reg::to_id(m.species.as_str()), crate::reg::to_id(m.base_species.as_str()), m.slot]))
                .collect::<Vec<_>>());
            let payoff = match self.node_fill(reg, encoder, request, &rows, &cols)? {
                Ok(flat) => flat,
                Err(why) => {
                    self.notes.insert(why);
                    return Ok(false);
                }
            };
            self.fills += 1;
            self.spent += (rows.len() * cols.len()) as u64;
            let Some(eq) = solve_zero(&payoff, rows.len(), cols.len()) else {
                return Ok(false);
            };
            made.push((
                weight,
                Ok(Child {
                    pos: child_pos,
                    rows,
                    cols,
                    payoff,
                    eq,
                    level: level + 1,
                    up: match target {
                        Target::Root(cell) => Up::Root(cell),
                        Target::Node(at, cell) => Up::Node(at, cell),
                    },
                    children: BTreeMap::new(),
                    refused: BTreeSet::new(),
                }),
            ));
        }
        let mut kept: Branches = Vec::with_capacity(made.len());
        for (weight, got) in made {
            match got {
                Err(value) => kept.push((weight, Kid::Done(value))),
                Ok(child) => {
                    self.nodes.push(child);
                    kept.push((weight, Kid::Node(self.nodes.len() - 1)));
                }
            }
        }
        match target {
            Target::Root(cell) => {
                self.root.children.insert(cell, kept);
            }
            Target::Node(at, cell) => {
                self.nodes[at].children.insert(cell, kept);
            }
        }
        self.deepest = self.deepest.max(level + 1);
        Ok(true)
    }

    fn kid_value(&self, kid: &Kid) -> f64 {
        match kid {
            Kid::Done(v) => *v,
            Kid::Node(at) => self.nodes[*at].eq.value,
        }
    }

    fn cell_value(&self, kids: &Branches) -> f64 {
        kids.iter().map(|(w, k)| w * self.kid_value(k)).sum()
    }

    /// Write the refined cell's value and re-solve every node up to the root
    /// (`deepen._propagate`). A node whose LP fails keeps its last answer, and stops it.
    fn propagate(&mut self, target: Target) {
        let mut here = target;
        loop {
            match here {
                Target::Root((k, i, j)) => {
                    let value = self.cell_value(&self.root.children[&(k, i, j)]);
                    let n = self.root.n();
                    self.root.prices[k][i * n + j] = if self.root.side == 0 { value } else { -value };
                    self.root.solve();
                    return;
                }
                Target::Node(at, cell) => {
                    let value = self.cell_value(&self.nodes[at].children[&cell]);
                    let node = &mut self.nodes[at];
                    let n = node.n();
                    node.payoff[cell.0 * n + cell.1] = value;
                    match solve_zero(&node.payoff, node.rows.len(), n) {
                        Some(eq) => node.eq = eq,
                        None => return,
                    }
                    here = match node.up {
                        Up::Root(cell) => Target::Root(cell),
                        Up::Node(parent, cell) => Target::Node(parent, cell),
                    };
                }
            }
        }
    }

    fn support_rows(&self) -> Vec<usize> {
        (0..self.root.m()).filter(|&i| self.root.eq.x[i] > SUPPORT).collect()
    }

    fn support_cols(&self) -> Vec<Vec<usize>> {
        self.root.eq.ys.iter().map(|y| (0..y.len()).filter(|&j| y[j] > SUPPORT).collect()).collect()
    }

    fn outside(&self, role: usize) -> Vec<SideAct> {
        let Some(oracle) = &self.oracle else { return Vec::new() };
        let menu = if role == 0 { &self.root.own } else { &self.root.other };
        let on: HashSet<String> = menu.iter().map(choice).collect();
        oracle.candidates[role]
            .iter()
            .filter(|a| {
                let c = choice(a);
                !on.contains(&c) && !oracle.dropped[role].contains(&c)
            })
            .cloned()
            .collect()
    }

    fn known(&self, a: &SideAct, b: &SideAct) -> bool {
        self.oracle.as_ref().is_some_and(|o| o.known.contains_key(&(choice(a), choice(b))))
    }

    /// Fill `rows` x `cols` in every completion and remember each cell (`_BeliefOracle._ask`).
    fn ask(&mut self, reg: &Reg, encoder: &crate::encode::Encoder, rows: &[SideAct], cols: &[SideAct]) -> Result<(), Value> {
        if rows.is_empty() || cols.is_empty() {
            return Ok(());
        }
        let prices = self.root_fill(reg, encoder, rows, cols)?;
        let k = prices.len();
        let oracle = self.oracle.as_mut().expect("an oracle");
        for (r, a) in rows.iter().enumerate() {
            for (c, b) in cols.iter().enumerate() {
                oracle.known.insert((choice(a), choice(b)), prices.iter().map(|p| p[r * cols.len() + c]).collect());
            }
        }
        let cells = (rows.len() * cols.len() * k) as u64;
        oracle.probed += cells;
        self.spent += cells;
        self.fills += 1;
        Ok(())
    }

    /// `_BeliefOracle.probe`: the outside actions against the other side's support.
    fn probe(&mut self, reg: &Reg, encoder: &crate::encode::Encoder) -> Result<(), Value> {
        let union: BTreeSet<usize> = self.support_cols().into_iter().flatten().collect();
        let support = self.support_rows();
        let (out_rows, out_cols) = (self.outside(0), self.outside(1));
        let rows: Vec<SideAct> = out_rows
            .into_iter()
            .filter(|a| union.iter().any(|&j| !self.known(a, &self.root.other[j])))
            .collect();
        let cols_needed: Vec<SideAct> = union
            .iter()
            .filter(|&&j| rows.iter().any(|a| !self.known(a, &self.root.other[j])))
            .map(|&j| self.root.other[j].clone())
            .collect();
        let outside_cols: Vec<SideAct> = out_cols
            .into_iter()
            .filter(|b| support.iter().any(|&i| !self.known(&self.root.own[i], b)))
            .collect();
        let rows_needed: Vec<SideAct> = support
            .iter()
            .filter(|&&i| outside_cols.iter().any(|b| !self.known(&self.root.own[i], b)))
            .map(|&i| self.root.own[i].clone())
            .collect();
        if (rows.is_empty() || cols_needed.is_empty()) && (outside_cols.is_empty() || rows_needed.is_empty()) {
            return Ok(());
        }
        self.ask(reg, encoder, &rows, &cols_needed)?;
        self.ask(reg, encoder, &rows_needed, &outside_cols)?;
        self.oracle.as_mut().expect("an oracle").probes += 1;
        Ok(())
    }

    /// `_BeliefOracle.widen`: the outside action that gains most joins, if one gains.
    fn widen(&mut self, reg: &Reg, encoder: &crate::encode::Encoder) -> Result<bool, Value> {
        if self.oracle.as_ref().is_none_or(|o| o.stalled) {
            return Ok(false);
        }
        let root = &self.root;
        let (m, n) = (root.m(), root.n());
        let value = root.eq.value;
        let support = self.support_rows();
        let cols = self.support_cols();
        let earned: Vec<f64> = (0..root.prices.len())
            .map(|k| {
                let p = &root.prices[k];
                (0..m).map(|i| root.eq.x[i] * (0..n).map(|j| p[i * n + j] * root.eq.ys[k][j]).sum::<f64>()).sum()
            })
            .collect();
        let oracle = self.oracle.as_ref().expect("an oracle");
        let mut best: Option<(f64, usize, SideAct)> = None;
        for role in 0..2 {
            for action in self.outside(role) {
                let key = choice(&action);
                let gain = if role == 0 {
                    let mut got = 0.0;
                    for k in 0..root.prices.len() {
                        for &j in &cols[k] {
                            got += root.w[k] * root.eq.ys[k][j] * oracle.known[&(key.clone(), choice(&root.other[j]))][k];
                        }
                    }
                    got - value
                } else {
                    let mut gain = 0.0;
                    for k in 0..root.prices.len() {
                        let cost: f64 =
                            support.iter().map(|&i| root.eq.x[i] * oracle.known[&(choice(&root.own[i]), key.clone())][k]).sum();
                        gain += root.w[k] * (earned[k] - cost).max(0.0);
                    }
                    gain
                };
                if gain > ORACLE_TOLERANCE && best.as_ref().is_none_or(|(b, _, _)| gain > *b) {
                    best = Some((gain, role, action));
                }
            }
        }
        let Some((_, role, action)) = best else { return Ok(false) };
        let key = choice(&action);
        let others: Vec<SideAct> = if role == 0 { self.root.other.clone() } else { self.root.own.clone() };
        let pair = |o: &SideAct| if role == 0 { (key.clone(), choice(o)) } else { (choice(o), key.clone()) };
        let missing: Vec<SideAct> = others.iter().filter(|o| !self.known_pair(&pair(o))).cloned().collect();
        if role == 0 {
            self.ask(reg, encoder, std::slice::from_ref(&action), &missing)?;
        } else {
            self.ask(reg, encoder, &missing, std::slice::from_ref(&action))?;
        }
        let oracle = self.oracle.as_ref().expect("an oracle");
        let lines: Vec<&Vec<f64>> = others.iter().map(|o| &oracle.known[&pair(o)]).collect();
        let before = (self.root.prices.clone(), self.root.eq.clone());
        for (k, p) in self.root.prices.iter_mut().enumerate() {
            if role == 0 {
                p.extend(lines.iter().map(|l| l[k]));
            } else {
                let mut grown = Vec::with_capacity(m * (n + 1));
                for i in 0..m {
                    grown.extend_from_slice(&p[i * n..(i + 1) * n]);
                    grown.push(lines[i][k]);
                }
                *p = grown;
            }
        }
        if role == 0 {
            self.root.own.push(action);
        } else {
            self.root.other.push(action);
        }
        if !self.root.solve() {
            (self.root.prices, self.root.eq) = before;
            if role == 0 {
                self.root.own.pop();
            } else {
                self.root.other.pop();
            }
            self.oracle.as_mut().expect("an oracle").dropped[role].insert(key);
            return Ok(false);
        }
        self.oracle.as_mut().expect("an oracle").widened += 1;
        self.swap_out(role, &key);
        let oracle = self.oracle.as_mut().expect("an oracle");
        if oracle.widened as usize >= oracle.cap {
            oracle.stalled = true;
        }
        Ok(true)
    }

    /// Widen the root mid-read (`deepen._grow_belief`, IKA-354): `add` is side 0's and side 1's
    /// actions; those not on the root's menus yet are appended (this side's as rows, the other
    /// side's as columns) and filled at depth 1 in every completion, and the root is solved
    /// again. Every refined cell keeps its index and its tree. A root the LP cannot solve is put
    /// back whole. Returns the actions added.
    fn grow(&mut self, reg: &Reg, encoder: &crate::encode::Encoder, add: [Vec<SideAct>; 2]) -> Result<usize, Value> {
        let [a0, a1] = add;
        let (mine, theirs) = if self.root.side == 0 { (a0, a1) } else { (a1, a0) };
        let fresh = |list: Vec<SideAct>, menu: &[SideAct]| -> Vec<SideAct> {
            let mut seen: HashSet<String> = menu.iter().map(choice).collect();
            list.into_iter().filter(|a| seen.insert(choice(a))).collect()
        };
        let rows = fresh(mine, &self.root.own);
        let cols = fresh(theirs, &self.root.other);
        if rows.is_empty() && cols.is_empty() {
            return Ok(0);
        }
        let before = (self.root.prices.clone(), self.root.eq.clone(), self.root.own.clone(), self.root.other.clone());
        let spent = self.spent;
        if !rows.is_empty() {
            let other = self.root.other.clone();
            let lines = self.root_fill(reg, encoder, &rows, &other)?;
            for (p, line) in self.root.prices.iter_mut().zip(lines) {
                p.extend(line);
            }
            self.root.own.extend(rows.iter().cloned());
            self.spent += (rows.len() * other.len() * self.root.comps.len()) as u64;
            self.fills += 1;
        }
        if !cols.is_empty() {
            let own = self.root.own.clone();
            let lines = self.root_fill(reg, encoder, &own, &cols)?;
            let (m, n, c) = (own.len(), self.root.other.len(), cols.len());
            for (p, line) in self.root.prices.iter_mut().zip(lines) {
                let mut grown = Vec::with_capacity(m * (n + c));
                for i in 0..m {
                    grown.extend_from_slice(&p[i * n..(i + 1) * n]);
                    grown.extend_from_slice(&line[i * c..(i + 1) * c]);
                }
                *p = grown;
            }
            self.root.other.extend(cols.iter().cloned());
            self.spent += (m * c * self.root.comps.len()) as u64;
            self.fills += 1;
        }
        if !self.root.solve() {
            (self.root.prices, self.root.eq, self.root.own, self.root.other) = before;
            self.grown_cells += self.spent - spent;
            return Ok(0);
        }
        if let Some(oracle) = self.oracle.as_mut() {
            for (role, list) in [(0, &rows), (1, &cols)] {
                let known: HashSet<String> = oracle.candidates[role].iter().map(choice).collect();
                oracle.candidates[role].extend(list.iter().filter(|a| !known.contains(&choice(a))).cloned());
            }
        }
        self.grown += (rows.len() + cols.len()) as u64;
        self.grown_cells += self.spent - spent;
        Ok(rows.len() + cols.len())
    }

    fn known_pair(&self, pair: &(String, String)) -> bool {
        self.oracle.as_ref().is_some_and(|o| o.known.contains_key(pair))
    }

    /// `_BeliefOracle._swap_out`: a weightless action of the role, with no refined or refused
    /// cell, leaves -- the one that loses most.
    fn swap_out(&mut self, role: usize, added: &str) {
        let root = &self.root;
        let (m, n) = (root.m(), root.n());
        let (weightless, loss): (Vec<bool>, Vec<f64>) = if role == 0 {
            (root.eq.x.iter().map(|v| *v <= WEIGHTLESS).collect(), root.eq.row_loss.clone())
        } else {
            (
                (0..n).map(|j| root.eq.ys.iter().all(|y| y[j] <= WEIGHTLESS)).collect(),
                (0..n).map(|j| (0..root.prices.len()).map(|k| root.w[k] * root.eq.col_loss[k][j]).sum()).collect(),
            )
        };
        let busy: HashSet<usize> = root
            .children
            .keys()
            .chain(root.refused.iter())
            .map(|c| if role == 0 { c.1 } else { c.2 })
            .collect();
        let menu = if role == 0 { &root.own } else { &root.other };
        let left = &self.oracle.as_ref().expect("an oracle").left;
        let free: Vec<usize> = (0..menu.len())
            .filter(|&i| weightless[i] && !busy.contains(&i) && choice(&menu[i]) != added && !left.contains(&choice(&menu[i])))
            .collect();
        let Some(&out) = free.iter().max_by(|a, b| loss[**a].total_cmp(&loss[**b]).then(a.cmp(b))) else {
            return;
        };
        let before = (self.root.prices.clone(), self.root.eq.clone(), self.root.own.clone(), self.root.other.clone());
        for p in self.root.prices.iter_mut() {
            *p = if role == 0 {
                (0..m).filter(|&i| i != out).flat_map(|i| p[i * n..(i + 1) * n].to_vec()).collect()
            } else {
                (0..m).flat_map(|i| (0..n).filter(|&j| j != out).map(move |j| (i, j))).map(|(i, j)| p[i * n + j]).collect()
            };
        }
        let gone = if role == 0 { self.root.own.remove(out) } else { self.root.other.remove(out) };
        if !self.root.solve() {
            (self.root.prices, self.root.eq, self.root.own, self.root.other) = before;
            return;
        }
        let moved = |c: (usize, usize, usize)| -> (usize, usize, usize) {
            match role {
                0 if c.1 > out => (c.0, c.1 - 1, c.2),
                1 if c.2 > out => (c.0, c.1, c.2 - 1),
                _ => c,
            }
        };
        let children = std::mem::take(&mut self.root.children);
        self.root.children = children.into_iter().map(|(c, kids)| (moved(c), kids)).collect();
        self.root.refused = std::mem::take(&mut self.root.refused).into_iter().map(moved).collect();
        for (cell, kids) in &self.root.children {
            for (_, kid) in kids {
                if let Kid::Node(at) = kid {
                    self.nodes[*at].up = Up::Root(*cell);
                }
            }
        }
        let oracle = self.oracle.as_mut().expect("an oracle");
        oracle.left.insert(choice(&gone));
        oracle.swapped += 1;
    }

    /// Why each deepened line stopped: (finished game, nothing worth a step, at the guard,
    /// the budget stopped first) -- `deepen._Watch`.
    fn lines(&self) -> [u64; 4] {
        let mut out = [0u64; 4];
        let mut stack: Vec<&Branches> = self.root.children.values().collect();
        while let Some(kids) = stack.pop() {
            for (_, kid) in kids {
                match kid {
                    Kid::Done(_) => out[0] += 1,
                    Kid::Node(at) => {
                        let node = &self.nodes[*at];
                        if !node.children.is_empty() {
                            stack.extend(node.children.values());
                            continue;
                        }
                        let n = node.n();
                        let worth = node
                            .signal()
                            .iter()
                            .enumerate()
                            .any(|(c, s)| *s > 0.0 && !node.refused.contains(&(c / n, c % n)));
                        out[if !worth {
                            1
                        } else if node.level >= self.levels {
                            2
                        } else {
                            3
                        }] += 1;
                    }
                }
            }
        }
        out
    }

    /// The read as it stands.
    fn snapshot(&self) -> Value {
        let root = &self.root;
        let flat: Vec<f64> = root.prices.iter().flatten().copied().collect();
        let oracle = self.oracle.as_ref();
        json!({
            "own": codes(&root.own),
            "other": codes(&root.other),
            "x": root.eq.x,
            "ys": root.eq.ys,
            "value": root.eq.value,
            "prices": f64s_text(&flat),
            "cells": self.spent,
            "refines": self.refines,
            "fills": self.fills,
            "expanded": self.expanded,
            "depth": 1 + self.deepest,
            "refused": self.refused_count,
            "oracle": oracle.is_some(),
            "probed": oracle.map_or(0, |o| o.probed),
            "probes": oracle.map_or(0, |o| o.probes),
            "widened": oracle.map_or(0, |o| o.widened),
            "swapped": oracle.map_or(0, |o| o.swapped),
            "classes": root.prices.len(),
            "steps": self.steps,
            "exhausted": self.exhausted,
            "levels": self.levels,
            "lines": self.lines(),
            "guarded": self.guarded,
            "nodes": self.nodes.len(),
            "childPasses": self.child_passes,
            "childRanked": self.child_ranked,
            "waitUs": self.wait_us,
            "grown": self.grown,
            "grownCells": self.grown_cells,
            "elapsedMs": self.elapsed.as_secs_f64() * 1e3,
            "unmodelled": self.notes.iter().collect::<Vec<_>>(),
        })
    }

    /// Steps until the budget, the slice or the tree runs out: why it stopped.
    fn run(
        &mut self,
        reg: &Reg,
        encoder: &crate::encode::Encoder,
        cells: Option<u64>,
        ms: Option<f64>,
        steps: Option<u64>,
    ) -> Result<&'static str, Value> {
        let started = Instant::now();
        let mut taken = 0u64;
        let why = loop {
            if self.exhausted {
                break "exhausted";
            }
            if cells.is_some_and(|c| self.spent >= c) {
                break "budget";
            }
            if ms.is_some_and(|t| started.elapsed().as_secs_f64() * 1e3 >= t) {
                break "slice";
            }
            if steps.is_some_and(|s| taken >= s) {
                break "steps";
            }
            taken += 1;
            self.steps += 1;
            STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if self.oracle.is_some() {
                self.probe(reg, encoder)?;
                if self.widen(reg, encoder)? {
                    continue;
                }
            }
            let Some(target) = self.best() else {
                self.exhausted = true;
                break "exhausted";
            };
            if self.expand(reg, encoder, target)? {
                self.expanded += 1;
                self.propagate(target);
            } else {
                self.refused_count += 1;
                match target {
                    Target::Root(cell) => {
                        self.root.refused.insert(cell);
                    }
                    Target::Node(at, cell) => {
                        self.nodes[at].refused.insert(cell);
                    }
                }
            }
        };
        self.elapsed += started.elapsed();
        Ok(why)
    }
}

fn start(reg: &Reg, encoder: &crate::encode::Encoder, value: &Value) -> Result<Value, Value> {
    let fault = |e: String| json!({ "fault": e });
    let node = value;
    if node["netScoresEnds"].as_bool().unwrap_or(false) {
        return Err(fault("the port's deepening scores a finished game by its result; not with netScoresEnds".into()));
    }
    let side = value["side"].as_u64().filter(|s| *s < 2).ok_or_else(|| fault("deepenStart without a side".into()))? as usize;
    let other_side = 1 - side;
    let menus = value["menus"].as_array().filter(|m| m.len() == 2).ok_or_else(|| fault("deepenStart without two menus".into()))?;
    let lists = [read_codes(&menus[0], "menus[0]").map_err(fault)?, read_codes(&menus[1], "menus[1]").map_err(fault)?];
    if lists[0].is_empty() || lists[1].is_empty() {
        return Err(fault("deepenStart: a menu is empty".into()));
    }
    // The other side's completions: its jobs, renumbered from p0, and nothing of this side's.
    let jobs = node["jobs"].as_array().ok_or_else(|| fault("deepenStart: the node has no jobs".into()))?;
    let mut base = node.clone();
    let mut kept_jobs = Vec::new();
    let mut comps = Vec::new();
    for (idx, job) in jobs.iter().enumerate() {
        if job["side"].as_u64() != Some(other_side as u64) {
            continue;
        }
        comps.push(crate::held::position(&node[format!("p{idx}")]));
        kept_jobs.push(job.clone());
    }
    if comps.is_empty() {
        return Err(fault("deepenStart: no completion of the other side".into()));
    }
    if let Some(map) = base.as_object_mut() {
        map.retain(|k, _| {
            let completion = k.len() > 1 && k.starts_with('p') && k[1..].chars().all(|c| c.is_ascii_digit());
            !completion && !matches!(k.as_str(), "menus" | "outside" | "weights" | "qServer")
        });
    }
    base["jobs"] = Value::Array(kept_jobs);
    base["hidden"][side] = json!([]);
    let root_pos = crate::held::position(&node["position"]);
    base["position"] = json!({ "held": crate::held::keep(&root_pos) });
    let weights: Vec<f64> = value["weights"]
        .as_array()
        .ok_or_else(|| fault("deepenStart without weights".into()))?
        .iter()
        .map(|w| w.as_f64().ok_or_else(|| fault("a weight is not a number".into())))
        .collect::<Result<_, _>>()?;
    if weights.len() != comps.len() {
        return Err(fault(format!("{} weights for {} completions", weights.len(), comps.len())));
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return Err(fault("the completions' weights sum to 0".into()));
    }
    let mut child_base = json!({
        "kind": "beliefNode",
        "budget": node["budget"],
        "hidden": [[], []],
        "carded": [false, false],
        "jobs": [{ "side": 1, "slots": [], "exact": true, "root": true, "pk": [] }],
        "server": node["server"],
        "netScoresEnds": false,
        "patch": node["patch"],
    });
    if let Some(e) = node.get("encoding").filter(|e| !e.is_null()) {
        child_base["encoding"] = e.clone();
    }
    let [l0, l1] = lists;
    let (own, other) = if side == 0 { (l0, l1) } else { (l1, l0) };
    let oracle = match value.get("outside").filter(|o| !o.is_null()) {
        None => None,
        Some(outside) => {
            let given = outside.as_array().filter(|o| o.len() == 2).ok_or_else(|| fault("outside is two lists".into()))?;
            let out = [read_codes(&given[0], "outside[0]").map_err(fault)?, read_codes(&given[1], "outside[1]").map_err(fault)?];
            let roles = if side == 0 { [&out[0], &out[1]] } else { [&out[1], &out[0]] };
            let mut candidates: [Vec<SideAct>; 2] = [Vec::new(), Vec::new()];
            for (role, menu) in [(0, &own), (1, &other)] {
                let mut seen = HashSet::new();
                for a in roles[role].iter().chain(menu.iter()) {
                    if seen.insert(choice(a)) {
                        candidates[role].push(a.clone());
                    }
                }
            }
            let cap = 4 * (candidates[0].len() + candidates[1].len()) + 4;
            Some(Oracle {
                candidates,
                known: HashMap::new(),
                dropped: [HashSet::new(), HashSet::new()],
                left: HashSet::new(),
                cap,
                stalled: false,
                probes: 0,
                probed: 0,
                widened: 0,
                swapped: 0,
            })
        }
    };
    let started = Instant::now();
    let mut read = Read {
        base,
        child_base,
        budget: Budget::from_json(&node["budget"]),
        root: Root {
            side,
            own,
            other,
            comps,
            w: weights.iter().map(|w| w / total).collect(),
            prices: Vec::new(),
            eq: Eq { value: 0.0, x: Vec::new(), ys: Vec::new(), row_loss: Vec::new(), col_loss: Vec::new() },
            children: BTreeMap::new(),
            refused: BTreeSet::new(),
        },
        nodes: Vec::new(),
        oracle,
        levels: value["levels"].as_u64().unwrap_or(8) as usize,
        discount: value["discount"].as_f64(),
        branches: value["branches"].as_u64().unwrap_or(3).max(1) as usize,
        child_width: value["childWidth"].as_u64().unwrap_or(8).max(1) as usize,
        q_server: value["qServer"].clone(),
        properties: value["properties"].as_bool().unwrap_or(true),
        rules: crate::encode::EncodeRules { mega_from_slots: value["megaFromSlots"].as_bool().unwrap_or(false) },
        notes: BTreeSet::new(),
        spent: 0,
        refines: 0,
        fills: 0,
        expanded: 0,
        deepest: 0,
        refused_count: 0,
        guarded: 0,
        steps: 0,
        exhausted: false,
        child_passes: 0,
        child_ranked: 0,
        wait_us: 0.0,
        grown: 0,
        grown_cells: 0,
        elapsed: std::time::Duration::ZERO,
    };
    let (own, other) = (read.root.own.clone(), read.root.other.clone());
    read.root.prices = read.root_fill(reg, encoder, &own, &other)?;
    read.fills = 0;
    if !read.root.solve() {
        return Err(fault("the depth-1 game was not solved".into()));
    }
    read.elapsed = started.elapsed();
    let id = NEXT.with(|next| {
        let id = next.get();
        next.set(id + 1);
        id
    });
    let mut answer = read.snapshot();
    answer["id"] = json!(id);
    READS.with(|reads| reads.borrow_mut().insert(id, read));
    STARTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(answer)
}

/// `deepenStart`.
pub fn deepen_start(reg: &Reg, encoder: &crate::encode::Encoder, value: &Value) -> Value {
    start(reg, encoder, value).unwrap_or_else(|failed| failed)
}

/// `deepenRun`.
pub fn deepen_run(reg: &Reg, encoder: &crate::encode::Encoder, value: &Value) -> Value {
    let Some(id) = value["id"].as_u64() else {
        return json!({ "fault": "deepenRun without an id" });
    };
    let Some(mut read) = READS.with(|reads| reads.borrow_mut().remove(&id)) else {
        return json!({ "fault": format!("no deepening is held under {id} (the port restarted?)") });
    };
    // IKA-354's widening of the root, asked before the slice (the analysis mode's `grow`).
    let grown = match value.get("grow").and_then(Value::as_array).filter(|g| g.len() == 2) {
        None => Ok(0),
        Some(lists) => match (read_codes(&lists[0], "grow[0]"), read_codes(&lists[1], "grow[1]")) {
            (Ok(a), Ok(b)) => read.grow(reg, encoder, [a, b]),
            (Err(e), _) | (_, Err(e)) => Err(json!({ "fault": e })),
        },
    };
    let got = match grown {
        Ok(_) => read.run(reg, encoder, value["cells"].as_u64(), value["ms"].as_f64(), value["steps"].as_u64()),
        Err(failed) => Err(failed),
    };
    let answer = match got {
        Ok(why) => {
            let mut answer = read.snapshot();
            answer["id"] = json!(id);
            answer["stopped"] = json!(why);
            answer
        }
        Err(failed) => failed,
    };
    READS.with(|reads| reads.borrow_mut().insert(id, read));
    answer
}

/// `deepenEnd`.
pub fn deepen_end(value: &Value) -> Value {
    let id = value["id"].as_u64().unwrap_or(0);
    let had = READS.with(|reads| reads.borrow_mut().remove(&id)).is_some();
    json!({ "ended": had })
}

/// `deepenCounts`: the positive control.
pub fn counts() -> Value {
    json!({
        "started": STARTED.load(std::sync::atomic::Ordering::Relaxed),
        "steps": STEPS.load(std::sync::atomic::Ordering::Relaxed),
        "held": READS.with(|reads| reads.borrow().len()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bayesian_game_of_one_completion_is_the_zero_sum_game() {
        let p = vec![0.2, 0.7, 0.6, 0.3];
        let zero = solve_zero(&p, 2, 2).unwrap();
        let bayes = solve_bayes(&[p.clone()], 2, 2, &[1.0]).unwrap();
        assert!((zero.value - bayes.value).abs() < 1e-9);
        assert!((zero.value - 0.45).abs() < 1e-9, "{}", zero.value);
        for (a, b) in zero.x.iter().zip(&bayes.x) {
            assert!((a - b).abs() < 1e-9);
        }
        // Mixed: no action loses at equilibrium.
        assert!(zero.row_loss.iter().all(|l| l.abs() < 1e-9));
    }

    #[test]
    fn the_heap_opens_the_largest_bound_first_and_the_first_met_on_a_tie() {
        let mut heap = BinaryHeap::new();
        heap.push(Open { bound: 0.5, order: 2, at: Some(2), inherited: None });
        heap.push(Open { bound: 0.9, order: 3, at: Some(3), inherited: None });
        heap.push(Open { bound: 0.5, order: 1, at: Some(1), inherited: None });
        let order: Vec<usize> = std::iter::from_fn(|| heap.pop()).map(|o| o.at.unwrap()).collect();
        assert_eq!(order, vec![3, 1, 2]);
    }
}
