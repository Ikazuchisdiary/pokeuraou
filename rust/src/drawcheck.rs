//! IKA-469 `sampleCheck`: does a played turn drawn along one path (`resolve::sample_turn`)
//! draw from the distribution `resolve_turn`'s branches give?
//!
//! Two answers for one position and pair of actions:
//!
//! * **Every path.** `sample_turn` is run once per sequence of answers to its chance points
//!   (an odometer over them, up to `maxPaths` paths). Each path's probability is the product
//!   of the normalised weights it took. The paths' outcomes, folded by `merge_branches`'s and
//!   `merge_suspended`'s tests, are compared with `resolve_turn`'s branches and pauses:
//!   `maxDiff` is the largest difference of probability over the outcomes of either side.
//! * **Draws.** `samples` draws from one PCG64 seeded with `seed`, each matched to one of
//!   `resolve_turn`'s outcomes (`counts`), or to none (`unmatched`).
//!
//! `budget` is the enumeration's and `sampleBudget` (when given) the draws'. A test asks the
//! draws under the exact budget a game plays and the enumeration under the same budget with a
//! cap high enough that nothing is narrowed or dropped: the exact budget's own enumeration
//! narrows most turns with two damaging moves (records/IKA-469.md).
//!
//! A diagnostic for the tests (`tests/test_sample_turn.py`); nothing on the game's road asks it.

use crate::advance::Pcg64;
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::{
    branch_fingerprint, parse_actions_list, resolve_turn, same_branch, same_pause, sample_turn,
    Budget, Drawn, Sampler, Suspended,
};
use serde_json::{json, Value};
use std::collections::HashMap;

/// One outcome: a finished position or a pause.
enum Leaf<'a> {
    Finished(Position),
    Paused(Suspended<'a>),
}

impl Leaf<'_> {
    fn fingerprint(&self) -> u64 {
        match self {
            Leaf::Finished(pos) => branch_fingerprint(pos, None),
            Leaf::Paused(pause) => branch_fingerprint(&pause.turn.pos, Some(&pause.remaining)),
        }
    }

    fn same(&self, other: &Leaf) -> bool {
        match (self, other) {
            (Leaf::Finished(a), Leaf::Finished(b)) => same_branch(a, b),
            (Leaf::Paused(a), Leaf::Paused(b)) => same_pause(a, b),
            _ => false,
        }
    }
}

/// Outcomes with their probabilities, equal ones folded into one.
#[derive(Default)]
struct Folded<'a> {
    items: Vec<(f64, Leaf<'a>)>,
    buckets: HashMap<u64, Vec<usize>>,
}

impl<'a> Folded<'a> {
    fn find(&self, leaf: &Leaf) -> Option<usize> {
        self.buckets.get(&leaf.fingerprint())?.iter().copied().find(|i| self.items[*i].1.same(leaf))
    }

    fn add(&mut self, probability: f64, leaf: Leaf<'a>) {
        match self.find(&leaf) {
            Some(i) => self.items[i].0 += probability,
            None => {
                self.buckets.entry(leaf.fingerprint()).or_default().push(self.items.len());
                self.items.push((probability, leaf));
            }
        }
    }
}

/// Answers a drawn turn's chance points along one path after another: the points the
/// current path has met replay its answers, and a new point takes its first option with
/// weight. `path` holds (the answer, the normalised weights) of each point.
#[derive(Default)]
struct Odometer {
    path: Vec<(usize, Vec<f64>)>,
    depth: usize,
}

impl crate::resolve::Chooser for Odometer {
    fn pick(&mut self, weights: &[f64]) -> Result<usize, String> {
        let at = if self.depth < self.path.len() {
            self.path[self.depth].0
        } else {
            let sum: f64 = weights.iter().sum();
            let normal: Vec<f64> = weights.iter().map(|w| w / sum).collect();
            let first = normal.iter().position(|w| *w > 0.0).ok_or("no option has weight")?;
            self.path.push((first, normal));
            first
        };
        self.depth += 1;
        Ok(at)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn with_odometer<T>(sampler: &Sampler, f: impl FnOnce(&mut Odometer) -> T) -> T {
    let mut draws = sampler.borrow_mut();
    let any: &mut dyn std::any::Any = draws.chooser.as_any_mut();
    f(any.downcast_mut::<Odometer>().expect("an odometer"))
}

fn leaf_of(drawn: Drawn) -> Option<Leaf> {
    match drawn {
        Drawn::Nothing => None,
        Drawn::Finished(pos) => Some(Leaf::Finished(pos)),
        Drawn::Paused(pause) => Some(Leaf::Paused(pause)),
    }
}

pub fn sample_check(reg: &Reg, value: &Value) -> Value {
    let position = crate::held::position(&value["position"]);
    if &*position.format != reg.format_id.as_str() {
        return json!({ "refused": "position is for another regulation" });
    }
    let actions = [parse_actions_list(&value["actions"][0]), parse_actions_list(&value["actions"][1])];
    let budget = Budget::from_json(&value["budget"]);
    // The budget the draws are made under (the enumeration's when absent).
    let drawn_budget = match value.get("sampleBudget").filter(|v| !v.is_null()) {
        Some(v) => Budget::from_json(v),
        None => budget,
    };
    let max_paths = value["maxPaths"].as_u64().unwrap_or(0) as usize;
    let samples = value["samples"].as_u64().unwrap_or(0) as usize;
    let seed = value["seed"].as_u64().unwrap_or(0);

    let clock = std::time::Instant::now();
    let result = match resolve_turn(reg, &position, &actions, budget) {
        Err(reason) => return json!({ "refused": reason }),
        Ok(result) => result,
    };
    let resolve_us = clock.elapsed().as_secs_f64() * 1e6;
    let exact = result.exact;
    let (branches, pauses) = (result.branches.len(), result.suspended.len());
    let mut enumerated = Folded::default();
    for branch in result.branches {
        enumerated.add(branch.probability, Leaf::Finished(branch.position));
    }
    for mut pause in result.suspended {
        // A pause carries the budget it was resolved under, which `same_pause` compares.
        pause.turn.budget = drawn_budget;
        let probability = pause.probability;
        enumerated.add(probability, Leaf::Paused(pause));
    }
    let total: f64 = enumerated.items.iter().map(|(p, _)| *p).sum();
    let probabilities: Vec<f64> = enumerated.items.iter().map(|(p, _)| *p / total.max(f64::MIN_POSITIVE)).collect();

    // Every path, as an odometer over the answers to the chance points.
    let mut paths_answer = Value::Null;
    let mut max_diff = Value::Null;
    let mut path_draws = Value::Null;
    if max_paths > 0 {
        let mut walked = Folded::default();
        let odometer = crate::resolve::sampler(Box::new(Odometer::default()));
        let mut paths = 0usize;
        let mut deepest = 0usize;
        let mut over = false;
        loop {
            with_odometer(&odometer, |o| o.depth = 0);
            let sampled = match sample_turn(reg, &position, &actions, drawn_budget, &odometer) {
                Err(reason) => return json!({ "fault": format!("a path was refused: {reason}") }),
                Ok(sampled) => sampled,
            };
            let (depth, length, probability) =
                with_odometer(&odometer, |o| (o.depth, o.path.len(), o.path.iter().map(|(at, w)| w[*at]).product::<f64>()));
            if depth != length {
                return json!({ "fault": "a path asked fewer chance points than the one it replays" });
            }
            paths += 1;
            deepest = deepest.max(length);
            if let Some(leaf) = leaf_of(sampled.drawn) {
                walked.add(probability, leaf);
            }
            // The next path: the last point that has an untried option with weight.
            let more = with_odometer(&odometer, |o| {
                while let Some((at, weights)) = o.path.last_mut() {
                    if let Some(next) = (*at + 1..weights.len()).find(|j| weights[*j] > 0.0) {
                        *at = next;
                        return true;
                    }
                    o.path.pop();
                }
                false
            });
            if !more {
                break;
            }
            if paths >= max_paths {
                over = true;
                break;
            }
        }
        if !over {
            let mut diff = 0.0f64;
            for (p, leaf) in &walked.items {
                let q = enumerated.find(leaf).map_or(0.0, |i| probabilities[i]);
                diff = diff.max((p - q).abs());
            }
            for (i, (_, leaf)) in enumerated.items.iter().enumerate() {
                let p = walked.find(leaf).map_or(0.0, |j| walked.items[j].0);
                diff = diff.max((p - probabilities[i]).abs());
            }
            paths_answer = json!(paths);
            max_diff = json!(diff);
            path_draws = json!(deepest);
        }
    }

    // Draws from one generator.
    let mut counts = vec![0u64; enumerated.items.len()];
    let mut unmatched = 0u64;
    let mut nothing = 0u64;
    let mut inexact = 0u64;
    let mut draws = 0u64;
    let rng = crate::resolve::sampler(Box::new(Pcg64::seeded(&[seed])));
    let clock = std::time::Instant::now();
    for _ in 0..samples {
        let before = rng.borrow().count;
        let sampled = match sample_turn(reg, &position, &actions, drawn_budget, &rng) {
            Err(reason) => return json!({ "fault": format!("a draw was refused: {reason}") }),
            Ok(sampled) => sampled,
        };
        draws += (rng.borrow().count - before) as u64;
        if !sampled.exact {
            inexact += 1;
        }
        match leaf_of(sampled.drawn) {
            None => nothing += 1,
            Some(leaf) => match enumerated.find(&leaf) {
                Some(i) => counts[i] += 1,
                None => unmatched += 1,
            },
        }
    }
    let sample_us = clock.elapsed().as_secs_f64() * 1e6 / samples.max(1) as f64;
    json!({
        "resolveUs": resolve_us,
        "sampleUs": sample_us,
        "exact": exact,
        "branches": branches,
        "suspended": pauses,
        "outcomes": enumerated.items.len(),
        "probabilities": probabilities,
        "paths": paths_answer,
        "deepest": path_draws,
        "maxDiff": max_diff,
        "counts": counts,
        "unmatched": unmatched,
        "nothing": nothing,
        "inexactDraws": inexact,
        "draws": draws,
    })
}
