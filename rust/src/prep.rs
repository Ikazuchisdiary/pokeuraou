//! IKA-458 stage 3.5 `prepGame`: a game's preparation in one request.
//!
//! `poolplay.generate_pool` drew the pair and its seats from `default_rng([seed, index])`,
//! read the pair's selection solve (`SolvedSelections.entry`), drew both sides' four from it
//! (`BookEntry.draw`), built the two bench priors (`BenchPrior.of`), built the turn-1 position
//! (`selfplay._opening`) and ran the leads' switch-ins on it (`port.apply_lead_abilities`, one
//! request per draw they open). All of it is here: the caller names the pool, the seed and the
//! game's index, and the answer is the pair, the picks, the selection's policies and mixtures
//! (what the record carries), the start position and the generator's state after all that.
//!
//! * The pool (its teams' sets and pairs) crosses once: `poolDefine`, a line that is not
//!   answered. The solves are read from the store directory by this process, as the caller
//!   did; a pair with no file is `unsolved` and the caller solves it (it needs the leaf) and
//!   asks again.
//! * The generator is numpy's `PCG64` seeded as `default_rng([seed, index])` is (`advance.rs`),
//!   so the same index plays the same game as the Python road does, up to `np.exp` in the
//!   selection's exploration share (this crate's `f64::exp`).
//! * The game's two sheets and priors are defined here for `hiddenStep` under the caller's
//!   number (`hiddenbench::insert_game`), so they never cross.
//!
//! What this does not do is the selection solve itself, which needs the leaf; that stays with
//! the caller until the leaf is served from here.

use crate::advance::{choose_p, Pcg64};
use crate::hiddenbench::{insert_game, make_pokemon, sheet_set, Game, Prior, SheetSet};
use crate::lp::{f64s, f64s_text, sum_f64, sum_of};
use crate::position::{Field, Position, Side};
use crate::reg::{to_id, Reg};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// The games prepared, the positive control of the stage.
pub static PREPARED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct Team {
    id: String,
    sets: Vec<SheetSet>,
}

struct Pool {
    teams: Vec<Team>,
    pairs: Vec<(usize, usize)>,
    index_of: HashMap<(usize, usize), usize>,
}

/// One pair's solve as `BookEntry` stores it, for a one-class selection game.
struct Entry {
    selections: Vec<Vec<usize>>,
    our_strategy: Vec<f64>,
    our_ev_loss: Vec<f64>,
    their_strategy: Vec<f64>,
    their_ev_loss: Vec<f64>,
    class_weight: f64,
    value: f64,
}

thread_local! {
    static POOLS: RefCell<HashMap<u64, Pool>> = RefCell::new(HashMap::new());
    /// The solves read, by pool, pair and `store|tag` (a store of another tag is another run).
    static ENTRIES: RefCell<HashMap<(u64, usize, usize, String), Rc<Entry>>> = RefCell::new(HashMap::new());
}

/// `poolDefine`: a pool's teams and pairs under the caller's number.
pub fn define_pool(value: &Value) {
    let Some(id) = value["id"].as_u64() else { return };
    let teams: Vec<Team> = value["teams"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|t| Team {
                    id: t["id"].as_str().unwrap_or_default().to_string(),
                    sets: t["sets"].as_array().map(|s| s.iter().map(sheet_set).collect()).unwrap_or_default(),
                })
                .collect()
        })
        .unwrap_or_default();
    let pairs: Vec<(usize, usize)> = value["pairs"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| Some((p[0].as_u64()? as usize, p[1].as_u64()? as usize)))
                .collect()
        })
        .unwrap_or_default();
    let index_of = pairs.iter().enumerate().map(|(k, p)| (*p, k)).collect();
    POOLS.with(|pools| pools.borrow_mut().insert(id, Pool { teams, pairs, index_of }));
    ENTRIES.with(|entries| entries.borrow_mut().retain(|key, _| key.0 != id));}

/// `selection_book.explore_mixture`, with numpy's sums (`sum_f64`) and `exp`.
fn explore_mixture(equilibrium: &[f64], ev_loss: &[f64], epsilon: f64, temperature: f64) -> Result<Vec<f64>, String> {
    if equilibrium.len() != ev_loss.len() {
        return Err(format!("{} strategy against {} losses", equilibrium.len(), ev_loss.len()));
    }
    if !(0.0..=1.0).contains(&epsilon) {
        return Err(format!("epsilon must be a probability, got {epsilon}"));
    }
    if temperature <= 0.0 {
        return Err(format!("temperature must be > 0, got {temperature}"));
    }
    let n = equilibrium.len();
    let total = sum_f64(equilibrium);
    let base: Vec<f64> = if total > 0.0 {
        equilibrium.iter().map(|v| v / total).collect()
    } else {
        vec![1.0 / n as f64; n]
    };
    if epsilon == 0.0 {
        return Ok(base);
    }
    let shaped: Vec<f64> = if temperature.is_infinite() {
        vec![1.0 / n as f64; n]
    } else {
        let mut scores: Vec<f64> = ev_loss.iter().map(|l| -(l.max(0.0)) / temperature).collect();
        let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for s in &mut scores {
            *s -= top;
        }
        let mut shaped: Vec<f64> = scores.iter().map(|s| s.exp()).collect();
        let sum = sum_f64(&shaped);
        for v in &mut shaped {
            *v /= sum;
        }
        shaped
    };
    let mut mixed: Vec<f64> =
        base.iter().zip(&shaped).map(|(b, s)| (1.0 - epsilon) * b + epsilon * s).collect();
    let sum = sum_f64(&mixed);
    for v in &mut mixed {
        *v /= sum;
    }
    Ok(mixed)
}

/// `np.searchsorted(np.cumsum(p), u, side="right")`, kept below `len`.
fn cumulative_index(p: &[f64], u: f64) -> usize {
    let mut run = 0.0;
    let mut at = 0;
    for v in p {
        run += v;
        if run <= u {
            at += 1;
        }
    }
    at.min(p.len() - 1)
}

fn vector(value: &Value) -> Result<Vec<f64>, String> {
    value
        .as_array()
        .ok_or("a vector is not a list")?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| "a vector holds a non-number".to_string()))
        .collect()
}

enum Loaded {
    Entry(Rc<Entry>, bool),
    Unsolved,
    Stop(String),
    Fault(String),
}

/// The pair's solve with the lower index as the row player: from this process's memory, or
/// from the store's file (waiting out another worker's rename onto it, as `_read_shared`).
fn load(pool_id: u64, pool: &Pool, lo: usize, hi: usize, store: &str, tag: &str) -> Loaded {
    let key = (pool_id, lo, hi, format!("{store}|{tag}"));
    if let Some(found) = ENTRIES.with(|e| e.borrow().get(&key).cloned()) {
        return Loaded::Entry(found, false);
    }
    let path = std::path::Path::new(store).join(format!("{lo:03}-{hi:03}.json"));
    if !path.exists() {
        return Loaded::Unsolved;
    }
    let mut bytes = None;
    for attempt in 0..50 {
        match std::fs::read(&path) {
            Ok(read) => {
                bytes = Some(read);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied && attempt < 49 => {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(error) => return Loaded::Fault(format!("{}: {error}", path.display())),
        }
    }
    let data: Value = match serde_json::from_slice(&bytes.unwrap_or_default()) {
        Ok(data) => data,
        Err(error) => return Loaded::Fault(format!("{}: {error}", path.display())),
    };
    let teams = [pool.teams[lo].id.as_str(), pool.teams[hi].id.as_str()];
    let listed: Vec<&str> =
        data["teams"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    if data["tag"].as_str() != Some(tag) || listed != teams {
        return Loaded::Stop(format!(
            "{} was solved for tag {} and teams {}, not {:?} and {:?}: another pool or another leaf. \
             Give this run its own store.",
            path.display(),
            data["tag"],
            data["teams"],
            tag,
            teams
        ));
    }
    let entry = &data["entry"];
    let parsed = (|| -> Result<Entry, String> {
        let weights = vector(&entry["classWeights"])?;
        let theirs = entry["theirStrategies"].as_array().ok_or("no theirStrategies")?;
        let their_loss = entry["theirEvLoss"].as_array().ok_or("no theirEvLoss")?;
        if weights.len() != 1 || theirs.len() != 1 || their_loss.len() != 1 {
            return Err("a selection game with more than one class of the opponent".into());
        }
        Ok(Entry {
            selections: entry["selections"]
                .as_array()
                .ok_or("no selections")?
                .iter()
                .map(|s| {
                    s.as_array()
                        .map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
                        .unwrap_or_default()
                })
                .collect(),
            our_strategy: vector(&entry["ourStrategy"])?,
            our_ev_loss: vector(&entry["ourEvLoss"])?,
            their_strategy: vector(&theirs[0])?,
            their_ev_loss: vector(&their_loss[0])?,
            class_weight: weights[0],
            value: entry["value"].as_f64().ok_or("no value")?,
        })
    })();
    match parsed {
        Err(why) => Loaded::Fault(format!("{}: {why}", path.display())),
        Ok(entry) => {
            let entry = Rc::new(entry);
            ENTRIES.with(|e| e.borrow_mut().insert(key, entry.clone()));
            Loaded::Entry(entry, true)
        }
    }
}

/// `selfplay._opening`: the turn-1 position before the leads' switch-ins.
pub(crate) fn opening(reg: &Reg, sides: [&[&SheetSet]; 2]) -> Result<Position, String> {
    let mut built = Vec::new();
    for (index, sets) in sides.iter().enumerate() {
        let mut party: Vec<Rc<crate::position::Pokemon>> = Vec::new();
        for (slot, set) in sets.iter().enumerate() {
            let mut mon = make_pokemon(reg, set, slot)?;
            mon.active_index = if slot < reg.active_per_side { Some(slot) } else { None };
            party.push(Rc::new(mon));
        }
        let mega_capable_slots =
            party.iter().filter(|m| reg.holds_mega_stone(m.species, m.item)).map(|m| m.slot).collect();
        built.push(Side {
            id: Rc::from(format!("p{}", index + 1)),
            name: Rc::from(format!("p{}", index + 1)),
            active: (0..reg.active_per_side).map(Some).collect(),
            pokemon: party,
            side_conditions: Vec::new(),
            slot_conditions: vec![Vec::new(); reg.active_per_side],
            mega_used: false,
            mega_capable_slots,
        });
    }
    let second = built.pop().ok_or("no sides")?;
    let first = built.pop().ok_or("no sides")?;
    Ok(Position {
        format: Rc::from(reg.format_id.as_str()),
        sides: [first, second],
        turn: 1,
        field: Field::default(),
        request_state: Rc::from("move"),
        ended: false,
        winner: None,
    })
}

fn float(value: &Value) -> Option<f64> {
    f64s(value).ok().and_then(|v| v.first().copied())
}

pub fn prep_game(reg: &Reg, value: &Value) -> Value {
    let (Some(pool_id), Some(seed), Some(index)) =
        (value["pool"].as_u64(), value["seed"].as_u64(), value["index"].as_u64())
    else {
        return json!({ "fault": "prepGame without a pool, a seed and an index" });
    };
    let (Some(epsilon), Some(temperature)) = (float(&value["epsilon"]), float(&value["temperature"])) else {
        return json!({ "fault": "prepGame without epsilon and temperature" });
    };
    let store = value["store"].as_str().unwrap_or_default();
    let tag = value["tag"].as_str().unwrap_or_default();
    let hide_bench = value["hideBench"].as_bool().unwrap_or(false);
    let game_number = value["game"].as_u64();
    POOLS.with(|pools| {
        let pools = pools.borrow();
        let Some(pool) = pools.get(&pool_id) else {
            return json!({ "fault": format!("prepGame: no pool {pool_id} is defined") });
        };
        let mut rng = Pcg64::seeded(&[seed, index]);
        // `pool.draw_pair`: a uniform pair, then a fair coin for the seats.
        let subset: Option<Vec<(usize, usize)>> = value["pairs"].as_array().map(|a| {
            a.iter().filter_map(|p| Some((p[0].as_u64()? as usize, p[1].as_u64()? as usize))).collect()
        });
        let drawn_from: &[(usize, usize)] = subset.as_deref().unwrap_or(&pool.pairs);
        if drawn_from.is_empty() {
            return json!({ "fault": "prepGame: no pairs to draw from" });
        }
        let k_drawn = rng.integers(drawn_from.len() as u64) as usize;
        let swap = rng.next_double() < 0.5;
        let (first, second) = drawn_from[k_drawn];
        let (a, b) = if swap { (second, first) } else { (first, second) };
        let k = if subset.is_some() {
            match pool.index_of.get(&(a.min(b), a.max(b))) {
                Some(k) => *k,
                None => return json!({ "fault": "prepGame: a pair outside the pool" }),
            }
        } else {
            k_drawn
        };
        if a >= pool.teams.len() || b >= pool.teams.len() {
            return json!({ "fault": "prepGame: a team outside the pool" });
        }
        let (lo, hi) = (a.min(b), a.max(b));
        let (entry, read) = match load(pool_id, pool, lo, hi, store, tag) {
            Loaded::Entry(entry, read) => (entry, read),
            Loaded::Unsolved => return json!({ "unsolved": [lo, hi] }),
            Loaded::Stop(why) => return json!({ "stop": why }),
            Loaded::Fault(why) => return json!({ "fault": why }),
        };
        // The solve read with the seat-0 team as the row player (`SolvedSelections.entry`).
        let (our_strategy, our_loss, their_strategy, their_loss, solve_value) = if a <= b {
            (&entry.our_strategy, &entry.our_ev_loss, &entry.their_strategy, &entry.their_ev_loss, entry.value)
        } else {
            (&entry.their_strategy, &entry.their_ev_loss, &entry.our_strategy, &entry.our_ev_loss, 1.0 - entry.value)
        };
        // `BookEntry.draw`: ours first, then the class, then theirs.
        let drawn = (|| -> Result<_, String> {
            let ours = explore_mixture(our_strategy, our_loss, epsilon, temperature)?;
            let our_index = cumulative_index(&ours, rng.next_double());
            // One class: its weight over itself is exactly 1, and the draw is still taken.
            let class_weight = if a <= b { entry.class_weight } else { 1.0 };
            let _class = cumulative_index(&[class_weight / class_weight], rng.next_double());
            let theirs = explore_mixture(their_strategy, their_loss, epsilon, temperature)?;
            let their_index = cumulative_index(&theirs, rng.next_double());
            Ok((ours, theirs, our_index, their_index))
        })();
        let (ours, theirs, our_index, their_index) = match drawn {
            Ok(drawn) => drawn,
            Err(why) => return json!({ "fault": why }),
        };
        let picks = [entry.selections[our_index].clone(), entry.selections[their_index].clone()];
        let sets = [&pool.teams[a].sets, &pool.teams[b].sets];
        let mut chosen: [Vec<&SheetSet>; 2] = [Vec::new(), Vec::new()];
        for side in 0..2 {
            for i in &picks[side] {
                match sets[side].get(*i) {
                    Some(set) => chosen[side].push(set),
                    None => return json!({ "fault": "prepGame: a pick outside the six" }),
                }
            }
        }
        let start = match opening(reg, [&chosen[0], &chosen[1]]) {
            Ok(start) => start,
            Err(why) => return json!({ "fault": why }),
        };
        let led = crate::resolve::commands::lead_phase(reg, start, |weights| {
            // `rng.choice(len(weights), p=[w / total for w in weights])`.
            let total = sum_of(weights.iter().copied());
            let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
            choose_p(&p, &mut rng)
        });
        let position = match led {
            Ok(position) => position,
            Err(why) => return json!({ "fault": why }),
        };
        if let (true, Some(number)) = (hide_bench, game_number) {
            let foe_weight = if a <= b { entry.class_weight } else { 1.0 };
            let foe_prior: Vec<f64> = theirs.iter().map(|p| foe_weight * p).collect();
            let species = |side: usize| -> Vec<String> { sets[side].iter().map(|s| to_id(&s.species)).collect() };
            let prior = |side: usize, probabilities: &Vec<f64>| Prior {
                selections: entry.selections.clone(),
                probabilities: probabilities.clone(),
                species: species(side),
            };
            // `BenchPrior.of`: side 0's is our mixture, side 1's the one class's mixture
            // weighted by the class's weight (1.0 here).
            let game = Game {
                sheets: [sets[0].clone(), sets[1].clone()],
                priors: [Some(prior(0, &ours)), Some(prior(1, &foe_prior))],
                made: HashMap::new(),
            };
            insert_game(number, game);
        }
        PREPARED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        json!({
            "pair": k,
            "teams": [a, b],
            "mirror": a == b,
            "picks": picks,
            "ownPolicy": f64s_text(our_strategy),
            "foePolicy": f64s_text(their_strategy),
            "ownMixture": f64s_text(&ours),
            "foeMixture": f64s_text(&theirs),
            "value": solve_value,
            "read": read,
            "position": position.to_json(),
            "rng": rng.to_json_full(),
        })
    })
}
