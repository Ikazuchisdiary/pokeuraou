//! IKA-458 stage 3.4 `hiddenStep`: what a side has shown, and every position its unseen
//! slots could hold, in one request.
//!
//! `selfplay.play_game` carried each side's seen identities from decision to decision
//! (`hidden.seen_identities`), turned them into the position's seen slots, priced every
//! way of filling the unseen slots with the selection prior (`BenchPrior.weights`) and
//! built each completed position (`hidden.completions`, `substitute`, `_make_pokemon`).
//! All of it is here. The caller sends the position, the identities carried so far and the
//! game's lead pairs; the answer is the identities now, the slots seen, and each side's
//! completions: the species and slots filled, the weight, and the completed position kept in
//! this process under a number (`held::keep`) that later requests name. The caller never
//! reads a completed position, so none crosses the pipe.
//!
//! The two sheets and the two selection priors do not change in a game, so they cross once:
//! `hiddenGame` defines them under the caller's number (a line that is not answered).
//!
//! A game this process cannot read (no such number) and a prior it cannot use are a `fault`:
//! the caller then builds the completions as before. A sheet that cannot explain the board
//! is an `error` per side, as `completions`' `ValueError` was.

use crate::lp::python_sum;
use crate::position::{Pokemon, Position, STAT_IDS};
use crate::reg::{to_id, Reg};
use serde_json::{json, Value};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;

#[derive(Clone)]
pub(crate) struct SheetSet {
    pub(crate) species: String,
    pub(crate) ability: String,
    pub(crate) item: Option<String>,
    pub(crate) nature: String,
    pub(crate) sp: [i64; 6],
    pub(crate) moves: Vec<String>,
}

pub(crate) struct Prior {
    pub(crate) selections: Vec<Vec<usize>>,
    pub(crate) probabilities: Vec<f64>,
    /// The sheet's species as ids, in the order `selections` indexes.
    pub(crate) species: Vec<String>,
}

pub(crate) struct Game {
    pub(crate) sheets: [Vec<SheetSet>; 2],
    pub(crate) priors: [Option<Prior>; 2],
    /// The Pokemon a set makes in a party slot, built once (`_make_pokemon`).
    pub(crate) made: HashMap<(usize, usize, usize), Rc<Pokemon>>,
}

thread_local! {
    static GAMES: RefCell<HashMap<u64, Game>> = RefCell::new(HashMap::new());
}

/// Past this many games the table starts again (a bound; a worker plays one at a time).
const GAMES_MAX: usize = 16;

pub(crate) fn sheet_set(value: &Value) -> SheetSet {
    let mut sp = [0i64; 6];
    for (index, name) in STAT_IDS.iter().enumerate() {
        sp[index] = value["sp"].get(*name).and_then(Value::as_i64).unwrap_or(0);
    }
    SheetSet {
        species: value["species"].as_str().unwrap_or_default().to_string(),
        ability: value["ability"].as_str().unwrap_or_default().to_string(),
        item: value.get("item").and_then(Value::as_str).map(String::from),
        nature: value["nature"].as_str().unwrap_or_default().to_string(),
        sp,
        moves: value["moves"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect())
            .unwrap_or_default(),
    }
}

fn prior(value: &Value) -> Option<Prior> {
    if value.is_null() {
        return None;
    }
    Some(Prior {
        selections: value["selections"]
            .as_array()?
            .iter()
            .map(|s| s.as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|v| v as usize).collect()).unwrap_or_default())
            .collect(),
        // Base64, not JSON numbers: serde_json reads a decimal one ulp off at times.
        probabilities: crate::lp::f64s(&value["probabilities"]).ok()?,
        species: value["species"].as_array()?.iter().filter_map(Value::as_str).map(to_id).collect(),
    })
}

/// `hiddenGame`: a game's two sheets and priors under the caller's number.
pub fn define_game(value: &Value) {
    let Some(id) = value["id"].as_u64() else { return };
    let sheets_of = |side: usize| -> Vec<SheetSet> {
        value["sheets"][side].as_array().map(|a| a.iter().map(sheet_set).collect()).unwrap_or_default()
    };
    let game = Game {
        sheets: [sheets_of(0), sheets_of(1)],
        priors: [prior(&value["priors"][0]), prior(&value["priors"][1])],
        made: HashMap::new(),
    };
    insert_game(id, game);
}

/// A game under the caller's number: the table starts again past `GAMES_MAX`.
pub(crate) fn insert_game(id: u64, game: Game) {
    GAMES.with(|games| {
        let mut games = games.borrow_mut();
        if games.len() >= GAMES_MAX {
            games.clear();
        }
        games.insert(id, game);
    });
}

/// `hidden._shows_itself`: whether the position alone says this Pokemon has been out.
fn shows_itself(mon: &Pokemon) -> bool {
    mon.active_index.is_some()
        || mon.fainted
        || mon.hp != mon.maxhp
        || mon.status.is_some()
        || mon.boosts.iter().any(|b| *b != 0)
        || !mon.volatiles.is_empty()
        || mon.is_mega
}

fn identity(mon: &Pokemon) -> String {
    to_id(mon.base_species.as_str())
}

/// `hidden.shown_species`: the ids a side has revealed, each Mega's base form alongside.
fn shown_species(position: &Position, side: usize, shown: &BTreeSet<usize>) -> HashSet<String> {
    let mut out = HashSet::new();
    for mon in &position.sides[side].pokemon {
        if !shown.contains(&mon.slot) {
            continue;
        }
        if !mon.transformed {
            out.insert(to_id(mon.species.as_str()));
        }
        out.insert(to_id(mon.base_species.as_str()));
    }
    out
}

/// `selection_book.bench_weights`: the prior's mass on each way of filling the unseen
/// slots, keyed by the sorted ids of the sheet members not yet seen, in the prior's order.
fn bench_weights(
    prior: &Prior,
    seen: &HashSet<String>,
    leads: Option<&[String]>,
) -> Result<Vec<(Vec<String>, f64)>, String> {
    let sheet: HashSet<&str> = prior.species.iter().map(String::as_str).collect();
    let led: Option<HashSet<String>> = match leads {
        None => None,
        Some(names) => {
            let led: HashSet<String> =
                names.iter().map(|n| to_id(n)).filter(|n| sheet.contains(n.as_str())).collect();
            if led.len() != 2 {
                return Err(format!("leads {names:?} name {} sheet members, not the 2 that led", led.len()));
            }
            Some(led)
        }
    };
    let want: HashSet<&str> = seen.iter().map(String::as_str).filter(|n| sheet.contains(n)).collect();
    let mut out: Vec<(Vec<String>, f64)> = Vec::new();
    for (selection, probability) in prior.selections.iter().zip(&prior.probabilities) {
        if *probability <= 0.0 {
            continue;
        }
        let brought: Vec<&str> = selection.iter().map(|i| prior.species[*i].as_str()).collect();
        if !want.iter().all(|w| brought.contains(w)) {
            continue;
        }
        if let Some(led) = &led {
            let front: HashSet<String> = brought.iter().take(2).map(|s| s.to_string()).collect();
            if &front != led {
                continue;
            }
        }
        let mut key: Vec<String> =
            brought.iter().filter(|n| !want.contains(*n)).map(|n| n.to_string()).collect();
        key.sort();
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some((_, mass)) => *mass += *probability,
            None => out.push((key, *probability)),
        }
    }
    let total = python_sum(out.iter().map(|(_, v)| *v));
    if total > 0.0 {
        for (_, v) in &mut out {
            *v /= total;
        }
        Ok(out)
    } else {
        Ok(Vec::new())
    }
}

/// `selfplay._make_pokemon`: a fresh Pokemon from a set, at full health, in party slot `slot`.
pub(crate) fn make_pokemon(reg: &Reg, set: &SheetSet, slot: usize) -> Result<Pokemon, String> {
    let species = reg
        .species
        .get(&crate::id::Id::new(&set.species))
        .ok_or_else(|| format!("unknown species {}", set.species))?;
    let stats = reg.stats_from_sp(species, &set.nature, &set.sp)?;
    let mut moves = Vec::new();
    for name in &set.moves {
        let found = reg
            .moves
            .get(&crate::id::Id::new(name))
            .ok_or_else(|| format!("unknown move {name}"))?;
        let pp = found.raw["startPP"].as_i64().ok_or_else(|| format!("move {name} has no startPP"))?;
        moves.push(json!({ "id": name, "pp": pp, "maxpp": pp }));
    }
    let mut sp = serde_json::Map::new();
    for (index, name) in STAT_IDS.iter().enumerate() {
        sp.insert((*name).into(), json!(set.sp[index]));
    }
    let item = match &set.item {
        Some(item) => json!(item),
        None => Value::Null,
    };
    Ok(Pokemon::from_json(&json!({
        "slot": slot,
        "species": set.species,
        "baseSpecies": set.species,
        "types": species.types,
        "ability": set.ability,
        "nature": set.nature,
        "moves": moves,
        "hp": stats[0],
        "maxhp": stats[0],
        "item": item,
        "baseItem": item,
        "sp": sp,
        "level": reg.level,
    })))
}

fn strings(value: &Value) -> Vec<String> {
    value.as_array().map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect()).unwrap_or_default()
}

/// One side's completions as `hidden.completions` lists them, or why it cannot.
fn side_completions(
    reg: &Reg,
    game: &mut Game,
    position: &Position,
    side: usize,
    shown: &BTreeSet<usize>,
    weights: Option<&Vec<(Vec<String>, f64)>>,
) -> Result<Vec<Value>, String> {
    let party = &position.sides[side].pokemon;
    let hidden: Vec<usize> = party.iter().map(|m| m.slot).filter(|s| !shown.contains(s)).collect();
    if hidden.is_empty() {
        return Ok(vec![json!({ "exact": true, "species": [], "slots": [], "weight": 1.0, "held": Value::Null })]);
    }
    let on_board = shown_species(position, side, shown);
    let candidates: Vec<usize> = (0..game.sheets[side].len())
        .filter(|k| !on_board.contains(&to_id(&game.sheets[side][*k].species)))
        .collect();
    if candidates.len() < hidden.len() {
        return Err(format!(
            "{} unseen slots on side {side} but only {} sheet members unaccounted for",
            hidden.len(),
            candidates.len()
        ));
    }
    // itertools.combinations(range(len(candidates)), len(hidden)), in its order.
    let (n, r) = (candidates.len(), hidden.len());
    let mut at: Vec<usize> = (0..r).collect();
    let mut made: Vec<(Vec<usize>, f64)> = Vec::new();
    loop {
        let picked: Vec<usize> = at.iter().map(|i| candidates[*i]).collect();
        let mut key: Vec<&str> = picked.iter().map(|k| game.sheets[side][*k].species.as_str()).collect();
        key.sort();
        let weight = weights
            .and_then(|w| w.iter().find(|(k, _)| k.iter().map(String::as_str).eq(key.iter().copied())))
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        made.push((picked, weight));
        // The next combination.
        let mut i = r;
        let found = loop {
            if i == 0 {
                break None;
            }
            i -= 1;
            if at[i] != i + n - r {
                break Some(i);
            }
        };
        let Some(i) = found else { break };
        at[i] += 1;
        for j in i + 1..r {
            at[j] = at[j - 1] + 1;
        }
    }
    let total = python_sum(made.iter().map(|(_, w)| *w));
    let count = made.len();
    let mut out = Vec::with_capacity(count);
    for (picked, weight) in made {
        let share = if total <= 0.0 { 1.0 / count as f64 } else { weight / total };
        let mut completed = position.clone();
        {
            let mons = &mut completed.sides[side].pokemon;
            for (slot, k) in hidden.iter().zip(&picked) {
                let cached = match game.made.get(&(side, *k, *slot)) {
                    Some(mon) => mon.clone(),
                    None => {
                        let mon = Rc::new(make_pokemon(reg, &game.sheets[side][*k], *slot)?);
                        game.made.insert((side, *k, *slot), mon.clone());
                        mon
                    }
                };
                for mon in mons.iter_mut() {
                    if mon.slot == *slot {
                        *mon = cached.clone();
                    }
                }
            }
        }
        let capable: Vec<usize> = completed.sides[side]
            .pokemon
            .iter()
            .filter(|m| reg.holds_mega_stone(m.species, m.item))
            .map(|m| m.slot)
            .collect();
        completed.sides[side].mega_capable_slots = capable;
        let pk: Vec<Value> = completed.sides[side]
            .pokemon
            .iter()
            .map(|m| json!([to_id(m.species.as_str()), to_id(m.base_species.as_str()), m.slot]))
            .collect();
        let held = crate::held::keep(&completed);
        out.push(json!({
            "exact": false,
            "species": picked.iter().map(|k| game.sheets[side][*k].species.clone()).collect::<Vec<_>>(),
            "slots": hidden,
            "weight": share,
            "held": held,
            "pk": pk,
        }));
    }
    Ok(out)
}

pub fn hidden_step(reg: &Reg, value: &Value) -> Value {
    let Some(id) = value["game"].as_u64() else {
        return json!({ "fault": "hiddenStep without a game" });
    };
    let position = crate::held::position(&value["position"]);
    GAMES.with(|games| {
        let mut games = games.borrow_mut();
        let Some(game) = games.get_mut(&id) else {
            return json!({ "fault": format!("hiddenStep: no game {id} is defined") });
        };
        let mut notes: Vec<String> = Vec::new();
        let mut sides = Vec::new();
        let mut seen_out = Vec::new();
        let mut shown_out = Vec::new();
        let mut shown_sets = Vec::new();
        for side in 0..2 {
            let mut seen: BTreeSet<String> = strings(&value["seen"][side]).into_iter().collect();
            for mon in &position.sides[side].pokemon {
                if shows_itself(mon) {
                    seen.insert(identity(mon));
                }
            }
            let shown: BTreeSet<usize> = position.sides[side]
                .pokemon
                .iter()
                .filter(|m| shows_itself(m) || seen.contains(&identity(m)))
                .map(|m| m.slot)
                .collect();
            seen_out.push(seen.iter().cloned().collect::<Vec<_>>());
            shown_out.push(shown.iter().copied().collect::<Vec<_>>());
            shown_sets.push(shown);
        }
        let mut failed = false;
        for side in 0..2 {
            if failed {
                sides.push(Value::Null);
                continue;
            }
            let shown = &shown_sets[side];
            let leads = if value["leads"][side].is_null() { None } else { Some(strings(&value["leads"][side])) };
            let weights = match &game.priors[side] {
                None => None,
                Some(prior) => {
                    match bench_weights(prior, &shown_species(&position, side, shown), leads.as_deref()) {
                        Err(why) => {
                            sides.push(json!({ "error": why }));
                            failed = true;
                            continue;
                        }
                        Ok(found) if found.is_empty() => {
                            notes.push(format!(
                                "bench weights: side {side}'s selection prior explains nothing on board"
                            ));
                            None
                        }
                        Ok(found) => Some(found),
                    }
                }
            };
            match side_completions(reg, game, &position, side, shown, weights.as_ref()) {
                Ok(items) => sides.push(json!({ "items": items })),
                Err(why) => {
                    sides.push(json!({ "error": why }));
                    failed = true;
                }
            }
        }
        json!({ "seen": seen_out, "shown": shown_out, "sides": sides, "notes": notes })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prior() -> Prior {
        // Six species a..f: two selections lead (a, b), one leads (b, a).
        Prior {
            selections: vec![vec![0, 1, 2, 3], vec![0, 1, 4, 5], vec![1, 0, 2, 3]],
            probabilities: vec![0.5, 0.25, 0.25],
            species: ["a", "b", "c", "d", "e", "f"].map(String::from).to_vec(),
        }
    }

    #[test]
    fn the_prior_is_conditioned_on_what_is_seen_and_who_led() {
        let seen: HashSet<String> = ["a", "b"].map(String::from).into();
        let all = bench_weights(&prior(), &seen, None).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0], (vec!["c".to_string(), "d".to_string()], 0.75));
        assert_eq!(all[1], (vec!["e".to_string(), "f".to_string()], 0.25));
        // Led (a, b) as a set: the order inside the pair does not matter.
        let led = ["a".to_string(), "b".to_string()];
        assert_eq!(bench_weights(&prior(), &seen, Some(&led)).unwrap(), all);
        // Nobody led (c, d): nothing explains the board.
        let wrong = ["c".to_string(), "d".to_string()];
        assert!(bench_weights(&prior(), &seen, Some(&wrong)).unwrap().is_empty());
        // A lead pair naming fewer than two sheet members is an error.
        assert!(bench_weights(&prior(), &seen, Some(&["a".to_string()])).is_err());
    }
}
