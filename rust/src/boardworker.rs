//! IKA-458 stage 3.7: the board's worker as this executable (`pokeuraou-damage pool-match-worker`).
//!
//! `tools/pool_match.py` played a board's games in Python: per queue index `i` it played game
//! `i // 2` with the tested arm at side `i % 2` (`poolplay.pool_match_game`), each arm with its own
//! leaf, width, Q, selection store and executable, and wrote the record with its provenance. This
//! process does the same game itself, with the functions the port's requests call:
//!
//! * the pair and its seats from `default_rng([seed, game])` (`pool.draw_pair`);
//! * each side's four from its own stream `default_rng([seed, game, 1, side])` and the solve of the
//!   arm sitting there (`poolplay.draw_side`), solved here with that arm's leaf and written to that
//!   arm's store when the store has no file; an arm without a store (hp-share) draws four of six
//!   uniformly (`teams.pick_four_indices`, numpy's `choice` without replacement);
//! * each side's bench belief from the solve of the arm across (`BenchPrior.of`), uniform for an
//!   arm without one;
//! * the turn-1 position from the game's stream, and the game by `playGame` with each side's own
//!   node (`playgame.rs`'s `nodes`; one node when the two arms are the same agent).
//!
//! What a run of the board shares comes from one JSON file `rustworker.board_config` writes
//! (`--config`): the pool, the arms (their leaves' and Qs' servers, widths, stores, executables), the
//! two record templates (tested arm at side 0, at side 1) in `write_game`'s order. Per worker: the
//! queue, the inference servers, the games file and the summary file.
//!
//! ```text
//! pokeuraou-damage pool-match-worker --config C --inference HOST:PORT [--baseline-inference HOST:PORT]
//!     --games-out FILE [--out FILE] (--queue HOST:PORT | --first-index K --indices N)
//! ```

use crate::advance::Pcg64;
use crate::encode::Encoder;
use crate::hiddenbench::{identity, insert_game, shown_species, Game, Prior, SheetSet};
use crate::position::Position;
use crate::prep::{mixture, start_position, stored, Entry, Stored};
use crate::reg::{to_id, Reg};
use crate::selection::{solve_pair, TeamOf};
use crate::served::Server;
use crate::shm::Shared;
use crate::worker::{arg, ordered, ordered_raw, self_switch_decision, teams_of, Queue, Team};
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::rc::Rc;
use std::time::Instant;

/// The pool's number in this process (`poolDefine`'s `id`).
const POOL: u64 = 2;

/// The keys of a record this worker writes per game; every other key of the template is copied.
const PER_GAME: &[&str] = &[
    "ownTeam", "foeTeam", "foeArchetype", "outcome", "turns", "searchSeconds", "foeSpecies", "ownSix",
    "foeSix", "ownPick", "foePick", "unmodelled", "leads", "finalPosition", "endReason", "decisions",
    "gameIndex", "pool",
];

/// One arm of the board.
struct Arm {
    name: String,
    /// Its leaf: an index into the config's `servers` (the context's order).
    leaf: usize,
    limit: u64,
    /// Its selection store and the tag its files carry; None draws uniformly (hp-share).
    store: Option<(String, String, String)>,
    /// Its node's static request, with its servers' blocks and addresses filled in.
    node: Value,
}

#[derive(Default)]
struct Stats {
    /// Per seat (tested at side 0, at side 1): wins, played, unfinished, seconds.
    seats: [(u64, u64, u64, f64); 2],
    handoffs: u64,
    solves: u64,
    loaded: u64,
    reused: u64,
    solve_seconds: f64,
    solved_here: HashSet<(usize, usize, usize)>,
    moves: u64,
    split: u64,
    replacements: u64,
    selfswitches: u64,
    /// Per arm: the selection it drew by, and the belief it held (the echo).
    selections: [HashMap<&'static str, u64>; 2],
}

pub fn main(args: &[String]) {
    if let Err(why) = run(args) {
        eprintln!("pool-match-worker: {why}");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let config_path = arg(args, "--config").ok_or("--config is required")?;
    let address = arg(args, "--inference").ok_or("--inference is required")?;
    let baseline_address = arg(args, "--baseline-inference").unwrap_or_else(|| address.clone());
    let games_out = arg(args, "--games-out").ok_or("--games-out is required")?;
    let summary_out = arg(args, "--out");
    let config: Value = serde_json::from_slice(&std::fs::read(&config_path).map_err(|e| format!("{config_path}: {e}"))?)
        .map_err(|e| format!("{config_path}: {e}"))?;
    let regulation = config["regulation"].as_str().ok_or("the config has no regulation")?.to_string();
    let reg = Reg::load(&regulation).map_err(|e| format!("regulation: {e}"))?;
    let encoder = Encoder::new(&reg);

    // Each leaf's and each Q's block, this process's own.
    let pid = std::process::id();
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut blocks: Vec<Shared> = Vec::new();
    let mut servers: Vec<Value> = Vec::new();
    for (k, s) in config["servers"].as_array().ok_or("the config has no servers")?.iter().enumerate() {
        let bytes = s["bytes"].as_u64().ok_or("a server without bytes")? as usize;
        let block = Shared::create(&format!("pokeuraou_b{pid}_{stamp:x}_v{k}"), bytes).ok_or("could not create a leaf's block")?;
        let mut server = s.clone();
        let at = if s["role"] == "baseline-inference" { &baseline_address } else { &address };
        server["address"] = json!(at);
        server["shm"] = json!(block.name());
        server["bytes"] = json!(block.capacity());
        if let Some(map) = server.as_object_mut() {
            map.remove("role");
        }
        servers.push(server);
        blocks.push(block);
    }
    let mut qs: Vec<Value> = Vec::new();
    for (k, q) in config["qServers"].as_array().ok_or("the config has no qServers")?.iter().enumerate() {
        let bytes = q["bytes"].as_u64().ok_or("a Q without bytes")? as usize;
        let block = Shared::create(&format!("pokeuraou_b{pid}_{stamp:x}_q{k}"), bytes).ok_or("could not create a Q's block")?;
        let mut server = q.clone();
        server["address"] = json!(address);
        server["shm"] = json!(block.name());
        server["bytes"] = json!(block.capacity());
        qs.push(server);
        blocks.push(block);
    }
    let leaf_servers: Vec<Server> = servers.iter().map(Server::from_json).collect::<Result<_, _>>()?;
    let mut arms: Vec<Arm> = Vec::new();
    for a in config["arms"].as_array().ok_or("the config has no arms")? {
        let leaf = a["leaf"].as_u64().ok_or("an arm without a leaf")? as usize;
        let q = a["q"].as_u64().ok_or("an arm without a Q")? as usize;
        let mut node = a["node"].clone();
        let mut server = servers.get(leaf).ok_or("an arm's leaf is not a server")?.clone();
        if let Some(map) = server.as_object_mut() {
            map.remove("netScoresEnds");
            map.remove("encoding");
        }
        node["server"] = server;
        node["qServer"] = qs.get(q).ok_or("an arm's Q is not a server")?.clone();
        if let Some(binary) = a["binary"].as_str() {
            node["binary"] = json!(binary);
            node["regulation"] = json!(regulation);
        }
        let store = match a["store"].as_str() {
            Some(store) => Some((
                store.to_string(),
                a["tag"].as_str().ok_or("an arm with a store but no tag")?.to_string(),
                a["model"].as_str().unwrap_or_default().to_string(),
            )),
            None => None,
        };
        arms.push(Arm {
            name: a["name"].as_str().unwrap_or_default().to_string(),
            leaf,
            limit: a["limit"].as_u64().ok_or("an arm without a limit")?,
            store,
            node,
        });
    }
    if arms.len() != 2 {
        return Err("a board has two arms".into());
    }
    let pool_value = &config["pool"];
    let teams = teams_of(pool_value)?;
    crate::prep::define_pool(&json!({ "id": POOL, "teams": pool_value["teams"], "pairs": pool_value["pairs"] }));
    let pool_pairs: Vec<(usize, usize)> = pool_pairs(&pool_value["pairs"]);
    let subset: Option<Vec<(usize, usize)>> = config["pairs"].as_array().map(|_| pool_pairs_of(&config["pairs"]));
    let templates: Vec<Vec<(String, Value)>> = config["records"]
        .as_array()
        .ok_or("the config has no records")?
        .iter()
        .map(|t| {
            t.as_array()
                .map(|pairs| pairs.iter().map(|p| (p[0].as_str().unwrap_or_default().to_string(), p[1].clone())).collect())
                .unwrap_or_default()
        })
        .collect();
    if templates.len() != 2 {
        return Err("the config has not two record templates".into());
    }
    for template in &templates {
        let missing: Vec<&&str> = PER_GAME.iter().filter(|k| !template.iter().any(|(t, _)| t == *k)).collect();
        if !missing.is_empty() {
            return Err(format!("the record template has no {missing:?}"));
        }
    }
    let label = config["label"].as_str().unwrap_or_default().to_string();
    let seed = config["seed"].as_u64().ok_or("the config has no seed")?;
    let max_turns = config["maxTurns"].as_u64().ok_or("the config has no maxTurns")?;
    let epsilon = config["epsilon"].as_f64().ok_or("the config has no epsilon")?;
    let temperature = config["temperature"].as_f64().ok_or("the config has no temperature")?;
    let ctx = Run {
        reg: &reg,
        encoder: &encoder,
        teams: &teams,
        arms: &arms,
        servers: &servers,
        leaf_servers: &leaf_servers,
        pool_value,
        pool_pairs: &pool_pairs,
        subset: subset.as_deref(),
        templates: &templates,
        config: &config,
        seed,
        max_turns,
        epsilon,
        temperature,
    };

    let mut queue = match arg(args, "--queue") {
        Some(at) => Some(Queue::connect(&at)?),
        None => None,
    };
    let first: u64 = arg(args, "--first-index").map(|v| v.parse().map_err(|_| "--first-index")).transpose()?.unwrap_or(0);
    let count: u64 = arg(args, "--indices").map(|v| v.parse().map_err(|_| "--indices")).transpose()?.unwrap_or(0);
    if queue.is_none() && count == 0 {
        return Err("give --queue, or --indices (with --first-index)".into());
    }
    if let Some(parent) = std::path::Path::new(&games_out).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&games_out)
        .map_err(|e| format!("{games_out}: {e}"))?;
    eprintln!(
        "rust pool-match worker (IKA-458 stage 3.7): pool {} ({} teams), seed {seed}, {label}",
        pool_value["id"].as_str().unwrap_or("?"),
        teams.len()
    );
    for (k, arm) in arms.iter().enumerate() {
        eprintln!(
            "  {} arm: leaf {} ({}) / width {} / selection {} / belief {}{}",
            if k == 0 { "tested" } else { "other" },
            arm.name,
            servers[arm.leaf]["model"].as_str().unwrap_or("?"),
            arm.limit,
            if arm.store.is_some() { "solved" } else { "uniform" },
            if arm.store.is_some() { "solved" } else { "uniform" },
            arm.node["binary"].as_str().map(|b| format!(" / port {}", crate::armexe::fingerprint(b))).unwrap_or_default(),
        );
    }

    let started = Instant::now();
    let mut stats = Stats::default();
    let mut number: u64 = 0;
    let mut next_plain = first;
    loop {
        let index = match queue.as_mut() {
            Some(q) => match q.take()? {
                Some(index) => index,
                None => break,
            },
            None => {
                if next_plain >= first + count {
                    break;
                }
                next_plain += 1;
                next_plain - 1
            }
        };
        number += 1;
        let began = Instant::now();
        let which = (index % 2) as usize;
        if let Some(line) = play_one(&ctx, index, number, &mut stats)? {
            out.write_all(line.as_bytes())
                .and_then(|()| out.write_all(b"\n"))
                .and_then(|()| out.flush())
                .map_err(|e| format!("{games_out}: {e}"))?;
            stats.seats[which].3 += began.elapsed().as_secs_f64();
        }
        if let Some(q) = queue.as_mut() {
            q.finish(index)?;
        }
    }
    report(&ctx, &stats, &label, started.elapsed().as_secs_f64(), summary_out.as_deref())
}

fn pool_pairs(value: &Value) -> Vec<(usize, usize)> {
    value
        .as_array()
        .map(|a| a.iter().filter_map(|p| Some((p[0].as_u64()? as usize, p[1].as_u64()? as usize))).collect())
        .unwrap_or_default()
}

fn pool_pairs_of(value: &Value) -> Vec<(usize, usize)> {
    pool_pairs(value)
}

/// What every game of the run reads.
struct Run<'a> {
    reg: &'a Reg,
    encoder: &'a Encoder<'a>,
    teams: &'a [Team],
    arms: &'a [Arm],
    servers: &'a [Value],
    leaf_servers: &'a [Server],
    pool_value: &'a Value,
    pool_pairs: &'a [(usize, usize)],
    subset: Option<&'a [(usize, usize)]>,
    templates: &'a [Vec<(String, Value)>],
    config: &'a Value,
    seed: u64,
    max_turns: u64,
    epsilon: f64,
    temperature: f64,
}

/// A solve read with the seat-0 team as the row player (`SolvedSelections.entry`, `transposed`).
struct Oriented {
    entry: Rc<Entry>,
    flipped: bool,
}

impl Oriented {
    fn ours(&self) -> (&[f64], &[f64]) {
        if self.flipped {
            (&self.entry.their_strategy, &self.entry.their_ev_loss)
        } else {
            (&self.entry.our_strategy, &self.entry.our_ev_loss)
        }
    }

    fn theirs(&self) -> (&[f64], &[f64]) {
        if self.flipped {
            (&self.entry.our_strategy, &self.entry.our_ev_loss)
        } else {
            (&self.entry.their_strategy, &self.entry.their_ev_loss)
        }
    }

    fn class_weight(&self) -> f64 {
        if self.flipped { 1.0 } else { self.entry.class_weight }
    }
}

/// `arm.solver.entry(a, b)`: from the arm's store, solved here with its leaf when the store has no file.
fn entry_of(ctx: &Run, arm_index: usize, a: usize, b: usize, stats: &mut Stats) -> Result<Option<Oriented>, String> {
    let arm = &ctx.arms[arm_index];
    let Some((store, tag, model)) = &arm.store else { return Ok(None) };
    let (lo, hi) = (a.min(b), a.max(b));
    let key = (arm_index, lo, hi);
    let entry = match stored(POOL, lo, hi, store, tag)? {
        Stored::Entry(entry, read) => {
            if read && !stats.solved_here.contains(&key) {
                stats.loaded += 1;
            } else {
                stats.reused += 1;
            }
            entry
        }
        Stored::Unsolved => {
            let team = |t: usize| TeamOf {
                id: &ctx.teams[t].id,
                name: &ctx.teams[t].name,
                sets: &ctx.teams[t].sets,
                book_sets: &ctx.teams[t].book_sets,
            };
            let solved = solve_pair(
                ctx.reg, ctx.encoder, &ctx.leaf_servers[arm.leaf], &team(lo), &team(hi), lo, hi, store, tag, model,
                &ctx.config["notes"],
            )?;
            stats.solves += 1;
            stats.solve_seconds += solved.seconds;
            stats.solved_here.insert(key);
            match stored(POOL, lo, hi, store, tag)? {
                Stored::Entry(entry, _) => entry,
                Stored::Unsolved => return Err(format!("the pair {lo}-{hi} was solved and written but is not in {store}")),
            }
        }
    };
    Ok(Some(Oriented { entry, flipped: a > b }))
}

/// `_gen_mask`: the smallest all-ones mask at least `max`.
fn gen_mask(max: u64) -> u64 {
    let mut mask = max;
    for shift in [1, 2, 4, 8, 16, 32] {
        mask |= mask >> shift;
    }
    mask
}

/// numpy's `Generator.choice(n, size=k, replace=False)` for a small `n` (Floyd's algorithm, then
/// the shuffle of the drawn `k`).
fn choice_without_replacement(rng: &mut Pcg64, n: u64, k: u64) -> Vec<u64> {
    let mask = gen_mask((1.2 * k as f64) as u64);
    let mut hash = vec![u64::MAX; (mask + 1) as usize];
    let mut idx = vec![0u64; k as usize];
    for j in (n - k)..n {
        let val = rng.integers(j + 1);
        let mut loc = val & mask;
        while hash[loc as usize] != u64::MAX && hash[loc as usize] != val {
            loc = (loc + 1) & mask;
        }
        if hash[loc as usize] == u64::MAX {
            hash[loc as usize] = val;
            idx[(j - n + k) as usize] = val;
        } else {
            loc = j & mask;
            while hash[loc as usize] != u64::MAX {
                loc = (loc + 1) & mask;
            }
            hash[loc as usize] = j;
            idx[(j - n + k) as usize] = j;
        }
    }
    for i in (1..k).rev() {
        let j = rng.integers(i + 1);
        idx.swap(i as usize, j as usize);
    }
    idx
}

/// `teams.pick_four_indices`: which `size` come, then which two of them lead (in drawn order).
fn pick_four(rng: &mut Pcg64, available: usize, size: usize) -> Vec<usize> {
    let chosen = choice_without_replacement(rng, available as u64, size as u64);
    let mut leads = choice_without_replacement(rng, size as u64, 2);
    leads.sort_unstable();
    let mut out: Vec<usize> = leads.iter().map(|&i| chosen[i as usize] as usize).collect();
    out.extend((0..size as u64).filter(|i| !leads.contains(i)).map(|i| chosen[i as usize] as usize));
    out
}

/// `poolplay.draw_side`: the side's four from its arm's mixture (`np.cumsum`, divided by its last,
/// `searchsorted(side="right")`), or uniformly without a solve.
fn draw_side(ctx: &Run, entry: Option<&Oriented>, side: usize, rng: &mut Pcg64, six: usize) -> Result<Vec<usize>, String> {
    let Some(entry) = entry else { return Ok(pick_four(rng, six, ctx.reg.picked_team_size)) };
    let (strategy, loss) = if side == 0 { entry.ours() } else { entry.theirs() };
    let mixed = mixture(strategy, loss, ctx.epsilon, ctx.temperature)?;
    let mut cumulative = Vec::with_capacity(mixed.len());
    let mut run = 0.0;
    for v in &mixed {
        run += v;
        cumulative.push(run);
    }
    let last = *cumulative.last().ok_or("an empty mixture")?;
    for c in &mut cumulative {
        *c /= last;
    }
    let u = rng.next_double();
    let at = cumulative.iter().filter(|c| **c <= u).count().min(cumulative.len() - 1);
    Ok(entry.entry.selections[at].clone())
}

fn play_one(ctx: &Run, index: u64, number: u64, stats: &mut Stats) -> Result<Option<String>, String> {
    let reg = ctx.reg;
    let which = (index % 2) as usize;
    let game_index = index / 2;
    let mut rng = Pcg64::seeded(&[ctx.seed, game_index]);
    // `pool.draw_pair`: a uniform pair, then a fair coin for the seats.
    let drawn_from = ctx.subset.unwrap_or(ctx.pool_pairs);
    if drawn_from.is_empty() {
        return Err("no pairs to draw from".into());
    }
    let k_drawn = rng.integers(drawn_from.len() as u64) as usize;
    let swap = rng.next_double() < 0.5;
    let (first, second) = drawn_from[k_drawn];
    let (a, b) = if swap { (second, first) } else { (first, second) };
    let k = match ctx.subset {
        None => k_drawn,
        Some(_) => ctx.pool_pairs.iter().position(|p| *p == (a.min(b), a.max(b))).ok_or("a pair outside the pool")?,
    };
    let mirror = a == b;
    // Which arm sits at each side.
    let at_side = if which == 0 { [0usize, 1] } else { [1, 0] };
    let entries = [entry_of(ctx, at_side[0], a, b, stats)?, entry_of(ctx, at_side[1], a, b, stats)?];
    let teams = [&ctx.teams[a], &ctx.teams[b]];
    let mut picks: Vec<Vec<usize>> = Vec::new();
    for side in 0..2 {
        let stream = if cfg!(feature = "ika458-board-control") { 0 } else { side as u64 };
        let mut side_rng = Pcg64::seeded(&[ctx.seed, game_index, 1, stream]);
        picks.push(draw_side(ctx, entries[side].as_ref(), side, &mut side_rng, teams[side].sets.len())?);
        let arm = at_side[side];
        *stats.selections[arm].entry(if entries[side].is_some() { "solved" } else { "uniform" }).or_insert(0) += 1;
    }
    // `bench_prior[s]` prices side s's bench and is read by side 1 - s: from the arm across's solve.
    let species = |side: usize| -> Vec<String> { teams[side].sets.iter().map(|s| to_id(&s.species)).collect() };
    let about = |side: usize| -> Result<Option<Prior>, String> {
        let Some(entry) = entries[1 - side].as_ref() else { return Ok(None) };
        let probabilities = if side == 0 {
            let (s, l) = entry.ours();
            mixture(s, l, ctx.epsilon, ctx.temperature)?
        } else {
            let (s, l) = entry.theirs();
            let w = entry.class_weight();
            mixture(s, l, ctx.epsilon, ctx.temperature)?.into_iter().map(|p| w * p).collect()
        };
        Ok(Some(Prior { selections: entry.entry.selections.clone(), probabilities, species: species(side) }))
    };
    let priors = [about(0)?, about(1)?];
    let mut chosen: [Vec<&SheetSet>; 2] = [Vec::new(), Vec::new()];
    for side in 0..2 {
        for i in &picks[side] {
            chosen[side].push(teams[side].sets.get(*i).ok_or("a pick outside the six")?);
        }
    }
    let start = start_position(reg, [&chosen[0], &chosen[1]], &mut rng)?;
    insert_game(number, Game { sheets: [teams[0].sets.clone(), teams[1].sets.clone()], priors, made: HashMap::new() });

    let mut record_leads = Vec::new();
    let mut leads = Vec::new();
    for side in 0..2 {
        let mut active: Vec<_> = start.sides[side].pokemon.iter().filter(|m| m.active_index.is_some()).collect();
        active.sort_by_key(|m| m.active_index);
        record_leads.push(json!(active.iter().map(|m| identity(m)).collect::<Vec<_>>()));
        let slots: BTreeSet<usize> = active.iter().map(|m| m.slot).collect();
        let mut shown: Vec<String> = shown_species(&start, side, &slots).into_iter().collect();
        shown.sort();
        leads.push(json!(shown));
    }

    // Each side's node: its arm's, with the widths of the two arms in their seats.
    let widths = json!([ctx.arms[at_side[0]].limit, ctx.arms[at_side[1]].limit]);
    let nodes: Vec<Value> = (0..2)
        .map(|side| {
            let mut node = ctx.arms[at_side[side]].node.clone();
            node["widths"] = widths.clone();
            node
        })
        .collect();
    // The context's servers: the leaves in the config's order, each side naming its arm's.
    let context = json!({
        "servers": ctx.servers,
        "leaf": [ctx.arms[at_side[0]].leaf, ctx.arms[at_side[1]].leaf],
        "hidden": { "game": number, "seen": [[], []], "leads": leads },
    });
    let play = json!({
        "kind": "playGame",
        "position": start.to_json(),
        "rng": rng.to_json_full(),
        "context": context,
        "nodes": nodes,
        "exactBudget": ctx.config["exactBudget"],
        "steps": ctx.max_turns * 2,
    });
    let played = crate::playgame::play_game(reg, ctx.encoder, &play);
    if let Some(fault) = played.get("fault") {
        return Err(format!("playGame: {fault}"));
    }
    if let Some(error) = played.get("serverError") {
        return Err(format!("inference failed: {error}"));
    }
    let mut notes: BTreeSet<String> = BTreeSet::new();
    let take_notes = |list: &Value, notes: &mut BTreeSet<String>| {
        for note in list.as_array().into_iter().flatten() {
            if let Some(text) = note.as_str() {
                notes.insert(text.to_string());
            }
        }
    };
    let mut decisions: Vec<Value> = Vec::new();
    let mut seconds = [0.0f64; 2];
    for event in played["events"].as_array().into_iter().flatten() {
        take_notes(&event["stepNotes"], &mut notes);
        if event["kind"] == "replacement" {
            stats.replacements += 1;
            decisions.push(event["decision"].clone());
            take_notes(&event["answer"]["notes"], &mut notes);
            continue;
        }
        stats.moves += 1;
        take_notes(&event["nodeNotes"], &mut notes);
        match event.get("sideSeconds") {
            // Two nodes: each side its own search.
            Some(split) => {
                stats.split += 1;
                seconds[0] += split[0].as_f64().unwrap_or(0.0);
                seconds[1] += split[1].as_f64().unwrap_or(0.0);
            }
            // One node for two agents: each would have spent all of it alone (`one_agent=False`).
            None => {
                let spent = event["seconds"].as_f64().unwrap_or(0.0);
                seconds[0] += spent;
                seconds[1] += spent;
            }
        }
        decisions.push(event["decision"].clone());
        take_notes(&event["advanceNotes"], &mut notes);
        if let Some(chain) = event.get("chain") {
            take_notes(&chain["notes"], &mut notes);
            for step in chain["nodes"].as_array().into_iter().flatten() {
                stats.selfswitches += 1;
                decisions.push(self_switch_decision(step));
            }
        }
    }
    let stop = played["stop"].as_str().unwrap_or_default();
    if stop == "handoff" {
        stats.handoffs += 1;
        stats.seats[which].2 += 1;
        eprintln!("  handoff in index {index}: {} (not written)", played["why"].as_str().unwrap_or_default());
        return Ok(None);
    }
    let finale = Position::from_json(&played["position"]);
    let end_reason = if stop == "unresolved" {
        "unresolved"
    } else if finale.ended {
        if finale.winner.is_some() { "wipeout" } else { "draw" }
    } else {
        "turn-cap"
    };
    let outcome = match (&finale.winner, finale.ended) {
        (Some(winner), true) => Some(if **winner == *finale.sides[0].id { 1.0 } else { 0.0 }),
        _ => None,
    };
    let Some(outcome) = outcome else {
        stats.seats[which].2 += 1;
        return Ok(None);
    };
    let turns = played["turns"].as_i64().unwrap_or(0);
    let tested_won = if which == 0 { outcome > 0.5 } else { outcome < 0.5 };
    stats.seats[which].0 += u64::from(tested_won);
    stats.seats[which].1 += 1;

    let team_json = |t: &Team, pick: &[usize]| -> Value { json!(pick.iter().map(|i| t.record_sets[*i].clone()).collect::<Vec<_>>()) };
    let own_team = team_json(teams[0], &picks[0]);
    let foe_team = team_json(teams[1], &picks[1]);
    let pool_text = ordered(&[
        ("id", ctx.pool_value["id"].clone()),
        ("sha256", ctx.pool_value["sha256"].clone()),
        ("pair", json!(k)),
        ("teams", json!([teams[0].id, teams[1].id])),
        ("mirror", json!(mirror)),
        ("picks", json!(picks)),
        ("epsilon", json!(ctx.epsilon)),
        ("temperature", json!(ctx.temperature)),
    ]);
    let template = &ctx.templates[which];
    let mut pairs: Vec<(&str, String)> = Vec::with_capacity(template.len());
    for (key, fixed) in template {
        let value = match key.as_str() {
            "pool" => {
                pairs.push((key.as_str(), pool_text.clone()));
                continue;
            }
            "ownTeam" => own_team.clone(),
            "foeTeam" => foe_team.clone(),
            "foeArchetype" => json!(if mirror { "mirror" } else { "pool" }),
            "outcome" => json!(outcome),
            "turns" => json!(turns),
            "searchSeconds" => json!(seconds),
            "foeSpecies" => json!(foe_team.as_array().map(|m| m.iter().map(|s| s["species"].clone()).collect::<Vec<_>>())),
            "ownSix" => json!(teams[0].species),
            "foeSix" => json!(teams[1].species),
            "ownPick" => json!(picks[0]),
            "foePick" => json!(picks[1]),
            "unmodelled" => json!(notes),
            "leads" => json!(record_leads),
            "finalPosition" => played["position"].clone(),
            "endReason" => json!(end_reason),
            "decisions" => Value::Array(std::mem::take(&mut decisions)),
            "gameIndex" => json!(game_index),
            _ => fixed.clone(),
        };
        pairs.push((key.as_str(), value.to_string()));
    }
    Ok(Some(ordered_raw(&pairs)))
}

/// The worker's last lines and its summary file, in the shape `tools/pool_match.py` wrote them.
fn report(ctx: &Run, stats: &Stats, label: &str, elapsed: f64, out: Option<&str>) -> Result<(), String> {
    let (mut wins, mut played) = (0u64, 0u64);
    let tested = &ctx.arms[0];
    let other = &ctx.arms[1];
    for which in 0..2 {
        let (seat_wins, seat_played, unfinished, seconds) = stats.seats[which];
        let rate = if seat_played > 0 { seat_wins as f64 / seat_played as f64 * 100.0 } else { f64::NAN };
        println!(
            "  {label} = side {which}: {seat_played} games, won {rate:.1}% ({:.2} s/game, unfinished {unfinished})",
            seconds / seat_played.max(1) as f64
        );
        if let Some(path) = out {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            let line = ordered(&[
                ("seat", json!(format!("{label} = side {which}"))),
                ("seed", json!(ctx.seed)),
                ("model", json!(tested.name)),
                ("baseline", json!(other.name)),
                ("pool", ctx.pool_value["id"].clone()),
                ("limit", json!(tested.limit)),
                ("baselineLimit", json!(other.limit)),
                ("played", json!(seat_played)),
                ("wins", json!(seat_wins)),
                ("unfinished", json!(unfinished)),
                ("seconds", json!(seconds)),
            ]);
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("{path}: {e}"))?;
            file.write_all(line.as_bytes()).and_then(|()| file.write_all(b"\n")).map_err(|e| format!("{path}: {e}"))?;
        }
        wins += seat_wins;
        played += seat_played;
    }
    if played > 0 {
        println!("\n  both seats, {played} games: {label} won {:.1}%", wins as f64 / played as f64 * 100.0);
    }
    for (k, arm) in ctx.arms.iter().enumerate() {
        eprintln!(
            "  echo, {} {}: selection {:?}",
            if k == 0 { "tested" } else { "other" },
            arm.name,
            stats.selections[k]
        );
    }
    eprintln!(
        "  rust pool-match worker: {elapsed:.1}s; move nodes {} ({} with each side's own node), replacements {}, \
         self-switches {}, handoffs {} (not written); selection: {} solved here ({:.1}s), {} read, {} reused; \
         requests to other executables {}",
        stats.moves,
        stats.split,
        stats.replacements,
        stats.selfswitches,
        stats.handoffs,
        stats.solves,
        stats.solve_seconds,
        stats.loaded,
        stats.reused,
        crate::armexe::ASKED.load(std::sync::atomic::Ordering::Relaxed),
    );
    Ok(())
}
