//! IKA-458 stage 3.7: the generation worker as this executable (`pokeuraou-damage selfplay-worker`).
//!
//! `tools/selfplay.py --pool` was a Python process that, for each game the queue handed it,
//! asked a port process to prepare the game (`prepGame`), solved a pair's selection game with
//! the leaf when the store had no file for it, asked the port to play the game (`playGame`),
//! wrote the steps into a `GameRecord` and the record as one JSON line, and told the queue the
//! game was done. This process does all of it itself, calling the same functions the port's
//! requests call, so nothing crosses a pipe and no Python runs per game.
//!
//! What is the same for every game of a run -- the pool and its sheets in the shapes the record
//! and the store write, the seed, the selection's exploration, the store and its tag, the static
//! parts of `playGame`'s request (the leaf's and the Q's servers, the widths, the budgets) and
//! the record's fields that do not depend on the game, in `GameRecord.to_json`'s order -- comes
//! from one JSON file `rustworker.py` writes once (`--config`). What differs per worker comes on
//! the command line: the queue, the inference server and the output file.
//!
//! ```text
//! pokeuraou-damage selfplay-worker --config C --inference HOST:PORT --out FILE
//!     (--queue HOST:PORT | --first-game K --games N) [--threads T]
//! ```
//!
//! A game the loop leaves to its caller (`playGame`'s `handoff`) has no caller here: it is not
//! written, the worker says so on stderr and counts it (1,200 generation games had none,
//! records/IKA-458.md section 16). A fault, a stop or a server error ends the worker with the
//! message, as `portlp.PortFault` ended the Python one; the queue gives its game to another.

use crate::encode::Encoder;
use crate::hiddenbench::{identity, sheet_set, shown_species, SheetSet};
use crate::lp::{f64s, f64s_text};
use crate::position::Position;
use crate::reg::Reg;
use crate::selection::{solve_pair, TeamOf};
use crate::served::Server;
use crate::shm::Shared;
use serde_json::{json, Value};
use std::collections::{BTreeSet, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Instant;

/// The keys of a record this worker writes per game; every other key of the template is copied.
const PER_GAME: &[&str] = &[
    "ownTeam", "foeTeam", "foeArchetype", "outcome", "turns", "searchSeconds", "foeSpecies", "ownSix",
    "foeSix", "ownPick", "foePick", "ownSelectionPolicy", "foeSelectionPolicy", "ownSelectionMixture",
    "foeSelectionMixture", "selectionValue", "unmodelled", "leads", "finalPosition", "endReason",
    "decisions", "pool", "gameIndex",
];

/// The pool's number in this process (`poolDefine`'s `id`).
const POOL: u64 = 1;

pub(crate) struct Team {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) sets: Vec<SheetSet>,
    pub(crate) species: Vec<String>,
    pub(crate) record_sets: Vec<Value>,
    pub(crate) book_sets: Value,
}

/// The config's pool's teams, in the shapes the record and the store write.
pub(crate) fn teams_of(pool: &Value) -> Result<Vec<Team>, String> {
    Ok(pool["teams"]
        .as_array()
        .ok_or("the config's pool has no teams")?
        .iter()
        .map(|t| Team {
            id: t["id"].as_str().unwrap_or_default().to_string(),
            name: t["name"].as_str().unwrap_or_default().to_string(),
            sets: t["sets"].as_array().map(|s| s.iter().map(sheet_set).collect()).unwrap_or_default(),
            species: t["sets"]
                .as_array()
                .map(|s| s.iter().map(|x| x["species"].as_str().unwrap_or_default().to_string()).collect())
                .unwrap_or_default(),
            record_sets: t["recordSets"].as_array().cloned().unwrap_or_default(),
            book_sets: t["bookSets"].clone(),
        })
        .collect())
}

pub(crate) struct Queue {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
}

impl Queue {
    pub(crate) fn connect(address: &str) -> Result<Queue, String> {
        let stream = TcpStream::connect(address).map_err(|e| format!("the queue at {address}: {e}"))?;
        let _ = stream.set_nodelay(true);
        let writer = stream.try_clone().map_err(|e| e.to_string())?;
        Ok(Queue { reader: BufReader::new(stream), writer })
    }

    /// `WorkClient.take`: the next game's number, or None when the queue is empty.
    pub(crate) fn take(&mut self) -> Result<Option<u64>, String> {
        self.writer.write_all(b"get\n").map_err(|e| e.to_string())?;
        let mut line = String::new();
        self.reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() || line == "done" {
            return Ok(None);
        }
        line.parse().map(Some).map_err(|_| format!("the queue sent {line:?}"))
    }

    /// `WorkClient.finish`.
    pub(crate) fn finish(&mut self, index: u64) -> Result<(), String> {
        self.writer.write_all(format!("ok {index}\n").as_bytes()).map_err(|e| e.to_string())
    }
}

#[derive(Default)]
struct Stats {
    games: u64,
    finished: u64,
    discarded: u64,
    handoffs: u64,
    wins: u64,
    decisions: u64,
    turns: i64,
    mirrors: u64,
    mirror_wins: u64,
    mirror_draws: u64,
    solves: u64,
    solved_here: HashSet<(usize, usize)>,
    loaded: u64,
    reused: u64,
    solve_seconds: f64,
    solve_positions: usize,
    worst_antisymmetry: f64,
    prep_seconds: f64,
    play_seconds: f64,
    write_seconds: f64,
    moves: u64,
    replacements: u64,
    selfswitches: u64,
}

pub(crate) fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|at| args.get(at + 1)).cloned()
}

pub fn main(args: &[String]) {
    if let Err(why) = run(args) {
        eprintln!("selfplay-worker: {why}");
        std::process::exit(1);
    }
}

/// `JSON` text of `value` with the keys in the order given, for a record that reads as Python's.
pub(crate) fn ordered(pairs: &[(&str, Value)]) -> String {
    let raw: Vec<(&str, String)> = pairs.iter().map(|(k, v)| (*k, v.to_string())).collect();
    ordered_raw(&raw)
}

/// `ordered` of values already written as JSON text.
pub(crate) fn ordered_raw(pairs: &[(&str, String)]) -> String {
    let mut out = String::from("{");
    for (k, (key, text)) in pairs.iter().enumerate() {
        if k > 0 {
            out.push_str(", ");
        }
        out.push_str(&Value::String((*key).to_string()).to_string());
        out.push_str(": ");
        out.push_str(text);
    }
    out.push('}');
    out
}

pub(crate) fn set_shm(server: &mut Value, address: &str, block: &Shared) {
    server["address"] = json!(address);
    server["shm"] = json!(block.name());
    server["bytes"] = json!(block.capacity());
}

fn run(args: &[String]) -> Result<(), String> {
    if let Some(threads) = arg(args, "--threads") {
        match threads.parse::<usize>() {
            Ok(t) if t >= 1 => crate::par::set_threads(t),
            _ => return Err("--threads takes a positive number".into()),
        }
    }
    let config_path = arg(args, "--config").ok_or("--config is required")?;
    let address = arg(args, "--inference").ok_or("--inference is required")?;
    let out_path = arg(args, "--out").ok_or("--out is required")?;
    let config: Value = serde_json::from_slice(
        &std::fs::read(&config_path).map_err(|e| format!("{config_path}: {e}"))?,
    )
    .map_err(|e| format!("{config_path}: {e}"))?;
    let reg = Reg::load(config["regulation"].as_str().ok_or("the config has no regulation")?)
        .map_err(|e| format!("regulation: {e}"))?;
    let encoder = Encoder::new(&reg);

    // The leaf's and the Q's blocks, this process's own (`RemoteValue._block`, `RemoteQ._block`).
    let pid = std::process::id();
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let value_bytes = config["value"]["bytes"].as_u64().ok_or("the config has no value.bytes")? as usize;
    let q_bytes = config["q"]["bytes"].as_u64().ok_or("the config has no q.bytes")? as usize;
    let value_block = Shared::create(&format!("pokeuraou_w{pid}_{stamp:x}_v"), value_bytes)
        .ok_or("could not create the leaf's shared block")?;
    let q_block = Shared::create(&format!("pokeuraou_w{pid}_{stamp:x}_q"), q_bytes)
        .ok_or("could not create the Q's shared block")?;
    let mut context = config["context"].clone();
    for server in context["servers"].as_array_mut().ok_or("the config's context has no servers")? {
        set_shm(server, &address, &value_block);
    }
    let mut node = config["node"].clone();
    set_shm(&mut node["server"], &address, &value_block);
    set_shm(&mut node["qServer"], &address, &q_block);
    let leaf_server = Server::from_json(&node["server"])?;

    // The pool, defined once (`poolDefine`) and kept in the shapes the record and the store write.
    let pool_value = &config["pool"];
    let teams = teams_of(pool_value)?;
    crate::prep::define_pool(&json!({ "id": POOL, "teams": pool_value["teams"], "pairs": pool_value["pairs"] }));
    let template: Vec<(String, Value)> = config["record"]
        .as_array()
        .ok_or("the config has no record template")?
        .iter()
        .map(|p| (p[0].as_str().unwrap_or_default().to_string(), p[1].clone()))
        .collect();
    let seed = config["seed"].as_u64().ok_or("the config has no seed")?;
    let max_turns = config["maxTurns"].as_u64().ok_or("the config has no maxTurns")?;
    let epsilon = config["epsilon"].as_f64().ok_or("the config has no epsilon")?;
    let temperature = config["temperature"].as_f64().ok_or("the config has no temperature")?;
    let store = config["store"].as_str().ok_or("the config has no store")?.to_string();
    let tag = config["tag"].as_str().ok_or("the config has no tag")?.to_string();
    let model = config["model"].as_str().unwrap_or_default().to_string();

    let mut queue = match arg(args, "--queue") {
        Some(at) => Some(Queue::connect(&at)?),
        None => None,
    };
    let first: u64 = arg(args, "--first-game").map(|v| v.parse().map_err(|_| "--first-game")).transpose()?.unwrap_or(0);
    let count: u64 = arg(args, "--games").map(|v| v.parse().map_err(|_| "--games")).transpose()?.unwrap_or(0);
    if queue.is_none() && count == 0 {
        return Err("give --queue, or --games (with --first-game)".into());
    }
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .map_err(|e| format!("{out_path}: {e}"))?;
    eprintln!(
        "rust selfplay worker (IKA-458 stage 3.7): pool {} ({} teams), seed {seed}, leaf {model} on {address}, store {store}",
        pool_value["id"].as_str().unwrap_or("?"),
        teams.len()
    );

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
        let line = play_one(
            &reg, &encoder, &leaf_server, &teams, pool_value, &config, &template, &context, &node, seed, index,
            number, max_turns, epsilon, temperature, &store, &tag, &model, &mut stats,
        )?;
        if let Some(line) = line {
            let began = Instant::now();
            out.write_all(line.as_bytes()).and_then(|()| out.write_all(b"\n")).and_then(|()| out.flush())
                .map_err(|e| format!("{out_path}: {e}"))?;
            stats.write_seconds += began.elapsed().as_secs_f64();
        }
        if let Some(q) = queue.as_mut() {
            q.finish(index)?;
        }
    }
    report(&stats, started.elapsed().as_secs_f64(), &out_path, &store);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn play_one(
    reg: &Reg,
    encoder: &Encoder,
    leaf_server: &Server,
    teams: &[Team],
    pool_value: &Value,
    config: &Value,
    template: &[(String, Value)],
    context: &Value,
    node: &Value,
    seed: u64,
    index: u64,
    number: u64,
    max_turns: u64,
    epsilon: f64,
    temperature: f64,
    store: &str,
    tag: &str,
    model: &str,
    stats: &mut Stats,
) -> Result<Option<String>, String> {
    let began = Instant::now();
    // The positive control of the comparison with the Python worker: the game after the one named.
    let seeded = if cfg!(feature = "ika458-worker-control") { index + 1 } else { index };
    let request = json!({
        "kind": "prepGame",
        "pool": POOL,
        "seed": seed,
        "index": seeded,
        "pairs": config["pairs"],
        "store": store,
        "tag": tag,
        "epsilon": f64s_text(&[epsilon]),
        "temperature": f64s_text(&[temperature]),
        "hideBench": true,
        "game": number,
    });
    let mut answer = crate::prep::prep_game(reg, &request);
    if let Some(pair) = answer.get("unsolved") {
        // The pair has no file yet: solved here with the leaf and written for every worker.
        let lo = pair[0].as_u64().ok_or("unsolved without a pair")? as usize;
        let hi = pair[1].as_u64().ok_or("unsolved without a pair")? as usize;
        let row = TeamOf { id: &teams[lo].id, name: &teams[lo].name, sets: &teams[lo].sets, book_sets: &teams[lo].book_sets };
        let col = TeamOf { id: &teams[hi].id, name: &teams[hi].name, sets: &teams[hi].sets, book_sets: &teams[hi].book_sets };
        let solved = solve_pair(reg, encoder, leaf_server, &row, &col, lo, hi, store, tag, model, &config["notes"])?;
        stats.solves += 1;
        stats.solved_here.insert((lo, hi));
        stats.solve_seconds += solved.seconds;
        stats.solve_positions += solved.positions;
        stats.worst_antisymmetry = stats.worst_antisymmetry.max(solved.antisymmetry);
        answer = crate::prep::prep_game(reg, &request);
        if answer.get("unsolved").is_some() {
            return Err(format!("the pair {lo}-{hi} was solved and written but prepGame found no file"));
        }
    }
    if let Some(stop) = answer.get("stop") {
        return Err(stop.as_str().unwrap_or_default().to_string());
    }
    if let Some(fault) = answer.get("fault") {
        return Err(format!("prepGame: {fault}"));
    }
    let a = answer["teams"][0].as_u64().ok_or("prepGame without teams")? as usize;
    let b = answer["teams"][1].as_u64().ok_or("prepGame without teams")? as usize;
    let key = (a.min(b), a.max(b));
    // What `SolvedSelections.canonical` counted: a file is loaded once, a solve made here is reused.
    if answer["read"].as_bool() == Some(true) && !stats.solved_here.contains(&key) {
        stats.loaded += 1;
    } else {
        stats.reused += 1;
    }
    let picks: Vec<Vec<usize>> = (0..2)
        .map(|side| {
            answer["picks"][side]
                .as_array()
                .map(|p| p.iter().filter_map(Value::as_u64).map(|v| v as usize).collect())
                .unwrap_or_default()
        })
        .collect();
    let start = Position::from_json(&answer["position"]);
    // The pair each side led with: identities in `active` order for the record, and the
    // revealed species (`shown_species` of the active slots) the bench belief conditions on.
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
    stats.prep_seconds += began.elapsed().as_secs_f64();

    let began = Instant::now();
    let mut context = context.clone();
    context["hidden"] = json!({ "game": number, "seen": [[], []], "leads": leads });
    let play = json!({
        "kind": "playGame",
        "position": answer["position"],
        "rng": answer["rng"],
        "context": context,
        "node": node,
        "exactBudget": config["exactBudget"],
        "steps": max_turns * 2,
    });
    let played = crate::playgame::play_game(reg, encoder, &play);
    stats.play_seconds += began.elapsed().as_secs_f64();
    if let Some(fault) = played.get("fault") {
        return Err(format!("playGame: {fault}"));
    }
    if let Some(error) = played.get("serverError") {
        return Err(format!("inference failed: {error}"));
    }
    let mut notes: BTreeSet<String> = BTreeSet::new();
    let mut take_notes = |list: &Value, notes: &mut BTreeSet<String>| {
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
        let spent = event["seconds"].as_f64().unwrap_or(0.0);
        seconds[0] += spent / 2.0;
        seconds[1] += spent / 2.0;
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
    stats.games += 1;
    let mirror = a == b;
    if mirror {
        stats.mirrors += 1;
    }
    if stop == "handoff" {
        stats.handoffs += 1;
        stats.discarded += 1;
        eprintln!("  handoff in game {index}: {} (not written)", played["why"].as_str().unwrap_or_default());
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
        stats.discarded += 1;
        return Ok(None);
    };
    let turns = played["turns"].as_i64().unwrap_or(0);
    stats.finished += 1;
    stats.wins += u64::from(outcome > 0.5);
    stats.decisions += decisions.len() as u64;
    stats.turns += turns;
    if mirror {
        stats.mirror_wins += u64::from(outcome > 0.5);
        stats.mirror_draws += u64::from(outcome == 0.5);
    }

    let began = Instant::now();
    let team = |t: usize, pick: &[usize]| -> Value { json!(pick.iter().map(|i| teams[t].record_sets[*i].clone()).collect::<Vec<_>>()) };
    let own_team = team(a, &picks[0]);
    let foe_team = team(b, &picks[1]);
    let floats = |key: &str| -> Result<Value, String> { Ok(json!(f64s(&answer[key])?)) };
    let pool_text = ordered(&[
        ("id", pool_value["id"].clone()),
        ("sha256", pool_value["sha256"].clone()),
        ("teams", json!([teams[a].id, teams[b].id])),
        ("names", json!([teams[a].name, teams[b].name])),
        ("pair", answer["pair"].clone()),
        ("mirror", json!(mirror)),
        ("benchPrior", json!(["solved", "solved"])),
    ]);
    let mut pairs: Vec<(&str, String)> = Vec::with_capacity(template.len());
    for (key, fixed) in template {
        if key == "pool" {
            // The pool object in its own key order (`generate_pool`'s).
            pairs.push((key.as_str(), pool_text.clone()));
            continue;
        }
        let value = match key.as_str() {
            "ownTeam" => own_team.clone(),
            "foeTeam" => foe_team.clone(),
            "foeArchetype" => json!(if mirror { "mirror" } else { "pool" }),
            "outcome" => json!(outcome),
            "turns" => json!(turns),
            "searchSeconds" => json!(seconds),
            "foeSpecies" => json!(foe_team.as_array().map(|m| m.iter().map(|s| s["species"].clone()).collect::<Vec<_>>())),
            "ownSix" => json!(teams[a].species),
            "foeSix" => json!(teams[b].species),
            "ownPick" => json!(picks[0]),
            "foePick" => json!(picks[1]),
            "ownSelectionPolicy" => floats("ownPolicy")?,
            "foeSelectionPolicy" => floats("foePolicy")?,
            "ownSelectionMixture" => floats("ownMixture")?,
            "foeSelectionMixture" => floats("foeMixture")?,
            "selectionValue" => answer["value"].clone(),
            "unmodelled" => json!(notes),
            "leads" => json!(record_leads),
            "finalPosition" => played["position"].clone(),
            "endReason" => json!(end_reason),
            "decisions" => Value::Array(std::mem::take(&mut decisions)),
            "gameIndex" => json!(index),
            _ => fixed.clone(),
        };
        pairs.push((key.as_str(), value.to_string()));
    }
    let missing: Vec<&&str> = PER_GAME.iter().filter(|k| !template.iter().any(|(t, _)| t == *k)).collect();
    if !missing.is_empty() {
        return Err(format!("the record template has no {missing:?}"));
    }
    let line = ordered_raw(&pairs);
    stats.write_seconds += began.elapsed().as_secs_f64();
    Ok(Some(line))
}

/// `_apply_chain`'s `Decision` of one self-switch node, as `GameRecord.to_json` writes it.
pub(crate) fn self_switch_decision(step: &Value) -> Value {
    let chooser = step["chooser"].as_u64().unwrap_or(0);
    let options: Vec<Value> = step["options"].as_array().cloned().unwrap_or_default();
    let best = step["best"].as_u64().unwrap_or(0) as usize;
    let mut policy = vec![0.0f64; options.len()];
    if best < policy.len() {
        policy[best] = 1.0;
    }
    let waiting = json!(["pass"]);
    let chosen = options.get(best).cloned().unwrap_or(Value::Null);
    let (own_actions, own_policy, foe_actions, foe_policy, own_chosen, foe_chosen) = if chooser == 0 {
        (json!(options), json!(policy), waiting.clone(), json!([1.0]), chosen, json!("pass"))
    } else {
        (waiting.clone(), json!([1.0]), json!(options), json!(policy), json!("pass"), chosen)
    };
    json!({
        "turn": step["turn"],
        "kind": "selfswitch",
        "position": step["position"],
        "ownActions": own_actions,
        "ownPolicy": own_policy,
        "foeActions": foe_actions,
        "foePolicy": foe_policy,
        "searchValue": step["value"].as_f64().unwrap_or(0.0),
        "ownChosen": own_chosen,
        "foeChosen": foe_chosen,
        "shownIdentities": step["shown"],
    })
}

/// The worker's last lines, in the shape `tools/selfplay.py` printed them, and its own clock.
fn report(stats: &Stats, elapsed: f64, out_path: &str, store: &str) {
    let finished = stats.finished.max(1) as f64;
    println!(
        "{} games in {elapsed:.1}s ({:.2}s each)\n  finished {}, discarded unfinished {}\n  seat-0 win rate {:.1}%\n  decisions {} ({:.1} per game), turns {:.1} per game\n  mirrors {} (wins {}, draws {})\n  -> {out_path}",
        stats.games,
        elapsed / stats.games.max(1) as f64,
        stats.finished,
        stats.discarded,
        stats.wins as f64 / finished * 100.0,
        stats.decisions,
        stats.decisions as f64 / finished,
        stats.turns as f64 / finished,
        stats.mirrors,
        stats.mirror_wins,
        stats.mirror_draws,
    );
    let solves = stats.solves.max(1) as f64;
    println!(
        "  selection: {} solves ({} reused, {} read from {store}), {:.1}s = {:.2}s a solve, {} positions, worst antisymmetry {:.2e}",
        stats.solves,
        stats.reused,
        stats.loaded,
        stats.solve_seconds,
        stats.solve_seconds / solves,
        stats.solve_positions,
        stats.worst_antisymmetry,
    );
    eprintln!(
        "  rust worker: move nodes {}, replacements {}, self-switches {}, handoffs {} (not written); \
         seconds: prepare {:.1}, solve {:.1}, play {:.1}, write {:.1}",
        stats.moves,
        stats.replacements,
        stats.selfswitches,
        stats.handoffs,
        stats.prep_seconds - stats.solve_seconds,
        stats.solve_seconds,
        stats.play_seconds,
        stats.write_seconds,
    );
}
