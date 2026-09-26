//! Recorded games -> the training arrays, for `tools/encode_dataset.py` (IKA-347).
//!
//! The loop `encode_dir` ran in Python, moved here: read each JSONL line, keep the finished
//! games, turn every decision's position into the encoder's arrays and the per-decision
//! columns beside them. Python still owns everything around it -- which files, the cache,
//! the meta, the npz -- and reads what this writes straight into numpy arrays.
//!
//! Where the Python loop spent its time on `data/selfplay-mc0` (40,000 games, 4.6 GB) was
//! `json.loads` of whole records -- most of whose bytes are policies and selection mixtures
//! nothing here reads -- `Position.from_json`, and the encoder. Here a record is read into a
//! typed struct that skips the fields it does not name, only the position becomes a
//! `Value`, and the files are split across threads (`--jobs`).
//!
//! Three things are done the way Python did them, because the arrays must be the same
//! bytes:
//!
//! - **floats are read from their text.** serde_json's default float parser is not
//!   correctly rounded, and `searchValue` goes f64 -> f32; the raw text through
//!   `str::parse::<f64>` is what Python's `float()` gives.
//! - **absent is not null.** `record.get("information", "open")` is "open" when the key is
//!   missing and "None" when it is null, so the meta fields travel as their raw JSON text
//!   (or as absent) and Python applies its own `.get` and `str()` to the few distinct ones.
//! - **order is first appearance.** Game numbers, opponent labels, meta keys and the unknown
//!   volatiles are numbered or listed in the order the sequential loop met them: each file
//!   is read on its own and the files are joined in their sorted order.
//!
//! A line that does not parse is skipped only when it is the last line of its file (a run
//! still writing leaves a partial one, which is what Python's `except JSONDecodeError`
//! was for). Anywhere else it stops the run: Python would have read a `NaN` that serde
//! refuses, and a silently skipped game would be a different dataset.
//!
//!     pokeuraou-damage encode-games <regulation.json> [--jobs N] [--limit N]
//!         [--kind K ...] -- <file.jsonl> ...
//!
//! Output on stdout: one JSON header line, then the arrays as raw little-endian buffers in
//! the order the header's `arrays` lists them.

use crate::encode::{Encoder, VOLATILES};
use crate::objective::hp_share;
use crate::position::Position;
use crate::reg::Reg;
use serde::de::IgnoredAny;
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// A key that is present, even as `null`, is `Some(raw)`; only a missing key is `None`.
fn present<'de, D: Deserializer<'de>>(d: D) -> Result<Option<&'de RawValue>, D::Error> {
    <&RawValue>::deserialize(d).map(Some)
}

#[derive(Deserialize)]
struct GameLine<'a> {
    #[serde(borrow, default, deserialize_with = "present")]
    outcome: Option<&'a RawValue>,
    #[serde(borrow, default)]
    decisions: Option<Vec<DecisionLine<'a>>>,
    #[serde(borrow, default, deserialize_with = "present")]
    provenance: Option<&'a RawValue>,
    #[serde(borrow, default, deserialize_with = "present", rename = "searchLimit")]
    search_limit: Option<&'a RawValue>,
    #[serde(borrow, default, deserialize_with = "present", rename = "searchObjective")]
    search_objective: Option<&'a RawValue>,
    #[serde(borrow, default, deserialize_with = "present", rename = "selectionSource")]
    selection_source: Option<&'a RawValue>,
    #[serde(borrow, default, deserialize_with = "present")]
    information: Option<&'a RawValue>,
    #[serde(borrow, default, deserialize_with = "present")]
    engine: Option<&'a RawValue>,
    #[serde(borrow, default, deserialize_with = "present", rename = "foeArchetype")]
    foe_archetype: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct DecisionLine<'a> {
    #[serde(borrow)]
    turn: &'a RawValue,
    kind: Value,
    position: Value,
    #[serde(rename = "ownActions")]
    own_actions: Vec<IgnoredAny>,
    #[serde(borrow, rename = "searchValue")]
    search_value: &'a RawValue,
    #[serde(borrow, default, rename = "foeSearchValue")]
    foe_search_value: Option<&'a RawValue>,
}

/// The meta fields, in the order the header lists them.
const META_FIELDS: [&str; 6] =
    ["provenance", "searchLimit", "searchObjective", "selectionSource", "information", "engine"];

/// Counts keyed by a value, listed in the order the values were first met.
#[derive(Default)]
struct Ordered<K: std::hash::Hash + Eq + Clone> {
    index: HashMap<K, usize>,
    items: Vec<(K, u64)>,
}

impl<K: std::hash::Hash + Eq + Clone> Ordered<K> {
    fn add(&mut self, key: K, count: u64) -> usize {
        match self.index.get(&key) {
            Some(&at) => {
                self.items[at].1 += count;
                at
            }
            None => {
                self.index.insert(key.clone(), self.items.len());
                self.items.push((key, count));
                self.items.len() - 1
            }
        }
    }
}

/// One file's worth: the encoder's arrays and the columns beside them, with game numbers
/// and opponent indices local to the file (renumbered when the files are joined).
#[derive(Default)]
struct Part {
    games: usize,
    species: Vec<i32>,
    ability: Vec<i32>,
    item: Vec<i32>,
    moves: Vec<i32>,
    mon: Vec<f32>,
    mask: Vec<f32>,
    side: Vec<f32>,
    field: Vec<f32>,
    outcome: Vec<f32>,
    game: Vec<i32>,
    turn: Vec<i16>,
    search_value: Vec<f32>,
    hp_share: Vec<f32>,
    kind: Vec<i8>,
    foe: Vec<i32>,
    foe_search_value: Vec<f32>,
    /// Opponent labels as raw JSON text (`None`: the key was missing).
    foes: Ordered<Option<String>>,
    meta: [Ordered<Option<String>>; 6],
    branching: Ordered<usize>,
    unknown: Ordered<String>,
}

fn raw_f64(raw: &RawValue, what: &str, file: &str, line: usize) -> Result<f64, String> {
    raw.get()
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("{file}:{line}: {what} is {} and not a number", raw.get()))
}

/// `(record.get("provenance") or {}).get("kind", "selfplay")`, for `--kind`.
fn provenance_kind(raw: Option<&RawValue>, file: &str, line: usize) -> Result<Value, String> {
    let fallback = Value::String("selfplay".to_string());
    let Some(raw) = raw else { return Ok(fallback) };
    let value: Value = serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
    match value {
        Value::Null => Ok(fallback),
        Value::Object(map) if map.is_empty() => Ok(fallback),
        Value::Object(map) => Ok(map.get("kind").cloned().unwrap_or(fallback)),
        other => Err(format!("{file}:{line}: provenance is {other}, not an object")),
    }
}

struct Job<'a> {
    reg: &'a Reg,
    kinds: &'a [String],
    /// Games to keep in this file at most (the sequential `--limit` road), else unbounded.
    limit: Option<usize>,
}

fn encode_file(job: &Job, path: &str) -> Result<Part, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let encoder = Encoder::new(job.reg);
    let m = encoder.widths.mons_per_side;
    let mut part = Part::default();
    let lines: Vec<&[u8]> = bytes.split(|b| *b == b'\n').collect();
    let last = lines.iter().rposition(|l| !l.iter().all(u8::is_ascii_whitespace));
    for (number, line) in lines.iter().enumerate() {
        if job.limit.is_some_and(|limit| part.games >= limit) {
            break;
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let record: GameLine = match serde_json::from_slice(line) {
            Ok(record) => record,
            Err(error) if Some(number) == last => {
                eprintln!("{path}:{}: skipping a partial last line ({error})", number + 1);
                continue;
            }
            Err(error) => return Err(format!("{path}:{}: {error}", number + 1)),
        };
        let at = number + 1;
        let outcome = match record.outcome {
            None => continue,
            Some(raw) if raw.get().trim() == "null" => continue,
            Some(raw) => raw_f64(raw, "outcome", path, at)?,
        };
        if !job.kinds.is_empty() {
            let kind = provenance_kind(record.provenance, path, at)?;
            if !job.kinds.iter().any(|k| kind.as_str() == Some(k.as_str())) {
                continue;
            }
        }
        let raws = [
            record.provenance,
            record.search_limit,
            record.search_objective,
            record.selection_source,
            record.information,
            record.engine,
        ];
        for (counter, raw) in part.meta.iter_mut().zip(raws) {
            counter.add(raw.map(|r| r.get().to_string()), 1);
        }
        let foe = part.foes.add(record.foe_archetype.map(|r| r.get().to_string()), 0) as i32;
        let decisions = record
            .decisions
            .ok_or_else(|| format!("{path}:{at}: a finished game without decisions"))?;
        let game = part.games as i32;
        let mut positions = Vec::with_capacity(decisions.len());
        let mut met: Ordered<String> = Ordered::default();
        for decision in &decisions {
            let position = Position::from_json(&decision.position);
            part.outcome.push(outcome as f32);
            part.game.push(game);
            let turn = raw_f64(decision.turn, "turn", path, at)?;
            if turn.fract() != 0.0 || !(i16::MIN as f64..=i16::MAX as f64).contains(&turn) {
                return Err(format!("{path}:{at}: turn {turn} does not fit int16"));
            }
            part.turn.push(turn as i16);
            part.search_value.push(raw_f64(decision.search_value, "searchValue", path, at)? as f32);
            part.foe_search_value.push(match decision.foe_search_value {
                None => f32::NAN,
                Some(raw) if raw.get().trim() == "null" => f32::NAN,
                Some(raw) => raw_f64(raw, "foeSearchValue", path, at)? as f32,
            });
            part.hp_share.push(hp_share(&position) as f32);
            part.kind.push(if decision.kind.as_str() == Some("move") { 0 } else { 1 });
            part.foe.push(foe);
            part.branching.add(decision.own_actions.len(), 1);
            // The unknown volatiles in the order `encode_mon` meets them, which is the order
            // Python's counter lists them in. The encoder's own map (a HashMap) has the counts.
            for one_side in &position.sides {
                for mon in one_side.pokemon.iter().take(m) {
                    for volatile in &mon.volatiles {
                        if !VOLATILES.contains(&volatile.id.as_str()) {
                            met.add(volatile.id.as_str().to_string(), 1);
                        }
                    }
                    for vid in &mon.unmodelled_volatiles {
                        met.add(vid.as_str().to_string(), 1);
                    }
                }
            }
            positions.push(position);
        }
        let borrowed: Vec<&Position> = positions.iter().collect();
        let encoded = encoder.encode_positions(&borrowed);
        // The order is read off the positions above; the counts must still be the encoder's.
        let walked: HashMap<String, usize> =
            met.items.iter().map(|(k, n)| (k.clone(), *n as usize)).collect();
        if walked != encoded.unknown_volatiles {
            return Err(format!(
                "{path}:{at}: unknown volatiles {walked:?} walked, {:?} encoded",
                encoded.unknown_volatiles
            ));
        }
        for (id, count) in met.items {
            part.unknown.add(id, count);
        }
        #[cfg(feature = "ika347-control")]
        let encoded = {
            // IKA-347's positive control: the turn feature written one slot along.
            let mut encoded = encoded;
            let width = encoder.widths.field;
            for row in encoded.field.chunks_mut(width) {
                row.swap(width - 2, width - 3);
            }
            encoded
        };
        part.species.extend_from_slice(&encoded.species);
        part.ability.extend_from_slice(&encoded.ability);
        part.item.extend_from_slice(&encoded.item);
        part.moves.extend_from_slice(&encoded.moves);
        part.mon.extend_from_slice(&encoded.mon);
        part.mask.extend_from_slice(&encoded.mask);
        part.side.extend_from_slice(&encoded.side);
        part.field.extend_from_slice(&encoded.field);
        part.games += 1;
    }
    Ok(part)
}

fn write_all<T: Copy, const N: usize>(
    out: &mut impl Write,
    values: &[T],
    bytes: fn(T) -> [u8; N],
) -> std::io::Result<()> {
    let mut buffer = Vec::with_capacity(values.len().min(1 << 16) * N);
    for chunk in values.chunks(1 << 16) {
        buffer.clear();
        for value in chunk {
            buffer.extend_from_slice(&bytes(*value));
        }
        out.write_all(&buffer)?;
    }
    Ok(())
}

pub fn main(args: &[String]) {
    let usage = "encode-games <regulation.json> [--jobs N] [--limit N] [--kind K ...] -- <file.jsonl> ...";
    let Some(regulation) = args.first() else {
        eprintln!("usage: {usage}");
        std::process::exit(2);
    };
    let mut jobs = 1usize;
    let mut limit = 0usize;
    let mut kinds: Vec<String> = Vec::new();
    let mut files: Vec<String> = Vec::new();
    let mut at = 1;
    while at < args.len() {
        let value = || -> String {
            args.get(at + 1).cloned().unwrap_or_else(|| {
                eprintln!("{} takes a value; usage: {usage}", args[at]);
                std::process::exit(2);
            })
        };
        match args[at].as_str() {
            "--jobs" => {
                jobs = value().parse().unwrap_or(0).max(1);
                at += 2;
            }
            "--limit" => {
                limit = value().parse().unwrap_or(0);
                at += 2;
            }
            "--kind" => {
                kinds.push(value());
                at += 2;
            }
            "--" => {
                files.extend(args[at + 1..].iter().cloned());
                break;
            }
            other => {
                eprintln!("unknown argument {other}; usage: {usage}");
                std::process::exit(2);
            }
        }
    }
    let reg = Reg::load(regulation).unwrap_or_else(|e| {
        eprintln!("regulation: {e}");
        std::process::exit(1);
    });
    let started = std::time::Instant::now();
    let parts = if limit > 0 {
        // `--limit` is a debugging road, read in order and stopped early.
        let mut parts = Vec::new();
        let mut kept = 0;
        for file in &files {
            if kept >= limit {
                break;
            }
            let job = Job { reg: &reg, kinds: &kinds, limit: Some(limit - kept) };
            let part = encode_file(&job, file).unwrap_or_else(|e| fail(&e));
            kept += part.games;
            parts.push(part);
        }
        parts
    } else {
        let job = Job { reg: &reg, kinds: &kinds, limit: None };
        let next = AtomicUsize::new(0);
        let slots: Vec<Mutex<Option<Result<Part, String>>>> =
            files.iter().map(|_| Mutex::new(None)).collect();
        std::thread::scope(|scope| {
            for _ in 0..jobs.min(files.len().max(1)) {
                scope.spawn(|| loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= files.len() {
                        break;
                    }
                    let result = encode_file(&job, &files[index]);
                    *slots[index].lock().unwrap() = Some(result);
                });
            }
        });
        slots
            .into_iter()
            .map(|slot| slot.into_inner().unwrap().expect("every file taken").unwrap_or_else(|e| fail(&e)))
            .collect()
    };

    // Join: game numbers offset by the games before, opponents renumbered by first
    // appearance, counters summed in file order.
    let mut foes: Ordered<Option<String>> = Ordered::default();
    let mut meta: [Ordered<Option<String>>; 6] = Default::default();
    let mut branching: Ordered<usize> = Ordered::default();
    let mut unknown: Ordered<String> = Ordered::default();
    let mut remaps: Vec<Vec<i32>> = Vec::new();
    for part in &parts {
        remaps.push(part.foes.items.iter().map(|(k, _)| foes.add(k.clone(), 0) as i32).collect());
        for (into, from) in meta.iter_mut().zip(&part.meta) {
            for (k, n) in &from.items {
                into.add(k.clone(), *n);
            }
        }
        for (k, n) in &part.branching.items {
            branching.add(*k, *n);
        }
        for (k, n) in &part.unknown.items {
            unknown.add(k.clone(), *n);
        }
    }
    let games: usize = parts.iter().map(|p| p.games).sum();
    let decisions: usize = parts.iter().map(|p| p.outcome.len()).sum();
    let seconds = started.elapsed().as_secs_f64();

    let encoder = Encoder::new(&reg);
    let header = json!({
        "games": games,
        "decisions": decisions,
        "monsPerSide": encoder.widths.mons_per_side,
        "monWidth": encoder.widths.mon,
        "sideWidth": encoder.widths.side,
        "fieldWidth": encoder.widths.field,
        "foeLabels": foes.items.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
        "meta": META_FIELDS
            .iter()
            .zip(&meta)
            .map(|(name, c)| (name.to_string(), json!(c.items)))
            .collect::<serde_json::Map<String, Value>>(),
        "branching": branching.items,
        "unknownVolatiles": unknown.items,
        "seconds": seconds,
        "control": cfg!(feature = "ika347-control"),
        // Python's encoder hands the indices over as int64, so they cross as int64.
        "arrays": [["species", "<i8"], ["ability", "<i8"], ["item", "<i8"],
                   ["moves", "<i8"], ["mon", "<f4"], ["mask", "<f4"], ["side", "<f4"],
                   ["field", "<f4"], ["outcome", "<f4"], ["game", "<i4"], ["turn", "<i2"],
                   ["search_value", "<f4"], ["hp_share", "<f4"], ["kind", "i1"],
                   ["foe", "<i4"], ["foe_search_value", "<f4"]],
    });
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(1 << 20, stdout.lock());
    let result = (|| -> std::io::Result<()> {
        writeln!(out, "{header}")?;
        for p in &parts {
            write_all(&mut out, &p.species, wide)?;
        }
        for p in &parts {
            write_all(&mut out, &p.ability, wide)?;
        }
        for p in &parts {
            write_all(&mut out, &p.item, wide)?;
        }
        for p in &parts {
            write_all(&mut out, &p.moves, wide)?;
        }
        for p in &parts {
            write_all(&mut out, &p.mon, f32::to_le_bytes)?;
        }
        for p in &parts {
            write_all(&mut out, &p.mask, f32::to_le_bytes)?;
        }
        for p in &parts {
            write_all(&mut out, &p.side, f32::to_le_bytes)?;
        }
        for p in &parts {
            write_all(&mut out, &p.field, f32::to_le_bytes)?;
        }
        for p in &parts {
            write_all(&mut out, &p.outcome, f32::to_le_bytes)?;
        }
        let mut offset = 0i32;
        for p in &parts {
            let shifted: Vec<i32> = p.game.iter().map(|g| g + offset).collect();
            write_all(&mut out, &shifted, i32::to_le_bytes)?;
            offset += p.games as i32;
        }
        for p in &parts {
            write_all(&mut out, &p.turn, i16::to_le_bytes)?;
        }
        for p in &parts {
            write_all(&mut out, &p.search_value, f32::to_le_bytes)?;
        }
        for p in &parts {
            write_all(&mut out, &p.hp_share, f32::to_le_bytes)?;
        }
        for p in &parts {
            write_all(&mut out, &p.kind, i8::to_le_bytes)?;
        }
        for (p, remap) in parts.iter().zip(&remaps) {
            let mapped: Vec<i32> = p.foe.iter().map(|f| remap[*f as usize]).collect();
            write_all(&mut out, &mapped, i32::to_le_bytes)?;
        }
        for p in &parts {
            write_all(&mut out, &p.foe_search_value, f32::to_le_bytes)?;
        }
        out.flush()
    })();
    if let Err(error) = result {
        fail(&format!("writing the arrays: {error}"));
    }
}

fn wide(value: i32) -> [u8; 8] {
    i64::from(value).to_le_bytes()
}

fn fail(message: &str) -> ! {
    eprintln!("encode-games: {message}");
    std::process::exit(1);
}
