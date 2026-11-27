//! IKA-458 stage 3.7: a pair's selection game solved by the Rust worker.
//!
//! `poolplay.SolvedSelections._solve` solved it in Python (`selection.solve_selection`, then
//! `book_entry`) when the store had no file for the pair: the 90 x 90 ordered selections, each
//! cell the leaf's value of the turn-1 position at the weights of the leads' draws, the matrix
//! solved as a one-class Bayesian game, and the entry written to `<store>/<lo>-<hi>.json` for
//! every worker to read. This does the same steps in the same order, so the leaf is asked the
//! same rows in the same requests (one batch of every opening's branches, cut at the server's
//! chunk; then the eight antisymmetry positions and their mirrors), and the file is the same
//! JSON `prep.rs` and `BookEntry.from_json` read.

use crate::encode::Encoder;
use crate::hiddenbench::SheetSet;
use crate::lp::{solve_bayesian, sum_f64};
use crate::position::Position;
use crate::reg::Reg;
use crate::served::{self, Server, Stats};
use serde_json::{json, Value};
use std::time::Instant;

/// `teams.all_selections(6, 4)`: every lead pair, then every back pair of the other four.
pub fn all_selections(team_size: usize, size: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for a in 0..team_size {
        for b in (a + 1)..team_size {
            let rest: Vec<usize> = (0..team_size).filter(|i| *i != a && *i != b).collect();
            // `itertools.combinations(rest, size - 2)`, here size - 2 == 2.
            assert_eq!(size, 4, "a selection of four");
            for x in 0..rest.len() {
                for y in (x + 1)..rest.len() {
                    out.push(vec![a, b, rest[x], rest[y]]);
                }
            }
        }
    }
    out
}

/// What the store needs of one team.
pub struct TeamOf<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub sets: &'a [SheetSet],
    /// `selection_book.set_to_json` of each set, for the entry's `classSets`.
    pub book_sets: &'a Value,
}

/// What a solve cost, for the worker's report.
pub struct Solve {
    pub seconds: f64,
    pub positions: usize,
    pub antisymmetry: f64,
}

/// `Position.swapped` through the JSON: the sides exchanged (ids and names kept to their seats)
/// and every effect's side-indexed `sourceSlot` flipped.
fn swapped(position: &Position) -> Position {
    fn flip(slot: &str) -> Option<String> {
        let b = slot.as_bytes();
        if b.len() >= 2 && (b[0] == b'0' || b[0] == b'1') && b[1].is_ascii_digit() {
            return Some(format!("{}{}", if b[0] == b'0' { '1' } else { '0' }, &slot[1..]));
        }
        if b.len() >= 3 && b[0] == b'p' && (b[1] == b'1' || b[1] == b'2') {
            return Some(format!("p{}{}", if b[1] == b'1' { '2' } else { '1' }, &slot[2..]));
        }
        None
    }
    fn walk(value: &mut Value) {
        match value {
            Value::Object(map) => {
                if let Some(Value::String(slot)) = map.get("sourceSlot") {
                    if let Some(flipped) = flip(slot) {
                        map.insert("sourceSlot".into(), Value::String(flipped));
                    }
                }
                for (_, v) in map.iter_mut() {
                    walk(v);
                }
            }
            Value::Array(items) => items.iter_mut().for_each(walk),
            _ => {}
        }
    }
    let mut text = position.to_json();
    if let Some(sides) = text["sides"].as_array_mut() {
        sides.swap(0, 1);
        let (id0, id1) = (sides[0]["id"].clone(), sides[1]["id"].clone());
        let (n0, n1) = (sides[0]["name"].clone(), sides[1]["name"].clone());
        sides[0]["id"] = id1;
        sides[1]["id"] = id0;
        sides[0]["name"] = n1;
        sides[1]["name"] = n0;
    }
    walk(&mut text);
    Position::from_json(&text)
}

/// The leaf's value of each position (side 0's win probability), one batch cut at the server's
/// chunk as `RemoteValue.__call__` cuts it.
fn score(encoder: &Encoder, server: &Server, positions: &[&Position]) -> Result<Vec<f64>, String> {
    let mut stats = Stats::default();
    let mut out = Vec::with_capacity(positions.len());
    // Encoded a chunk at a time, as `RemoteValue.__call__` does, so one batch is never all in memory.
    for chunk in positions.chunks(server.batch.max(1)) {
        let encoded = encoder.encode_positions(chunk);
        let rows = [chunk.len()];
        let batch = served::begin(server, &encoder.widths, &[&encoded], &rows, &mut stats).map_err(failed)?;
        out.extend(served::finish(server, batch, &mut stats).map_err(failed)?);
    }
    Ok(out)
}

fn failed(why: served::Failed) -> String {
    match why {
        served::Failed::Refused { error, oom, .. } => {
            format!("the inference server refused the selection's leaves{}: {error}", if oom { " (out of memory)" } else { "" })
        }
        served::Failed::Broken(error) => format!("the inference server: {error}"),
    }
}

/// `SolvedSelections._solve` and `_write`: the pair `lo`-`hi` (row player `lo`) solved and its
/// file written into `store` under `tag`.
#[allow(clippy::too_many_arguments)]
pub fn solve_pair(
    reg: &Reg,
    encoder: &Encoder,
    server: &Server,
    row: &TeamOf,
    col: &TeamOf,
    lo: usize,
    hi: usize,
    store: &str,
    tag: &str,
    model: &str,
    notes: &Value,
) -> Result<Solve, String> {
    let started = Instant::now();
    if row.sets.len() != col.sets.len() {
        return Err("the two sheets are not the same size".into());
    }
    let selections = all_selections(row.sets.len(), reg.picked_team_size);
    let n = selections.len();
    fn pick<'s>(sets: &'s [SheetSet], sel: &[usize]) -> Vec<&'s SheetSet> {
        sel.iter().map(|i| &sets[*i]).collect()
    }
    // Every opening's branches, in (ours, theirs) order, flattened as `solve_selection` does.
    let mut positions: Vec<Position> = Vec::new();
    let mut owner: Vec<(usize, f64)> = Vec::new();
    let mut branched = false;
    for (i, ours) in selections.iter().enumerate() {
        let mine = pick(row.sets, ours);
        for (j, theirs) in selections.iter().enumerate() {
            let other = pick(col.sets, theirs);
            let opening = crate::prep::opening(reg, [&mine, &other])?;
            let branches = crate::resolve::commands::lead_branches(reg, &opening)?;
            if branches.len() != 1 {
                branched = true;
            }
            for (weight, position) in branches {
                owner.push((i * n + j, weight));
                positions.push(position);
            }
        }
    }
    let refs: Vec<&Position> = positions.iter().collect();
    let values = score(encoder, server, &refs)?;
    let mut matrix = vec![0.0f64; n * n];
    if !branched {
        matrix.copy_from_slice(&values);
    } else {
        // `np.bincount(owner, weights=weights * values)`: each cell 0.0 plus its branches in order.
        for ((cell, weight), value) in owner.iter().zip(&values) {
            matrix[*cell] += weight * value;
        }
    }
    let evaluated_cells = positions.len();

    // The antisymmetry check: our first selection against the first eight of theirs, each with
    // the leads' draws taken as their first option (`position_from_sets` without a generator).
    let mut sample = Vec::new();
    for theirs in selections.iter().take(8) {
        let opening = crate::prep::opening(reg, [&pick(row.sets, &selections[0]), &pick(col.sets, theirs)])?;
        sample.push(crate::resolve::commands::lead_phase(reg, opening, |_| Ok(0))?);
    }
    let mirrored: Vec<Position> = sample.iter().map(swapped).collect();
    let forward = score(encoder, server, &sample.iter().collect::<Vec<_>>())?;
    let backward = score(encoder, server, &mirrored.iter().collect::<Vec<_>>())?;
    let antisymmetry = forward.iter().zip(&backward).map(|(f, b)| (f + b - 1.0).abs()).fold(0.0f64, f64::max);
    let evaluated = evaluated_cells + sample.len() + mirrored.len();

    let solved = solve_bayesian(&[(matrix.clone(), n, n)], &[1.0], 1e-9).map_err(|why| format!("the selection game: {why:?}"))?;
    // `assemble_bayesian` for one class of weight 1.
    let value = solved.value();
    let y = &solved.ys[0];
    let x = &solved.x;
    let row_ev: Vec<f64> = (0..n).map(|i| 0.0 + 1.0 * sum_f64(&(0..n).map(|j| matrix[i * n + j] * y[j]).collect::<Vec<_>>())).collect();
    let row_ev_loss: Vec<f64> = row_ev.iter().map(|e| (value - e).max(0.0)).collect();
    let per_action: Vec<f64> = (0..n).map(|j| sum_f64(&(0..n).map(|i| x[i] * matrix[i * n + j]).collect::<Vec<_>>())).collect();
    let low = per_action.iter().copied().fold(f64::INFINITY, f64::min);
    let col_ev_loss: Vec<f64> = per_action.iter().map(|p| (p - low).max(0.0)).collect();
    let (their_strategy, their_loss) = if lo == hi { (x.clone(), row_ev_loss.clone()) } else { (y.clone(), col_ev_loss) };
    let seconds = started.elapsed().as_secs_f64();
    let entry = json!({
        "key": format!("{}|{}", row.id, col.id),
        "player": col.name,
        "place": 0,
        "selections": selections,
        "ourStrategy": x,
        "ourEvLoss": row_ev_loss,
        "classWeights": [1.0],
        "classSets": [col.book_sets],
        "theirStrategies": [their_strategy],
        "theirEvLoss": [their_loss],
        "value": value,
        "dualityGap": (solved.value_row - solved.value_col).abs(),
        "antisymmetryError": antisymmetry,
        "seconds": seconds,
        "model": model,
        "notes": notes,
    });
    let took = started.elapsed().as_secs_f64();
    let payload = json!({
        "tag": tag,
        "pair": [lo, hi],
        "teams": [row.id, col.id],
        "seconds": took,
        "entry": entry,
    });
    write(store, lo, hi, &payload)?;
    Ok(Solve { seconds: took, positions: evaluated, antisymmetry })
}

/// `SolvedSelections._write`: atomically, and another worker's file of the same pair kept.
fn write(store: &str, lo: usize, hi: usize, payload: &Value) -> Result<(), String> {
    let dir = std::path::Path::new(store);
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join(format!("{lo:03}-{hi:03}.json"));
    let temporary = dir.join(format!("{lo:03}-{hi:03}.json.{}.tmp", std::process::id()));
    std::fs::write(&temporary, payload.to_string().as_bytes()).map_err(|e| format!("{}: {e}", temporary.display()))?;
    match std::fs::rename(&temporary, &path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::PermissionDenied && path.exists() => {
            let _ = std::fs::remove_file(&temporary);
            Ok(())
        }
        Err(error) => Err(format!("{}: {error}", path.display())),
    }
}
