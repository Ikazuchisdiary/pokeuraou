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
//!
//! IKA-463: the same fill (`fill`) answers the `selectionSolve` request, which a person's game,
//! the analysis mode and the tools ask through `selection.solve_selection`: every spread class's
//! matrix, scored on the caller's inference server, and the Bayesian game solved here.

use crate::encode::{EncodeRules, Encoder};
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
fn score(
    encoder: &Encoder,
    server: &Server,
    rules: EncodeRules,
    positions: &[&Position],
    stats: &mut Stats,
) -> Result<Vec<f64>, served::Failed> {
    let mut out = Vec::with_capacity(positions.len());
    // Encoded a chunk at a time, as `RemoteValue.__call__` does, so one batch is never all in memory.
    for chunk in positions.chunks(server.batch.max(1)) {
        let encoded = encoder.encode_positions_with(chunk, rules);
        let rows = [chunk.len()];
        let batch = served::begin(server, &encoder.widths, &[&encoded], &rows, stats)?;
        out.extend(served::finish(server, batch, stats)?);
    }
    Ok(out)
}

/// Why a selection game has no matrices: a rule the port refused, or the server's failure.
pub(crate) enum FillFailed {
    Refused(String),
    Server(served::Failed),
}

/// The 90 x 90 matrices of a selection game (`selection.solve_selection`'s fill): per spread
/// class of the other side's six, every opening's branches scored in one batch and a cell the
/// weighted sum of its branches, row-major; then the antisymmetry check of the first class (our
/// first selection against its first eight, and their mirrors). The matrices, the check, and
/// the positions scored.
pub(crate) fn fill(
    reg: &Reg,
    encoder: &Encoder,
    server: &Server,
    rules: EncodeRules,
    row: &[SheetSet],
    classes: &[&[SheetSet]],
    stats: &mut Stats,
) -> Result<(Vec<Vec<f64>>, f64, usize), FillFailed> {
    let refused = FillFailed::Refused;
    let Some(first) = classes.first() else { return Err(refused("no spread classes to solve over".into())) };
    if classes.iter().any(|c| c.len() != row.len()) {
        return Err(refused("the two sheets are not the same size".into()));
    }
    let selections = all_selections(row.len(), reg.picked_team_size);
    let n = selections.len();
    fn pick<'s>(sets: &'s [SheetSet], sel: &[usize]) -> Vec<&'s SheetSet> {
        sel.iter().map(|i| &sets[*i]).collect()
    }
    let mut matrices = Vec::with_capacity(classes.len());
    let mut evaluated = 0usize;
    for col in classes {
        // Every opening's branches, in (ours, theirs) order, flattened as `solve_selection` does.
        let mut positions: Vec<Position> = Vec::new();
        let mut owner: Vec<(usize, f64)> = Vec::new();
        let mut branched = false;
        for (i, ours) in selections.iter().enumerate() {
            let mine = pick(row, ours);
            for (j, theirs) in selections.iter().enumerate() {
                let other = pick(col, theirs);
                let opening = crate::prep::opening(reg, [&mine, &other]).map_err(refused)?;
                let branches = crate::resolve::commands::lead_branches(reg, &opening).map_err(refused)?;
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
        let values = score(encoder, server, rules, &refs, stats).map_err(FillFailed::Server)?;
        let mut matrix = vec![0.0f64; n * n];
        if !branched {
            matrix.copy_from_slice(&values);
        } else {
            // `np.bincount(owner, weights=weights * values)`: each cell 0.0 plus its branches in order.
            for ((cell, weight), value) in owner.iter().zip(&values) {
                matrix[*cell] += weight * value;
            }
        }
        evaluated += positions.len();
        matrices.push(matrix);
    }

    // The antisymmetry check: our first selection against the first eight of theirs, each with
    // the leads' draws taken as their first option (`position_from_sets` without a generator).
    let mut sample = Vec::new();
    for theirs in selections.iter().take(8) {
        let opening = crate::prep::opening(reg, [&pick(row, &selections[0]), &pick(first, theirs)]).map_err(refused)?;
        sample.push(crate::resolve::commands::lead_phase(reg, opening, |_| Ok(0)).map_err(refused)?);
    }
    let mirrored: Vec<Position> = sample.iter().map(swapped).collect();
    let forward = score(encoder, server, rules, &sample.iter().collect::<Vec<_>>(), stats).map_err(FillFailed::Server)?;
    let backward = score(encoder, server, rules, &mirrored.iter().collect::<Vec<_>>(), stats).map_err(FillFailed::Server)?;
    let antisymmetry = forward.iter().zip(&backward).map(|(f, b)| (f + b - 1.0).abs()).fold(0.0f64, f64::max);
    evaluated += sample.len() + mirrored.len();
    Ok((matrices, antisymmetry, evaluated))
}

/// `selectionSolve` (IKA-463): `selection.solve_selection` in one request. The matrices filled
/// and scored on the caller's inference server (`fill`), and the Bayesian game over the spread
/// classes solved as `beliefFold` solves side 0's game (`solve_games`).
///
/// { "kind": "selectionSolve", "ours": [set], "classes": [[set]], "server": {...},
///   "encoding": {"megaFromSlots": b}, "solve": [ { "side": 0, "weights": b64 } ] }
/// -> `beliefFold`'s answer (`matrices`, `answers`, `lps`) and `antisymmetry`, `evaluated`,
///    `served`; or `fault`, or `serverError`.
pub fn selection_solve(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    let fault = |e: String| json!({ "fault": e });
    let sets = |v: &Value| -> Vec<SheetSet> {
        v.as_array().map(|a| a.iter().map(crate::hiddenbench::sheet_set).collect()).unwrap_or_default()
    };
    let server = match Server::from_json(&value["server"]) {
        Ok(server) => server,
        Err(why) => return fault(why),
    };
    let rules = EncodeRules {
        mega_from_slots: value
            .get("encoding")
            .and_then(|e| e.get("megaFromSlots"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    let row = sets(&value["ours"]);
    let classes: Vec<Vec<SheetSet>> = value["classes"].as_array().map(|a| a.iter().map(sets).collect()).unwrap_or_default();
    let class_refs: Vec<&[SheetSet]> = classes.iter().map(Vec::as_slice).collect();
    let mut stats = Stats::default();
    let (matrices, antisymmetry, evaluated) = match fill(reg, encoder, &server, rules, &row, &class_refs, &mut stats) {
        Ok(filled) => filled,
        Err(FillFailed::Refused(why)) => return fault(format!("the selection game: {why}")),
        Err(FillFailed::Server(why)) => return crate::beliefnode::server_failed(why),
    };
    let n = all_selections(row.len(), reg.picked_team_size).len();
    // Side 0's game over the classes: the jobs are the other side's (`solve_asks`).
    let sides = vec![1i64; matrices.len()];
    let mut answer = match crate::beliefnode::solve_games(value, &matrices, &sides, n, n) {
        Ok(answer) => answer,
        Err(why) => return fault(why),
    };
    answer["kind"] = json!("selectionSolve");
    answer["antisymmetry"] = json!(antisymmetry);
    answer["evaluated"] = json!(evaluated);
    answer["served"] = json!({
        "requests": stats.requests,
        "rows": stats.rows,
        "waitUs": stats.wait_ns as f64 / 1e3,
        "copyUs": stats.copy_ns as f64 / 1e3,
    });
    answer
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
    let mut stats = Stats::default();
    let (mut matrices, antisymmetry, evaluated) =
        match fill(reg, encoder, server, EncodeRules::default(), row.sets, &[col.sets], &mut stats) {
            Ok(filled) => filled,
            Err(FillFailed::Refused(why)) => return Err(why),
            Err(FillFailed::Server(why)) => return Err(failed(why)),
        };
    let matrix = matrices.remove(0);

    #[cfg(feature = "lp-dump")]
    let _tag = crate::lpbench::tag("selection");
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
