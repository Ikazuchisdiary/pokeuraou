//! IKA-457 stage 2.2: a replacement node's whole solve in one request.
//!
//! `selfplay._do_replacement_node` read the node of a replacement phase one completion at a
//! time: for each completion of the other side's bench (`spreads[1 - side]`, side 0's then side
//! 1's) the port filled every pair of replacements (`replacementsEncoded`, the phase's positions
//! encoded), Python scored the rows on the inference server (`RemoteValue.from_encoded`, which
//! settles the ended leaves), folded a phase that draws into its cell by weight
//! (`np.bincount`), and then solved each side's Bayesian game (`solve_bayesian`). Here one
//! request does all of it, in the same order and so with the same requests to the server, and
//! hands back every completion's matrix and the solved games.
//!
//! ```text
//! { "kind": "replacementNode", "p0": P, "p1": P, ...,   // completion k's position
//!   "pairs": [[a, b], ...],                              // row-major over the options
//!   "rows": m, "cols": n,
//!   "jobs": [ { "side": s, "leaf": i } ],                 // in `spreads` order, side 0's first
//!   "servers": [ { address, model, shm, bytes, batch, merge, netScoresEnds, encoding? } ],
//!   "solve": [ { "side": s, "weights": b64 } ] }          // as `beliefFold`
//! ```
//!
//! A pair the port refuses, or a request it cannot read, is a `fault`: Python then does the node
//! itself and meets the same refusal. Fills are made on the thread that reads the request.

use crate::beliefnode::{server_failed, solve_games};
use crate::encode::Encoder;
use crate::reg::Reg;
use crate::served::{Server, Stats};
use serde_json::{json, Value};

pub fn replacement_node(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    #[cfg(feature = "lp-dump")]
    let _tag = crate::lpbench::tag("replacement");
    match node(reg, encoder, value) {
        Ok(answer) | Err(answer) => answer,
    }
}

/// `encode.settle`: an ended leaf is its result, after the pass; how many ended.
fn settle(values: &mut [f64], header: &Value, net_scores_ends: bool) -> u64 {
    let mut ended = 0u64;
    for pair in header["decided"].as_array().map(Vec::as_slice).unwrap_or_default() {
        if let (Some(i), Some(v)) = (pair.get(0).and_then(Value::as_u64), pair.get(1).and_then(Value::as_f64)) {
            if (i as usize) < values.len() && !v.is_nan() {
                ended += 1;
                if !net_scores_ends {
                    values[i as usize] = v;
                }
            }
        }
    }
    ended
}

/// One completion's leaves of the replacement node, filled and encoded but not yet scored:
/// the phase's header, the encoded rows, and their count.
pub(crate) struct EncodedJob {
    header: Value,
    pub(crate) encoded: crate::encode::Encoded,
    pub(crate) rows: usize,
}

/// Every pair's phase of one completion filled and encoded (IKA-458 stage 3.6a: `node`'s
/// per-job work without a request's JSON). An `Err` is a fault to hand back.
pub(crate) fn encode_job(
    reg: &Reg,
    encoder: &Encoder,
    position: &crate::position::Position,
    pairs: &[[Vec<crate::resolve::SlotAction>; 2]],
    rules: crate::encode::EncodeRules,
) -> Result<EncodedJob, Value> {
    let per_row = (2 * encoder.widths.mons_per_side).max(1);
    let (header, encoded) = crate::resolve::commands::replacements_encoded_of(reg, encoder, position, rules, pairs)
        .map_err(|why| json!({ "fault": format!("the port refused a replacement phase: {why}") }))?;
    let rows = encoded.species.len() / per_row;
    Ok(EncodedJob { header, encoded, rows })
}

/// One completion's matrix of the replacement node from its scored leaves (`values`, one a row
/// of `job`): an ended leaf settled, a phase that draws folded into its cell by weight. The
/// matrix (row-major) and the fill's account for Python's counters; an `Err` is a fault.
pub(crate) fn job_matrix(
    job: &EncodedJob,
    mut values: Vec<f64>,
    net_scores_ends: bool,
    ended: &mut u64,
    m: usize,
    n: usize,
) -> Result<(Vec<f64>, Value), Value> {
    let fault = |e: String| json!({ "fault": e });
    let (header, rows) = (&job.header, job.rows);
    *ended += settle(&mut values, header, net_scores_ends);
    let cell_values = match header.get("branches").and_then(Value::as_array).filter(|b| !b.is_empty()) {
        Some(branches) => {
            if branches.len() != values.len() {
                return Err(fault("the branches are not the rows".into()));
            }
            let mut out = vec![0.0f64; m * n];
            for (row, branch) in branches.iter().enumerate() {
                let cell = branch.get(0).and_then(Value::as_u64).ok_or_else(|| fault("a branch's cell".into()))? as usize;
                let weight = branch.get(1).and_then(Value::as_f64).ok_or_else(|| fault("a branch's weight".into()))?;
                if cell >= out.len() {
                    return Err(fault("a branch's cell is outside the node".into()));
                }
                out[cell] += weight * values[row];
            }
            out
        }
        None => {
            if values.len() != m * n {
                return Err(fault("the rows are not one a pair".into()));
            }
            values
        }
    };
    Ok((cell_values, json!({ "leaves": rows, "mega": header["encoding"]["megaFromSlots"], "pairs": m * n })))
}

/// The server's graph bucket of a request of `rows` rows (`infer_gpu::bucket_of`), or None for
/// a request the server cuts into several graphs (or an empty one).
fn bucket(rows: usize) -> Option<usize> {
    (1..=crate::infer_gpu::MAX_ROWS).contains(&rows).then(|| crate::infer_gpu::bucket_of(rows))
}

/// IKA-473: whether `score_jobs` packs (`POKEURAOU_REPLACEMENT_PACK`: unset or `1` packs, `0`
/// sends one request a job as before, for the test that holds the two to the same values).
/// Kept in the shipping build on purpose: CI's test runs the shipping build, `0` only sends more
/// requests and gives the same games, and any value but `0` or `1` stops the node.
fn packing() -> Result<bool, String> {
    match std::env::var("POKEURAOU_REPLACEMENT_PACK").as_deref() {
        Err(_) | Ok("1") => Ok(true),
        Ok("0") => Ok(false),
        Ok(other) => Err(format!("POKEURAOU_REPLACEMENT_PACK is {other:?}; it takes 0 or 1")),
    }
}

/// IKA-473's control (the counting build only, `POKEURAOU_PACK_IGNORE_BUCKET`): jobs are packed
/// up to 4,096 rows whatever their buckets, so a job's rows can run on another graph.
#[cfg(feature = "served-report")]
fn ignore_bucket() -> bool {
    std::env::var_os("POKEURAOU_PACK_IGNORE_BUCKET").is_some()
}
#[cfg(not(feature = "served-report"))]
fn ignore_bucket() -> bool {
    false
}

/// IKA-473: the jobs (by their row counts) in the groups `score_jobs` sends together, as
/// ranges of consecutive jobs: a group's jobs and their sum share one graph bucket.
fn packs(rows: &[usize]) -> Vec<std::ops::Range<usize>> {
    let mut out = Vec::new();
    let mut first = 0usize;
    let ignore = ignore_bucket();
    while first < rows.len() {
        let own = bucket(rows[first]);
        let mut stop = first + 1;
        let mut total = rows[first];
        if ignore {
            while stop < rows.len() && rows[stop] > 0 && total + rows[stop] <= crate::infer_gpu::MAX_ROWS {
                total += rows[stop];
                stop += 1;
            }
        } else if own.is_some() {
            while stop < rows.len() && bucket(rows[stop]) == own && bucket(total + rows[stop]) == own {
                total += rows[stop];
                stop += 1;
            }
        }
        out.push(first..stop);
        first = stop;
    }
    out
}

/// IKA-473: the rows of `jobs` scored on `server`, each job's values apart. Consecutive jobs go
/// in one request while every one of them and their sum fall in the same graph bucket of the
/// server. The server runs a request on the graph of its bucket, and a row's value there
/// depends only on that row's inputs, so each job gets the values its own request gave, with
/// fewer round trips (a replacement node asked once a completion, mostly 1 to 4 rows each).
pub(crate) fn score_jobs(
    server: &Server,
    widths: &crate::encode::Widths,
    jobs: &[&EncodedJob],
    stats: &mut Stats,
) -> Result<Vec<Vec<f64>>, Value> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(jobs.len());
    let counts: Vec<usize> = jobs.iter().map(|j| j.rows).collect();
    let groups = if packing().map_err(|e| json!({ "fault": e }))? {
        packs(&counts)
    } else {
        (0..counts.len()).map(|j| j..j + 1).collect()
    };
    for group in groups {
        let blocks: Vec<&crate::encode::Encoded> = jobs[group.clone()].iter().map(|j| &j.encoded).collect();
        let rows: Vec<usize> = counts[group].to_vec();
        let batch = crate::served::begin(server, widths, &blocks, &rows, stats).map_err(server_failed)?;
        let values = crate::served::finish(server, batch, stats).map_err(server_failed)?;
        let mut at = 0usize;
        for r in rows {
            out.push(values[at..at + r].to_vec());
            at += r;
        }
    }
    Ok(out)
}

fn node(reg: &Reg, encoder: &Encoder, value: &Value) -> Result<Value, Value> {
    let fault = |e: String| json!({ "fault": e });
    let m = value["rows"].as_u64().ok_or_else(|| fault("replacementNode without rows".into()))? as usize;
    let n = value["cols"].as_u64().ok_or_else(|| fault("replacementNode without cols".into()))? as usize;
    let servers_in = value["servers"].as_array().ok_or_else(|| fault("replacementNode without servers".into()))?;
    let mut servers: Vec<Server> = Vec::with_capacity(servers_in.len());
    for s in servers_in {
        servers.push(Server::from_json(s).map_err(fault)?);
    }
    let jobs = value["jobs"].as_array().ok_or_else(|| fault("replacementNode without jobs".into()))?;
    let pairs = &value["pairs"];
    if pairs.as_array().map(Vec::len) != Some(m * n) {
        return Err(fault("the pairs are not rows x cols long".into()));
    }
    let per_row = (2 * encoder.widths.mons_per_side).max(1);
    let mut stats: Vec<Stats> = servers.iter().map(|_| Stats::default()).collect();
    let mut ended: Vec<u64> = vec![0; servers.len()];
    let mut matrices: Vec<Vec<f64>> = Vec::with_capacity(jobs.len());
    let mut sides: Vec<i64> = Vec::with_capacity(jobs.len());
    let mut fills: Vec<Value> = Vec::with_capacity(jobs.len());
    for (k, job) in jobs.iter().enumerate() {
        let side = job["side"].as_i64().ok_or_else(|| fault("a job without side".into()))?;
        let leaf = job["leaf"].as_u64().ok_or_else(|| fault("a job without leaf".into()))? as usize;
        let server = servers.get(leaf).ok_or_else(|| fault("a job's leaf is not a server".into()))?;
        let net_scores_ends = servers_in[leaf].get("netScoresEnds").and_then(Value::as_bool).unwrap_or(false);
        let mut asked = json!({ "position": value[format!("p{k}")], "pairs": pairs });
        if let Some(e) = servers_in[leaf].get("encoding").filter(|e| !e.is_null()) {
            asked["encoding"] = e.clone();
        }
        let (header, encoded) = crate::resolve::commands::replacements_encoded(reg, encoder, &asked)
            .map_err(|why| fault(format!("the port refused a replacement phase: {why}")))?;
        let rows = encoded.species.len() / per_row;
        let batch = crate::served::begin(server, &encoder.widths, &[&encoded], &[rows], &mut stats[leaf])
            .map_err(server_failed)?;
        let mut values = crate::served::finish(server, batch, &mut stats[leaf]).map_err(server_failed)?;
        ended[leaf] += settle(&mut values, &header, net_scores_ends);
        // A phase that draws is every outcome at its weight: the cell is their weighted sum, in
        // the order of the rows (`np.bincount`, IKA-352).
        let cell_values = match header.get("branches").and_then(Value::as_array).filter(|b| !b.is_empty()) {
            Some(branches) => {
                if branches.len() != values.len() {
                    return Err(fault("the branches are not the rows".into()));
                }
                let mut out = vec![0.0f64; m * n];
                for (row, branch) in branches.iter().enumerate() {
                    let cell = branch.get(0).and_then(Value::as_u64).ok_or_else(|| fault("a branch's cell".into()))? as usize;
                    let weight = branch.get(1).and_then(Value::as_f64).ok_or_else(|| fault("a branch's weight".into()))?;
                    if cell >= out.len() {
                        return Err(fault("a branch's cell is outside the node".into()));
                    }
                    out[cell] += weight * values[row];
                }
                out
            }
            None => {
                if values.len() != m * n {
                    return Err(fault("the rows are not one a pair".into()));
                }
                values
            }
        };
        fills.push(json!({
            "leaves": rows,
            "mega": header["encoding"]["megaFromSlots"],
            "pairs": m * n,
        }));
        matrices.push(cell_values);
        sides.push(side);
    }
    let mut answer = solve_games(value, &matrices, &sides, m, n).map_err(fault)?;
    answer["kind"] = json!("replacementNode");
    answer["fills"] = json!(fills);
    answer["served"] = json!(stats
        .iter()
        .zip(&ended)
        .map(|(s, e)| json!({
            "requests": s.requests,
            "rows": s.rows,
            "ended": e,
            "waitUs": s.wait_ns as f64 / 1e3,
            "copyUs": s.copy_ns as f64 / 1e3,
        }))
        .collect::<Vec<_>>());
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_ended_leaf_is_its_result_unless_the_net_scores_ends() {
        let header = json!({ "decided": [[1, 1.0], [2, 0.0]] });
        let mut values = vec![0.3, 0.4, 0.5];
        assert_eq!(settle(&mut values, &header, false), 2);
        assert_eq!(values, vec![0.3, 1.0, 0.0]);
        let mut values = vec![0.3, 0.4, 0.5];
        assert_eq!(settle(&mut values, &header, true), 2);
        assert_eq!(values, vec![0.3, 0.4, 0.5]);
    }

    #[test]
    fn completions_go_together_only_while_they_keep_their_graph_bucket() {
        // Small jobs share the 64-row bucket until their sum would leave it.
        assert_eq!(packs(&[2, 4, 3, 1]), vec![0..4]);
        assert_eq!(packs(&[30, 30, 5]), vec![0..2, 2..3]);
        // A job of another bucket goes alone, and so does one the server would cut.
        assert_eq!(packs(&[2, 100, 3]), vec![0..1, 1..2, 2..3]);
        assert_eq!(packs(&[0, 2]), vec![0..1, 1..2]);
        assert_eq!(packs(&[5000, 5000]), vec![0..1, 1..2]);
        assert_eq!(packs(&[]), Vec::<std::ops::Range<usize>>::new());
        // Two jobs of the 1,280-row bucket: their sum is in another.
        assert_eq!(packs(&[1100, 1100]), vec![0..1, 1..2]);
        assert_eq!(packs(&[70, 70]), vec![0..1, 1..2]);
    }
}
