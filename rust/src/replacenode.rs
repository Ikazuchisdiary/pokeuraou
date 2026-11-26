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

/// One completion's matrix of the replacement node, from a position and the pairs of options
/// already read (IKA-458 stage 3.6a: `node`'s per-job work without a request's JSON): every
/// pair's phase filled and encoded, scored on `server`, an ended leaf settled, a phase that draws
/// folded into its cell by weight. The matrix (row-major) and the fill's account for Python's
/// counters; an `Err` is an answer to hand back (a fault, or the server's failure).
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_job(
    reg: &Reg,
    encoder: &Encoder,
    position: &crate::position::Position,
    pairs: &[[Vec<crate::resolve::SlotAction>; 2]],
    rules: crate::encode::EncodeRules,
    server: &Server,
    net_scores_ends: bool,
    stats: &mut Stats,
    ended: &mut u64,
    m: usize,
    n: usize,
) -> Result<(Vec<f64>, Value), Value> {
    let fault = |e: String| json!({ "fault": e });
    let per_row = (2 * encoder.widths.mons_per_side).max(1);
    let (header, encoded) = crate::resolve::commands::replacements_encoded_of(reg, encoder, position, rules, pairs)
        .map_err(|why| fault(format!("the port refused a replacement phase: {why}")))?;
    let rows = encoded.species.len() / per_row;
    let batch = crate::served::begin(server, &encoder.widths, &[&encoded], &[rows], stats).map_err(server_failed)?;
    let mut values = crate::served::finish(server, batch, stats).map_err(server_failed)?;
    *ended += settle(&mut values, &header, net_scores_ends);
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
}
