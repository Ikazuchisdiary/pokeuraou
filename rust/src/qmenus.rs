//! IKA-389 `qMenus`: a depth-2 read's children's menus by the Q, in one crossing.
//!
//! `deepen._q_menus` built each child's two menus in the worker's Python: both sides' legal
//! pools (`qhead.legal_pool`), the Q's arrays (`qrank._pool_arrays`: the position's
//! encoding, `qhead.encode_actions`, the port's `qfeatures`), one `q_batch` request to the
//! inference server per `Q_MENU_CHUNK` children (`RemoteQ.batched`), the game on each
//! matrix (`equilibrium.solve`, or its fallback the mean against every reply), and each
//! side's `width` best by `narrow` (no cover): highest score first, ties by the choice text.
//! Here the port does all of it and hands back the menus alone.
//!
//! Every piece is the Python's to the bit: the pools action for action (`legal.rs`), the
//! arrays byte for byte into the same block in `RemoteQ._lay`'s layout, the same requests
//! (so the server answers the same matrices), the LP `lp.rs` already solves as scipy does,
//! and the scores by numpy's own BLAS (`blas.rs`), since the menus' order among the
//! equilibrium's support is decided by the last bits of those products.

use crate::blas::Blas;
use crate::encode::{EncodeRules, Encoder};
use crate::legal::{self, SideAct};
use crate::position::Position;
use crate::reg::Reg;
use crate::served::{self, Failed, Server, Stats};
use serde_json::{json, Value};

/// Numbers per candidate the Q reads from the port (`qhead.FEATURE_WIDTH`).
const FEATURES: usize = crate::qfeatures::WIDTH;

/// The positive control and the clocks of the process's `qMenus`, for `lpCounts`-like reads.
pub static MENUS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn align(offset: usize) -> usize {
    (offset + 63) & !63
}

/// One child's request: its arrays in `qrank._pool_arrays`' order (name, dtype, shape, bytes).
struct Arrays {
    named: Vec<(&'static str, &'static str, Vec<usize>, Vec<u8>)>,
    shape: (usize, usize),
}

fn i64_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| (*v as i64).to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `qrank._pool_arrays` of one position and its two pools.
fn pool_arrays(
    reg: &Reg,
    encoder: &Encoder,
    rules: EncodeRules,
    pos: &Position,
    pools: &[Vec<SideAct>; 2],
    properties: bool,
) -> Result<Arrays, String> {
    let enc = encoder.encode_positions_with(&[pos], rules);
    let w = &encoder.widths;
    let m = w.mons_per_side;
    let mut named: Vec<(&'static str, &'static str, Vec<usize>, Vec<u8>)> = vec![
        ("species", "<i8", vec![1, 2, m], i64_bytes(&enc.species)),
        ("ability", "<i8", vec![1, 2, m], i64_bytes(&enc.ability)),
        ("item", "<i8", vec![1, 2, m], i64_bytes(&enc.item)),
        ("moves", "<i8", vec![1, 2, m, 4], i64_bytes(&enc.moves)),
        ("mon", "<f4", vec![1, 2, m, w.mon], f32_bytes(&enc.mon)),
        ("mask", "<f4", vec![1, 2, m], f32_bytes(&enc.mask)),
        ("side", "<f4", vec![1, 2, w.side], f32_bytes(&enc.side)),
        ("field", "<f4", vec![1, w.field], f32_bytes(&enc.field)),
    ];
    for (side, name) in ["acts0", "acts1"].into_iter().enumerate() {
        let acts = legal::encode_actions(&encoder.vocab, pos, side, &pools[side])?;
        named.push((name, "<i4", vec![pools[side].len(), legal::SLOTS, legal::ACTION_FIELDS], i32_bytes(&acts)));
    }
    if properties {
        // `qhead.port_features`: zeros for both sides where the port refuses the position.
        let mut rows: Vec<Vec<[f64; FEATURES]>> = Vec::with_capacity(2);
        for side in 0..2 {
            let candidates: Vec<Vec<crate::resolve::SlotAction>> =
                pools[side].iter().map(|a| a.iter().map(legal::Act::slot_action).collect()).collect();
            match crate::qfeatures::side_features(reg, pos, side, &candidates) {
                Ok(got) => rows.push(got),
                Err(_) => {
                    rows = vec![vec![[0.0; FEATURES]; pools[0].len()], vec![[0.0; FEATURES]; pools[1].len()]];
                    break;
                }
            }
        }
        for (side, name) in ["feats0", "feats1"].into_iter().enumerate() {
            let flat: Vec<f32> = rows[side].iter().flatten().map(|v| *v as f32).collect();
            named.push((name, "<f4", vec![pools[side].len(), FEATURES], f32_bytes(&flat)));
        }
    }
    Ok(Arrays { named, shape: (pools[0].len(), pools[1].len()) })
}

/// `RemoteQ.batched` of one chunk: the arrays laid into the Q's block as `RemoteQ._lay`
/// lays them, one `q_batch` request, and each child's matrix read back.
fn ask_chunk(server: &Server, chunk: &[Arrays], stats: &mut Stats) -> Result<Vec<Vec<f64>>, Failed> {
    served::with_q_client(server, |client| {
        let broken = |e: std::io::Error| Failed::Broken(e.to_string());
        let copied = std::time::Instant::now();
        let mut items = Vec::with_capacity(chunk.len());
        let mut at = 0usize;
        for arrays in chunk {
            let mut layout = Vec::with_capacity(arrays.named.len());
            let mut offset = 0usize;
            let mut places = Vec::with_capacity(arrays.named.len());
            for (name, dtype, shape, bytes) in &arrays.named {
                offset = align(offset);
                layout.push(json!({
                    "name": name, "offset": offset + at, "shape": shape, "dtype": dtype, "nbytes": bytes.len(),
                }));
                places.push(offset + at);
                offset += bytes.len();
            }
            let (n0, n1) = arrays.shape;
            let result_offset = align(at + offset);
            at = align(result_offset + n0 * n1 * 8);
            if at > server.capacity {
                // `RemoteQ._lay`'s own RuntimeError, raised by the caller as it is.
                return Err(Failed::Broken(format!(
                    "\u{0}tooBig Q requests up to ({n0}, {n1}) do not fit {} bytes",
                    server.capacity
                )));
            }
            for ((_, _, _, bytes), place) in arrays.named.iter().zip(&places) {
                client.block.write_at(*place, bytes).map_err(broken)?;
            }
            items.push((json!({ "layout": layout, "shape": [n0, n1], "result_offset": result_offset }), result_offset, n0 * n1));
        }
        stats.copy_ns += copied.elapsed().as_nanos() as u64;
        let request = json!({
            "op": "q_batch",
            "model": server.model,
            "shm": server.shm,
            "items": items.iter().map(|(item, _, _)| item.clone()).collect::<Vec<_>>(),
        });
        served::ask(client, &request, stats)?;
        let mut out = Vec::with_capacity(items.len());
        for (_, result_offset, cells) in &items {
            let mut raw = vec![0u8; cells * 8];
            client.block.read_at(*result_offset, &mut raw).map_err(broken)?;
            out.push(raw.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect());
        }
        Ok(out)
    })
}

/// numpy's `q.mean(axis=1)`: each row's pairwise sum over its length.
fn row_means(q: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..m).map(|i| crate::lp::numpy_sum(&q[i * n..(i + 1) * n]) / n as f64).collect()
}

/// numpy's `q.mean(axis=0)`: the rows added in order, over their count.
fn col_means(q: &[f64], m: usize, n: usize) -> Vec<f64> {
    if n == 1 {
        // One column is contiguous, and numpy sums it pairwise.
        return vec![crate::lp::numpy_sum(q) / m as f64];
    }
    let mut out = q[..n].to_vec();
    for i in 1..m {
        for j in 0..n {
            out[j] += q[i * n + j];
        }
    }
    out.iter().map(|v| v / m as f64).collect()
}

/// Both sides' scores on one matrix (`_q_menus`: the row values against the column mix,
/// minus the column values against the row mix; the means where the game has no answer),
/// and whether the game was solved.
fn scores(blas: &Blas, q: &[f64], m: usize, n: usize, eps: f64) -> ([Vec<f64>; 2], bool) {
    match crate::lp::solve(q, m, n, eps) {
        Ok(solved) => {
            let row = blas.a_y(q, m, n, &solved.ys[0]);
            let col: Vec<f64> = blas.x_a(&solved.x, q, m, n).into_iter().map(|v| -v).collect();
            ([row, col], true)
        }
        Err(_) => {
            let col: Vec<f64> = col_means(q, m, n).into_iter().map(|v| -v).collect();
            ([row_means(q, m, n), col], false)
        }
    }
}

/// `narrow(..., rank=..., cover=False).actions`: the `width` highest, ties by choice text.
fn top(pool: &[SideAct], scores: &[f64], width: usize) -> Vec<usize> {
    let choices: Vec<String> = pool.iter().map(legal::choice).collect();
    let mut order: Vec<usize> = (0..pool.len()).collect();
    order.sort_by(|&a, &b| {
        (-scores[a])
            .partial_cmp(&-scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| choices[a].as_bytes().cmp(choices[b].as_bytes()))
    });
    order.truncate(width);
    order
}

pub fn q_menus(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    let Some(list) = value["requests"].as_array() else {
        return json!({ "error": "`qMenus` without a list of requests" });
    };
    let width = value["width"].as_u64().unwrap_or(0) as usize;
    if width < 1 {
        return json!({ "error": "`qMenus`: width must be at least 1" });
    }
    let chunk = value["chunk"].as_u64().unwrap_or(16).max(1) as usize;
    let eps = value.get("eps").and_then(Value::as_f64).unwrap_or(1e-9);
    let properties = value.get("properties").and_then(Value::as_bool).unwrap_or(true);
    let rules = EncodeRules {
        mega_from_slots: value.get("megaFromSlots").and_then(Value::as_bool).unwrap_or(false),
    };
    let server = match Server::from_json(&value["server"]) {
        Err(e) => return json!({ "error": e }),
        Ok(server) => server,
    };
    let blas = match Blas::from_json(&value["blas"]) {
        Err(e) => return json!({ "error": e }),
        Ok(blas) => blas,
    };
    let started = std::time::Instant::now();
    let positions: Vec<Position> = list.iter().map(|one| crate::held::position(&one["position"])).collect();
    for pos in &positions {
        if &*pos.format != reg.format_id.as_str() {
            return json!({ "error": format!("position is {} but the regulation is {}", pos.format, reg.format_id) });
        }
    }
    let pools: Vec<[Vec<SideAct>; 2]> =
        positions.iter().map(|pos| [legal::legal_pool(reg, pos, 0), legal::legal_pool(reg, pos, 1)]).collect();
    let legal_ns = started.elapsed().as_nanos() as u64;
    let wanted: Vec<usize> = (0..positions.len()).filter(|&k| !pools[k][0].is_empty() && !pools[k][1].is_empty()).collect();
    let lps_before = crate::lp::LPS.load(std::sync::atomic::Ordering::Relaxed);
    let mut stats = Stats::default();
    let mut menus: Vec<Value> = vec![json!([[], []]); positions.len()];
    let mut unsolved = 0u64;
    let mut arrays_ns = 0u64;
    for group in wanted.chunks(chunk) {
        let laid = std::time::Instant::now();
        let mut arrays = Vec::with_capacity(group.len());
        for &k in group {
            match pool_arrays(reg, encoder, rules, &positions[k], &pools[k], properties) {
                Ok(a) => arrays.push(a),
                Err(e) => return json!({ "invalid": e }),
            }
        }
        arrays_ns += laid.elapsed().as_nanos() as u64;
        let matrices = match ask_chunk(&server, &arrays, &mut stats) {
            Ok(got) => got,
            Err(failed) => {
                return match failed {
                    Failed::Refused { error, oom, cap_gb } => {
                        json!({ "serverError": error, "oom": oom, "capGb": cap_gb })
                    }
                    Failed::Broken(error) => match error.strip_prefix("\u{0}tooBig ") {
                        Some(text) => json!({ "tooBig": text }),
                        None => json!({ "serverError": error, "broken": true }),
                    },
                };
            }
        };
        for (&k, q) in group.iter().zip(&matrices) {
            let (m, n) = (pools[k][0].len(), pools[k][1].len());
            let (both, solved) = scores(&blas, q, m, n, eps);
            unsolved += u64::from(!solved);
            let mut menu = Vec::with_capacity(2);
            for side in 0..2 {
                let kept = top(&pools[k][side], &both[side], width);
                menu.push(Value::Array(
                    kept.iter()
                        .map(|&i| Value::Array(pools[k][side][i].iter().map(legal::Act::to_json).collect()))
                        .collect(),
                ));
            }
            menus[k] = Value::Array(menu);
        }
    }
    MENUS.fetch_add(positions.len() as u64, std::sync::atomic::Ordering::Relaxed);
    json!({
        "kind": "qMenus",
        "menus": menus,
        "asked": wanted.len(),
        "unsolved": unsolved,
        "lps": crate::lp::LPS.load(std::sync::atomic::Ordering::Relaxed) - lps_before,
        "served": {
            "requests": stats.requests,
            "waitUs": stats.wait_ns as f64 / 1e3,
            "copyUs": stats.copy_ns as f64 / 1e3,
        },
        "legalUs": legal_ns as f64 / 1e3,
        "arraysUs": arrays_ns as f64 / 1e3,
    })
}

/// `legal`: both sides' legal pools of each position, as `qhead.legal_pool` lists them --
/// for the check that holds the port's to the Python's (`tests/test_port_menus.py`).
pub fn legal_command(reg: &Reg, value: &Value) -> Value {
    let Some(list) = value["requests"].as_array() else {
        return json!({ "error": "`legal` without a list of requests" });
    };
    let pools: Vec<Value> = list
        .iter()
        .map(|one| {
            let pos = crate::held::position(&one["position"]);
            let side_pool = |side: usize| -> Value {
                let pool = if one.get("raw").and_then(Value::as_bool) == Some(true) {
                    legal::side_actions(reg, &pos, side)
                } else {
                    legal::legal_pool(reg, &pos, side)
                };
                Value::Array(pool.iter().map(|a| Value::Array(a.iter().map(legal::Act::to_json).collect())).collect())
            };
            json!([side_pool(0), side_pool(1)])
        })
        .collect();
    json!({ "kind": "legal", "pools": pools })
}

/// `qArrays`: `qrank._pool_arrays` of each position with its legal pools, the arrays as
/// base64 -- for the check that holds them to the Python's bytes.
pub fn arrays_command(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    let Some(list) = value["requests"].as_array() else {
        return json!({ "error": "`qArrays` without a list of requests" });
    };
    let properties = value.get("properties").and_then(Value::as_bool).unwrap_or(true);
    let rules = EncodeRules {
        mega_from_slots: value.get("megaFromSlots").and_then(Value::as_bool).unwrap_or(false),
    };
    let mut out = Vec::with_capacity(list.len());
    for one in list {
        let pos = crate::held::position(&one["position"]);
        let pools = [legal::legal_pool(reg, &pos, 0), legal::legal_pool(reg, &pos, 1)];
        match pool_arrays(reg, encoder, rules, &pos, &pools, properties) {
            Err(e) => out.push(json!({ "invalid": e })),
            Ok(arrays) => {
                let mut named = serde_json::Map::new();
                for (name, dtype, shape, bytes) in &arrays.named {
                    named.insert(
                        (*name).to_string(),
                        json!({ "dtype": dtype, "shape": shape, "data": crate::lp::b64_encode(bytes) }),
                    );
                }
                out.push(Value::Object(named));
            }
        }
    }
    json!({ "kind": "qArrays", "arrays": out })
}

/// `qScores`: `scores` and `top` of given matrices and pools' choice order, for the check
/// that the ranking is `_q_menus`' (numpy's BLAS, the mean fallback, the tie order).
pub fn scores_command(value: &Value) -> Value {
    let blas = match Blas::from_json(&value["blas"]) {
        Err(e) => return json!({ "error": e }),
        Ok(blas) => blas,
    };
    let eps = value.get("eps").and_then(Value::as_f64).unwrap_or(1e-9);
    let force_mean = value.get("mean").and_then(Value::as_bool).unwrap_or(false);
    let Some(games) = value["games"].as_array() else {
        return json!({ "error": "`qScores` without games" });
    };
    let mut out = Vec::with_capacity(games.len());
    for game in games {
        let m = game["rows"].as_u64().unwrap_or(0) as usize;
        let n = game["cols"].as_u64().unwrap_or(0) as usize;
        let data = match crate::lp::b64_decode(game["data"].as_str().unwrap_or("")) {
            Ok(d) => d,
            Err(e) => return json!({ "error": e }),
        };
        let q: Vec<f64> = data.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect();
        if q.len() != m * n || m == 0 || n == 0 {
            return json!({ "error": "`qScores`: a matrix of the wrong size" });
        }
        let (both, solved) = if force_mean {
            let col: Vec<f64> = col_means(&q, m, n).into_iter().map(|v| -v).collect();
            ([row_means(&q, m, n), col], false)
        } else {
            scores(&blas, &q, m, n, eps)
        };
        let bits = |v: &[f64]| crate::lp::b64_encode(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        out.push(json!({ "rows": bits(&both[0]), "cols": bits(&both[1]), "solved": solved }));
    }
    json!({ "kind": "qScores", "answers": out })
}
