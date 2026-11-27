//! IKA-389: positions' menus by the Q, built in the port.
//!
//! `deepen._q_menus` built each child's two menus in the worker's Python: both sides' legal
//! pools (`qhead.legal_pool`), the Q's arrays (`qrank._pool_arrays`: the position's
//! encoding, `qhead.encode_actions`, the port's `qfeatures`), one `q_batch` request to the
//! inference server per `Q_MENU_CHUNK` children (`RemoteQ.batched`), the game on each
//! matrix (`equilibrium.solve`, or its fallback the mean against every reply), and each
//! side's `width` best by `narrow` (no cover): highest score first, ties by the choice text.
//! Here the port does all of it: `menus_of` for the port's deepening (`deepen.rs`), and
//! `rootMenus` / `moveNode` for a move node's own two menus. The `qMenus` request that sent
//! the deepening's children from Python went in IKA-464.
//!
//! The pools are the Python's action for action (`legal.rs`), the arrays are laid byte for
//! byte into the same block in `RemoteQ._lay`'s layout, the same requests go to the server,
//! and the LP is `lp.rs`'s. The scores are plain row and column products (IKA-460: they were
//! numpy's own BLAS, to give the same order among the equilibrium's tied support).

use crate::encode::{EncodeRules, Encoder};
use crate::legal::{self, SideAct};
use crate::position::Position;
use crate::reg::Reg;
use crate::served::{self, Failed, Server, Stats};
use serde_json::{json, Value};

/// Numbers per candidate the Q reads from the port (`qhead.FEATURE_WIDTH`).
const FEATURES: usize = crate::qfeatures::WIDTH;

/// The positions this process built Q menus for (`menus_of`, `rootMenus`), a positive control.
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

/// Which request the server is sent: `batched` (`q_batch`) or `matrices` (`q` / `q_many`).
#[derive(Clone, Copy)]
enum Op {
    Batch,
    Matrices,
}

/// `RemoteQ.batched` of one chunk: the arrays laid into the Q's block as `RemoteQ._lay`
/// lays them, one `q_batch` request, and each child's matrix read back.
fn ask_chunk(server: &Server, chunk: &[Arrays], stats: &mut Stats, op: Op) -> Result<Vec<Vec<f64>>, Failed> {
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
        let listed: Vec<Value> = items.iter().map(|(item, _, _)| item.clone()).collect();
        let request = match op {
            // One forward pass for the children (`RemoteQ.batched`).
            Op::Batch => json!({ "op": "q_batch", "model": server.model, "shm": server.shm, "items": listed }),
            // `RemoteQ.matrices` (IKA-457 stage 2.3): one position as `q`, several as `q_many`,
            // each answered as its own pass.
            Op::Matrices if listed.len() == 1 => {
                let mut one = json!({ "op": "q", "model": server.model, "shm": server.shm });
                for (key, v) in listed[0].as_object().expect("an item is an object") {
                    one[key] = v.clone();
                }
                one
            }
            Op::Matrices => json!({ "op": "q_many", "model": server.model, "shm": server.shm, "items": listed }),
        };
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

/// Each row's mean.
fn row_means(q: &[f64], m: usize, n: usize) -> Vec<f64> {
    (0..m).map(|i| crate::lp::sum_f64(&q[i * n..(i + 1) * n]) / n as f64).collect()
}

/// Each column's mean.
fn col_means(q: &[f64], m: usize, n: usize) -> Vec<f64> {
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
fn scores(q: &[f64], m: usize, n: usize, eps: f64) -> ([Vec<f64>; 2], bool) {
    match crate::lp::solve(q, m, n, eps) {
        Ok(solved) => {
            let y = &solved.ys[0];
            let row: Vec<f64> = (0..m).map(|i| crate::lp::dot(&q[i * n..(i + 1) * n], y)).collect();
            let mut col = vec![0.0f64; n];
            for (i, xi) in solved.x.iter().enumerate() {
                for (c, v) in col.iter_mut().zip(&q[i * n..(i + 1) * n]) {
                    *c -= xi * v;
                }
            }
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
    // The positive control of a board arm read through another executable (`armexe.rs`).
    let width = if cfg!(feature = "ika458-arm-control") { width.saturating_sub(1).max(1) } else { width };
    order.truncate(width);
    order
}

/// `menus_of`'s answer: each position's two menus (both empty where a side has no legal
/// action), and what the server was asked.
pub(crate) struct QMenus {
    pub(crate) menus: Vec<[Vec<SideAct>; 2]>,
    pub(crate) stats: Stats,
}

/// Menus by the Q for typed positions (IKA-464: the port's deepening asks it for a refined
/// cell's children): each side's `width` best by the Q solved over both legal pools, no
/// cover, one `q_batch` request per `chunk` positions. A failure comes back as the answer's
/// fields (`invalid`, `tooBig` or `serverError`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn menus_of(
    reg: &Reg,
    encoder: &Encoder,
    positions: &[Position],
    width: usize,
    server: &Server,
    rules: EncodeRules,
    properties: bool,
    eps: f64,
    chunk: usize,
) -> Result<QMenus, Value> {
    let pools: Vec<[Vec<SideAct>; 2]> =
        positions.iter().map(|pos| [legal::legal_pool(reg, pos, 0), legal::legal_pool(reg, pos, 1)]).collect();
    let wanted: Vec<usize> = (0..positions.len()).filter(|&k| !pools[k][0].is_empty() && !pools[k][1].is_empty()).collect();
    let mut stats = Stats::default();
    let mut menus: Vec<[Vec<SideAct>; 2]> = vec![[Vec::new(), Vec::new()]; positions.len()];
    for group in wanted.chunks(chunk.max(1)) {
        let mut arrays = Vec::with_capacity(group.len());
        for &k in group {
            match pool_arrays(reg, encoder, rules, &positions[k], &pools[k], properties) {
                Ok(a) => arrays.push(a),
                Err(e) => return Err(json!({ "invalid": e })),
            }
        }
        let matrices = ask_chunk(server, &arrays, &mut stats, Op::Batch).map_err(|failed| match failed {
            Failed::Refused { error, oom, cap_gb } => json!({ "serverError": error, "oom": oom, "capGb": cap_gb }),
            Failed::Broken(error) => match error.strip_prefix("\u{0}tooBig ") {
                Some(text) => json!({ "tooBig": text }),
                None => json!({ "serverError": error, "broken": true }),
            },
        })?;
        for (&k, q) in group.iter().zip(&matrices) {
            let (m, n) = (pools[k][0].len(), pools[k][1].len());
            let (both, _solved) = scores(q, m, n, eps);
            for side in 0..2 {
                let kept = top(&pools[k][side], &both[side], width);
                menus[k][side] = kept.iter().map(|&i| pools[k][side][i].clone()).collect();
            }
        }
    }
    MENUS.fetch_add(positions.len() as u64, std::sync::atomic::Ordering::Relaxed);
    Ok(QMenus { menus, stats })
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
/// that the ranking is `_q_menus`' (the mean fallback, the tie order).
pub fn scores_command(value: &Value) -> Value {
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
            scores(&q, m, n, eps)
        };
        let bits = |v: &[f64]| crate::lp::b64_encode(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        out.push(json!({ "rows": bits(&both[0]), "cols": bits(&both[1]), "solved": solved }));
    }
    json!({ "kind": "qScores", "answers": out })
}

/// `narrow._slot_key` of one slot of an action: the identity of its option, for the cover.
fn slot_key(k: usize, act: &legal::Act) -> String {
    match act {
        legal::Act::Move { move_id, target, mega, .. } => {
            format!("{k}:move:{}:{target:?}:{}", move_id.as_str(), u8::from(*mega))
        }
        legal::Act::Switch { party, .. } => format!("{k}:switch:{party}"),
        legal::Act::Pass { .. } => format!("{k}:pass"),
    }
}

/// How many slot options of `pool` the `kept` actions leave off (`Narrowed.uncovered`'s length)
/// when the pool was cut (a pool within the limit is whole and leaves none).
fn uncovered(pool: &[SideAct], kept: &[usize], limit: usize) -> usize {
    if pool.len() <= limit {
        return 0;
    }
    let keys = |action: &SideAct| -> Vec<String> { action.iter().enumerate().map(|(k, a)| slot_key(k, a)).collect() };
    let all: std::collections::HashSet<String> = pool.iter().flat_map(keys).collect();
    let taken: std::collections::HashSet<String> = kept.iter().flat_map(|&i| keys(&pool[i])).collect();
    all.difference(&taken).count()
}

/// IKA-457 stage 2.3 `rootMenus`: a move node's two menus by the Q, in one crossing.
///
/// `_menus` with a `q-nocover` fill built them in the worker's Python: both sides' legal pools,
/// `qrank.prefetch`'s one or two requests (`RemoteQ.matrices`: side `s` asks its own pool on the
/// node against the other side's pool on the view it ranks from, the arrays of that view), the
/// Q's games (`portlp.solve_many`) and each side's ranking, `narrow(..., cover=False)` of it. Here
/// the port lists the pools (`legal.rs`), lays the arrays into the Q's block, sends the same `q`
/// or `q_many` request, solves the games, scores with numpy's BLAS and cuts each menu at its
/// width. The same bits as the Python road (`tests/test_root_menus_port.py`).
///
/// ```text
/// { "kind": "rootMenus", "position": P, "v0": P, "v1": P, "same": bool, "widths": [w0, w1],
///   "properties": bool, "megaFromSlots": bool, "server": S }
/// ```
///
/// `same` is whether the two sides rank from one view object (and so, with equal pools, from one
/// request). An empty pool, or a position of another regulation, is a `fault`.
pub fn root_menus(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    match root_menus_node(reg, encoder, value) {
        Ok((answer, _kept)) => answer,
        Err(answer) => answer,
    }
}

/// `rootMenus`' answer and the two menus themselves (the actions, for `moveNode`, which solves
/// the node over them in the same request). The Q's server is `qServer` when the request names
/// one beside the node's own `server`.
pub(crate) fn root_menus_node(
    reg: &Reg,
    encoder: &Encoder,
    value: &Value,
) -> Result<(Value, [Vec<SideAct>; 2]), Value> {
    let args = MenusArgs::from_json(value)?;
    let pos = crate::held::position(&value["position"]);
    let views: Vec<Position> = [&value["v0"], &value["v1"]].into_iter().map(crate::held::position).collect();
    let same = value.get("same").and_then(Value::as_bool).unwrap_or(false);
    menus_core(reg, encoder, &args, &pos, [&views[0], &views[1]], same)
}

/// What `rootMenus` reads that does not change from node to node of a game: the cut widths, the
/// Q's server, the BLAS to score with, how the arrays are laid out. Typed, so a game's loop
/// (`playgame.rs`) builds it once and does not write a request for each node.
pub(crate) struct MenusArgs {
    pub(crate) widths: [usize; 2],
    pub(crate) eps: f64,
    pub(crate) properties: bool,
    pub(crate) rules: EncodeRules,
    pub(crate) server: Server,
}

impl MenusArgs {
    pub(crate) fn from_json(value: &Value) -> Result<MenusArgs, Value> {
        let fault = |e: String| json!({ "fault": e });
        let widths: Vec<usize> = value["widths"]
            .as_array()
            .filter(|w| w.len() == 2)
            .and_then(|w| w.iter().map(|x| x.as_u64().map(|n| n as usize)).collect())
            .ok_or_else(|| fault("`rootMenus` without two widths".into()))?;
        if widths.iter().any(|w| *w < 1) {
            return Err(fault("`rootMenus`: a width must be at least 1".into()));
        }
        Ok(MenusArgs {
            widths: [widths[0], widths[1]],
            eps: value.get("eps").and_then(Value::as_f64).unwrap_or(1e-9),
            properties: value.get("properties").and_then(Value::as_bool).unwrap_or(true),
            rules: EncodeRules { mega_from_slots: value.get("megaFromSlots").and_then(Value::as_bool).unwrap_or(false) },
            server: Server::from_json(value.get("qServer").filter(|s| !s.is_null()).unwrap_or(&value["server"]))
                .map_err(fault)?,
        })
    }
}

/// `rootMenus` over a node's position and the two views each side ranks from. `same` is whether
/// the two sides rank from one view object (and so, with equal pools, from one request).
pub(crate) fn menus_core(
    reg: &Reg,
    encoder: &Encoder,
    args: &MenusArgs,
    pos: &Position,
    views: [&Position; 2],
    same: bool,
) -> Result<(Value, [Vec<SideAct>; 2]), Value> {
    let fault = |e: String| json!({ "fault": e });
    let MenusArgs { widths, eps, properties, rules, server } = args;
    let (eps, properties, rules) = (*eps, *properties, *rules);
    for p in std::iter::once(pos).chain(views.iter().copied()) {
        if &*p.format != reg.format_id.as_str() {
            return Err(fault(format!("position is {} but the regulation is {}", p.format, reg.format_id)));
        }
    }
    let started = std::time::Instant::now();
    // Side `s` ranks its pool on the node against the other side's pool on its view.
    let mut pools: Vec<[Vec<SideAct>; 2]> = Vec::with_capacity(2);
    for s in 0..2 {
        let own = legal::legal_pool(reg, &pos, s);
        let foe = legal::legal_pool(reg, &views[s], 1 - s);
        if own.is_empty() || foe.is_empty() {
            return Err(fault("a side has no legal action".into()));
        }
        pools.push(if s == 0 { [own, foe] } else { [foe, own] });
    }
    let legal_ns = started.elapsed().as_nanos() as u64;
    let choices = |p: &[Vec<SideAct>; 2]| -> [Vec<String>; 2] {
        [p[0].iter().map(legal::choice).collect(), p[1].iter().map(legal::choice).collect()]
    };
    let one_request = same && choices(&pools[0]) == choices(&pools[1]);
    let request_of = [0usize, if one_request { 0 } else { 1 }];
    let requests: Vec<usize> = if one_request { vec![0] } else { vec![0, 1] };
    let lps_before = crate::lp::LPS.load(std::sync::atomic::Ordering::Relaxed);
    let laid = std::time::Instant::now();
    let mut arrays = Vec::with_capacity(requests.len());
    for &r in &requests {
        arrays.push(
            pool_arrays(reg, encoder, rules, &views[r], &pools[r], properties).map_err(|e| json!({ "invalid": e }))?,
        );
    }
    let arrays_ns = laid.elapsed().as_nanos() as u64;
    let mut stats = Stats::default();
    let matrices = ask_chunk(&server, &arrays, &mut stats, Op::Matrices).map_err(|failed| match failed {
        Failed::Refused { error, oom, cap_gb } => json!({ "serverError": error, "oom": oom, "capGb": cap_gb }),
        Failed::Broken(error) => match error.strip_prefix("\u{0}tooBig ") {
            Some(text) => json!({ "tooBig": text }),
            None => json!({ "serverError": error, "broken": true }),
        },
    })?;
    let mut menus = Vec::with_capacity(2);
    let mut kept_actions: [Vec<SideAct>; 2] = [Vec::new(), Vec::new()];
    let mut cuts = Vec::with_capacity(2);
    let mut cells = Vec::with_capacity(2);
    let mut unsolved = 0u64;
    let mut solved_once: Vec<Option<([Vec<f64>; 2], bool)>> = vec![None; requests.len()];
    for s in 0..2 {
        let r = request_of[s];
        let (m, n) = (pools[r][0].len(), pools[r][1].len());
        if solved_once[r].is_none() {
            let got = scores(&matrices[r], m, n, eps);
            unsolved += u64::from(!got.1);
            solved_once[r] = Some(got);
        }
        let both = &solved_once[r].as_ref().expect("solved just above").0;
        let kept = top(&pools[r][s], &both[s], widths[s]);
        cuts.push(uncovered(&pools[r][s], &kept, widths[s]));
        kept_actions[s] = kept.iter().map(|&i| pools[r][s][i].clone()).collect();
        menus.push(Value::Array(
            kept.iter().map(|&i| Value::Array(pools[r][s][i].iter().map(legal::Act::to_json).collect())).collect(),
        ));
        cells.push(m * n);
    }
    MENUS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok((json!({
        "kind": "rootMenus",
        "menus": menus,
        "uncovered": cuts,
        "cells": cells,
        "requests": requests.len(),
        "unsolved": unsolved,
        "lps": crate::lp::LPS.load(std::sync::atomic::Ordering::Relaxed) - lps_before,
        "served": { "requests": stats.requests, "waitUs": stats.wait_ns as f64 / 1e3, "copyUs": stats.copy_ns as f64 / 1e3 },
        "legalUs": legal_ns as f64 / 1e3,
        "arraysUs": arrays_ns as f64 / 1e3,
    }), kept_actions))
}

/// A slot action as `rustnode.dump_action` writes it, which `parse_slot_action` reads.
pub(crate) fn act_json(act: &legal::Act) -> Value {
    match act {
        legal::Act::Move { slot, index, move_id, target, mega } => json!({
            "kind": "move", "slot": slot, "moveIndex": index, "moveId": move_id.as_str(),
            "target": target, "mega": mega,
        }),
        legal::Act::Switch { slot, party, species } => json!({
            "kind": "switch", "slot": slot, "partyIndex": party, "species": species.as_str(),
        }),
        legal::Act::Pass { slot } => json!({ "kind": "pass", "slot": slot }),
    }
}

/// Showdown's ID normalisation (`regulation.to_id`): lowercase, everything but letters and digits out.
fn to_id(name: &str) -> String {
    name.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect()
}

/// What `beliefnode._action_features` reads off an action: whether it forces a switch, attacks,
/// and which party members its switches bring in.
pub(crate) fn action_feature(reg: &Reg, action: &SideAct) -> Value {
    const PHAZING: [&str; 4] = ["dragontail", "circlethrow", "whirlwind", "roar"];
    let mut phazes = false;
    let mut attacks = false;
    let mut switches: Vec<Value> = Vec::new();
    for act in action {
        match act {
            legal::Act::Move { move_id, .. } => {
                phazes |= PHAZING.contains(&move_id.as_str());
                attacks |= reg
                    .moves
                    .get(move_id)
                    .is_some_and(|m| m.category != crate::reg::Category::Status);
            }
            legal::Act::Switch { party, species, .. } => switches.push(json!([party, to_id(species.as_str())])),
            legal::Act::Pass { .. } => {}
        }
    }
    json!({ "p": phazes, "a": attacks, "s": switches })
}

/// IKA-457 stage 2.4 `moveNode`: a move node's menus and the node over them in one request.
///
/// `rootMenus`' request and `beliefNode`'s (minus `ours`, `theirs`, `rows`, `cols`, `rowFeat` and
/// `colFeat`, which come from the menus built here), with the Q's server named `qServer`. The
/// answer is `beliefNode`'s and `rootMenus`' under `rootMenus`. A fault in either is the answer.
pub fn move_node(reg: &Reg, encoder: &Encoder, value: &Value) -> Value {
    let got = match root_menus_node(reg, encoder, value) {
        Ok(got) => got,
        Err(answer) => return answer,
    };
    move_node_over(reg, encoder, got, value.clone())
}

/// `moveNode`'s second half: the node over the menus `got` (`menus_core`'s answer), asked as
/// `asked` says (`beliefNode`'s request without the actions).
pub(crate) fn move_node_over(
    reg: &Reg,
    encoder: &Encoder,
    got: (Value, [Vec<SideAct>; 2]),
    mut asked: Value,
) -> Value {
    let (menus_answer, kept) = got;
    asked["ours"] = Value::Array(kept[0].iter().map(|a| Value::Array(a.iter().map(act_json).collect())).collect());
    asked["theirs"] = Value::Array(kept[1].iter().map(|a| Value::Array(a.iter().map(act_json).collect())).collect());
    asked["rows"] = json!(kept[0].len());
    asked["cols"] = json!(kept[1].len());
    asked["rowFeat"] = Value::Array(kept[0].iter().map(|a| action_feature(reg, a)).collect());
    asked["colFeat"] = Value::Array(kept[1].iter().map(|a| action_feature(reg, a)).collect());
    let mut answer = crate::beliefplan::belief_node(reg, encoder, &asked);
    if answer.get("fault").is_some() || answer.get("serverError").is_some() {
        return answer;
    }
    answer["rootMenus"] = menus_answer;
    answer
}
