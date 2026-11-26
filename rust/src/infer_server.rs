//! IKA-461: the inference server in Rust (`pokeuraou-damage inference-server`).
//!
//! It answers the protocol of `tools/inference_server.py` / `src/pokeuraou/inference.py`:
//! one JSON line per request over a socket, the arrays and the answers in a named shared
//! block (`RemoteValue._block`, `served.rs`). The ops it knows are `ping`, `describe`,
//! `score` (and `score_segments`, scored as one block: the rows are the same and the order
//! is the same), and `stages` (for the tests: every step of the first net's forward pass).
//! A Q arm's ops (`q`, `q_many`, `q_batch`, `describe_q`) are passed on, line for line, to
//! the Python server named by `--q-forward` until the Q arm is moved here.
//!
//! One thread a connection. A request takes a free slot (`infer_gpu::Slot`), copies its rows
//! into the slot's pinned memory, replays a CUDA Graph and sleeps until the card is done.

use crate::encode::Encoder;
use crate::infer_gpu::{self, Geometry, Gpu, Net, QItem, QNet, QSlot, Slot, MAX_ROWS, Q_ARRAYS};
use crate::reg::Reg;
use crate::rnet;
use crate::shm::Shared;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

const ARRAYS: [&str; 8] = ["species", "ability", "item", "moves", "mon", "mask", "side", "field"];

struct Arm {
    name: String,
    files: Vec<String>,
    nets: Vec<Net>,
}

#[derive(Default)]
struct Stats {
    requests: AtomicU64,
    rows: AtomicU64,
    /// Nanoseconds: in `score` altogether, waiting for a slot, in the copy into pinned memory,
    /// in the graph launch and the wait for the card.
    handled_ns: AtomicU64,
    slot_wait_ns: AtomicU64,
    copy_ns: AtomicU64,
    gpu_ns: AtomicU64,
    q_forwarded: AtomicU64,
    /// Items answered by a Q arm of this server.
    q_served: AtomicU64,
}

struct QArm {
    name: String,
    file: String,
    net: QNet,
}

struct Server {
    gpu: Gpu,
    arms: Vec<Arm>,
    q_arms: Vec<QArm>,
    /// Arms with no net (`--hp-share-arm NAME`): `payoff.HP_SHARE` read off the encoding.
    hp_arms: Vec<String>,
    q_slots: Mutex<Vec<QSlot>>,
    q_free: Condvar,
    geometry: Geometry,
    slots: Mutex<Vec<Slot>>,
    free: Condvar,
    q_forward: Option<String>,
    stats: Stats,
}

impl Server {
    fn take_slot(&self) -> Slot {
        let mut slots = self.slots.lock().unwrap();
        loop {
            if let Some(slot) = slots.pop() {
                return slot;
            }
            slots = self.free.wait(slots).unwrap();
        }
    }

    fn give_slot(&self, slot: Slot) {
        self.slots.lock().unwrap().push(slot);
        self.free.notify_one();
    }

    fn take_q_slot(&self) -> QSlot {
        let mut slots = self.q_slots.lock().unwrap();
        loop {
            if let Some(slot) = slots.pop() {
                return slot;
            }
            slots = self.q_free.wait(slots).unwrap();
        }
    }

    fn give_q_slot(&self, slot: QSlot) {
        self.q_slots.lock().unwrap().push(slot);
        self.q_free.notify_one();
    }
}

/// A Q request's items (`q`: the request itself; `q_many`, `q_batch`: its `items`): their
/// arrays staged into the slot, and where each answer goes.
fn stage_q(
    slot: &mut QSlot,
    shared: &mut Shared,
    geom: Geometry,
    items: &[&Value],
) -> Result<(Vec<QItem>, Vec<usize>, usize, usize), String> {
    let row_bytes = geom.row_bytes();
    let mut staged = Vec::new();
    let mut result_offsets = Vec::new();
    let (mut at, mut cells) = (0usize, 0usize);
    for item in items {
        let layout = item["layout"].as_array().ok_or("a Q item without `layout`")?;
        let mut offsets = [0usize; 12];
        let mut counts = [0usize; 12];
        for (k, name) in Q_ARRAYS.iter().enumerate() {
            let entry = layout
                .iter()
                .find(|e| e["name"].as_str() == Some(name))
                .ok_or(format!("a Q item without `{name}`"))?;
            let shape: Vec<usize> =
                entry["shape"].as_array().ok_or("layout `shape`")?.iter().map(|n| n.as_u64().unwrap_or(0) as usize).collect();
            let count: usize = shape.iter().product();
            let dtype = entry["dtype"].as_str().unwrap_or("");
            let src = entry["offset"].as_u64().ok_or("layout `offset`")? as usize;
            let ints = k < 4 || (8..10).contains(&k);
            at = (at + 63) & !63;
            let dst = slot.stage_mut(at, count * 4)?;
            match (ints, dtype) {
                (true, "<i4") | (false, "<f4") => shared.read_at(src, dst).map_err(|e| e.to_string())?,
                (true, "<i8") => {
                    let mut raw = vec![0u8; count * 8];
                    shared.read_at(src, &mut raw).map_err(|e| e.to_string())?;
                    for (d, s) in dst.chunks_exact_mut(4).zip(raw.chunks_exact(8)) {
                        let v = i64::from_le_bytes(s.try_into().unwrap());
                        d.copy_from_slice(&(v as i32).to_le_bytes());
                    }
                }
                _ => return Err(format!("`{name}` has dtype {dtype}")),
            }
            if k < 8 && (shape[0] != 1 || count * 4 != row_bytes[k]) {
                return Err(format!("`{name}` is not one row of this server's encoder"));
            }
            offsets[k] = at;
            counts[k] = shape[0];
            at += count * 4;
        }
        let (n0, n1) = (counts[8], counts[9]);
        let asked = item["shape"].as_array().map(|s| (s[0].as_u64().unwrap_or(0) as usize, s[1].as_u64().unwrap_or(0) as usize));
        if asked != Some((n0, n1)) || counts[10] != n0 || counts[11] != n1 {
            return Err(format!("a Q item's pools are {n0} x {n1} and its shape says {asked:?}"));
        }
        staged.push(QItem { offsets, n0, n1, cells_at: cells });
        result_offsets.push(item["result_offset"].as_u64().ok_or("`result_offset` is missing")? as usize);
        cells += n0 * n1;
    }
    Ok((staged, result_offsets, at, cells))
}

fn q_request(server: &Server, req: &Value, attached: &mut HashMap<String, Shared>) -> Result<Value, String> {
    let name = req["model"].as_str().ok_or("`model` is missing")?;
    let arm = server
        .q_arms
        .iter()
        .find(|a| a.name == name)
        .ok_or(format!("no Q arm named {name:?}; have {:?}", server.q_arms.iter().map(|a| &a.name).collect::<Vec<_>>()))?;
    if req["op"].as_str() == Some("describe_q") {
        return Ok(json!({
            "ok": true, "files": [arm.file], "fingerprint": arm.net.fingerprint,
            "properties": arm.net.properties, "sha256": [arm.net.sha256],
        }));
    }
    let items: Vec<&Value> = match req["items"].as_array() {
        Some(list) => list.iter().collect(),
        None => vec![req],
    };
    let mut needed = 0usize;
    for item in &items {
        let rows = item["shape"].as_array().map(|s| s[0].as_u64().unwrap_or(0) * s[1].as_u64().unwrap_or(0)).unwrap_or(0);
        needed = needed.max(item["result_offset"].as_u64().unwrap_or(0) as usize + rows as usize * 8);
        for e in item["layout"].as_array().into_iter().flatten() {
            needed = needed.max(e["offset"].as_u64().unwrap_or(0) as usize + e["nbytes"].as_u64().unwrap_or(0) as usize);
        }
    }
    let shared = attach(attached, req["shm"].as_str().ok_or("`shm` is missing")?, needed)?;
    let geom = server.geometry;
    let mut slot = server.take_q_slot();
    let result = (|| -> Result<(), String> {
        let started = Instant::now();
        let (staged, result_offsets, used, cells) = stage_q(&mut slot, shared, geom, &items)?;
        server.stats.copy_ns.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        let on_card = Instant::now();
        let values = slot.matrices(&server.gpu, &arm.net, &staged, used, cells)?;
        server.stats.gpu_ns.fetch_add(on_card.elapsed().as_nanos() as u64, Relaxed);
        for (item, at) in staged.iter().zip(&result_offsets) {
            let bytes: Vec<u8> =
                values[item.cells_at..item.cells_at + item.n0 * item.n1].iter().flat_map(|v| v.to_le_bytes()).collect();
            shared.write_at(*at, &bytes).map_err(|e| e.to_string())?;
        }
        Ok(())
    })();
    server.give_q_slot(slot);
    result?;
    server.stats.q_served.fetch_add(items.len() as u64, Relaxed);
    Ok(json!({ "ok": true }))
}

fn error_reply(message: impl std::fmt::Display) -> Value {
    json!({ "ok": false, "error": message.to_string() })
}

struct Item {
    offset: usize,
    dtype: String,
    shape: Vec<usize>,
}

fn layout_of(req: &Value) -> Result<(Vec<Item>, usize), String> {
    let layout = req["layout"].as_array().ok_or("`layout` is missing")?;
    let mut items: Vec<Option<Item>> = (0..8).map(|_| None).collect();
    let mut needed = 0usize;
    for entry in layout {
        let name = entry["name"].as_str().ok_or("a layout entry without a name")?;
        let Some(k) = ARRAYS.iter().position(|a| *a == name) else { continue };
        let offset = entry["offset"].as_u64().ok_or("layout `offset`")? as usize;
        let nbytes = entry["nbytes"].as_u64().ok_or("layout `nbytes`")? as usize;
        let shape = entry["shape"]
            .as_array()
            .ok_or("layout `shape`")?
            .iter()
            .map(|n| n.as_u64().unwrap_or(0) as usize)
            .collect();
        let dtype = entry["dtype"].as_str().ok_or("layout `dtype`")?.to_string();
        needed = needed.max(offset + nbytes);
        items[k] = Some(Item { offset, dtype, shape });
    }
    let items = items
        .into_iter()
        .enumerate()
        .map(|(k, i)| i.ok_or(format!("the layout has no `{}`", ARRAYS[k])))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((items, needed))
}

/// Rows [first, first + n) of every array into the slot's pinned inputs.
fn fill(slot: &mut Slot, shared: &mut Shared, items: &[Item], first: usize, n: usize) -> Result<(), String> {
    let geom = slot.geometry();
    let row_bytes = geom.row_bytes();
    for (k, item) in items.iter().enumerate() {
        let ints = k < 4;
        let wide = match (ints, item.dtype.as_str()) {
            (true, "<i4") | (false, "<f4") => false,
            (true, "<i8") => true,
            (_, other) => return Err(format!("`{}` has dtype {other}", ARRAYS[k])),
        };
        let per_row: usize = item.shape.iter().skip(1).product();
        let elem = if wide { 8 } else { 4 };
        if per_row * 4 != row_bytes[k] {
            return Err(format!(
                "`{}` has rows of {per_row} values; this server's encoder has {}",
                ARRAYS[k],
                row_bytes[k] / 4
            ));
        }
        let at = item.offset + first * per_row * elem;
        if wide {
            let mut raw = vec![0u8; n * per_row * 8];
            shared.read_at(at, &mut raw).map_err(|e| e.to_string())?;
            let out = slot.input(k, 0, n);
            for (dst, src) in out.chunks_exact_mut(4).zip(raw.chunks_exact(8)) {
                let v = i64::from_le_bytes(src.try_into().unwrap());
                dst.copy_from_slice(&(v as i32).to_le_bytes());
            }
        } else {
            shared.read_at(at, slot.input(k, 0, n)).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn attach<'a>(
    attached: &'a mut HashMap<String, Shared>,
    name: &str,
    needed: usize,
) -> Result<&'a mut Shared, String> {
    let stale = attached.get(name).is_none_or(|s| s.capacity() < needed);
    if stale {
        attached.remove(name);
        let block = Shared::attach(name, needed).ok_or(format!("no shared block named {name}"))?;
        attached.insert(name.to_string(), block);
    }
    Ok(attached.get_mut(name).unwrap())
}

/// `payoff.HP_SHARE` of one row, from the encoding (IKA-460 step 4): each side's remaining HP
/// (the live members' HP, each rounded as `encode.py` rounds it) over its total, side 0's share
/// of the two; 1/2 when neither has any. `mon` is the row's `[2][m][wm]` floats, `mask` its
/// `[2][m]`.
fn hp_share_row(mon: &[f32], mask: &[f32], m: usize, wm: usize) -> f64 {
    use crate::encode::{MON_FAINTED, MON_HP_FRACTION, MON_HP_SCALE, MON_MAXHP_SCALED};
    let mut remaining = [0.0f64; 2];
    for (side, share) in remaining.iter_mut().enumerate() {
        let (mut live, mut total) = (0.0f64, 0.0f64);
        for k in 0..m {
            if mask[side * m + k] <= 0.0 {
                continue;
            }
            let row = &mon[(side * m + k) * wm..(side * m + k + 1) * wm];
            let maxhp = (row[MON_MAXHP_SCALED] as f64 * MON_HP_SCALE).round_ties_even();
            let hp = (row[MON_HP_FRACTION] as f64 * maxhp).round_ties_even();
            total += maxhp;
            if row[MON_FAINTED] <= 0.0 {
                live += hp;
            }
        }
        *share = if total > 0.0 { live / total } else { 0.0 };
    }
    let both = remaining[0] + remaining[1];
    if both > 0.0 { remaining[0] / both } else { 0.5 }
}

/// Op `score` on an hp-share arm: no net, the answer is read off the shared block's `mon` and
/// `mask` arrays (rows of `<f4`), on the CPU.
fn score_hp_share(
    server: &Server,
    req: &Value,
    attached: &mut HashMap<String, Shared>,
) -> Result<Value, String> {
    let rows = req["rows"].as_u64().ok_or("`rows` is missing")? as usize;
    let result_offset = req["result_offset"].as_u64().ok_or("`result_offset` is missing")? as usize;
    let (items, needed) = layout_of(req)?;
    let needed = needed.max(result_offset + rows * 8);
    let shared = attach(attached, req["shm"].as_str().ok_or("`shm` is missing")?, needed)?;
    let (m, wm) = (server.geometry.m, server.geometry.wm);
    let (mon_item, mask_item) = (&items[4], &items[5]);
    if mon_item.dtype != "<f4" || mask_item.dtype != "<f4" {
        return Err(format!("hp-share reads `<f4` mon and mask, not {} and {}", mon_item.dtype, mask_item.dtype));
    }
    if mon_item.shape.iter().skip(1).product::<usize>() != 2 * m * wm {
        return Err(format!("`mon` has rows of {} values; this server's encoder has {}",
                           mon_item.shape.iter().skip(1).product::<usize>(), 2 * m * wm));
    }
    let mut mon_bytes = vec![0u8; rows * 2 * m * wm * 4];
    let mut mask_bytes = vec![0u8; rows * 2 * m * 4];
    shared.read_at(mon_item.offset, &mut mon_bytes).map_err(|e| e.to_string())?;
    shared.read_at(mask_item.offset, &mut mask_bytes).map_err(|e| e.to_string())?;
    let floats = |bytes: &[u8]| -> Vec<f32> {
        bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
    };
    let (mon, mask) = (floats(&mon_bytes), floats(&mask_bytes));
    let mut out = Vec::with_capacity(rows * 8);
    for r in 0..rows {
        let value = hp_share_row(
            &mon[r * 2 * m * wm..(r + 1) * 2 * m * wm],
            &mask[r * 2 * m..(r + 1) * 2 * m],
            m,
            wm,
        );
        out.extend_from_slice(&value.to_le_bytes());
    }
    shared.write_at(result_offset, &out).map_err(|e| e.to_string())?;
    server.stats.requests.fetch_add(1, Relaxed);
    server.stats.rows.fetch_add(rows as u64, Relaxed);
    Ok(json!({ "ok": true, "rows": rows }))
}

fn score(server: &Server, req: &Value, attached: &mut HashMap<String, Shared>) -> Result<Value, String> {
    let started = Instant::now();
    let name = req["model"].as_str().ok_or("`model` is missing")?;
    if server.hp_arms.iter().any(|a| a == name) {
        return score_hp_share(server, req, attached);
    }
    let arm = server
        .arms
        .iter()
        .position(|a| a.name == name)
        .ok_or(format!("no model named {name:?}; have {:?}", server.arms.iter().map(|a| &a.name).collect::<Vec<_>>()))?;
    let rows = req["rows"].as_u64().ok_or("`rows` is missing")? as usize;
    let result_offset = req["result_offset"].as_u64().ok_or("`result_offset` is missing")? as usize;
    let (items, needed) = layout_of(req)?;
    let needed = needed.max(result_offset + rows * 8);
    let shared = attach(attached, req["shm"].as_str().ok_or("`shm` is missing")?, needed)?;
    if rows == 0 {
        return Ok(json!({ "ok": true, "rows": 0 }));
    }
    let waiting = Instant::now();
    let mut slot = server.take_slot();
    server.stats.slot_wait_ns.fetch_add(waiting.elapsed().as_nanos() as u64, Relaxed);
    let done = (|| -> Result<(), String> {
        let mut first = 0;
        while first < rows {
            let n = (rows - first).min(MAX_ROWS);
            let copied = Instant::now();
            fill(&mut slot, shared, &items, first, n)?;
            server.stats.copy_ns.fetch_add(copied.elapsed().as_nanos() as u64, Relaxed);
            let on_card = Instant::now();
            let values = slot.score(arm, n)?;
            server.stats.gpu_ns.fetch_add(on_card.elapsed().as_nanos() as u64, Relaxed);
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            shared.write_at(result_offset + first * 8, &bytes).map_err(|e| e.to_string())?;
            first += n;
        }
        Ok(())
    })();
    server.give_slot(slot);
    done?;
    server.stats.requests.fetch_add(1, Relaxed);
    server.stats.rows.fetch_add(rows as u64, Relaxed);
    server.stats.handled_ns.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
    Ok(json!({ "ok": true, "rows": rows }))
}

/// Op "stages": the same request as `score` (at most MAX_ROWS rows) plus `stage_offset`;
/// the first net's every step is written there as float32, each at a 64-byte boundary.
fn stages(server: &Server, req: &Value, attached: &mut HashMap<String, Shared>) -> Result<Value, String> {
    let name = req["model"].as_str().ok_or("`model` is missing")?;
    let arm = server.arms.iter().find(|a| a.name == name).ok_or(format!("no model named {name:?}"))?;
    let rows = req["rows"].as_u64().ok_or("`rows` is missing")? as usize;
    if rows == 0 || rows > MAX_ROWS {
        return Err(format!("stages takes 1..={MAX_ROWS} rows"));
    }
    let mut at = req["stage_offset"].as_u64().ok_or("`stage_offset` is missing")? as usize;
    let (items, needed) = layout_of(req)?;
    let capacity = req["stage_bytes"].as_u64().unwrap_or(0) as usize;
    let shared = attach(attached, req["shm"].as_str().ok_or("`shm` is missing")?, needed.max(at + capacity))?;
    let mut slot = server.take_slot();
    let result = (|| {
        fill(&mut slot, shared, &items, 0, rows)?;
        slot.stages(&server.gpu, &arm.nets, rows)
    })();
    server.give_slot(slot);
    let mut listed = Vec::new();
    for (stage, width, values) in result? {
        at = (at + 63) & !63;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        shared.write_at(at, &bytes).map_err(|e| e.to_string())?;
        listed.push(json!({ "name": stage, "width": width, "offset": at, "count": values.len() }));
        at += bytes.len();
    }
    Ok(json!({ "ok": true, "stages": listed }))
}

fn handle(server: Arc<Server>, stream: TcpStream) {
    let _ = stream.set_nodelay(true);
    let Ok(writer_half) = stream.try_clone() else { return };
    let mut reader = BufReader::new(stream);
    let mut writer = writer_half;
    let mut attached: HashMap<String, Shared> = HashMap::new();
    let mut q: Option<(BufReader<TcpStream>, TcpStream)> = None;
    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(writer, "{}", error_reply(format!("bad request: {e}")));
                continue;
            }
        };
        let op = request["op"].as_str().unwrap_or("").to_string();
        let reply = match op.as_str() {
            "ping" => Ok(json!({ "ok": true, "models": server.arms.iter().map(|a| a.name.clone())
                .chain(server.hp_arms.iter().cloned()).collect::<Vec<_>>() })),
            "describe" => {
                let mut arms: serde_json::Map<String, Value> =
                    server.arms.iter().map(|a| (a.name.clone(), json!(a.files))).collect();
                for name in &server.hp_arms {
                    arms.insert(name.clone(), json!(["hp-share"]));
                }
                Ok(json!({ "ok": true, "arms": arms }))
            }
            "score" | "score_segments" => score(&server, &request, &mut attached),
            "stages" => stages(&server, &request, &mut attached),
            "q" | "q_many" | "q_batch" | "describe_q" if !server.q_arms.is_empty() => {
                q_request(&server, &request, &mut attached)
            }
            "q" | "q_many" | "q_batch" | "describe_q" => forward_q(&server, &mut q, line.trim_end()),
            other => Err(format!("unknown op {other:?}")),
        };
        let reply = reply.unwrap_or_else(error_reply);
        let mut text = reply.to_string();
        text.push('\n');
        if writer.write_all(text.as_bytes()).and_then(|_| writer.flush()).is_err() {
            break;
        }
    }
}

/// A Q request, line for line, to the Python server and its answer back.
fn forward_q(
    server: &Server,
    link: &mut Option<(BufReader<TcpStream>, TcpStream)>,
    line: &str,
) -> Result<Value, String> {
    let Some(address) = &server.q_forward else {
        return Err("this server has no Q arm (start it with --q-forward ADDRESS)".into());
    };
    if link.is_none() {
        let stream = TcpStream::connect(address).map_err(|e| format!("Q server at {address}: {e}"))?;
        let _ = stream.set_nodelay(true);
        let writer = stream.try_clone().map_err(|e| e.to_string())?;
        *link = Some((BufReader::new(stream), writer));
    }
    let (reader, writer) = link.as_mut().unwrap();
    let outcome = (|| {
        writer.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
        writer.write_all(b"\n").map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
        let mut reply = String::new();
        if reader.read_line(&mut reply).map_err(|e| e.to_string())? == 0 {
            return Err("the Q server closed the connection".to_string());
        }
        serde_json::from_str::<Value>(reply.trim()).map_err(|e| e.to_string())
    })();
    if outcome.is_err() {
        *link = None;
    } else {
        server.stats.q_forwarded.fetch_add(1, Relaxed);
    }
    outcome
}

#[cfg(windows)]
fn process_cpu_seconds() -> f64 {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn GetProcessTimes(p: *mut std::ffi::c_void, c: *mut FileTime, e: *mut FileTime, k: *mut FileTime, u: *mut FileTime) -> i32;
    }
    let (mut c, mut e, mut k, mut u) = (FileTime::default(), FileTime::default(), FileTime::default(), FileTime::default());
    // SAFETY: the pseudo-handle is always valid and the four structs are ours.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
    let ticks = |t: &FileTime| ((t.high as u64) << 32 | t.low as u64) as f64 * 1e-7;
    ticks(&k) + ticks(&u)
}

#[cfg(not(windows))]
fn process_cpu_seconds() -> f64 {
    0.0
}

fn usage() -> ! {
    eprintln!(
        "usage: inference-server --regulation <regulation.json> --arm NAME FILE.rnet [FILE.rnet ...] \
         [--arm ...] [--hp-share-arm NAME] [--q-forward HOST:PORT] [--host H] [--port P] [--slots N] [--dll-dir DIR] \
         [--report-every SECONDS]"
    );
    std::process::exit(2);
}

pub fn main(args: &[String]) {
    let mut regulation = None;
    let mut arm_specs: Vec<(String, Vec<String>)> = Vec::new();
    let mut hp_specs: Vec<String> = Vec::new();
    // POKEURAOU_RUST_SERVER_REPORT: seconds between the report lines, for a run that is shorter
    // than a minute (a launcher does not pass --report-every).
    let every = std::env::var("POKEURAOU_RUST_SERVER_REPORT").ok().and_then(|s| s.parse().ok());
    let (mut host, mut port, mut slots, mut report) =
        ("127.0.0.1".to_string(), 0u16, 4usize, every.unwrap_or(60.0f64));
    let mut q_forward = None;
    let mut q_specs: Vec<(String, String)> = Vec::new();
    let mut dll_dir = None;
    let mut i = 0;
    let value = |i: usize| args.get(i + 1).cloned().unwrap_or_else(|| usage());
    while i < args.len() {
        match args[i].as_str() {
            "--regulation" => {
                regulation = Some(value(i));
                i += 2;
            }
            "--arm" => {
                let name = value(i);
                let mut files = Vec::new();
                i += 2;
                while i < args.len() && !args[i].starts_with("--") {
                    files.push(args[i].clone());
                    i += 1;
                }
                if files.is_empty() {
                    usage();
                }
                arm_specs.push((name, files));
            }
            "--hp-share-arm" => {
                hp_specs.push(value(i));
                i += 2;
            }
            "--q-forward" => {
                q_forward = Some(value(i));
                i += 2;
            }
            "--q-arm" => {
                let file = args.get(i + 2).cloned().unwrap_or_else(|| usage());
                q_specs.push((value(i), file));
                i += 3;
            }
            "--host" => {
                host = value(i);
                i += 2;
            }
            "--port" => {
                port = value(i).parse().unwrap_or_else(|_| usage());
                i += 2;
            }
            "--slots" => {
                slots = value(i).parse().unwrap_or_else(|_| usage());
                i += 2;
            }
            "--dll-dir" => {
                dll_dir = Some(value(i));
                i += 2;
            }
            "--report-every" => {
                report = value(i).parse().unwrap_or_else(|_| usage());
                i += 2;
            }
            _ => usage(),
        }
    }
    let (Some(regulation), false) = (regulation, arm_specs.is_empty() && hp_specs.is_empty()) else { usage() };
    // Where CUDA 12's DLLs are: --dll-dir, else POKEURAOU_CUDA_DLLS, else the bin folder of an
    // installed CUDA 12 Toolkit (`CUDA_PATH_V12_*`), else whatever PATH already holds.
    let toolkit = || {
        let named = std::env::vars()
            .filter(|(k, _)| k.starts_with("CUDA_PATH_V12"))
            .map(|(_, v)| format!("{v}{}bin", std::path::MAIN_SEPARATOR))
            .next();
        // A shell opened before the Toolkit was installed has no such variable: look where
        // its installer puts it.
        named.or_else(|| {
            let root = std::path::Path::new(&std::env::var("ProgramFiles").ok()?)
                .join("NVIDIA GPU Computing Toolkit")
                .join("CUDA");
            let mut found: Vec<_> = std::fs::read_dir(root)
                .ok()?
                .flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("v12"))
                .map(|e| e.path().join("bin"))
                .filter(|p| p.is_dir())
                .collect();
            found.sort();
            found.pop().map(|p| p.to_string_lossy().into_owned())
        })
    };
    if let Some(dir) = dll_dir.or_else(|| std::env::var("POKEURAOU_CUDA_DLLS").ok()).or_else(toolkit) {
        let path = std::env::var("PATH").unwrap_or_default();
        let sep = if cfg!(windows) { ";" } else { ":" };
        std::env::set_var("PATH", format!("{dir}{sep}{path}"));
    }
    let started = Instant::now();
    let reg = Reg::load(&regulation).unwrap_or_else(|e| {
        eprintln!("regulation: {e}");
        std::process::exit(1);
    });
    let widths = Encoder::new(&reg).widths;
    let geom = Geometry { m: widths.mons_per_side, wm: widths.mon, ws: widths.side, wf: widths.field };
    // cudarc panics when a library is not found; say what to do about it instead.
    let gpu = std::panic::catch_unwind(|| Gpu::new(0)).unwrap_or_else(|_| {
        Err("CUDA 12's DLLs (nvrtc64_120_0, cublas64_12) were not found: pass --dll-dir, or set \
             POKEURAOU_CUDA_DLLS, or install the CUDA 12 Toolkit"
            .to_string())
    });
    let gpu = gpu.unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let v = &gpu.versions;
    eprintln!(
        "  GPU {} (compute {}.{}), driver {}.{}, NVRTC {}.{}, cuBLAS {}.{}.{}",
        v.device,
        v.capability.0,
        v.capability.1,
        v.driver / 1000,
        (v.driver % 1000) / 10,
        v.nvrtc.0,
        v.nvrtc.1,
        v.cublas / 10000,
        (v.cublas % 10000) / 100,
        v.cublas % 100
    );
    let stream = gpu.ctx.new_stream().expect("a stream");
    let mut arms = Vec::new();
    let mut widest = (0usize, 0usize, 0usize);
    for (name, files) in &arm_specs {
        let mut nets = Vec::new();
        for file in files {
            let net = rnet::read(file).and_then(|r| Net::upload(&stream, &r)).unwrap_or_else(|e| {
                eprintln!("arm {name}: {e}");
                std::process::exit(1);
            });
            let w = Slot::widths_of(&net);
            widest = (widest.0.max(w.0), widest.1.max(w.1), widest.2.max(w.2));
            nets.push(net);
        }
        let sources = nets.iter().map(|n| n.source.clone()).collect();
        arms.push(Arm { name: name.clone(), files: sources, nets });
    }
    stream.synchronize().expect("weights uploaded");
    let mut pool = Vec::new();
    // A server of hp-share arms alone has no net to score on the card: no slots.
    for _ in 0..(if arms.is_empty() { 0 } else { slots }) {
        let mut slot = Slot::new(&gpu, geom, arms.len(), widest).unwrap_or_else(|e| {
            eprintln!("slot: {e}");
            std::process::exit(1);
        });
        for arm in &arms {
            slot.capture_arm(&gpu, &arm.nets).unwrap_or_else(|e| {
                eprintln!("capture: {e}");
                std::process::exit(1);
            });
        }
        pool.push(slot);
    }
    // Q arms (--q-arm NAME FILE.rnet): the net on the card, and slots of their own.
    let mut q_arms: Vec<QArm> = Vec::new();
    for (name, file) in &q_specs {
        let net = rnet::read(file).and_then(|r| QNet::upload(&stream, &r)).unwrap_or_else(|e| {
            eprintln!("Q arm {name}: {e}");
            std::process::exit(1);
        });
        let shown = Path::new(file).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        q_arms.push(QArm { name: name.clone(), file: shown, net });
    }
    stream.synchronize().expect("Q weights uploaded");
    let mut q_pool = Vec::new();
    if let Some(first) = q_arms.first() {
        for _ in 0..slots {
            q_pool.push(QSlot::new(&gpu, geom, &first.net).unwrap_or_else(|e| {
                eprintln!("Q slot: {e}");
                std::process::exit(1);
            }));
        }
        for arm in &q_arms {
            if arm.net.widths() != first.net.widths() {
                eprintln!("Q arms {} and {} differ in width; one server holds Q arms of one shape", first.name, arm.name);
                std::process::exit(1);
            }
        }
    }
    let server = Arc::new(Server {
        gpu,
        arms,
        q_arms,
        hp_arms: hp_specs,
        q_slots: Mutex::new(q_pool),
        q_free: Condvar::new(),
        geometry: geom,
        slots: Mutex::new(pool),
        free: Condvar::new(),
        q_forward,
        stats: Stats::default(),
    });
    let listener = TcpListener::bind((host.as_str(), port)).unwrap_or_else(|e| {
        eprintln!("bind: {e}");
        std::process::exit(1);
    });
    let address = listener.local_addr().expect("address");
    // First line of stdout, as the Python server does.
    println!("{address}");
    let _ = std::io::stdout().flush();
    for arm in &server.arms {
        eprintln!(
            "  arm {}: {}{}",
            arm.name,
            arm.files.join(", "),
            if arm.nets.len() > 1 { " (ensemble, logits averaged)" } else { "" }
        );
    }
    for name in &server.hp_arms {
        eprintln!("  arm {name}: hp-share (read off the encoding, no net)");
    }
    for arm in &server.q_arms {
        eprintln!("  Q arm {}: {} (sha256 {})", arm.name, arm.file, &arm.net.sha256.chars().take(12).collect::<String>());
    }
    eprintln!(
        "  {} slots x {} graphs a net-set, ready in {:.1} s; Q requests {}",
        slots,
        infer_gpu::BUCKETS,
        started.elapsed().as_secs_f64(),
        if !server.q_arms.is_empty() {
            "answered here".to_string()
        } else {
            server.q_forward.as_deref().map(|a| format!("forwarded to {a}")).unwrap_or("refused".into())
        }
    );
    if report > 0.0 {
        let server = server.clone();
        std::thread::spawn(move || {
            let mut last = (0u64, 0u64, 0.0f64);
            loop {
                std::thread::sleep(Duration::from_secs_f64(report));
                let s = &server.stats;
                let (req, rows) = (s.requests.load(Relaxed), s.rows.load(Relaxed));
                let cpu = process_cpu_seconds();
                let per = |ns: &AtomicU64| ns.load(Relaxed) as f64 / 1e6 / req.max(1) as f64;
                eprintln!(
                    "  {req} requests, {rows} rows ({} since the last line); per request {:.2} ms in all, \
                     {:.2} ms waiting for a slot, {:.2} ms copying, {:.2} ms on the card; Q items {}; \
                     process CPU {:.1} s (+{:.1})",
                    req - last.0,
                    per(&s.handled_ns),
                    per(&s.slot_wait_ns),
                    per(&s.copy_ns),
                    per(&s.gpu_ns),
                    s.q_forwarded.load(Relaxed) + s.q_served.load(Relaxed),
                    cpu,
                    cpu - last.2
                );
                last = (req, rows, cpu);
            }
        });
    }
    for stream in listener.incoming().flatten() {
        let server = server.clone();
        std::thread::spawn(move || handle(server, stream));
    }
}

#[cfg(test)]
mod tests {
    use super::hp_share_row;
    use crate::encode::{MON_FAINTED, MON_HP_FRACTION, MON_HP_SCALE, MON_MAXHP_SCALED};

    const WM: usize = MON_FAINTED + 2;

    /// One row of `m` members a side: `(hp fraction, max HP, fainted)` per member.
    fn row(sides: [&[(f32, f64, bool)]; 2], m: usize) -> (Vec<f32>, Vec<f32>) {
        let mut mon = vec![0.0f32; 2 * m * WM];
        let mut mask = vec![0.0f32; 2 * m];
        for (s, members) in sides.iter().enumerate() {
            for (k, (fraction, maxhp, fainted)) in members.iter().enumerate() {
                let at = (s * m + k) * WM;
                mon[at + MON_HP_FRACTION] = *fraction;
                mon[at + MON_MAXHP_SCALED] = (*maxhp / MON_HP_SCALE) as f32;
                mon[at + MON_FAINTED] = if *fainted { 1.0 } else { 0.0 };
                mask[s * m + k] = 1.0;
            }
        }
        (mon, mask)
    }

    #[test]
    fn the_share_is_the_live_hp_over_the_total_hp_of_each_side() {
        // Side 0: 100 of 100 and a fainted 200; side 1: half of 200 and 100 of 100.
        let (mon, mask) = row(
            [&[(1.0, 100.0, false), (0.0, 200.0, true)], &[(0.5, 200.0, false), (1.0, 100.0, false)]],
            4,
        );
        let got = hp_share_row(&mon, &mask, 4, WM);
        let (a, b) = (100.0 / 300.0, 200.0 / 300.0);
        assert!((got - a / (a + b)).abs() < 1e-12, "{got}");
        // The control that can fail: the fainted member's HP does not count, a live one's does.
        let (mon, mask) = row([&[(1.0, 100.0, false)], &[(1.0, 100.0, false)]], 4);
        assert_eq!(hp_share_row(&mon, &mask, 4, WM), 0.5);
        let (mon, mask) = row([&[(0.5, 100.0, false)], &[(1.0, 100.0, false)]], 4);
        assert!(hp_share_row(&mon, &mask, 4, WM) < 0.5);
    }

    #[test]
    fn nothing_alive_on_either_side_is_one_half() {
        let (mon, mask) = row([&[(0.0, 100.0, true)], &[(0.0, 100.0, true)]], 4);
        assert_eq!(hp_share_row(&mon, &mask, 4, WM), 0.5);
    }
}
