//! IKA-386: the port asks the inference server itself.
//!
//! A depth-2 read's sub-games went port -> Python -> server -> Python -> port: the port
//! filled and encoded each node (`fills`), Python copied the arrays into its shared block
//! and asked the server (`inference.RemoteValue.from_encoded`), and Python sent the span
//! blocks and the values back to the port to fold and solve (`folds`, IKA-381). Here the
//! port writes the same arrays into the same block and sends the server the same request
//! on a connection of its own, so the leaves never cross into Python at all.
//!
//! The request is `RemoteValue.from_encoded`'s: `op` "score", the arm's name, the block's
//! name, the rows, each array's place in the block (64-byte aligned, in `inference.ARRAYS`'
//! order, with numpy's dtype text), and where the answer goes. A batch longer than
//! `batch` (`inference.CHUNK_ROWS`) is cut where `RemoteValue` cuts it. The server answers a
//! request by its row count, so the same rows in the same requests are the same values.
//!
//! The block is the Python leaf's own (`RemoteValue._block`): Python does not touch it while
//! the port is answering, and the server has it attached already.

use crate::encode::Encoded;
use crate::shm::Shared;
use serde_json::{json, Value};
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Instant;

/// Where a request goes: the server, the arm, the block, the chunk and the merged road.
pub struct Server {
    pub address: String,
    pub model: String,
    pub shm: String,
    pub capacity: usize,
    pub batch: usize,
    pub merge: bool,
}

impl Server {
    pub fn from_json(value: &Value) -> Result<Server, String> {
        let text = |key: &str| {
            value.get(key).and_then(Value::as_str).map(String::from).ok_or(format!("`server` without `{key}`"))
        };
        Ok(Server {
            address: text("address")?,
            model: text("model")?,
            shm: text("shm")?,
            capacity: value.get("bytes").and_then(Value::as_u64).ok_or("`server` without `bytes`")? as usize,
            batch: value.get("batch").and_then(Value::as_u64).unwrap_or(8192).max(1) as usize,
            merge: value.get("merge").and_then(Value::as_bool).unwrap_or(false),
        })
    }
}

/// What the server was asked and what it cost, for the caller's counters.
#[derive(Default)]
pub struct Stats {
    pub requests: u64,
    pub rows: u64,
    /// Nanoseconds blocked on an answer (`RemoteValue.waited`). A batch's last request is
    /// answered while the port fills and solves, so this is the part of the server's time the
    /// port could not hide, not all of it.
    pub wait_ns: u64,
    /// Nanoseconds laying the arrays into the block (`RemoteValue.copied`).
    pub copy_ns: u64,
}

/// Why a batch has no values: the server said no (`ok: false`, perhaps out of memory), or
/// the connection or the block failed.
pub enum Failed {
    Refused { error: String, oom: bool, cap_gb: Option<f64> },
    Broken(String),
}

pub struct Client {
    address: String,
    shm: String,
    capacity: usize,
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    pub block: Shared,
}

thread_local! {
    /// One connection and one attachment, kept while the caller names the same ones.
    static CLIENT: RefCell<Option<Client>> = const { RefCell::new(None) };
    /// IKA-389: the same for the Q's block (`qrank.RemoteQ._block`), a connection of its own
    /// so the leaf's and the Q's do not replace each other.
    static Q_CLIENT: RefCell<Option<Client>> = const { RefCell::new(None) };
}

fn connect(server: &Server) -> Result<Client, Failed> {
    let stream = TcpStream::connect(&server.address)
        .map_err(|e| Failed::Broken(format!("the inference server at {}: {e}", server.address)))?;
    let _ = stream.set_nodelay(true);
    let writer = stream.try_clone().map_err(|e| Failed::Broken(e.to_string()))?;
    let block = Shared::attach(&server.shm, server.capacity)
        .ok_or_else(|| Failed::Broken(format!("could not attach the leaf's block {}", server.shm)))?;
    Ok(Client {
        address: server.address.clone(),
        shm: server.shm.clone(),
        capacity: server.capacity,
        reader: BufReader::new(stream),
        writer,
        block,
    })
}

/// The arrays of `inference.ARRAYS`, in order: name, numpy's dtype text, and the shape of a row.
fn fields(widths: &crate::encode::Widths) -> [(&'static str, &'static str, Vec<usize>); 8] {
    let m = widths.mons_per_side;
    [
        ("species", "<i4", vec![2, m]),
        ("ability", "<i4", vec![2, m]),
        ("item", "<i4", vec![2, m]),
        ("moves", "<i4", vec![2, m, 4]),
        ("mon", "<f4", vec![2, m, widths.mon]),
        ("mask", "<f4", vec![2, m]),
        ("side", "<f4", vec![2, widths.side]),
        ("field", "<f4", vec![widths.field]),
    ]
}

/// The bytes of field `k` of a block (little-endian, as `write_body` writes them).
fn field_bytes(encoded: &Encoded, k: usize) -> &[u8] {
    #[cfg(not(target_endian = "little"))]
    compile_error!("the served road writes the arrays as they lie in memory");
    fn bytes<T>(v: &[T]) -> &[u8] {
        // SAFETY: i32 and f32 have no padding, and every byte pattern of them is readable.
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }
    match k {
        0 => bytes(&encoded.species),
        1 => bytes(&encoded.ability),
        2 => bytes(&encoded.item),
        3 => bytes(&encoded.moves),
        4 => bytes(&encoded.mon),
        5 => bytes(&encoded.mask),
        6 => bytes(&encoded.side),
        _ => bytes(&encoded.field),
    }
}

fn align(offset: usize) -> usize {
    (offset + 63) & !63
}

/// A batch sent (`begin`) whose last request's answer has not been read yet (`finish`).
pub struct Batch {
    values: Vec<f64>,
    /// The last request's rows and where its answer lies in the block; None when every
    /// answer is in (an empty batch).
    last: Option<(usize, usize, usize)>,
}

fn with_client<T>(
    server: &Server,
    call: impl FnOnce(&mut Client) -> Result<T, Failed>,
) -> Result<T, Failed> {
    with_client_of(&CLIENT, server, call)
}

/// IKA-389: `with_client` on the Q's own connection.
pub fn with_q_client<T>(
    server: &Server,
    call: impl FnOnce(&mut Client) -> Result<T, Failed>,
) -> Result<T, Failed> {
    with_client_of(&Q_CLIENT, server, call)
}

/// One request line to the server and its reply, `ok` or the server's refusal.
pub fn ask(client: &mut Client, request: &Value, stats: &mut Stats) -> Result<Value, Failed> {
    let broken = |e: std::io::Error| Failed::Broken(e.to_string());
    let mut line = request.to_string();
    line.push('\n');
    client.writer.write_all(line.as_bytes()).map_err(broken)?;
    client.writer.flush().map_err(broken)?;
    stats.requests += 1;
    let waited = Instant::now();
    let mut reply = String::new();
    if client.reader.read_line(&mut reply).map_err(broken)? == 0 {
        return Err(Failed::Broken("the inference server closed the connection".into()));
    }
    stats.wait_ns += waited.elapsed().as_nanos() as u64;
    let reply: Value = serde_json::from_str(reply.trim())
        .map_err(|e| Failed::Broken(format!("the inference server's reply: {e}")))?;
    if reply.get("ok").and_then(Value::as_bool) != Some(true) {
        return Err(Failed::Refused {
            error: reply
                .get("error")
                .map(|e| e.as_str().map(String::from).unwrap_or(e.to_string()))
                .unwrap_or_default(),
            oom: reply.get("oom").and_then(Value::as_bool).unwrap_or(false),
            cap_gb: reply.get("capGb").and_then(Value::as_f64),
        });
    }
    Ok(reply)
}

fn with_client_of<T>(
    key: &'static std::thread::LocalKey<RefCell<Option<Client>>>,
    server: &Server,
    call: impl FnOnce(&mut Client) -> Result<T, Failed>,
) -> Result<T, Failed> {
    key.with(|cell| {
        let mut held = cell.borrow_mut();
        let stale = held.as_ref().is_none_or(|c| {
            c.address != server.address || c.shm != server.shm || c.capacity != server.capacity
        });
        if stale {
            *held = None;
            *held = Some(connect(server)?);
        }
        let answered = call(held.as_mut().expect("connected just above"));
        if let Err(Failed::Broken(_)) = &answered {
            // A connection in an unknown state is not used again.
            *held = None;
        }
        answered
    })
}

/// Every row of `blocks`, one after another (`port.score_stacked`'s one batch), sent to the
/// server in requests of at most `server.batch` rows, in order. Every request's answer but
/// the last is read here; the last is left with the server (`finish` reads it), so the
/// caller can fill and solve while the server scores. `rows[b]` is block b's row count. The
/// blocks' arrays are in the block by the time this returns and are not needed again.
pub fn begin(
    server: &Server,
    widths: &crate::encode::Widths,
    blocks: &[&Encoded],
    rows: &[usize],
    stats: &mut Stats,
) -> Result<Batch, Failed> {
    let total: usize = rows.iter().sum();
    let mut batch = Batch { values: vec![0.0f64; total], last: None };
    if total == 0 {
        return Ok(batch);
    }
    with_client(server, |client| {
        let mut start = 0;
        while start < total {
            let stop = (start + server.batch).min(total);
            let result_offset = send(client, server, widths, blocks, rows, start, stop, stats)?;
            if stop < total {
                receive(client, &mut batch.values[start..stop], result_offset, stats)?;
            } else {
                batch.last = Some((start, stop, result_offset));
            }
            start = stop;
        }
        Ok(())
    })?;
    Ok(batch)
}

/// The batch's values, once the last request's answer is read.
pub fn finish(server: &Server, mut batch: Batch, stats: &mut Stats) -> Result<Vec<f64>, Failed> {
    if let Some((start, stop, result_offset)) = batch.last.take() {
        with_client(server, |client| {
            receive(client, &mut batch.values[start..stop], result_offset, stats)
        })?;
    }
    Ok(batch.values)
}

/// Rows [start, stop) of the batch laid into the block and asked for; where the answer goes.
#[allow(clippy::too_many_arguments)]
fn send(
    client: &mut Client,
    server: &Server,
    widths: &crate::encode::Widths,
    blocks: &[&Encoded],
    rows: &[usize],
    start: usize,
    stop: usize,
    stats: &mut Stats,
) -> Result<usize, Failed> {
    let fields = fields(widths);
    let per_row: Vec<usize> = fields.iter().map(|(_, _, shape)| shape.iter().product::<usize>() * 4).collect();
    let broken = |e: std::io::Error| Failed::Broken(e.to_string());
    let n = stop - start;
    let copied = Instant::now();
    let mut layout = Vec::with_capacity(fields.len());
    let mut offset = 0usize;
    for (k, (name, dtype, shape)) in fields.iter().enumerate() {
        offset = align(offset);
        let nbytes = n * per_row[k];
        let mut shape_all = vec![n];
        shape_all.extend_from_slice(shape);
        layout.push(json!({
            "name": name, "offset": offset, "shape": shape_all, "dtype": dtype, "nbytes": nbytes,
        }));
        offset += nbytes;
    }
    let result_offset = align(offset);
    let needed = result_offset + n * 8;
    if needed > client.capacity {
        return Err(Failed::Refused {
            error: format!(
                "one chunk of {n} rows needs {:.0} MB and the buffer is {:.0} MB",
                needed as f64 / 1e6,
                client.capacity as f64 / 1e6
            ),
            oom: false,
            cap_gb: None,
        });
    }
    // The rows [start, stop) of every block, field by field.
    for (k, item) in layout.iter().enumerate() {
        let mut at = item["offset"].as_u64().unwrap() as usize;
        let mut first = 0usize;
        for (block, &count) in blocks.iter().zip(rows) {
            let (lo, hi) = (start.max(first), stop.min(first + count));
            if lo < hi {
                let bytes = field_bytes(block, k);
                let piece = &bytes[(lo - first) * per_row[k]..(hi - first) * per_row[k]];
                client.block.write_at(at, piece).map_err(broken)?;
                at += piece.len();
            }
            first += count;
        }
    }
    stats.copy_ns += copied.elapsed().as_nanos() as u64;
    let mut request = json!({
        "op": "score",
        "model": server.model,
        "shm": server.shm,
        "rows": n,
        "layout": layout,
        "result_offset": result_offset,
    });
    if server.merge {
        request["merge"] = json!(true);
    }
    let mut line = request.to_string();
    line.push('\n');
    client.writer.write_all(line.as_bytes()).map_err(broken)?;
    client.writer.flush().map_err(broken)?;
    stats.requests += 1;
    stats.rows += n as u64;
    Ok(result_offset)
}

/// A request's answer: the reply line, then its values out of the block.
fn receive(client: &mut Client, out: &mut [f64], result_offset: usize, stats: &mut Stats) -> Result<(), Failed> {
    let broken = |e: std::io::Error| Failed::Broken(e.to_string());
    let waited = Instant::now();
    let mut reply = String::new();
    if client.reader.read_line(&mut reply).map_err(broken)? == 0 {
        return Err(Failed::Broken("the inference server closed the connection".into()));
    }
    stats.wait_ns += waited.elapsed().as_nanos() as u64;
    let reply: Value = serde_json::from_str(reply.trim())
        .map_err(|e| Failed::Broken(format!("the inference server's reply: {e}")))?;
    if reply.get("ok").and_then(Value::as_bool) != Some(true) {
        return Err(Failed::Refused {
            error: reply
                .get("error")
                .map(|e| e.as_str().map(String::from).unwrap_or(e.to_string()))
                .unwrap_or_default(),
            oom: reply.get("oom").and_then(Value::as_bool).unwrap_or(false),
            cap_gb: reply.get("capGb").and_then(Value::as_f64),
        });
    }
    let mut raw = vec![0u8; out.len() * 8];
    client.block.read_at(result_offset, &mut raw).map_err(broken)?;
    for (slot, bytes) in out.iter_mut().zip(raw.chunks_exact(8)) {
        *slot = f64::from_le_bytes(bytes.try_into().unwrap());
    }
    Ok(())
}
