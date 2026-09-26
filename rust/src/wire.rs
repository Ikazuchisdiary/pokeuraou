//! What the node's wire costs this process (IKA-302), kept apart from what it resolves.
//!
//! Four clocks, because a crossing is four different pieces of work with four different
//! fixes: reading a request's text into a `Value`, building a `Position` out of one, writing
//! a `Position` back into a `Value`, and writing an answer's `Value` out as text. The first
//! and the last are kept per request kind, with the bytes each way, so a table can rank the
//! crossings by what they carry. Written next to the cell threads' report when the process
//! ends (`POKEURAOU_PORT_THREADS_REPORT`), as `wire-<pid>.json`; nothing without it.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Nanoseconds and calls of one clock.
pub struct Clock {
    ns: AtomicU64,
    calls: AtomicU64,
}

impl Clock {
    const fn new() -> Clock {
        Clock { ns: AtomicU64::new(0), calls: AtomicU64::new(0) }
    }

    pub fn add(&self, started: Instant) {
        self.ns.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
    }

    fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "s": self.ns.load(Ordering::Relaxed) as f64 / 1e9,
            "calls": self.calls.load(Ordering::Relaxed),
        })
    }
}

/// `Position::from_json`, wherever it is called (on the cell threads too).
pub static POSITION_READ: Clock = Clock::new();
/// `Position::to_json`, wherever it is called.
pub static POSITION_WRITE: Clock = Clock::new();

/// Per request kind: [lines, text-read ns, bytes in, text-write ns, bytes out, answer ns].
static KINDS: Mutex<BTreeMap<String, [u64; 6]>> = Mutex::new(BTreeMap::new());

/// One request's crossing: its kind, the text it came as and what reading that cost, the
/// answer's text and what writing it cost, and the whole answer's time.
pub fn note(kind: &str, read_ns: u64, bytes_in: usize, write_ns: u64, bytes_out: usize, answer_ns: u64) {
    if let Ok(mut kinds) = KINDS.lock() {
        let row = kinds.entry(kind.to_string()).or_insert([0; 6]);
        row[0] += 1;
        row[1] += read_ns;
        row[2] += bytes_in as u64;
        row[3] += write_ns;
        row[4] += bytes_out as u64;
        row[5] += answer_ns;
    }
}

/// The request being answered on this thread: kind, read ns, bytes in, write ns, bytes out.
struct Crossing {
    kind: String,
    read_ns: u64,
    bytes_in: usize,
    write_ns: u64,
    bytes_out: usize,
}

thread_local! {
    static CROSSING: std::cell::RefCell<Option<Crossing>> = const { std::cell::RefCell::new(None) };
}

/// A request has been read: what it is, and what reading its text cost.
pub fn begin(kind: &str, read_ns: u64, bytes_in: usize) {
    CROSSING.with(|c| {
        *c.borrow_mut() = Some(Crossing {
            kind: kind.to_string(),
            read_ns,
            bytes_in,
            write_ns: 0,
            bytes_out: 0,
        })
    });
}

/// Text written for the request being answered (a header, or a whole answer).
pub fn wrote(started: Instant, bytes: usize) {
    let ns = started.elapsed().as_nanos() as u64;
    CROSSING.with(|c| {
        if let Some(crossing) = c.borrow_mut().as_mut() {
            crossing.write_ns += ns;
            crossing.bytes_out += bytes;
        }
    });
}

/// The request is answered; `started` is when its line was in hand.
pub fn end(started: Instant) {
    let answer_ns = started.elapsed().as_nanos() as u64;
    CROSSING.with(|c| {
        if let Some(x) = c.borrow_mut().take() {
            note(&x.kind, x.read_ns, x.bytes_in, x.write_ns, x.bytes_out, answer_ns);
        }
    });
}

pub fn report() -> serde_json::Value {
    let kinds = KINDS.lock().map(|k| k.clone()).unwrap_or_default();
    serde_json::json!({
        "positionRead": POSITION_READ.json(),
        "positionWrite": POSITION_WRITE.json(),
        "kinds": kinds.iter().map(|(kind, row)| (kind.clone(), serde_json::json!({
            "lines": row[0],
            "readS": row[1] as f64 / 1e9,
            "bytesIn": row[2],
            "writeS": row[3] as f64 / 1e9,
            "bytesOut": row[4],
            "answerS": row[5] as f64 / 1e9,
        }))).collect::<serde_json::Map<_, _>>(),
    })
}
