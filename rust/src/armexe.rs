//! IKA-458 stage 3.7: an arm of a board read through another executable (`--rust-binary`, IKA-413).
//!
//! A board that measures a port change plays one arm's move-node search -- its menus and its node --
//! through the executable before the change. The Python worker sent those requests to a second
//! port process of that executable. The Rust worker does the same: the first time an arm names an
//! executable, this process starts it as a child (`<exe> node <regulation>`) and keeps it for the
//! run; each of that arm's move nodes is one `moveNode` line to it and one line back. The child holds
//! none of this process's positions, so the request writes them out (`playgame.rs`'s `plain`), and it
//! scores on the same inference server through the same shared block, which this process leaves idle
//! while it waits. The game itself, the replacements and the mid-turn choices stay with this process.

use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct Arm {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl Drop for Arm {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

thread_local! {
    static ARMS: RefCell<HashMap<String, Arm>> = RefCell::new(HashMap::new());
}

/// The requests answered by another executable, the positive control that an arm reached it.
pub static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn start(exe: &str, regulation: &str) -> Result<Arm, String> {
    let mut child = Command::new(exe)
        .args(["node", regulation])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("could not start it: {e}"))?;
    let input = child.stdin.take().ok_or("no stdin")?;
    let output = BufReader::new(child.stdout.take().ok_or("no stdout")?);
    Ok(Arm { child, input, output })
}

/// One request to `exe`'s node process and its answer.
pub(crate) fn ask(exe: &str, regulation: &str, request: &Value) -> Result<Value, String> {
    ARMS.with(|arms| {
        let mut arms = arms.borrow_mut();
        if !arms.contains_key(exe) {
            arms.insert(exe.to_string(), start(exe, regulation)?);
        }
        let arm = arms.get_mut(exe).expect("inserted above");
        let mut line = request.to_string();
        line.push('\n');
        let answered = (|| -> Result<Value, String> {
            arm.input.write_all(line.as_bytes()).map_err(|e| e.to_string())?;
            arm.input.flush().map_err(|e| e.to_string())?;
            let mut reply = String::new();
            if arm.output.read_line(&mut reply).map_err(|e| e.to_string())? == 0 {
                return Err("the process closed its output".into());
            }
            serde_json::from_str(reply.trim()).map_err(|e| format!("its answer: {e}"))
        })();
        if answered.is_err() {
            arms.remove(exe);
        }
        ASKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        answered
    })
}

/// The executable's path and size, for the worker's echo (the record carries its sha256).
pub(crate) fn fingerprint(exe: &str) -> String {
    match std::fs::metadata(exe) {
        Ok(meta) => format!("{exe} ({} bytes)", meta.len()),
        Err(e) => format!("{exe} ({e})"),
    }
}
