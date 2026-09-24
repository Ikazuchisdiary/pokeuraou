//! One whole node -- a matrix of cells -- resolved and scored in one crossing.
//!
//! `batched_payoffs` is already the project's node-level boundary, so it is the right
//! place to put a process boundary too: a position and two lists of actions go in, a
//! payoff matrix comes out, and the leaves never cross. A cell this port refuses is named
//! in `refused` and the caller fills it in Python, which is what makes a partial port
//! usable rather than merely measurable.
//!
//! The protocol is JSONL over stdio, the same shape the sim-bridge already uses: one
//! request per line, one response per line, and the process stays warm. An encoded node's
//! arrays are the exception: they are megabytes rather than kilobytes, so when the request
//! names a block of shared memory they go there instead and only the header line crosses
//! the pipe. The header says which road they took.

use crate::objective;
use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::{parse_actions_list, resolve_turn, turn_value, Budget, SlotAction};
use crate::shm;
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::io::{BufRead, Write};

pub struct Request {
    pub position: Position,
    pub ours: Vec<Vec<SlotAction>>,
    pub theirs: Vec<Vec<SlotAction>>,
    pub budget: Budget,
    pub objectives: Vec<String>,
    /// Return the encoder's arrays and a fold instead of payoffs, for a learned leaf.
    pub encode: bool,
    /// Which cells to fill, or all of them.
    ///
    /// An equilibrium does not need the whole matrix -- an unplayed action only has to be
    /// shown not to beat the opponent's mixed strategy, and that strategy sits on a few
    /// columns -- so the caller asks for what it needs and comes back for more.
    pub cells: Option<Vec<(usize, usize)>>,
    /// Where to put an encoded node's arrays instead of the pipe, if the caller offered a
    /// block and this node fits in it. Absent means the pipe, which is what a caller that
    /// has not been taught about the block sends.
    pub shm: Option<shm::Target>,
    /// Which encoding rule the asking leaf wants (`EncodingRules`, IKA-141). Absent means
    /// the current one, which is what every caller but a fix-measuring match sends.
    pub encoding: crate::encode::EncodeRules,
}

impl Request {
    /// The (row, column) pairs to resolve, in the order they will be reported.
    pub fn wanted_cells(&self) -> Vec<(usize, usize)> {
        match &self.cells {
            Some(listed) => listed
                .iter()
                .copied()
                .filter(|(i, j)| *i < self.ours.len() && *j < self.theirs.len())
                .collect(),
            None => (0..self.ours.len())
                .flat_map(|i| (0..self.theirs.len()).map(move |j| (i, j)))
                .collect(),
        }
    }
}

pub fn parse_request(value: &Value) -> Result<Request, String> {
    let position = Position::from_json(&value["position"]);
    let read = |key: &str| -> Vec<Vec<SlotAction>> {
        value[key]
            .as_array()
            .map(|list| list.iter().map(parse_actions_list).collect())
            .unwrap_or_default()
    };
    let objectives: Vec<String> = value["objectives"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).map(String::from).collect())
        .unwrap_or_default();
    let encode = value.get("encode").and_then(Value::as_bool).unwrap_or(false);
    // An encoded node may still be asked for named objectives beside the learned leaf --
    // the analyser runs one as a cross-check -- so the names are checked either way.
    for name in &objectives {
        if objective::by_name(name).is_none() {
            return Err(format!("objective not available in the port: {name}"));
        }
    }
    let cells = value.get("cells").and_then(Value::as_array).map(|listed| {
        listed
            .iter()
            .filter_map(|pair| {
                let pair = pair.as_array()?;
                Some((pair.first()?.as_u64()? as usize, pair.get(1)?.as_u64()? as usize))
            })
            .collect()
    });
    // A `shm` key at all means the caller will hold a block for the arrays. The name and
    // the size come together or not at all -- a name with no size would mean writing into
    // a block of unknown length, which is the one mistake this side must not be able to
    // make -- and neither of them means "I hold none yet, ask me for the size you need".
    let shm = value.get("shm").map(|block| shm::Target {
        name: block.get("name").and_then(Value::as_str).map(String::from),
        capacity: block.get("bytes").and_then(Value::as_u64).unwrap_or(0) as usize,
    });
    let encoding = crate::encode::EncodeRules {
        mega_from_slots: value
            .get("encoding")
            .and_then(|e| e.get("megaFromSlots"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    };
    Ok(Request {
        encoding,
        encode,
        cells,
        shm,
        position,
        ours: read("ours"),
        theirs: read("theirs"),
        budget: Budget::from_json(&value["budget"]),
        objectives,
    })
}

/// Fills the matrix, leaving the cells this port refuses for the caller.
pub fn fill(reg: &Reg, request: &Request) -> Value {
    let rows = request.ours.len();
    let cols = request.theirs.len();
    let scorers: Vec<fn(&Position) -> f64> = request
        .objectives
        .iter()
        .map(|name| objective::by_name(name).unwrap())
        .collect();

    let mut payoffs: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![0.0; cols]; rows]; scorers.len().max(1)];
    let mut exact = vec![vec![false; cols]; rows];
    let mut refused: Vec<Value> = Vec::new();
    let mut unmodelled: BTreeSet<String> = BTreeSet::new();

    for (i, j) in request.wanted_cells() {
        {
            let actions = [request.ours[i].clone(), request.theirs[j].clone()];
            match resolve_turn(reg, &request.position, &actions, request.budget) {
                Err(reason) => refused.push(json!([i, j, reason])),
                Ok(result) => {
                    exact[i][j] = result.exact;
                    let mut failed: Option<String> = None;
                    for (index, score) in scorers.iter().enumerate() {
                        match turn_value(reg, &result, *score, 0, &mut unmodelled) {
                            Ok(value) => payoffs[index][i][j] = value,
                            Err(reason) => {
                                failed = Some(reason);
                                break;
                            }
                        }
                    }
                    if let Some(reason) = failed {
                        refused.push(json!([i, j, reason]));
                    }
                }
            }
        }
    }

    json!({
        "payoffs": payoffs,
        "exact": exact,
        "refused": refused,
        "unmodelled": unmodelled.into_iter().collect::<Vec<_>>(),
    })
}

/// One turn, for advancing a game rather than filling a matrix.
///
/// The weights come back first and the caller samples an index with its own generator --
/// which is what keeps a generated game bit-identical to one played without this bridge,
/// since `numpy.random.Generator.choice` is the stream. Then it asks for that one
/// position. Returning every branch instead would be right and useless: an exact budget
/// produces a hundred-odd of them, and a hundred positions is two megabytes of JSON per
/// turn against the thirty kilobytes this costs.
pub fn resolve_one(reg: &Reg, value: &Value) -> Value {
    // The same turn twice in a row is the caller's `weights` and then its `branch`
    // (IKA-264): resolved once, and the second answered from the first.
    let key = resolve_key(value);
    LAST_RESOLVE.with(|cell| {
        let mut last = cell.borrow_mut();
        if last.as_ref().is_none_or(|kept| kept.key != key) {
            *last = None;
            match resolve_fresh(reg, value, key) {
                Err(refused) => return refused,
                Ok(fresh) => *last = Some(fresh),
            }
        }
        let kept = last.as_ref().expect("resolved just above");
        let selected = value.get("select").and_then(Value::as_u64).map(|k| k as usize);
        let chosen = match selected {
            None => Value::Null,
            Some(index) => match kept.positions.get(index) {
                None => return json!({ "refused": "branch index out of range" }),
                Some(position) => position.to_json(),
            },
        };
        json!({
            "branches": kept.weights,
            "suspended": kept.paused,
            "exact": kept.exact,
            "unmodelled": kept.unmodelled,
            "position": chosen,
        })
    })
}

/// A resolved turn as `resolve_one` answers from it: the weights, and the branches'
/// positions for the `select` that follows.
struct KeptTurn {
    key: String,
    weights: Vec<f64>,
    paused: Vec<f64>,
    exact: bool,
    unmodelled: Vec<String>,
    positions: Vec<Position>,
}

thread_local! {
    /// The last `resolve` turn this process answered (IKA-264). One: the repeat it is for
    /// is the very next request, and anything else asked in between replaces it.
    static LAST_RESOLVE: std::cell::RefCell<Option<KeptTurn>> =
        const { std::cell::RefCell::new(None) };
}

/// What makes two `resolve` requests the same turn: everything but `select`.
fn resolve_key(value: &Value) -> String {
    format!("{}\u{1}{}\u{1}{}", value["position"], value["actions"], value["budget"])
}

fn resolve_fresh(reg: &Reg, value: &Value, key: String) -> Result<KeptTurn, Value> {
    let position = Position::from_json(&value["position"]);
    if &*position.format != reg.format_id.as_str() {
        return Err(json!({ "refused": "position is for another regulation" }));
    }
    let actions_value = &value["actions"];
    let actions = [
        parse_actions_list(&actions_value[0]),
        parse_actions_list(&actions_value[1]),
    ];
    let budget = Budget::from_json(&value["budget"]);
    let result = match resolve_turn(reg, &position, &actions, budget) {
        Err(reason) => return Err(json!({ "refused": reason })),
        Ok(result) => result,
    };
    // A suspension's continuation state cannot cross a process boundary, but its weight
    // can -- and the caller only has to resolve the turn itself when it actually draws
    // one, which is rarer than merely having one.
    Ok(KeptTurn {
        key,
        weights: result.branches.iter().map(|b| b.probability).collect(),
        paused: result.suspended.iter().map(|s| s.probability).collect(),
        exact: result.exact,
        unmodelled: result.unmodelled.iter().cloned().collect(),
        positions: result.branches.into_iter().map(|b| b.position).collect(),
    })
}

/// Scores a pool of candidates, which is what decides the menu the matrix is filled from.
///
/// Separate from `fill` because it is a different question with a different shape: no
/// turns are resolved, only damage calculated, and what comes back is one number per
/// candidate rather than a matrix.
fn score_pool(reg: &Reg, value: &Value) -> Value {
    let position = Position::from_json(&value["position"]);
    if &*position.format != reg.format_id.as_str() {
        return json!({
            "error": format!(
                "position is {} but the regulation is {}", position.format, reg.format_id
            )
        });
    }
    let side = value["side"].as_u64().unwrap_or(0) as usize;
    let candidates: Vec<Vec<crate::resolve::SlotAction>> = value["candidates"]
        .as_array()
        .map(|list| list.iter().map(crate::resolve::parse_actions_list).collect())
        .unwrap_or_default();
    match crate::score::score_candidates(reg, &position, side, &candidates) {
        Err(reason) => json!({ "refused": reason }),
        Ok(scored) => json!({
            "scores": scored.iter().map(|s| s.score).collect::<Vec<_>>(),
            "detail": scored
                .iter()
                .map(|s| {
                    s.detail
                        .iter()
                        .map(|c| json!([c.slot, c.target_slot, c.is_foe, c.signed, c.exact]))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        }),
    }
}

/// JSONL over stdio: one request per line, one response per line.
///
/// The lock is held across the loop rather than taken per line, because an encoded node
/// whose body needs a bigger block than the caller holds asks for one and reads the answer
/// -- so `answer` needs the same stdin the requests arrive on.
pub fn serve(reg: &Reg) {
    let encoder = crate::encode::Encoder::new(reg);
    let mut shared = shm::Cache::default();
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let mut stdout = std::io::stdout();
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if line.trim().is_empty() {
            continue;
        }
        // A panic answering one node must not cost the caller the rest of the run. The
        // node it panicked on is refused, which the caller already knows how to fill in
        // Python, and the process stays up. Two generation workers lost a thousand games
        // each to a failure that took the bridge down for good.
        let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            answer(reg, &encoder, &mut shared, &mut input, &line, &mut stdout)
        }));
        match answered {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                // Loudly. Every one of these used to be a bare `break`, so the process
                // exited cleanly with nothing on stderr and the caller was left holding
                // "the Rust node sent 0 of 64219920 bytes:" with no reason after the colon.
                eprintln!("node: the pipe failed ({error}); stopping");
                break;
            }
            Err(_) => {
                // `catch_unwind` has already printed the panic and where it came from.
                let refusal = json!({ "refused": "the port panicked on this node" });
                if writeln!(stdout, "{refusal}").and_then(|()| stdout.flush()).is_err() {
                    break;
                }
            }
        }
    }
}

/// The header as text, carrying what its own serialisation cost.
///
/// A number cannot be inside the object that measures it, so the object is serialised and
/// the two timings are appended before the closing brace. `headerUs` is the serialisation
/// alone; building the spans and folds that make it large is `foldUs`, which `fill`
/// measures where it happens.
fn with_timings(header: &Value, parse_us: f64) -> String {
    let started = std::time::Instant::now();
    let mut text = header.to_string();
    let header_us = started.elapsed().as_secs_f64() * 1e6;
    if !text.ends_with('}') {
        return text;
    }
    text.pop();
    let separator = if text.ends_with('{') { "" } else { "," };
    text.push_str(&format!(
        "{separator}\"parseUs\":{parse_us:.1},\"headerUs\":{header_us:.1}}}"
    ));
    text
}

/// Puts an encoded node's arrays where the caller can get at them, and says where.
///
/// Three outcomes, and the header names which one: into the block the request offered
/// (`shm`), into a block the caller is asked for and then makes (`grow`, one extra line
/// each way), or down the pipe behind the header (`pipe`). The asking is what keeps the
/// block the right size without anybody guessing: the caller cannot know how many leaves
/// a node has until it is resolved, and by then this side knows exactly.
fn place_body<R: BufRead, W: Write>(
    encoded: &crate::encode::Encoded,
    leaf_values: &[f64],
    body_bytes: usize,
    target: &shm::Target,
    shared: &mut shm::Cache,
    input: &mut R,
    stdout: &mut W,
    header: &mut Value,
    parse_us: f64,
) -> std::io::Result<()> {
    if shm::place(shared, target, body_bytes, |sink| {
        crate::encoded_node::write_body(sink, encoded, leaf_values)
    })
    .is_some()
    {
        header["via"] = json!("shm");
        writeln!(stdout, "{}", with_timings(header, parse_us))?;
        return stdout.flush();
    }
    // Nothing here fits: ask, and let the caller answer with a block or with the pipe.
    header["via"] = json!("grow");
    writeln!(stdout, "{}", with_timings(header, parse_us))?;
    stdout.flush()?;
    let mut reply = String::new();
    if input.read_line(&mut reply)? == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "the caller was asked for a block and went away",
        ));
    }
    let offered: Value = serde_json::from_str(reply.trim()).unwrap_or(Value::Null);
    let grown = shm::Target {
        name: offered.get("name").and_then(Value::as_str).map(String::from),
        capacity: offered.get("bytes").and_then(Value::as_u64).unwrap_or(0) as usize,
    };
    if shm::place(shared, &grown, body_bytes, |sink| {
        crate::encoded_node::write_body(sink, encoded, leaf_values)
    })
    .is_some()
    {
        writeln!(stdout, "{}", json!({ "via": "shm" }))?;
        return stdout.flush();
    }
    // The caller said the pipe, or made a block this could not open. Either way the body
    // still has to arrive, and the pipe is the road that always works.
    writeln!(stdout, "{}", json!({ "via": "pipe" }))?;
    crate::encoded_node::write_body(stdout, encoded, leaf_values)?;
    stdout.flush()
}

/// Answers one request onto `stdout`. `Err` means the pipe is gone and serving is over.
fn answer<R: BufRead, W: Write>(
    reg: &Reg,
    encoder: &crate::encode::Encoder,
    shared: &mut shm::Cache,
    input: &mut R,
    line: &str,
    stdout: &mut W,
) -> std::io::Result<()> {
    // What reading the request costs, for the caller's table. The other candidate for
    // this crossing's remaining cost is the header going the other way, and the two have
    // different fixes, so they are reported apart rather than as one "JSON" row.
    let parse_started = std::time::Instant::now();
    let parsed = serde_json::from_str::<Value>(line);
    let parse_us = parse_started.elapsed().as_secs_f64() * 1e6;
    let response = match parsed {
        Err(error) => json!({ "error": error.to_string() }),
        Ok(value) if value["kind"].as_str() == Some("resolve") => resolve_one(reg, &value),
        Ok(value) if value["kind"].as_str() == Some("score") => score_pool(reg, &value),
        // A pause's replacements with their leaves encoded (IKA-209): the arrays take the
        // same roads as an encoded node's.
        Ok(value) if value["kind"].as_str() == Some("alternativesEncoded") => {
            match crate::resolve::commands::alternatives_encoded(reg, encoder, &value) {
                Err(reason) => json!({ "refused": reason }),
                Ok((mut header, encoded, leaf_values)) => {
                    let body_bytes = header["bytes"].as_u64().unwrap_or(0) as usize;
                    let target = value.get("shm").map(|block| shm::Target {
                        name: block.get("name").and_then(Value::as_str).map(String::from),
                        capacity: block.get("bytes").and_then(Value::as_u64).unwrap_or(0) as usize,
                    });
                    return match &target {
                        Some(target) => place_body(
                            &encoded, &leaf_values, body_bytes, target, shared, input,
                            stdout, &mut header, parse_us,
                        ),
                        None => {
                            header["via"] = json!("pipe");
                            writeln!(stdout, "{}", with_timings(&header, parse_us))?;
                            crate::encoded_node::write_body(stdout, &encoded, &leaf_values)?;
                            stdout.flush()
                        }
                    };
                }
            }
        }
        Ok(value) if crate::resolve::commands::handles(value["kind"].as_str()) => {
            crate::resolve::commands::answer(reg, &value)
        }
        Ok(value) => match parse_request(&value) {
            Err(reason) => json!({ "error": reason }),
            Ok(request) => {
                if &*request.position.format != reg.format_id.as_str() {
                    json!({
                        "error": format!(
                            "position is {} but the regulation is {}",
                            request.position.format, reg.format_id
                        )
                    })
                } else if request.encode {
                    // The arrays into the caller's block when there is one, and a header
                    // line either way saying where they went. A block is filled *before*
                    // its header is sent, because the header is what says to read it.
                    let (mut header, encoded, leaf_values) =
                        crate::encoded_node::fill(reg, encoder, &request);
                    let body_bytes = header["bytes"].as_u64().unwrap_or(0) as usize;
                    return match &request.shm {
                        Some(target) => place_body(
                            &encoded, &leaf_values, body_bytes, target, shared, input,
                            stdout, &mut header, parse_us,
                        ),
                        None => {
                            header["via"] = json!("pipe");
                            writeln!(stdout, "{}", with_timings(&header, parse_us))?;
                            crate::encoded_node::write_body(stdout, &encoded, &leaf_values)?;
                            stdout.flush()
                        }
                    };
                } else {
                    fill(reg, &request)
                }
            }
        },
    };
    writeln!(stdout, "{response}")?;
    stdout.flush()
}
