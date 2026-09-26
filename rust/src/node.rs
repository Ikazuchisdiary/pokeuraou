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
    /// The two action lists as the request wrote them, for a `fills` node that reads its
    /// cells off another's by their actions (IKA-295). Empty everywhere else.
    pub ours_json: Vec<Value>,
    pub theirs_json: Vec<Value>,
    /// The position as the request wrote it, for the cell threads to parse their own
    /// copies from (IKA-32: a `Position` holds `Rc`s and cannot be shared between
    /// threads). `Null` when there is no pool, which is every run at one thread.
    pub position_json: Value,
    /// An encoded node's spans in the header as JSON rather than in the body (IKA-302):
    /// only for the test that holds the two roads to the same lists.
    pub json_spans: bool,
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
    let position = crate::held::position(&value["position"]);
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
    let position_json = match crate::par::Pool::global() {
        Some(_) => crate::held::json(&value["position"]),
        None => Value::Null,
    };
    Ok(Request {
        json_spans: value.get("jsonSpans").and_then(Value::as_bool).unwrap_or(false),
        ours_json: Vec::new(),
        theirs_json: Vec::new(),
        position_json,
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
    fill_on(reg, request, crate::par::Pool::global())
}

/// `fill`, with its cells on `pool` when there is one (IKA-32).
pub fn fill_on(reg: &Reg, request: &Request, pool: Option<&crate::par::Pool>) -> Value {
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

    if let Some(pool) = pool.filter(|_| !request.position_json.is_null()) {
        let wanted = request.wanted_cells();
        let answered = fill_cells(pool, reg, request, &scorers, &wanted);
        for ((i, j), cell) in wanted.into_iter().zip(answered) {
            unmodelled.extend(cell.notes);
            match cell.outcome {
                Err(reason) => refused.push(json!([i, j, reason])),
                Ok((was_exact, values, failed)) => {
                    exact[i][j] = was_exact;
                    for (index, value) in values.into_iter().enumerate() {
                        payoffs[index][i][j] = value;
                    }
                    if let Some(reason) = failed {
                        refused.push(json!([i, j, reason]));
                    }
                }
            }
        }
        return json!({
            "payoffs": payoffs,
            "exact": exact,
            "refused": refused,
            "unmodelled": unmodelled.into_iter().collect::<Vec<_>>(),
        });
    }

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

/// One cell of `fill`, answered on a cell thread (IKA-32).
struct FilledCell {
    /// The resolver's refusal, or (exact, the values up to the first objective that
    /// failed, that objective's reason) -- exactly what `fill`'s own loop writes.
    outcome: Result<(bool, Vec<f64>, Option<String>), String>,
    notes: BTreeSet<String>,
}

/// `fill`'s loop body, one cell per item, on the pool. Each thread resolves from its own
/// parse of the position; the notes are a set, so their union does not care who added
/// what first, and the rest is written back cell by cell in `wanted`'s order.
fn fill_cells(
    pool: &crate::par::Pool,
    reg: &Reg,
    request: &Request,
    scorers: &[fn(&Position) -> f64],
    wanted: &[(usize, usize)],
) -> Vec<FilledCell> {
    let (ours, theirs, budget) = (&request.ours, &request.theirs, request.budget);
    let position_json = &request.position_json;
    pool.map_with(
        wanted.len(),
        || Position::from_json(position_json),
        |position, k| {
            let (i, j) = wanted[k];
            let mut notes = BTreeSet::new();
            let actions = [ours[i].clone(), theirs[j].clone()];
            let outcome = resolve_turn(reg, position, &actions, budget).map(|result| {
                let mut values = Vec::with_capacity(scorers.len());
                let mut failed = None;
                for score in scorers {
                    match turn_value(reg, &result, *score, 0, &mut notes) {
                        Ok(value) => values.push(value),
                        Err(reason) => {
                            failed = Some(reason);
                            break;
                        }
                    }
                }
                (result.exact, values, failed)
            });
            FilledCell { outcome, notes }
        },
    )
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
    let position = crate::held::position(&value["position"]);
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
    let position = crate::held::position(&value["position"]);
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

/// Both sides' candidate features for the candidate model (IKA-274): `qfeatures`
/// answers `WIDTH` numbers per candidate of each side's list, from the same calculator the
/// `score` command uses. A position the calculator refuses is refused whole.
fn qfeatures(reg: &Reg, value: &Value) -> Value {
    let position = crate::held::position(&value["position"]);
    if &*position.format != reg.format_id.as_str() {
        return json!({
            "error": format!(
                "position is {} but the regulation is {}", position.format, reg.format_id
            )
        });
    }
    let mut out = Vec::with_capacity(2);
    for side in 0..2usize {
        let candidates: Vec<Vec<crate::resolve::SlotAction>> = value["candidates"][side]
            .as_array()
            .map(|list| list.iter().map(crate::resolve::parse_actions_list).collect())
            .unwrap_or_default();
        match crate::qfeatures::side_features(reg, &position, side, &candidates) {
            Err(reason) => return json!({ "refused": reason }),
            Ok(rows) => out.push(rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>()),
        }
    }
    json!({ "width": crate::qfeatures::WIDTH, "features": out })
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
        let started = std::time::Instant::now();
        let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            answer(reg, &encoder, &mut shared, &mut input, &line, &mut stdout)
        }));
        crate::wire::end(started);
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
    // IKA-32: the cell threads' counters, for a run whose workers cannot be asked (a
    // generation run's). A file per process in the named directory; nothing without it.
    if let Ok(dir) = std::env::var("POKEURAOU_PORT_THREADS_REPORT") {
        let path = std::path::Path::new(&dir).join(format!("port-{}.json", std::process::id()));
        if let Err(error) = std::fs::write(&path, format!("{}\n", crate::par::report())) {
            eprintln!("node: could not write {}: {error}", path.display());
        }
        // IKA-302: and what the wire cost, beside it.
        let path = std::path::Path::new(&dir).join(format!("wire-{}.json", std::process::id()));
        if let Err(error) = std::fs::write(&path, format!("{}\n", crate::wire::report())) {
            eprintln!("node: could not write {}: {error}", path.display());
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
    crate::wire::wrote(started, text.len());
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
    tail: &[u8],
    body_bytes: usize,
    target: &shm::Target,
    shared: &mut shm::Cache,
    input: &mut R,
    stdout: &mut W,
    header: &mut Value,
    parse_us: f64,
) -> std::io::Result<()> {
    if shm::place(shared, target, body_bytes, |sink| {
        crate::encoded_node::write_body(sink, encoded, leaf_values, tail)
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
        crate::encoded_node::write_body(sink, encoded, leaf_values, tail)
    })
    .is_some()
    {
        writeln!(stdout, "{}", json!({ "via": "shm" }))?;
        return stdout.flush();
    }
    // The caller said the pipe, or made a block this could not open. Either way the body
    // still has to arrive, and the pipe is the road that always works.
    writeln!(stdout, "{}", json!({ "via": "pipe" }))?;
    crate::encoded_node::write_body(stdout, encoded, leaf_values, tail)?;
    stdout.flush()
}

/// Many header-only requests in one line each way (IKA-295).
///
/// Each is answered by the function that answers it alone, in the order sent, so an answer
/// is the one its own line would have had. What goes is the wait between one answer and
/// the next request: a depth-2 pass asked for a turn per refined cell and two scores per
/// sub-game, a line each. A panic on one refuses that one, as it would have refused its
/// own line, and the rest are answered.
fn many(reg: &Reg, value: &Value) -> Value {
    let Some(list) = value["requests"].as_array() else {
        return json!({ "error": "`many` without a list of requests" });
    };
    let answer_one = |one: &Value| -> Value {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match one["kind"].as_str() {
            Some("score") => score_pool(reg, one),
            Some("qfeatures") => qfeatures(reg, one),
            kind if crate::resolve::commands::handles(kind) => {
                crate::resolve::commands::answer(reg, one)
            }
            _ => json!({ "error": "`many` answers `score` and the turn commands only" }),
        }))
        .unwrap_or_else(|_| json!({ "refused": "the port panicked on this node" }))
    };
    // Each request parses its own position and answers in JSON, so nothing of one thread's
    // is seen by another's: on the pool they go one per item and come back in the order sent
    // (IKA-32).
    let answers: Vec<Value> = match crate::par::Pool::global() {
        Some(pool) if list.len() > 1 => {
            // A cell thread cannot see this thread's held positions (IKA-302): each
            // request goes to it with its position written out.
            let owned: Vec<Value> = list
                .iter()
                .map(|one| {
                    let mut one = one.clone();
                    if crate::held::held_id(&one["position"]).is_some() {
                        one["position"] = crate::held::json(&one["position"]);
                    }
                    // Nor keep one for this thread: a branch goes back written out only.
                    if one.get("refs").is_some() {
                        one["refs"] = json!(false);
                    }
                    one
                })
                .collect();
            pool.map_with(owned.len(), || (), |_, k| answer_one(&owned[k]))
        }
        _ => list.iter().map(answer_one).collect(),
    };
    json!({ "kind": "many", "answers": answers })
}

/// Several encoded nodes in one crossing (IKA-295): one header line naming each node's own
/// header, and one body that is their bodies one after another.
///
/// Every node is `encoded_node::fill` of its own request, so its arrays are the bytes its
/// own crossing would have carried; the reader cuts the body at each node's `bytes`. Only
/// encoded nodes are taken (a depth-2 pass's sub-games), and a request that would have been
/// an error alone makes the whole line an error, which the caller treats as it treated one.
fn fills<R: BufRead, W: Write>(
    reg: &Reg,
    encoder: &crate::encode::Encoder,
    shared: &mut shm::Cache,
    input: &mut R,
    stdout: &mut W,
    value: &Value,
    parse_us: f64,
) -> std::io::Result<()> {
    let fail = |stdout: &mut W, reason: String| -> std::io::Result<()> {
        writeln!(stdout, "{}", json!({ "error": reason }))?;
        stdout.flush()
    };
    let Some(list) = value["requests"].as_array() else {
        return fail(stdout, "`fills` without a list of requests".into());
    };
    let mut headers: Vec<Value> = Vec::with_capacity(list.len());
    let mut bodies: Vec<(crate::encode::Encoded, Vec<f64>, Vec<u8>)> =
        Vec::with_capacity(list.len());
    // A node another node of this crossing reads its turns off (`like`), by index.
    let mut kept: Vec<Option<crate::encoded_node::Kept>> = Vec::with_capacity(list.len());
    let mut total = 0usize;
    // IKA-32 stage 2: plain nodes -- none keeps its turns for another or reads another's --
    // go one per pool thread, each resolved and encoded there whole (no pool inside it),
    // with that thread's own encoder, and come back in the order sent. A node's bytes are
    // what its own crossing would have carried (the cells, the shared leaves and the
    // encoding are the node's alone), so the body is the same; what spreads is a crossing
    // of many small nodes -- the deepening's children, 8 x 8 each -- that the cell pool
    // could only split into a few cells a thread.
    let plain = list.len() > 1
        && list.iter().all(|one| {
            !one.get("keep").and_then(Value::as_bool).unwrap_or(false)
                && one.get("like").is_none_or(Value::is_null)
        });
    if let (Some(pool), true) = (crate::par::Pool::global(), plain) {
        let owned: Vec<Value> = list
            .iter()
            .map(|one| {
                let mut one = one.clone();
                // A pool thread cannot see this thread's held positions (IKA-302).
                if crate::held::held_id(&one["position"]).is_some() {
                    one["position"] = crate::held::json(&one["position"]);
                }
                one
            })
            .collect();
        let format = reg.format_id.as_str();
        type Node = (Value, crate::encode::Encoded, Vec<f64>, Vec<u8>);
        let answers: Vec<Result<Node, String>> = pool.map_with(
            owned.len(),
            || crate::encode::Encoder::new(reg),
            |encoder, k| {
                let request = parse_request(&owned[k])?;
                if &*request.position.format != format {
                    return Err(format!(
                        "position is {} but the regulation is {}",
                        request.position.format, format
                    ));
                }
                if !request.encode {
                    return Err("`fills` answers encoded nodes only".into());
                }
                let (header, encoded, leaf_values, spans, _kept) =
                    crate::encoded_node::fill_shared_on(reg, encoder, &request, None, false, None);
                Ok((header, encoded, leaf_values, spans))
            },
        );
        for answer in answers {
            match answer {
                Err(reason) => return fail(stdout, reason),
                Ok((header, encoded, leaf_values, spans)) => {
                    total += header["bytes"].as_u64().unwrap_or(0) as usize;
                    headers.push(header);
                    bodies.push((encoded, leaf_values, spans));
                }
            }
        }
        crate::par::count_node_items(list.len());
    }
    for one in if bodies.is_empty() { &list[..] } else { &list[..0] } {
        let mut request = match parse_request(one) {
            Err(reason) => return fail(stdout, reason),
            Ok(request) => request,
        };
        let listed = |key: &str| one[key].as_array().cloned().unwrap_or_default();
        let like_of = one.get("like").filter(|l| !l.is_null());
        let keep = one.get("keep").and_then(Value::as_bool).unwrap_or(false);
        if keep || like_of.is_some() {
            request.ours_json = listed("ours");
            request.theirs_json = listed("theirs");
        }
        if &*request.position.format != reg.format_id.as_str() {
            return fail(
                stdout,
                format!(
                    "position is {} but the regulation is {}",
                    request.position.format, reg.format_id
                ),
            );
        }
        if !request.encode {
            return fail(stdout, "`fills` answers encoded nodes only".into());
        }
        // Only an earlier node, and only one that was kept; anything else resolves.
        let like = like_of.and_then(|spec| {
            let index = spec.get("node")?.as_u64()? as usize;
            let side = spec.get("side")?.as_u64()? as usize;
            let slots: Vec<usize> = spec
                .get("slots")?
                .as_array()?
                .iter()
                .filter_map(|s| s.as_u64().map(|s| s as usize))
                .collect();
            let earlier = kept.get(index)?.as_ref()?;
            crate::encoded_node::Like::new(
                earlier,
                &request.position,
                side,
                slots,
                request.ours_json.clone(),
                request.theirs_json.clone(),
            )
        });
        let (mut header, encoded, leaf_values, spans, keeping) =
            crate::encoded_node::fill_shared(reg, encoder, &request, like.as_ref(), keep);
        if like_of.is_some() && like.is_none() {
            // Asked to read off a node and could not: said, so the caller can count it.
            header["readOff"] = json!(null);
        }
        total += header["bytes"].as_u64().unwrap_or(0) as usize;
        headers.push(header);
        bodies.push((encoded, leaf_values, spans));
        kept.push(keeping);
    }
    let write_all = |sink: &mut dyn Write| -> std::io::Result<()> {
        for (encoded, leaf_values, spans) in &bodies {
            crate::encoded_node::write_body(sink, encoded, leaf_values, spans)?;
        }
        Ok(())
    };
    let mut header = json!({ "kind": "encodedMany", "nodes": headers, "bytes": total });
    let target = value.get("shm").map(|block| shm::Target {
        name: block.get("name").and_then(Value::as_str).map(String::from),
        capacity: block.get("bytes").and_then(Value::as_u64).unwrap_or(0) as usize,
    });
    let Some(target) = target else {
        header["via"] = json!("pipe");
        writeln!(stdout, "{}", with_timings(&header, parse_us))?;
        write_all(stdout)?;
        return stdout.flush();
    };
    // `place_body`'s three roads, for a body written by `write_all`.
    if shm::place(shared, &target, total, |sink| write_all(sink)).is_some() {
        header["via"] = json!("shm");
        writeln!(stdout, "{}", with_timings(&header, parse_us))?;
        return stdout.flush();
    }
    header["via"] = json!("grow");
    writeln!(stdout, "{}", with_timings(&header, parse_us))?;
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
    if shm::place(shared, &grown, total, |sink| write_all(sink)).is_some() {
        writeln!(stdout, "{}", json!({ "via": "shm" }))?;
        return stdout.flush();
    }
    writeln!(stdout, "{}", json!({ "via": "pipe" }))?;
    write_all(stdout)?;
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
    crate::wire::begin(
        match &parsed {
            Err(_) => "error",
            Ok(value) => match value["kind"].as_str() {
                Some(kind) => kind,
                None if value.get("encode").and_then(Value::as_bool) == Some(true) => "fill_encoded",
                None => "fill",
            },
        },
        parse_started.elapsed().as_nanos() as u64,
        line.len(),
    );
    let response = match parsed {
        Err(error) => json!({ "error": error.to_string() }),
        // IKA-302: a position to hold, and the end of a decision. Neither is answered:
        // they go down the pipe ahead of the request that needs them.
        Ok(value) if value["kind"].as_str() == Some("hold") => {
            if let Some(id) = value["id"].as_u64() {
                crate::held::define(id, &value["position"]);
            }
            return Ok(());
        }
        Ok(value) if value["kind"].as_str() == Some("forget") => {
            crate::held::forget();
            return Ok(());
        }
        Ok(value) if value["kind"].as_str() == Some("resolve") => resolve_one(reg, &value),
        Ok(value) if value["kind"].as_str() == Some("score") => score_pool(reg, &value),
        Ok(value) if value["kind"].as_str() == Some("qfeatures") => qfeatures(reg, &value),
        Ok(value) if value["kind"].as_str() == Some("many") => many(reg, &value),
        // The cell threads' own account (IKA-32): how many, and how much ran on them.
        Ok(value) if value["kind"].as_str() == Some("parallel") => crate::par::report(),
        Ok(value) if value["kind"].as_str() == Some("fills") => {
            return fills(reg, encoder, shared, input, stdout, &value, parse_us);
        }
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
                            &encoded, &leaf_values, &[], body_bytes, target, shared, input,
                            stdout, &mut header, parse_us,
                        ),
                        None => {
                            header["via"] = json!("pipe");
                            writeln!(stdout, "{}", with_timings(&header, parse_us))?;
                            crate::encoded_node::write_body(stdout, &encoded, &leaf_values, &[])?;
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
                    let (mut header, encoded, leaf_values, spans) =
                        crate::encoded_node::fill(reg, encoder, &request);
                    let body_bytes = header["bytes"].as_u64().unwrap_or(0) as usize;
                    return match &request.shm {
                        Some(target) => place_body(
                            &encoded, &leaf_values, &spans, body_bytes, target, shared, input,
                            stdout, &mut header, parse_us,
                        ),
                        None => {
                            header["via"] = json!("pipe");
                            writeln!(stdout, "{}", with_timings(&header, parse_us))?;
                            crate::encoded_node::write_body(stdout, &encoded, &leaf_values, &spans)?;
                            stdout.flush()
                        }
                    };
                } else {
                    fill(reg, &request)
                }
            }
        },
    };
    let written = std::time::Instant::now();
    let text = response.to_string();
    crate::wire::wrote(written, text.len());
    writeln!(stdout, "{text}")?;
    stdout.flush()
}
