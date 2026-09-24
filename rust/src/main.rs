//! The port's entry point: the node process (`node`) and the self-tests over fixture files.
//!
//!     cargo run --release -- damage    <regulation.json> <cases.json> [repeats]
//!     cargo run --release -- roundtrip <turns.json>
//!     cargo run --release -- turns     <regulation.json> <turns.json> [repeats]
//!     cargo run --release -- node      <regulation.json>          # JSONL over stdio
//!     cargo run --release -- encode    <regulation.json> <turns.json> <out.bin> [--mega-from-slots]
//!
//! The self-tests (`damage`, `roundtrip`, `turns`, `encode`) read fixture files Python's
//! resolver wrote (`cases*.json`, `turns*.json`, kept out of git): they are regression tests
//! against Python's answers as of when each file was written. The tools that wrote them went
//! with Python's resolver (IKA-212), so the files are kept and not regenerated. The chain of
//! verification is Showdown -> port since IKA-212: the oracle tests and `tools/diff_turn.py`.

#[macro_use]
mod events;
mod battler;
#[cfg(feature = "count-allocations")]
mod counting_alloc;
mod damage;
mod damage_callback;
mod effects;
mod encode;
mod encoded_node;
mod fixedpoint;
mod id;
mod inert;
mod modelled;
mod moveinfo;
mod node;
mod objective;
mod moves;
mod position;
mod reg;
mod resolve;
mod score;
mod shm;
mod speed;
mod terrain;

#[cfg(feature = "count-allocations")]
#[global_allocator]
static ALLOCATOR: counting_alloc::Counting = counting_alloc::Counting;

use serde_json::Value;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("damage") => damage_main(&args[2..]),
        Some("roundtrip") => roundtrip_main(&args[2..]),
        Some("turns") => turns_main(&args[2..]),
        Some("node") => node_main(&args[2..]),
        Some("encode") => encode_main(&args[2..]),
        Some("clones") => clones_main(&args[2..]),
        _ => {
            eprintln!(
                "usage:\n  {0} damage <regulation.json> <cases.json> [repeats]\n  \
                 {0} roundtrip <turns.json>\n  {0} turns <regulation.json> <turns.json> [repeats]",
                args[0]
            );
            std::process::exit(2);
        }
    }
}

/// Encodes the positions of a fixture, for the encoder differential.
///
/// Writes a JSON header line and then the raw little-endian buffers, which is what the
/// node protocol does too -- the arrays are the network's input and JSON would be both
/// larger and lossy about float32.
fn encode_main(args: &[String]) {
    use std::io::Write;

    let reg = reg::Reg::load(&args[0]).unwrap_or_else(|e| {
        eprintln!("regulation: {e}");
        std::process::exit(1);
    });
    let text = std::fs::read_to_string(&args[1]).expect("fixture");
    let doc: Value = serde_json::from_str(&text).expect("fixture json");
    require_same_format(&doc, &reg);
    let positions: Vec<position::Position> = doc["positions"]
        .as_array()
        .expect("positions")
        .iter()
        .map(position::Position::from_json)
        .collect();
    let borrowed: Vec<&position::Position> = positions.iter().collect();

    let encoder = encode::Encoder::new(&reg);
    // Revision 1's `can_mega` rule, for comparing an arm that plays it (IKA-141).
    let rules = encode::EncodeRules {
        mega_from_slots: args.iter().skip(3).any(|a| a == "--mega-from-slots"),
    };
    let started = Instant::now();
    let encoded = encoder.encode_positions_with(&borrowed, rules);
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "encoded {} positions in {:.3} s = {:.1} us each",
        positions.len(),
        elapsed,
        elapsed / positions.len() as f64 * 1e6
    );

    let mut out = std::io::BufWriter::new(std::fs::File::create(&args[2]).expect("output"));
    let header = serde_json::json!({
        "positions": positions.len(),
        "monsPerSide": encoder.widths.mons_per_side,
        "monWidth": encoder.widths.mon,
        "sideWidth": encoder.widths.side,
        "fieldWidth": encoder.widths.field,
        "types": encoder.vocab.types,
        "unknownVolatiles": encoded.unknown_volatiles,
        "encoding": { "megaFromSlots": rules.mega_from_slots },
    });
    writeln!(out, "{header}").expect("header");
    for value in &encoded.species {
        out.write_all(&value.to_le_bytes()).expect("species");
    }
    for value in &encoded.ability {
        out.write_all(&value.to_le_bytes()).expect("ability");
    }
    for value in &encoded.item {
        out.write_all(&value.to_le_bytes()).expect("item");
    }
    for value in &encoded.moves {
        out.write_all(&value.to_le_bytes()).expect("moves");
    }
    for value in &encoded.mon {
        out.write_all(&value.to_le_bytes()).expect("mon");
    }
    for value in &encoded.mask {
        out.write_all(&value.to_le_bytes()).expect("mask");
    }
    for value in &encoded.side {
        out.write_all(&value.to_le_bytes()).expect("side");
    }
    for value in &encoded.field {
        out.write_all(&value.to_le_bytes()).expect("field");
    }
}

/// Serves whole nodes over stdio, so a caller can fill a matrix in one crossing.
fn node_main(args: &[String]) {
    let reg = reg::Reg::load(&args[0]).unwrap_or_else(|e| {
        eprintln!("regulation: {e}");
        std::process::exit(1);
    });
    node::serve(&reg);
}

// ---------------------------------------------------------------------------
// The position model: does it survive a round trip through Rust unchanged?
// ---------------------------------------------------------------------------

/// A fixture is for one regulation and so is a `Reg`; comparing across two is comparing
/// two different games. The damage differential has always checked this; the turn one did
/// not, and was quietly holding an M-C build to an M-B fixture.
fn require_same_format(doc: &Value, reg: &reg::Reg) {
    let fixture = doc["format_id"].as_str().unwrap_or("");
    if fixture != reg.format_id {
        eprintln!(
            "the fixture is for {fixture} but the regulation is {}; \
             pass configs/regulations/{fixture}.json",
            reg.format_id
        );
        std::process::exit(1);
    }
}

/// How long does cloning a position take?
///
/// The resolver clones one per branch and per action, and a `Position` owns a String and a
/// dozen Vecs, so a clone is tens of allocations on an allocator nobody chose. Whether that
/// is worth restructuring for is a measurement, not an opinion -- this is the measurement.
fn clones_main(args: &[String]) {
    let text = std::fs::read_to_string(&args[0]).expect("turns file");
    let doc: Value = serde_json::from_str(&text).expect("turns json");
    let positions: Vec<position::Position> = doc["positions"]
        .as_array()
        .expect("positions")
        .iter()
        .map(position::Position::from_json)
        .collect();
    let repeats: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(200);
    // What a Battler costs to build, since the resolver builds hundreds a turn. Only when
    // a regulation is named, because loading one is the expensive half of this subcommand.
    if let Some(Ok(reg)) = args.get(2).map(|path| reg::Reg::load(path)) {
        let mut sink = 0i64;
        let mons: Vec<&position::Pokemon> = positions
            .iter()
            .flat_map(|p| p.sides.iter().flat_map(|s| s.pokemon.iter().map(|m| &**m)))
            .take(400)
            .collect();
        let started = std::time::Instant::now();
        for _ in 0..500 {
            for mon in &mons {
                if let Ok(battler) = battler::Battler::from_pokemon(&reg, mon) {
                    sink += battler.hp;
                }
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        let n = 500 * mons.len();
        println!(
            "battlers: {n} in {elapsed:.3} s = {:.3} us each [sink {sink}]",
            elapsed / n as f64 * 1e6
        );
    }
    // What one small allocation costs, since a turn makes 557 of them. The pattern is the
    // resolver's: allocate, use, free, in a tight loop where the block is reused.
    {
        let mut sink = 0usize;
        let started = std::time::Instant::now();
        for i in 0..2_000_000usize {
            let v: Vec<u64> = vec![i as u64; 8];
            sink += v.len();
            std::hint::black_box(&v);
        }
        let elapsed = started.elapsed().as_secs_f64();
        println!(
            "one 64-byte alloc+free: {:.1} ns [sink {sink}]",
            elapsed / 2_000_000.0 * 1e9
        );
    }
    println!(
        "sizes: Position {} B, Side {} B, Pokemon {} B, Effect {} B, MoveSlot {} B",
        std::mem::size_of::<position::Position>(),
        std::mem::size_of::<position::Side>(),
        std::mem::size_of::<position::Pokemon>(),
        std::mem::size_of::<position::Effect>(),
        std::mem::size_of::<position::MoveSlot>(),
    );

    let started = std::time::Instant::now();
    let mut sink = 0usize;
    for _ in 0..repeats {
        for position in &positions {
            let copy = position.clone();
            sink += copy.sides.len() + copy.turn as usize;
            std::hint::black_box(&copy);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let count = repeats * positions.len();
    println!(
        "clones: {count} in {elapsed:.3} s = {:.3} us each [sink {sink}]",
        elapsed / count as f64 * 1e6
    );
}

fn roundtrip_main(args: &[String]) {
    let text = std::fs::read_to_string(&args[0]).expect("turns file");
    let doc: Value = serde_json::from_str(&text).expect("turns json");
    let positions = doc["positions"].as_array().expect("positions");

    let mut mismatched = 0usize;
    let mut shown = 0usize;
    for (index, original) in positions.iter().enumerate() {
        let parsed = position::Position::from_json(original);
        let again = parsed.to_json();
        if &again == original {
            continue;
        }
        mismatched += 1;
        if shown < 3 {
            shown += 1;
            println!("\n#{index} differs:");
            report_json_diff("", original, &again, &mut 0);
        }
    }
    println!(
        "\nround trip: {}/{} positions identical ({} differ)",
        positions.len() - mismatched,
        positions.len(),
        mismatched
    );
    if mismatched > 0 {
        std::process::exit(1);
    }
}

/// Prints where two JSON documents first differ, rather than dumping both.
pub fn report_json_diff(path: &str, left: &Value, right: &Value, shown: &mut usize) {
    if *shown >= 12 || left == right {
        return;
    }
    match (left, right) {
        (Value::Object(a), Value::Object(b)) => {
            for (key, value) in a {
                let child = format!("{path}/{key}");
                match b.get(key) {
                    None => {
                        println!("  {child}: python has {value}, rust has nothing");
                        *shown += 1;
                    }
                    Some(other) => report_json_diff(&child, value, other, shown),
                }
            }
            for key in b.keys() {
                if !a.contains_key(key) {
                    println!("  {path}/{key}: rust has {}, python has nothing", b[key]);
                    *shown += 1;
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                println!("  {path}: python has {} items {}, rust has {} {}", a.len(), serde_json::to_string(a).unwrap_or_default(), b.len(), serde_json::to_string(b).unwrap_or_default());
                *shown += 1;
                return;
            }
            for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                report_json_diff(&format!("{path}[{index}]"), x, y, shown);
            }
        }
        _ => {
            println!("  {path}: python {left}, rust {right}");
            *shown += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Whole turns
// ---------------------------------------------------------------------------

fn turns_main(args: &[String]) {
    let reg = reg::Reg::load(&args[0]).unwrap_or_else(|e| {
        eprintln!("regulation: {e}");
        std::process::exit(1);
    });
    let text = std::fs::read_to_string(&args[1]).expect("turns file");
    let doc: Value = serde_json::from_str(&text).expect("turns json");
    let repeats: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

    require_same_format(&doc, &reg);
    let positions: Vec<position::Position> = doc["positions"]
        .as_array()
        .expect("positions")
        .iter()
        .map(position::Position::from_json)
        .collect();
    let raw_positions = doc["positions"].as_array().unwrap();
    let cases = doc["cases"].as_array().expect("cases");

    let mut refused = 0usize;
    let mut matched = 0usize;
    let mut wrong = 0usize;
    let mut shown = 0usize;
    let mut reasons: std::collections::BTreeMap<String, usize> = Default::default();

    let mut refused_indices: Vec<usize> = Vec::new();
    for (case_index, case) in cases.iter().enumerate() {
        let index = case["position"].as_u64().unwrap() as usize;
        let actions = resolve::parse_actions(case);
        let budget = resolve::Budget::from_json(&case["budget"]);
        match resolve::resolve_turn(&reg, &positions[index], &actions, budget) {
            Err(reason) => {
                refused += 1;
                refused_indices.push(case_index);
                *reasons.entry(reason.to_string()).or_default() += 1;
            }
            Ok(result) => {
                let expect = &case["expect"];
                match compare_turn(&result, expect, raw_positions) {
                    None => matched += 1,
                    Some(explanation) => {
                        wrong += 1;
                        if shown < 5 {
                            shown += 1;
                            println!("\nmismatch: {explanation}");
                            print_case_header(case, &positions[index]);
                            let rust: Vec<String> = result
                                .branches
                                .iter()
                                .map(|b| format!("{:.4}", b.probability))
                                .collect();
                            let python: Vec<String> = expect["branches"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|b| format!("{:.4}", b["probability"].as_f64().unwrap()))
                                .collect();
                            println!("  python weights {}", python.join(" "));
                            println!("  rust   weights {}", rust.join(" "));
                        }
                    }
                }
            }
        }
    }

    let total = cases.len();
    println!(
        "\nturns: {total} cases -- {matched} exact, {wrong} wrong, {refused} refused \
         ({:.1}% resolved, {:.2}% of resolved wrong)",
        (total - refused) as f64 / total as f64 * 100.0,
        wrong as f64 / ((total - refused).max(1)) as f64 * 100.0
    );
    if !reasons.is_empty() {
        println!("\nrefusals, heaviest first:");
        let mut sorted: Vec<_> = reasons.into_iter().collect();
        sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        for (reason, count) in sorted.iter().take(25) {
            println!("  {count:>5}  {reason}");
        }
    }

    std::fs::write(
        "refused.json",
        serde_json::to_string(&refused_indices).unwrap(),
    )
    .ok();

    if repeats > 0 {
        // Everything that is not the resolver is hoisted out of the loop, because the
        // Python benchmark it is compared against does the same.
        // Only the turns Rust actually resolves are timed: the rest cost Python's time in
        // a hybrid, and `tools/bench_turn_cases.py` timed those separately (until IKA-212).
        let prepared: Vec<(usize, [Vec<resolve::SlotAction>; 2], resolve::Budget)> = cases
            .iter()
            .enumerate()
            .filter(|(case_index, _)| !refused_indices.contains(case_index))
            .map(|(_, case)| {
                (
                    case["position"].as_u64().unwrap() as usize,
                    resolve::parse_actions(case),
                    resolve::Budget::from_json(&case["budget"]),
                )
            })
            .collect();
        let mut sink = 0usize;
        resolve::reset_counters();
        #[cfg(feature = "count-allocations")]
        counting_alloc::reset();
        let started = Instant::now();
        for _ in 0..repeats {
            for (index, actions, budget) in &prepared {
                if let Ok(result) =
                    resolve::resolve_turn(&reg, &positions[*index], actions, *budget)
                {
                    sink += result.branches.len();
                }
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        let calls = repeats as f64 * prepared.len() as f64;
        println!(
            "\nspeed (resolved turns only): {calls:.0} turns in {elapsed:.3} s = {:.1} us/turn ({:.0} turns/s) [sink {sink}]",
            elapsed / calls * 1e6,
            calls / elapsed
        );
        let clones = position::CLONES.load(std::sync::atomic::Ordering::Relaxed) as f64;
        println!("  position clones: {clones:.0} = {:.1} per turn", clones / calls);
        let hits = damage::CALLS.load(std::sync::atomic::Ordering::Relaxed) as f64;
        println!("  damage calls:    {hits:.0} = {:.1} per turn", hits / calls);
        let copies = position::UNSHARED.load(std::sync::atomic::Ordering::Relaxed) as f64;
        println!("  forced copies:   {copies:.0} = {:.1} per turn", copies / calls);
        let relaxed = std::sync::atomic::Ordering::Relaxed;
        let builds = battler::BUILDS.load(relaxed) as f64;
        let resorts = resolve::RESORTS.load(relaxed) as f64;
        let residuals = resolve::RESIDUALS.load(relaxed) as f64;
        println!("  battlers built:  {builds:.0} = {:.1} per turn", builds / calls);
        println!("  queue re-sorts:  {resorts:.0} = {:.1} per turn", resorts / calls);
        println!("  residual phases: {residuals:.0} = {:.1} per turn", residuals / calls);
        #[cfg(feature = "count-allocations")]
        counting_alloc::report(calls);
        #[cfg(feature = "profile")]
        for (index, name) in resolve::PHASE_NAMES.iter().enumerate() {
            let ns = resolve::PHASE_NS[index].load(relaxed) as f64;
            let allocs = resolve::PHASE_ALLOCS[index].load(relaxed) as f64;
            println!(
                "  {name:<16} {:>7.2} us/turn ({:>4.1}%), {:>6.1} allocs/turn",
                ns / calls / 1000.0,
                ns / 1000.0 / (elapsed * 1e6) * 100.0,
                allocs / calls
            );
        }
    }
}

fn print_case_header(case: &Value, pos: &position::Position) {
    let describe = |side: usize| -> String {
        (0..pos.sides[side].active.len())
            .map(|slot| match pos.mon_at(side, slot) {
                None => "-".to_string(),
                Some(mon) => format!("{}", mon.species),
            })
            .collect::<Vec<_>>()
            .join("+")
    };
    println!("  {} vs {}", describe(0), describe(1));
    println!("  ours   {}", case["ours"]);
    println!("  theirs {}", case["theirs"]);
}

fn compare_turn(
    got: &resolve::TurnResult,
    expect: &Value,
    raw_positions: &[Value],
) -> Option<String> {
    if got.is_suspended() != expect["suspended"].as_bool().unwrap_or(false) {
        return Some(format!(
            "suspended: python {}, rust {}",
            expect["suspended"],
            got.is_suspended()
        ));
    }
    // The reported effects are part of the answer: a caller prints them, and a port that
    // resolved a turn without reporting what it approximated would quietly shrink that
    // list.
    let wanted_notes: Vec<&str> = expect["unmodelled"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let got_notes: Vec<&str> = got.unmodelled.iter().map(String::as_str).collect();
    if wanted_notes != got_notes {
        return Some(format!(
            "unmodelled: python {wanted_notes:?}, rust {got_notes:?}"
        ));
    }
    let want = expect["branches"].as_array().unwrap();
    if got.branches.len() != want.len() {
        return Some(format!(
            "branch count: python {}, rust {}",
            want.len(),
            got.branches.len()
        ));
    }
    for (index, (branch, wanted)) in got.branches.iter().zip(want.iter()).enumerate() {
        let probability = wanted["probability"].as_f64().unwrap();
        if (branch.probability - probability).abs() > 1e-12 {
            return Some(format!(
                "branch {index} probability: python {probability}, rust {}",
                branch.probability
            ));
        }
        let expected = &raw_positions[wanted["position"].as_u64().unwrap() as usize];
        let actual = branch.position.to_json();
        if &actual != expected {
            let mut shown = 0;
            report_json_diff(&format!("branch{index}"), expected, &actual, &mut shown);
            return Some(format!("branch {index} position differs"));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Damage, unchanged from the first spike
// ---------------------------------------------------------------------------

fn damage_main(args: &[String]) {
    use battler::{Battler, BattlerCase, FieldCase, FieldState};
    use moveinfo::MoveContext;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct Expect {
        rolls: Vec<i64>,
        effectiveness: f64,
        type_mod: i64,
        immune: bool,
    }

    #[derive(Debug, Deserialize)]
    struct Case {
        #[serde(rename = "move")]
        move_id: String,
        defender_side: usize,
        spread: bool,
        crit: bool,
        base_power_override: Option<i64>,
        attacker: BattlerCase,
        defender: BattlerCase,
        #[serde(default)]
        attacker_is_defender: bool,
        field: FieldCase,
        move_ctx: Option<MoveContext>,
        expect: Expect,
    }

    #[derive(Debug, Deserialize)]
    struct Cases {
        format_id: String,
        cases: Vec<Case>,
    }

    let repeats: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(200);
    let reg = reg::Reg::load(&args[0]).unwrap_or_else(|e| {
        eprintln!("regulation: {e}");
        std::process::exit(1);
    });
    let text = std::fs::read_to_string(&args[1]).expect("cases");
    let cases: Cases = serde_json::from_str(&text).expect("cases json");
    if cases.format_id != reg.format_id {
        eprintln!("cases are for {} but the regulation is {}", cases.format_id, reg.format_id);
        std::process::exit(1);
    }
    println!("{} cases, regulation {}", cases.cases.len(), reg.format_id);

    let run = |case: &Case| {
        let attacker: Battler = (&case.attacker).into();
        let defender: Battler = (&case.defender).into();
        let field: FieldState = (&case.field).into();
        damage::calculate(
            &reg,
            &attacker,
            &defender,
            &case.move_id,
            &field,
            case.defender_side,
            case.spread,
            case.crit,
            case.move_ctx.as_ref(),
            case.base_power_override,
            case.attacker_is_defender,
        )
    };

    let mut mismatched = 0usize;
    let mut shown = 0usize;
    for (index, case) in cases.cases.iter().enumerate() {
        let got = run(case);
        let same = got.rolls.as_slice() == case.expect.rolls.as_slice()
            && (got.effectiveness - case.expect.effectiveness).abs() < 1e-12
            && got.type_mod == case.expect.type_mod
            && got.immune == case.expect.immune;
        if same {
            continue;
        }
        mismatched += 1;
        if shown < 8 {
            shown += 1;
            println!(
                "\n#{index} {} : {} ({}) -> {} ({})",
                case.move_id,
                case.attacker.species,
                case.attacker.ability,
                case.defender.species,
                case.defender.ability
            );
            println!("  python {:?}", case.expect.rolls);
            println!("  rust   {:?}", got.rolls);
        }
    }
    let total = cases.cases.len();
    println!(
        "\nagreement: {}/{} exact ({:.3}% divergent)",
        total - mismatched,
        total,
        mismatched as f64 / total as f64 * 100.0
    );

    let mut sink = 0i64;
    let started = Instant::now();
    for _ in 0..repeats {
        for case in &cases.cases {
            sink = sink.wrapping_add(run(case).rolls[0]);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let calls = repeats as f64 * total as f64;
    println!(
        "speed: {calls:.0} calls in {elapsed:.3} s = {:.3} us/call ({:.0} calls/s) [sink {sink}]",
        elapsed / calls * 1e6,
        calls / elapsed
    );
}
