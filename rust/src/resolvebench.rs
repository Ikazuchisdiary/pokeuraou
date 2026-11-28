//! IKA-471: the fills' turns, replayed on their own.
//!
//! `dump-fills` (a feature, off by default) writes every fill request the belief plan makes to
//! `$POKEURAOU_FILL_DUMP/<pid>.jsonl`, one in `$POKEURAOU_FILL_DUMP_EVERY` (default 1).
//! `fill-bench <regulation.json> <dump.jsonl>... [--repeats N] [--digest] [--limit N] [--max-acted N|off]` reads
//! them back and resolves each fill's cells as `encoded_node::fill` does on one thread: the
//! turn, then every pause resumed (`resume_all`). It does not encode. `--digest` prints a hash
//! of every answer (branch weights, positions, pauses, notes, refusals), so two builds can be
//! held to the same turns; without it, it times the resolving.

use crate::position::Position;
use crate::reg::Reg;
use crate::resolve::{resolve_turn, resume_alternatives, sharing_budget, TurnResult};
use serde_json::Value;
use std::hash::Hasher;

#[cfg(feature = "dump-fills")]
pub fn dump_fill(request: &Value) {
    use std::cell::Cell;
    use std::io::Write;
    thread_local! {
        static SEEN: Cell<u64> = const { Cell::new(0) };
    }
    let Ok(dir) = std::env::var("POKEURAOU_FILL_DUMP") else { return };
    let every: u64 = std::env::var("POKEURAOU_FILL_DUMP_EVERY").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let seen = SEEN.with(|s| {
        let n = s.get();
        s.set(n + 1);
        n
    });
    if seen % every.max(1) != 0 {
        return;
    }
    let path = format!("{dir}/{}.jsonl", std::process::id());
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let mut whole = request.clone();
        whole["position"] = crate::held::json(&request["position"]);
        let mut line = serde_json::to_vec(&whole).unwrap();
        line.push(b'\n');
        let _ = file.write_all(&line);
    }
}

#[cfg(feature = "count-damage")]
thread_local! {
    static DAMAGE_SEEN: std::cell::RefCell<(std::collections::HashSet<u64>, usize, usize)> =
        std::cell::RefCell::new(Default::default());
}

/// IKA-471: one damage calculation's whole input, counted against the inputs already met in
/// this fill (`count-damage`).
#[cfg(feature = "count-damage")]
pub fn count_damage(input: &str) {
    let mut h = fnv();
    h.write(input.as_bytes());
    let key = h.finish();
    DAMAGE_SEEN.with(|seen| {
        let mut seen = seen.borrow_mut();
        seen.1 += 1;
        if !seen.0.insert(key) {
            seen.2 += 1;
        }
    });
}

#[cfg(feature = "count-damage")]
fn damage_new_fill() {
    DAMAGE_SEEN.with(|seen| seen.borrow_mut().0.clear());
}

struct Fnv(u64);

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn fnv() -> Fnv {
    Fnv(0xcbf2_9ce4_8422_2325)
}

fn eat_str(h: &mut Fnv, s: &str) {
    h.write(s.as_bytes());
    h.write(&[0xff]);
}

fn eat_position(h: &mut Fnv, pos: &Position) {
    eat_str(h, &format!("{pos:?}"));
}

/// Every pause of `result` resumed, as `encoded_node::resume_all` does, hashed into `h` when
/// there is one. Returns the turns resolved.
fn walk<'a>(reg: &'a Reg, result: &TurnResult<'a>, depth: usize, h: &mut Option<Fnv>) -> usize {
    let mut turns = 0;
    if let Some(h) = h.as_mut() {
        h.write_u8(result.exact as u8);
        h.write_usize(result.branches.len());
        for b in &result.branches {
            h.write_u64(b.probability.to_bits());
            eat_position(h, &b.position);
        }
        for note in &result.unmodelled {
            eat_str(h, note);
        }
        h.write_usize(result.suspended.len());
    }
    for pause in &result.suspended {
        if let Some(h) = h.as_mut() {
            h.write_u64(pause.probability.to_bits());
            eat_position(h, &pause.turn.pos);
        }
        if depth >= 4 {
            continue;
        }
        let shared = sharing_budget(pause, result.suspended.len());
        match resume_alternatives(reg, shared.as_ref().unwrap_or(pause)) {
            Err(e) => {
                if let Some(h) = h.as_mut() {
                    eat_str(h, &e);
                }
            }
            Ok((chooser, options)) => {
                if let Some(h) = h.as_mut() {
                    h.write_usize(chooser.map(|c| c + 1).unwrap_or(0));
                    h.write_usize(options.len());
                }
                for one in &options {
                    turns += 1 + walk(reg, one, depth + 1, h);
                }
            }
        }
    }
    turns
}

/// A sampler of the calling thread's stack (feature `sampler`, Windows only): another thread
/// suspends it every millisecond or so, reads its context and unwinds it with the x64 unwind
/// tables (`RtlVirtualUnwind`). `stop` writes the samples to `path` (per sample the frame count,
/// then the absolute return addresses, innermost first) and `path.modules` (base, size, name).
#[cfg(feature = "sampler")]
pub(crate) mod sampler {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    type Handle = *mut core::ffi::c_void;
    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct Context {
        raw: [u8; 1232],
    }
    #[repr(C)]
    struct ModuleInfo {
        base: *mut core::ffi::c_void,
        size: u32,
        entry: *mut core::ffi::c_void,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> Handle;
        fn GetCurrentProcess() -> Handle;
        fn DuplicateHandle(sp: Handle, sh: Handle, tp: Handle, th: *mut Handle, access: u32, inherit: i32, options: u32) -> i32;
        fn SuspendThread(h: Handle) -> u32;
        fn ResumeThread(h: Handle) -> u32;
        fn GetThreadContext(h: Handle, ctx: *mut Context) -> i32;
        fn RtlLookupFunctionEntry(pc: u64, image_base: *mut u64, history: *mut core::ffi::c_void) -> *mut core::ffi::c_void;
        fn RtlVirtualUnwind(
            kind: u32, image_base: u64, pc: u64, entry: *mut core::ffi::c_void, ctx: *mut Context,
            handler_data: *mut *mut core::ffi::c_void, establisher: *mut u64, pointers: *mut core::ffi::c_void,
        ) -> *mut core::ffi::c_void;
        fn K32EnumProcessModules(p: Handle, mods: *mut Handle, cb: u32, needed: *mut u32) -> i32;
        fn K32GetModuleInformation(p: Handle, m: Handle, info: *mut ModuleInfo, cb: u32) -> i32;
        fn K32GetModuleBaseNameW(p: Handle, m: Handle, name: *mut u16, size: u32) -> u32;
    }

    const MAX_FRAMES: usize = 48;

    pub struct Sampler {
        stop: Arc<AtomicBool>,
        join: Option<std::thread::JoinHandle<Vec<u64>>>,
    }

    struct Send_(Handle);
    unsafe impl Send for Send_ {}

    fn rd(ctx: &Context, at: usize) -> u64 {
        u64::from_le_bytes(ctx.raw[at..at + 8].try_into().unwrap())
    }
    fn wr(ctx: &mut Context, at: usize, v: u64) {
        ctx.raw[at..at + 8].copy_from_slice(&v.to_le_bytes());
    }

    pub fn start() -> Sampler {
        let mut handle: Handle = std::ptr::null_mut();
        unsafe {
            DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &mut handle, 0, 0, 2);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let target = Send_(handle);
        let join = std::thread::spawn(move || {
            let target = target;
            // Reserved up front: nothing on this thread may allocate while the other is
            // suspended, because it may be suspended inside the heap holding its lock.
            let mut out: Vec<u64> = Vec::with_capacity(1 << 24);
            let mut frames = [0u64; MAX_FRAMES];
            let mut ctx = Context { raw: [0; 1232] };
            while !flag.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(1));
                let mut n = 0usize;
                unsafe {
                    if SuspendThread(target.0) == u32::MAX {
                        continue;
                    }
                    ctx.raw[0x30..0x34].copy_from_slice(&0x0010_000Bu32.to_le_bytes());
                    if GetThreadContext(target.0, &mut ctx) != 0 {
                        let mut c = ctx;
                        while n < MAX_FRAMES {
                            let rip = rd(&c, 0xF8);
                            if rip == 0 {
                                break;
                            }
                            frames[n] = rip;
                            n += 1;
                            let mut image_base = 0u64;
                            let entry = RtlLookupFunctionEntry(rip, &mut image_base, std::ptr::null_mut());
                            if entry.is_null() {
                                let rsp = rd(&c, 0x98);
                                wr(&mut c, 0xF8, *(rsp as *const u64));
                                wr(&mut c, 0x98, rsp + 8);
                            } else {
                                let mut data = std::ptr::null_mut();
                                let mut est = 0u64;
                                RtlVirtualUnwind(0, image_base, rip, entry, &mut c, &mut data, &mut est, std::ptr::null_mut());
                            }
                        }
                    }
                    ResumeThread(target.0);
                }
                if n > 0 && out.len() + n + 1 <= out.capacity() {
                    out.push(n as u64);
                    out.extend_from_slice(&frames[..n]);
                }
            }
            out
        });
        Sampler { stop, join: Some(join) }
    }

    impl Sampler {
        pub fn stop(mut self, path: &str) {
            self.stop.store(true, Ordering::Relaxed);
            let samples = self.join.take().unwrap().join().unwrap();
            let mut bytes = Vec::with_capacity(samples.len() * 8);
            let mut count = 0;
            let mut k = 0;
            while k < samples.len() {
                count += 1;
                k += samples[k] as usize + 1;
            }
            for s in &samples {
                bytes.extend_from_slice(&s.to_le_bytes());
            }
            std::fs::write(path, bytes).unwrap();
            let mut listing = String::new();
            unsafe {
                let process = GetCurrentProcess();
                let mut mods: Vec<Handle> = vec![std::ptr::null_mut(); 1024];
                let mut needed = 0u32;
                K32EnumProcessModules(process, mods.as_mut_ptr(), (mods.len() * 8) as u32, &mut needed);
                for m in mods.iter().take(needed as usize / 8) {
                    let mut info = ModuleInfo { base: std::ptr::null_mut(), size: 0, entry: std::ptr::null_mut() };
                    K32GetModuleInformation(process, *m, &mut info, std::mem::size_of::<ModuleInfo>() as u32);
                    let mut name = [0u16; 260];
                    let len = K32GetModuleBaseNameW(process, *m, name.as_mut_ptr(), 260) as usize;
                    listing.push_str(&format!("{} {} {}
", info.base as u64, info.size, String::from_utf16_lossy(&name[..len])));
                }
            }
            std::fs::write(format!("{path}.modules"), listing).unwrap();
            eprintln!("samples {count} -> {path}");
        }
    }
}

pub fn main(args: &[String]) {
    let reg = Reg::load(&args[0]).unwrap_or_else(|e| {
        eprintln!("regulation: {e}");
        std::process::exit(1);
    });
    let mut files = Vec::new();
    let mut repeats = 1usize;
    let mut digest = false;
    let mut limit = usize::MAX;
    let mut k = 1;
    while k < args.len() {
        match args[k].as_str() {
            "--repeats" => {
                repeats = args[k + 1].parse().unwrap();
                k += 1;
            }
            "--limit" => {
                limit = args[k + 1].parse().unwrap();
                k += 1;
            }
            "--digest" => digest = true,
            // IKA-471: how many slots may have moved in a state the fill cache keeps; `off` for none.
            "--max-acted" => {
                let n = if args[k + 1] == "off" { usize::MAX } else { args[k + 1].parse().unwrap() };
                crate::resolve::fill_cache::MAX_ACTED.store(n, std::sync::atomic::Ordering::Relaxed);
                k += 1;
            }
            other => files.push(other.to_string()),
        }
        k += 1;
    }
    let mut requests = Vec::new();
    'files: for file in &files {
        let text = std::fs::read_to_string(file).expect("dump");
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if requests.len() >= limit {
                break 'files;
            }
            let value: Value = serde_json::from_str(line).expect("dump line");
            requests.push(crate::node::parse_request(&value).expect("request"));
        }
    }
    let cells: usize = requests.iter().map(|r| r.wanted_cells().len()).sum();
    eprintln!(
        "fills {} cells {}; bytes: Turn {} Position {} Pokemon {} Battler {} FieldState {}",
        requests.len(),
        cells,
        std::mem::size_of::<crate::resolve::Turn>(),
        std::mem::size_of::<Position>(),
        std::mem::size_of::<crate::position::Pokemon>(),
        std::mem::size_of::<crate::battler::Battler>(),
        std::mem::size_of::<crate::battler::FieldState>(),
    );
    if digest {
        let mut whole = fnv();
        let mut turns = 0usize;
        for (index, request) in requests.iter().enumerate() {
            let _cache = crate::resolve::fill_cache::begin(&reg);
            let mut h = Some(fnv());
            for (i, j) in request.wanted_cells() {
                let actions = [request.ours[i].clone(), request.theirs[j].clone()];
                let hh = h.as_mut().unwrap();
                hh.write_usize(i);
                hh.write_usize(j);
                turns += 1;
                match resolve_turn(&reg, &request.position, &actions, request.budget) {
                    Err(e) => eat_str(h.as_mut().unwrap(), &e),
                    Ok(result) => turns += walk(&reg, &result, 0, &mut h),
                }
            }
            let fill_hash = h.unwrap().finish();
            println!("fill {index} {fill_hash:016x}");
            whole.write_u64(fill_hash);
        }
        println!("digest {:016x} turns {turns}", whole.finish());
        return;
    }
    let relaxed = std::sync::atomic::Ordering::Relaxed;
    crate::resolve::reset_counters();
    let damage_before = crate::damage::CALLS.load(relaxed);
    let clones_before = crate::position::CLONES.load(relaxed);
    let builds_before = crate::battler::BUILDS.load(relaxed);
    let mut turns = 0usize;
    #[cfg(feature = "sampler")]
    let sampling = std::env::var("POKEURAOU_SAMPLES").ok().map(|path| (path, sampler::start()));
    let started = std::time::Instant::now();
    for _ in 0..repeats {
        for request in &requests {
            #[cfg(feature = "count-damage")]
            damage_new_fill();
            #[cfg(feature = "count-execute")]
            crate::resolve::execute_count::new_fill();
            let _cache = crate::resolve::fill_cache::begin(&reg);
            for (i, j) in request.wanted_cells() {
                let actions = [request.ours[i].clone(), request.theirs[j].clone()];
                turns += 1;
                if let Ok(result) = resolve_turn(&reg, &request.position, &actions, request.budget) {
                    turns += walk(&reg, &result, 0, &mut None);
                }
            }
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    #[cfg(feature = "sampler")]
    if let Some((path, running)) = sampling {
        running.stop(&path);
    }
    let n = turns as f64;
    println!(
        "turns {turns} in {elapsed:.3} s = {:.2} us/turn; per turn: damage calls {:.2} position clones {:.2} battler builds {:.2}",
        elapsed / n * 1e6,
        (crate::damage::CALLS.load(relaxed) - damage_before) as f64 / n,
        (crate::position::CLONES.load(relaxed) - clones_before) as f64 / n,
        (crate::battler::BUILDS.load(relaxed) - builds_before) as f64 / n,
    );
    #[cfg(feature = "count-execute")]
    crate::resolve::execute_count::COUNTS.with(|c| {
        let c = c.borrow();
        let pct = |x: u64| x as f64 / c.ns.max(1) as f64 * 100.0;
        println!(
            "execute calls {} ({:.1} ms); state+action met earlier in the fill: exact {} ({:.1}% of execute time),              attacks masked {} ({:.1}%)",
            c.calls, c.ns as f64 / 1e6, c.exact, pct(c.exact_ns), c.masked, pct(c.masked_ns)
        );
        println!(
            "residual phases {} ({:.1} ms), of a state met earlier in the fill {} ({:.1}% of their time)",
            c.residual[0],
            c.residual[1] as f64 / 1e6,
            c.residual[2],
            c.residual[3] as f64 / c.residual[1].max(1) as f64 * 100.0
        );
        for (depth, row) in c.by_depth.iter().enumerate() {
            println!(
                "  slots acted before {depth}: calls {} ({:.1}% of execute time), masked repeats {} ({:.1}%)",
                row[0], pct(row[1]), row[2], pct(row[3])
            );
        }
    });
    #[cfg(feature = "count-damage")]
    DAMAGE_SEEN.with(|seen| {
        let seen = seen.borrow();
        println!("damage calls {} repeats of an input met earlier in the same fill {} ({:.1}%)",
            seen.1, seen.2, seen.2 as f64 / seen.1.max(1) as f64 * 100.0);
    });
    #[cfg(feature = "profile")]
    for (index, name) in crate::resolve::PHASE_NAMES.iter().enumerate() {
        let ns = crate::resolve::PHASE_NS[index].load(relaxed) as f64;
        println!("  {name:<16} {:>7.2} us/turn ({:>4.1}%)", ns / n / 1000.0, ns / 1000.0 / (elapsed * 1e6) * 100.0);
    }
}
