//! IKA-470: a dump of the LPs the port solves, and a bench that solves a dump under HiGHS
//! option sets (feature `lp-dump`).
//!
//! * `POKEURAOU_LP_DUMP=<path>` writes `<path>.<pid>`: per LP four u32 (`n_ub`, rows, cols,
//!   `free_from`), then f64 `c[cols]`, `a[rows*cols]` (row-major), `b_ub[n_ub]`, `b_eq[rows-n_ub]`.
//!   `POKEURAOU_LP_DUMP_EVERY=k` keeps one LP in k.
//! * `lp-bench <dump> [--repeats N] [--limit N] -- <name>:<opt=v,opt=v> ...`: each set solves every
//!   LP N times (the fastest of N per LP), and is compared with the first set.

use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

static FILES: Mutex<Vec<(&'static str, std::fs::File)>> = Mutex::new(Vec::new());

thread_local! {
    /// Which request the LPs of this thread belong to; a dump file per tag (`<path>.<tag>.<pid>`).
    pub static TAG: std::cell::Cell<&'static str> = const { std::cell::Cell::new("node") };
}

/// Marks the LPs this thread solves until the guard drops (the replacement node, the selection solve).
pub fn tag(name: &'static str) -> TagGuard {
    TAG.with(|t| t.set(name));
    TagGuard
}

pub struct TagGuard;

impl Drop for TagGuard {
    fn drop(&mut self) {
        TAG.with(|t| t.set("node"));
    }
}
static SEEN: AtomicU64 = AtomicU64::new(0);

pub fn dump(c: &[f64], a: &[f64], n_ub: usize, b_ub: &[f64], b_eq: &[f64], free_from: usize) {
    let Ok(base) = std::env::var("POKEURAOU_LP_DUMP") else { return };
    let every: u64 = std::env::var("POKEURAOU_LP_DUMP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    if SEEN.fetch_add(1, Ordering::Relaxed) % every.max(1) != 0 {
        return;
    }
    let which = TAG.with(|t| t.get());
    let mut guard = FILES.lock().unwrap();
    let at = match guard.iter().position(|(t, _)| *t == which) {
        Some(at) => at,
        None => {
            let file = std::fs::File::create(format!("{base}.{which}.{}", std::process::id())).expect("dump file");
            guard.push((which, file));
            guard.len() - 1
        }
    };
    let file = &mut guard[at].1;
    let mut buf: Vec<u8> = Vec::with_capacity(16 + 8 * (c.len() + a.len() + b_ub.len() + b_eq.len()));
    for v in [n_ub, n_ub + b_eq.len(), c.len(), free_from] {
        buf.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for part in [c, a, b_ub, b_eq] {
        for v in part {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    file.write_all(&buf).unwrap();
}

struct Lp {
    n_ub: usize,
    rows: usize,
    cols: usize,
    free_from: usize,
    c: Vec<f64>,
    a: Vec<f64>,
    b_ub: Vec<f64>,
    b_eq: Vec<f64>,
}

fn read(path: &str, limit: usize) -> Vec<Lp> {
    let bytes = std::fs::read(path).expect("dump");
    let mut at = 0;
    let mut out = Vec::new();
    let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    while at + 16 <= bytes.len() && out.len() < limit {
        let (n_ub, rows, cols, free_from) = (u32_at(at), u32_at(at + 4), u32_at(at + 8), u32_at(at + 12));
        at += 16;
        let mut take = |n: usize| -> Vec<f64> {
            let v = (0..n)
                .map(|i| f64::from_le_bytes(bytes[at + 8 * i..at + 8 * i + 8].try_into().unwrap()))
                .collect();
            at += 8 * n;
            v
        };
        let c = take(cols);
        let a = take(rows * cols);
        let b_ub = take(n_ub);
        let b_eq = take(rows - n_ub);
        out.push(Lp { n_ub, rows, cols, free_from, c, a, b_ub, b_eq });
    }
    out
}

pub fn main(args: &[String]) {
    let path = &args[0];
    let (mut repeats, mut limit) = (3usize, usize::MAX);
    let mut sets: Vec<(String, Vec<(String, String)>)> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--repeats" => {
                repeats = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--limit" => {
                limit = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--" => i += 1,
            spec => {
                let (name, opts) = spec.split_once(':').unwrap_or((spec, ""));
                let opts = opts
                    .split(',')
                    .filter_map(|kv| kv.split_once('='))
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect();
                sets.push((name.to_string(), opts));
                i += 1;
            }
        }
    }
    let lps = read(path, limit);
    let cells: usize = lps.iter().map(|l| l.rows * l.cols).sum();
    println!("set\tus/LP\trun us/LP\tfailed\tmax|dobj|\tobj_worse\tsolution_changed\tmax|dx|\tmax_viol\tp95 us\tp99 us");
    // Per LP, every set in turn (so a drift of the machine falls on all sets alike): one
    // untimed solve makes the set's instance, then the fastest of `repeats` timed solves.
    let ns = sets.len();
    let mut times = vec![vec![0.0f64; lps.len()]; ns];
    let mut runs = vec![vec![0.0f64; lps.len()]; ns];
    let mut objs = vec![vec![f64::NAN; lps.len()]; ns];
    let mut xs: Vec<Vec<Option<Vec<f64>>>> = vec![vec![None; lps.len()]; ns];
    let mut viol = vec![0.0f64; ns];
    for (k, lp) in lps.iter().enumerate() {
        for (s, (_, opts)) in sets.iter().enumerate() {
            crate::lp::set_extra_options(opts.clone());
            let _ = crate::lp::solve_lp(&lp.c, &lp.a, lp.n_ub, &lp.b_ub, &lp.b_eq, lp.free_from);
            let (mut best, mut best_run) = (f64::INFINITY, 0.0);
            let mut answer = None;
            for _ in 0..repeats {
                let r0 = crate::lp::NS[2].load(Ordering::Relaxed);
                let t = std::time::Instant::now();
                let r = crate::lp::solve_lp(&lp.c, &lp.a, lp.n_ub, &lp.b_ub, &lp.b_eq, lp.free_from);
                let dt = t.elapsed().as_secs_f64();
                let r1 = crate::lp::NS[2].load(Ordering::Relaxed);
                if dt < best {
                    best = dt;
                    best_run = (r1 - r0) as f64 * 1e-9;
                }
                answer = Some(r);
            }
            times[s][k] = best;
            runs[s][k] = best_run;
            if let Some(Ok(x)) = answer {
                objs[s][k] = lp.c.iter().zip(&x).map(|(c, x)| c * x).sum::<f64>();
                // The worst violation of `a x <= b` (first n_ub rows), `a x = b` and `x >= 0`.
                let mut v = 0.0f64;
                for r in 0..lp.rows {
                    let ax: f64 = (0..lp.cols).map(|j| lp.a[r * lp.cols + j] * x[j]).sum();
                    v = v.max(if r < lp.n_ub { ax - lp.b_ub[r] } else { (ax - lp.b_eq[r - lp.n_ub]).abs() });
                }
                for j in 0..lp.free_from {
                    v = v.max(-x[j]);
                }
                viol[s] = viol[s].max(v);
                if std::env::var("LPBENCH_PRINT").is_ok() {
                    println!("lp {k} set {s} rows {} cols {} obj {:.12}", lp.rows, lp.cols, objs[s][k]);
                }
                xs[s][k] = Some(x);
            }
        }
    }
    for (s, (name, _)) in sets.iter().enumerate() {
        let failed = xs[s].iter().filter(|x| x.is_none()).count();
        let (mut dobj, mut worse, mut changed, mut dx) = (0.0f64, 0, 0, 0.0f64);
        for k in 0..lps.len() {
            if objs[s][k].is_nan() || objs[0][k].is_nan() {
                continue;
            }
            dobj = dobj.max((objs[s][k] - objs[0][k]).abs());
            if objs[s][k] > objs[0][k] + 1e-9 {
                worse += 1; // minimisation: a larger objective is a worse value
            }
            if let (Some(x), Some(y)) = (&xs[s][k], &xs[0][k]) {
                let d = x.iter().zip(y).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                dx = dx.max(d);
                if d > 1e-9 {
                    changed += 1;
                }
            }
        }
        let mut sorted = times[s].clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize] * 1e6;
        println!(
            "{name}\t{:.1}\t{:.1}\t{failed}\t{dobj:.2e}\t{worse}\t{changed}\t{dx:.2e}\t{:.1e}\t{:.0}\t{:.0}",
            times[s].iter().sum::<f64>() * 1e6 / lps.len() as f64,
            runs[s].iter().sum::<f64>() * 1e6 / lps.len() as f64,
            viol[s],
            q(0.95),
            q(0.99)
        );
    }
}
