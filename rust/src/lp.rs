//! IKA-381: the equilibria of matrix games, by HiGHS, as `equilibrium.py` solves them.
//!
//! The same model handed to the same solver with the same options: `equilibrium._lp` builds
//! a column-wise matrix from the dense one (the nonzeros, column by column), bounds `x >= 0`
//! below `free_from` and free from it on, and runs HiGHS with `linprog(method="highs")`'s
//! defaults -- presolve on, the dual simplex. The HiGHS is the one scipy 1.18.1 builds
//! (`vendor/HiGHS`, 1.12.0, `build.rs`). Success is linprog's: HiGHS optimal, then linprog's
//! residual check (`_check_result`, 1e-9). The strategies are cleaned as `_clean` cleans
//! them, with numpy's pairwise sum, so a solve here is the Python solve to the bit wherever
//! HiGHS, built by another compiler, takes the same pivots (counted in records/IKA-381.md).
//!
//! Two commands use it, both off unless a caller asks (`portlp.py`):
//! * `lp` -- a list of games (a matrix, or a Bayesian game's matrices and weights), each
//!   answered with both LPs' values and the cleaned strategies.
//! * `folds` -- sub-games whose leaves a caller has scored: each matrix folded from its
//!   node's spans (the bytes `fills` sent) and the leaf values, the way `port._folded` folds
//!   it (numpy's dot, in the order this machine's OpenBLAS takes it), then solved; the value.

use serde_json::{json, Value};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::atomic::{AtomicU64, Ordering};

type HighsInt = c_int;

extern "C" {
    fn Highs_create() -> *mut c_void;
    fn Highs_destroy(highs: *mut c_void);
    fn Highs_run(highs: *mut c_void) -> HighsInt;
    #[allow(clippy::too_many_arguments)]
    fn Highs_passLp(
        highs: *mut c_void,
        num_col: HighsInt,
        num_row: HighsInt,
        num_nz: HighsInt,
        a_format: HighsInt,
        sense: HighsInt,
        offset: f64,
        col_cost: *const f64,
        col_lower: *const f64,
        col_upper: *const f64,
        row_lower: *const f64,
        row_upper: *const f64,
        a_start: *const HighsInt,
        a_index: *const HighsInt,
        a_value: *const f64,
    ) -> HighsInt;
    fn Highs_setBoolOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt)
        -> HighsInt;
    fn Highs_setIntOptionValue(highs: *mut c_void, option: *const c_char, value: HighsInt)
        -> HighsInt;
    fn Highs_setStringOptionValue(
        highs: *mut c_void,
        option: *const c_char,
        value: *const c_char,
    ) -> HighsInt;
    fn Highs_getModelStatus(highs: *const c_void) -> HighsInt;
    fn Highs_getSolution(
        highs: *const c_void,
        col_value: *mut f64,
        col_dual: *mut f64,
        row_value: *mut f64,
        row_dual: *mut f64,
    ) -> HighsInt;
    fn Highs_getObjectiveValue(highs: *const c_void) -> f64;
}

const STATUS_ERROR: HighsInt = -1;
const MATRIX_COLWISE: HighsInt = 1;
const SENSE_MINIMIZE: HighsInt = 1;
const MODEL_OPTIMAL: HighsInt = 7;
const SIMPLEX_DUAL: HighsInt = 1;

/// LPs solved by this process (the positive control: a caller's LPs went through here).
pub static LPS: AtomicU64 = AtomicU64::new(0);
/// Games (`lp`) and sub-games (`folds`) answered.
pub static GAMES: AtomicU64 = AtomicU64::new(0);
pub static FOLDS: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds in making instances, passing models, running and reading solutions (a probe).
pub static NS: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

fn tick(slot: usize, since: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    NS[slot].fetch_add((now - since).as_nanos() as u64, Ordering::Relaxed);
    now
}

/// A HiGHS instance with `equilibrium._highs_options` set (kept per thread, `Held`).
struct Instance(*mut c_void);

impl Instance {
    fn new() -> Instance {
        // SAFETY: `Highs_create` returns a fresh instance or aborts.
        let highs = unsafe { Highs_create() };
        let set_bool = |name: &[u8], value: bool| unsafe {
            Highs_setBoolOptionValue(highs, name.as_ptr() as *const c_char, value as HighsInt)
        };
        let set_int = |name: &[u8], value: HighsInt| unsafe {
            Highs_setIntOptionValue(highs, name.as_ptr() as *const c_char, value)
        };
        // `equilibrium._highs_options`, in its order.
        // SAFETY: the names and values are NUL-terminated literals.
        unsafe {
            Highs_setStringOptionValue(
                highs,
                b"presolve\0".as_ptr() as *const c_char,
                b"on\0".as_ptr() as *const c_char,
            );
        }
        set_int(b"highs_debug_level\0", 0);
        set_bool(b"log_to_console\0", false);
        set_bool(b"output_flag\0", false);
        set_int(b"simplex_strategy\0", SIMPLEX_DUAL);
        // One thread: the dual simplex is serial (`simplex_strategy` 1), and a scheduler of
        // more would be started and stopped with every instance (`Highs_destroy`). scipy
        // leaves the default; the captured LPs are the same to the bit either way.
        set_int(b"threads\0", THREADS);
        Instance(highs)
    }
}

/// HiGHS's `threads` option for every instance (see `Instance::new`).
const THREADS: HighsInt = 1;

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: made by `Highs_create`, dropped once.
        unsafe { Highs_destroy(self.0) }
    }
}

extern "C" {
    fn Highs_clearModel(highs: *mut c_void) -> HighsInt;
}

thread_local! {
    static KEPT: std::cell::RefCell<Option<Instance>> = const { std::cell::RefCell::new(None) };
}

/// A thread keeps its instance and clears its model between LPs (`Highs_clearModel`: the
/// model, the basis and the solution go; the options stay). Python makes a fresh `_Highs` per
/// LP; the kept one answered 59,050 captured LPs to the same bits and saves making and
/// destroying one (45 us of 170 an LP). `POKEURAOU_LP_FRESH=1` makes a fresh one per LP.
fn reuse() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("POKEURAOU_LP_FRESH").is_ok_and(|v| v == "1"))
}

/// The instance an LP runs on: fresh, or the thread's kept one with its model cleared.
struct Held {
    instance: Option<Instance>,
    keep: bool,
}

impl Held {
    fn take() -> Held {
        if !reuse() {
            return Held { instance: Some(Instance::new()), keep: false };
        }
        let kept = KEPT.with(|k| k.borrow_mut().take());
        let instance = match kept {
            Some(instance) => {
                // SAFETY: a live instance.
                unsafe { Highs_clearModel(instance.0) };
                instance
            }
            None => Instance::new(),
        };
        Held { instance: Some(instance), keep: true }
    }
}

impl std::ops::Deref for Held {
    type Target = Instance;
    fn deref(&self) -> &Instance {
        self.instance.as_ref().unwrap()
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        if self.keep {
            let instance = self.instance.take();
            KEPT.with(|k| *k.borrow_mut() = instance);
        }
    }
}

/// `equilibrium._lp`: minimise `c.x` subject to `a_ub x <= b_ub`, `a_eq x = b_eq`, `x >= 0`
/// below `free_from` and free from it on. `a` is the dense `[a_ub; a_eq]`, row-major, with
/// `n_ub` rows of the first. The solution, or linprog's failure message.
fn solve_lp(
    c: &[f64],
    a: &[f64],
    n_ub: usize,
    b_ub: &[f64],
    b_eq: &[f64],
    free_from: usize,
) -> Result<Vec<f64>, String> {
    let cols = c.len();
    let rows = n_ub + b_eq.len();
    debug_assert_eq!(a.len(), rows * cols);
    let inf = f64::INFINITY;
    // `columns = a.T; nonzero = columns != 0`: column by column, the rows in order.
    let mut start: Vec<HighsInt> = Vec::with_capacity(cols + 1);
    let mut index: Vec<HighsInt> = Vec::with_capacity(rows * cols);
    let mut value: Vec<f64> = Vec::with_capacity(rows * cols);
    start.push(0);
    for k in 0..cols {
        for r in 0..rows {
            let v = a[r * cols + k];
            if v != 0.0 {
                index.push(r as HighsInt);
                value.push(v);
            }
        }
        start.push(index.len() as HighsInt);
    }
    let mut lower = vec![0.0; cols];
    for v in lower.iter_mut().skip(free_from) {
        *v = -inf;
    }
    let upper = vec![inf; cols];
    let mut row_lower = vec![-inf; n_ub];
    row_lower.extend_from_slice(b_eq);
    let mut row_upper = b_ub.to_vec();
    row_upper.extend_from_slice(b_eq);
    LPS.fetch_add(1, Ordering::Relaxed);
    let clock = std::time::Instant::now();
    let highs = Held::take();
    let clock = tick(0, clock);
    // SAFETY: every array is as long as the counts passed with it.
    let passed = unsafe {
        Highs_passLp(
            highs.0,
            cols as HighsInt,
            rows as HighsInt,
            value.len() as HighsInt,
            MATRIX_COLWISE,
            SENSE_MINIMIZE,
            0.0,
            c.as_ptr(),
            lower.as_ptr(),
            upper.as_ptr(),
            row_lower.as_ptr(),
            row_upper.as_ptr(),
            start.as_ptr(),
            index.as_ptr(),
            value.as_ptr(),
        )
    };
    if passed == STATUS_ERROR {
        return Err("HiGHS refused the model".into());
    }
    let clock = tick(1, clock);
    // SAFETY: a model was passed.
    let ran = unsafe { Highs_run(highs.0) };
    let clock = tick(2, clock);
    if ran == STATUS_ERROR {
        let status = unsafe { Highs_getModelStatus(highs.0) };
        return Err(format!("HiGHS model status {status}"));
    }
    let status = unsafe { Highs_getModelStatus(highs.0) };
    if status != MODEL_OPTIMAL {
        return Err(format!("HiGHS model status {status}"));
    }
    let mut x = vec![0.0; cols];
    let mut col_dual = vec![0.0; cols];
    let mut row_value = vec![0.0; rows];
    let mut row_dual = vec![0.0; rows];
    // SAFETY: the buffers are as long as the model's columns and rows.
    unsafe {
        Highs_getSolution(
            highs.0,
            x.as_mut_ptr(),
            col_dual.as_mut_ptr(),
            row_value.as_mut_ptr(),
            row_dual.as_mut_ptr(),
        );
    }
    let fun = unsafe { Highs_getObjectiveValue(highs.0) };
    drop(highs);
    tick(3, clock);
    // linprog's `_check_result`, status 0, tolerance 1e-9.
    let tol = (1e-9f64).sqrt() * 10.0;
    let slack: Vec<f64> = row_upper.iter().zip(&row_value).map(|(u, v)| u - v).collect();
    let nans = x.iter().any(|v| v.is_nan()) || fun.is_nan() || slack.iter().any(|v| v.is_nan());
    let feasible = !nans && {
        let bounds_ok = x
            .iter()
            .zip(lower.iter().zip(&upper))
            .all(|(v, (lo, hi))| *v >= lo - tol && *v <= hi + tol);
        let slack_ok = !slack[..n_ub].iter().any(|s| *s < -tol);
        let con_ok = !slack[n_ub..].iter().any(|s| s.abs() > tol);
        bounds_ok && slack_ok && con_ok
    };
    if !feasible {
        return Err(format!(
            "The solution does not satisfy the constraints within the required tolerance of {tol:.2E}"
        ));
    }
    Ok(x)
}

/// `equilibrium._maximin`: the row player's LP of `payoff` (m x n, row-major).
fn maximin(payoff: &[f64], m: usize, n: usize) -> Result<(f64, Vec<f64>), String> {
    let cols = m + 1;
    let mut c = vec![0.0; cols];
    c[m] = -1.0;
    let mut a = vec![0.0; (n + 1) * cols];
    for j in 0..n {
        for i in 0..m {
            a[j * cols + i] = -payoff[i * n + j];
        }
        a[j * cols + m] = 1.0;
    }
    for i in 0..m {
        a[n * cols + i] = 1.0;
    }
    let x = solve_lp(&c, &a, n, &vec![0.0; n], &[1.0], m)?;
    Ok((x[m], x[..m].to_vec()))
}

/// numpy's sum of a float64 vector (`pairwise_sum`, the reduction `ndarray.sum` takes).
pub fn numpy_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        numpy_sum(&a[..n2]) + numpy_sum(&a[n2..])
    }
}

/// `equilibrium._clean`: small negatives and dust to zero, renormalised; uniform if none.
pub fn clean(p: &[f64], eps: f64) -> Vec<f64> {
    let mut q: Vec<f64> = p.iter().map(|&v| if v < eps { 0.0 } else { v }).collect();
    let total = numpy_sum(&q);
    if total <= 0.0 {
        return vec![1.0 / q.len() as f64; q.len()];
    }
    for v in &mut q {
        *v /= total;
    }
    q
}

/// A solved game: both LPs' values and the cleaned strategies (`equilibrium.solve`'s parts;
/// the caller builds the rest as `solve` does).
pub struct Solved {
    pub value_row: f64,
    pub value_col: f64,
    pub x: Vec<f64>,
    pub ys: Vec<Vec<f64>>,
}

impl Solved {
    /// `solve`'s value: the mean of the two LPs'.
    pub fn value(&self) -> f64 {
        0.5 * (self.value_row + self.value_col)
    }
}

/// Why a game has no answer: `Invalid` is `solve`'s ValueError, `Failed` its EquilibriumError.
#[derive(Debug)]
pub enum Unsolved {
    Invalid(String),
    Failed(String),
}

/// `equilibrium.solve` of an m x n matrix (row-major).
pub fn solve(payoff: &[f64], m: usize, n: usize, eps: f64) -> Result<Solved, Unsolved> {
    if m == 0 || n == 0 || payoff.len() != m * n {
        return Err(Unsolved::Invalid(format!("payoff must be a non-empty 2-D array, got {m} x {n}")));
    }
    if payoff.iter().any(|v| !v.is_finite()) {
        return Err(Unsolved::Invalid("payoff contains non-finite entries".into()));
    }
    let (value_row, x_raw) =
        maximin(payoff, m, n).map_err(|why| Unsolved::Failed(format!("row LP failed: {why}")))?;
    // The column player minimises A: the row player of -A^T.
    let mut negated = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            negated[j * m + i] = -payoff[i * n + j];
        }
    }
    let (value_col_neg, y_raw) =
        maximin(&negated, n, m).map_err(|why| Unsolved::Failed(format!("row LP failed: {why}")))?;
    Ok(Solved {
        value_row,
        value_col: -value_col_neg,
        x: clean(&x_raw, eps),
        ys: vec![clean(&y_raw, eps)],
    })
}

/// `equilibrium.solve_bayesian`: `mats[k]` (m x n_k, row-major) against class k of weight
/// `weights[k]` (normalised here, as there).
pub fn solve_bayesian(
    mats: &[(Vec<f64>, usize, usize)],
    weights: &[f64],
    eps: f64,
) -> Result<Solved, Unsolved> {
    if mats.is_empty() {
        return Err(Unsolved::Invalid("no matrices to solve".into()));
    }
    let m = mats[0].1;
    if mats.iter().any(|(_, rows, _)| *rows != m) {
        return Err(Unsolved::Invalid("every class must offer us the same actions".into()));
    }
    for (data, rows, cols) in mats {
        if rows * cols == 0 || data.len() != rows * cols || data.iter().any(|v| !v.is_finite()) {
            return Err(Unsolved::Invalid("a payoff matrix is empty or non-finite".into()));
        }
    }
    if weights.len() != mats.len() {
        return Err(Unsolved::Invalid(format!("{} weights for {} matrices", weights.len(), mats.len())));
    }
    if weights.iter().cloned().fold(f64::INFINITY, f64::min) < 0.0 {
        return Err(Unsolved::Invalid("class weights must be non-negative".into()));
    }
    let total_w = numpy_sum(weights);
    let w: Vec<f64> = weights.iter().map(|v| v / total_w).collect();
    let k = mats.len();
    let columns: Vec<usize> = mats.iter().map(|(_, _, cols)| *cols).collect();
    let total: usize = columns.iter().sum();

    // `_bayesian_maximin`: x_0..x_{m-1}, then v_0..v_{k-1}.
    let row = {
        let cols = m + k;
        let mut c = vec![0.0; cols];
        for v in c.iter_mut().skip(m) {
            *v = -1.0;
        }
        let mut a = vec![0.0; (total + 1) * cols];
        let mut at = 0;
        for (index, (data, _, n)) in mats.iter().enumerate() {
            let scale = -w[index];
            for j in 0..*n {
                for i in 0..m {
                    a[(at + j) * cols + i] = scale * data[i * n + j];
                }
                a[(at + j) * cols + m + index] = 1.0;
            }
            at += n;
        }
        for i in 0..m {
            a[total * cols + i] = 1.0;
        }
        let x = solve_lp(&c, &a, total, &vec![0.0; total], &[1.0], m)
            .map_err(|why| Unsolved::Failed(format!("Bayesian row LP failed: {why}")))?;
        (numpy_sum(&x[m..]), x[..m].to_vec())
    };
    // `_bayesian_minimax`: y per class (concatenated), then u.
    let col = {
        let cols = total + 1;
        let mut c = vec![0.0; cols];
        c[total] = 1.0;
        let mut a = vec![0.0; (m + k) * cols];
        let mut offset = 0;
        for (index, (data, _, n)) in mats.iter().enumerate() {
            for i in 0..m {
                for j in 0..*n {
                    a[i * cols + offset + j] = w[index] * data[i * n + j];
                }
            }
            offset += n;
        }
        for i in 0..m {
            a[i * cols + total] = -1.0;
        }
        let mut offset = 0;
        for (index, n) in columns.iter().enumerate() {
            for j in 0..*n {
                a[(m + index) * cols + offset + j] = 1.0;
            }
            offset += n;
        }
        let x = solve_lp(&c, &a, m, &vec![0.0; m], &vec![1.0; k], total)
            .map_err(|why| Unsolved::Failed(format!("Bayesian column LP failed: {why}")))?;
        let mut ys = Vec::with_capacity(k);
        let mut offset = 0;
        for n in &columns {
            ys.push(clean(&x[offset..offset + n], eps));
            offset += n;
        }
        (x[total], ys)
    };
    Ok(Solved { value_row: row.0, value_col: col.0, x: clean(&row.1, eps), ys: col.1 })
}

// ---------------------------------------------------------------------------
// The fold: `port._folded`'s `values[indices] @ weights`, as numpy computes it here
// ---------------------------------------------------------------------------

/// numpy's 1-D float64 dot on this machine: OpenBLAS's SkylakeX `ddot` (numpy's
/// `scipy-openblas` picks that core on the Zen 5 it runs on). Blocks of 32 in four 8-wide
/// FMA accumulators, folded to four 4-wide ones, then blocks of 16 into those, their sum
/// ((a0 + a1) + a2) + a3, its halves, its pair; then the tail by FMA onto that. Checked
/// against numpy on random vectors of every length to 1,000 (`records/IKA-381.md`). Another
/// core (another machine) sums in another order, and the fold's values move in the last
/// place there -- the LP is the same either way.
pub fn numpy_dot(x: &[f64], y: &[f64]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("fma") {
            // SAFETY: the CPU has FMA.
            return unsafe { dot_fma(x, y) };
        }
    }
    dot_generic(x, y)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "fma")]
unsafe fn dot_fma(x: &[f64], y: &[f64]) -> f64 {
    dot_generic(x, y)
}

#[inline(always)]
fn dot_generic(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    let n1 = n & !15;
    let mut dot = 0.0;
    if n1 > 0 {
        let mut z = [[0.0f64; 8]; 4];
        let mut i = 0;
        let n32 = n1 & !31;
        while i < n32 {
            for (r, acc) in z.iter_mut().enumerate() {
                for (lane, a) in acc.iter_mut().enumerate() {
                    let k = i + 8 * r + lane;
                    *a = x[k].mul_add(y[k], *a);
                }
            }
            i += 32;
        }
        let mut acc = [[0.0f64; 4]; 4];
        for r in 0..4 {
            for k in 0..4 {
                acc[r][k] = z[r][k] + z[r][k + 4];
            }
        }
        while i < n1 {
            for (r, row) in acc.iter_mut().enumerate() {
                for (lane, a) in row.iter_mut().enumerate() {
                    let k = i + 4 * r + lane;
                    *a = x[k].mul_add(y[k], *a);
                }
            }
            i += 16;
        }
        let mut s = [0.0f64; 4];
        for k in 0..4 {
            s[k] = ((acc[0][k] + acc[1][k]) + acc[2][k]) + acc[3][k];
        }
        let h0 = s[0] + s[2];
        let h1 = s[1] + s[3];
        dot = h0 + h1;
    }
    for i in n1..n {
        dot = y[i].mul_add(x[i], dot);
    }
    dot
}

/// A sub-game's matrix from its node's spans (`encoded_node::write_body`'s span block:
/// every span's weights, then rows, columns and lengths, then every span's leaf indices)
/// and the leaves' values, as `port._folded` builds it (a span with no weights stays 0).
pub fn fold(
    block: &[u8],
    count: usize,
    total: usize,
    values: &[f64],
    m: usize,
    n: usize,
) -> Result<Vec<f64>, String> {
    let need = total * 8 + count * 12 + total * 4;
    if block.len() < need {
        return Err(format!("a span block of {} bytes for {count} spans of {total} leaves", block.len()));
    }
    let f64_at = |at: usize| f64::from_le_bytes(block[at..at + 8].try_into().unwrap());
    let u32_at = |at: usize| u32::from_le_bytes(block[at..at + 4].try_into().unwrap()) as usize;
    let weights_at = 0;
    let rows_at = total * 8;
    let cols_at = rows_at + count * 4;
    let lengths_at = cols_at + count * 4;
    let indices_at = lengths_at + count * 4;
    let mut payoff = vec![0.0; m * n];
    let mut start = 0;
    let mut xs: Vec<f64> = Vec::new();
    let mut ws: Vec<f64> = Vec::new();
    for s in 0..count {
        let (i, j, length) = (u32_at(rows_at + 4 * s), u32_at(cols_at + 4 * s), u32_at(lengths_at + 4 * s));
        let end = start + length;
        if end > total || i >= m || j >= n {
            return Err("a span outside its node".into());
        }
        if length > 0 {
            xs.clear();
            ws.clear();
            for t in start..end {
                let leaf = u32_at(indices_at + 4 * t);
                xs.push(*values.get(leaf).ok_or("a span's leaf past the values")?);
                ws.push(f64_at(weights_at + 8 * t));
            }
            payoff[i * n + j] = numpy_dot(&xs, &ws);
        }
        start = end;
    }
    Ok(payoff)
}

/// A cell whose turn stopped for a replacement (`fold.py`'s tree): chance is the weighted
/// mean of its parts, a replacement the option its chooser likes most.
pub enum FoldTree {
    Leaf(usize),
    Best(i64, Vec<FoldTree>),
    Avg(Vec<(f64, FoldTree)>),
}

/// JSON read with its numbers kept as text, for the fold trees: Python writes a weight as its
/// shortest round-trip text, and `str::parse::<f64>` (correctly rounded) reads back the double
/// the port first wrote -- serde_json's default reader is not always the nearest double.
enum Json<'a> {
    Num(&'a str),
    Arr(Vec<Json<'a>>),
    Obj(Vec<(&'a str, Json<'a>)>),
    Other,
}

struct Reader<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> Reader<'a> {
    fn skip(&mut self) {
        let bytes = self.text.as_bytes();
        while self.at < bytes.len() && bytes[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.text.as_bytes().get(self.at).copied()
    }

    fn expect(&mut self, byte: u8) -> Result<(), String> {
        if self.peek() == Some(byte) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("a fold tree's text: expected '{}' at {}", byte as char, self.at))
        }
    }

    fn string(&mut self) -> Result<&'a str, String> {
        self.expect(b'"')?;
        let bytes = self.text.as_bytes();
        let start = self.at;
        while self.at < bytes.len() && bytes[self.at] != b'"' {
            if bytes[self.at] == b'\\' {
                self.at += 1;
            }
            self.at += 1;
        }
        let end = self.at;
        self.expect(b'"')?;
        Ok(&self.text[start..end])
    }

    fn value(&mut self) -> Result<Json<'a>, String> {
        match self.peek() {
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                if self.peek() == Some(b']') {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                loop {
                    items.push(self.value()?);
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        _ => break,
                    }
                }
                self.expect(b']')?;
                Ok(Json::Arr(items))
            }
            Some(b'{') => {
                self.at += 1;
                let mut fields = Vec::new();
                if self.peek() == Some(b'}') {
                    self.at += 1;
                    return Ok(Json::Obj(fields));
                }
                loop {
                    let key = self.string()?;
                    self.expect(b':')?;
                    fields.push((key, self.value()?));
                    match self.peek() {
                        Some(b',') => self.at += 1,
                        _ => break,
                    }
                }
                self.expect(b'}')?;
                Ok(Json::Obj(fields))
            }
            Some(b'"') => {
                self.string()?;
                Ok(Json::Other)
            }
            Some(_) => {
                let bytes = self.text.as_bytes();
                let start = self.at;
                while self.at < bytes.len()
                    && !matches!(bytes[self.at], b',' | b']' | b'}')
                    && !bytes[self.at].is_ascii_whitespace()
                {
                    self.at += 1;
                }
                Ok(Json::Num(&self.text[start..self.at]))
            }
            None => Err("a fold tree's text ended early".into()),
        }
    }
}

fn number<T: std::str::FromStr>(value: &Json) -> Result<T, String> {
    match value {
        Json::Num(text) => text.parse::<T>().map_err(|_| format!("not a number: {text}")),
        _ => Err("a fold tree's number is missing".into()),
    }
}

fn field<'b, 'a>(fields: &'b [(&'a str, Json<'a>)], key: &str) -> Option<&'b Json<'a>> {
    fields.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
}

fn fold_tree_of(value: &Json) -> Result<FoldTree, String> {
    let Json::Obj(fields) = value else {
        return Err("a fold node is not an object".into());
    };
    if let Some(leaf) = field(fields, "leaf") {
        return Ok(FoldTree::Leaf(number::<usize>(leaf)?));
    }
    if let Some(chooser) = field(fields, "best") {
        let Some(Json::Arr(options)) = field(fields, "options") else {
            return Err("a choice without options".into());
        };
        return Ok(FoldTree::Best(
            number::<i64>(chooser)?,
            options.iter().map(fold_tree_of).collect::<Result<_, _>>()?,
        ));
    }
    let Some(Json::Arr(parts)) = field(fields, "avg") else {
        return Err("a fold node of no known kind".into());
    };
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        let Json::Arr(pair) = part else {
            return Err("a chance part is not a pair".into());
        };
        if pair.len() != 2 {
            return Err("a chance part is not a pair".into());
        }
        out.push((number::<f64>(&pair[0])?, fold_tree_of(&pair[1])?));
    }
    Ok(FoldTree::Avg(out))
}

/// A node's replacement cells, `[[row, column, tree], ...]`, as `json.dumps` wrote
/// `EncodedNode.folded` (the port's own `folded` read back by Python).
pub fn folded_cells(text: &str) -> Result<Vec<(usize, usize, FoldTree)>, String> {
    let mut reader = Reader { text, at: 0 };
    let Json::Arr(cells) = reader.value()? else {
        return Err("the folded cells are not a list".into());
    };
    let mut out = Vec::with_capacity(cells.len());
    for cell in &cells {
        let Json::Arr(parts) = cell else {
            return Err("a folded cell is not a list".into());
        };
        if parts.len() != 3 {
            return Err("a folded cell is not [row, column, tree]".into());
        }
        out.push((number::<usize>(&parts[0])?, number::<usize>(&parts[1])?, fold_tree_of(&parts[2])?));
    }
    Ok(out)
}

/// IKA-386: a fold tree as the port built it (`encoded_node`'s `turn_leaves`), not read back
/// from text: its weights are the doubles the header would have carried, and Python reads that
/// header's shortest round-trip text back to the same doubles.
pub fn fold_tree_from_value(value: &Value) -> Result<FoldTree, String> {
    let Some(fields) = value.as_object() else {
        return Err("a fold node is not an object".into());
    };
    if let Some(leaf) = fields.get("leaf") {
        return Ok(FoldTree::Leaf(leaf.as_u64().ok_or("a fold's leaf is not an index")? as usize));
    }
    if let Some(chooser) = fields.get("best") {
        let options = fields.get("options").and_then(Value::as_array).ok_or("a choice without options")?;
        return Ok(FoldTree::Best(
            chooser.as_i64().ok_or("a choice's chooser is not a number")?,
            options.iter().map(fold_tree_from_value).collect::<Result<_, _>>()?,
        ));
    }
    let parts = fields.get("avg").and_then(Value::as_array).ok_or("a fold node of no known kind")?;
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        let pair = part.as_array().filter(|p| p.len() == 2).ok_or("a chance part is not a pair")?;
        out.push((pair[0].as_f64().ok_or("a chance weight is not a number")?, fold_tree_from_value(&pair[1])?));
    }
    Ok(FoldTree::Avg(out))
}

/// IKA-386: a sub-game's matrix from its node as the port filled it -- the span block and the
/// replacement cells' trees (`[[row, column, tree], ...]`) -- and its leaves' values, and its
/// solved value: `folds`' answer for the node, without the crossing.
pub fn fold_and_solve(
    block: &[u8],
    count: usize,
    total: usize,
    folded: &[Value],
    values: &[f64],
    m: usize,
    n: usize,
) -> Result<Result<Solved, Unsolved>, String> {
    FOLDS.fetch_add(1, Ordering::Relaxed);
    let mut payoff = fold(block, count, total, values, m, n)?;
    for cell in folded {
        let parts = cell.as_array().filter(|p| p.len() == 3).ok_or("a folded cell is not [row, column, tree]")?;
        let i = parts[0].as_u64().ok_or("a folded cell's row")? as usize;
        let j = parts[1].as_u64().ok_or("a folded cell's column")? as usize;
        if i >= m || j >= n {
            return Err("a folded cell outside its node".into());
        }
        payoff[i * n + j] = fold_value(&fold_tree_from_value(&parts[2])?, values)?;
    }
    Ok(solve(&payoff, m, n, 1e-9))
}

/// Python's `sum` of floats (3.12: Neumaier's compensated sum, from the int 0 it starts at).
pub fn python_sum(items: impl IntoIterator<Item = f64>) -> f64 {
    let mut items = items.into_iter();
    let Some(first) = items.next() else {
        return 0.0;
    };
    let mut f = 0.0 + first;
    let mut c = 0.0;
    for x in items {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// `fold.fold_value`: a tree collapsed against one value per leaf.
pub fn fold_value(tree: &FoldTree, values: &[f64]) -> Result<f64, String> {
    Ok(match tree {
        FoldTree::Leaf(index) => *values.get(*index).ok_or("a fold's leaf past the values")?,
        FoldTree::Best(chooser, options) => {
            let mut best: Option<f64> = None;
            for option in options {
                let v = fold_value(option, values)?;
                // Python's max / min: the first of the best, replaced only by a better one.
                best = Some(match best {
                    None => v,
                    Some(b) if (*chooser == 0 && v > b) || (*chooser != 0 && v < b) => v,
                    Some(b) => b,
                });
            }
            best.unwrap_or(0.0)
        }
        FoldTree::Avg(parts) => {
            let total = python_sum(parts.iter().map(|(w, _)| *w));
            if total <= 0.0 {
                0.0
            } else {
                let mut products = Vec::with_capacity(parts.len());
                for (w, part) in parts {
                    products.push(w * fold_value(part, values)?);
                }
                python_sum(products) / total
            }
        }
    })
}

// ---------------------------------------------------------------------------
// The wire: f64 arrays as base64 of their little-endian bytes
// ---------------------------------------------------------------------------

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn b64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

pub fn b64_decode(text: &str) -> Result<Vec<u8>, String> {
    let mut table = [255u8; 256];
    for (k, c) in B64.iter().enumerate() {
        table[*c as usize] = k as u8;
    }
    let raw = text.as_bytes();
    if raw.len() % 4 != 0 {
        return Err("base64 of a length not a multiple of 4".into());
    }
    let mut out = Vec::with_capacity(raw.len() / 4 * 3);
    for quad in raw.chunks(4) {
        let mut n = 0u32;
        let mut pad = 0;
        for &c in quad {
            n <<= 6;
            if c == b'=' {
                pad += 1;
            } else {
                let v = table[c as usize];
                if v == 255 || pad > 0 {
                    return Err("not base64".into());
                }
                n |= v as u32;
            }
        }
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Ok(out)
}

fn f64s(value: &Value) -> Result<Vec<f64>, String> {
    let text = value.as_str().ok_or("an array is not base64 text")?;
    let bytes = b64_decode(text)?;
    if bytes.len() % 8 != 0 {
        return Err("an f64 array of a length not a multiple of 8".into());
    }
    Ok(bytes.chunks(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect())
}

fn f64s_text(values: &[f64]) -> String {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    b64_encode(&bytes)
}

fn matrix(value: &Value) -> Result<(Vec<f64>, usize, usize), String> {
    let m = value["rows"].as_u64().ok_or("a game without rows")? as usize;
    let n = value["cols"].as_u64().ok_or("a game without cols")? as usize;
    let data = f64s(&value["data"])?;
    if data.len() != m * n {
        return Err(format!("{} values for a {m} x {n} game", data.len()));
    }
    Ok((data, m, n))
}

pub fn unsolved_json(why: Unsolved) -> Value {
    match why {
        Unsolved::Invalid(reason) => json!({ "invalid": reason }),
        Unsolved::Failed(reason) => json!({ "failed": reason }),
    }
}

/// `lp`: each game (`{"rows", "cols", "data"}`, or `{"bayes": [games], "weights"}`) solved.
pub fn lp_command(value: &Value) -> Value {
    let Some(games) = value["games"].as_array() else {
        return json!({ "error": "`lp` without a list of games" });
    };
    let eps = value.get("eps").and_then(Value::as_f64).unwrap_or(1e-9);
    let before = LPS.load(Ordering::Relaxed);
    let mut answers = Vec::with_capacity(games.len());
    for game in games {
        GAMES.fetch_add(1, Ordering::Relaxed);
        let solved = if let Some(classes) = game.get("bayes").and_then(Value::as_array) {
            let mats: Result<Vec<_>, String> = classes.iter().map(matrix).collect();
            let weights = f64s(&game["weights"]);
            match (mats, weights) {
                (Ok(mats), Ok(weights)) => solve_bayesian(&mats, &weights, eps),
                (Err(e), _) | (_, Err(e)) => return json!({ "error": e }),
            }
        } else {
            match matrix(game) {
                Ok((data, m, n)) => solve(&data, m, n, eps),
                Err(e) => return json!({ "error": e }),
            }
        };
        answers.push(match solved {
            Ok(s) => json!({
                "valueRow": s.value_row,
                "valueCol": s.value_col,
                "x": f64s_text(&s.x),
                "ys": s.ys.iter().map(|y| f64s_text(y)).collect::<Vec<_>>(),
            }),
            Err(why) => unsolved_json(why),
        });
    }
    json!({ "kind": "lp", "answers": answers, "lps": LPS.load(Ordering::Relaxed) - before })
}

/// `folds`: each sub-game folded from its spans and leaf values (or given whole, `data`) and
/// solved; its value (`solve(payoff).value`).
pub fn folds_command(value: &Value) -> Value {
    let Some(nodes) = value["nodes"].as_array() else {
        return json!({ "error": "`folds` without a list of nodes" });
    };
    let before = LPS.load(Ordering::Relaxed);
    let mut answers = Vec::with_capacity(nodes.len());
    for node in nodes {
        FOLDS.fetch_add(1, Ordering::Relaxed);
        let read = || -> Result<(Vec<f64>, usize, usize), String> {
            if node.get("data").is_some() {
                // A matrix already whole (no forward pass to share, or a cell folded from a
                // replacement's tree): solved as it is.
                return matrix(node);
            }
            let m = node["rows"].as_u64().ok_or("a node without rows")? as usize;
            let n = node["cols"].as_u64().ok_or("a node without cols")? as usize;
            let count = node["spanCount"].as_u64().ok_or("a node without spanCount")? as usize;
            let total = node["spanLeaves"].as_u64().ok_or("a node without spanLeaves")? as usize;
            let block = b64_decode(node["spans"].as_str().ok_or("a node without spans")?)?;
            let values = f64s(&node["values"])?;
            let mut payoff = fold(&block, count, total, &values, m, n)?;
            // `port._folded`: the cells a replacement stopped, after the spans.
            if let Some(text) = node.get("folded").and_then(Value::as_str) {
                for (i, j, tree) in folded_cells(text)? {
                    if i >= m || j >= n {
                        return Err("a folded cell outside its node".into());
                    }
                    payoff[i * n + j] = fold_value(&tree, &values)?;
                }
            }
            Ok((payoff, m, n))
        };
        let (payoff, m, n) = match read() {
            Ok(got) => got,
            Err(e) => return json!({ "error": e }),
        };
        answers.push(match solve(&payoff, m, n, 1e-9) {
            Ok(s) => json!({ "value": s.value() }),
            Err(why) => unsolved_json(why),
        });
    }
    json!({ "kind": "folds", "answers": answers, "lps": LPS.load(Ordering::Relaxed) - before })
}

/// The counters, for `parallel`'s report and the tests.
pub fn report() -> Value {
    json!({
        "lps": LPS.load(Ordering::Relaxed),
        "games": GAMES.load(Ordering::Relaxed),
        "folds": FOLDS.load(Ordering::Relaxed),
        "ns": NS.iter().map(|n| n.load(Ordering::Relaxed)).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_pennies_is_a_half_each() {
        let s = solve(&[1.0, 0.0, 0.0, 1.0], 2, 2, 1e-9).unwrap();
        assert!((s.value() - 0.5).abs() < 1e-12);
        assert!((s.x[0] - 0.5).abs() < 1e-9 && (s.ys[0][1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn a_dominated_row_gets_nothing() {
        // Row 1 is worse than row 0 everywhere.
        let s = solve(&[0.9, 0.6, 0.1, 0.2], 2, 2, 1e-9).unwrap();
        assert_eq!(s.x, vec![1.0, 0.0]);
        assert!((s.value() - 0.6).abs() < 1e-12);
        assert_eq!(s.ys[0], vec![0.0, 1.0]);
    }

    #[test]
    fn one_class_bayesian_is_the_plain_game() {
        let payoff = [0.3, 0.8, 0.6, 0.2, 0.5, 0.4];
        let plain = solve(&payoff, 2, 3, 1e-9).unwrap();
        let bayes = solve_bayesian(&[(payoff.to_vec(), 2, 3)], &[1.0], 1e-9).unwrap();
        assert!((plain.value() - bayes.value()).abs() < 1e-12);
    }

    #[test]
    fn invalid_games_are_refused_as_python_refuses_them() {
        assert!(matches!(solve(&[], 0, 0, 1e-9), Err(Unsolved::Invalid(_))));
        assert!(matches!(solve(&[f64::NAN], 1, 1, 1e-9), Err(Unsolved::Invalid(_))));
    }

    #[test]
    fn python_sum_compensates_and_the_tree_folds_as_fold_py() {
        // 1e16 + 1 + 1 - 1e16: the running sum loses both ones; Python's keeps them.
        assert_eq!(python_sum([1e16, 1.0, 1.0, -1e16]), 2.0);
        assert_eq!(python_sum([]), 0.0);
        let tree = FoldTree::Avg(vec![
            (0.25, FoldTree::Leaf(0)),
            (0.75, FoldTree::Best(1, vec![FoldTree::Leaf(1), FoldTree::Leaf(2)])),
        ]);
        let v = fold_value(&tree, &[0.2, 0.9, 0.4]).unwrap();
        assert_eq!(v, python_sum([0.25 * 0.2, 0.75 * 0.4]) / 1.0);
        assert_eq!(fold_value(&FoldTree::Best(0, vec![]), &[]).unwrap(), 0.0);
        // As `json.dumps` writes `EncodedNode.folded`: the weights read back to the bit.
        let text = r#"[[1, 0, {"avg": [[0.1, {"leaf": 0}], [2.5e-05, {"best": 1, "options": [{"leaf": 1}, {"leaf": 2}]}]]}]]"#;
        let cells = folded_cells(text).unwrap();
        assert_eq!((cells[0].0, cells[0].1), (1, 0));
        let v = fold_value(&cells[0].2, &[0.2, 0.9, 0.4]).unwrap();
        assert_eq!(v, python_sum([0.1 * 0.2, 2.5e-05 * 0.4]) / python_sum([0.1, 2.5e-05]));
        // IKA-386: the same tree as the port built it, not read back from text.
        let built: Value = serde_json::from_str(text).unwrap();
        let from_value = fold_tree_from_value(&built[0][2]).unwrap();
        assert_eq!(fold_value(&from_value, &[0.2, 0.9, 0.4]).unwrap().to_bits(), v.to_bits());
    }

    #[test]
    fn base64_round_trips() {
        for len in 0..40 {
            let bytes: Vec<u8> = (0..len).map(|k| (k * 37 + 11) as u8).collect();
            assert_eq!(b64_decode(&b64_encode(&bytes)).unwrap(), bytes);
        }
        assert_eq!(b64_encode(b"Man"), "TWFu");
        assert_eq!(b64_encode(b"Ma"), "TWE=");
    }

    #[test]
    fn numpy_sum_is_pairwise() {
        // Eight accumulators from eight on: numpy's answers (2.5.3), not the running sum's 0.
        let mut a = vec![1e16];
        a.extend([1.0; 7]);
        a.push(-1e16);
        assert_eq!(numpy_sum(&a), 6.0);
        assert_eq!(a.iter().fold(0.0, |s, v| s + v), 0.0);
        let mut b = vec![1e16];
        b.extend([1.0; 20]);
        b.push(-1e16);
        assert_eq!(numpy_sum(&b), 16.0);
    }

    #[test]
    fn the_fold_skips_an_empty_span_and_weighs_the_rest() {
        // Two spans: (0, 0) of leaves [1, 0] weights [0.25, 0.75]; (1, 1) empty.
        let mut block = Vec::new();
        for w in [0.25f64, 0.75] {
            block.extend_from_slice(&w.to_le_bytes());
        }
        for v in [0u32, 1] {
            block.extend_from_slice(&v.to_le_bytes()); // rows
        }
        for v in [0u32, 1] {
            block.extend_from_slice(&v.to_le_bytes()); // cols
        }
        for v in [2u32, 0] {
            block.extend_from_slice(&v.to_le_bytes()); // lengths
        }
        for v in [1u32, 0] {
            block.extend_from_slice(&v.to_le_bytes()); // indices
        }
        let payoff = fold(&block, 2, 2, &[0.4, 0.8], 2, 2).unwrap();
        let first = 0.75f64.mul_add(0.4, 0.25f64.mul_add(0.8, 0.0));
        assert_eq!(payoff, vec![first, 0.0, 0.0, 0.0]);
    }
}
