//! IKA-389: numpy's matrix-vector products, from numpy's own BLAS.
//!
//! A Q's ranking (`deepen._q_menus`) scores each candidate by `equilibrium.assemble`'s
//! `row_ev = a @ y` and `col_ev = x @ a`, and at an equilibrium every action in the support
//! has the same expected value up to the last bits -- so which of them sorts first is decided
//! by those bits, and the menu is only the Python's menu if they are numpy's. numpy hands both
//! products to OpenBLAS's `dgemv` (`matmul.c.src`'s `gemv`: column-major, transposed, `lda`
//! the row length), whose kernel sums in its own blocked order, chosen at run time for the
//! machine's core. So the port asks the very library numpy loaded -- its path and symbol
//! come from the caller (`portmenus.blas`), never from here -- and gets the same bits by
//! construction on any machine. The library's thread count does not move a bit (measured,
//! records/IKA-389.md), so the port's copy is set to one thread.

use std::sync::OnceLock;

type Gemv = unsafe extern "C" fn(
    order: i32,
    trans: i32,
    m: i64,
    n: i64,
    alpha: f64,
    a: *const f64,
    lda: i64,
    x: *const f64,
    incx: i64,
    beta: f64,
    y: *mut f64,
    incy: i64,
);
type SetThreads = unsafe extern "C" fn(threads: i32);

const COL_MAJOR: i32 = 102;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

struct Library {
    path: String,
    gemv: Gemv,
}

// SAFETY: a function pointer into a library that is never unloaded.
unsafe impl Send for Library {}
unsafe impl Sync for Library {}

static LOADED: OnceLock<Result<Library, String>> = OnceLock::new();

#[cfg(windows)]
mod os {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }
    pub fn open(path: &str) -> *mut c_void {
        let wide: Vec<u16> = path.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: a NUL-terminated UTF-16 path that outlives the call.
        unsafe { LoadLibraryW(wide.as_ptr()) }
    }
    pub fn symbol(module: *mut c_void, name: &str) -> *mut c_void {
        let text: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
        // SAFETY: a live module and a NUL-terminated name.
        unsafe { GetProcAddress(module, text.as_ptr()) }
    }
}

#[cfg(unix)]
mod os {
    use std::ffi::c_void;
    const RTLD_NOW: i32 = 2;
    extern "C" {
        fn dlopen(path: *const u8, flags: i32) -> *mut c_void;
        fn dlsym(module: *mut c_void, name: *const u8) -> *mut c_void;
    }
    pub fn open(path: &str) -> *mut c_void {
        let text: Vec<u8> = path.bytes().chain(std::iter::once(0)).collect();
        // SAFETY: a NUL-terminated path.
        unsafe { dlopen(text.as_ptr(), RTLD_NOW) }
    }
    pub fn symbol(module: *mut c_void, name: &str) -> *mut c_void {
        let text: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
        // SAFETY: a live module and a NUL-terminated name.
        unsafe { dlsym(module, text.as_ptr()) }
    }
}

fn load(path: &str, gemv: &str, threads: Option<&str>) -> Result<Library, String> {
    let module = os::open(path);
    if module.is_null() {
        return Err(format!("could not load numpy's BLAS at {path}"));
    }
    let found = os::symbol(module, gemv);
    if found.is_null() {
        return Err(format!("{path} has no {gemv}"));
    }
    if let Some(name) = threads {
        let set = os::symbol(module, name);
        if !set.is_null() {
            // SAFETY: the library's own `openblas_set_num_threads`, as numpy's threadpool
            // tools call it.
            unsafe { std::mem::transmute::<*mut std::ffi::c_void, SetThreads>(set)(1) };
        }
    }
    // SAFETY: the symbol is `cblas_dgemv` of a 64-bit-integer OpenBLAS (`scipy_openblas64`),
    // whose signature `Gemv` is.
    let gemv = unsafe { std::mem::transmute::<*mut std::ffi::c_void, Gemv>(found) };
    Ok(Library { path: path.to_string(), gemv })
}

/// numpy's BLAS, loaded once for the process from what the caller named.
fn library(path: &str, gemv: &str, threads: Option<&str>) -> Result<&'static Library, String> {
    let loaded = LOADED.get_or_init(|| load(path, gemv, threads));
    match loaded {
        Err(e) => Err(e.clone()),
        Ok(lib) if lib.path != path => Err(format!("numpy's BLAS is {} here, not {path}", lib.path)),
        Ok(lib) => Ok(lib),
    }
}

/// What a request names: the library, its `cblas_dgemv` and its thread setter.
pub struct Blas {
    lib: &'static Library,
}

impl Blas {
    pub fn from_json(value: &serde_json::Value) -> Result<Blas, String> {
        let path = value["path"].as_str().ok_or("`blas` without `path`")?;
        let gemv = value["gemv"].as_str().ok_or("`blas` without `gemv`")?;
        let threads = value.get("threads").and_then(serde_json::Value::as_str);
        Ok(Blas { lib: library(path, gemv, threads)? })
    }

    /// `a @ y` for a row-major m x n `a`, as numpy computes it for m, n > 1 (`dgemv`).
    pub fn a_y(&self, a: &[f64], m: usize, n: usize, y: &[f64]) -> Vec<f64> {
        assert!(a.len() == m * n && y.len() == n);
        let mut out = vec![0.0f64; m];
        // SAFETY: `a` holds m*n, `y` n and `out` m doubles, as the call reads and writes them.
        unsafe {
            (self.lib.gemv)(
                COL_MAJOR, TRANS, n as i64, m as i64, 1.0, a.as_ptr(), n as i64, y.as_ptr(), 1, 0.0,
                out.as_mut_ptr(), 1,
            )
        };
        out
    }

    /// `x @ a` for a row-major m x n `a`, as numpy computes it for m, n > 1 (`dgemv`).
    pub fn x_a(&self, x: &[f64], a: &[f64], m: usize, n: usize) -> Vec<f64> {
        assert!(a.len() == m * n && x.len() == m);
        let mut out = vec![0.0f64; n];
        // SAFETY: as `a_y`.
        unsafe {
            (self.lib.gemv)(
                COL_MAJOR, NO_TRANS, n as i64, m as i64, 1.0, a.as_ptr(), n as i64, x.as_ptr(), 1, 0.0,
                out.as_mut_ptr(), 1,
            )
        };
        out
    }
}
