//! IKA-461: the value net's forward pass on the GPU, for the Rust inference server.
//!
//! `cudarc` loads the driver, NVRTC and cuBLAS at run time (`dynamic-loading`), so the
//! build needs no CUDA toolkit; the DLLs are CUDA 12's, found on `PATH` (the server puts
//! `--dll-dir` there first; the launcher passes the `.venv`'s `torch/lib`). The kernels are
//! `infer.cu`, compiled by NVRTC for the card's architecture at start-up.
//!
//! A *slot* is one stream with its own cuBLAS handle, buffers and CUDA Graphs. A request
//! takes a slot, copies its rows into the slot's pinned memory, replays the graph of the
//! row count's bucket (host-to-device copies, every kernel of every net of the arm, and the
//! copy of the answers back are all in the graph) and sleeps on a blocking-sync event until
//! the card is done. Rows between the request's and its bucket's size are whatever the
//! slot held before: they are computed and thrown away.

use crate::rnet::RNet;
use cudarc::cublas::{result as bres, sys as bsys, CudaBlas};
use cudarc::driver::sys as dsys;
use cudarc::driver::{
    result as dres, CudaContext, CudaEvent, CudaFunction, CudaGraph, CudaSlice, CudaStream,
    DevicePtr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::Arc;

const KERNELS: &str = include_str!("infer.cu");

/// Most rows one graph handles; longer requests are cut here.
pub const MAX_ROWS: usize = 4096;

/// The rows a graph is captured for: 64-row steps up to 1,024, then 256-row steps.
pub fn bucket_rows(bucket: usize) -> usize {
    if bucket < 16 {
        (bucket + 1) * 64
    } else {
        1024 + (bucket - 15) * 256
    }
}

pub const BUCKETS: usize = 16 + (MAX_ROWS - 1024) / 256;

pub fn bucket_of(rows: usize) -> usize {
    debug_assert!(rows >= 1 && rows <= MAX_ROWS);
    if rows <= 1024 {
        (rows - 1) / 64
    } else {
        15 + (rows - 1024).div_ceil(256)
    }
}

/// The shape of a row of the encoder's arrays (`encode::Widths`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub m: usize,
    pub wm: usize,
    pub ws: usize,
    pub wf: usize,
}

impl Geometry {
    /// Bytes of one row of each of the eight arrays, in `inference.ARRAYS` order.
    pub fn row_bytes(&self) -> [usize; 8] {
        let t = 2 * self.m;
        [t * 4, t * 4, t * 4, t * 4 * 4, t * self.wm * 4, t * 4, 2 * self.ws * 4, self.wf * 4]
    }
}

struct Dims {
    sd: usize,
    ad: usize,
    id: usize,
    md: usize,
    mon: usize,
    side: usize,
    head: usize,
    mon_cut: usize,
    side_cut: usize,
    field_cut: usize,
    id_lo: usize,
    n_ids: usize,
    active: usize,
    mon_in: usize,
    pooled: usize,
    head_in: usize,
    vocab: [usize; 4],
}

/// One net's weights on the card, as raw device pointers kept alive by `_hold`.
pub struct Net {
    dims: Dims,
    ptr: HashMap<String, u64>,
    /// "trunk." for a Q file (the value net is its trunk), "" for a value net.
    prefix: &'static str,
    _hold: Vec<CudaSlice<f32>>,
    pub source: String,
}

impl Net {
    pub fn upload(stream: &Arc<CudaStream>, rnet: &RNet) -> Result<Net, String> {
        let h = &rnet.header;
        let num = |path: &[&str]| -> Result<usize, String> {
            let mut v = h;
            for key in path {
                v = v.get(*key).ok_or(format!("the .rnet header has no `{}`", path.join(".")))?;
            }
            v.as_u64().map(|n| n as usize).ok_or(format!("`{}` is not a number", path.join(".")))
        };
        let mut ptr = HashMap::new();
        let mut hold = Vec::new();
        for (name, (_shape, values)) in &rnet.tensors {
            let dev = stream.memcpy_stod(values).map_err(|e| e.to_string())?;
            ptr.insert(name.clone(), dev.device_ptr(stream).0);
            hold.push(dev);
        }
        let is_q = h.get("kind").and_then(|k| k.as_str()) == Some("q");
        let prefix: &'static str = if is_q { "trunk." } else { "" };
        let vocab = |name: &str| {
            rnet.tensors.get(&format!("{prefix}{name}")).map(|t| t.0[0]).ok_or(format!("no tensor {prefix}{name}"))
        };
        let dims = Dims {
            sd: num(&["dims", "species"])?,
            ad: num(&["dims", "ability"])?,
            id: num(&["dims", "item"])?,
            md: num(&["dims", "move"])?,
            mon: num(&["dims", "mon"])?,
            side: num(&["dims", "side"])?,
            head: num(&["dims", "head"])?,
            mon_cut: num(&["mon_cut"])?,
            side_cut: num(&["side_cut"])?,
            field_cut: num(&["field_cut"])?,
            id_lo: num(&["id_lo"])?,
            n_ids: num(&["n_ids"])?,
            active: num(&["active_feature"])?,
            mon_in: num(&["mon_in"])?,
            pooled: num(&["pooled"])?,
            head_in: num(&["head_in"])?,
            vocab: [
                vocab("species.weight")?,
                vocab("ability.weight")?,
                vocab("item.weight")?,
                vocab("move.weight")?,
            ],
        };
        if dims.mon > 256 || dims.side > 256 || dims.head > 256 {
            return Err("a layer wider than 256 needs a wider block in infer.cu".into());
        }
        let trunk = ["mon_mlp.0.weight", "mon_mlp.1.weight", "mon_mlp.3.weight", "mon_mlp.4.weight",
            "side_mlp.0.weight", "side_mlp.1.weight"];
        let head = ["head.0.weight", "head.1.weight", "head.4.weight", "head.5.weight", "head.8.weight", "head.8.bias"];
        for name in trunk.iter().chain(if is_q { [].iter() } else { head.iter() }) {
            if !ptr.contains_key(&format!("{prefix}{name}")) {
                return Err(format!("the .rnet has no tensor {prefix}{name}"));
            }
        }
        Ok(Net { dims, ptr, prefix, _hold: hold, source: rnet.source.clone() })
    }

    /// A trunk tensor's device pointer (a Q file's trunk is under "trunk.").
    fn w(&self, name: &str) -> u64 {
        self.ptr[&format!("{}{name}", self.prefix)]
    }

    /// A tensor by its full name (the Q's own layers).
    pub fn raw(&self, name: &str) -> u64 {
        *self.ptr.get(name).unwrap_or_else(|| panic!("the .rnet has no tensor {name}"))
    }
}

pub struct Kernels {
    mon_features: CudaFunction,
    bias_ln_gelu: CudaFunction,
    pool: CudaFunction,
    head_input: CudaFunction,
    head_final: CudaFunction,
    finalize: CudaFunction,
    add_bias: CudaFunction,
    props_pool: CudaFunction,
    attend: CudaFunction,
    ln_residual: CudaFunction,
    action_input: CudaFunction,
    slot_sum: CudaFunction,
    pair_matrix: CudaFunction,
}

/// The versions of the libraries this process loaded, for the start-up line.
pub struct Versions {
    pub driver: i32,
    pub nvrtc: (i32, i32),
    pub cublas: i32,
    pub device: String,
    pub capability: (i32, i32),
}

pub struct Gpu {
    pub ctx: Arc<CudaContext>,
    pub kernels: Kernels,
    pub versions: Versions,
}

fn arch_of(cc: (i32, i32)) -> &'static str {
    match cc {
        (12, 0) => "sm_120",
        (10, 0) => "sm_100",
        (9, 0) => "sm_90",
        (8, 9) => "sm_89",
        (8, 6) => "sm_86",
        (8, 0) => "sm_80",
        _ => "compute_80",
    }
}

impl Gpu {
    pub fn new(ordinal: usize) -> Result<Gpu, String> {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("CUDA driver: {e}"))?;
        // One stream per slot and no cross-stream use of a buffer: the tracking only gets in the
        // way of capturing a graph.
        unsafe { ctx.disable_event_tracking() };
        let capability = ctx.compute_capability().map_err(|e| e.to_string())?;
        let device = ctx.name().map_err(|e| e.to_string())?;
        let mut driver = 0i32;
        unsafe { dsys::cuDriverGetVersion(&mut driver) };
        let (mut nvrtc_major, mut nvrtc_minor) = (0i32, 0i32);
        unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut nvrtc_major, &mut nvrtc_minor) };
        if nvrtc_major != 12 {
            return Err(format!(
                "NVRTC {nvrtc_major}.{nvrtc_minor} was loaded; this server was written against \
                 CUDA 12's (the torch 2.11+cu128 DLLs): pass --dll-dir of that folder"
            ));
        }
        let opts = CompileOptions { arch: Some(arch_of(capability)), ..Default::default() };
        let ptx = compile_ptx_with_opts(KERNELS, opts).map_err(|e| format!("NVRTC: {e}"))?;
        let module = ctx.load_module(ptx).map_err(|e| e.to_string())?;
        let f = |name: &str| module.load_function(name).map_err(|e| format!("{name}: {e}"));
        let kernels = Kernels {
            mon_features: f("mon_features")?,
            bias_ln_gelu: f("bias_ln_gelu")?,
            pool: f("pool")?,
            head_input: f("head_input")?,
            head_final: f("head_final")?,
            finalize: f("finalize")?,
            add_bias: f("add_bias")?,
            props_pool: f("props_pool")?,
            attend: f("attend")?,
            ln_residual: f("ln_residual")?,
            action_input: f("action_input")?,
            slot_sum: f("slot_sum")?,
            pair_matrix: f("pair_matrix")?,
        };
        let stream = ctx.new_stream().map_err(|e| e.to_string())?;
        let blas = CudaBlas::new(stream).map_err(|e| format!("cuBLAS: {e}"))?;
        let mut cublas = 0i32;
        unsafe { bsys::cublasGetVersion_v2(*blas.handle(), &mut cublas) };
        if cublas / 10000 != 12 {
            return Err(format!("cuBLAS {cublas} was loaded; CUDA 12's is expected (--dll-dir)"));
        }
        Ok(Gpu {
            ctx,
            kernels,
            versions: Versions {
                driver,
                nvrtc: (nvrtc_major, nvrtc_minor),
                cublas,
                device,
                capability,
            },
        })
    }
}

/// Raw pinned host memory.
struct Pinned {
    ptr: *mut u8,
    len: usize,
}

impl Pinned {
    fn new(len: usize) -> Result<Pinned, String> {
        let ptr = unsafe { dres::malloc_host(len.max(64), 0) }.map_err(|e| e.to_string())? as *mut u8;
        unsafe { std::ptr::write_bytes(ptr, 0, len.max(64)) };
        Ok(Pinned { ptr, len })
    }
    fn slice_mut(&mut self, from: usize, n: usize) -> &mut [u8] {
        assert!(from + n <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(from), n) }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        unsafe {
            let _ = dres::free_host(self.ptr as *mut _);
        }
    }
}

pub struct Slot {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    event: CudaEvent,
    geom: Geometry,
    row_bytes: [usize; 8],
    /// The eight input arrays on the card and in pinned memory, MAX_ROWS rows each.
    dev_in: Vec<CudaSlice<u8>>,
    host_in: Vec<Pinned>,
    // Activations, shared by every net of the slot (they run one after another).
    feat: CudaSlice<f32>,
    h0: CudaSlice<f32>,
    h1: CudaSlice<f32>,
    pooled: CudaSlice<f32>,
    sides: CudaSlice<f32>,
    head_in: CudaSlice<f32>,
    ha: CudaSlice<f32>,
    hb: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    out_dev: CudaSlice<f64>,
    host_out: Pinned,
    _workspace: CudaSlice<u8>,
    /// graphs[arm][bucket]
    graphs: Vec<Vec<CudaGraph>>,
    /// The stages of the last `debug` run, for tests.
    pub debug_names: Vec<&'static str>,
}

// A slot is used by one thread at a time (the server hands it out under a lock).
unsafe impl Send for Slot {}

fn ptr<T>(slice: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    slice.device_ptr(stream).0
}

/// y (rows, n) = x (rows, k) . w (n, k)^T + beta * y, all row-major: the transposed product in
/// cuBLAS's column-major terms.
#[allow(clippy::too_many_arguments)]
fn sgemm_nt(blas: &CudaBlas, w: u64, x: u64, y: u64, n: usize, k: usize, rows: usize, beta: f32) -> Result<(), String> {
    let one = 1.0f32;
    unsafe {
        bres::sgemm(
            *blas.handle(),
            bsys::cublasOperation_t::CUBLAS_OP_T,
            bsys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            rows as i32,
            k as i32,
            &one,
            w as *const f32,
            k as i32,
            x as *const f32,
            k as i32,
            &beta,
            y as *mut f32,
            n as i32,
        )
    }
    .map_err(|e| e.to_string())
}

fn cfg(grid: usize, block: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: (grid as u32, 1, 1), block_dim: (block as u32, 1, 1), shared_mem_bytes: 0 }
}

impl Slot {
    /// `widest`: the largest (mon_in, pooled, head_in) of the nets the slot will run.
    pub fn new(gpu: &Gpu, geom: Geometry, arms: usize, widest: (usize, usize, usize)) -> Result<Slot, String> {
        let e = |e: cudarc::driver::DriverError| e.to_string();
        let stream = gpu.ctx.new_stream().map_err(e)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| e.to_string())?;
        let workspace = stream.alloc_zeros::<u8>(32 << 20).map_err(e)?;
        unsafe {
            bsys::cublasSetWorkspace_v2(*blas.handle(), ptr(&workspace, &stream) as *mut _, 32 << 20)
        }
        .result()
        .map_err(|e| e.to_string())?;
        let row_bytes = geom.row_bytes();
        let mut dev_in = Vec::new();
        let mut host_in = Vec::new();
        for bytes in row_bytes {
            dev_in.push(stream.alloc_zeros::<u8>(bytes * MAX_ROWS).map_err(e)?);
            host_in.push(Pinned::new(bytes * MAX_ROWS)?);
        }
        let t = 2 * geom.m * MAX_ROWS;
        let r = MAX_ROWS;
        let wide = 256usize.max(1);
        let buf = |n: usize| stream.alloc_zeros::<f32>(n).map_err(e);
        let slot = Slot {
            event: gpu
                .ctx
                .new_event(Some(dsys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
                .map_err(e)?,
            feat: buf(t * widest.0)?,
            h0: buf(t * wide)?,
            h1: buf(t * wide)?,
            pooled: buf(2 * r * widest.1)?,
            sides: buf(2 * r * wide)?,
            head_in: buf(2 * r * widest.2)?,
            ha: buf(2 * r * wide)?,
            hb: buf(2 * r * wide)?,
            logits: buf(8 * r)?,
            out_dev: stream.alloc_zeros::<f64>(r).map_err(e)?,
            host_out: Pinned::new(r * 8)?,
            _workspace: workspace,
            graphs: (0..arms).map(|_| Vec::new()).collect(),
            debug_names: Vec::new(),
            stream,
            blas,
            geom,
            row_bytes,
            dev_in,
            host_in,
        };
        Ok(slot)
    }

    /// A net's (mon_in, pooled, head_in), the widths a slot's buffers are sized by.
    pub fn widths_of(net: &Net) -> (usize, usize, usize) {
        (net.dims.mon_in, net.dims.pooled, net.dims.head_in)
    }

    fn gemm(&self, w: u64, x: u64, y: u64, n: usize, k: usize, rows: usize) -> Result<(), String> {
        sgemm_nt(&self.blas, w, x, y, n, k, rows, 0.0)
    }

    fn bln(&self, k: &Kernels, x: u64, rows: usize, n: usize, b: u64, g: u64, be: u64) -> Result<(), String> {
        let n = n as i32;
        let mut l = self.stream.launch_builder(&k.bias_ln_gelu);
        l.arg(&x).arg(&b).arg(&g).arg(&be).arg(&n);
        unsafe { l.launch(cfg(rows, 256)) }.map(|_| ()).map_err(|e| e.to_string())
    }

    /// Every kernel of one net on the first `rows` rows of the slot's input arrays, its
    /// logits into `logits[net_index * rows ..]`. `stage` is called after each step with
    /// the buffer, its element count and its row width (for the tests' op "stages").
    #[allow(clippy::too_many_arguments)]
    fn run_net(
        &self,
        k: &Kernels,
        net: &Net,
        rows: usize,
        net_index: usize,
        mut stage: impl FnMut(&'static str, u64, usize, usize),
    ) -> Result<(), String> {
        let d = &net.dims;
        let g = self.geom;
        let s = &self.stream;
        let t = 2 * g.m * rows;
        let inp: Vec<u64> = self.dev_in.iter().map(|b| ptr(b, s)).collect();
        let (feat, h0, h1) = (ptr(&self.feat, s), ptr(&self.h0, s), ptr(&self.h1, s));
        let (pooled, sides, head_in) = (ptr(&self.pooled, s), ptr(&self.sides, s), ptr(&self.head_in, s));
        let (ha, hb) = (ptr(&self.ha, s), ptr(&self.hb, s));
        let i = |v: usize| v as i32;
        {
            let (sd, ad, idd, md) = (i(d.sd), i(d.ad), i(d.id), i(d.md));
            let (nsp, nab, nit, nmv) = (i(d.vocab[0]), i(d.vocab[1]), i(d.vocab[2]), i(d.vocab[3]));
            let (wm, id_lo, n_ids, cut, mon_in) = (i(g.wm), i(d.id_lo), i(d.n_ids), i(d.mon_cut), i(d.mon_in));
            let (es, ea, ei, em) =
                (net.w("species.weight"), net.w("ability.weight"), net.w("item.weight"), net.w("move.weight"));
            let mut l = s.launch_builder(&k.mon_features);
            l.arg(&inp[0]).arg(&inp[1]).arg(&inp[2]).arg(&inp[3]).arg(&inp[4]);
            l.arg(&es).arg(&ea).arg(&ei).arg(&em);
            l.arg(&sd).arg(&ad).arg(&idd).arg(&md).arg(&nsp).arg(&nab).arg(&nit).arg(&nmv);
            l.arg(&wm).arg(&id_lo).arg(&n_ids).arg(&cut).arg(&mon_in).arg(&feat);
            unsafe { l.launch(cfg(t, 128)) }.map_err(|e| e.to_string())?;
        }
        stage("features", feat, t * d.mon_in, d.mon_in);
        self.gemm(net.w("mon_mlp.0.weight"), feat, h0, d.mon, d.mon_in, t)?;
        self.bln(k, h0, t, d.mon, net.w("mon_mlp.0.bias"), net.w("mon_mlp.1.weight"), net.w("mon_mlp.1.bias"))?;
        self.gemm(net.w("mon_mlp.3.weight"), h0, h1, d.mon, d.mon, t)?;
        self.bln(k, h1, t, d.mon, net.w("mon_mlp.3.bias"), net.w("mon_mlp.4.weight"), net.w("mon_mlp.4.bias"))?;
        stage("mon", h1, t * d.mon, d.mon);
        {
            let (m, wm, act, ws, sc, dd) = (i(g.m), i(g.wm), i(d.active), i(g.ws), i(d.side_cut), i(d.mon));
            let mut l = s.launch_builder(&k.pool);
            l.arg(&m).arg(&h1).arg(&inp[5]).arg(&inp[4]).arg(&wm).arg(&act).arg(&inp[6]);
            l.arg(&ws).arg(&sc).arg(&dd).arg(&pooled);
            unsafe { l.launch(cfg(2 * rows, 256)) }.map_err(|e| e.to_string())?;
        }
        stage("pooled", pooled, 2 * rows * d.pooled, d.pooled);
        self.gemm(net.w("side_mlp.0.weight"), pooled, sides, d.side, d.pooled, 2 * rows)?;
        self.bln(k, sides, 2 * rows, d.side, net.w("side_mlp.0.bias"), net.w("side_mlp.1.weight"), net.w("side_mlp.1.bias"))?;
        stage("sides", sides, 2 * rows * d.side, d.side);
        {
            let (r, sd, wf, fc) = (i(rows), i(d.side), i(g.wf), i(d.field_cut));
            let mut l = s.launch_builder(&k.head_input);
            l.arg(&r).arg(&sides).arg(&sd).arg(&inp[7]).arg(&wf).arg(&fc).arg(&head_in);
            unsafe { l.launch(cfg(2 * rows, 256)) }.map_err(|e| e.to_string())?;
        }
        stage("head_in", head_in, 2 * rows * d.head_in, d.head_in);
        self.gemm(net.w("head.0.weight"), head_in, ha, d.head, d.head_in, 2 * rows)?;
        self.bln(k, ha, 2 * rows, d.head, net.w("head.0.bias"), net.w("head.1.weight"), net.w("head.1.bias"))?;
        self.gemm(net.w("head.4.weight"), ha, hb, d.head, d.head, 2 * rows)?;
        self.bln(k, hb, 2 * rows, d.head, net.w("head.4.bias"), net.w("head.5.weight"), net.w("head.5.bias"))?;
        stage("head", hb, 2 * rows * d.head, d.head);
        {
            let (r, hd) = (i(rows), i(d.head));
            let logits = ptr(&self.logits, s) + (net_index * rows * 4) as u64;
            let (w, b) = (net.w("head.8.weight"), net.w("head.8.bias"));
            let mut l = s.launch_builder(&k.head_final);
            l.arg(&r).arg(&hb).arg(&hd).arg(&w).arg(&b).arg(&logits);
            unsafe { l.launch(cfg(rows, 256)) }.map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn run_arm(
        &self,
        k: &Kernels,
        nets: &[Net],
        rows: usize,
        mut stage: impl FnMut(&'static str, u64, usize, usize),
    ) -> Result<(), String> {
        for (index, net) in nets.iter().enumerate() {
            self.run_net(k, net, rows, index, &mut stage)?;
        }
        let (r, n) = (rows as i32, nets.len() as i32);
        let (logits, out) = (ptr(&self.logits, &self.stream), ptr(&self.out_dev, &self.stream));
        let mut l = self.stream.launch_builder(&k.finalize);
        l.arg(&r).arg(&n).arg(&logits).arg(&out);
        unsafe { l.launch(cfg(rows.div_ceil(256), 256)) }.map(|_| ()).map_err(|e| e.to_string())
    }

    fn copy_in(&self, rows: usize) -> Result<(), String> {
        for (k, dev) in self.dev_in.iter().enumerate() {
            let n = rows * self.row_bytes[k];
            let src = unsafe { std::slice::from_raw_parts(self.host_in[k].ptr, n) };
            unsafe { dres::memcpy_htod_async(ptr(dev, &self.stream), src, self.stream.cu_stream()) }
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn copy_out(&self, rows: usize) -> Result<(), String> {
        let dst = unsafe { std::slice::from_raw_parts_mut(self.host_out.ptr as *mut f64, rows) };
        unsafe { dres::memcpy_dtoh_async(dst, ptr(&self.out_dev, &self.stream), self.stream.cu_stream()) }
            .map_err(|e| e.to_string())
    }

    /// Warm every net up eagerly once, then capture one graph per bucket for this arm.
    pub fn capture_arm(&mut self, gpu: &Gpu, nets: &[Net]) -> Result<(), String> {
        let k = &gpu.kernels;
        self.run_arm(k, nets, MAX_ROWS, |_, _, _, _| {})?;
        self.stream.synchronize().map_err(|e| e.to_string())?;
        let mut graphs = Vec::with_capacity(BUCKETS);
        for bucket in 0..BUCKETS {
            let rows = bucket_rows(bucket);
            self.stream
                .begin_capture(dsys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| e.to_string())?;
            let result = (|| {
                self.copy_in(rows)?;
                self.run_arm(k, nets, rows, |_, _, _, _| {})?;
                self.copy_out(rows)
            })();
            let graph = self
                .stream
                .end_capture(dsys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
                .map_err(|e| e.to_string())?;
            result?;
            graphs.push(graph.ok_or("the capture came back empty")?);
        }
        let at = self.graphs.iter().position(|g| g.is_empty()).ok_or("more arms than slots were built for")?;
        self.graphs[at] = graphs;
        Ok(())
    }

    /// The pinned input for array `k`, `rows` rows from row `first`.
    pub fn input(&mut self, k: usize, first: usize, rows: usize) -> &mut [u8] {
        let rb = self.row_bytes[k];
        self.host_in[k].slice_mut(first * rb, rows * rb)
    }

    pub fn geometry(&self) -> Geometry {
        self.geom
    }

    /// Scores the `rows` rows laid into `input` (at most MAX_ROWS) with arm `arm`'s graph;
    /// the values are valid until the next call.
    pub fn score(&mut self, arm: usize, rows: usize) -> Result<&[f64], String> {
        let bucket = bucket_of(rows);
        self.graphs[arm][bucket].launch().map_err(|e| e.to_string())?;
        self.event.record(&self.stream).map_err(|e| e.to_string())?;
        self.event.synchronize().map_err(|e| e.to_string())?;
        Ok(unsafe { std::slice::from_raw_parts(self.host_out.ptr as *const f64, rows) })
    }

    /// Runs the arm eagerly on the rows in `input` and returns every stage's values (for the
    /// tests of each kernel against torch): (name, row width, values).
    pub fn stages(&mut self, gpu: &Gpu, nets: &[Net], rows: usize) -> Result<Vec<(&'static str, usize, Vec<f32>)>, String> {
        self.copy_in(rows)?;
        let mut seen: Vec<(&'static str, u64, usize, usize)> = Vec::new();
        // Only the first net's stages: the ensemble's members are checked one at a time.
        self.run_net(&gpu.kernels, &nets[0], rows, 0, |name, p, n, w| seen.push((name, p, n, w)))?;
        self.stream.synchronize().map_err(|e| e.to_string())?;
        let mut out = Vec::new();
        for (name, p, n, w) in seen {
            let mut host = vec![0f32; n];
            unsafe { dres::memcpy_dtoh_async(&mut host, p, self.stream.cu_stream()) }.map_err(|e| e.to_string())?;
            self.stream.synchronize().map_err(|e| e.to_string())?;
            out.push((name, w, host));
        }
        // The logit and the probability, from the full arm.
        self.run_arm(&gpu.kernels, nets, rows, |_, _, _, _| {})?;
        self.copy_out(rows)?;
        self.stream.synchronize().map_err(|e| e.to_string())?;
        let probs = unsafe { std::slice::from_raw_parts(self.host_out.ptr as *const f64, rows) };
        out.push(("probability", 1, probs.iter().map(|&v| v as f32).collect()));
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------
// The Q net (`qhead.build_net`). One request is a few items; each item is one position, two
// pools of actions and the port's features for them, and its answer is an N0 x N1 matrix.

/// The most actions in one side's pool of one item.
pub const Q_POOL: usize = 1024;
/// Bytes of one request's arrays, as staged (a request's shared block is 8 MB).
const Q_STAGE: usize = 12 << 20;
/// Cells of all of one request's matrices.
const Q_CELLS: usize = 1 << 20;

/// The arrays of an item, in the order they are staged.
pub const Q_ARRAYS: [&str; 12] = [
    "species", "ability", "item", "moves", "mon", "mask", "side", "field", "acts0", "acts1", "feats0",
    "feats1",
];

pub struct QDims {
    act: usize,
    pair: usize,
    rank: usize,
    feats: usize,
    heads: usize,
    props: usize,
}

/// A Q net on the card: the trunk (`Net`, tensors under "trunk.") and the layers after it.
pub struct QNet {
    pub net: Net,
    q: QDims,
    pub fingerprint: String,
    pub sha256: String,
    pub properties: bool,
}

impl QNet {
    pub fn upload(stream: &Arc<CudaStream>, rnet: &RNet) -> Result<QNet, String> {
        let h = &rnet.header;
        if h.get("kind").and_then(|k| k.as_str()) != Some("q") {
            return Err("not a Q .rnet (kind is not \"q\")".into());
        }
        let num = |key: &str| -> Result<usize, String> {
            h["q"][key].as_u64().map(|n| n as usize).ok_or(format!("the .rnet's q.{key}"))
        };
        let net = Net::upload(stream, rnet)?;
        for name in [
            "move_props", "mon_props.weight", "mon_props.bias", "attend.in_proj_weight", "attend.in_proj_bias",
            "attend.out_proj.weight", "attend.out_proj.bias", "attend_norm.weight", "attend_norm.bias",
            "context.0.weight", "context.1.weight", "kind.weight", "target.weight", "slot_pos.weight",
            "slot_mlp.0.weight", "slot_mlp.1.weight", "slot_mlp.3.weight", "slot_mlp.3.bias",
            "act_mlp.0.weight", "act_mlp.1.weight", "act_mlp.4.weight", "act_mlp.5.weight",
            "first_pair.weight", "first_pair.bias", "second_pair.weight", "pair_out.weight", "pair_out.bias",
            "first_low.weight", "first_low.bias", "second_low.weight", "second_low.bias",
        ] {
            if !net.ptr.contains_key(name) {
                return Err(format!("the Q .rnet has no tensor {name}"));
            }
        }
        let props = rnet.tensors["move_props"].0[1];
        let q = QDims {
            act: num("act_dim")?,
            pair: num("pair_dim")?,
            rank: num("rank")?,
            feats: num("features")?,
            heads: num("heads")?,
            props,
        };
        if q.act > 256 || q.heads * 32 > 256 || net.dims.mon % q.heads != 0 {
            return Err("a Q wider than infer.cu's blocks".into());
        }
        Ok(QNet {
            net,
            q,
            fingerprint: h["vocab_fingerprint"].as_str().unwrap_or("").to_string(),
            sha256: h["sha256"].as_str().unwrap_or("").to_string(),
            properties: h["q"]["properties"].as_bool().unwrap_or(false),
        })
    }

    pub fn widths(&self) -> (usize, usize, usize) {
        Slot::widths_of(&self.net)
    }
}

/// Where one item's arrays are in the staged block, and its matrix in the output.
pub struct QItem {
    pub offsets: [usize; 12],
    pub n0: usize,
    pub n1: usize,
    pub cells_at: usize,
}

pub struct QSlot {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    event: CudaEvent,
    geom: Geometry,
    stage_dev: CudaSlice<u8>,
    stage_host: Pinned,
    out_dev: CudaSlice<f64>,
    out_host: Pinned,
    _workspace: CudaSlice<u8>,
    feat: CudaSlice<f32>,
    h0: CudaSlice<f32>,
    h1: CudaSlice<f32>,
    props: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    att: CudaSlice<f32>,
    hq: CudaSlice<f32>,
    hfin: CudaSlice<f32>,
    pooled: CudaSlice<f32>,
    sides: CudaSlice<f32>,
    ctx_in: CudaSlice<f32>,
    ctx: CudaSlice<f32>,
    x: CudaSlice<f32>,
    s0: CudaSlice<f32>,
    s1: CudaSlice<f32>,
    act_in: CudaSlice<f32>,
    a0: CudaSlice<f32>,
    u: [CudaSlice<f32>; 2],
    fp: [CudaSlice<f32>; 2],
    sp: [CudaSlice<f32>; 2],
    fl: [CudaSlice<f32>; 2],
    sl: [CudaSlice<f32>; 2],
    /// IKA-473: one item's arrays at fixed places (`q_fixed`), what the graphs read.
    fixed_dev: CudaSlice<u8>,
    fixed: [usize; 12],
    /// IKA-473: the trunk's graph and each side's graph by pool size, captured on first use
    /// (keyed by the Q net's address too: a server's Q arms share the slots).
    trunk_graphs: HashMap<usize, CudaGraph>,
    side_graphs: HashMap<(usize, usize, usize), CudaGraph>,
    /// IKA-473: the trunks and sides whose capture failed. They run eagerly from then on, so a
    /// capture that keeps failing is not tried again on every request.
    trunk_failed: std::collections::HashSet<usize>,
    side_failed: std::collections::HashSet<(usize, usize, usize)>,
}

unsafe impl Send for QSlot {}

/// IKA-473: how the Q is run (`POKEURAOU_Q_GRAPHS`): graphs (unset or `1`, the default), eager
/// (`0`), or both with the answers compared bit for bit (`check`). Any other value stops the
/// server at its start. Kept in the shipping build on purpose: `0` is the way back to the eager
/// road without a rebuild, and `check` is how a run on the card shows the two roads agree.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum QMode {
    Graphs,
    Eager,
    Check,
}

impl QMode {
    pub fn from_env() -> Result<QMode, String> {
        match std::env::var("POKEURAOU_Q_GRAPHS").as_deref() {
            Err(_) | Ok("1") => Ok(QMode::Graphs),
            Ok("0") => Ok(QMode::Eager),
            Ok("check") => Ok(QMode::Check),
            Ok(other) => Err(format!("POKEURAOU_Q_GRAPHS is {other:?}; it takes 0, 1 or check")),
        }
    }
}

/// IKA-473: the most side graphs a Q slot keeps. A side graph is captured for each (net, side,
/// pool size) met; past this many, a new size runs eagerly (the same kernels, so the same answer).
pub const Q_SIDE_GRAPHS: usize = 256;

/// IKA-473: Q graphs captured by this process.
pub static Q_CAPTURED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// IKA-473: Q captures that failed; each one's piece runs eagerly from then on.
pub static Q_CAPTURE_FAILED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// IKA-473's control (the counting build only, `POKEURAOU_Q_GRAPH_FAULT=shape`): each side graph
/// is captured for one action fewer than its pool, so the pool's last row is stale.
#[cfg(feature = "served-report")]
fn shape_fault() -> bool {
    static FAULT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FAULT.get_or_init(|| std::env::var("POKEURAOU_Q_GRAPH_FAULT").as_deref() == Ok("shape"))
}
#[cfg(not(feature = "served-report"))]
fn shape_fault() -> bool {
    false
}

/// Where an item's twelve arrays lie in `fixed_dev`: the position's eight one after another at
/// 64-byte boundaries (as `stage_q` lays them), then each side's actions and features with room
/// for Q_POOL rows.
fn q_fixed(geom: Geometry, feats: usize) -> ([usize; 12], usize) {
    let row = geom.row_bytes();
    let acts = crate::legal::SLOTS * crate::legal::ACTION_FIELDS * 4;
    let mut out = [0usize; 12];
    let mut at = 0usize;
    for k in 0..12 {
        at = (at + 63) & !63;
        out[k] = at;
        at += match k {
            0..=7 => row[k],
            8 | 9 => Q_POOL * acts,
            _ => Q_POOL * feats.max(1) * 4,
        };
    }
    (out, (at + 63) & !63)
}

impl QSlot {
    pub fn new(gpu: &Gpu, geom: Geometry, q: &QNet) -> Result<QSlot, String> {
        let e = |e: cudarc::driver::DriverError| e.to_string();
        let stream = gpu.ctx.new_stream().map_err(e)?;
        let blas = CudaBlas::new(stream.clone()).map_err(|e| e.to_string())?;
        let workspace = stream.alloc_zeros::<u8>(32 << 20).map_err(e)?;
        unsafe { bsys::cublasSetWorkspace_v2(*blas.handle(), ptr(&workspace, &stream) as *mut _, 32 << 20) }
            .result()
            .map_err(|e| e.to_string())?;
        let d = &q.net.dims;
        let t = 2 * geom.m;
        let buf = |n: usize| stream.alloc_zeros::<f32>(n.max(1)).map_err(e);
        let (act, pair, rank) = (q.q.act, q.q.pair, q.q.rank);
        let pools = |w: usize| -> Result<[CudaSlice<f32>; 2], String> { Ok([buf(Q_POOL * w)?, buf(Q_POOL * w)?]) };
        Ok(QSlot {
            event: gpu.ctx.new_event(Some(dsys::CUevent_flags::CU_EVENT_BLOCKING_SYNC)).map_err(e)?,
            stage_dev: stream.alloc_zeros::<u8>(Q_STAGE).map_err(e)?,
            stage_host: Pinned::new(Q_STAGE)?,
            out_dev: stream.alloc_zeros::<f64>(Q_CELLS).map_err(e)?,
            out_host: Pinned::new(Q_CELLS * 8)?,
            feat: buf(t * d.mon_in)?,
            h0: buf(t * d.mon)?,
            h1: buf(t * d.mon)?,
            props: buf(t * q.q.props)?,
            qkv: buf(t * 3 * d.mon)?,
            att: buf(t * d.mon)?,
            hq: buf(t * d.mon)?,
            hfin: buf(t * d.mon)?,
            pooled: buf(2 * d.pooled)?,
            sides: buf(2 * d.side)?,
            ctx_in: buf(2 * d.head_in)?,
            ctx: buf(2 * act)?,
            x: buf(Q_POOL * 2 * (16 + d.mon + d.md + q.q.props + 1 + 8 + d.mon + d.mon))?,
            s0: buf(Q_POOL * 2 * act)?,
            s1: buf(Q_POOL * 2 * act)?,
            act_in: buf(Q_POOL * (2 * act + q.q.feats))?,
            a0: buf(Q_POOL * act)?,
            u: pools(act)?,
            fp: pools(pair)?,
            sp: pools(pair)?,
            fl: pools(rank)?,
            sl: pools(rank)?,
            fixed_dev: stream.alloc_zeros::<u8>(q_fixed(geom, q.q.feats).1).map_err(e)?,
            fixed: q_fixed(geom, q.q.feats).0,
            trunk_graphs: HashMap::new(),
            side_graphs: HashMap::new(),
            trunk_failed: Default::default(),
            side_failed: Default::default(),
            _workspace: workspace,
            stream,
            blas,
            geom,
        })
    }

    pub fn stage_mut(&mut self, at: usize, n: usize) -> Result<&mut [u8], String> {
        if at + n > Q_STAGE {
            return Err(format!("a Q request's arrays are over {} MB", Q_STAGE >> 20));
        }
        Ok(self.stage_host.slice_mut(at, n))
    }

    fn p<T>(&self, s: &CudaSlice<T>) -> u64 {
        ptr(s, &self.stream)
    }

    fn lin(&self, w: u64, x: u64, y: u64, n: usize, k: usize, rows: usize, beta: f32) -> Result<(), String> {
        sgemm_nt(&self.blas, w, x, y, n, k, rows, beta)
    }

    fn bln(&self, k: &Kernels, x: u64, rows: usize, n: usize, b: u64, g: u64, be: u64) -> Result<(), String> {
        let n = n as i32;
        let mut l = self.stream.launch_builder(&k.bias_ln_gelu);
        l.arg(&x).arg(&b).arg(&g).arg(&be).arg(&n);
        unsafe { l.launch(cfg(rows, 256)) }.map(|_| ()).map_err(|e| e.to_string())
    }

    fn bias(&self, k: &Kernels, x: u64, rows: usize, n: usize, b: u64) -> Result<(), String> {
        let (n, total) = (n as i32, (rows * n) as i32);
        let mut l = self.stream.launch_builder(&k.add_bias);
        l.arg(&x).arg(&b).arg(&n).arg(&total);
        unsafe { l.launch(cfg((total as usize).div_ceil(256).max(1), 256)) }.map(|_| ()).map_err(|e| e.to_string())
    }

    /// The matrices of `items` (their arrays staged in `stage_mut`, `used` bytes) in `cells`
    /// doubles, valid until the next call.
    ///
    /// IKA-473: with `graphs`, each item's arrays go to their fixed places (`q_fixed`) and the
    /// trunk and each side's actions and pair projections replay graphs captured on first use;
    /// the matrix of both sides stays one eager kernel. The graphs hold the kernels and shapes
    /// the eager road launches, so the answer is the eager road's (`QMode::Check` compares
    /// them on every request). Where the graph road fails, the request is answered on the eager
    /// road and the failure comes back beside the values for the caller to note.
    pub fn matrices(
        &mut self,
        gpu: &Gpu,
        q: &QNet,
        items: &[QItem],
        used: usize,
        cells: usize,
        graphs: bool,
    ) -> Result<(&[f64], Option<String>), String> {
        if cells > Q_CELLS {
            return Err(format!("a Q request's matrices are over {Q_CELLS} cells"));
        }
        let k = &gpu.kernels;
        let mut failed = None;
        if graphs {
            for item in items {
                if let Err(why) = self.item_graphs(k, q, item) {
                    failed = Some(why);
                    break;
                }
            }
        }
        if !graphs || failed.is_some() {
            let src = unsafe { std::slice::from_raw_parts(self.stage_host.ptr, used) };
            unsafe { dres::memcpy_htod_async(self.p(&self.stage_dev), src, self.stream.cu_stream()) }
                .map_err(|e| e.to_string())?;
            for item in items {
                self.item(k, q, item)?;
            }
        }
        let dst = unsafe { std::slice::from_raw_parts_mut(self.out_host.ptr as *mut f64, cells) };
        unsafe { dres::memcpy_dtoh_async(dst, self.p(&self.out_dev), self.stream.cu_stream()) }
            .map_err(|e| e.to_string())?;
        self.event.record(&self.stream).map_err(|e| e.to_string())?;
        self.event.synchronize().map_err(|e| e.to_string())?;
        Ok((unsafe { std::slice::from_raw_parts(self.out_host.ptr as *const f64, cells) }, failed))
    }

    /// One item on the eager road, its arrays where `stage_q` staged them.
    fn item(&self, k: &Kernels, q: &QNet, item: &QItem) -> Result<(), String> {
        let base = self.p(&self.stage_dev);
        let at: Vec<u64> = item.offsets.iter().map(|o| base + *o as u64).collect();
        self.trunk(k, q, &at)?;
        for (side, n) in [item.n0, item.n1].into_iter().enumerate() {
            self.side(k, q, &at, side, n)?;
        }
        self.pair(k, q, item)
    }

    /// One item on the graph road: its arrays copied to their fixed places, the trunk's graph
    /// and each side's graph replayed (each captured the first time it is met), and the pair
    /// matrix launched.
    fn item_graphs(&mut self, k: &Kernels, q: &QNet, item: &QItem) -> Result<(), String> {
        for side in [item.n0, item.n1] {
            if side > Q_POOL {
                return Err(format!("a pool of {side} actions is over {}", Q_POOL));
            }
        }
        let base = self.p(&self.fixed_dev);
        // The position's eight arrays lie as `q_fixed` places them, so one copy moves them.
        let copy = |from: usize, to: usize, n: usize| -> Result<(), String> {
            if n == 0 {
                return Ok(());
            }
            let src = unsafe { std::slice::from_raw_parts(self.stage_host.ptr.add(from), n) };
            unsafe { dres::memcpy_htod_async(base + to as u64, src, self.stream.cu_stream()) }.map_err(|e| e.to_string())
        };
        let position_end = self.fixed[7] + self.geom.row_bytes()[7];
        if (0..8).any(|j| item.offsets[j] - item.offsets[0] != self.fixed[j]) {
            return Err("a Q item's position arrays are not laid as the graphs read them".into());
        }
        copy(item.offsets[0], 0, position_end)?;
        let acts = crate::legal::SLOTS * crate::legal::ACTION_FIELDS * 4;
        for side in 0..2 {
            let n = [item.n0, item.n1][side];
            copy(item.offsets[8 + side], self.fixed[8 + side], n * acts)?;
            copy(item.offsets[10 + side], self.fixed[10 + side], n * q.q.feats * 4)?;
        }
        let at: Vec<u64> = self.fixed.iter().map(|o| base + *o as u64).collect();
        let net_id = q as *const QNet as usize;
        let mut trunks = std::mem::take(&mut self.trunk_graphs);
        let replayed = (|| -> Result<(), String> {
            if self.trunk_failed.contains(&net_id) {
                return self.trunk(k, q, &at);
            }
            if !trunks.contains_key(&net_id) {
                match self.capture(|slot| slot.trunk(k, q, &at)) {
                    Ok(graph) => {
                        trunks.insert(net_id, graph);
                    }
                    Err(why) => {
                        self.capture_failed(&why);
                        self.trunk_failed.insert(net_id);
                        return self.trunk(k, q, &at);
                    }
                }
            }
            trunks[&net_id].launch().map_err(|e| e.to_string())
        })();
        self.trunk_graphs = trunks;
        replayed?;
        for (side, n) in [item.n0, item.n1].into_iter().enumerate() {
            if n == 0 {
                continue;
            }
            let key = (net_id, side, n);
            let mut sides = std::mem::take(&mut self.side_graphs);
            let replayed = (|| -> Result<(), String> {
                if !sides.contains_key(&key) {
                    if sides.len() >= Q_SIDE_GRAPHS || self.side_failed.contains(&key) {
                        return self.side(k, q, &at, side, n);
                    }
                    let rows = if shape_fault() { n.max(2) - 1 } else { n };
                    match self.capture(|slot| slot.side(k, q, &at, side, rows)) {
                        Ok(graph) => {
                            sides.insert(key, graph);
                        }
                        Err(why) => {
                            self.capture_failed(&why);
                            self.side_failed.insert(key);
                            return self.side(k, q, &at, side, n);
                        }
                    }
                }
                sides[&key].launch().map_err(|e| e.to_string())
            })();
            self.side_graphs = sides;
            replayed?;
        }
        self.pair(k, q, item)
    }

    /// A capture that failed: counted, and the first few in the log.
    fn capture_failed(&self, why: &str) {
        if Q_CAPTURE_FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
            eprintln!("  a Q capture failed; that piece runs eagerly from now on: {why}");
        }
    }

    /// `run` once eagerly (the kernels loaded and cuBLAS set up before a capture), then
    /// captured as a graph on the slot's stream. The caller replays it for this request too.
    fn capture(&self, run: impl Fn(&QSlot) -> Result<(), String>) -> Result<CudaGraph, String> {
        run(self)?;
        self.stream
            .begin_capture(dsys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| e.to_string())?;
        let result = run(self);
        let graph = self
            .stream
            .end_capture(dsys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
            .map_err(|e| e.to_string())?;
        result?;
        Q_CAPTURED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        graph.ok_or_else(|| "a Q capture came back empty".to_string())
    }

    /// The trunk on the item's position (`at`: its twelve arrays): the Pokemon rows after
    /// attention (`hfin`) and each side's context (`ctx`).
    fn trunk(&self, k: &Kernels, q: &QNet, at: &[u64]) -> Result<(), String> {
        let (net, qd) = (&q.net, &q.q);
        let d = &net.dims;
        let g = self.geom;
        let s = &self.stream;
        let t = 2 * g.m;
        let i = |v: usize| v as i32;
        // The leaf's trunk on this position: Pokemon rows (+ move properties), the side vectors.
        {
            let (sd, ad, idd, md) = (i(d.sd), i(d.ad), i(d.id), i(d.md));
            let (nsp, nab, nit, nmv) = (i(d.vocab[0]), i(d.vocab[1]), i(d.vocab[2]), i(d.vocab[3]));
            let (wm, id_lo, n_ids, cut, mon_in) = (i(g.wm), i(d.id_lo), i(d.n_ids), i(d.mon_cut), i(d.mon_in));
            let (es, ea, ei, em) =
                (net.w("species.weight"), net.w("ability.weight"), net.w("item.weight"), net.w("move.weight"));
            let feat = self.p(&self.feat);
            let mut l = s.launch_builder(&k.mon_features);
            l.arg(&at[0]).arg(&at[1]).arg(&at[2]).arg(&at[3]).arg(&at[4]);
            l.arg(&es).arg(&ea).arg(&ei).arg(&em);
            l.arg(&sd).arg(&ad).arg(&idd).arg(&md).arg(&nsp).arg(&nab).arg(&nit).arg(&nmv);
            l.arg(&wm).arg(&id_lo).arg(&n_ids).arg(&cut).arg(&mon_in).arg(&feat);
            unsafe { l.launch(cfg(t, 128)) }.map_err(|e| e.to_string())?;
        }
        let (feat, h0, h1) = (self.p(&self.feat), self.p(&self.h0), self.p(&self.h1));
        self.lin(net.w("mon_mlp.0.weight"), feat, h0, d.mon, d.mon_in, t, 0.0)?;
        self.bln(k, h0, t, d.mon, net.w("mon_mlp.0.bias"), net.w("mon_mlp.1.weight"), net.w("mon_mlp.1.bias"))?;
        self.lin(net.w("mon_mlp.3.weight"), h0, h1, d.mon, d.mon, t, 0.0)?;
        self.bln(k, h1, t, d.mon, net.w("mon_mlp.3.bias"), net.w("mon_mlp.4.weight"), net.w("mon_mlp.4.bias"))?;
        // mon = mon_mlp(...) + mon_props(mean of the moves' properties)
        let props = self.p(&self.props);
        {
            let (pd, nmv) = (i(qd.props), i(d.vocab[3]));
            let move_props = net.raw("move_props");
            let mut l = s.launch_builder(&k.props_pool);
            l.arg(&at[3]).arg(&move_props).arg(&pd).arg(&nmv).arg(&props);
            unsafe { l.launch(cfg(t, 128)) }.map_err(|e| e.to_string())?;
        }
        self.lin(net.raw("mon_props.weight"), props, h1, d.mon, qd.props, t, 1.0)?;
        self.bias(k, h1, t, d.mon, net.raw("mon_props.bias"))?;
        // the side vectors from the rows before attention
        let (pooled, sides) = (self.p(&self.pooled), self.p(&self.sides));
        {
            let (m, wm, act, ws, sc, dd) = (i(g.m), i(g.wm), i(d.active), i(g.ws), i(d.side_cut), i(d.mon));
            let mut l = s.launch_builder(&k.pool);
            l.arg(&m).arg(&h1).arg(&at[5]).arg(&at[4]).arg(&wm).arg(&act).arg(&at[6]);
            l.arg(&ws).arg(&sc).arg(&dd).arg(&pooled);
            unsafe { l.launch(cfg(2, 256)) }.map_err(|e| e.to_string())?;
        }
        self.lin(net.w("side_mlp.0.weight"), pooled, sides, d.side, d.pooled, 2, 0.0)?;
        self.bln(k, sides, 2, d.side, net.w("side_mlp.0.bias"), net.w("side_mlp.1.weight"), net.w("side_mlp.1.bias"))?;
        // attention over the rows: hfin = attend_norm(h1 + attention(h1))
        let (qkv, att, hq, hfin) = (self.p(&self.qkv), self.p(&self.att), self.p(&self.hq), self.p(&self.hfin));
        self.lin(net.raw("attend.in_proj_weight"), h1, qkv, 3 * d.mon, d.mon, t, 0.0)?;
        self.bias(k, qkv, t, 3 * d.mon, net.raw("attend.in_proj_bias"))?;
        {
            let (tokens, heads, hd) = (i(t), i(qd.heads), i(d.mon / qd.heads));
            let mut l = s.launch_builder(&k.attend);
            l.arg(&qkv).arg(&at[5]).arg(&tokens).arg(&heads).arg(&hd).arg(&att);
            unsafe { l.launch(cfg(t, d.mon)) }.map_err(|e| e.to_string())?;
        }
        self.lin(net.raw("attend.out_proj.weight"), att, hq, d.mon, d.mon, t, 0.0)?;
        self.bias(k, hq, t, d.mon, net.raw("attend.out_proj.bias"))?;
        {
            let (gamma, beta, n) = (net.raw("attend_norm.weight"), net.raw("attend_norm.bias"), i(d.mon));
            let mut l = s.launch_builder(&k.ln_residual);
            l.arg(&h1).arg(&hq).arg(&gamma).arg(&beta).arg(&hfin).arg(&n);
            unsafe { l.launch(cfg(t, 256)) }.map_err(|e| e.to_string())?;
        }
        // the context of each side: context(sides, field)
        let (ctx_in, ctx) = (self.p(&self.ctx_in), self.p(&self.ctx));
        {
            let (r, sd, wf, fc) = (1i32, i(d.side), i(g.wf), i(d.field_cut));
            let mut l = s.launch_builder(&k.head_input);
            l.arg(&r).arg(&sides).arg(&sd).arg(&at[7]).arg(&wf).arg(&fc).arg(&ctx_in);
            unsafe { l.launch(cfg(2, 256)) }.map_err(|e| e.to_string())?;
        }
        self.lin(net.raw("context.0.weight"), ctx_in, ctx, qd.act, d.head_in, 2, 0.0)?;
        self.bln(k, ctx, 2, qd.act, net.raw("context.0.bias"), net.raw("context.1.weight"), net.raw("context.1.bias"))
    }

    /// Side `side`'s `n` actions -> vectors (`u`), and their pair projections (`fp`, `sp`, `fl`,
    /// `sl`). Nothing for an empty pool (the eager road's projections of no rows were no-ops).
    fn side(&self, k: &Kernels, q: &QNet, at: &[u64], side: usize, n: usize) -> Result<(), String> {
        let (net, qd) = (&q.net, &q.q);
        let d = &net.dims;
        let g = self.geom;
        let s = &self.stream;
        let i = |v: usize| v as i32;
        let (hfin, ctx) = (self.p(&self.hfin), self.p(&self.ctx));
        let slot_width = 16 + d.mon + d.md + qd.props + 1 + 8 + d.mon + d.mon;
        {
            if n == 0 {
                return Ok(());
            }
            if n > Q_POOL {
                return Err(format!("a pool of {n} actions is over {}", Q_POOL));
            }
            let (x, s0, s1, act_in, a0, u) = (
                self.p(&self.x), self.p(&self.s0), self.p(&self.s1), self.p(&self.act_in), self.p(&self.a0),
                self.p(&self.u[side]),
            );
            {
                let (sd, m, hd, md, pd, nmv) = (i(side), i(g.m), i(d.mon), i(d.md), i(qd.props), i(d.vocab[3]));
                let (kind_w, target_w, move_w, move_props) =
                    (net.raw("kind.weight"), net.raw("target.weight"), net.w("move.weight"), net.raw("move_props"));
                let mut l = s.launch_builder(&k.action_input);
                l.arg(&at[8 + side]).arg(&hfin).arg(&sd).arg(&m).arg(&hd).arg(&kind_w).arg(&target_w);
                l.arg(&move_w).arg(&md).arg(&move_props).arg(&pd).arg(&nmv).arg(&x);
                unsafe { l.launch(cfg(n * 2, 256)) }.map_err(|e| e.to_string())?;
            }
            self.lin(net.raw("slot_mlp.0.weight"), x, s0, qd.act, slot_width, n * 2, 0.0)?;
            self.bln(k, s0, n * 2, qd.act, net.raw("slot_mlp.0.bias"), net.raw("slot_mlp.1.weight"), net.raw("slot_mlp.1.bias"))?;
            self.lin(net.raw("slot_mlp.3.weight"), s0, s1, qd.act, qd.act, n * 2, 0.0)?;
            {
                let (bias, pos) = (net.raw("slot_mlp.3.bias"), net.raw("slot_pos.weight"));
                let (ad, nf) = (i(qd.act), i(qd.feats));
                let ctx_side = ctx + (side * qd.act * 4) as u64;
                let mut l = s.launch_builder(&k.slot_sum);
                l.arg(&s1).arg(&bias).arg(&pos).arg(&ctx_side).arg(&at[10 + side]).arg(&ad).arg(&nf).arg(&act_in);
                unsafe { l.launch(cfg(n, 256)) }.map_err(|e| e.to_string())?;
            }
            let width = 2 * qd.act + qd.feats;
            self.lin(net.raw("act_mlp.0.weight"), act_in, a0, qd.act, width, n, 0.0)?;
            self.bln(k, a0, n, qd.act, net.raw("act_mlp.0.bias"), net.raw("act_mlp.1.weight"), net.raw("act_mlp.1.bias"))?;
            self.lin(net.raw("act_mlp.4.weight"), a0, u, qd.act, qd.act, n, 0.0)?;
            self.bln(k, u, n, qd.act, net.raw("act_mlp.4.bias"), net.raw("act_mlp.5.weight"), net.raw("act_mlp.5.bias"))?;
        }
        // the pair head: first_pair / second_pair / first_low / second_low of every vector
        {
            let (u, fp, sp, fl, sl) =
                (self.p(&self.u[side]), self.p(&self.fp[side]), self.p(&self.sp[side]), self.p(&self.fl[side]), self.p(&self.sl[side]));
            self.lin(net.raw("first_pair.weight"), u, fp, qd.pair, qd.act, n, 0.0)?;
            self.bias(k, fp, n, qd.pair, net.raw("first_pair.bias"))?;
            self.lin(net.raw("second_pair.weight"), u, sp, qd.pair, qd.act, n, 0.0)?;
            self.lin(net.raw("first_low.weight"), u, fl, qd.rank, qd.act, n, 0.0)?;
            self.bias(k, fl, n, qd.rank, net.raw("first_low.bias"))?;
            self.lin(net.raw("second_low.weight"), u, sl, qd.rank, qd.act, n, 0.0)?;
            self.bias(k, sl, n, qd.rank, net.raw("second_low.bias"))?;
        }
        Ok(())
    }

    /// The item's N0 x N1 matrix from both sides' projections, into the output at its cells.
    fn pair(&self, k: &Kernels, q: &QNet, item: &QItem) -> Result<(), String> {
        let (net, qd) = (&q.net, &q.q);
        let s = &self.stream;
        let i = |v: usize| v as i32;
        let out = self.p(&self.out_dev) + (item.cells_at * 8) as u64;
        let (n0, n1, pd, rank) = (i(item.n0), i(item.n1), i(qd.pair), i(qd.rank));
        let (w, b) = (net.raw("pair_out.weight"), net.raw("pair_out.bias"));
        let (fpu, spu, flu, slu) = (self.p(&self.fp[0]), self.p(&self.sp[0]), self.p(&self.fl[0]), self.p(&self.sl[0]));
        let (fpv, spv, flv, slv) = (self.p(&self.fp[1]), self.p(&self.sp[1]), self.p(&self.fl[1]), self.p(&self.sl[1]));
        let mut l = s.launch_builder(&k.pair_matrix);
        l.arg(&n0).arg(&n1).arg(&pd).arg(&rank).arg(&fpu).arg(&spu).arg(&flu).arg(&slu);
        l.arg(&fpv).arg(&spv).arg(&flv).arg(&slv).arg(&w).arg(&b).arg(&out);
        let launch = LaunchConfig {
            grid_dim: (item.n1.div_ceil(128).max(1) as u32, item.n0.max(1) as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { l.launch(launch) }.map(|_| ()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_count_has_a_bucket_that_holds_it_and_is_the_smallest() {
        for rows in 1..=MAX_ROWS {
            let b = bucket_of(rows);
            assert!(b < BUCKETS, "{rows}");
            assert!(bucket_rows(b) >= rows, "{rows} in bucket {b} of {}", bucket_rows(b));
            if b > 0 {
                assert!(bucket_rows(b - 1) < rows, "{rows} would fit bucket {}", b - 1);
            }
        }
        assert_eq!(bucket_rows(BUCKETS - 1), MAX_ROWS);
    }

    #[test]
    fn a_rows_bytes_follow_the_encoders_arrays() {
        let g = Geometry { m: 4, wm: 100, ws: 38, wf: 25 };
        // species, ability, item (2 x m i32), moves (2 x m x 4 i32), mon, mask, side, field.
        assert_eq!(g.row_bytes(), [32, 32, 32, 128, 3200, 32, 304, 100]);
    }
}
