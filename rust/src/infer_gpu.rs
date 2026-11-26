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
        let vocab = |name: &str| rnet.tensors.get(name).map(|t| t.0[0]).ok_or(format!("no tensor {name}"));
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
        for name in [
            "mon_mlp.0.weight", "mon_mlp.1.weight", "mon_mlp.3.weight", "mon_mlp.4.weight",
            "side_mlp.0.weight", "side_mlp.1.weight", "head.0.weight", "head.1.weight",
            "head.4.weight", "head.5.weight", "head.8.weight", "head.8.bias",
        ] {
            if !ptr.contains_key(name) {
                return Err(format!("the .rnet has no tensor {name}"));
            }
        }
        Ok(Net { dims, ptr, _hold: hold, source: rnet.source.clone() })
    }

    fn w(&self, name: &str) -> u64 {
        self.ptr[name]
    }
}

pub struct Kernels {
    mon_features: CudaFunction,
    bias_ln_gelu: CudaFunction,
    pool: CudaFunction,
    head_input: CudaFunction,
    head_final: CudaFunction,
    finalize: CudaFunction,
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
        // y (rows, n) = x (rows, k) . w (n, k)^T, row-major: the transposed product in
        // cuBLAS's column-major terms.
        let (one, zero) = (1.0f32, 0.0f32);
        unsafe {
            bres::sgemm(
                *self.blas.handle(),
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
                &zero,
                y as *mut f32,
                n as i32,
            )
        }
        .map_err(|e| e.to_string())
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
