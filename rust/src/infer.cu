// IKA-461: the value net's forward pass, one kernel per step of `value.ValueNet.forward`.
// Compiled by NVRTC when the inference server starts (`infer_gpu.rs`). Row-major float32
// throughout; the matrix products are cuBLAS's. Each kernel has a stage test against torch
// (`tests/test_rust_inference_server.py`, op "stages").

#define EPS 1e-5f

__device__ __forceinline__ float block_sum(float v, float* shared) {
    // 256 threads. Fixed order, so a given input always gives the same answer.
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) shared[warp] = v;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x < 8) total = shared[threadIdx.x];
    for (int o = 4; o > 0; o >>= 1) total += __shfl_down_sync(0xffu, total, o);
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = total;
    __syncthreads();
    float out = shared[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ int safe_index(int id, int vocab) {
    return ((unsigned)id < (unsigned)vocab) ? id : 0;
}

// The first layer's input of the per-Pokemon MLP (`ValueNet.mon_inputs`): the species,
// ability and item embeddings, the four moves' mean embedding, the numeric columns, the
// embeddings of the move ids carried in the numeric block, and the columns after them.
extern "C" __global__ void mon_features(
    const int* species, const int* ability, const int* item, const int* moves,
    const float* mon, const float* es, const float* ea, const float* ei, const float* em,
    int sd, int ad, int id_dim, int md, int nsp, int nab, int nit, int nmv,
    int wm, int id_lo, int n_ids, int cut, int mon_in, float* out) {
    int t = blockIdx.x;
    int sp = safe_index(species[t], nsp), ab = safe_index(ability[t], nab);
    int it = safe_index(item[t], nit);
    int mv[4];
    float count = 0.0f;
    for (int k = 0; k < 4; ++k) {
        int m = moves[t * 4 + k];
        mv[k] = safe_index(m, nmv);
        if (m > 0) count += 1.0f;
    }
    float denom = count < 1.0f ? 1.0f : count;
    int o1 = sd, o2 = o1 + ad, o3 = o2 + id_dim, o4 = o3 + md, o5 = o4 + id_lo;
    int o6 = o5 + n_ids * md;
    for (int c = threadIdx.x; c < mon_in; c += blockDim.x) {
        float v;
        if (c < o1) v = es[sp * sd + c];
        else if (c < o2) v = ea[ab * ad + (c - o1)];
        else if (c < o3) v = ei[it * id_dim + (c - o2)];
        else if (c < o4) {
            int j = c - o3;
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) {
                int m = moves[t * 4 + k];
                s += em[mv[k] * md + j] * (m > 0 ? 1.0f : 0.0f);
            }
            v = s / denom;
        } else if (c < o5) v = mon[(size_t)t * wm + (c - o4)];
        else if (c < o6) {
            int q = c - o5, e = q / md, j = q % md;
            int id = safe_index((int)mon[(size_t)t * wm + id_lo + e], nmv);
            v = em[id * md + j];
        } else v = mon[(size_t)t * wm + id_lo + n_ids + (c - o6)];
        out[(size_t)t * mon_in + c] = v;
    }
}

// x[row] = gelu(layer_norm(x[row] + bias)) over n <= 256 columns, in place.
extern "C" __global__ void bias_ln_gelu(float* x, const float* bias, const float* gamma,
                                        const float* beta, int n) {
    __shared__ float shared[32];
    float* row = x + (size_t)blockIdx.x * n;
    int c = threadIdx.x;
    float v = c < n ? row[c] + bias[c] : 0.0f;
    float mean = block_sum(v, shared) / (float)n;
    float d = c < n ? v - mean : 0.0f;
    float var = block_sum(d * d, shared) / (float)n;
    float rstd = rsqrtf(var + EPS);
    if (c < n) {
        float y = d * rstd * gamma[c] + beta[c];
        row[c] = 0.5f * y * (1.0f + erff(y * 0.70710678118654752f));
    }
}

// Pools each side's Pokemon (`_masked_mean`, `_masked_max` over the active and the bench
// ones) and appends the side's own columns. out: (rows * 2, 4 * d + side_cut).
extern "C" __global__ void pool(
    int m, const float* mon_h, const float* mask, const float* mon, int wm, int act_col,
    const float* side, int ws, int side_cut, int d, float* out) {
    int b = blockIdx.x;  // row * 2 + side
    int p = 4 * d + side_cut;
    float* o = out + (size_t)b * p;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float asum = 0.0f, acnt = 0.0f, bsum = 0.0f, bcnt = 0.0f;
        float amax = __int_as_float(0xff800000), bmax = amax;  // -inf
        for (int k = 0; k < m; ++k) {
            size_t t = (size_t)b * m + k;
            float v = mon_h[t * d + c];
            float present = mask[t];
            float is_act = mon[t * wm + act_col];
            float wa = present * is_act, wb = present * (1.0f - is_act);
            asum += v * wa;
            acnt += wa;
            bsum += v * wb;
            bcnt += wb;
            if (wa > 0.0f && v > amax) amax = v;
            if (wb > 0.0f && v > bmax) bmax = v;
        }
        o[c] = asum / (acnt < 1.0f ? 1.0f : acnt);
        o[d + c] = isfinite(amax) ? amax : 0.0f;
        o[2 * d + c] = bsum / (bcnt < 1.0f ? 1.0f : bcnt);
        o[3 * d + c] = isfinite(bmax) ? bmax : 0.0f;
    }
    for (int c = threadIdx.x; c < side_cut; c += blockDim.x)
        o[4 * d + c] = side[(size_t)b * ws + c];
}

// The head's input for both orders: rows [0, r) are (ours, theirs, field), rows [r, 2r)
// are (theirs, ours, field). sides: (r * 2, s).
extern "C" __global__ void head_input(int r, const float* sides, int s, const float* field,
                                      int wf, int field_cut, float* out) {
    int i = blockIdx.x;
    int row = i < r ? i : i - r;
    const float* first = sides + ((size_t)row * 2 + (i < r ? 0 : 1)) * s;
    const float* second = sides + ((size_t)row * 2 + (i < r ? 1 : 0)) * s;
    int width = 2 * s + field_cut;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        float v;
        if (c < s) v = first[c];
        else if (c < 2 * s) v = second[c - s];
        else v = field[(size_t)row * wf + (c - 2 * s)];
        out[(size_t)i * width + c] = v;
    }
}

// logit = head(ours, theirs) - head(theirs, ours) for one net: the last Linear (h -> 1).
extern "C" __global__ void head_final(int r, const float* h, int hd, const float* w,
                                      const float* bias, float* logits) {
    __shared__ float shared[32];
    int row = blockIdx.x;
    float f = 0.0f, mi = 0.0f;
    for (int c = threadIdx.x; c < hd; c += blockDim.x) {
        f += h[(size_t)row * hd + c] * w[c];
        mi += h[((size_t)row + r) * hd + c] * w[c];
    }
    f = block_sum(f, shared);
    mi = block_sum(mi, shared);
    if (threadIdx.x == 0) logits[row] = (f + bias[0]) - (mi + bias[0]);
}

// The mean of the nets' logits, then sigmoid in float32 and widened to double.
extern "C" __global__ void finalize(int r, int nets, const float* logits, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= r) return;
    float s = logits[i];
    for (int k = 1; k < nets; ++k) s += logits[(size_t)k * r + i];
    float mean = nets > 1 ? s / (float)nets : s;
    out[i] = (double)(1.0f / (1.0f + expf(-mean)));
}
