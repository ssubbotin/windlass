/*
 * glm_layer_runner.cuh — one GLM-5.2 decoder layer, decode form (one token).
 *
 * Transcribed from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa
 * (transformers 5.14.1). Citations below are `M:LINE` into that file.
 *
 * Layer order (M:602-620):
 *   residual = h;  h = input_layernorm(h);  h = self_attn(h);  h = residual + h
 *   residual = h;  h = post_attention_layernorm(h);  h = mlp(h);  h = residual + h
 *
 * Attention is MLA in *absorbed* form: the KV cache holds the compressed
 * kv_c [max_seq, kv_lora_rank] (post kv_a_layernorm) plus k_rot [max_seq, qk_rope]
 * (rope'd, NOT normalised), and kv_b_proj is absorbed into q / out instead of
 * being materialised per cached position. Algebraically identical; see Step 4.3
 * of the plan.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>

#include "glm_primitives.cuh"
#include "glm_kernels.cuh"
#include "glm_loader.cuh"

namespace glm {

struct ExpertSource {
    virtual uint8_t* get(uint32_t layer, uint32_t expert) = 0;

    // Optional Task-12 prefetch pair. run_moe knows all K routed experts the
    // instant routing completes, so it announces them once (prefetch) and then
    // consumes them one at a time (get_async). A source that has nothing to
    // overlap ignores prefetch and get_async degrades to get() — which is what
    // test_glm_layer.cu's checkpoint-backed source does. Neither call changes
    // WHICH bytes are fetched, only when the fetch is started.
    virtual void prefetch(uint32_t /*layer*/, const int32_t* /*experts*/,
                          uint32_t /*n*/, cudaStream_t /*stream*/) {}
    virtual uint8_t* get_async(uint32_t layer, uint32_t expert,
                               cudaStream_t /*stream*/) { return get(layer, expert); }
    virtual ~ExpertSource() = default;
};

// Sub-offsets inside one packed 20 MB expert block, matching
// repack_experts_glm.py. Recomputed from Config at init, not hardcoded here.
struct ExpertLayout {
    size_t gw_off, gw_len, gs_off, gs_len;
    size_t uw_off, uw_len, us_off, us_len;
    size_t dw_off, dw_len, ds_off, ds_len;
    size_t total;
    static ExpertLayout from(const Config& c);
};

inline ExpertLayout ExpertLayout::from(const Config& c) {
    const size_t H = c.hidden, I = c.moe_inter;
    ExpertLayout l{};
    size_t o = 0;
    l.gw_off = o; l.gw_len = I * H / 2;  o += l.gw_len;
    l.gs_off = o; l.gs_len = I * H / 32; o += l.gs_len;
    l.uw_off = o; l.uw_len = I * H / 2;  o += l.uw_len;
    l.us_off = o; l.us_len = I * H / 32; o += l.us_len;
    l.dw_off = o; l.dw_len = H * I / 2;  o += l.dw_len;
    l.ds_off = o; l.ds_len = H * I / 32; o += l.ds_len;
    l.total = o;
    return l;
}

// Optional substep capture, for the oracle test only. run_layer copies the
// named intermediates out before they are overwritten (s_norm and s_acc are both
// reused by run_moe, and d_hidden is updated twice). Null in production, and
// every field is independently optional; no allocation and no cost when unset.
// This is a Task-5 addition on top of the plan's Step 4 spec — it changes no
// published signature and no arithmetic.
struct Trace {
    float* post_input_norm  = nullptr;   // [hidden]
    float* attn_out         = nullptr;   // [hidden]  self_attn output, pre-residual
    float* post_attn_hidden = nullptr;   // [hidden]  after the first residual add
};
inline Trace* g_trace = nullptr;

// Optional wall-clock accounting, for infer_glm --timing only. Null in every
// other configuration, and NOTHING here changes what run_layer/run_moe compute:
// the only added work is a cudaStreamSynchronize at each section boundary, which
// is what makes the attribution meaningful in the first place (every launch below
// is asynchronous, so without the barriers all four buckets would collapse into
// whichever call happened to block). The barriers themselves cost time and DO
// depress throughput slightly, so the gate figure must be measured with --timing
// OFF and the breakdown read as a proportion, not as an absolute budget.
//
// Buckets:
//   attn   — input_layernorm through o_proj and the first residual add.
//   router — post_attention_layernorm, the router matvec and the top-k select
//            (MoE layers); the whole dense MLP on layers < first_k_dense_replace.
//   fetch  — time inside ExpertSource::get, i.e. the cache hit/miss path.
//   expert — the MXFP4 matvecs for the K routed experts and the shared expert.
struct Timing {
    double attn = 0, router = 0, fetch = 0, expert = 0;
    uint64_t layers = 0, fetches = 0;
    void reset() { attn = router = fetch = expert = 0; layers = fetches = 0; }
    double total() const { return attn + router + fetch + expert; }
};
inline Timing* g_timing = nullptr;

inline double glm_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

// Block reductions over <= 1024 threads. Both call __syncthreads() unconditionally,
// so every thread of the block must reach them.
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float s[32];
    const uint32_t lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    const uint32_t nwarp = (blockDim.x + 31) / 32;
    v = (threadIdx.x < nwarp) ? s[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    return s[0];
}

__device__ __forceinline__ float block_reduce_max(float v) {
    __shared__ float s[32];
    const uint32_t lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_reduce_max(v);
    if (lane == 0) s[wid] = v;
    __syncthreads();
    const uint32_t nwarp = (blockDim.x + 31) / 32;
    v = (threadIdx.x < nwarp) ? s[threadIdx.x] : -INFINITY;
    if (wid == 0) v = warp_reduce_max(v);
    if (threadIdx.x == 0) s[0] = v;
    __syncthreads();
    return s[0];
}

// --- RoPE, interleaved-in / halves-out -------------------------------------
// GLM applies DeepSeek-style *interleaved* RoPE (M:133-169, called at M:396 and
// M:240). Reading M:164-168:
//     q1, q2 = q[..., 0::2], q[..., 1::2]
//     q_embed = cat([q1*cos - q2*sin, q2*cos + q1*sin], dim=-1)
// i.e. the INPUT pairs are adjacent (2i, 2i+1) but the OUTPUT is the
// halves layout [rot_dim/2 | rot_dim/2]. That is a permutation of the rope
// slice, applied identically to q and k, so the q·k dot product is unchanged.
// cos/sin are cat(freqs, freqs) (M:126) and only the first half is used
// (M:161-162), so the table is rot_dim/2 = 32 entries per position with
// inv_freq[i] = 1 / theta^(2i / rot_dim), rot_dim = qk_rope_head_dim = 64
// (M:112-114 with dim = config.head_dim = qk_rope_head_dim, C:153).
__global__ void rope_interleave_slice(
    float* __restrict__ x,            // [n_vec, vec_dim]; rope slice at [rope_off, rope_off+rot)
    const float* __restrict__ cos_p,  // [rot/2]
    const float* __restrict__ sin_p,  // [rot/2]
    uint32_t vec_dim, uint32_t rope_off, uint32_t rot)
{
    const uint32_t v    = blockIdx.x;
    const uint32_t i    = threadIdx.x;
    const uint32_t half = rot >> 1;
    const bool     act  = (i < half);
    float* p = x + (size_t)v * vec_dim + rope_off;
    // Read both members of the interleaved pair before any write; the two halves
    // of the output alias the input slice. No early `return` before the barrier —
    // every thread of the block must reach __syncthreads().
    float a = 0.f, b = 0.f, cc = 0.f, ss = 0.f;
    if (act) { a = p[2 * i]; b = p[2 * i + 1]; cc = cos_p[i]; ss = sin_p[i]; }
    __syncthreads();                      // every pair read before any write
    if (act) {
        p[i]        = a * cc - b * ss;
        p[i + half] = b * cc + a * ss;
    }
}

// Host-side table for one position. rot = qk_rope_head_dim = 64, theta = 8e6.
inline void rope_pos(float* cos_out, float* sin_out, uint32_t pos,
                     uint32_t rot, float theta) {
    for (uint32_t i = 0; i < rot / 2; i++) {
        const float inv = 1.0f / std::pow(theta, (float)(2 * i) / (float)rot);
        cos_out[i] = std::cos((float)pos * inv);
        sin_out[i] = std::sin((float)pos * inv);
    }
}

// --- MLA absorption --------------------------------------------------------
// q_abs[h][c] = sum_{d < qk_nope} q_pass[h][d] * kv_b[h*(qk_nope+v_head) + d][c]
// One warp per output element.
__global__ void mla_absorb_q(
    const float*    __restrict__ q,      // [n_heads, qk_head]  (nope | rope)
    const uint16_t* __restrict__ kv_b,   // [n_heads*(qk_nope+v_head), kv_lora]
    float*          __restrict__ q_abs,  // [n_heads, kv_lora]
    uint32_t n_heads, uint32_t qk_head, uint32_t qk_nope,
    uint32_t v_head, uint32_t kv_lora)
{
    const uint32_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= n_heads * kv_lora) return;
    const uint32_t h = idx / kv_lora, c = idx % kv_lora;
    const float* qh = q + (size_t)h * qk_head;
    const uint16_t* w = kv_b + (size_t)h * (qk_nope + v_head) * kv_lora;
    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < qk_nope; d += 32)
        acc += qh[d] * bf16_to_f32(__ldg(w + (size_t)d * kv_lora + c));
    acc = warp_reduce_sum(acc);
    if (threadIdx.x == 0) q_abs[idx] = acc;
}

// attn_out[h][v] = sum_c ctx[h][c] * kv_b[h*(qk_nope+v_head) + qk_nope + v][c]
__global__ void mla_project_v(
    const float*    __restrict__ ctx,    // [n_heads, kv_lora]
    const uint16_t* __restrict__ kv_b,
    float*          __restrict__ out,    // [n_heads, v_head]
    uint32_t n_heads, uint32_t qk_nope, uint32_t v_head, uint32_t kv_lora)
{
    const uint32_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= n_heads * v_head) return;
    const uint32_t h = idx / v_head, v = idx % v_head;
    const float* ch = ctx + (size_t)h * kv_lora;
    const uint16_t* w = kv_b + ((size_t)h * (qk_nope + v_head) + qk_nope + v) * kv_lora;
    float acc = 0.0f;
    for (uint32_t c = threadIdx.x; c < kv_lora; c += 32)
        acc += ch[c] * bf16_to_f32(__ldg(w + c));
    acc = warp_reduce_sum(acc);
    if (threadIdx.x == 0) out[idx] = acc;
}

// One block per head. Causal over t in [0, T); T = pos + 1 (this token included).
// s[t] = (q_abs[h] . kv_c[t] + q_rot[h] . k_rot[t]) * scaling
__global__ void mla_decode_absorbed(
    const float* __restrict__ q,        // [n_heads, qk_head]
    const float* __restrict__ q_abs,    // [n_heads, kv_lora]
    const float* __restrict__ kv_c,     // [T, kv_lora]
    const float* __restrict__ k_rot,    // [T, qk_rope]
    float*       __restrict__ ctx,      // [n_heads, kv_lora]
    uint32_t T, uint32_t qk_head, uint32_t qk_nope, uint32_t qk_rope,
    uint32_t kv_lora, float scaling)
{
    extern __shared__ float sh[];       // T floats
    const uint32_t h = blockIdx.x;
    const float* qa = q_abs + (size_t)h * kv_lora;
    const float* qr = q + (size_t)h * qk_head + qk_nope;

    for (uint32_t t = threadIdx.x; t < T; t += blockDim.x) {
        float s = 0.0f;
        const float* kc = kv_c + (size_t)t * kv_lora;
        for (uint32_t c = 0; c < kv_lora; c++) s += qa[c] * kc[c];
        const float* kr = k_rot + (size_t)t * qk_rope;
        for (uint32_t r = 0; r < qk_rope; r++) s += qr[r] * kr[r];
        sh[t] = s * scaling;
    }
    __syncthreads();

    // Block-wide max then sum, then the weighted accumulation into ctx.
    float m = -INFINITY;
    for (uint32_t t = threadIdx.x; t < T; t += blockDim.x) m = fmaxf(m, sh[t]);
    m = block_reduce_max(m);
    float z = 0.0f;
    for (uint32_t t = threadIdx.x; t < T; t += blockDim.x) {
        const float e = __expf(sh[t] - m);
        sh[t] = e; z += e;
    }
    z = block_reduce_sum(z);
    const float inv_z = 1.0f / z;

    for (uint32_t c = threadIdx.x; c < kv_lora; c += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t t = 0; t < T; t++) acc += sh[t] * kv_c[(size_t)t * kv_lora + c];
        ctx[(size_t)h * kv_lora + c] = acc * inv_z;
    }
}

// --- router ----------------------------------------------------------------
// Single block. Transcribes GlmMoeDsaTopkRouter.forward (M:485-510) for the
// n_group == topk_group == 1 case, where the group mask is all-ones and the
// masked_fill at M:503 is a no-op.
__global__ void router_sigmoid_topk(
    const float* __restrict__ logits,   // [E] f32, already computed by matvec_bf16
    const float* __restrict__ bias,     // [E] f32, e_score_correction_bias
    float*   __restrict__ topk_w,       // [K]
    int32_t* __restrict__ topk_i,       // [K]
    uint32_t E, uint32_t K, float routed_scaling, bool norm_topk_prob)
{
    extern __shared__ float sc[];       // 2*E floats: [0,E) = score, [E,2E) = score+bias
    for (uint32_t e = threadIdx.x; e < E; e += blockDim.x) {
        const float s = 1.0f / (1.0f + __expf(-logits[e]));   // M:488  scores = router_logits.sigmoid()
        sc[e]     = s;
        sc[E + e] = s + bias[e];                              // M:489  scores_for_choice
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    // Selection is on score+bias (M:504); the WEIGHT is gathered from the
    // bias-free score (M:505). Getting this backwards is the classic trap.
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
        int best = -1; float bv = -INFINITY;
        for (uint32_t e = 0; e < E; e++)
            if (sc[E + e] > bv) { bv = sc[E + e]; best = (int)e; }
        sc[E + best] = -INFINITY;                             // consume
        topk_i[k] = best;
        topk_w[k] = sc[best];
        sum += sc[best];
    }
    if (norm_topk_prob) {                                     // M:506-508
        const float inv = 1.0f / (sum + 1e-20f);
        for (uint32_t k = 0; k < K; k++) topk_w[k] *= inv;
    }
    for (uint32_t k = 0; k < K; k++) topk_w[k] *= routed_scaling;  // M:509
}

__global__ void silu_mul(const float* g, const float* u, float* out, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (g[i] / (1.0f + __expf(-g[i]))) * u[i];   // hidden_act == "silu" (J, C:113)
}

__global__ void axpy(const float* x, float a, float* acc, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += a * x[i];
}

// --- run_moe ---------------------------------------------------------------
// d_hidden is the *pre-norm* residual stream after attention; run_moe applies
// post_attention_layernorm itself and writes the block output (WITHOUT the
// residual) to d_out.
inline void run_moe(const Config& c, const LayerWeights& w, const ExpertLayout& lay,
                    ExpertSource& src, uint32_t layer,
                    const float* d_hidden, float* d_out, cudaStream_t stream)
{
    const uint32_t H = c.hidden, I = c.moe_inter, E = c.n_routed_experts,
                   K = c.n_experts_per_tok;
    // Shared expert's intermediate size is moe_intermediate_size * n_shared_experts
    // (M:563-565) — 2048 * 1 here, but derive it rather than reusing I.
    const uint32_t SI = c.moe_inter * c.n_shared;

    // ALIASING: run_layer calls run_moe(..., d_hidden, w.s_acc, stream), so
    // d_out == w.s_acc. d_out is the accumulator ONLY. Every intermediate that is
    // later axpy'd into it must live in a different buffer — hence w.s_expert for
    // the routed down-projection and w.s_shared for the shared one. Writing the
    // down-proj into w.s_acc and then axpy'ing w.s_acc into d_out would evaluate
    // `d_out += a * d_out`, which compiles, runs, and is silently wrong.
    const double t_moe0 = g_timing ? glm_now() : 0.0;

    rms_norm_bf16<<<1, 256, 0, stream>>>(d_hidden, w.post_attn_ln, w.s_norm, H, c.rms_eps);

    // ---- dense layers (layer < first_k_dense_replace): plain GlmMoeDsaMLP (M:456-469)
    if (layer < c.dense_first) {
        const uint32_t DI = c.dense_inter;                      // 12288
        launch_matvec_bf16(w.mlp_gate, w.s_norm, w.s_gate, DI, H, stream);
        launch_matvec_bf16(w.mlp_up,   w.s_norm, w.s_up,   DI, H, stream);
        silu_mul<<<(DI + 255) / 256, 256, 0, stream>>>(w.s_gate, w.s_up, w.s_mid, DI);
        launch_matvec_bf16(w.mlp_down, w.s_mid, d_out, H, DI, stream);
        if (g_timing) {
            cudaStreamSynchronize(stream);
            g_timing->router += glm_now() - t_moe0;   // dense MLP, bucketed with router
        }
        return;
    }

    // ---- router ----
    launch_matvec_bf16(w.router_w, w.s_norm, w.s_logits, E, H, stream);
    router_sigmoid_topk<<<1, 256, 2 * E * sizeof(float), stream>>>(
        w.s_logits, w.router_bias, w.s_w, w.s_idx, E, K, c.routed_scaling, c.norm_topk_prob);

    // Fixed-size staging for the K selected experts. load_config refuses any
    // config with num_experts_per_tok > MAX_TOPK, so K <= MAX_TOPK holds here.
    int32_t h_idx[MAX_TOPK]; float h_w[MAX_TOPK];
    cudaMemcpyAsync(h_idx, w.s_idx, K * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_w,   w.s_w,   K * sizeof(float),   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // The sync above is unconditional (the host reads h_idx/h_w next), so the
    // router bucket is already a true elapsed figure with no barrier added.
    if (g_timing) g_timing->router += glm_now() - t_moe0;
    const double t_exp0 = g_timing ? glm_now() : 0.0;
    double fetch_acc = 0.0;

    cudaMemsetAsync(d_out, 0, H * sizeof(float), stream);

    // Announce all K experts before touching any of them, so the source can
    // start every read now instead of after the previous expert's matvecs.
    // Ordering against the compute already enqueued on `stream` is the
    // source's responsibility (see glm_expert_cache.cuh).
    {
        const double t_p0 = g_timing ? glm_now() : 0.0;
        src.prefetch(layer, h_idx, K, stream);
        if (g_timing) fetch_acc += glm_now() - t_p0;
    }

    // ---- routed experts ----
    for (uint32_t k = 0; k < K; k++) {
        const double t_f0 = g_timing ? glm_now() : 0.0;
        uint8_t* blk = src.get_async(layer, (uint32_t)h_idx[k], stream);
        if (g_timing) { fetch_acc += glm_now() - t_f0; g_timing->fetches++; }
        const uint8_t* gw = blk + lay.gw_off; const uint8_t* gs = blk + lay.gs_off;
        const uint8_t* uw = blk + lay.uw_off; const uint8_t* us = blk + lay.us_off;
        const uint8_t* dw = blk + lay.dw_off; const uint8_t* ds = blk + lay.ds_off;
        launch_dequant_matvec_mxfp4(gw, gs, w.s_norm, w.s_gate, I, H, stream);
        launch_dequant_matvec_mxfp4(uw, us, w.s_norm, w.s_up,   I, H, stream);
        silu_mul<<<(I + 255) / 256, 256, 0, stream>>>(w.s_gate, w.s_up, w.s_mid, I);
        launch_dequant_matvec_mxfp4(dw, ds, w.s_mid, w.s_expert, H, I, stream);
        axpy<<<(H + 255) / 256, 256, 0, stream>>>(w.s_expert, h_w[k], d_out, H);
    }

    // ---- shared expert, on the SAME normalized input, added AFTER (M:573) ----
    launch_dequant_matvec_mxfp4(w.sh_gate_w, w.sh_gate_s, w.s_norm, w.s_gate, SI, H, stream);
    launch_dequant_matvec_mxfp4(w.sh_up_w,   w.sh_up_s,   w.s_norm, w.s_up,   SI, H, stream);
    silu_mul<<<(SI + 255) / 256, 256, 0, stream>>>(w.s_gate, w.s_up, w.s_mid, SI);
    launch_dequant_matvec_mxfp4(w.sh_down_w, w.sh_down_s, w.s_mid, w.s_shared, H, SI, stream);
    axpy<<<(H + 255) / 256, 256, 0, stream>>>(w.s_shared, 1.0f, d_out, H);

    if (g_timing) {
        cudaStreamSynchronize(stream);
        g_timing->fetch  += fetch_acc;
        g_timing->expert += (glm_now() - t_exp0) - fetch_acc;
    }
}

// --- run_layer -------------------------------------------------------------
inline void run_layer(const Config& c, const LayerWeights& w, const ExpertLayout& lay,
                      ExpertSource& src, uint32_t layer, uint32_t pos,
                      float* d_hidden, cudaStream_t stream)
{
    const uint32_t H = c.hidden, HN = c.n_heads, DH = c.qk_head, L = c.kv_lora_rank;
    const double t_attn0 = g_timing ? glm_now() : 0.0;

    // ---- input_layernorm -> attention ----
    rms_norm_bf16<<<1, 256, 0, stream>>>(d_hidden, w.input_ln, w.s_norm, H, c.rms_eps);
    if (g_trace && g_trace->post_input_norm)
        cudaMemcpyAsync(g_trace->post_input_norm, w.s_norm, H * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

    launch_matvec_bf16(w.q_a_proj, w.s_norm, w.s_qa, c.q_lora_rank, H, stream);
    // LORA_NORM_EPS (1e-6), NOT c.rms_eps (1e-5) — M:339 passes no eps, so the
    // GlmMoeDsaRMSNorm class default at M:49 applies.
    rms_norm_bf16<<<1, 256, 0, stream>>>(w.s_qa, w.q_a_ln, w.s_qa, c.q_lora_rank, LORA_NORM_EPS);
    launch_matvec_bf16(w.q_b_proj, w.s_qa, w.s_q, HN * DH, c.q_lora_rank, stream);

    launch_matvec_bf16(w.kv_a_proj_with_mqa, w.s_norm, w.s_kv, L + c.qk_rope, H, stream);
    float* kv_row  = w.kv_c_cache  + (size_t)pos * L;
    float* rot_row = w.k_rot_cache + (size_t)pos * c.qk_rope;
    // LORA_NORM_EPS again — M:347 also passes no eps (M:49 default).
    rms_norm_bf16<<<1, 256, 0, stream>>>(w.s_kv, w.kv_a_ln, kv_row, L, LORA_NORM_EPS);
    cudaMemcpyAsync(rot_row, w.s_kv + L, c.qk_rope * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);   // NOT normalised (M:390-391)

    // RoPE at `pos`: q_rot is the trailing qk_rope of each of HN heads; k_rot is one vector.
    const uint32_t half = c.qk_rope / 2;                 // 32
    std::vector<float> h_cos(half), h_sin(half);
    rope_pos(h_cos.data(), h_sin.data(), pos, c.qk_rope, c.rope_theta);
    cudaMemcpyAsync(w.d_cos, h_cos.data(), half * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(w.d_sin, h_sin.data(), half * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    // h_cos/h_sin are pageable host memory, so these copies are synchronous with
    // respect to the host and the vectors may be destroyed on return. If they are
    // ever changed to pinned memory, add a cudaStreamSynchronize here.
    // PERF: pageable H2D also means an implicit host/stream sync once per layer per
    // token — 78 per decode step. Negligible next to 8 MXFP4 expert matvecs per
    // layer, but it is the first thing to hoist if the RoPE table is ever
    // precomputed for all positions instead of rebuilt per step.
    rope_interleave_slice<<<HN, half, 0, stream>>>(
        w.s_q, w.d_cos, w.d_sin, DH, c.qk_nope, c.qk_rope);
    rope_interleave_slice<<<1, half, 0, stream>>>(
        rot_row, w.d_cos, w.d_sin, c.qk_rope, 0, c.qk_rope);

    const uint32_t T = pos + 1;
    // Exactness bound for the indexer-free path — see Step 4.3.
    if (T > c.index_topk) {
        fprintf(stderr, "run_layer: seq %u exceeds index_topk %u; the dense path is "
                        "no longer exact\n", T, c.index_topk);
        std::abort();
    }

    mla_absorb_q<<<(HN * L + 3) / 4, dim3(32, 4), 0, stream>>>(
        w.s_q, w.kv_b_proj, w.s_qabs, HN, DH, c.qk_nope, c.v_head, L);
    mla_decode_absorbed<<<HN, 256, T * sizeof(float), stream>>>(
        w.s_q, w.s_qabs, w.kv_c_cache, w.k_rot_cache, w.s_ctx,
        T, DH, c.qk_nope, c.qk_rope, L, 1.0f / std::sqrt((float)DH));
    mla_project_v<<<(HN * c.v_head + 3) / 4, dim3(32, 4), 0, stream>>>(
        w.s_ctx, w.kv_b_proj, w.s_attn, HN, c.qk_nope, c.v_head, L);
    launch_matvec_bf16(w.o_proj, w.s_attn, w.s_acc, H, HN * c.v_head, stream);
    if (g_trace && g_trace->attn_out)
        cudaMemcpyAsync(g_trace->attn_out, w.s_acc, H * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

    // ---- residual ----
    residual_add<<<(H + 255) / 256, 256, 0, stream>>>(d_hidden, w.s_acc, d_hidden, H);
    if (g_trace && g_trace->post_attn_hidden)
        cudaMemcpyAsync(g_trace->post_attn_hidden, d_hidden, H * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

    if (g_timing) {
        cudaStreamSynchronize(stream);
        g_timing->attn += glm_now() - t_attn0;
        g_timing->layers++;
    }

    // ---- MLP / MoE + residual ----  (run_moe applies post_attention_layernorm itself)
    run_moe(c, w, lay, src, layer, d_hidden, w.s_acc, stream);
    residual_add<<<(H + 255) / 256, 256, 0, stream>>>(d_hidden, w.s_acc, d_hidden, H);
}

// --- whole-model chain -----------------------------------------------------
// The three tensors that live outside any decoder layer, plus the two [hidden]
// working buffers the chain needs. Kept separate from LayerWeights because none
// of them is per-layer and because the caller owns their lifetime.
struct ChainWeights {
    uint16_t* embed      = nullptr;   // model.embed_tokens.weight [vocab, hidden] bf16
    uint16_t* final_norm = nullptr;   // model.norm.weight         [hidden]        bf16
    uint16_t* lm_head    = nullptr;   // lm_head.weight            [vocab, hidden] bf16
    float*    d_hidden   = nullptr;   // [hidden] the residual stream
    float*    d_hnorm    = nullptr;   // [hidden] model.norm output, feeds lm_head
};

// Optional per-layer observation point, for the chain test only. Fires after
// run_layer returns, while that layer's own scratch (s_norm = post_post_norm,
// s_logits, s_idx, s_w, s_shared, s_acc = moe_out) still holds this step's
// values — every layer owns its own scratch, but the NEXT token's pass over the
// same layer overwrites it, so a substep fixture must be snapshotted here.
// Enqueue copies on `stream`; they are ordered against run_layer's work.
// Null in production; no cost when unset, and no arithmetic depends on it.
struct ChainHook {
    virtual void after_layer(uint32_t layer, uint32_t pos, const LayerWeights& w,
                             const float* d_hidden, cudaStream_t stream) = 0;
    virtual ~ChainHook() = default;
};

// out[i] = bf16_to_f32(embed[token * hidden + i])
__global__ void embed_lookup_bf16(const uint16_t* __restrict__ tbl, uint32_t token,
                                  float* __restrict__ out, uint32_t hidden) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) out[i] = bf16_to_f32(tbl[(size_t)token * hidden + i]);
}

// Runs `n_tokens` decode steps at absolute positions [pos0, pos0 + n_tokens),
// then model.norm + lm_head on the LAST token only — logits for earlier
// positions are never a generation step and the chain reference does not
// produce them either.
//
// DEVIATION from the plan's Interfaces sketch, which reads
//   run_chain(cfg, layers, layout, src, tokens, n_tokens, d_logits, stream).
// Three additions, none of which changes what the function computes:
//   * `cw`     — embed / model.norm / lm_head are not in LayerWeights and the
//                sketch named no other home for them.
//   * `pos0`   — without it, generating token t would have to replay the whole
//                prompt from position 0 (the KV cache lives in LayerWeights and
//                is keyed by absolute position). For a 29-token prompt and 5
//                generated tokens that is 155 forward steps instead of 33, and
//                ~5x the expert-cache traffic, for identical output.
//   * `hook`   — see ChainHook. Pass nullptr in production.
// Layer 78 (the MTP head) is deliberately not run: c.n_layers is 78, i.e.
// layers 0..77.
inline void run_chain(const Config& c, const std::vector<LayerWeights>& W,
                      const ChainWeights& cw, const ExpertLayout& lay,
                      ExpertSource& src,
                      const uint32_t* tokens, uint32_t n_tokens, uint32_t pos0,
                      float* d_logits, ChainHook* hook, cudaStream_t stream)
{
    const uint32_t H = c.hidden;
    for (uint32_t i = 0; i < n_tokens; i++) {
        const uint32_t pos = pos0 + i;
        embed_lookup_bf16<<<(H + 255) / 256, 256, 0, stream>>>(
            cw.embed, tokens[i], cw.d_hidden, H);
        for (uint32_t l = 0; l < c.n_layers; l++) {
            run_layer(c, W[l], lay, src, l, pos, cw.d_hidden, stream);
            if (hook) hook->after_layer(l, pos, W[l], cw.d_hidden, stream);
        }
    }
    rms_norm_bf16<<<1, 256, 0, stream>>>(cw.d_hidden, cw.final_norm, cw.d_hnorm,
                                         H, c.rms_eps);
    launch_matvec_bf16(cw.lm_head, cw.d_hnorm, d_logits, c.vocab, H, stream);
}

} // namespace glm
