/*
 * glm_kernels.cuh — kernels specific to the GLM-5.2 MXFP4 port.
 *
 * OCP MXFP4 microscaling layout:
 *   weight : [N, K/2] uint8, two FP4 E2M1 values per byte, low nibble = even col
 *   scale  : [N, K/32] uint8, E8M0 (value = 2^(byte-127), 0xFF = NaN)
 *   dequant: w[r,c] = e2m1(nibble) * 2^(scale[r, c/32] - 127)
 *
 * Compute: y[N] = W . x  (x is f32)
 *
 * Constraints: K must be a multiple of 32. True for all GLM-5.2 expert
 * tensors (K is 6144 or 2048).
 *
 * Nibble order: "low nibble = even column" (lo-even), VERIFIED against
 * transformers 5.14.1's own MXFP4 dequant,
 * transformers.integrations.mxfp4._convert_moe_packed_tensors (idx_lo =
 * blk & 0x0F feeds out[..., 0::2], idx_hi = blk >> 4 feeds out[..., 1::2]),
 * run as an external oracle over the real
 * model.layers.3.mlp.experts.0.gate_proj.weight tensor from
 * amd/GLM-5.2-MXFP4 ([2048, 6144], cuda_infer/verify_nibble_order.py).
 * lo-even: max abs diff 0.0 against the oracle. hi-even (nibbles swapped):
 * max abs diff 2.03125e-01 -- of order the weight magnitudes themselves, as
 * expected for a wrong ordering. (2026-07-31.)
 *
 * Task 3's own diagnostic (whole-matrix statistics: absmax, std, frac_zero)
 * was inconclusive by construction and could not have settled this: MXFP4's
 * scale block is 32 values = 16 bytes, so both nibbles of every byte always
 * share one scale (see the launch comment below), and swapping lo/hi is
 * exactly a within-pair permutation of which value lands in column 2b vs
 * 2b+1 -- it never changes the multiset of dequantized values in the
 * matrix, so any whole-matrix aggregate statistic is order-invariant by
 * construction. This file's lo-even convention is now settled by the
 * element-wise oracle comparison above, not by that earlier check.
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// warp_reduce_sum / bf16_to_f32 / matvec_bf16 live here. glm_primitives.cuh has
// no dependency on this file, so the include is not circular.
#include "glm_primitives.cuh"

namespace glm {

// E2M1: sign(1) | exp(2) | mantissa(1), bias 1.
// exp==0 -> subnormal (0, 0.5); else 2^(e-1) * (1 + m/2).
__device__ __constant__ float d_e2m1[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float e8m0_to_f32(uint8_t b) {
    // 0xFF is the E8M0 NaN encoding. Return NaN rather than falling through to
    // exp2f(128) = Inf, which would silently poison a whole 32-value block with a
    // finite-looking overflow instead of failing visibly. The CPU reference and the
    // format spec both treat 0xFF as NaN; the kernel must agree. The branch is free
    // in a memory-bound kernel. (Added 2026-07-31.)
    if (b == 0xFF) return __int_as_float(0x7fffffff);
    return exp2f((float)b - 127.0f);
}

// One block = one output row. blockDim.x = 128.
// Each thread walks the row's packed bytes with stride 128; the two nibbles of
// a byte are columns 2b and 2b+1, which always share a scale block because
// 32 values = 16 bytes and (2b)/32 == b/16.
__global__ void dequant_matvec_mxfp4(
    const uint8_t* __restrict__ W,      // [N, K/2]
    const uint8_t* __restrict__ S,      // [N, K/32]
    const float*   __restrict__ x,      // [K]
    float*         __restrict__ y,      // [N]
    uint32_t N, uint32_t K)
{
    const uint32_t row = blockIdx.x;
    if (row >= N) return;
    const uint32_t nbytes = K >> 1;
    const uint32_t nscale = K >> 5;
    const uint8_t* wr = W + (size_t)row * nbytes;
    const uint8_t* sr = S + (size_t)row * nscale;
    const uint32_t tid = threadIdx.x;

    float acc = 0.0f;
    #pragma unroll 1
    for (uint32_t b = tid; b < nbytes; b += 128) {
        const uint8_t byte = wr[b];
        const float scale = e8m0_to_f32(sr[b >> 4]);
        const float w0 = d_e2m1[byte & 0xF];
        const float w1 = d_e2m1[byte >> 4];
        acc = fmaf(w0 * scale, x[2 * b],     acc);
        acc = fmaf(w1 * scale, x[2 * b + 1], acc);
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);

    __shared__ float warp_sum[4];
    const int wid = tid >> 5, lane = tid & 31;
    if (lane == 0) warp_sum[wid] = acc;
    __syncthreads();
    if (tid == 0)
        y[row] = warp_sum[0] + warp_sum[1] + warp_sum[2] + warp_sum[3];
}

inline void launch_dequant_matvec_mxfp4(
    const uint8_t* d_W, const uint8_t* d_S,
    const float* d_x, float* d_y,
    uint32_t N, uint32_t K, cudaStream_t stream = 0)
{
    dequant_matvec_mxfp4<<<dim3(N), dim3(128), 0, stream>>>(d_W, d_S, d_x, d_y, N, K);
}

// --- RoPE, interleaved-in / halves-out -------------------------------------
// MOVED here from glm_layer_runner.cuh by Task 4, byte-for-byte: the main
// attention path and the DSA indexer apply the SAME rotation to different
// slices of different vectors, and Task 3 had to duplicate it because the
// original lived in the header that includes this one. One definition now,
// two callers, `rope_off` being the only difference (main path: qk_nope = 192,
// indexer: 0). Nothing about the kernel body, its launch shape (blockDim =
// rot/2) or its arithmetic changed in the move, which is what makes the
// unification safe against test_glm_chain's substep gate.
//
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
//
// TRAP (Task 1, finding 6): the main path ropes the TRAILING 64 of its 256-dim
// head (rope_off = qk_nope); the indexer ropes the LEADING 64 of its 128-dim
// head (rope_off = 0). Opposite ends of the vector, same rotation.
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

// ============================================================================
// DSA sparse-attention indexer (Task 3)
// ============================================================================
//
// Transcribed from GlmMoeDsaIndexer.forward, transformers 5.14.1
// `models/glm_moe_dsa/modeling_glm_moe_dsa.py` (cited as M:LINE below); the
// line-accurate extraction lives in docs/plans/2026-08-01-dsa-indexer-and-serve.md
// (Task 1). Decode form: one query token at absolute position `pos`, T = pos+1
// keys, matching the rest of this port.
//
//   q      = wq_b(q_resid)                       [H*D]   M:232-234
//   k      = k_norm(wk(x))                       [D]     M:236
//   rope(q[..., 0:R]), rope(k[0:R])              interleaved, M:237-240
//   k -> indexer KV cache at `pos` (bf16)                M:244-245
//   scores[h,t] = relu( (q[h] . k[t]) * D**-0.5 )        M:247-248
//   w[h]        = weights_proj(x)[h] * H**-0.5   fp32    M:251
//   index[t]    = sum_h w[h] * scores[h,t]               M:252
//   topk(index, min(index_topk, T))                      M:262-263
//
// FIVE THINGS THAT WOULD COMPILE AND BE SILENTLY WRONG — all of them established
// from the source by Task 1, none of them detectable by the <=index_topk
// bit-identity gate Task 4 uses:
//
//  1. `k_norm` is nn.LayerNorm(128, eps=1e-6) (M:199), NOT an RMSNorm. It
//     subtracts the mean and adds a bias. Every other norm in this model is an
//     RMSNorm and rms_norm_bf16() is sitting right next door in
//     glm_primitives.cuh; reaching for it is the natural mistake.
//  2. The eps is a hardcoded `1e-6` literal at construction, not
//     config.rms_norm_eps (1e-5). INDEXER_LN_EPS below, deliberately its own
//     constant and deliberately not Config::rms_eps.
//  3. RoPE goes on the LEADING R=64 of the indexer's 128 dims (M:234, M:237).
//     The MAIN attention path ropes the TRAILING 64 of its 256
//     (glm_layer_runner.cuh, rope_off = qk_nope). Same (cos,sin) table, same
//     interleaved convention, opposite end of the vector.
//  4. ReLU comes BEFORE the head combination (M:248 then M:252), and there are
//     TWO distinct scales: D**-0.5 on the qk product and H**-0.5 on the head
//     weights. Combining first and rectifying after is a different function.
//  5. The selected indices are NOT causal in general (Task 1 measured
//     [0,1,2,10] for query row 2 at topk=4). Causality here is structural
//     rather than enforced: this is a decode-form port, T = pos+1, so every
//     cached key is already causal. Anything that ever calls
//     launch_indexer_decode with keys beyond `pos` must mask them itself, and
//     must not treat the returned set as pre-filtered.
//
// The indexer's RoPE is rope_interleave_slice (above) with rope_off = 0. Task 3
// had a private copy of it, `indexer_rope_lead`, because the original lived in
// glm_layer_runner.cuh, which includes this header. Task 4 moved the kernel here
// and deleted the copy — there is exactly one rotation in the repo now.

// LayerNorm eps for indexer.k_norm — M:199, a literal, NOT Config::rms_eps.
static const float INDEXER_LN_EPS = 1e-6f;

// Round-to-nearest-even f32 -> bf16, matching torch's cast. The indexer KV
// cache stores bf16 (M:244-245 keeps k in its own dtype), so this rounding is
// part of the arithmetic, not a storage detail.
__device__ __forceinline__ uint16_t f32_to_bf16_rne(float f) {
    const uint32_t u = __float_as_uint(f);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)0x7FC0;   // NaN stays NaN
    const uint32_t rnd = ((u >> 16) & 1u) + 0x7FFFu;
    return (uint16_t)((u + rnd) >> 16);
}

// Block-wide sum, broadcast to every thread. `red` must be >= 33 floats of
// shared memory; slot 32 is the broadcast slot, so it never collides with the
// per-warp slots (at most 32 warps -> slots 0..31). Ends on a barrier, so the
// same scratch can be reused by a second call.
__device__ __forceinline__ float indexer_block_sum(float v, float* red) {
    const uint32_t lane = threadIdx.x & 31u, wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) red[wid] = v;
    __syncthreads();
    const uint32_t nwarp = (blockDim.x + 31u) >> 5;
    v = (threadIdx.x < nwarp) ? red[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_reduce_sum(v);
    if (threadIdx.x == 0) red[32] = v;
    __syncthreads();
    const float out = red[32];
    __syncthreads();
    return out;
}

// indexer.k_norm — nn.LayerNorm(dim, eps) with weight AND bias, both bf16.
// out[i] = (x[i] - mean) / sqrt(var + eps) * w[i] + b[i],  var biased (1/N).
// Safe in place (x == out): every read of x finishes before the write loop,
// and each thread writes only the elements it read.
//
// 1/sqrtf rather than rsqrtf: rsqrtf is a ~2-ulp approximation, and the whole
// point of this kernel is that a 1e-6-vs-1e-5 eps difference must be visible.
// One block per token — the cost is irrelevant.
__global__ void indexer_k_layernorm(
    const float*    __restrict__ x,
    const uint16_t* __restrict__ w,
    const uint16_t* __restrict__ b,
    float*          __restrict__ out,
    uint32_t dim, float eps)
{
    __shared__ float red[33];
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) acc += x[i];
    const float mean = indexer_block_sum(acc, red) / (float)dim;

    float vacc = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = x[i] - mean;
        vacc += d * d;
    }
    const float var = indexer_block_sum(vacc, red) / (float)dim;
    const float inv = 1.0f / sqrtf(var + eps);

    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
        out[i] = (x[i] - mean) * inv * bf16_to_f32(w[i]) + bf16_to_f32(b[i]);
}

// Append the post-norm, post-RoPE key to the indexer's own KV cache, in bf16.
__global__ void indexer_store_k_bf16(
    const float* __restrict__ k, uint16_t* __restrict__ row, uint32_t dim)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) row[i] = f32_to_bf16_rne(k[i]);
}

// w[h] = (weights_proj @ x)[h] * head_scale, all fp32 (M:251 — weights_proj is
// forced to fp32 by _keep_in_fp32_modules and its input is explicitly upcast).
// One warp per head.
__global__ void indexer_head_weights(
    const float* __restrict__ W,     // [H, hidden] fp32
    const float* __restrict__ x,     // [hidden]    fp32
    float*       __restrict__ out,   // [H]
    uint32_t H, uint32_t hidden, float head_scale)
{
    const uint32_t h = blockIdx.x * blockDim.y + threadIdx.y;
    if (h >= H) return;
    const float* wr = W + (size_t)h * hidden;
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < hidden; i += 32) acc += wr[i] * __ldg(x + i);
    acc = warp_reduce_sum(acc);
    if (threadIdx.x == 0) out[h] = acc * head_scale;
}

// scores[h,t] = relu( (q[h] . k[t]) * qk_scale ). One warp per (h,t); the
// linear index is h*T + t so that consecutive warps walk consecutive keys with
// the same query row.
__global__ void indexer_scores_relu(
    const float*    __restrict__ q,        // [H, D] fp32, post-RoPE
    const uint16_t* __restrict__ k_cache,  // [T, D] bf16, post-norm post-RoPE
    float*          __restrict__ scores,   // [H, T] fp32
    uint32_t H, uint32_t T, uint32_t D, float qk_scale)
{
    const uint32_t idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= H * T) return;
    const uint32_t h = idx / T, t = idx - h * T;
    const float*    qh = q + (size_t)h * D;
    const uint16_t* kt = k_cache + (size_t)t * D;
    float acc = 0.0f;
    for (uint32_t d = threadIdx.x; d < D; d += 32)
        acc += qh[d] * bf16_to_f32(__ldg(kt + d));
    acc = warp_reduce_sum(acc);
    // relu AFTER the scale and BEFORE the head combination (M:248). Combining
    // first and rectifying after is a different function.
    if (threadIdx.x == 0) scores[idx] = fmaxf(acc * qk_scale, 0.0f);
}

// index[t] = sum_h w[h] * scores[h,t]  (M:252). One thread per key.
__global__ void indexer_combine(
    const float* __restrict__ scores,   // [H, T]
    const float* __restrict__ w,        // [H]
    float*       __restrict__ out,      // [T]
    uint32_t H, uint32_t T)
{
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    float acc = 0.0f;
    for (uint32_t h = 0; h < H; h++) acc = fmaf(w[h], scores[(size_t)h * T + t], acc);
    out[t] = acc;
}

// Order-preserving map float -> uint32: a > b  <=>  key(a) > key(b), for every
// pair of non-NaN floats including negatives and both zeros.
__device__ __forceinline__ uint32_t indexer_ordered_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// top-k over `index` (M:262-263), k = min(index_topk, T) applied by the caller.
// Single block, MSD radix select on the ordered key: four 8-bit passes find the
// exact k-th largest value, then one parallel pass emits everything strictly
// above it and one serial pass tops up from the ties.
//
// ORDER OF `out` IS NOT MEANINGFUL. torch passes no `sorted=` and defaults to
// descending, but the indices are only ever scattered into a bool mask
// (M:423-427), so only the SET matters. Ties are genuinely under-determined —
// after the relu a large fraction of index scores are exactly 0.0, and which
// zeros fill the tail is arbitrary in both implementations. This kernel breaks
// ties by lowest index, which is deterministic but is NOT torch's rule; no test
// may assert index-for-index equality against torch above the tie boundary.
__global__ void indexer_topk(
    const float* __restrict__ index,   // [T]
    uint32_t T, uint32_t k,
    int32_t*  __restrict__ out,        // [k]
    uint32_t* __restrict__ out_n)      // may be null
{
    __shared__ uint32_t hist[256];
    __shared__ uint32_t sh_prefix, sh_kk, sh_cnt;
    const uint32_t tid = threadIdx.x, nthr = blockDim.x;

    if (k >= T) {                       // topk = min(index_topk, T): every key
        for (uint32_t i = tid; i < T; i += nthr) out[i] = (int32_t)i;
        if (tid == 0 && out_n) *out_n = T;
        return;
    }

    uint32_t prefix = 0, kk = k;
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        // High bits already pinned by `prefix`. `1u << 32` is UB, hence the
        // explicit pass-0 case (nothing is pinned yet).
        const uint32_t hi = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
        for (uint32_t i = tid; i < 256; i += nthr) hist[i] = 0;
        __syncthreads();
        for (uint32_t i = tid; i < T; i += nthr) {
            const uint32_t u = indexer_ordered_key(index[i]);
            if ((u & hi) == (prefix & hi)) atomicAdd(&hist[(u >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            uint32_t cum = 0; int d = 255;
            for (; d > 0; d--) { if (cum + hist[d] >= kk) break; cum += hist[d]; }
            sh_prefix = prefix | ((uint32_t)d << shift);
            sh_kk     = kk - cum;
        }
        __syncthreads();
        prefix = sh_prefix; kk = sh_kk;
        __syncthreads();
    }
    // `prefix` is now the ordered key of the k-th largest score and `kk` (>= 1)
    // is how many of the elements equal to it are needed.
    if (tid == 0) sh_cnt = 0;
    __syncthreads();
    for (uint32_t i = tid; i < T; i += nthr)
        if (indexer_ordered_key(index[i]) > prefix)
            out[atomicAdd(&sh_cnt, 1u)] = (int32_t)i;
    __syncthreads();
    if (tid == 0) {
        uint32_t n = sh_cnt, need = kk;
        for (uint32_t i = 0; i < T && need; i++)
            if (indexer_ordered_key(index[i]) == prefix) { out[n++] = (int32_t)i; need--; }
        if (out_n) *out_n = n;
    }
}

// --- the two halves of one indexer forward ---------------------------------
// Split because they run at different rates: the KEY is computed once per
// position and lives in the cache forever; the QUERY side re-reads the whole
// cache on every token. Task 4 calls launch_indexer_decode, which is both.
//
// `x` is the raw input_layernorm output (M:603 -> M:408), `hidden` = 6144 wide,
// NOT the compressed KV and NOT q_resid. `q_resid` is the SAME tensor the main
// attention path already computed — q_a_layernorm(q_a_proj(x)), M:385 -> M:409
// — so do not recompute it. d_cos/d_sin must hold the table for the position
// being processed (rot/2 entries each); run_layer already uploads exactly that
// table for the main path and the indexer shares it.
//
// Everything from `s_*` on is caller-owned scratch (Task 4 puts it in
// LayerWeights).

// k[pos] = rope_lead( k_norm( wk(x) ) ), stored to the indexer KV cache in bf16.
inline void launch_indexer_key(
    const uint16_t* wk,            // [D, hidden] bf16
    const uint16_t* k_norm_w,      // [D] bf16
    const uint16_t* k_norm_b,      // [D] bf16
    const float*    x,             // [hidden] fp32
    const float*    d_cos, const float* d_sin,
    uint16_t*       k_cache,       // [max_seq, D] bf16
    float*          s_k,           // [D] fp32 scratch
    uint32_t D, uint32_t rot, uint32_t hidden, uint32_t pos,
    float ln_eps, cudaStream_t stream = 0)
{
    launch_matvec_bf16(wk, x, s_k, D, hidden, stream);
    indexer_k_layernorm<<<1, 128, 0, stream>>>(s_k, k_norm_w, k_norm_b, s_k, D, ln_eps);
    rope_interleave_slice<<<1, rot / 2, 0, stream>>>(s_k, d_cos, d_sin, D, 0, rot);
    indexer_store_k_bf16<<<(D + 127) / 128, 128, 0, stream>>>(
        s_k, k_cache + (size_t)pos * D, D);
}

// The query side over the first `T` cached keys: scores, head combination,
// top-k. `T` is pos+1 in decode form, which is what makes every cached key
// causal here — see trap 5 in the header.
inline void launch_indexer_query(
    const uint16_t* wq_b,          // [H*D, q_lora] bf16
    const float*    weights_proj,  // [H, hidden]   fp32
    const float*    q_resid,       // [q_lora] fp32
    const float*    x,             // [hidden] fp32
    const float*    d_cos, const float* d_sin,
    const uint16_t* k_cache,       // [T, D] bf16
    float* s_q, float* s_w, float* s_scores, float* s_index,
    int32_t* out_idx, uint32_t* out_n,
    uint32_t H, uint32_t D, uint32_t rot, uint32_t hidden, uint32_t q_lora,
    uint32_t T, uint32_t index_topk, cudaStream_t stream = 0)
{
    const uint32_t k = (index_topk < T) ? index_topk : T;    // M:262
    const float qk_scale   = rsqrtf((float)D);  // 128**-0.5, M:201/M:247
    const float head_scale = rsqrtf((float)H);  // 32**-0.5,  M:251 — a SECOND, different scale

    launch_matvec_bf16(wq_b, q_resid, s_q, H * D, q_lora, stream);
    rope_interleave_slice<<<H, rot / 2, 0, stream>>>(s_q, d_cos, d_sin, D, 0, rot);
    indexer_head_weights<<<(H + 7) / 8, dim3(32, 8), 0, stream>>>(
        weights_proj, x, s_w, H, hidden, head_scale);
    indexer_scores_relu<<<(H * T + 7) / 8, dim3(32, 8), 0, stream>>>(
        s_q, k_cache, s_scores, H, T, D, qk_scale);
    indexer_combine<<<(T + 255) / 256, 256, 0, stream>>>(s_scores, s_w, s_index, H, T);
    indexer_topk<<<1, 256, 0, stream>>>(s_index, T, k, out_idx, out_n);
}

// One indexer forward for one query token at absolute position `pos`.
inline void launch_indexer_decode(
    const uint16_t* wq_b, const uint16_t* wk,
    const uint16_t* k_norm_w, const uint16_t* k_norm_b,
    const float* weights_proj, const float* q_resid, const float* x,
    const float* d_cos, const float* d_sin,
    uint16_t* k_cache,
    float* s_q, float* s_k, float* s_w, float* s_scores, float* s_index,
    int32_t* out_idx, uint32_t* out_n,
    uint32_t H, uint32_t D, uint32_t rot, uint32_t hidden, uint32_t q_lora,
    uint32_t pos, uint32_t index_topk, float ln_eps, cudaStream_t stream = 0)
{
    launch_indexer_key(wk, k_norm_w, k_norm_b, x, d_cos, d_sin, k_cache, s_k,
                       D, rot, hidden, pos, ln_eps, stream);
    launch_indexer_query(wq_b, weights_proj, q_resid, x, d_cos, d_sin, k_cache,
                         s_q, s_w, s_scores, s_index, out_idx, out_n,
                         H, D, rot, hidden, q_lora, pos + 1, index_topk, stream);
}

// --- index_mask (Task 4) ---------------------------------------------------
// M:423-427:
//     index_mask = ones([B,S,T], bool)
//     index_mask.scatter_(-1, topk_indices, False)          # True == DROP
//     attn_mask  = causal_mask.masked_fill(index_mask, finfo(dtype).min)
//
// Two properties of that composition, both load-bearing, both from Task 1:
//
//  * The CAUSAL mask is the BASE. masked_fill only writes `min` where index_mask
//    is True, so a causally-forbidden key stays forbidden whether or not the
//    indexer named it. The index set never *un*-masks anything.
//  * The selected indices are NOT causal. Task 1 measured topk=4 returning
//    [0,1,2,10] for query row 2. So causality must be applied INDEPENDENTLY and
//    the index set must never be treated as pre-filtered.
//
// In this decode-form port the causal base is structural: the query is at
// absolute position `pos`, the attention loop runs t in [0, T) with T = pos+1,
// and every t in that range is causally allowed. The scatter below therefore
// enforces causality by REFUSING to clear any index outside [0, T) — a
// non-causal index simply unmasks nothing, which is exactly what
// `causal.masked_fill(...)` does to it. Widening the attention loop past `pos`
// (a prefill kernel, say) would break that and needs a real causal test here.
//
// Layout: one uint8 per key, 1 == drop. A separate byte array rather than the
// additive float mask transformers uses, because the score buffer here is
// shared memory owned by mla_decode_absorbed and the mask has to be built
// before that kernel launches.

__global__ void index_mask_fill(uint8_t* __restrict__ m, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) m[i] = 1;
}

__global__ void index_mask_scatter(const int32_t* __restrict__ idx, uint32_t n,
                                   uint8_t* __restrict__ m, uint32_t T) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int32_t v = idx[i];
    if (v >= 0 && (uint32_t)v < T) m[v] = 0;   // causality: out-of-range unmasks nothing
}

// Builds the [T] drop-mask for one query position from `n` raw indexer indices.
inline void launch_build_index_mask(const int32_t* d_idx, uint32_t n,
                                    uint8_t* d_mask, uint32_t T,
                                    cudaStream_t stream = 0) {
    index_mask_fill<<<(T + 255) / 256, 256, 0, stream>>>(d_mask, T);
    if (n) index_mask_scatter<<<(n + 255) / 256, 256, 0, stream>>>(d_idx, n, d_mask, T);
}

} // namespace glm
