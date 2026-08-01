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

} // namespace glm
