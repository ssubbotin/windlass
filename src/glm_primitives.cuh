/*
 * glm_primitives.cuh — small CUDA primitives shared by the GLM layer runner.
 *
 * Copyright (c) 2026 Sergey Subbotin <ssubbotin@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Four generic building blocks: two warp-shuffle reductions, an RMS norm with
 * bf16 weights, and an elementwise residual add.
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// One warp per output row; ROWS_PER_BLOCK warps per block.
#define ROWS_PER_BLOCK 8

__device__ __forceinline__ float bf16_to_f32(uint16_t bf16) {
    return __uint_as_float((uint32_t)bf16 << 16);
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__global__ void rms_norm_bf16(
    const float* __restrict__ x,
    const uint16_t* __restrict__ weight,
    float* __restrict__ out,
    uint32_t dim,
    float eps
) {
    __shared__ float shared[32];
    float acc = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
        acc += x[i] * x[i];

    acc = warp_reduce_sum(acc);
    uint32_t wid = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    if (lane == 0) shared[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (lane == 0) shared[0] = acc;
    }
    __syncthreads();

    float rms = rsqrtf(shared[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x)
        out[i] = x[i] * rms * bf16_to_f32(weight[i]);
}

// ============================================================================
// 5. Residual add: out = a + b
// ============================================================================


__global__ void residual_add(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    uint32_t dim
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) out[i] = a[i] + b[i];
}

static inline void launch_rms_norm_bf16(const float* x, const uint16_t* w, float* out, uint32_t dim, float eps, cudaStream_t s = 0) {
    rms_norm_bf16<<<1, 256, 0, s>>>(x, w, out, dim, eps);
}

static inline void launch_residual_add(const float* a, const float* b, float* out, uint32_t dim, cudaStream_t s = 0) {
    residual_add<<<(dim+255)/256, 256, 0, s>>>(a, b, out, dim);
}

__global__ void matvec_bf16(
    const uint16_t* __restrict__ W,       // [out_dim, in_dim] bf16
    const float*    __restrict__ x,       // [in_dim] f32
    float*          __restrict__ out,     // [out_dim] f32
    uint32_t out_dim,
    uint32_t in_dim
) {
    const uint32_t lane    = threadIdx.x;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t row     = blockIdx.x * ROWS_PER_BLOCK + warp_id;
    if (row >= out_dim) return;

    const uint16_t* w_row = W + (size_t)row * in_dim;
    float acc = 0.0f;
    for (uint32_t i = lane; i < in_dim; i += 32) {
        acc += bf16_to_f32(__ldg(w_row + i)) * __ldg(x + i);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) out[row] = acc;
}

static inline void launch_matvec_bf16(
    const uint16_t* W, const float* x, float* out,
    uint32_t out_dim, uint32_t in_dim,
    cudaStream_t stream = 0
) {
    dim3 block(32, ROWS_PER_BLOCK);
    dim3 grid((out_dim + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
    matvec_bf16<<<grid, block, 0, stream>>>(W, x, out, out_dim, in_dim);
}
