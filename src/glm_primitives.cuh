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
