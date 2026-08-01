#!/usr/bin/env python3
"""verify_nibble_order.py — settle lo-even vs hi-even against transformers'
own MXFP4 dequant (Task 3b).

Task 3's whole-matrix statistics check (absmax/std/frac_zero) cannot
discriminate nibble order: a 32-value scale block is exactly 16 bytes, so
both nibbles of every byte always share one scale, and swapping lo/hi is a
pure permutation of which value lands in column 2b vs 2b+1 -- it never
changes the dequantized multiset. This script instead dequantizes the real
probe tensor with transformers' own kernel
(transformers.integrations.mxfp4._convert_moe_packed_tensors, transformers
5.14.1) as an external oracle, and compares it element-wise against our two
candidate orderings.

Usage (inside a venv with transformers >= 5.12):
    python3 tools/verify_nibble_order.py \
        --probe-dir /tmp/glm-probe
"""
import argparse
import json
import os

import numpy as np
import torch

from glm_mxfp4_numpy import dequant as _lo_even_dequant

# Same E2M1 code table as glm_kernels.cuh / test_glm_mxfp4.cu.
E2M1 = np.array([
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float64)


def load_tensor(path_base):
    meta = json.load(open(path_base + ".json"))
    assert meta["dtype"] == "U8"
    arr = np.fromfile(path_base + ".bin", dtype=np.uint8)
    arr = arr.reshape(meta["shape"])
    return arr


def our_dequant(blocks: np.ndarray, scales: np.ndarray, hi_even: bool) -> np.ndarray:
    """Reimplementation of the kernel's math (lo-even, or swapped hi-even).

    lo-even is the settled ordering (cuda_infer/glm_mxfp4_numpy.py is now the
    single implementation of it); this function still computes both orderings
    locally so this script keeps its own side-by-side comparison logic.
    """
    if not hi_even:
        return _lo_even_dequant(blocks, scales).astype(np.float64)

    N, nbytes = blocks.shape
    K = nbytes * 2
    lo = blocks & 0x0F
    hi = (blocks >> 4) & 0x0F
    even_idx, odd_idx = hi, lo

    w_even = E2M1[even_idx]              # [N, nbytes] -> columns 0,2,4,...
    w_odd = E2M1[odd_idx]                # [N, nbytes] -> columns 1,3,5,...

    # scale index per byte column b is b // 16 (block of 32 values = 16 bytes)
    block_of_byte = np.arange(nbytes) // 16
    scale_pow = scales.astype(np.int32) - 127
    scale_f = np.ldexp(1.0, scale_pow)              # [N, nscale]
    scale_per_byte = scale_f[:, block_of_byte]        # [N, nbytes]

    out = np.empty((N, K), dtype=np.float64)
    out[:, 0::2] = w_even * scale_per_byte
    out[:, 1::2] = w_odd * scale_per_byte
    return out


def oracle_dequant(blocks_u8: np.ndarray, scales_u8: np.ndarray) -> np.ndarray:
    """Dequant via transformers' own MXFP4 conversion path (the external oracle)."""
    from transformers.integrations.mxfp4 import _convert_moe_packed_tensors

    N, nbytes = blocks_u8.shape
    nscale = scales_u8.shape[1]
    assert nbytes == nscale * 16, f"{nbytes=} {nscale=}"

    blocks_t = torch.from_numpy(blocks_u8.copy()).reshape(1, N, nscale, 16)
    scales_t = torch.from_numpy(scales_u8.copy()).reshape(1, N, nscale)

    out = _convert_moe_packed_tensors(blocks_t, scales_t, dtype=torch.float64)
    # _convert_moe_packed_tensors ends with .transpose(1, 2); undo that plus the
    # dummy leading dim we added so the result lines up [N, K] row-major, same
    # layout as the raw .bin file (row = output channel, col = input channel).
    assert out.shape == (1, K_of(nscale), N)
    out = out.squeeze(0).transpose(0, 1).contiguous()
    return out.numpy()


def K_of(nscale):
    return nscale * 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", default="/tmp/glm-probe")
    args = ap.parse_args()

    base = os.path.join(
        args.probe_dir, "model.layers.3.mlp.experts.0.gate_proj.weight"
    )
    blocks = load_tensor(base)              # [2048, 3072] U8
    scales = load_tensor(base + "_scale")   # [2048, 192]  U8

    N, nbytes = blocks.shape
    K = nbytes * 2
    print(f"probe tensor: blocks {blocks.shape} scales {scales.shape} -> K={K}")

    oracle = oracle_dequant(blocks, scales)
    print(f"oracle shape {oracle.shape}, dtype {oracle.dtype}")

    lo_even = our_dequant(blocks, scales, hi_even=False)
    hi_even = our_dequant(blocks, scales, hi_even=True)

    diff_lo = np.abs(lo_even - oracle)
    diff_hi = np.abs(hi_even - oracle)

    max_lo = float(diff_lo.max())
    max_hi = float(diff_hi.max())

    print(f"max abs diff, lo-even vs oracle: {max_lo:.6e}")
    print(f"max abs diff, hi-even vs oracle: {max_hi:.6e}")

    exact_thresh = 1e-6
    magnitude_thresh = 1e-3  # weight-scale differences are ~0.1; anything
                              # this large can only mean "wrong ordering"

    lo_exact = max_lo < exact_thresh
    hi_exact = max_hi < exact_thresh

    if lo_exact and not hi_exact and max_hi > magnitude_thresh:
        print("VERDICT: lo-even (matches oracle exactly; hi-even does not)")
    elif hi_exact and not lo_exact and max_lo > magnitude_thresh:
        print("VERDICT: hi-even (matches oracle exactly; lo-even does not)")
    else:
        print("VERDICT: BLOCKED -- neither ordering unambiguously matches the "
              "oracle, or both do. Do not trust either result; the oracle "
              "wiring or the E2M1 table needs re-checking.")


if __name__ == "__main__":
    main()
