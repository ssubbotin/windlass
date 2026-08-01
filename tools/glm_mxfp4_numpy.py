#!/usr/bin/env python3
"""glm_mxfp4_numpy.py — reference OCP MXFP4 dequantization in numpy.

Layout must match cuda_infer/glm_kernels.cuh: [N, K/2] packed FP4 E2M1 pairs
(low nibble = even column) with [N, K/32] E8M0 scale bytes.

Nibble order (lo-even) and the E2M1/E8M0 formulas were verified exactly
(0.0 diff) against transformers.integrations.mxfp4._convert_moe_packed_tensors
in Task 3b (see verify_nibble_order.py, which now imports dequant() from
this module instead of carrying its own copy).
"""
import numpy as np

E2M1 = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.,
                 -0., -.5, -1., -1.5, -2., -3., -4., -6.], dtype=np.float32)

def dequant(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    assert w.dtype == np.uint8 and s.dtype == np.uint8, (w.dtype, s.dtype)
    N, half = w.shape
    K = half * 2
    assert s.shape == (N, K // 32), (s.shape, (N, K // 32))
    out = np.empty((N, K), dtype=np.float32)
    out[:, 0::2] = E2M1[w & 0xF]
    out[:, 1::2] = E2M1[w >> 4]
    out *= np.repeat(np.exp2(s.astype(np.float32) - 127.0), 32, axis=1)
    return out

def _self_test():
    # Hand-computed fixture. One row, K=32 (= 16 packed bytes, 1 scale byte).
    # The first 8 bytes walk all 16 E2M1 codes exactly once, low nibble first:
    #   0x21 -> lo 0x1=0.5, hi 0x2=1.0      0x43 -> lo 0x3=1.5, hi 0x4=2.0
    #   0x65 -> lo 0x5=3.0, hi 0x6=4.0      0x87 -> lo 0x7=6.0, hi 0x8=-0.0
    #   0xA9 -> lo 0x9=-0.5, hi 0xA=-1.0    0xCB -> lo 0xB=-1.5, hi 0xC=-2.0
    #   0xED -> lo 0xD=-3.0, hi 0xE=-4.0    0x0F -> lo 0xF=-6.0, hi 0x0=0.0
    # Scale byte 128 = 2^(128-127) = 2.0, so every decoded value doubles.
    #
    # This fixture is NOT self-referential: the expected values are written out by
    # hand, so it fails if the nibble order flips (adjacent pairs swap), if any E2M1
    # table entry is wrong (that entry's column is wrong), or if the E8M0 exponent
    # direction is inverted (every value scales by 0.5 instead of 2.0). The earlier
    # version divided the output by a scale recomputed with the same formula under
    # test and only checked membership in the E2M1 set — it could not fail on
    # anything but a shape bug. (Amended 2026-07-31.)
    w = np.array([[0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F] + [0x00] * 8],
                 dtype=np.uint8)
    s = np.array([[128]], dtype=np.uint8)
    expected = np.array([[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0,
                          -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0, 0.0] + [0.0] * 16],
                        dtype=np.float32)
    d = dequant(w, s)
    assert d.shape == (1, 32), d.shape
    assert np.array_equal(d, expected), f"\ngot      {d}\nexpected {expected}"

    # Shape contract over a larger random case.
    rng = np.random.default_rng(0)
    N, K = 8, 64
    wr = rng.integers(0, 256, size=(N, K // 2), dtype=np.uint8)
    sr = rng.integers(120, 135, size=(N, K // 32), dtype=np.uint8)
    assert dequant(wr, sr).shape == (N, K)
    print("glm_mxfp4_numpy self-test PASS")

if __name__ == "__main__":
    _self_test()
