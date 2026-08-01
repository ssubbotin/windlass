#!/usr/bin/env python3
"""ref_glm_indexer.py — numpy reference for the GLM-5.2 DSA indexer (Task 3).

Generates random inputs at the REAL shapes (q_lora 2048, hidden 6144, 32 heads
x 128 dims) plus the reference outputs of every stage, and writes them as .npy
for src/test_glm_indexer.cu to compare against. No checkpoint and no
transformers involved: this is an independent reimplementation of the spec
extracted in Task 1 (docs/plans/2026-08-01-dsa-indexer-and-serve.md), written
in a different language from the CUDA under test, so a shared implementation
slip is at least not free.

It also writes four DEFECTIVE reference variants, one per trap Task 1 recorded.
The defect is injected on the REFERENCE side rather than into the kernels: the
comparison metric is a difference between the two implementations, so the
separation is the same either way, and no defect branch ends up shipping in
glm_kernels.cuh. Baseline agreement proves the CUDA side is trap-free; the
variants prove the comparison would have noticed if it were not.

  rmsnorm        k_norm as an RMSNorm (no mean subtraction, no bias) instead of
                 a LayerNorm  -- and rms_norm_bf16() is right next to the kernel.
  eps            LayerNorm eps 1e-5 (config.rms_norm_eps) instead of the
                 hardcoded 1e-6.
  rope_trailing  RoPE on the TRAILING 64 of the 128 dims, which is what the MAIN
                 attention path does to ITS 256-dim heads.
  relu_after     relu applied after the head combination instead of before it.

Usage:
  python3 tools/ref_glm_indexer.py --out /tmp/idxfix --scale 1.0
  python3 tools/ref_glm_indexer.py --out /tmp/idxfix_eps --scale 0.01

`--scale` multiplies the hidden state only. It exists because the eps defect is
detectable only in proportion to eps/var(k): at scale 1.0 the post-wk key
variance is O(1) and a 1e-6-vs-1e-5 eps moves the result by parts in 1e6 --
i.e. into the fp32 noise floor. Both regimes are reported rather than only the
flattering one.
"""
import argparse
import os

import numpy as np

H, D, ROT = 32, 128, 64          # index_n_heads, index_head_dim, qk_rope_head_dim
HIDDEN, Q_LORA = 6144, 2048
ROPE_THETA = 8e6
LN_EPS = 1e-6                    # M:199 -- a literal, NOT config.rms_norm_eps
RMS_EPS = 1e-5                   # config.rms_norm_eps, the wrong one


def to_bf16(x):
    """float32 -> bf16 bit pattern, round-to-nearest-even (torch's cast)."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rnd = ((u >> 16) & 1).astype(np.uint32) + np.uint32(0x7FFF)
    return ((u + rnd) >> 16).astype(np.uint16)


def from_bf16(u):
    return (np.ascontiguousarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16(x):
    """Round a float32 array through bf16 and back."""
    return from_bf16(to_bf16(x))


def rope(v, cos, sin):
    """Interleaved RoPE, halves-out (M:161-168). v: [..., rot]."""
    a, b = v[..., 0::2], v[..., 1::2]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(np.float32)


def layer_norm(x, w, b, eps):
    mu = x.mean(dtype=np.float32)
    var = ((x - mu) ** 2).mean(dtype=np.float32)
    return ((x - mu) / np.sqrt(var + eps) * w + b).astype(np.float32)


def rms_norm(x, w, eps):
    """The trap: no mean subtraction, no bias."""
    r = np.sqrt((x * x).mean(dtype=np.float32) + eps)
    return (x / r * w).astype(np.float32)


def indexer(defect, wq_b, wk, kn_w, kn_b, wproj, q_resid, x_all, cos, sin, topk):
    """A full T-position indexer pass. `defect` is None or one of the four names.

    Every one of the T keys goes through k_norm and RoPE, which is what makes
    the k_norm controls meaningful: a fixture that computes only the LAST key
    and takes the other T-1 from a random cache moves exactly one of T index
    scores when k_norm is wrong, and the RMSNorm control then shows a full
    2048/2048 top-k overlap. Measured, not assumed -- that was this file's
    first version.
    """
    T = x_all.shape[0]
    pos = T - 1
    x = x_all[pos]
    # ---- query side: wq_b consumes q_resid (2048), NOT the hidden state ----
    q = (wq_b @ q_resid).astype(np.float32).reshape(H, D)
    # ---- key side: wk consumes the raw hidden state (6144), every position ----
    k_raw = (x_all @ wk.T).astype(np.float32)                      # [T, D]
    k = np.empty_like(k_raw)
    for t in range(T):
        if defect == "rmsnorm":
            k[t] = rms_norm(k_raw[t], kn_w, LN_EPS)
        elif defect == "eps":
            k[t] = layer_norm(k_raw[t], kn_w, kn_b, RMS_EPS)
        else:
            k[t] = layer_norm(k_raw[t], kn_w, kn_b, LN_EPS)

    if defect == "rope_trailing":                 # the MAIN path's slice, not the indexer's
        q[:, D - ROT:] = rope(q[:, D - ROT:], cos[pos], sin[pos])
        k[:, D - ROT:] = rope(k[:, D - ROT:], cos, sin)
    else:
        q[:, :ROT] = rope(q[:, :ROT], cos[pos], sin[pos])
        k[:, :ROT] = rope(k[:, :ROT], cos, sin)

    cache = to_bf16(k)                                             # [T, D] bf16
    kf = from_bf16(cache).reshape(-1, D)

    qk = ((q @ kf.T) * np.float32(D ** -0.5)).astype(np.float32)    # [H, T]
    w = ((wproj @ x) * np.float32(H ** -0.5)).astype(np.float32)    # [H]
    if defect == "relu_after":
        scores = qk
        index = np.maximum((w[:, None] * scores).sum(axis=0, dtype=np.float32), 0.0)
    else:
        scores = np.maximum(qk, 0.0)
        index = (w[:, None] * scores).sum(axis=0, dtype=np.float32)
    index = index.astype(np.float32)

    kk = min(topk, index.shape[0])
    sel = np.argsort(-index, kind="stable")[:kk].astype(np.int32)
    return dict(q=q.reshape(-1), k=k, k_bf16=cache, w=w, scores=scores.astype(np.float32),
                index=index, topk=sel, cache=cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=1024, help="T, total keys including this one")
    ap.add_argument("--topk", type=int, default=512,
                    help="index_topk; below --seq so the top-k actually selects")
    ap.add_argument("--scale", type=float, default=1.0, help="hidden-state magnitude")
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    T, pos = a.seq, a.seq - 1

    # Weights at plausible magnitudes: projections ~N(0, 0.02) as in a trained
    # bf16 checkpoint, LayerNorm weight near 1, bias small.
    wq_b_b = to_bf16(rng.normal(0, 0.02, (H * D, Q_LORA)).astype(np.float32))
    wk_b = to_bf16(rng.normal(0, 0.02, (D, HIDDEN)).astype(np.float32))
    kn_w_b = to_bf16(rng.normal(1.0, 0.1, D).astype(np.float32))
    kn_b_b = to_bf16(rng.normal(0.0, 0.05, D).astype(np.float32))
    wproj = rng.normal(0, 0.02, (H, HIDDEN)).astype(np.float32)     # fp32 on purpose (M:643)

    q_resid = rng.normal(0, 1.0, Q_LORA).astype(np.float32)
    # One hidden state per position: every key goes through wk / k_norm / RoPE,
    # so the norm and RoPE controls move all T index scores rather than one.
    x_all = (rng.normal(0, 1.0, (T, HIDDEN)) * a.scale).astype(np.float32)
    x = x_all[pos]

    # RoPE table per position, same construction as glm_layer_runner.cuh::rope_pos.
    inv = (1.0 / ROPE_THETA ** (np.arange(0, ROT, 2, dtype=np.float64) / ROT))
    ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]
    cos = np.cos(ang).astype(np.float32)                            # [T, ROT/2]
    sin = np.sin(ang).astype(np.float32)

    args = (from_bf16(wq_b_b).reshape(H * D, Q_LORA), from_bf16(wk_b).reshape(D, HIDDEN),
            from_bf16(kn_w_b), from_bf16(kn_b_b), wproj, q_resid, x_all,
            cos, sin, a.topk)
    ref = indexer(None, *args)

    def save(name, arr):
        np.save(os.path.join(a.out, name + ".npy"), np.ascontiguousarray(arr))

    save("wq_b", wq_b_b); save("wk", wk_b)
    save("k_norm_w", kn_w_b); save("k_norm_b", kn_b_b)
    save("weights_proj", wproj)
    save("q_resid", q_resid); save("x_all", x_all)
    save("cos", cos); save("sin", sin)
    save("ref_q", ref["q"]); save("ref_k", ref["k"]); save("ref_k_bf16", ref["k_bf16"])
    save("ref_w", ref["w"]); save("ref_index", ref["index"]); save("ref_topk", ref["topk"])

    lines = [f"H {H}", f"D {D}", f"rot {ROT}", f"hidden {HIDDEN}", f"q_lora {Q_LORA}",
             f"T {T}", f"pos {pos}", f"topk {a.topk}", f"scale {a.scale}"]
    # sep_<defect> is the reference's OWN prediction of how far this defect moves
    # the index scores on this fixture. test_glm_indexer.cu uses it to decide
    # whether the fixture is even capable of exercising the control: the eps trap
    # moves the result in proportion to eps/var(wk(x)), so at realistic
    # magnitudes it is below the fp32 noise floor and no test on these shapes can
    # catch it. Saying so is the honest outcome; silently gating on it is not.
    for name in ("rmsnorm", "eps", "rope_trailing", "relu_after"):
        d = indexer(name, *args)
        save("ref_index_" + name, d["index"])
        save("ref_topk_" + name, d["topk"])
        over = len(set(d["topk"].tolist()) & set(ref["topk"].tolist()))
        sep = _rel(ref["index"], d["index"])
        lines.append(f"sep_{name} {sep:.6e}")
        print(f"  defect {name:14s} index rel={sep:.4e} "
              f"topk overlap {over}/{len(ref['topk'])}")
    open(os.path.join(a.out, "meta.txt"), "w").write("\n".join(lines) + "\n")

    kraw = (from_bf16(wk_b).reshape(D, HIDDEN) @ x).astype(np.float32)  # last position
    print(f"wrote {a.out}: T={T} topk={a.topk} scale={a.scale}")
    print(f"  var(wk(x)) = {kraw.var():.6e}   (eps {LN_EPS:g} vs {RMS_EPS:g} matters in "
          f"proportion to eps/var)")
    print(f"  index score: min {ref['index'].min():.4e} max {ref['index'].max():.4e} "
          f"exact zeros {int((ref['index'] == 0).sum())}/{T}")
    print(f"  relu zeros in scores: {int((ref['scores'] == 0).sum())}/{ref['scores'].size}")


def _rel(ref, got):
    return float(np.abs(ref - got).max() / (np.abs(ref).max() + 1e-300))


if __name__ == "__main__":
    main()
