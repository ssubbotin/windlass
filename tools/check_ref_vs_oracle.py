#!/usr/bin/env python3
"""check_ref_vs_oracle.py -- Step 2 of the GLM-5.2 port: cross-check
ref_glm_chain.py against the single-layer transformers oracle.

This is the load-bearing evidence that the numpy chain reference is trustworthy.
The full-chain comparison (Task 10) rests on it, because 753B cannot be loaded
under transformers and the reference is the only ground truth there is.

Both sides are fed the SAME injected hidden state -- the oracle's own
`layer{L}_input_hidden.npy` -- and the check runs at BOTH:
  * layer 0, exercising the dense MLP branch (layer < first_k_dense_replace), and
  * layer 3, exercising the MoE branch.
A layer-3-only pass leaves layers 0-2 of every chain run unverified.

WHY THIS CHECKS SUBSTEPS AND NOT JUST THE LAYER OUTPUT
------------------------------------------------------
The layer output is `h_in + attn + mlp` and is dominated by `h_in`, so a relative
error normalized by max|output| divides every MoE-side mistake by the residual.
Measured on this fixture, an output-only comparison at tol 2e-2 passes a bf16
`e_score_correction_bias` (1.00e-2), `routed_scaling_factor = 1.0` (1.05e-2) and a
flipped MXFP4 nibble order (1.94e-2) -- i.e. it would certify a badly wrong
reference. Run this script with --negative-controls to reproduce that.

Usage (must run under the venv only if you also want transformers; plain numpy is
enough for this script):
  python3 check_ref_vs_oracle.py \
      --model ./glm52-mxfp4 --packed ./packed_experts \
      --oracle-layer0 glm-oracle-layer0 --oracle-layer3 glm-oracle
  ... --negative-controls      # additionally re-demonstrate that it can fail

Exits 0 if every assertion passes, 1 otherwise.
"""
import argparse
import os
import sys

import numpy as np

import ref_glm_chain as R

# --- Tolerances. Do not raise these without re-deriving the band below. -------

HIDDEN_TOL = 2e-2       # on the layer output hidden state, as originally specified

# SUBSTEP_TOL usable band, measured on this fixture (bf16 oracle vs f32 reference):
#
#   lower bound  5.65e-03  -- NEG "layer 0 LORA_NORM_EPS = rms_eps (1e-5)", a real
#                             defect that this check does NOT catch. Anything at or
#                             below this is indistinguishable from oracle noise.
#   upper bound  1.09e-02  -- NEG "layer 3 RoPE halves instead of interleaved", the
#                             tightest defect this check DOES catch.
#
# So the real headroom is 1.09e-02 / 1.00e-02 = 1.09x, NOT the ~2.4x that a naive
# comparison against the worst passing baseline (4.13e-03, layer0 attn_out) would
# suggest. Raising SUBSTEP_TOL loses the layer-3 RoPE-convention catch FIRST, and
# loses it almost immediately. Lowering it below ~5.7e-3 starts failing baselines.
# If an oracle regeneration moves these numbers, re-run --negative-controls and
# re-derive the band; do not just widen the tolerance.
SUBSTEP_TOL = 1e-2

# On topk_w, relative to the largest oracle weight. The "weight gathered from the
# BIASED score" defect lands at 5.77e-03 against a 3.17e-03 threshold, so this
# assertion -- not topk_idx -- is what catches it.
TOPK_W_REL = 1e-2

SUBSTEPS = ["post_input_norm", "attn_out", "post_attn_hidden", "post_post_norm",
            "router_logits", "shared_out", "moe_out"]


def _last(a):
    """Oracle tensors are [1, seq, hidden] or [seq, ...]; take the last position."""
    return a[0, -1] if a.ndim == 3 else a[-1]


def _rel(a, b):
    return float(np.abs(a - b).max() / (np.abs(a).max() + 1e-4))


def run_layer(store, cfg, layer, oracle_dir, packed):
    """Drive the reference over the oracle's own input, one position at a time.

    Returns (output_hidden, taps) at the LAST position -- the oracle runs a
    length-8 sequence, so all 8 KV rows must be accumulated before comparing.
    """
    inj_path = os.path.join(oracle_dir, f"layer{layer}_input_hidden.npy")
    inj, cos, sin, W = R.inject_setup(store, cfg, inj_path, layer)
    cache, taps, h = {"k": [], "v": []}, {}, None
    for pos in range(inj.shape[0]):
        taps = {}
        h, cache = R.layer_forward(inj[pos], W, cfg, pos, cos, sin, cache,
                                   packed, layer, taps)
    return h, taps


def compare(tag, layer, oracle_dir, h, taps, verbose=True):
    """Assert the reference against every tensor the oracle dumped."""
    results = []

    def emit(name, value, ok, extra=""):
        results.append((name, value, ok))
        if verbose:
            print(f"  {name:22s} {value}  {'OK' if ok else 'FAIL'}{extra}")

    a = np.load(os.path.join(oracle_dir, f"layer{layer}_output_hidden.npy"))[0, -1]
    rel = _rel(a, h)
    if verbose:
        print(f"{tag}")
        print(f"  {tag} hidden max_rel: {rel:.4e} {'OK' if rel < HIDDEN_TOL else 'FAIL'}")
    results.append(("output_hidden", f"max_rel {rel:.4e}", rel < HIDDEN_TOL))

    for name in SUBSTEPS:
        p = os.path.join(oracle_dir, f"layer{layer}_{name}.npy")
        if not os.path.exists(p):
            continue                    # dense layers have no router / shared expert
        rel = _rel(_last(np.load(p)), taps[name])
        emit(name, f"max_rel {rel:.4e}", rel < SUBSTEP_TOL)

    p = os.path.join(oracle_dir, f"layer{layer}_topk_idx.npy")
    if os.path.exists(p):
        oi = np.load(p)[-1]
        ow = np.load(os.path.join(oracle_dir, f"layer{layer}_topk_w.npy"))[-1]
        ri, rw = taps["topk_idx"], taps["topk_w"]
        same = set(oi.tolist()) == set(ri.tolist())
        emit("topk_idx", f"exact set match: {same}", same,
             f"\n      oracle {sorted(oi.tolist())}\n      ref    {sorted(ri.tolist())}"
             if verbose else "")
        if same:
            om = dict(zip(oi.tolist(), ow.tolist()))
            rm = dict(zip(ri.tolist(), rw.tolist()))
            d = max(abs(om[k] - rm[k]) for k in om)
            thr = TOPK_W_REL * max(abs(v) for v in om.values())
            emit("topk_w", f"max_abs {d:.4e} thr {thr:.4e} (sum={sum(ow):.4f})", d < thr)
        else:
            # Selection already diverged; a weight comparison would be meaningless.
            results.append(("topk_w", "skipped (topk_idx diverged)", False))

    return all(ok for _, _, ok in results), results


# ---------------------------------------------------------------------------
# Negative controls -- re-demonstrate that the check above CAN fail.
# ---------------------------------------------------------------------------
# Each entry perturbs exactly one documented fact and is reverted afterwards.
# The point is not the perturbations themselves but the standing proof that a
# green run means something. A check nobody has ever seen fail is not evidence.

def _bf16(x):
    """Round-to-nearest-even-ish bf16 cast, as a torch .to(bfloat16) would do."""
    return ((x.view(np.uint32) + 0x8000) & 0xFFFF0000).view(np.float32)


def negative_controls(store, cfg, packed, oracle):
    import contextlib

    @contextlib.contextmanager
    def patch(obj, attr, value):
        old = getattr(obj, attr)
        setattr(obj, attr, value)
        try:
            yield
        finally:
            setattr(obj, attr, old)

    @contextlib.contextmanager
    def cfgkey(key, value):
        old = cfg[key]
        cfg[key] = value
        try:
            yield
        finally:
            cfg[key] = old

    @contextlib.contextmanager
    def bias_bf16():
        """Cast e_score_correction_bias to bf16 -- the Task 4 defect."""
        orig = R.load_layer_weights

        def patched(st, c, layer):
            W = orig(st, c, layer)
            if "gate.e_score_correction_bias" in W:
                W["gate.e_score_correction_bias"] = _bf16(W["gate.e_score_correction_bias"])
            return W
        with patch(R, "load_layer_weights", patched):
            yield

    @contextlib.contextmanager
    def rmsnorm_delta():
        """The (1 + w) delta parameterization GLM-5.2 does NOT use."""
        orig = R.rms_norm
        with patch(R, "rms_norm", lambda x, w, eps: orig(x, 1.0 + w.astype(np.float32), eps)):
            yield

    @contextlib.contextmanager
    def rope_halves():
        """Halves layout instead of the interleaved adjacent-pair layout."""
        def halves(x, cos, sin):
            hf = x.shape[-1] // 2
            a, c = x[..., :hf], x[..., hf:]
            return np.concatenate([a * cos - c * sin, c * cos + a * sin], axis=-1)
        with patch(R, "rope_interleave", halves):
            yield

    @contextlib.contextmanager
    def weight_from_biased():
        """Gather the top-k weight from the BIASED score (M:505 says unbiased)."""
        orig = R.route

        def patched(x, gw, bias, top_k, rs, ntp):
            lg, idx, _ = orig(x, gw, bias, top_k, rs, ntp)
            sc = 1.0 / (1.0 + np.exp(-lg)) + bias.astype(np.float32)
            w = sc[idx].copy()
            w /= (w.sum() + 1e-20)
            return lg, idx, w * rs
        with patch(R, "route", patched):
            yield

    @contextlib.contextmanager
    def nibble_flip():
        """hi-even instead of the verified lo-even MXFP4 nibble order."""
        orig = R.dequant

        def flipped(w, s):
            d = orig(w, s)
            out = np.empty_like(d)
            out[:, 0::2] = d[:, 1::2]
            out[:, 1::2] = d[:, 0::2]
            return out
        with patch(R, "dequant", flipped):
            yield

    @contextlib.contextmanager
    def lora_eps_wrong():
        with patch(R, "LORA_NORM_EPS", 1e-5):
            yield

    controls = [
        ("baseline", 0, None),
        ("baseline", 3, None),
        ("e_score_correction_bias -> bf16", 3, bias_bf16),
        ("rms_norm in (1+w) delta form", 0, rmsnorm_delta),
        ("rms_norm in (1+w) delta form", 3, rmsnorm_delta),
        ("RoPE halves instead of interleaved", 0, rope_halves),
        ("RoPE halves instead of interleaved", 3, rope_halves),
        ("topk weight from BIASED score", 3, weight_from_biased),
        ("routed_scaling 1.0 instead of 2.5", 3, lambda: cfgkey("routed_scaling", 1.0)),
        ("MXFP4 nibble order flipped", 3, nibble_flip),
        ("LORA_NORM_EPS = rms_eps (1e-5)", 0, lora_eps_wrong),
        ("LORA_NORM_EPS = rms_eps (1e-5)", 3, lora_eps_wrong),
    ]

    print("\n--- negative controls (a defect MUST show as 'caught') ---")
    print(f"{'perturbation':40s} {'L':>2s}  {'out-only':>9s}  {'full check':>10s}  worst substep")
    missed = []
    for label, layer, ctx in controls:
        mgr = ctx() if ctx is not None else None
        if mgr is not None:
            mgr.__enter__()
        try:
            h, taps = run_layer(store, cfg, layer, oracle[layer], packed)
            ok, results = compare(label, layer, oracle[layer], h, taps, verbose=False)
        finally:
            if mgr is not None:
                mgr.__exit__(None, None, None)
        out_rel = float(results[0][1].split()[-1])
        subs = [(n, v) for n, v, o in results[1:] if v.startswith("max_rel")]
        worst_n, worst_v = max(subs, key=lambda t: float(t[1].split()[-1])) if subs else ("-", "-")
        base = label == "baseline"
        out_v = "pass" if out_rel < HIDDEN_TOL else "CAUGHT"
        full_v = "pass" if ok else "caught"
        verdict_bad = (base and not ok) or (not base and ok)
        if verdict_bad:
            missed.append((label, layer))
        print(f"{label:40s} {layer:2d}  {out_v:>9s}  {full_v:>10s}  "
              f"{worst_n} {worst_v.split()[-1]}{'   <-- MISSED' if verdict_bad else ''}")
    print("\nKnown-undetectable (documented, source-verified instead): "
          "LORA_NORM_EPS. Any other MISSED line is new and must be investigated.")
    return missed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="GLM-5.2 checkpoint directory")
    ap.add_argument("--packed", required=True, help="packed_experts_glm directory")
    ap.add_argument("--oracle-layer0", default="glm-oracle-layer0",
                    help="dump_glm_oracle.py --layer 0 output directory (dense path)")
    ap.add_argument("--oracle-layer3", default="glm-oracle",
                    help="dump_glm_oracle.py --layer 3 output directory (MoE path)")
    ap.add_argument("--negative-controls", action="store_true",
                    help="also perturb one documented fact at a time and show the "
                         "check failing -- proof that a green run means something")
    args = ap.parse_args()

    oracle = {0: args.oracle_layer0, 3: args.oracle_layer3}
    for layer, d in oracle.items():
        f = os.path.join(d, f"layer{layer}_input_hidden.npy")
        if not os.path.exists(f):
            ap.error(f"missing oracle fixture {f}; regenerate with "
                     f"dump_glm_oracle.py --layer {layer} --out {d}/")

    store = R.ShardStore(args.model)
    cfg = R.load_config(args.model)

    ok = True
    for tag, layer in (("layer0 (dense)", 0), ("layer3 (MoE)", 3)):
        h, taps = run_layer(store, cfg, layer, oracle[layer], args.packed)
        good, _ = compare(tag, layer, oracle[layer], h, taps)
        ok &= good

    if args.negative_controls:
        missed = negative_controls(store, cfg, args.packed, oracle)
        ok &= not [m for m in missed if m[0] != "LORA_NORM_EPS = rms_eps (1e-5)"]

    print(f"\npeak RSS: {R.peak_rss_gb():.2f} GB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
