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

AND WHY THAT IS NOT ENOUGH ON ITS OWN
------------------------------------
Both fixtures above are 8 tokens long, i.e. far below `index_topk` (2048), where
the DSA indexer's mask is uniformly False and its arithmetic cannot move a single
output bit. `--oracle-long` adds Task 5's 4096-token fixtures, where it can: layer
2 (indexer owner) and layer 3 (a "shared" consumer of layer 2's selection). Only
that pass constrains `k_norm`, the two scale factors, the RoPE slice, the ReLU
placement and the head combination. See the block above `run_long`.

Usage (must run under the venv only if you also want transformers; plain numpy is
enough for this script):
  python3 check_ref_vs_oracle.py \
      --model ./glm52-mxfp4 --packed ./packed_experts \
      --oracle-layer0 glm-oracle-layer0 --oracle-layer3 glm-oracle
  ... --oracle-long /home/user1/glm-oracle-t5   # + the indexer, at seq 4096
  ... --negative-controls      # additionally re-demonstrate that it can fail

Exits 0 if every assertion passes, 1 otherwise.
"""
import argparse
import os
import sys
import time

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
    assert inj.shape[0] <= cfg["index_topk"], "use run_long past index_topk"
    cache, taps, h = {"k": [], "v": [], "ik": []}, {}, None
    for pos in range(inj.shape[0]):
        taps = {}
        # A "shared" layer run in isolation has no leader in this process. Below
        # index_topk the selection cannot matter -- topk(min(index_topk,T))
        # returns every index and the mask is uniformly False -- so the arange
        # stand-in dump_glm_oracle.py passes as prev_topk_indices (M:335-337) is
        # the same mask the real leader would have produced, bit for bit. Above
        # index_topk it would NOT be, which is why run_long uses a real leader.
        share = R.IndexShare(np.arange(pos + 1, dtype=np.int32))
        h, cache = R.layer_forward(inj[pos], W, cfg, pos, cos, sin, cache,
                                   packed, layer, taps, share)
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


def check_index_share(store, cfg, packed, oracle_dir):
    """IndexShare's structural contract, which no tolerance can express.

    The numeric checks above run one layer at a time and so never exercise
    propagation. These three do, and they are the ones that would let a chain run
    silently degrade into a dense one:

      * a "shared" layer with nothing published must FAIL, not fall back;
      * a "shared" layer must pass its input through UNCHANGED (M:417-419), or
        propagation stops at the first consumer instead of crossing the group;
      * the ownership map must match the checkpoint, so members consume the right
        leader.
    """
    results = []

    def emit(name, value, ok):
        results.append((name, value, ok))
        print(f"  {name:34s} {value}  {'OK' if ok else 'FAIL'}")

    types = cfg["indexer_types"]
    owners = [i for i, t in enumerate(types) if t == "full"]
    emit("indexer owners", f"{len(owners)} at {owners[:4]}...{owners[-2:]}",
         len(owners) == 21 and owners[:3] == [0, 1, 2] and owners[3] == 6
         and owners[-1] == 74 and all(o % 4 == 2 for o in owners[2:]))
    # Every member's leader is the nearest preceding "full" layer; groups are
    # {2,3,4,5}, {6,7,8,9}, ... Checked by construction of the layer loop, but a
    # gap would mean some layer consumes a leader four layers too early.
    leaders = []
    cur = None
    for i, t in enumerate(types):
        cur = i if t == "full" else cur
        leaders.append(cur)
    emit("every layer has a leader", f"first None at {leaders.index(None) if None in leaders else 'never'}",
         None not in leaders)

    inj_path = os.path.join(oracle_dir, "layer3_input_hidden.npy")
    inj, cos, sin, W = R.inject_setup(store, cfg, inj_path, 3)

    try:
        R.layer_forward(inj[0], W, cfg, 0, cos, sin, {"k": [], "v": [], "ik": []},
                        packed, 3, None, R.IndexShare(None))
        emit("member with no indices raises", "returned normally", False)
    except RuntimeError as e:
        emit("member with no indices raises", f"RuntimeError: {str(e)[:40]}...", True)

    # Index 0 is in range at pos 0; the rest are not, which also exercises
    # index_drop_mask's "a non-causal index unmasks nothing" rule. At least one
    # in-range index is required -- a fully dropped row is a softmax over all
    # -inf, i.e. NaN. That cannot arise from a real selection (the top-k always
    # contains causally valid keys with finite scores) but it can arise from a
    # hand-written test, and it did.
    published = np.array([0, 7, 3, 0, 5], np.int32)
    share = R.IndexShare(published)
    R.layer_forward(inj[0], W, cfg, 0, cos, sin, {"k": [], "v": [], "ik": []},
                    packed, 3, None, share)
    emit("member passes indices through", f"{share.indices.tolist()}",
         share.indices is published)
    return all(ok for _, _, ok in results)


# ---------------------------------------------------------------------------
# Long context (seq > index_topk): the indexer is no longer a no-op
# ---------------------------------------------------------------------------
# Below index_topk the DSA mask is uniformly False and the reference's indexer
# arithmetic is unobservable -- Task 1 proved that, and Task 4 turned it into a
# bit-identity gate. Everything in this section exists because that gate says
# nothing about the numbers. The fixtures are Task 5's: dump_glm_oracle.py at
# --seq 4096 for layer 2 (an indexer OWNER, dense MLP) and layer 3 (a "shared"
# consumer, MoE), the latter dumped with --prev-topk from the former.
#
# Two passes per layer, for the same reason test_glm_layer runs two:
#   native  the reference picks its own keys. Includes the selection difference.
#   arith   the ORACLE's selection is forced in. Isolates the arithmetic from the
#           selection, so "attn_out is off" stops being unfalsifiable.
# Never gate on the index SET: Task 5 measured 2043 of 2048 keys shared between
# two correct implementations, the 5 stragglers all within 2e-4 of the selection
# boundary. That drift is legitimate and is larger here (fp32 vs bf16 q and k).

# Gates. Derived here from measurement on THIS comparison; Task 5's ulp gates are
# deliberately not imported, because numpy-fp32-vs-transformers-bf16 is a
# different comparison from CUDA-fp32-vs-transformers-bf16 (and this metric is
# relative-to-peak, not ulp).
#
# LONG_TOL_ARITH -- oracle's selection forced in, so only arithmetic is compared.
#   lower bound  3.25e-03  worst baseline substep (L2 post_input_norm; L3 2.29e-03).
#                          That substep is the injected input's own bf16 dtype and
#                          everything downstream of it is SMALLER (arith attn_out
#                          1.31e-03 / 7.86e-04) -- no accumulation, no residual.
#   upper bound  1.28e-02  topk_swap N=1, the tightest defect this pass catches.
#   Headroom 3.9x; the gate sits 3.1x above the floor and 1.28x below the catch.
#   Raising it past ~1.28e-02 loses the one-key selection catch FIRST.
LONG_TOL_ARITH = 1.0e-2
# LONG_TOL_NATIVE -- the reference picks its own keys, so the two boundary keys it
#   disagrees with transformers about are inside the baseline (worst 1.38e-02, L3
#   attn_out; L2 1.22e-02). Nothing constrains this from above: every selection
#   defect is measured on the pinned `arith` pass, where the gate is 4x tighter,
#   and every arithmetic defect separates at >=11x. Placed at 2.2x the floor,
#   the same 2x rule Task 5 used.
LONG_TOL_NATIVE = 3.0e-2
# LONG_TOL_SCORE -- the index-score row itself. Measured floor 4.00e-03, made of
#   two bf16 terms of similar size (transformers keeps the indexer's q and its key
#   cache in bf16, this reference keeps both in fp32). 5.0x margin.
LONG_TOL_SCORE = 2.0e-2

LONG_SUBSTEPS = ["post_input_norm", "attn_out", "post_attn_hidden",
                 "post_post_norm", "router_logits", "shared_out", "moe_out"]


def long_setup(store, cfg, oracle_dir, layer):
    """Load the 4096-token fixture and prefill every position but the last.

    Only the LAST position is compared, but its attention reads all 4096 keys, so
    the whole prefix has to exist. Those K/V rows depend only on the injected
    hidden states, never on attention output, so they are computed in a handful
    of GEMMs (R.prefill_kv) instead of 4095 discarded full-layer forwards.
    """
    inj = np.load(os.path.join(oracle_dir, f"layer{layer}_input_hidden.npy"))
    inj = (inj[0] if inj.ndim == 3 else inj).astype(np.float32)
    T = inj.shape[0]
    assert T > cfg["index_topk"], f"seq {T} <= index_topk; nothing new is exercised"
    cos, sin = R.rope_tables(T + 8, cfg["rot"], cfg["rope_theta"])
    W = R.load_layer_weights(store, cfg, layer)
    t0 = time.time()
    pre = R.prefill_kv(inj[:T - 1], W, cfg, cos, sin, layer)
    print(f"  layer {layer}: prefilled {T - 1} positions in {time.time() - t0:.1f}s")
    return inj, cos, sin, W, pre


def long_pass(inj, cos, sin, W, pre, cfg, packed, layer, share_idx=None, force=None):
    """Run the last position on a private copy of the prefilled cache."""
    cache = {k: list(v) for k, v in pre.items()}
    taps, share = {}, R.IndexShare(share_idx)
    h, _ = R.layer_forward(inj[-1], W, cfg, inj.shape[0] - 1, cos, sin, cache,
                           packed, layer, taps, share, force)
    return h, taps, share.indices


def prefill_equiv(inj, cos, sin, W, cfg, packed, layer, n=16):
    """prefill_kv must equal the per-position path it replaces.

    Same expressions, different matmul shapes, so this is a numerical claim and
    not a structural one -- measure it rather than asserting it. Reported as an
    absolute difference against the K/V magnitudes involved.
    """
    a = R.prefill_kv(inj[:n], W, cfg, cos, sin, layer)
    b = {"k": [], "v": [], "ik": []}
    for pos in range(n):
        R.layer_forward(inj[pos], W, cfg, pos, cos, sin, b, packed, layer, None,
                        R.IndexShare(np.arange(pos + 1, dtype=np.int32)))
    out = {}
    for key in ("k", "v", "ik"):
        if not a[key]:
            continue
        out[key] = _rel(np.stack(b[key]), np.stack(a[key]))
    return out


def _score_report(taps, oracle_dir, layer, sel_oracle):
    """Index-score gate plus reported-only q / w / set-overlap diagnostics.

    Task 5's `headscale` control is the reason this gate exists: multiplying
    every head gate by a positive constant multiplies the whole index score,
    leaves the ranking and therefore the selected set untouched, and produces a
    BIT-IDENTICAL attention output. An output-only check cannot see it -- that is
    a property of top-k, not of this fixture -- so a whole class of defect (any
    monotone rescaling of the score: a wrong 128**-0.5, a wrong 32**-0.5, a
    scaled weights_proj) is invisible without comparing the score vector itself.
    """
    rows = []
    for name, tap_name in (("indexer_q_last", "indexer_q"),
                           ("indexer_w_last", "indexer_w"),
                           ("indexer_index_score_last", "indexer_index_score")):
        p = os.path.join(oracle_dir, f"layer{layer}_{name}.npy")
        if not os.path.exists(p) or tap_name not in taps:
            continue
        rows.append((tap_name, _rel(np.load(p), taps[tap_name])))
    ov = None
    if "indexer_topk" in taps and sel_oracle is not None:
        ov = len(set(taps["indexer_topk"].tolist()) & set(sel_oracle.tolist()))
    return rows, ov


def run_long(store, cfg, packed, oracle_dir, verbose=True, prep=None):
    """Layer 2 (owner) then layer 3 (consumer) at 4096. Returns (ok, results)."""
    results = []

    def emit(name, value, ok):
        results.append((name, value, ok))
        if verbose:
            print(f"  {name:34s} {value}  {'OK' if ok else 'FAIL'}")

    sel_o2 = np.load(os.path.join(oracle_dir, "layer2_topk_indices.npy"))[0, -1]
    sel_o3 = np.load(os.path.join(oracle_dir, "layer3_topk_indices.npy"))[0, -1]
    # Layer 3 is "shared": its selection IS layer 2's, passed through unchanged
    # (M:417-419). If that ever stops holding, the propagation model is wrong.
    assert np.array_equal(sel_o2, sel_o3), \
        "oracle layer3 topk_indices differ from layer2's -- propagation model is wrong"

    prep = prep if prep is not None else {}
    leader_sel = None
    for layer, sel_oracle in ((2, sel_o2), (3, sel_o3)):
        if layer not in prep:
            prep[layer] = long_setup(store, cfg, oracle_dir, layer)
        inj, cos, sin, W, pre = prep[layer]
        if layer == 2 and verbose:
            eq = prefill_equiv(inj, cos, sin, W, cfg, packed, layer)
            print("  prefill_kv vs per-position path (16 pos): "
                  + "  ".join(f"{k} {v:.3e}" for k, v in eq.items()))

        for tag, share_idx, force in (
                ("native", leader_sel, None),
                ("arith", sel_oracle, sel_oracle)):
            # `native` on layer 3 consumes the reference's OWN layer-2 selection,
            # the analogue of test_glm_layer --leader-layer 2. On layer 2 the
            # indexer publishes its own, so share_idx is ignored there.
            h, taps, published = long_pass(inj, cos, sin, W, pre, cfg, packed,
                                           layer, share_idx, force)
            if layer == 2 and tag == "native":
                leader_sel = published
            tol = LONG_TOL_ARITH if tag == "arith" else LONG_TOL_NATIVE
            a = np.load(os.path.join(oracle_dir, f"layer{layer}_output_hidden.npy"))[0, -1]
            emit(f"L{layer} {tag} output_hidden", f"max_rel {_rel(a, h):.4e}",
                 _rel(a, h) < tol)
            for name in LONG_SUBSTEPS:
                p = os.path.join(oracle_dir, f"layer{layer}_{name}.npy")
                if not os.path.exists(p):
                    continue
                rel = _rel(_last(np.load(p)), taps[name])
                emit(f"L{layer} {tag} {name}", f"max_rel {rel:.4e}", rel < tol)
            p = os.path.join(oracle_dir, f"layer{layer}_topk_idx.npy")
            if os.path.exists(p):
                oi = set(np.load(p)[-1].tolist())
                emit(f"L{layer} {tag} router topk_idx",
                     f"exact set match: {oi == set(taps['topk_idx'].tolist())}",
                     oi == set(taps["topk_idx"].tolist()))
            if tag == "native":
                rows, ov = _score_report(taps, oracle_dir, layer, sel_oracle)
                for nm, rel in rows:
                    gated = nm == "indexer_index_score"
                    if gated:
                        emit(f"L{layer} {nm}", f"max_rel {rel:.4e}", rel < LONG_TOL_SCORE)
                    elif verbose:
                        print(f"  {'L%d %s' % (layer, nm):34s} max_rel {rel:.4e}  "
                              f"(reported, not gated)")
                if ov is not None and verbose:
                    n = len(sel_oracle)
                    print(f"  {'L%d topk_set' % layer:34s} {ov} of {n} shared "
                          f"({100.0 * ov / n:.4f}%)  (reported, NEVER gated)")
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


# ---------------------------------------------------------------------------
# Negative controls for the indexer (seq 4096 only)
# ---------------------------------------------------------------------------
# Task 5 measured two of these as UNDETECTABLE against a bf16 transformers oracle
# and explained why. They are run anyway and reported as not separating: "this
# suite cannot see X" is a result, and a gate tightened to manufacture the catch
# would fail the correct reference.
LONG_UNDETECTABLE = {
    # Same size as one of the two bf16 terms that set the index-score floor
    # (q and k each contribute ~2.4e-03). Task 5: no comparison against a bf16
    # transformers oracle can separate this eps at these magnitudes, and the
    # obvious remedy -- rounding q to bf16 to match transformers -- would only
    # remove one of the two terms.
    "indexer k_norm eps 1e-6 -> 1e-5",
    # Selecting 2047 or 2049 keys instead of 2048. The 2048th key carries almost
    # no attention weight, so adding or dropping it moves the output by less than
    # the two boundary keys the two implementations already disagree on.
    "index_topk 2047 (one key fewer)",
    "index_topk 2049 (one key more)",
}


def _ik_prefill(inj, cos, sin, W, cfg, chunk=256):
    """Recompute only the indexer key cache -- the MLA K/V is unaffected by any
    of the key-side indexer defects, so re-running prefill_kv would be waste."""
    out = []
    for c0 in range(0, inj.shape[0], chunk):
        xb = inj[c0:c0 + chunk]
        x = R.rms_norm(xb, W["input_layernorm"], cfg["rms_eps"])
        out.extend(R.indexer_keys(x, W, cfg,
                                  np.arange(c0, c0 + xb.shape[0]), cos, sin))
    return out


def _eval_long(cfg, packed, oracle_dir, prep, layer, share_idx=None, force=None,
               ik=None):
    """One `native` pass at 4096. Returns (worst_substep, worst_rel, score_rel)."""
    inj, cos, sin, W, pre = prep[layer]
    pre2 = dict(pre)
    if ik is not None:
        pre2["ik"] = ik
    h, taps, published = long_pass(inj, cos, sin, W, pre2, cfg, packed, layer,
                                   share_idx, force)
    a = np.load(os.path.join(oracle_dir, f"layer{layer}_output_hidden.npy"))[0, -1]
    worst = ("output_hidden", _rel(a, h))
    for name in LONG_SUBSTEPS:
        p = os.path.join(oracle_dir, f"layer{layer}_{name}.npy")
        if os.path.exists(p):
            rel = _rel(_last(np.load(p)), taps[name])
            if rel > worst[1]:
                worst = (name, rel)
    score = None
    p = os.path.join(oracle_dir, f"layer{layer}_indexer_index_score_last.npy")
    if "indexer_index_score" in taps and os.path.exists(p):
        score = _rel(np.load(p), taps["indexer_index_score"])
    return worst[0], worst[1], score, published


def long_controls(store, cfg, packed, oracle_dir, prep):
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
    def wproj(fn):
        """Perturb weights_proj in place on the already-loaded layer weights."""
        W = prep[2][3]
        old = W["ix_weights_proj"]
        W["ix_weights_proj"] = fn(old)
        try:
            yield
        finally:
            W["ix_weights_proj"] = old

    @contextlib.contextmanager
    def cfgkey(key, value):
        old = cfg[key]
        cfg[key] = value
        try:
            yield
        finally:
            cfg[key] = old

    def knorm_as_rms(x, w, b, eps):
        """The trap: RMSNorm instead of LayerNorm -- no mean subtraction, no bias.
        This repo's rms_norm is one import away, which is exactly the risk."""
        x = x.astype(np.float32)
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w

    def keys_trailing(x, W, cfg_, pos, cos, sin):
        """RoPE on the TRAILING 64 -- the MAIN attention path's slice, not the
        indexer's (M:237 vs M:387). The slice ends are opposite."""
        Rr = cfg_["qk_rope"]
        k = x.astype(np.float32) @ W["ix_wk"].T
        k = R.layer_norm(k, W["ix_k_norm_w"], W["ix_k_norm_b"], R.INDEXER_LN_EPS)
        return np.concatenate([k[..., :-Rr],
                               R.rope_interleave(k[..., -Rr:], cos[pos], sin[pos])], -1)

    def scores_trailing(q_resid, x, K_ix, W, cfg_, pos, cos, sin):
        H, D, Rr = cfg_["index_heads"], cfg_["index_dim"], cfg_["qk_rope"]
        q = (q_resid.astype(np.float32) @ W["ix_wq_b"].T).reshape(H, D)
        q = np.concatenate([q[:, :-Rr],
                            R.rope_interleave(q[:, -Rr:], cos[pos], sin[pos])], -1)
        s = np.maximum((q @ K_ix.T) * np.float32(D ** -0.5), 0.0)
        w = (W["ix_weights_proj"] @ x.astype(np.float32)) * np.float32(H ** -0.5)
        return q, w, (w[:, None] * s).sum(0, dtype=np.float32)

    def scores_relu_after(q_resid, x, K_ix, W, cfg_, pos, cos, sin):
        """relu AFTER the head combination instead of before it (M:248)."""
        H, D, Rr = cfg_["index_heads"], cfg_["index_dim"], cfg_["qk_rope"]
        q = (q_resid.astype(np.float32) @ W["ix_wq_b"].T).reshape(H, D)
        q = np.concatenate([R.rope_interleave(q[:, :Rr], cos[pos], sin[pos]), q[:, Rr:]], -1)
        s = (q @ K_ix.T) * np.float32(D ** -0.5)
        w = (W["ix_weights_proj"] @ x.astype(np.float32)) * np.float32(H ** -0.5)
        return q, w, np.maximum((w[:, None] * s).sum(0, dtype=np.float32), 0.0)

    sel_o = np.load(os.path.join(oracle_dir, "layer2_topk_indices.npy"))[0, -1]
    T = prep[2][0].shape[0]

    def swapped(n, rng=np.random.default_rng(20260802)):
        """n of the selected keys replaced by keys that were not selected."""
        chosen = set(sel_o.tolist())
        spare = np.array([i for i in range(T) if i not in chosen])
        out = sel_o.copy()
        out[rng.choice(len(out), n, replace=False)] = rng.choice(spare, n, replace=False)
        return out.astype(np.int32)

    @contextlib.contextmanager
    def rope_trailing():
        with patch(R, "indexer_keys", keys_trailing), \
             patch(R, "indexer_scores", scores_trailing):
            yield

    # (label, layer, pass, ctx factory, needs an indexer-key re-prefill, forced set)
    #
    # Which pass a control belongs to is not cosmetic -- it decides which baseline
    # and which gate it is judged against. An arithmetic defect changes the index
    # scores and therefore the selection, so it is a `native` defect. A topk_swap
    # perturbs the ORACLE's selection with everything else pinned, so it is an
    # `arith` defect and must be compared against the `arith` baseline; scoring it
    # against the looser `native` baseline would hide the sensitivity it exists to
    # measure.
    controls = [
        ("baseline", 2, "native", None, False, None),
        ("baseline", 3, "native", None, False, None),
        ("baseline", 3, "arith", None, False, sel_o),
        ("indexer k_norm as RMSNorm", 2, "native",
         lambda: patch(R, "layer_norm", knorm_as_rms), True, None),
        ("indexer k_norm eps 1e-6 -> 1e-5", 2, "native",
         lambda: patch(R, "INDEXER_LN_EPS", 1e-5), True, None),
        ("indexer RoPE on trailing 64", 2, "native", rope_trailing, True, None),
        ("relu AFTER head combination", 2, "native",
         lambda: patch(R, "indexer_scores", scores_relu_after), False, None),
        ("weights_proj head permutation", 2, "native",
         lambda: wproj(lambda w: w[::-1].copy()), False, None),
        ("weights_proj x 2 (scale only)", 2, "native",
         lambda: wproj(lambda w: w * 2.0), False, None),
        ("index_topk 2047 (one key fewer)", 2, "native",
         lambda: cfgkey("index_topk", 2047), False, None),
        ("index_topk 2049 (one key more)", 2, "native",
         lambda: cfgkey("index_topk", 2049), False, None),
        ("topk_swap N=1", 3, "arith", None, False, swapped(1)),
        ("topk_swap N=2", 3, "arith", None, False, swapped(2)),
        ("topk_swap N=8", 3, "arith", None, False, swapped(8)),
        ("topk_swap N=64", 3, "arith", None, False, swapped(64)),
    ]

    print("\n--- indexer negative controls, seq 4096 "
          "(a defect MUST show as 'caught') ---")
    print(f"{'perturbation':34s} {'L':>2s} {'pass':>6s}  {'worst substep':>30s}  "
          f"{'index_score':>11s}  verdict")
    missed, base = [], {}
    leader = None
    for label, layer, pass_, ctx, reprefill, force in controls:
        mgr = ctx() if ctx is not None else None
        if mgr is not None:
            mgr.__enter__()
        try:
            inj, cos, sin, W, _ = prep[2]
            ik = _ik_prefill(inj[:T - 1], cos, sin, W, cfg) if reprefill else None
            if layer == 3:
                # A consumer needs a leader's selection. `native` uses the
                # reference's OWN layer-2 indices (the analogue of --leader-layer
                # 2); `arith` uses the oracle's, optionally corrupted.
                share_idx = force if pass_ == "arith" else leader
                name, rel, score, _ = _eval_long(cfg, packed, oracle_dir, prep, 3,
                                                 share_idx=share_idx, force=force)
            else:
                name, rel, score, pub = _eval_long(cfg, packed, oracle_dir, prep, 2,
                                                   ik=ik, force=force)
                if label == "baseline":
                    leader = pub
        finally:
            if mgr is not None:
                mgr.__exit__(None, None, None)
        if label == "baseline":
            base[(layer, pass_)] = (rel, score)
        b_rel, b_score = base[(layer, pass_)]
        gate = LONG_TOL_ARITH if pass_ == "arith" else LONG_TOL_NATIVE
        caught = rel >= gate or (score is not None and score >= LONG_TOL_SCORE)
        bad = (label == "baseline" and caught) or (label != "baseline" and not caught)
        if bad:
            missed.append(label)
        sep = f"{rel / b_rel:.2f}x" if b_rel else "-"
        ssep = f" score {score / b_score:.0f}x" if (score and b_score) else ""
        print(f"{label:34s} {layer:2d} {pass_:>6s}  {name:>19s} {rel:.3e}  "
              f"{'-' if score is None else f'{score:.3e}'}  "
              f"{'caught' if caught else 'not caught':>10s}  "
              f"({sep}{ssep})"
              + ("   <-- MISSED" if bad and label not in LONG_UNDETECTABLE else ""))
    print("\nKnown-undetectable at output level, measured not assumed: "
          + "; ".join(sorted(LONG_UNDETECTABLE)) + ".")
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
    ap.add_argument("--oracle-long", default=None,
                    help="dump_glm_oracle.py --seq 4096 output directory holding "
                         "BOTH layer2_* and layer3_* (Task 5's fixtures). Enables "
                         "the indexer check; without it the indexer is a no-op.")
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

    print("IndexShare contract")
    ok &= check_index_share(store, cfg, args.packed, oracle[3])

    prep = {}
    if args.oracle_long:
        print("\n--- seq 4096: the indexer is live (layer 2 owner, layer 3 consumer) ---")
        good, _ = run_long(store, cfg, args.packed, args.oracle_long, prep=prep)
        ok &= good

    if args.negative_controls:
        missed = negative_controls(store, cfg, args.packed, oracle)
        ok &= not [m for m in missed if m[0] != "LORA_NORM_EPS = rms_eps (1e-5)"]
        if args.oracle_long:
            missed_l = long_controls(store, cfg, args.packed, args.oracle_long, prep)
            ok &= not [m for m in missed_l if m not in LONG_UNDETECTABLE]

    print(f"\npeak RSS: {R.peak_rss_gb():.2f} GB")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
