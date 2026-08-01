#!/usr/bin/env python3
"""ref_glm_chain.py -- full-model GLM-5.2 forward in numpy, streaming weights.

Direct transcription of transformers' GlmMoeDsaModel (transformers 5.14.1,
models/glm_moe_dsa/modeling_glm_moe_dsa.py). Non-expert weights stream from the
safetensors shards one layer at a time; routed experts stream from the packed
per-layer blocks written by repack_experts_glm.py.

This is deliberately the *direct* form -- no MLA absorption, no fused expert
layout. The CUDA path uses the absorbed MLA form, so an independent direct
implementation here means the chain comparison cross-checks the absorption
rather than repeating the same algebra twice.

Citations below are `modeling_glm_moe_dsa.py:LINE` (`M:`) and
`configuration_glm_moe_dsa.py:LINE` (`C:`), transformers 5.14.1.

Usage:
  ref_glm_chain.py --model DIR --packed DIR --prompt-ids 1,2,3 --tokens 5 --out DIR
  ref_glm_chain.py --model DIR --packed DIR --inject-hidden X.npy --inject-layer 3 \
      --inject-only --out DIR
"""
import argparse
import json
import os
import resource
import sys
import time

import numpy as np

from glm_mxfp4_numpy import dequant

# ---------------------------------------------------------------------------
# Numerically load-bearing primitives. Each one names the line it transcribes.
# ---------------------------------------------------------------------------

# GlmMoeDsaRMSNorm's class-default eps (M:49). q_a_layernorm (M:339) and
# kv_a_layernorm (M:347) are constructed WITHOUT an eps argument and so use this,
# NOT config.rms_norm_eps (1e-5), which reaches only input_layernorm /
# post_attention_layernorm (M:588-589) and model.norm (M:667).
LORA_NORM_EPS = 1e-6


def rms_norm(x, w, eps):
    """GlmMoeDsaRMSNorm.forward, M:57-62.

    PLAIN `w * x * rsqrt(mean(x^2) + eps)` in float32 -- there is NO (1 + w)
    delta parameterization anywhere in GLM-5.2, unlike the Qwen3.6 port. Getting
    this wrong is silent and produces plausible-looking output.

    `eps` is NOT one value: pass cfg["rms_eps"] (1e-5) for input_layernorm /
    post_attention_layernorm / model.norm, and LORA_NORM_EPS (1e-6) for
    q_a_layernorm / kv_a_layernorm.

    Note: M:62 casts back to the input dtype before multiplying by the weight;
    this reference multiplies in float32 throughout. Strictly more precise, far
    inside the comparison tolerances, mentioned only so a last-bit difference is
    not mistaken for a bug.
    """
    x = x.astype(np.float32)
    return w.astype(np.float32) * (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)))


def rope_tables(max_pos, rot, theta):
    """M:112-114 + M:119-130.

    `dim` is `config.head_dim`, which __post_init__ overwrites with
    qk_rope_head_dim (C:153) -- so rot == 64, NOT the 192 that config.json's
    head_dim key claims and NOT qk_head_dim. inv_freq has rot/2 = 32 entries;
    transformers builds emb = cat(freqs, freqs) and the interleaved apply then
    uses only the first half (M:161-162), so rot/2 columns is all we need.
    """
    inv = 1.0 / (theta ** (np.arange(0, rot, 2, dtype=np.float64) / rot))
    ang = np.arange(max_pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope_interleave(x, cos, sin):
    """apply_rotary_pos_emb_interleave, M:164-168.

    x: [..., rot]. Input pairs are ADJACENT (2i, 2i+1); the output is the
    HALVES layout cat([even*cos - odd*sin, odd*cos + even*sin]). That output
    permutation is applied identically to q and k, so q.k is unchanged -- but
    any elementwise comparison of an intermediate q_rot must use this layout.
    """
    a, b = x[..., 0::2], x[..., 1::2]
    return np.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)


def silu(x):
    return x / (1.0 + np.exp(-x))          # hidden_act == "silu" (C:113)


def route(x_norm, gate_w, bias, top_k, routed_scaling, norm_topk_prob):
    """GlmMoeDsaTopkRouter.forward, M:485-510, at n_group == topk_group == 1.

    At n_group == 1 the group-limited block (M:490-503) selects the single
    existing group, so score_mask is all-ones and the masked_fill is a no-op.
    Assert rather than silently generalizing.

    `bias` MUST arrive as float32. transformers registers it as an fp32 buffer
    (M:483) and lists it in _keep_in_fp32_modules_strict, on purpose: the values
    sit near 34, where the bf16 grid step is 0.25, while the whole sigmoid signal
    spans ~0.22 -- less than one grid cell. A bf16 bias collapses all 256 biases
    onto three values and changes 7 of the 8 selected experts.
    """
    assert bias.dtype == np.float32, f"e_score_correction_bias must be f32, got {bias.dtype}"
    logits = x_norm.astype(np.float32) @ gate_w.astype(np.float32).T   # M:487, float32
    scores = 1.0 / (1.0 + np.exp(-logits))                             # M:488 SIGMOID, not softmax
    choice = scores + bias.astype(np.float32)                          # M:489 e_score_correction_bias
    idx = np.argpartition(-choice, top_k)[:top_k]                      # M:504 selection uses the BIASED score
    w = scores[idx].copy()                                             # M:505 weight is the UNBIASED score
    if norm_topk_prob:
        w /= (w.sum() + 1e-20)                                         # M:506-508 renormalized
    w *= routed_scaling                                                # M:509 AFTER the renorm; 2.5
    return logits, idx.astype(np.int32), w


# ---------------------------------------------------------------------------
# safetensors streaming reader
# ---------------------------------------------------------------------------
# Deliberately a small self-contained reader rather than safetensors.safe_open:
#   * safe_open(framework="np") refuses BF16, and this checkpoint is almost
#     entirely BF16;
#   * framework="pt" would drag torch in and copy every tensor into RAM, while
#     what this reference needs is a *row slice* of a 1.9 GB lm_head.
# Everything below is backed by np.memmap, so untouched rows stay in the page
# cache and never count against RSS.

_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
             "U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8,
             "F8_E4M3": 1, "F8_E5M2": 1, "BOOL": 1}


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class ShardStore:
    """Memory-mapped access to a sharded safetensors checkpoint."""

    MAX_OPEN = 4

    def __init__(self, model_dir):
        self.dir = model_dir
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._open = {}          # shard -> (header, data_start, memmap)
        self._order = []

    def _shard(self, shard):
        if shard in self._open:
            self._order.remove(shard)
            self._order.append(shard)
            return self._open[shard]
        path = os.path.join(self.dir, shard)
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        while len(self._order) >= self.MAX_OPEN:
            drop = self._order.pop(0)
            del self._open[drop]
        self._open[shard] = (header, 8 + n, mm)
        self._order.append(shard)
        return self._open[shard]

    def has(self, name):
        return name in self.weight_map

    def shape(self, name):
        header, _, _ = self._shard(self.weight_map[name])
        return tuple(header[name]["shape"])

    def raw(self, name, row0=None, row1=None):
        """Return the tensor (or rows [row0, row1) of it) as numpy.

        BF16/F16 are widened to float32; F32 is returned as-is; U8 stays U8.
        """
        header, base, mm = self._shard(self.weight_map[name])
        ent = header[name]
        dt, shape = ent["dtype"], list(ent["shape"])
        off = ent["data_offsets"][0]
        isz = _ITEMSIZE[dt]
        row_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        if row0 is not None:
            off += row0 * row_elems * isz
            shape = [row1 - row0] + shape[1:]
        nbytes = int(np.prod(shape)) * isz
        buf = mm[base + off: base + off + nbytes]
        if dt == "BF16":
            return _bf16_to_f32(buf.view(np.uint16)).reshape(shape)
        if dt == "F16":
            return buf.view(np.float16).astype(np.float32).reshape(shape)
        if dt == "F32":
            return np.array(buf.view(np.float32).reshape(shape))
        if dt == "U8":
            return np.array(buf.reshape(shape))
        raise ValueError(f"unhandled dtype {dt} for {name}")

    def dequant_or_raw(self, name):
        """MXFP4 tensors carry a companion `_scale`; dense ones do not."""
        if self.has(name + "_scale"):
            return dequant(self.raw(name), self.raw(name + "_scale"))
        return self.raw(name)


# ---------------------------------------------------------------------------
# Weight loading -- one layer at a time
# ---------------------------------------------------------------------------

def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        c = json.load(f)
    rope = c.get("rope_parameters") or {}
    rope_type = rope.get("rope_type", "default")
    # M:361-366: the YaRN mscale correction on `scaling` applies only for a
    # non-default rope_type. This checkpoint is "default", so scaling stays
    # qk_head_dim ** -0.5.
    assert rope_type == "default", f"rope_type {rope_type!r} needs the mscale path"
    assert c["n_group"] == 1 and c["topk_group"] == 1, "grouped routing not transcribed"
    assert c["scoring_func"] == "sigmoid"
    assert c["hidden_act"] == "silu"
    assert not c["tie_word_embeddings"], "lm_head is expected to be its own tensor"
    qk_nope, qk_rope = c["qk_nope_head_dim"], c["qk_rope_head_dim"]
    return {
        "hidden": c["hidden_size"],
        "n_layers": c["num_hidden_layers"],
        "n_heads": c["num_attention_heads"],
        "qk_nope": qk_nope,
        "qk_rope": qk_rope,
        "qk_head": qk_nope + qk_rope,        # C:150; == config.qk_head_dim (256)
        "v_head": c["v_head_dim"],
        "q_lora": c["q_lora_rank"],
        "kv_lora": c["kv_lora_rank"],
        "rms_eps": c["rms_norm_eps"],
        "dense_inter": c["intermediate_size"],
        "moe_inter": c["moe_intermediate_size"],
        "n_shared": c["n_shared_experts"],
        "n_experts": c["n_routed_experts"],
        "top_k": c["num_experts_per_tok"],
        "routed_scaling": c["routed_scaling_factor"],
        "norm_topk_prob": c["norm_topk_prob"],
        "dense_first": c["first_k_dense_replace"],
        "index_topk": c["index_topk"],
        "vocab": c["vocab_size"],
        # C:153 -- __post_init__ overwrites head_dim with qk_rope_head_dim, so the
        # rotary table dim is 64. config.json's head_dim: 192 is dead.
        "rot": qk_rope,
        "rope_theta": rope["rope_theta"],
    }


def load_layer_weights(store, cfg, layer):
    """Stream one decoder layer's non-expert weights. ~660 MB in float32.

    The indexer submodule (self_attn.indexer.*) is deliberately NOT loaded: at
    T <= index_topk its top-k selection is every causally valid token, so it
    cannot change the attention output (see layer_forward's assert).
    """
    p = f"model.layers.{layer}."
    W = {
        "input_layernorm": store.raw(p + "input_layernorm.weight"),
        "post_attention_layernorm": store.raw(p + "post_attention_layernorm.weight"),
        "q_a_proj": store.dequant_or_raw(p + "self_attn.q_a_proj.weight"),
        "q_a_layernorm": store.raw(p + "self_attn.q_a_layernorm.weight"),
        "q_b_proj": store.dequant_or_raw(p + "self_attn.q_b_proj.weight"),
        "kv_a_proj_with_mqa": store.dequant_or_raw(p + "self_attn.kv_a_proj_with_mqa.weight"),
        "kv_a_layernorm": store.raw(p + "self_attn.kv_a_layernorm.weight"),
        "kv_b_proj": store.dequant_or_raw(p + "self_attn.kv_b_proj.weight"),
        "o_proj": store.dequant_or_raw(p + "self_attn.o_proj.weight"),
    }
    if layer < cfg["dense_first"]:
        W["gate_proj"] = store.dequant_or_raw(p + "mlp.gate_proj.weight")
        W["up_proj"] = store.dequant_or_raw(p + "mlp.up_proj.weight")
        W["down_proj"] = store.dequant_or_raw(p + "mlp.down_proj.weight")
    else:
        W["gate.weight"] = store.raw(p + "mlp.gate.weight")
        bias = store.raw(p + "mlp.gate.e_score_correction_bias")
        # Must stay F32 -- see route()'s docstring. Assert the checkpoint really
        # stores it that way rather than trusting the loader.
        assert bias.dtype == np.float32, "e_score_correction_bias is not F32 in the checkpoint"
        W["gate.e_score_correction_bias"] = bias
        W["sh_gate"] = store.dequant_or_raw(p + "mlp.shared_experts.gate_proj.weight")
        W["sh_up"] = store.dequant_or_raw(p + "mlp.shared_experts.up_proj.weight")
        W["sh_down"] = store.dequant_or_raw(p + "mlp.shared_experts.down_proj.weight")
    return W


# ---------------------------------------------------------------------------
# Packed routed experts
# ---------------------------------------------------------------------------
# repack_experts_glm.py writes, per expert, in this order:
#   gate_w | gate_s | up_w | up_s | down_w | down_s
# with sizes I*H/2, I*H/32, I*H/2, I*H/32, H*I/2, H*I/32 for I = 2048, H = 6144,
# i.e. 6291456, 393216, 6291456, 393216, 6291456, 393216 and a per-expert stride
# EXPERT_BYTES = 20054016.

_PACK_FH = {}


def _packed_handle(packed, layer):
    key = (packed, layer)
    if key not in _PACK_FH:
        _PACK_FH[key] = open(os.path.join(packed, f"layer_{layer}.bin"), "rb")
    return _PACK_FH[key]


def expert_sizes(cfg):
    I, H = cfg["moe_inter"], cfg["hidden"]
    return [(I, H // 2), (I, H // 32), (I, H // 2), (I, H // 32),
            (H, I // 2), (H, I // 32)]


def read_expert(packed, layer, e, cfg):
    """Read one expert's six sub-blocks out of the packed layer file."""
    sizes = expert_sizes(cfg)
    nb = [a * b for a, b in sizes]
    stride = sum(nb)
    fh = _packed_handle(packed, layer)
    fh.seek(e * stride)
    blob = np.frombuffer(fh.read(stride), dtype=np.uint8)
    outs, off = [], 0
    for (r, c), n in zip(sizes, nb):
        outs.append(blob[off:off + n].reshape(r, c))
        off += n
    assert off == stride
    return outs


# ---------------------------------------------------------------------------
# One decoder layer
# ---------------------------------------------------------------------------

# Substep names, matching dump_glm_oracle.py's SUBSTEPS exactly so the two can
# be compared name-for-name.
TAPS = ["post_input_norm", "attn_out", "post_attn_hidden", "post_post_norm",
        "router_logits", "topk_idx", "topk_w", "shared_out", "moe_out",
        "output_hidden"]


def layer_forward(h, W, cfg, pos, cos, sin, cache, packed, layer, taps=None):
    """One decoder layer for ONE token at absolute position `pos`.

    h: [hidden] float32. cache: dict with 'k' [T, n_heads, qk_head] and
    'v' [T, n_heads, v_head] float32 -- the materialized form transformers itself
    caches (M:402-403). Pre-norm block of M:602-620, attention from M:381-453.

    `taps`, if a dict, is filled with the intermediate tensors named in TAPS.
    The final residual add is NOT sensitive enough to validate this layer on its
    own -- it is dominated by the incoming hidden state, so a wrong router bias
    or a wrong routed_scaling_factor moves the layer OUTPUT by well under the
    comparison tolerance while moving `moe_out` by an order of magnitude more.
    Comparing the taps against the oracle's per-substep dumps is what actually
    pins the layer down.
    """
    HN, NP, RP, VD = cfg["n_heads"], cfg["qk_nope"], cfg["qk_rope"], cfg["v_head"]
    DH, L, eps = NP + RP, cfg["kv_lora"], cfg["rms_eps"]

    def tap(name, val):
        if taps is not None:
            taps[name] = np.asarray(val)

    resid = h
    x = rms_norm(h, W["input_layernorm"], eps)                             # M:603
    tap("post_input_norm", x)

    # --- MLA ---------------------------------------------------------------
    # LORA_NORM_EPS (1e-6), not eps (1e-5): M:339 / M:347 pass no eps, so the
    # GlmMoeDsaRMSNorm class default at M:49 applies to both LoRA norms.
    q_resid = rms_norm(x @ W["q_a_proj"].T, W["q_a_layernorm"], LORA_NORM_EPS)  # M:385, M:339
    q = (q_resid @ W["q_b_proj"].T).reshape(HN, DH)                        # M:386
    q_pass, q_rot = q[:, :NP], q[:, NP:]                                   # M:387 nope FIRST, rope LAST

    ckv = x @ W["kv_a_proj_with_mqa"].T                                    # M:389, [576]
    kc, k_rot = ckv[:L], ckv[L:]                                           # M:390
    kc = rms_norm(kc, W["kv_a_layernorm"], LORA_NORM_EPS)                  # M:391, M:347 -- normalizes ONLY
                                                                           #   the first 512; k_rot is NOT
                                                                           #   normalized. eps = 1e-6.
    kv = (kc @ W["kv_b_proj"].T).reshape(HN, NP + VD)                      # M:391, head-major rows
    k_pass, v = kv[:, :NP], kv[:, NP:]                                     # M:392

    q_rot = rope_interleave(q_rot, cos[pos], sin[pos])                     # M:396
    k_rot = rope_interleave(k_rot, cos[pos], sin[pos])                     # M:396, one shared vector (MQA)
    k = np.concatenate([k_pass, np.broadcast_to(k_rot, (HN, RP))], -1)     # M:397, M:400
    q = np.concatenate([q_pass, q_rot], -1)                                # M:399

    cache["k"].append(k)                                                   # M:402-403
    cache["v"].append(v)
    K = np.stack(cache["k"])
    V = np.stack(cache["v"])                                               # [T, HN, *]

    # scaling = qk_head_dim ** -0.5 (M:360). rope_type is "default" in this
    # checkpoint, so the YaRN mscale correction at M:361-366 does NOT apply.
    scale = DH ** -0.5
    s = np.einsum("hd,thd->ht", q, K).astype(np.float32) * scale
    # No DSA mask: with total length T <= index_topk (2048) the indexer's
    # topk(min(index_topk, T)) returns EVERY index (M:262-263), so index_mask is
    # all-False and the masked_fill at M:432 is a no-op -- only causality
    # survives, and decoding one token at a time is causal by construction.
    assert K.shape[0] <= cfg["index_topk"], "past index_topk the indexer-free path is not exact"
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    o = np.einsum("ht,thd->hd", p, V).reshape(HN * VD)                     # M:297
    attn_out = o @ W["o_proj"].T                                           # M:452
    tap("attn_out", attn_out)
    h = resid + attn_out                                                   # M:615
    tap("post_attn_hidden", h)

    # --- MLP / MoE ---------------------------------------------------------
    resid = h
    xn = rms_norm(h, W["post_attention_layernorm"], eps)                   # M:618
    tap("post_post_norm", xn)
    if layer < cfg["dense_first"]:
        y = (silu(xn @ W["gate_proj"].T) * (xn @ W["up_proj"].T)) @ W["down_proj"].T   # M:468
    else:
        logits, idx, wt = route(xn, W["gate.weight"], W["gate.e_score_correction_bias"],
                                cfg["top_k"], cfg["routed_scaling"], cfg["norm_topk_prob"])
        tap("router_logits", logits)
        tap("topk_idx", idx)
        tap("topk_w", wt)
        y = np.zeros(cfg["hidden"], np.float32)
        for e, a in zip(idx, wt):                                          # M:538-548
            gw, gs, uw, us, dw, ds = read_expert(packed, layer, int(e), cfg)
            g = xn @ dequant(gw, gs).T
            u = xn @ dequant(uw, us).T
            y += a * ((silu(g) * u) @ dequant(dw, ds).T)
        # shared expert: SAME normalized input, added AFTER the routed sum, and
        # NOT multiplied by routed_scaling_factor (M:567-573)
        shared = (silu(xn @ W["sh_gate"].T) * (xn @ W["sh_up"].T)) @ W["sh_down"].T
        tap("shared_out", shared)
        y = y + shared
    tap("moe_out", y)
    out = resid + y                                                        # M:620
    tap("output_hidden", out)
    return out, cache


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

def lm_head_chunked(store, h, vocab, chunk=8192):
    """M:794. lm_head.weight is [154880, 6144] BF16 = 1.9 GB; never materialize
    the whole thing in float32 (that would be 3.8 GB)."""
    out = np.empty(vocab, np.float32)
    for r0 in range(0, vocab, chunk):
        r1 = min(r0 + chunk, vocab)
        out[r0:r1] = store.raw("lm_head.weight", r0, r1) @ h
    return out


def embed_row(store, token_id):
    """model.embed_tokens.weight[id], BF16 -> F32, one row only."""
    return store.raw("model.embed_tokens.weight", token_id, token_id + 1)[0].copy()


def inject_setup(store, cfg, inject_path, layer):
    """Load a [seq, hidden] injected state plus the RoPE tables and layer weights.

    Shared by the --inject-hidden CLI path and check_ref_vs_oracle.py so both
    drive the reference identically.
    """
    inj = np.load(inject_path).astype(np.float32)
    if inj.ndim == 3:
        inj = inj[0]                                       # drop the batch axis
    assert inj.ndim == 2 and inj.shape[1] == cfg["hidden"], inj.shape
    cos, sin = rope_tables(inj.shape[0] + 8, cfg["rot"], cfg["rope_theta"])
    return inj, cos, sin, load_layer_weights(store, cfg, layer)


def peak_rss_gb():
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-ids", help="comma-separated token ids")
    ap.add_argument("--prompt-ids-file", help="file with comma/whitespace-separated token ids")
    ap.add_argument("--tokens", type=int, default=5)
    ap.add_argument("--dump-layer", action="append", type=int, default=None,
                    help="layer whose OUTPUT hidden state to dump at t=0 "
                         "(default: 0, 3, 40, 77)")
    ap.add_argument("--inject-hidden", help="[seq, hidden] float32 .npy to start from")
    ap.add_argument("--inject-layer", type=int, default=0)
    ap.add_argument("--inject-only", action="store_true",
                    help="run only --inject-layer; skip model.norm and lm_head")
    ap.add_argument("--dump-substeps", action="store_true",
                    help="with --inject-only, write ref_inject_{substep}_L{L}_p{pos}.npy; "
                         "in chain mode, write ref_chain_{substep}_L{layer}_t0.npy at every "
                         "--dump-layer. Task 10 gates per substep, not on the hidden state "
                         "alone -- a layer output is residual-dominated and hides MoE-side "
                         "error (see task-9-report.md section 4).")
    args = ap.parse_args()
    if args.inject_hidden and not args.inject_only:
        # --inject-hidden without --inject-only used to run layers L0..77 for every
        # position and then discard the result: the only np.save on this path is
        # inside `if args.inject_only`, and the function returns right after the
        # loop. It also reloaded all 78 layers' weights once per position,
        # O(seq * 78) shard reads, to produce nothing. Rejected rather than given
        # a meaning -- the tail-of-model variant has no consumer, and the Step 2
        # cross-check only ever wants the single-layer form.
        ap.error("--inject-hidden requires --inject-only "
                 "(the multi-layer injection path has no output and no consumer)")
    if args.dump_substeps and args.inject_hidden and not args.inject_only:
        ap.error("--dump-substeps requires --inject-only on the injection path")
    os.makedirs(args.out, exist_ok=True)

    store = ShardStore(args.model)
    cfg = load_config(args.model)
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Injection path (Step 2 cross-check against the single-layer oracle).
    # -----------------------------------------------------------------------
    # --inject-hidden PATH: a [seq, hidden] float32 array -- the SAME array the
    #   oracle was driven with (dump_glm_oracle.py's layer{L}_input_hidden.npy,
    #   squeezed of the leading batch axis). One row per position, fed in order so
    #   the KV cache fills exactly as it did in the oracle's length-`seq` run.
    # --inject-layer L: the layer that array is the input to.
    # --inject-only:   run ONLY layer L; skip model.norm and lm_head. Required --
    #                  see the argparse guard above.
    #
    # This genuinely STARTS from the supplied tensor: the embedding is never read
    # and layers 0..L-1 never run.
    if args.inject_hidden:
        L0 = args.inject_layer
        inj, cos, sin, W = inject_setup(store, cfg, args.inject_hidden, L0)
        cache = {"k": [], "v": []}
        for pos in range(inj.shape[0]):
            t0 = time.time()
            taps = {} if args.dump_substeps else None
            h, cache = layer_forward(inj[pos], W, cfg, pos, cos, sin,
                                     cache, args.packed, L0, taps)
            if taps:
                for name, val in taps.items():
                    np.save(os.path.join(args.out,
                                         f"ref_inject_{name}_L{L0}_p{pos}.npy"), val)
            path = os.path.join(args.out, f"ref_inject_hidden_L{L0}_p{pos}.npy")
            np.save(path, h.astype(np.float32))
            print(f"pos {pos}: {time.time() - t0:.1f}s -> {path}")
        print(f"peak RSS: {peak_rss_gb():.2f} GB   total {time.time() - t_start:.1f}s")
        return

    # -----------------------------------------------------------------------
    # Full chain: GlmMoeDsaModel.forward (M:712-732) + the head (M:794).
    # -----------------------------------------------------------------------
    if args.prompt_ids_file:
        text = open(args.prompt_ids_file).read()
        ids = [int(t) for t in text.replace(",", " ").split()]
    elif args.prompt_ids:
        ids = [int(t) for t in args.prompt_ids.split(",")]
    else:
        ap.error("one of --prompt-ids / --prompt-ids-file / --inject-hidden is required")

    dump_layers = set(args.dump_layer if args.dump_layer else [0, 3, 40, 77])
    n_layers = cfg["n_layers"]      # 78; layer 78 is the MTP head, out of scope
    total = len(ids) + args.tokens
    cos, sin = rope_tables(total + 8, cfg["rot"], cfg["rope_theta"])
    caches = {l: {"k": [], "v": []} for l in range(n_layers)}
    final_norm = store.raw("model.norm.weight")

    pos = 0
    tok = ids[0]
    generated = 0
    while True:
        t0 = time.time()
        h = embed_row(store, tok)
        for layer in range(n_layers):
            W = load_layer_weights(store, cfg, layer)          # streams, then drops
            # t=0 is the step that consumes the LAST prompt token and emits the
            # first generated token, so that is where the fixtures are taken.
            at_t0 = generated == 0 and pos == len(ids) - 1
            taps = {} if (args.dump_substeps and at_t0 and layer in dump_layers) else None
            h, caches[layer] = layer_forward(h, W, cfg, pos, cos, sin,
                                             caches[layer], args.packed, layer, taps)
            if taps:
                for name, val in taps.items():
                    np.save(os.path.join(args.out,
                                         f"ref_chain_{name}_L{layer}_t0.npy"), val)
            if layer in dump_layers and at_t0:
                np.save(os.path.join(args.out, f"ref_chain_hidden_L{layer}_t0.npy"),
                        h.astype(np.float32))
            del W
        if pos < len(ids) - 1:
            # still consuming the prompt; the logits here are not a generation step
            pos += 1
            tok = ids[pos]
            print(f"prompt pos {pos - 1}: {time.time() - t0:.1f}s")
            continue

        hn = rms_norm(h, final_norm, cfg["rms_eps"])           # M:728, model.norm
        logits = lm_head_chunked(store, hn, cfg["vocab"])      # M:794, [154880] float32
        topk = np.argsort(-logits)[:5].astype(np.int32)
        np.save(os.path.join(args.out, f"ref_chain_logits_{generated}.npy"), logits)
        np.save(os.path.join(args.out, f"ref_chain_topk_{generated}.npy"), topk)
        print(f"token {generated}: pos={pos} argmax={int(topk[0])} "
              f"{time.time() - t0:.1f}s  RSS={peak_rss_gb():.2f} GB", flush=True)
        generated += 1
        if generated >= args.tokens:
            break
        tok = int(topk[0])
        pos += 1

    print(f"peak RSS: {peak_rss_gb():.2f} GB   total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
