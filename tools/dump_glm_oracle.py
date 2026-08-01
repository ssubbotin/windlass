#!/usr/bin/env python3
"""dump_glm_oracle.py — run ONE GLM-5.2 decoder layer under transformers on CPU
and dump every substep tensor for the CUDA tests to compare against.

The full 753B model cannot be loaded; this instantiates a single decoder layer
from config and loads only that layer's weights from the local checkpoint.

Model type: glm_moe_dsa, architecture GlmMoeDsaForCausalLM. The decoder-layer
class is imported directly from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa
rather than discovered dynamically -- the module path and class name are known
ahead of time (see task-4 notes).

Usage (must run under the venv with transformers 5.14.1 / torch 2.11.0):
    python3 dump_glm_oracle.py \
        --model ./glm52-mxfp4 --layer 3 --out glm-oracle/

Above index_topk (Task 5) the indexer stops being a no-op, and two things that
did not matter below 2048 start to:

  * A "shared" layer's prev_topk_indices must be the REAL selection of its group
    leader. Below 2048 any full index list was equivalent (the top-k returns a
    permutation of all T indices there, so the mask is uniformly False); above
    2048 it is not. Dump the leader first and feed its saved indices in:

        python3 dump_glm_oracle.py --model M --layer 2 --seq 4096 --out O
        python3 dump_glm_oracle.py --model M --layer 3 --seq 4096 --out O \
            --prev-topk O/layer2_topk_indices.npy

  * indexer.weights_proj must be float32. transformers lists it in
    _keep_in_fp32_modules (modeling_glm_moe_dsa.py:643), but that is a
    from_pretrained-time behaviour and this script builds the layer on `meta` +
    load_state_dict, so it never runs here. --weights-proj-dtype replicates it
    (default fp32, i.e. the deployed layout); `bf16` exists only to MEASURE what
    the difference is worth.
"""
import argparse
import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaRotaryEmbedding,
    apply_rotary_pos_emb_interleave,
)

from glm_mxfp4_numpy import dequant

# Names of the substeps this script dumps, in the order the interface spec lists them.
SUBSTEPS = [
    "input_hidden", "post_input_norm", "attn_out", "post_attn_hidden",
    "post_post_norm", "router_logits", "topk_idx", "topk_w", "shared_out",
    "moe_out", "output_hidden",
]

_EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def load_layer_state(model_dir, layer, num_experts):
    """Return {param_name_without_layer_prefix: tensor} for one layer,
    dequantizing MXFP4 tensors into dense bf16.

    The checkpoint stores routed-expert weights per-expert
    (mlp.experts.{i}.{gate_proj,up_proj,down_proj}.weight[_scale]), matching a
    Linear-per-expert layout. GlmMoeDsaExperts, however, holds them as two
    fused 3D nn.Parameter tensors: gate_up_proj [E, 2*inter, hidden] (rows
    0:inter = gate_proj, inter:2*inter = up_proj, per expert.forward's
    `.chunk(2, dim=-1)` on the linear output) and down_proj [E, hidden, inter].
    This function dequantizes each per-expert tensor and reassembles them into
    that fused layout so load_state_dict matches by name.
    """
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    prefix = f"model.layers.{layer}."
    names = [n for n in idx["weight_map"] if n.startswith(prefix)]
    raw = {}
    handles = {}
    for n in names:
        shard = idx["weight_map"][n]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(model_dir, shard), framework="pt")
        raw[n] = handles[shard].get_tensor(n)

    def dq(name):
        w = raw[name].numpy().astype(np.uint8)
        s = raw[name + "_scale"].numpy().astype(np.uint8)
        return torch.from_numpy(dequant(w, s)).to(torch.bfloat16)

    state = {}
    expert_gate = [None] * num_experts
    expert_up = [None] * num_experts
    expert_down = [None] * num_experts

    for n, t in raw.items():
        if n.endswith("_scale"):
            continue
        short = n[len(prefix):]
        m = _EXPERT_RE.match(short)
        if m:
            eidx, kind = int(m.group(1)), m.group(2)
            val = dq(n)
            if kind == "gate_proj":
                expert_gate[eidx] = val
            elif kind == "up_proj":
                expert_up[eidx] = val
            else:
                expert_down[eidx] = val
            continue
        scale_name = n + "_scale"
        if scale_name in raw:
            state[short] = dq(n)
        elif t.dtype == torch.float32:
            # Keep F32 tensors in F32. In a GLM-5.2 layer the only one is
            # mlp.gate.e_score_correction_bias, and transformers keeps it in fp32
            # on purpose: modeling_glm_moe_dsa.py:483 registers it as a float32
            # buffer and :641 lists it in _keep_in_fp32_modules_strict, so even a
            # bfloat16 load never downcasts it.
            #
            # Downcasting it is catastrophic, not cosmetic. Its values sit at
            # 33.985-34.623 with std 0.138. bfloat16 has 7 explicit mantissa bits,
            # so for 32 <= x < 64 the grid step is 2^5 * 2^-7 = 0.25, and rounding
            # collapses all 256 distinct biases onto exactly three representable
            # values {34.0, 34.25, 34.5}.
            #
            # Selection is on sigmoid(logit) + bias, and the sigmoid term spans
            # only 0.220 at the position compared by test_glm_layer -- LESS than
            # one grid cell. So under a bf16 bias the expert ordering is decided by
            # which grid point each expert landed on, not by the model: the router
            # output is quantized away entirely and 7 of the 8 selected experts
            # change. (Found 2026-07-31 while validating the CUDA layer runner:
            # the fixture's own router_logits recomputed the CUDA path's top-8
            # exactly, but disagreed with the fixture's own topk_idx.)
            state[short] = t
        else:
            state[short] = t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t

    any_expert = any(e is not None for e in expert_gate + expert_up + expert_down)
    if not any_expert:
        # Dense layer (mlp_layer_types[layer] == "dense"): self.mlp is a plain
        # GlmMoeDsaMLP with its own gate_proj/up_proj/down_proj, already
        # handled by the generic per-name loop above. No routed experts exist
        # for this layer, so there is nothing to reassemble.
        return state

    assert all(e is not None for e in expert_gate), "missing gate_proj for some expert"
    assert all(e is not None for e in expert_up), "missing up_proj for some expert"
    assert all(e is not None for e in expert_down), "missing down_proj for some expert"

    state["mlp.experts.gate_up_proj"] = torch.stack(
        [torch.cat([expert_gate[e], expert_up[e]], dim=0) for e in range(num_experts)], dim=0
    )
    state["mlp.experts.down_proj"] = torch.stack(expert_down, dim=0)
    return state


@torch.no_grad()
def trace_indexer(ix, hidden_states, q_resid, position_embeddings, attention_mask,
                  position_ids):
    """A VERBATIM copy of GlmMoeDsaIndexer.forward (modeling_glm_moe_dsa.py:203-263)
    with capture points added, minus the `past_key_values` branch (this script
    never passes a cache, so `update_indexer` is not reached).

    It exists so that a CUDA/oracle disagreement can be localised to a tensor
    instead of stopping at "the outputs differ". Its trustworthiness is not
    assumed: main() asserts its returned indices are IDENTICAL to the ones the
    real module produced during the layer forward. If the copy has drifted from
    upstream in any way that matters, that assert fires.
    """
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings
    q = ix.wq_b(q_resid)
    q = q.view(batch_size, seq_len, ix.n_heads, ix.head_dim)
    q_rot, q_pass = torch.split(q, [ix.qk_rope_head_dim, ix.head_dim - ix.qk_rope_head_dim], dim=-1)

    k = ix.k_norm(ix.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [ix.qk_rope_head_dim, ix.head_dim - ix.qk_rope_head_dim], dim=-1)

    q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)

    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * ix.softmax_scale
    scores = F.relu(scores)

    weights = ix.weights_proj(hidden_states.to(ix.weights_proj.weight.dtype)).float() * (ix.n_heads**-0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    # Zero-score fraction, measured before any masking (Task 1 asked for this
    # number at 4096 before the comparison was designed): after the relu the
    # tail of the selection is genuinely under-determined.
    last = seq_len - 1
    zero_frac_row = float((index_scores[0, last] == 0.0).float().mean())
    zero_frac_heads = float((scores[0, last] == 0.0).float().mean())

    if attention_mask is not None:
        index_scores = index_scores + attention_mask
    else:
        key_positions = torch.arange(index_scores.shape[-1], device=index_scores.device)
        causal = key_positions[None, None, :] > position_ids[:, :, None]
        index_scores = index_scores.masked_fill(causal, float("-inf"))

    topk = min(ix.index_topk, index_scores.shape[-1])
    idx = index_scores.topk(topk, dim=-1).indices.to(torch.int32)

    cap = {
        "indexer_index_score_last": index_scores[0, last].float().cpu().numpy(),
        "indexer_w_last": weights[0, last].float().cpu().numpy(),
        "indexer_q_last": q[0, last].float().cpu().numpy(),
        "indexer_k": k[0].float().cpu().numpy(),
        "_zero_frac_row": zero_frac_row,
        "_zero_frac_heads": zero_frac_heads,
    }
    return idx, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="where the layer runs. At --seq 4096 the eager attention "
                         "is [1,64,S,S]; cuda is ~100x faster and the difference "
                         "against cpu is measured in task-5-report.md.")
    ap.add_argument("--prev-topk", default=None,
                    help="path to a leader layer's *_topk_indices.npy, required "
                         "for a \"shared\" layer once --seq exceeds index_topk")
    ap.add_argument("--weights-proj-dtype", default="fp32", choices=["fp32", "bf16"],
                    help="dtype of indexer.weights_proj. fp32 replicates "
                         "_keep_in_fp32_modules (M:643), which this script's "
                         "meta-build otherwise skips. bf16 is a measurement only.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    had_quant_cfg = getattr(cfg, "quantization_config", None) is not None
    if had_quant_cfg:
        # We build the decoder layer directly (never through
        # AutoModelForCausalLM.from_pretrained), so the quantized-module
        # replacement hooks never run regardless -- but strip it anyway so a
        # plain bf16 layer is unambiguous and nothing downstream can key off it.
        cfg.quantization_config = None
    cfg._attn_implementation = "eager"
    print(f"quantization_config present in checkpoint config: {had_quant_cfg} "
          f"({'neutralized' if had_quant_cfg else 'n/a'})")

    torch.manual_seed(args.seed)

    with torch.device("meta"):
        layer = GlmMoeDsaDecoderLayer(cfg, args.layer)
    layer = layer.to_empty(device="cpu")

    state = load_layer_state(args.model, args.layer, cfg.num_local_experts)
    missing, unexpected = layer.load_state_dict(state, strict=False)
    print(f"missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing sample:", missing[:5])
    if unexpected:
        print("  unexpected sample:", unexpected[:5])

    # --- _keep_in_fp32_modules, replicated by hand -------------------------
    # M:643 lists "indexer.weights_proj"; M:641 lists "e_score_correction_bias"
    # (strict). Both are from_pretrained-time behaviours. This script never goes
    # through from_pretrained -- it builds the layer on `meta` and calls
    # load_state_dict -- so NEITHER runs here, and a bf16 weights_proj would put
    # the oracle in a dtype layout no deployment ever uses. Task 1 flagged this
    # as invisible below 2048 and not invisible at Task 5.
    ix = getattr(layer.self_attn, "indexer", None)
    if ix is not None:
        want = torch.float32 if args.weights_proj_dtype == "fp32" else torch.bfloat16
        ix.weights_proj.to(want)
        print(f"indexer.weights_proj dtype forced to {ix.weights_proj.weight.dtype} "
              f"(_keep_in_fp32_modules M:643 replicated by hand; "
              f"{'DEPLOYED layout' if want == torch.float32 else 'MEASUREMENT ONLY'})")
        assert ix.weights_proj.weight.dtype == want
    if hasattr(layer.mlp, "gate"):
        b = layer.mlp.gate.e_score_correction_bias
        print(f"router e_score_correction_bias dtype {b.dtype} (must be float32)")
        assert b.dtype == torch.float32, "e_score_correction_bias must stay fp32 (M:641)"
    layer.eval()

    H = cfg.hidden_size
    # x is generated on CPU with a fixed seed so that --device cpu and
    # --device cuda see byte-identical inputs.
    x = torch.randn(1, args.seq, H, dtype=torch.bfloat16) * 0.02

    dev = torch.device(args.device)
    if dev.type != "cpu":
        layer = layer.to(dev)
        x = x.to(dev)
    print(f"device: {dev}")

    pos = torch.arange(args.seq, device=dev).unsqueeze(0)
    rotary = GlmMoeDsaRotaryEmbedding(cfg).to(dev)
    position_embeddings = rotary(x, pos)

    # Layer args.layer's indexer is "shared" for GLM-5.2 layers past the first
    # few (config.indexer_types[layer] == "shared" for layer 3): it has no
    # indexer of its own and reuses a previous "full"-indexer layer's top-k
    # selection via prev_topk_indices. Loading that other layer just to get a
    # real selection is unnecessary here: at seq_len(=args.seq) <=
    # index_topk(=2048) the settled-fact selection is "all causally valid
    # tokens" (Task 3), and GlmMoeDsaAttention.forward ORs the causal mask on
    # top of whatever indices are supplied regardless of content
    # (`index_mask | (key_positions > position_ids)`), so any prev_topk_indices
    # that names every position for every query is exactly equivalent to what
    # a real prior full indexer would produce at this trivial sequence length.
    #
    # ABOVE index_topk that equivalence is GONE -- the selection is a real
    # 2048-of-T subset and a "shared" layer's answer depends on WHICH subset its
    # leader picked. --prev-topk supplies it; refusing to guess is the point.
    is_shared = (getattr(cfg, "indexer_types", None) is not None
                 and cfg.indexer_types[args.layer] == "shared")
    if is_shared and args.prev_topk:
        pt = np.load(args.prev_topk)
        assert pt.shape[0] == 1 and pt.shape[1] == args.seq, (
            f"--prev-topk has shape {pt.shape}, expected (1, {args.seq}, topk)")
        prev_topk_indices = torch.from_numpy(pt.astype(np.int32)).to(dev)
        print(f"prev_topk_indices loaded from {args.prev_topk} {tuple(pt.shape)}")
    elif is_shared:
        if args.seq > cfg.index_topk:
            raise SystemExit(
                f"layer {args.layer} is \"shared\" and --seq {args.seq} > index_topk "
                f"{cfg.index_topk}: the arange stand-in is only equivalent BELOW "
                f"index_topk. Dump the group leader (layer "
                f"{max(i for i in range(args.layer + 1) if cfg.indexer_types[i] == 'full')}) "
                f"first and pass --prev-topk <that>/layer<L>_topk_indices.npy.")
        prev_topk_indices = (
            torch.arange(args.seq, dtype=torch.int32, device=dev)
            .view(1, 1, args.seq)
            .expand(1, args.seq, args.seq)
        )
    else:
        prev_topk_indices = None

    caught = {}

    def hook_out(name, index=None, cast=torch.float32):
        def fn(_m, _inp, out):
            t = out[index] if index is not None else (out[0] if isinstance(out, tuple) else out)
            t = t.detach()
            caught[name] = (t.to(cast) if cast is not None else t).cpu().numpy()
        return fn

    def hook_pre(name):
        def fn(_m, inp):
            caught[name] = inp[0].detach().to(torch.float32).cpu().numpy()
        return fn

    # q_resid, kept on-device: it is the traced indexer's second input (M:409).
    live = {}

    def hook_live(name):
        def fn(_m, _inp, out):
            live[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return fn

    is_moe_layer = hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "shared_experts")
    handles = [
        layer.input_layernorm.register_forward_hook(hook_out("post_input_norm")),
        layer.input_layernorm.register_forward_hook(hook_live("post_input_norm_dev")),
        layer.self_attn.register_forward_hook(hook_out("attn_out")),
        layer.post_attention_layernorm.register_forward_pre_hook(hook_pre("post_attn_hidden")),
        layer.post_attention_layernorm.register_forward_hook(hook_out("post_post_norm")),
        layer.mlp.register_forward_hook(hook_out("moe_out")),
        layer.self_attn.q_a_layernorm.register_forward_hook(hook_live("q_resid")),
    ]
    if is_moe_layer:
        # router_logits / topk_idx / topk_w / shared_out only exist for a
        # sparse (MoE) layer's mlp (GlmMoeDsaMoE); a dense layer's mlp is a
        # plain GlmMoeDsaMLP with no router or shared-expert submodule, so
        # only "moe_out" (really just "mlp output" there) is meaningful.
        handles += [
            layer.mlp.gate.register_forward_hook(hook_out("router_logits", index=0)),
            layer.mlp.gate.register_forward_hook(hook_out("topk_w", index=1)),
            layer.mlp.gate.register_forward_hook(hook_out("topk_idx", index=2, cast=None)),
            layer.mlp.shared_experts.register_forward_hook(hook_out("shared_out")),
        ]

    with torch.no_grad():
        out = layer(
            x,
            position_embeddings=position_embeddings,
            position_ids=pos,
            prev_topk_indices=prev_topk_indices,
        )
    for h in handles:
        h.remove()

    y, topk_indices = out[0], out[1]
    caught["input_hidden"] = x.detach().to(torch.float32).cpu().numpy()
    caught["output_hidden"] = y.detach().to(torch.float32).cpu().numpy()

    extra = {}
    if ix is not None:
        # The layer forward passes attention_mask=None through to the indexer
        # (M:406-415), so the trace must too, or the causal branch differs.
        with torch.no_grad():
            tidx, cap = trace_indexer(ix, live["post_input_norm_dev"],
                                      live["q_resid"], position_embeddings, None, pos)
        same = bool(torch.equal(tidx, topk_indices))
        print(f"traced indexer reproduces the module's indices EXACTLY: {same}")
        if not same:
            raise SystemExit(
                "the traced copy of GlmMoeDsaIndexer.forward disagrees with the "
                "real module -- the copy has drifted from upstream; fix it before "
                "trusting any diagnostic it dumps")
        print(f"index_score row {args.seq - 1}: exactly-zero fraction "
              f"{cap.pop('_zero_frac_row'):.4f}; per-head relu'd scores "
              f"exactly zero {cap.pop('_zero_frac_heads'):.4f}")
        extra.update(cap)

    if topk_indices is not None:
        extra["topk_indices"] = topk_indices.detach().cpu().numpy().astype(np.int32)
        print(f"  topk_indices: {tuple(extra['topk_indices'].shape)} "
              f"dtype={extra['topk_indices'].dtype}"
              f"{' (this layer OWNS the indexer)' if ix is not None else ' (propagated)'}")

    for tag, arr in extra.items():
        np.save(os.path.join(args.out, f"layer{args.layer}_{tag}.npy"), arr)

    for tag in SUBSTEPS:
        if tag not in caught:
            print(f"  {tag}: SKIPPED (dense layer has no router/shared-expert submodule)")
            continue
        arr = caught[tag]
        np.save(os.path.join(args.out, f"layer{args.layer}_{tag}.npy"), arr)
        print(f"  {tag}: {arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
