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
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaDecoderLayer,
    GlmMoeDsaRotaryEmbedding,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
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
    layer.eval()

    H = cfg.hidden_size
    x = torch.randn(1, args.seq, H, dtype=torch.bfloat16) * 0.02

    pos = torch.arange(args.seq).unsqueeze(0)
    rotary = GlmMoeDsaRotaryEmbedding(cfg)
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
    if getattr(cfg, "indexer_types", None) is not None and cfg.indexer_types[args.layer] == "shared":
        prev_topk_indices = (
            torch.arange(args.seq, dtype=torch.int32)
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
            caught[name] = (t.to(cast) if cast is not None else t).numpy()
        return fn

    def hook_pre(name):
        def fn(_m, inp):
            caught[name] = inp[0].detach().to(torch.float32).numpy()
        return fn

    is_moe_layer = hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "shared_experts")
    handles = [
        layer.input_layernorm.register_forward_hook(hook_out("post_input_norm")),
        layer.self_attn.register_forward_hook(hook_out("attn_out")),
        layer.post_attention_layernorm.register_forward_pre_hook(hook_pre("post_attn_hidden")),
        layer.post_attention_layernorm.register_forward_hook(hook_out("post_post_norm")),
        layer.mlp.register_forward_hook(hook_out("moe_out")),
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

    y = out[0] if isinstance(out, tuple) else out
    caught["input_hidden"] = x.detach().to(torch.float32).numpy()
    caught["output_hidden"] = y.detach().to(torch.float32).numpy()

    for tag in SUBSTEPS:
        if tag not in caught:
            print(f"  {tag}: SKIPPED (dense layer has no router/shared-expert submodule)")
            continue
        arr = caught[tag]
        np.save(os.path.join(args.out, f"layer{args.layer}_{tag}.npy"), arr)
        print(f"  {tag}: {arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
