# DSA indexer + serve mode — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** lift windlass's 2048-token context cap by implementing GLM-5.2's DSA sparse-attention indexer, then expose an OpenAI-compatible endpoint so it can serve real work — specifically, produce a genuine review of a real pull request.

**Why now:** correctness is established below 2048 tokens and throughput is measured. The blocker to useful work is neither — it is that `run_layer` aborts above `index_topk`, so every prompt is a toy prompt.

## Established facts

Verified against the live checkpoint and config, 2026-08-01:

```
index_topk        2048     index_n_heads   32
index_topk_freq   4        index_head_dim  128
kv_lora_rank      512      qk_rope_head_dim 64
max_position_embeddings 1048576
```

**Indexer weights exist on 22 layers**: `0, 1, 2, 6, 10, 14, … 74, 78`. Layer 78 is the MTP head and is out of scope, leaving **21 in-scope indexers**. Five tensors each:

```
self_attn.indexer.{wq_b.weight, wk.weight, k_norm.weight, k_norm.bias, weights_proj.weight}
```

Layers without their own indexer consume the group leader's indices. Groups are `{2,3,4,5}`, `{6,7,8,9}`, … led by the weight-owning layer; layers 0 and 1 own theirs outright.

**Current state:** `glm_loader.cuh` does not read indexer tensors at all — `grep indexer` returns nothing. The abort is `glm_layer_runner.cuh:470`. `ref_glm_chain.py` has the same limit, so the chain has no oracle above 2048 either.

**KV cache cost** is 576 f32 per position per layer (512 `kv_lora` + 64 rope): 1.5 GB at 8k, **5.9 GB at 32k**, 23.6 GB at 128k. Above ~32k the cache eats the expert pool, so 32k is the practical target.

**Prefill is already batched** — `infer_glm.cu:396` runs the whole prompt in one `run_chain` call.

## The free oracle

Below `index_topk`, the sparse path must equal the dense path **exactly** — not within a tolerance. Any divergence at seq ≤ 2048 is a bug with no judgement call involved. This is a stronger gate than anything used so far and it costs nothing.

Task 1 verified this against the source and corrected the mechanism: it is **not** that the indexer selects every causally-valid token. `topk = min(index_topk, T)`, so at `T ≤ 2048` the top-k returns a permutation of **all** T indices, `index_mask` is uniformly `False`, and the mask write is a no-op — bit-identical regardless of what the indexer computed. That makes the gate stronger *and* narrower than assumed: it proves the plumbing, and proves nothing about the indexer's arithmetic. See Task 1.

Above 2048, `dump_glm_oracle.py` builds a real `GlmMoeDsaDecoderLayer` from checkpoint weights; raising `--seq` past 2048 makes transformers compute true sparse attention. That is the ground truth for the indexer itself.

---

### Task 1: Extract the indexer specification from source

**DONE** — 2026-08-01. Source read: `transformers 5.14.1`,
`site-packages/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` (809 lines; generated from
`modular_glm_moe_dsa.py`, which is not what runs). Citations below are `modeling_glm_moe_dsa.py:LINE`
unless stated otherwise. Shapes and dtypes cross-checked against the live checkpoint
(`/home/user1/glm52-mxfp4/model.safetensors.index.json`); the ≤`index_topk` equivalence claim was
verified by executing the real `GlmMoeDsaIndexer` class, not by reading alone.

#### Answers

1. **`wq_b`** — `nn.Linear(q_lora_rank=2048 → index_n_heads*index_head_dim = 32*128 = 4096, bias=False)`
   (`:197`). It consumes **exactly the same tensor** the main attention path uses: `q_resid =
   q_a_layernorm(q_a_proj(hidden_states))` is computed once at `:385` and passed into the indexer at
   `:409`. There is no separate query projection — `wq_b` is a second head off the shared q-LoRA
   residual. Checkpoint shape confirms `[4096, 2048]`, bf16, unquantized.
2. **`wk`** — `nn.Linear(hidden_size=6144 → index_head_dim=128, bias=False)` (`:198`), applied to the
   **raw hidden state** entering the attention block (i.e. the `input_layernorm` output, `:603` → `:605`
   → `:408`), *not* the compressed KV and *not* `q_resid`. Single head: output is `[B,S,128]`,
   unsqueezed to `[B,S,1,128]` (`:236`). RoPE **is** applied, to the **first `qk_rope_head_dim`=64 of
   the 128 dims** (`:237`), interleaved (`apply_rotary_pos_emb_interleave`, `:240`), sharing the same
   `(cos,sin)` table as the main path — 64-dim table, `rope_theta = 8e6`. The trailing 64 dims are
   pass-through. Same split applies to `q` (`:234`).
3. **`k_norm`** — `nn.LayerNorm(128, eps=1e-6)` (`:199`). **LayerNorm, not RMSNorm** — mean-subtracting,
   with `elementwise_affine=True` giving both `weight` and `bias` (hence the fifth tensor). The eps is
   **explicitly passed as `1e-6`** at construction; `config.rms_norm_eps` (1e-5) is **not** used here.
   This is the same shape of trap as `q_a_layernorm`/`kv_a_layernorm`, but the opposite mechanism: those
   take the class default because no eps is passed, this one takes a hardcoded literal.
4. **`weights_proj`** — `nn.Linear(hidden_size=6144 → index_n_heads=32, bias=False)` (`:200`), producing
   one scalar gate per indexer head per query token. Combination (`:247`–`:252`):
   `scores[b,s,h,t] = relu( (q[b,s,h,:] · k[b,t,:]) * 128**-0.5 )`, then
   `index_score[b,s,t] = Σ_h weights[b,s,h] * scores[b,s,h,t]` with
   `weights = weights_proj(x) * 32**-0.5`. So it is a weighted sum over heads **after** ReLU — one
   score per (query,key) pair, heads are not selected independently.
5. **Top-k** — `index_scores.topk(topk, dim=-1).indices.to(torch.int32)` with
   `topk = min(index_topk, T)` (`:262`–`:263`). `sorted=` is **not passed**, so it defaults to `True`
   (descending) — but the ordering is **irrelevant**: the indices are only ever scattered into a bool
   mask (`:423`–`:427`), so only the selected *set* matters. Selection is **shared across all 32 heads**
   (it happens after the head combination in (4)), and shared across all 64 *attention* heads too.
   **Ties are not deterministic** — see Ambiguities.
6. **`index_mask`** (`:423`–`:432`): `ones([B,S,T], bool).scatter_(-1, topk_indices, False)` — `True`
   means *drop*. Composition with causal: `attention_mask.masked_fill(index_mask, finfo(dtype).min)`.
   The causal mask is the base and is **preserved** — `masked_fill` only writes `min` where index_mask
   is True, so a causally-forbidden key stays `min` whether or not the indexer named it. When the causal
   mask is `None` (SDPA fast path), causality is re-ORed explicitly (`:430`). **Selected indices can be
   non-causal** — verified empirically, see Findings — so a CUDA implementation must apply causality
   independently and must never treat the index set as already causal.
7. **`prev_topk_indices`** — a single variable threaded through the layer loop:
   `GlmMoeDsaModel.forward` initialises `topk_indices = None` (`:715`), passes it as `prev_topk_indices`
   into layer *i* (`:724`), and overwrites it with that layer's return (`:717`). The decoder layer
   returns whatever the attention returned (`:605`, `:621`); the attention returns `topk_indices` as its
   third value (`:453`). **Capture point:** the value returned is exactly the indexer's return
   (`:408`–`:415`) — captured *before* `index_mask` is built, so it is raw indices, unfiltered, with the
   non-causal junk still in it. A `"shared"` layer passes its input through unchanged (`:417`–`:419` →
   `:453`), so propagation continues across the whole group rather than stopping at the first consumer.
   A `"shared"` layer with `prev_topk_indices is None` raises (`:418`).

#### Layer ownership

`config.indexer_types` is **explicit in the checkpoint's `config.json`** (78 entries) — do not derive it.
It happens to equal the `__post_init__` formula
(`configuration_glm_moe_dsa.py:145`–`:149`, `freq = index_topk_freq = 4`,
`offset = index_skip_topk_offset = 3`):

```python
indexer_types[i] = "full" if (max(i - 3 + 1, 0) % 4) == 0 else "shared"
# -> full at 0, 1, 2, 6, 10, 14, ..., 74   (21 layers; verified equal to config.json)
```

Layer 78 owns indexer tensors in the checkpoint but has no `indexer_types` entry (MTP head, dropped by
`_keys_to_ignore_on_load_unexpected` at `:642`). Per Task 2, read the ownership map from the checkpoint
/ `config.json`, not from the formula.

#### Reference implementation (exact, transcribe this)

```python
# Per "full" layer. x = input_layernorm(hidden) [B,S,6144] bf16
#                   q_resid = q_a_layernorm(q_a_proj(x)) [B,S,2048] bf16  -- SHARED with main path
# H = 32, D = 128, R = 64 (rope dims), T = total key length (S + past)
SOFTMAX_SCALE = 128 ** -0.5      # 0.08838834764831845
HEAD_SCALE    = 32  ** -0.5      # 0.17677669529663687
LN_EPS        = 1e-6             # LayerNorm, NOT rms_norm_eps

# ---- query side ------------------------------------------------------- :232-234
q = wq_b(q_resid)                      # bf16 [B,S,4096]
q = q.view(B, S, 32, 128)              # [B,S,H,D]
q_rot, q_pass = q[..., :64], q[..., 64:]

# ---- key side --------------------------------------------------------- :236-237
k = wk(x)                              # bf16 [B,S,128]
# LayerNorm over the 128 dims: mean-subtract, /sqrt(var + 1e-6), *weight + bias
mu  = k.mean(-1, keepdim=True)
var = ((k - mu) ** 2).mean(-1, keepdim=True)      # biased (1/N) variance
k   = (k - mu) / sqrt(var + LN_EPS) * k_norm_w + k_norm_b
k_rot, k_pass = k[..., :64], k[..., 64:]

# ---- interleaved RoPE over the 64-dim slice --------------------------- :240, :161-168
# cos/sin are the FIRST HALF of the 64-wide cat(freqs,freqs) table -> 32 angles.
# pairs are (0,1),(2,3),...  BUT the output is written de-interleaved:
#   out[0:32]  = even * cos - odd * sin
#   out[32:64] = odd  * cos + even * sin
# This is the same helper the main attention path uses (:396) -- identical layout.
q_rot = rope_interleave(q_rot, cos, sin)
k_rot = rope_interleave(k_rot, cos, sin)
q = cat([q_rot, q_pass], -1)           # [B,S,H,128]
k = cat([k_rot, k_pass], -1)           # [B,S,128]

# ---- indexer KV cache -------------------------------------------------- :244-245
# Separate from the MLA cache. Stores the POST-norm, POST-RoPE k, one 128-vector
# per position per OWNING layer, in k's dtype (bf16). 21 owners -> 21*128*2 B/pos
# = 5376 B/pos = 176 MB at 32k. Independent of the 576-f32 MLA cache.
k_all = indexer_cache.append(k)        # [B,T,128]

# ---- scores ------------------------------------------------------------ :247-252
scores = relu( (q.float() @ k_all.float().transpose(-1,-2)) * SOFTMAX_SCALE )  # fp32 [B,S,H,T]
w      = weights_proj(x.float()).float() * HEAD_SCALE                          # fp32 [B,S,H]
index_scores = einsum('bsh,bsht->bst', w, scores)                              # fp32 [B,S,T]

# ---- causality + top-k -------------------------------------------------- :255-263
index_scores += attention_mask          # additive, finfo(bf16).min at forbidden keys
topk = min(2048, T)
topk_indices = index_scores.topk(topk, dim=-1).indices.to(int32)   # sorted=True (default)

# ---- mask ---------------------------------------------------------------- :423-432
index_mask = ones([B,S,T], bool); index_mask.scatter_(-1, topk_indices, False)
attn_mask  = causal_mask.masked_fill(index_mask, finfo(bf16).min)  # causal is the BASE
```

#### Findings that would have become silently-wrong CUDA

- **`k_norm` is a LayerNorm with a hardcoded `eps=1e-6`** (`:199`). Implementing it as the repo's
  existing RMSNorm — or reusing `rms_norm_eps = 1e-5` — is wrong twice over. It mean-subtracts and it
  adds a bias.
- **`wk` reads the hidden state, `wq_b` reads `q_resid`.** They are different inputs of different widths
  (6144 vs 2048). The indexer is not a self-contained projection pair off one tensor.
- **`weights_proj` is forced to fp32** by `_keep_in_fp32_modules = ["indexer.weights_proj"]` (`:643`),
  and its input is explicitly upcast (`:251`, `hidden_states.to(self.weights_proj.weight.dtype)`). Every
  other indexer projection runs in bf16. Same class of trap as the router's fp32 `e_score_correction_bias`.
- **Two scale factors, not one.** `128**-0.5` on the qk product (`:201`, `:247`) and `32**-0.5` on the
  head weights (`:251`). Missing the second scales every index score by 5.66× — harmless under a pure
  top-k, but it changes nothing only if you never add the mask; the additive mask makes the magnitude
  matter for tie/ordering behaviour near the boundary.
- **ReLU sits between the qk product and the head combination** (`:248`), not after it. Combining first
  and rectifying after is a different function.
- **The indexer's RoPE is interleaved**, matching the main path (`:240` vs `:396`) — but note it applies
  to dims `[0:64)` of the *indexer* 128-dim head, whereas the main path's rope slice is the *trailing*
  64 of its 256-dim head (`:387`, `q_pass, q_rot = split([192, 64])`). The slice ends are opposite.
  Getting this backwards produces plausible-looking garbage.
- **Selected indices are not guaranteed causal.** Verified by running the real class: at `S=16,
  index_topk=4`, query row 2 (whose only causal keys are 0,1,2) returned `[0,1,2,10]` — index 10 is a
  filler drawn from the `-inf` tail. Correctness survives only because the causal mask is the base of
  the `masked_fill` (`:432`) / is re-ORed (`:430`). CUDA must do the same; it must not skip the causal
  test on the grounds that the indexer already applied one.
- **The indexer has its own KV cache**, holding post-norm post-RoPE 128-dim keys
  (`cache_utils.py:271`–`:302`, `DynamicIndexedLayer.update_indexer`), stored in bf16, only for owning
  layers. It is *not* derivable from the MLA cache — `wk` reads the hidden state, which is not retained.
  Budget it separately: 5376 B/position total across the 21 owners.
- **The oracle dump path and the real model may disagree on `weights_proj` precision.**
  `_keep_in_fp32_modules` is a `from_pretrained`-time behaviour; `tools/dump_glm_oracle.py:163`–`:169`
  constructs the layer directly on `meta` and `load_state_dict`s into it, so the whole layer ends up at
  the default dtype rather than bf16-with-fp32-`weights_proj`. Below 2048 this is invisible (the mask is
  a no-op); at Task 5 it is not. Check the dumped dtypes before deriving a tolerance.

#### The ≤ `index_topk` free oracle — VERIFIED, but for a different reason than the plan states

The plan says the indexer "selects every causally-valid token". That is **not** the mechanism. The
mechanism is `topk = min(self.index_topk, index_scores.shape[-1])` (`:262`): when `T ≤ 2048`, `topk = T`,
so `.topk(T)` returns a **permutation of all T indices** — every key, causal or not. `index_mask` is then
uniformly `False` and `masked_fill` writes nothing.

Executed against the real `GlmMoeDsaIndexer` at `S=16, index_topk=2048`:

```
indices shape (1, 16, 16)
index_mask any-True (anything masked out): False
resulting mask bit-identical to dense causal mask: True
```

**The claim holds, in a stronger form than stated**: the attention mask is not merely equivalent to the
dense one, it is the *same object contents*, bit-for-bit, independent of the indexer's weights and
arithmetic. Task 4's bit-identity gate is sound. The corollary is that this gate proves **nothing about
the indexer's arithmetic** — a completely wrong `k_norm` eps or a missing `relu` still passes it. It
gates the *plumbing* (mask composition, index sharing, the removal of the abort); Task 5's 4096-token
oracle is the first test that touches the numbers. Do not let a green Task 4 create confidence in Task 3.

#### Prefill / decode / caching

The indexer runs on **every forward of every `"full"` layer** — prefill and decode alike, unconditionally
(`:406`–`:415`); there is no fast path that skips it. **Indices are recomputed for every query position
on every call and are never cached**; only the 128-dim indexer *keys* are cached (`:245`). At decode
(`S=1`) the cost is one `[1,32,128] @ [128,T]` product per owning layer plus a top-2048 over `T` — the
top-k, not the matmul, is the decode-time cost to watch.

#### Ambiguities — flagged, not invented

- **Tie-breaking in `topk` is not specified.** After `relu`, a large fraction of scores are exactly `0.0`;
  above 2048 keys, the selected set is genuinely under-determined and CUDA cannot be expected to match
  torch's choice index-for-index. Tasks 5/6 must compare **attention outputs**, or the selected set
  restricted to strictly-positive scores — never the raw index list. This has not yet been measured on
  real weights; measure the fraction of zero scores at 4096 before designing the Task 5 comparison.
- **The masked-entry ordering differs between the two mask paths.** With an explicit causal mask the
  code *adds* `finfo.min` (`:256`), so forbidden keys retain their relative order by pre-mask score; with
  `attention_mask is None` it `masked_fill`s `-inf` (`:260`), making them exactly equal. Which junk
  indices fill the tail therefore depends on the path. Harmless for output (all are re-masked), fatal if
  a test asserts on indices.
- **`LayerNorm` internal accumulation precision on bf16 input is torch/backend-defined.** The spec above
  assumes fp32 accumulation with a bf16 result, which is what ATen does on CUDA, but this was not
  measured. If Task 5 shows a `k_norm`-shaped residual, this is the first thing to check.
- **`index_topk` vs. the MLA key length are assumed equal** (`index_scores.shape[-1]` at `:262` vs
  `key_states.shape[2]` at `:424`). They come from two different caches. Equal in every path examined,
  but nothing enforces it; assert it in CUDA rather than assuming.

- [x] **Step 1:** Read the source, record findings with citations.
- [x] **Step 2:** Write the CUDA-level spec into this document.
- [x] **Step 3:** Commit.

---

### Task 2: Load the indexer weights

**Files:** `src/glm_loader.cuh`

Extend `LayerWeights` with the five indexer tensors, loaded only for the 21 owning layers; null elsewhere. Add an `indexer_owner[78]` map derived from the checkpoint rather than hardcoded, so a different model's layout does not silently mis-load.

- [ ] Load, with dtype and shape assertions per tensor.
- [ ] `free_layer` releases them.
- [ ] A loader test asserts: 21 owners, correct group leader for every non-owning layer, layer 78 never loaded.

---

### Task 3: Indexer forward and top-k

**Files:** `src/glm_kernels.cuh`, `src/glm_layer_runner.cuh`

Implement per Task 1's spec. Scores over 32 heads × 128 dim, combined via `weights_proj`, top-`index_topk` selected.

- [ ] Kernel plus launcher, following the existing block-reduction conventions.
- [ ] Unit test against a numpy reimplementation on random inputs, before any real weights.

---

### Task 4: IndexShare and the attention mask

**Files:** `src/glm_layer_runner.cuh`

Group leaders compute indices and cache them; members consume them. Apply `index_mask` in MLA attention, composed with the causal mask.

- [ ] Remove the `T > index_topk` abort.
- [ ] **Free-oracle gate:** at seq ≤ 2048 the sparse path must be **bit-identical** to the dense path. Not "within tolerance" — identical. Assert it in `test_glm_layer`.

---

### Task 5: Long-context single-layer oracle

**Files:** `tools/dump_glm_oracle.py`, `src/test_glm_layer.cu`

- [ ] Dump layer 2 (an indexer owner) and layer 3 (a consumer) at `--seq 4096`.
- [ ] Compare CUDA against it. Derive the tolerance from measured baseline as Task 10 did; do not import the ≤2048 figure.
- [ ] Negative controls: perturb the top-k selection, the head combination, and the k_norm eps. Each must be caught, with separation reported.

---

### Task 6: Indexer in the numpy reference

**Files:** `tools/ref_glm_chain.py`, `tools/check_ref_vs_oracle.py`

Without this the full chain has no oracle above 2048.

- [ ] Implement, validate against the transformers layer oracle at 4096.
- [ ] Extend `--negative-controls` with the three indexer defects from Task 5.

---

### Task 7: Full chain at long context

- [ ] Regenerate fixtures with a ~1400-token real prompt (a PR diff plus review instructions).
- [ ] Per-substep gating at layers 0/2/3/40/77 — note **layer 2 is added** because it is an indexer owner.
- [ ] Top-5 exact at all generated tokens.

---

### Task 8: Measure prefill on a realistic prompt — decision point

- [ ] Measure prefill separately from decode on the ~1400-token prompt, with `--timing`.
- [ ] Report seconds-to-first-token and total time for a 600-token completion.

**This decides whether the serve mode is worth building.** If a review costs 8 minutes, windlass is usable for batch work; at 40 minutes it is a demonstration. Report the number before proceeding.

---

### Task 9: Serve mode

**Files:** `src/serve_glm.cu` or `--serve` in `infer_glm.cu`

Port the pattern from `infer.cu:2579-3400` — OpenAI-compatible `/v1/chat/completions`, SSE streaming, `/v1/models`, `/health`. Tool calling is **not** required for the benchmark; skip it unless free.

- [ ] Single in-flight request; return 503 when busy rather than queueing silently.
- [ ] Honour `max_tokens`, and fail loudly if `prompt + max_tokens` exceeds the configured `max_seq`.

---

### Task 10: Run the PR review benchmark

- [ ] Add windlass to `bench_code_review.py`'s `MODELS`, pointing at the serve endpoint.
- [ ] Run against the three PRs it already uses.
- [ ] Report reviews alongside the other models' output, and state the context and `max_tokens` used so the comparison is honest.

**Acceptance:** windlass produces a coherent, substantive review of a real pull request — the thing it could not do before.

## Constraints

- No `printf` in device kernels — corrupts output on SM 12.0.
- Do not modify existing arithmetic outside the indexer path; `test_glm_chain` must still show top-5 exact and worst substep ≤1e-5 at short context after every change.
- Model weights, packed experts and `glm-ref/` are never deleted.
- `vllm-qwen36` may be stopped for GPU work; restart it immediately after, and confirm it serves.
- Commits carry no AI attribution and no co-author trailer.
