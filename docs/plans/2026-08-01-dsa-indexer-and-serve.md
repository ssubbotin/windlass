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

Below `index_topk`, the indexer selects **every causally-valid token**, so the sparse path must equal the dense path **exactly** — not within a tolerance. Any divergence at seq ≤ 2048 is a bug with no judgement call involved. This is a stronger gate than anything used so far and it costs nothing.

Above 2048, `dump_glm_oracle.py` builds a real `GlmMoeDsaDecoderLayer` from checkpoint weights; raising `--seq` past 2048 makes transformers compute true sparse attention. That is the ground truth for the indexer itself.

---

### Task 1: Extract the indexer specification from source

The Task 4b pattern, which caught RMSNorm form, router scaling and the dead `head_dim` before any of them became wrong CUDA. Read `transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py` and write the indexer's forward path into this plan as complete code.

**Must be answered with file:line citations:**
- What `wq_b` projects from and to; whether it consumes the same `q_a_layernorm` output as the main attention path or a separate projection.
- What `wk` projects from — the compressed KV, the raw hidden state, or something else — and whether RoPE is applied to it.
- `k_norm`: it has a **bias**, unlike every other norm in this model. Is it RMSNorm or LayerNorm? Which eps?
- What `weights_proj` produces and how per-head scores combine into one score per (query, key) pair.
- The exact top-k call: `sorted=` argument, tie behaviour, and whether selection is per-head or shared across heads.
- How the resulting indices become `index_mask`, and how that composes with the causal mask.
- How a non-owning layer receives `prev_topk_indices`, and at what point in the group leader's forward the indices are captured.

Flag anything genuinely ambiguous rather than inventing it.

- [ ] **Step 1:** Read the source, record findings with citations.
- [ ] **Step 2:** Write the CUDA-level spec into this document.
- [ ] **Step 3:** Commit.

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
