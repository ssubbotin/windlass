# Measurements

All figures from a single machine: RTX PRO 6000 Blackwell (96 GB, SM 12.0), 125 GB system RAM, Samsung 9100 PRO 4 TB NVMe (PCIe 5.0), CUDA 13.1. Model is GLM-5.2 at MXFP4 (`amd/GLM-5.2-MXFP4`), 753B total / 39B active.

## Memory budget, measured not estimated

The checkpoint's 438 GB partitions as:

| | measured |
|---|---|
| Routed experts (75 MoE layers × 256) | 385.0 GB |
| MTP head, layer 78 — never loaded | 19.9 GB |
| Resident (attention, embeddings, `lm_head`, shared experts, indexers) | **33.0 GB** |

One routed expert is 3 × 6144 × 2048 params = 20,054,016 bytes at MXFP4 (block-32: 16 payload bytes + 1 E8M0 scale byte per 32 values). Per token: 8 experts × 75 layers = **600 expert reads, ~12 GB nominal**.

Runtime allocation: 32.1 GB resident weights, **62.05 GB expert cache = 3,094 experts (16.1% residency)**, peak VRAM 95.53 / 101.97 GB.

## Correctness

Validated bottom-up, each stage against an independent reference.

| stage | reference | result |
|---|---|---|
| MXFP4 dequant | numpy, 8 seeds × 2 shapes | worst term-relative **4.74e-08** (bar 1e-5) |
| Nibble order | `transformers.integrations.mxfp4._convert_moe_packed_tensors` | exact **0.0**; wrong order differs by 2.03e-01 |
| Single layer (dense + MoE) | `transformers` `GlmMoeDsaDecoderLayer` | **≤2.24e-08** vs an fp32 rebuild of the oracle |
| Full 78-layer chain | numpy reference streaming the same packed files | worst substep **1.6928e-06** (`attn_out` L77), gate 1e-5 |
| End-to-end | greedy decode, self-fed argmax | **top-5 exact at all 5 tokens, in rank order** |

The chain reference itself was validated behaviourally: prompted with

```
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
```

it generates `    left = [x` — the opening of `left = [x for x in arr if x < pivot]`.

### Negative controls

A check that has never failed is not evidence. Each defect was injected into a copy and measured:

| injected defect | worst substep | vs. 1e-5 gate |
|---|---|---|
| MXFP4 nibble order flipped | 1.86e+00 | 1.9e+05× |
| `routed_scaling_factor` 1.0 instead of 2.5 | 9.45e-01 | 9.5e+04× |
| LoRA-norm eps 1e-5 instead of 1e-6 | 3.38e-01 | 3.4e+04× |

**The third defect produced the exactly correct top-5 tokens** while corrupting layer-40 and layer-77 expert selection. An end-to-end token gate would have shipped it; per-substep comparison caught it 36 times over.

Measured at the reference level, a final-hidden-state comparison catches **1 of 7** injected defects; per-substep catches **6 of 7**. The residual stream dilutes any single block's contribution by 21–56×.

## Throughput

Eight identical requests in one process, 60 tokens each, warm:

```
1.161  1.189  1.228  1.227  1.226  1.226  1.230  1.227 tok/s
```

Baseline binary in the same window: **0.901 tok/s flat**. Honest speedup **1.36×**.

### Measurement discipline

Page-cache state moves under you when the working set is 359 GB against 125 GB of RAM. The same binary measured **0.850 and 1.003 tok/s twenty minutes apart** — 18% drift, enough to invalidate a naive before/after pair. All deltas below come from an ABBA ordering with a closing baseline; a first pass was discarded when its closing baseline disagreed with its opening one.

Protocol identity was confirmed by measurement rather than assertion: baseline and optimised runs produced byte-identical cache counters (`hits=222125 misses=171475 evictions=168382`).

### Where the time goes

| | share of layer time |
|---|---|
| Expert fetch | **87.6%** (96.5% before optimisation) |
| Attention | 1.5% |
| Expert compute | 1.5% |
| Router | 0.4% |

If fetching were free, the compute floor would be ~18 tok/s. The deficit is entirely I/O.

### Optimisation deltas

| step | warm tok/s | delta | verdict |
|---|---|---|---|
| baseline | 0.913 | — | |
| pinned host staging | 0.968 | +6.0% | keep |
| double-buffered overlap | 1.032 | +6.6% | keep |
| batched issue (4 workers) | 1.346 | **+30.4%** | keep |
| 8 workers | 1.359 | +0.5% | discard, not significant |

The ranking is the opposite of the intuitive one. Pinned staging was expected to dominate and delivered least: the host-to-device copy was never the bottleneck. The pre-optimisation "3.6 GB/s" was a *read* rate at queue depth 1.

## Why it stops here

Two independent ceilings.

**The cache hit rate plateaus at 56.5%** — reached by request 2, with no warm-up curve (1.6% improvement across 8 requests). For comparison, a 397B model on a 24 GB card reached ~95% at 8% residency. Whether this is diffuse routing or the wrong eviction policy is **not established**: no residency sweep, no LRU/LFU/Belady comparison, and no access-trace analysis were run. The observation is solid; the causal attribution is not.

**The fetch path is bounded by a dependency chain, not by bandwidth.** Layer L+1's router cannot run until layer L completes, so outstanding reads are capped by the misses in the current layer — 261 misses/token ÷ 75 layers = **3.5 on average**, 8 at most. Four workers drain that queue; eight have nothing to issue. That is what `io8 == io4` measures.

Note the device ceiling itself was **never measured** — no `fio`, no `iostat`, and GPUDirect Storage was not tried (it bypasses the page cache, which serves roughly a third of this pool). Achieved throughput was 6.42 GB/s against the 10.47 GB/s that 2 tok/s would require. "The SSD is saturated" would be a claim beyond the evidence; "the pipeline cannot queue deeply enough to saturate it" is what the data supports.

## Independent corroboration

[Colibri](https://github.com/uv-genai/colibri) — pure C, CPU-first, same model, same technique, different architecture entirely — reports **0.05–1.23 tok/s**, peak 1.23 on AVX-512 with PCIe Gen5 storage. This engine measures **1.227 tok/s**.

Colibri also implements the speculative cross-layer prefetch named above as an escape, reporting **71.6% predictability** of the true top-8 by applying layer L+1's router to layer L's post-attention state — and finds it "remains neutral on well-saturated disk systems."

Two independent implementations converging on the same number, and an independent negative result on the most obvious escape, is stronger evidence than either alone.
