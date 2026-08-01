# windlass

**Run mixture-of-experts language models larger than your GPU, with routed experts streamed from NVMe.**

A single-GPU CUDA inference engine for MoE models whose weights do not fit in VRAM. Dense weights stay resident on the card; routed experts live on an SSD and are hauled up on demand through an LRU cache in VRAM. Pure C/CUDA, no framework dependencies.

Currently runs **GLM-5.2** (753B total / 39B active, MXFP4) on a single 96 GB card:

```
 32 GB  dense weights, resident in VRAM
 62 GB  LRU cache, ~3,094 of 19,200 routed experts
385 GB  routed experts on NVMe, fetched per layer per token
```

## Status: research engine, not a serving stack

Correctness is established. Throughput is not competitive, and the measurements below explain why — that result is the main thing this repository has to offer.

| | measured |
|---|---|
| Correctness | top-5 token ids exact vs. a numpy reference; worst substep **1.69e-06** against a derived 1e-5 gate |
| Single layers | **≤2.2e-08** against a `transformers` oracle |
| Throughput | **1.227 tok/s** warm (RTX PRO 6000 Blackwell, Samsung 9100 PRO) |
| Expert-cache hit rate | **56.5%** at 16.1% residency |

See [docs/RESULTS.md](docs/RESULTS.md) for the full measurements, the negative controls, and the throughput analysis.

## The finding

Expert fetch is **87.6% of layer time**. The obvious conclusion is "buy a faster SSD," and it is wrong.

Expert selection is **data-dependent per layer**: layer L+1's router cannot run until layer L finishes. A decode step therefore never has more than 8 — mean 3.5 — 20 MB reads in flight, regardless of device speed. The pipeline is structurally shallow-queued. Raising the I/O worker count from 4 to 8 changes nothing (+0.5%, not significant) because there is nothing more to issue.

The escapes are speculative cross-layer prefetch, popularity-resident pinning, or moving fewer bytes — not faster storage.

An independent implementation, [Colibri](https://github.com/uv-genai/colibri) (pure C, CPU-first), reports **1.23 tok/s** peak on GLM-5.2. This engine measures **1.227 tok/s** on a different architecture entirely. Two independent implementations converging suggests the wall belongs to the technique, not to either codebase.

## Build

Requires CUDA 12.8+ and a GPU with enough VRAM for the dense weights plus a useful cache.

```bash
make ARCH=sm_120        # Blackwell (RTX PRO 6000, RTX 50xx)
make ARCH=sm_89 tests   # Ada
```

## Use

```bash
# 1. Fetch weights (MXFP4 checkpoint, ~420 GB)
huggingface-cli download amd/GLM-5.2-MXFP4 --local-dir ./glm52-mxfp4

# 2. Repack routed experts into per-layer files with a fixed stride
python3 tools/repack_experts_glm.py --model ./glm52-mxfp4 --out ./packed_experts --layers 3-77

# 3. Verify the packed bytes against the checkpoint (samples 4 experts x 6 sub-tensors per layer)
python3 tools/repack_experts_glm.py --model ./glm52-mxfp4 --out ./packed_experts --layers 3 --verify-only

# 4. Generate
./infer_glm --model-dir ./glm52-mxfp4 --packed ./packed_experts \
            --prompt "def quicksort(arr):" --tokens 40 --io-threads 4
```

`--io-threads` selects the fetch strategy: `0` pinned staging only, `1` double-buffered overlap, `4` batched issue (best measured). Beyond 4 there is nothing left to overlap.

Tokenization is delegated to a Python sidecar using `AutoTokenizer`, so any model with a `tokenizer.json` works without a bespoke exporter.

## Verification

The correctness claims are backed by negative controls — checks are only meaningful if they can fail:

```bash
make tests
./test_glm_mxfp4                                  # MXFP4 dequant, 8 seeds x 2 shapes
./test_glm_expert_cache --packed-dir ./packed_experts --layer 3
./test_glm_layer  --model-dir ./glm52-mxfp4 --oracle glm-oracle --layer 3
./test_glm_chain  --model-dir ./glm52-mxfp4 --packed ./packed_experts --ref glm-ref
python3 tools/check_ref_vs_oracle.py --negative-controls
```

The last command injects known defects and asserts they are caught. Three of them — a flipped MXFP4 nibble order, `routed_scaling_factor` 1.0 instead of 2.5, and the wrong LoRA-norm epsilon — are separated from baseline by 3.4e+04x to 1.9e+05x.

One of those deserves emphasis: **the LoRA-epsilon defect produced the exactly correct top-5 tokens** while getting layer-40 and layer-77 expert selection wrong. An end-to-end token-match gate would have passed it. Only per-substep comparison caught it.

## Scope and limits

- **Contexts ≤2048 tokens.** GLM-5.2's DSA sparse-attention indexer is not implemented; the engine aborts past `index_topk`. Below that threshold the selection is all-tokens and the dense path is numerically exact.
- Single GPU. No tensor/pipeline parallelism.
- Greedy decode. No batching, no serving API.
- One model family so far.

## Licence

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Sergey Subbotin.

Model weights are **not** covered by this licence and are not distributed here. GLM-5.2's weights are released by Z.ai under MIT; the MXFP4 quantisation used above is published by AMD.

This is an independent implementation. No code from any other inference engine is included.
