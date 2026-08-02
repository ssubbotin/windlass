# windlass — session handoff

Read this first when resuming. Written 2026-08-02.

## Where we are in one paragraph

windlass runs GLM-5.2 (753B/39B, MXFP4) on one RTX PRO 6000 by streaming routed experts from NVMe. Correctness is established from MXFP4 dequant up to the full 78-layer chain. The DSA sparse-attention indexer is implemented, so the old 2048-token cap is gone. Prefill was 44× amortised in both the CUDA engine and the numpy reference. **Remaining work: a serve mode, then a real pull-request review** — the thing the engine could not do before.

## Repos

| | |
|---|---|
| **windlass** | `github.com/ssubbotin/windlass` — MIT, this repo, all code authored by Sergey Subbotin |
| `~/flash-moe` | fork of `danveloper/flash-moe`, **no licence upstream** — do not copy code from it |
| `ssubbotin/llama.cpp` @ `feature/moe-expert-gpu-cache` | MIT, the AMD hackathon deliverable, unrelated to this repo |
| `~/boostrap-llm` | RLMKX deployment notes; holds `bench_code_review.py`, the PR review benchmark Task 10 targets |

windlass was clean-extracted from the flash-moe fork: every file is Sergey's, the one shared header (`kernels.cuh`, 2217 lines) was reduced to 9 needed symbols in `src/glm_primitives.cuh`, and `tools/tokenizer_server.py` was added because `infer_glm` execs it at runtime.

**Commit identity is enforced** — `.githooks/pre-commit` rejects anything but `Sergey Subbotin <ssubbotin@gmail.com>`. A fresh clone must run `git config core.hooksPath .githooks`.

## Machine state

Everything runs on **aeronav-r6000** (`10.10.10.138`, behind the Navigator VPN). The local workstation has a 2 GB MX450 and **cannot build CUDA**.

```
build:     ssh aeronav-r6000 'cd /home/user1/windlass-build && make ARCH=sm_120 <target>'
sync:      rsync -a --exclude '.git' --exclude 'glm-ref/' ~/windlass/ aeronav-r6000:/home/user1/windlass-build/
checkpoint /home/user1/glm52-mxfp4          408 GB, 282/282 shards, byte-verified
packed     /home/user1/packed_experts_glm   359 GiB, 75 layers, content-verified
venv       /home/user1/glm-oracle-venv/bin/python3   transformers 5.14.1
```

**`vllm-qwen36` is stopped and staying stopped** — the owner released the machine and will say when to switch Qwen3.6 back. While it is down the RLMKX PR bots have no model. Restore with `sudo systemctl start vllm-qwen36` (it is still `enabled`, so it also returns on reboot).

**Expensive fixtures — do not delete:**

```
/home/user1/flash-moe/cuda_infer/glm-ref/   29-token chain fixtures, ~56 min to rebuild
/home/user1/t6b_1400/                       1400-token reference output, ~2 h and 84 GB RSS
/home/user1/t7_256/                         256-token all-78-layer reference, ~1 h
glm-oracle/, glm-oracle-layer0/             single-layer transformers oracles
```

## Plan progress

Two plans, both under `docs/plans/`, with SDD ledgers in `.superpowers/sdd/<plan>/progress.md` (gitignored — the per-task reports live there and are the real evidence base).

**Plan 1 — `2026-07-31-glm52-implementation-plan.md`: complete (12 tasks).** Built the engine, established correctness, measured throughput. Ended at a failed 2 tok/s gate.

**Plan 2 — `2026-08-01-dsa-indexer-and-serve.md`: Tasks 1–7 done, 8–10 remain.**

| task | state |
|---|---|
| 1 spec extraction · 2 weights · 3 kernels | done |
| 4 IndexShare + mask · **4b layer-major prefill (CUDA)** | done |
| 5 long-context oracle · 6 indexer in numpy ref · **6b layer-major (numpy)** | done |
| 7 full chain at long context | done |
| **8 prefill measurement** | largely answered by 4b — see below |
| **9 serve mode** | next |
| **10 PR review benchmark** | the goal |

## Key measurements

```
correctness   chain 1.6928e-06 worst substep, top-5 exact   (short context, stable across 7 tasks)
              single layers ≤2.2e-08 vs transformers oracle
              indexer forced-selection 0.2–0.4 ulp vs transformers at 4096
throughput    decode  1.227 tok/s warm
              prefill 9.41 tok/s at 1400 tokens (148.8 s) after 4b
cache         56.5% hit rate at 16.1% residency (3,094 of 19,200 experts, 62 GB)
expert reads  840,000 → 18,917 per 1400-token prefill; 252.2/layer vs a 256 ceiling
```

A PR review is therefore roughly **13 minutes**: 149 s prefill + ~600 tokens at ~1 tok/s.

## Findings that shape what comes next

**MoE routing is chaotically sensitive, so token-exact cross-implementation agreement is unachievable at depth.** Task 7 measured `1e-07 in → 2.4e-02 out`, saturating. Two experts 1.383e-05 apart at ranks 8/9 flip on floating-point noise, and ~105,000 routing decisions happen per prefill. Layers 0/2/3 agree; 40/77 diverge at 7.4e-02 for this reason, not from a defect. **Task 10 must not gate on matching another implementation.** The engine is correct; the comparison is chaotic.

**A `weights_proj` scale error is invisible in every output tensor** — top-k is scale-invariant. It is caught only by a separate gate on raw index scores (188×, 251×, 617× in three independent tests). Any new comparison needs both an output gate and an index-score gate; they are complementary, and neither alone catches both defect classes.

**Two defects are known-undetectable and are exempted by name, not by a loosened gate**: a one-key top-k error, and `k_norm` eps 1e-6 vs 1e-5 (2.4e-03, the size of the bf16 floor).

**Eleven checks that could not fail have been found in this project.** The pattern: a check whose expectation derives from the thing under test, or whose statistic is invariant to the error it targets. Every new gate gets a negative control before it is trusted.

**The sparse regime is not yet covered end to end.** 1400 < `index_topk` = 2048, so the drop mask is still a no-op in the full chain; only Tasks 5 and 6 exercise real sparsity, at single-layer scale.

## Next steps

1. **Task 8** — formally measure prefill on a realistic prompt. Largely answered by 4b (9.41 tok/s at 1400); run it for the record, cite the 4b report, not Task 4's stale 0.25 figure.
2. **Task 9 — serve mode.** Port `--serve` from `~/flash-moe/cuda_infer/infer.cu:2579-3400` (OpenAI-compatible `/v1/chat/completions`, SSE, `/v1/models`, `/health`). Tool calling is not needed. Single in-flight request; 503 when busy; fail loudly if `prompt + max_tokens > max_seq`.
3. **Task 10 — the PR review.** Add windlass to `~/boostrap-llm/bench_code_review.py`'s `MODELS`. Judge the review on quality and state the context and `max_tokens` used. Do **not** gate on token agreement.

## How to resume

The work uses `superpowers:subagent-driven-development`: one implementer subagent per task, a review after each, findings recorded in the ledger. Task briefs come from `scripts/task-brief <plan> <N>`. Every dispatch should carry the constraints the prior tasks measured — that is what has kept defects out of committed code.

The regression gate for every task in plan 2 is `test_glm_chain` reporting **1.6928e-06 and top-5 exact** at short context. It has not moved in seven tasks; any movement means something reached into the existing forward path.
