# windlass — session handoff

Read this first when resuming. Written 2026-08-02, updated 2026-08-07.

> **You are on branch `task-9-serve-mode`, one commit ahead of `master`.** Serve mode is written
> and compiles but **has never run on a GPU**. The first job of the next session is a GPU window
> to run `.superpowers/sdd/2026-08-01-dsa-indexer-and-serve/task-9-verify.sh` and `test_glm_chain`,
> then merge. Do not report serve mode as working until that has happened.

## Where we are in one paragraph

windlass runs GLM-5.2 (753B/39B, MXFP4) on one RTX PRO 6000 by streaming routed experts from NVMe. Correctness is established from MXFP4 dequant up to the full 78-layer chain. The DSA sparse-attention indexer is implemented, so the old 2048-token cap is gone. Prefill was 44× amortised in both the CUDA engine and the numpy reference. Task 8 measured a real review at **18 min 16 s** and Task 9 wrote the serve mode. **Remaining work: verify serve mode on a GPU, then the real pull-request review** — the thing the engine could not do before.

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

**`vllm-qwen36` is running again and is in real use.** It was restarted 2026-08-02 20:36 UTC, shortly after this file was first written, and served 10,154 requests in the following week. It holds 92.5 GB of 97.9 GB, so windlass cannot run alongside it — its expert cache alone wants 61 GB. Stop it for a GPU window (`sudo systemctl stop vllm-qwen36`), restart immediately after, and confirm `/v1/models` on :8000 returns 200 before releasing. While it is down the RLMKX PR bots have no model, so ask before taking the window.

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
| **8 prefill measurement** | done — decision point answered, build serve mode |
| **9 serve mode** | **written and compiling, NEVER RUN ON A GPU** — branch `task-9-serve-mode` |
| **10 PR review benchmark** | the goal |

## Key measurements

```
correctness   chain 1.6928e-06 worst substep, top-5 exact   (short context, stable across 7 tasks)
              single layers ≤2.2e-08 vs transformers oracle
              indexer forced-selection 0.2–0.4 ulp vs transformers at 4096
throughput    decode  1.227 tok/s warm (8 identical 60-token requests)
                      0.637 tok/s on a real review (Task 8, unique 600-token completion)
              prefill 9.41 tok/s at 1400 tokens (148.8 s) after 4b
                      9.39 tok/s at 1464 tokens (156.0 s), cold, Task 8 — reproduces 4b
cache         56.5% hit rate at 16.1% residency (3,094 of 19,200 experts, 62 GB)
expert reads  840,000 → 18,917 per 1400-token prefill; 252.2/layer vs a 256 ceiling
```

A PR review measures **18 min 16 s** for 600 tokens: 156 s prefill + 940 s decode. The earlier
13-minute estimate assumed ~1 tok/s decode; the real rate on a unique long generation is 0.637.
A complete `--no-think` review is **21–26 minutes**. With thinking ON it is 52–94 minutes, because
600 tokens of budget buy reasoning and no review at all.

## Findings that shape what comes next

**MoE routing is chaotically sensitive, so token-exact cross-implementation agreement is unachievable at depth.** Task 7 measured `1e-07 in → 2.4e-02 out`, saturating. Two experts 1.383e-05 apart at ranks 8/9 flip on floating-point noise, and ~105,000 routing decisions happen per prefill. Layers 0/2/3 agree; 40/77 diverge at 7.4e-02 for this reason, not from a defect. **Task 10 must not gate on matching another implementation.** The engine is correct; the comparison is chaotic.

**A `weights_proj` scale error is invisible in every output tensor** — top-k is scale-invariant. It is caught only by a separate gate on raw index scores (188×, 251×, 617× in three independent tests). Any new comparison needs both an output gate and an index-score gate; they are complementary, and neither alone catches both defect classes.

**Two defects are known-undetectable and are exempted by name, not by a loosened gate**: a one-key top-k error, and `k_norm` eps 1e-6 vs 1e-5 (2.4e-03, the size of the bf16 floor).

**Twelve checks that could not fail have been found in this project.** The pattern: a check whose expectation derives from the thing under test, or whose statistic is invariant to the error it targets. Every new gate gets a negative control before it is trusted. The twelfth came from Task 9 and is worth reading as a template: a test claimed to catch a whole-body substring scan for `max_tokens` in a request body, but a JSON encoder always escapes quotes inside message content, so content can never present a bare `"max_tokens"` to any scanner — the case was unfalsifiable in principle, not merely badly written. The reachable leak was a nested key, which is well-formed and unescaped.

**The serve mode's protocol layer is CUDA-free on purpose.** `src/glm_http.cuh` includes no CUDA and no model type, so `make test_glm_http && ./test_glm_http` builds and runs **on the local workstation**, which cannot build CUDA at all. 102 checks, about a second. Parsing bugs no longer cost a GPU window. Keep it that way — anything added there that pulls in `cuda_runtime.h` gives that up.

**The sparse regime is not yet covered end to end.** 1400 < `index_topk` = 2048, so the drop mask is still a no-op in the full chain; only Tasks 5 and 6 exercise real sparsity, at single-layer scale.

## Next steps

1. **Verify Task 9 on a GPU.** Ask before taking the window — Qwen3.6 is in real use. Then:

   ```
   ssh aeronav-r6000 'sudo systemctl stop vllm-qwen36'
   rsync -a --exclude '.git' --exclude 'glm-ref/' ~/windlass/ aeronav-r6000:/home/user1/windlass-build/
   ssh aeronav-r6000 'cd /home/user1/windlass-build && make ARCH=sm_120 infer_glm'
   # server, then in a second shell:
   ./infer_glm --model-dir /home/user1/glm52-mxfp4 --packed /home/user1/packed_experts_glm \
               --serve --port 8081 --max-seq 2048 --tokens 40 --no-think
   PORT=8081 ./task-9-verify.sh          # 8 groups: health, models, 404, 400, oversize-400,
                                          # non-streaming, streaming+keepalive, 503-when-busy
   make ARCH=sm_120 test_glm_chain && ./test_glm_chain    # the 1.6928e-06 gate
   ssh aeronav-r6000 'sudo systemctl start vllm-qwen36'   # confirm /v1/models on :8000 is 200
   ```

   Then merge `task-9-serve-mode` into `master`. The task report lists every design decision and
   the reasoning behind each; the send timeout and the immediate-503 threading are the two that
   are easy to break by "simplifying".

2. **Task 10 — the PR review.** Add windlass to `~/boostrap-llm/bench_code_review.py`'s `MODELS`, pointing at `http://10.10.10.138:8081/v1/chat/completions`. Set `"streaming": True` (the benchmark already supports it, and it is what survives a 156 s prefill) and `"timeout": 3600` — the default 120 s is far short of the 21–26 minutes a review takes. Run the server with `--no-think`: with thinking on, the budget goes entirely to reasoning and no review is produced. State the context, `max_tokens`, the no-think choice and that sampling is greedy, since the other models run `max_tokens: 16384` unconstrained. Judge on quality; do **not** gate on token agreement.

## How to resume

The work uses `superpowers:subagent-driven-development` where subagents are available: one implementer per task, a review after each, findings recorded in the ledger. Every dispatch should carry the constraints the prior tasks measured — that is what has kept defects out of committed code.

Task briefs are written by hand into the ledger directory (`task-N-brief.md`); earlier notes referred to a `scripts/task-brief` generator, but **no `scripts/` directory exists in this repo** and the briefs from Tasks 2–7 were all written directly.

The regression gate for every task in plan 2 is `test_glm_chain` reporting **1.6928e-06 and top-5 exact** at short context. It has not moved in seven tasks; any movement means something reached into the existing forward path.
