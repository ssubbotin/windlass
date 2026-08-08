# windlass — session handoff

Read this first when resuming. Written 2026-08-02, updated 2026-08-08.

> **Serve mode is verified and merged.** `task-9-verify.sh` 25 of 25 on a running server,
> `test_glm_chain` 1.6928e-06 with top-5 exact. **Task 10 — the pull-request review — is the
> only thing left**, and it is the point of both plans.

## Where we are in one paragraph

windlass runs GLM-5.2 (753B/39B, MXFP4) on one RTX PRO 6000 by streaming routed experts from NVMe. Correctness is established from MXFP4 dequant up to the full 78-layer chain. The DSA sparse-attention indexer is implemented, so the old 2048-token cap is gone. Prefill was 44× amortised in both the CUDA engine and the numpy reference. Task 8 measured a real review at **18 min 16 s** and Task 9 built and verified an OpenAI-compatible serve mode. **Remaining work: the real pull-request review** — the thing the engine could not do before.

## Repos

| | |
|---|---|
| **windlass** | `github.com/ssubbotin/windlass` — MIT, this repo, all code authored by Sergey Subbotin |
| `~/flash-moe` | fork of `danveloper/flash-moe`, **no licence upstream** — do not copy code from it |
| `ssubbotin/llama.cpp` @ `feature/moe-expert-gpu-cache` | MIT, the AMD hackathon deliverable, unrelated to this repo |
| `~/boostrap-llm` | private deployment notes; holds `bench_code_review.py`, the PR review benchmark Task 10 targets |

windlass was clean-extracted from the flash-moe fork: every file is Sergey's, the one shared header (`kernels.cuh`, 2217 lines) was reduced to 9 needed symbols in `src/glm_primitives.cuh`, and `tools/tokenizer_server.py` was added because `infer_glm` execs it at runtime.

**Commit identity is enforced** — `.githooks/pre-commit` rejects anything but `Sergey Subbotin <ssubbotin@gmail.com>`. A fresh clone must run `git config core.hooksPath .githooks`.

## Machine state

Everything runs on **gpu-box** (`its address`, behind a VPN). The local workstation has a 2 GB MX450 and **cannot build CUDA**.

```
build:     ssh gpu-box 'cd ~/windlass-build && make ARCH=sm_120 <target>'
sync:      rsync -a --exclude '.git' --exclude 'glm-ref/' ~/windlass/ gpu-box:~/windlass-build/
checkpoint ~/glm52-mxfp4          408 GB, 282/282 shards, byte-verified
packed     ~/packed_experts_glm   359 GiB, 75 layers, content-verified
venv       ~/glm-oracle-venv/bin/python3   transformers 5.14.1
```

**`vllm-service` is stopped** as of 2026-08-08, and the user said it is not needed back until **Monday 2026-08-10**. The GPU is free for Task 10 until then; after that, ask before taking a window. It holds 92.5 GB of 97.9 GB when up, so windlass cannot run alongside it — its expert cache alone wants 61 GB. Restart with `sudo systemctl start vllm-service` and confirm `/v1/models` on :8000 returns 200.

**Expensive fixtures — do not delete:**

```
~/flash-moe/cuda_infer/glm-ref/   29-token chain fixtures, ~56 min to rebuild
~/t6b_1400/                       1400-token reference output, ~2 h and 84 GB RSS
~/t7_256/                         256-token all-78-layer reference, ~1 h
glm-oracle/, glm-oracle-layer0/             single-layer transformers oracles
```

## Plan progress

Two plans, both under `docs/plans/`, with SDD ledgers in `.superpowers/sdd/<plan>/progress.md` (gitignored — the per-task reports live there and are the real evidence base).

**Plan 1 — `2026-07-31-glm52-implementation-plan.md`: complete (12 tasks).** Built the engine, established correctness, measured throughput. Ended at a failed 2 tok/s gate.

**Plan 2 — `2026-08-01-dsa-indexer-and-serve.md`: Tasks 1–9 done, Task 10 remains.**

| task | state |
|---|---|
| 1 spec extraction · 2 weights · 3 kernels | done |
| 4 IndexShare + mask · **4b layer-major prefill (CUDA)** | done |
| 5 long-context oracle · 6 indexer in numpy ref · **6b layer-major (numpy)** | done |
| 7 full chain at long context | done |
| **8 prefill measurement** | done — decision point answered, build serve mode |
| **9 serve mode** | done — verified on the GPU, 25/25, merged to `master` |
| **10 PR review benchmark** | **the only task left, and the goal** |

## Key measurements

```
correctness   chain 1.6928e-06 worst substep, top-5 exact   (short context, stable across 8 tasks)
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

**Thirteen checks that could not fail have been found in this project.** The pattern: a check whose expectation derives from the thing under test, or whose statistic is invariant to the error it targets. Every new gate gets a negative control before it is trusted. The last two both came from Task 9 and are worth reading as templates. The twelfth: a test claimed to catch a whole-body substring scan for `max_tokens`, but a JSON encoder always escapes quotes inside message content, so content can never present a bare `"max_tokens"` to any scanner — unfalsifiable in principle. The thirteenth: six `find()` calls checking the six fields of a response body all passed against a body carrying a stray quote no parser would accept, because **substring presence is invariant to anything between the substrings**. Where output is fully determined by its inputs, assert it character for character.

**Compilation and the host-side protocol suite together said nothing about two defects that a real server exposed in sixty seconds.** Every non-streaming response body was invalid JSON, and the stop-token set was wrong, so no completion ever ended on its own and turn scaffolding leaked into the answer. GLM-5.2 records `<|user|>` and `<|observation|>` as turn-enders **only in `generation_config.json`**; deriving stops from tokenizer metadata yields six plausible ids of which five are multimodal delimiters. Read the config. Both defects are written up in `task-9-report.md`.

**The serve mode's protocol layer is CUDA-free on purpose.** `src/glm_http.cuh` includes no CUDA and no model type, so `make test_glm_http && ./test_glm_http` builds and runs **on the local workstation**, which cannot build CUDA at all. 98 checks, about a second. Parsing bugs no longer cost a GPU window. Keep it that way — anything added there that pulls in `cuda_runtime.h` gives that up.

**The sparse regime is not yet covered end to end.** 1400 < `index_topk` = 2048, so the drop mask is still a no-op in the full chain; only Tasks 5 and 6 exercise real sparsity, at single-layer scale.

## Next steps

**Task 10 is the only remaining task.** Bringing the server up:

```
rsync -a --exclude '.git' --exclude 'glm-ref/' ~/windlass/ gpu-box:~/windlass-build/
ssh gpu-box 'cd ~/windlass-build && make ARCH=sm_120 infer_glm'
ssh -f gpu-box 'cd ~/windlass-build && nohup ./infer_glm \
    --model-dir ~/glm52-mxfp4 --packed ~/packed_experts_glm \
    --serve --port 8081 --max-seq 8192 --tokens 2048 --no-think > /tmp/serve.log 2>&1 </dev/null &'
# ~100 s to load. Confirm: curl -s http://127.0.0.1:8081/health  -> {"status":"ok","busy":false}
```

Re-run `PORT=8081 bash .superpowers/sdd/2026-08-01-dsa-indexer-and-serve/task-9-verify.sh` after
any change to the serving path — it is 25 checks and about four minutes. The send timeout and the
immediate-503 threading are the two design points easy to break by "simplifying"; the task report
gives the reasoning for each.

**The review itself.** Add windlass to `~/boostrap-llm/bench_code_review.py`'s `MODELS`, pointing at `http://its address:8081/v1/chat/completions`. Set `"streaming": True` (the benchmark already supports it, and it is what survives a 156 s prefill) and `"timeout": 3600` — the default 120 s is far short of the 21–26 minutes a review takes. Run the server with `--no-think`: with thinking on, the budget goes entirely to reasoning and no review is produced. State the context, `max_tokens`, the no-think choice and that sampling is greedy, since the other models run `max_tokens: 16384` unconstrained. Judge on quality; do **not** gate on token agreement.

## How to resume

The work uses `superpowers:subagent-driven-development` where subagents are available: one implementer per task, a review after each, findings recorded in the ledger. Every dispatch should carry the constraints the prior tasks measured — that is what has kept defects out of committed code.

Task briefs are written by hand into the ledger directory (`task-N-brief.md`); earlier notes referred to a `scripts/task-brief` generator, but **no `scripts/` directory exists in this repo** and the briefs from Tasks 2–7 were all written directly.

The regression gate for every task in plan 2 is `test_glm_chain` reporting **1.6928e-06 and top-5 exact** at short context. It has not moved in eight tasks; any movement means something reached into the existing forward path.
