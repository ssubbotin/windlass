# windlass — session handoff

Read this first when resuming. Written 2026-08-02, rewritten 2026-08-10, revised 2026-09-24.

> **Both plans that had tasks are finished.** Plan 1 (12 tasks) and Plan 2 (10 tasks) are complete;
> GLM-5.2 reviews real pull requests. Plan 3 is an optimisation backlog whose every item is either
> shipped or measured dead, except MTP speculation, which a 2026-09-24 review re-priced from 1.44x to
> about 1.06x. **Nothing is half-built and nothing is unverified.** `master` is clean.

## Where we are in one paragraph

windlass runs GLM-5.2 (753B/39B, MXFP4) on one RTX PRO 6000 by streaming routed experts from NVMe.
Correctness is established from MXFP4 dequant to the full 78-layer chain, the DSA indexer is in, and
serve mode is an OpenAI-compatible endpoint verified 25/25 on a live server. The engine **reviews
real pull requests** — the thing it could not do before — and decode has since gone from 1.058 to
1.623 tok/s. MTP speculation was the last open lever and is now priced at ~5% of decode unless its
acceptance measures 0.9 or better. What is left is one GPU window of measurements, then a decision:
prefill compute, MTP at k=1, or product work on a finished engine (see Next steps).

## Machine state — READ THIS BEFORE ANY GPU WORK

This repository is public. Host names, addresses, remote paths and the service that shares the GPU
are kept in **`CLAUDE.local.md`** (untracked). Read it before any GPU work, and never copy its
contents into a tracked file.

**The GPU is shared with a vLLM service that is in real use.** windlass cannot run alongside it —
the expert pool alone wants 61 GB. **Ask before taking a window**, stop the service for the window,
restart it afterwards and confirm it serves.

Do **not** `pkill -f infer_glm` from an ssh one-liner: the pattern matches the ssh command itself and
kills the shell. Find the pid with `pgrep -af "infer_glm --model"` and `kill -9` it.

Everything runs on the GPU box. The local workstation has a 2 GB MX450 and **cannot build CUDA** —
but see "what builds locally" below.

**Expensive fixtures on the box — do not delete:** the `glm-ref/` chain fixtures (~56 min to
rebuild), `t6b_1400/`, `t7_256/`, `glm-oracle*/`. Also two real expert routing traces in `/tmp`, the
input to `tools/replay_expert_trace.py`. Cheap to regenerate but only with a GPU window. Paths are in
`CLAUDE.local.md`.

## Plan progress

**Plan 1 — `2026-07-31-glm52-implementation-plan.md`: complete (12 tasks).**
**Plan 2 — `2026-08-01-dsa-indexer-and-serve.md`: complete (10 tasks).** Task 10 met its acceptance.
**Plan 3 — `2026-08-09-expert-tiering.md`: one item open, everything else shipped or dead.**

```
O_DIRECT                     DONE   +48.6%, output byte-identical
host expert slab             DONE   +3.2%, correct; family closed on fill cost
degeneration defences        DONE   29 host checks, off by default
popularity pinning           DEAD   monotonically harmful, killed 3x independently
lossless byte reduction      DEAD   entropy 3.769/4, zstd-19 = 0.8946, no redundancy
lossy draft experts          DEAD   dominated by MTP in every cell
shared-expert / IO overlap   DEAD   0.03 ms/layer against ~13 ms of fetch
MTP speculation              OPEN   re-priced: ceiling 1.18x at k=1; worth it only at acceptance >= 0.9
```

## Key measurements

```
correctness  chain 1.6928e-06 worst substep, top-5 exact  (stable across 9 tasks)
             single layers <=2.2e-08 vs transformers oracle
decode       1.058 tok/s baseline -> 1.572 (--o-direct) -> 1.623 (+ --host-cache 104)
             all three byte-identical output, 150-token paired harness
PR review    prefill 9.30 tok/s, decode 0.889, 21m30s per review  [PRE-O_DIRECT, now stale]
hit rate     52.7% DECODE-ONLY. The long-quoted 44% is a whole-run figure diluted by
             an all-miss prefill; do not use it in any projection.
NVMe         9.86 GB/s O_DIRECT 20 MB random; buffered 6.7-6.9 even with a warm cache
H2D/D2H      pinned 50.76 / 24.87 alone; 19.61 / 19.57 CONCURRENT; pageable H2D 29.17
host RAM     104 GB allocates and touches with zero swap-out
union        per-layer expert union over n tokens: 1.70x n=2, 2.33x n=3, 2.90x n=4
             => any speculation is capped at n/UNION[n]: 1.18x / 1.29x / 1.38x
ceiling      Belady oracle 3.64-4.02 tok/s. Tiering cannot reach a sub-4-minute review.
```

## Findings that shape what comes next

**The one open item is worth about 5%, not 1.44x.** MTP speculation drafts with the checkpoint's own
head (`num_nextn_predict_layers: 1`). Plan 3 priced it at 1.44x for k=2 at 0.8 acceptance, but its
table credited k+1 tokens while charging the expert union of only k positions. A verify step commits
at most one token per forwarded position, and k drafts forward k+1 positions. Charged correctly, with
the draft free:

```
drafts  positions   accept=1.0   accept=0.8   accept=0.6
k=1         2           1.18x        1.06x        0.94x
k=2         3           1.29x        1.05x        0.84x
k=3         4           1.38x        1.02x        0.75x
```

The draft is not free either: resident, layer 78 takes 18.07 GiB (~30%) out of the expert pool;
streamed, each draft fetches eight BF16 experts at 3.6x the bytes of an MXFP4 one. Neither cost is
priced yet. Build it only if acceptance measures 0.9 or better, and then at k=1. The full correction
is in Plan 3.

**Prefill may now be compute-bound, and nobody has priced it.** Task 4b: going from 256 to 1400
tokens added 14% more fetches but 75 s more prefill (73.5 -> 148.8 s). The extra fetches explain
about 10 s; the rest grows with token count. With O_DIRECT the fetch floor for a whole-store prefill
is ~38 s (379 GB at 9.86 GB/s). This is an inference from two runs, not a measurement; one `--timing`
prefill settles it. If it holds, larger PRs make it worse, and the batched expert matvecs are a
candidate for tensor cores.

**Hit rate was never the binding constraint; fill cost was.** A 40-agent invention council projected
2.0-2.4x from a three-tier design and the real win was one `open()` flag. Its simulators ranked
candidates on hit rate and none charged the cost of *filling* the tier. The replay I wrote to check
them repeated the error one level down by modelling a CPU memcpy as free. Every tier variant pays
about one expert-sized copy per fill and recovers well under half a copy per serve — the exclusive
rule pays on PCIe, the inclusive rule pays on the CPU, and neither inverts the ratio.
**Charge the write side of any future caching idea before believing its hit rate.**

**A correct null with a wrong mechanism steered two plans.** The old headline finding said the fetch
pipeline was "structurally shallow-queued" so faster storage could not help. The null it rested on
was real (io_threads 4->8, +0.5%) and the mechanism was wrong: reads were buffered, and buffered
reads cap at 6.7 GB/s even warm because each pays `copy_to_user`. That cost 48.6% sitting behind one
flag for two plans. When a null is explained, check the explanation independently of the null.

**Thirteen checks that could not fail have been found here.** Newest: six `find()` calls verified
six fields of a JSON body and all six passed against a body carrying a stray quote no parser would
accept — substring presence is invariant to anything *between* the substrings. Where output is fully
determined by its inputs, assert it character for character.

**MoE routing is chaotically sensitive**, so token-exact cross-implementation agreement is
unachievable at depth (`1e-07 in -> 2.4e-02 out`, saturating). Never gate on matching another
implementation.

## The gates, and which catches what

```
test_glm_chain          1.6928e-06 worst substep, top-5 exact. Arithmetic only.
byte-identical output   ./infer_glm ... --tokens 150 --ignore-eos, stdout compared to baseline.
                        Greedy + fixed prompt => identical bytes or something is wrong.
```

**These are complementary and the second is not optional.** The host-slab work produced three
separate data-corruption defects that `test_glm_chain` is structurally blind to: the arithmetic was
untouched and only the bytes fed into it were wrong. Byte-identical output caught every one, at a
cost of one run. Any change to the fetch, cache or tier path gets both.

## What builds and runs locally, with no GPU

```
make test_glm_http     && ./test_glm_http       # 98 checks, ~1 s
make test_glm_sampling && ./test_glm_sampling   # 29 checks, ~1 s
python3 tools/replay_expert_trace.py /tmp/route.bin 149   # needs a trace from the box
```

`src/glm_http.cuh` and `src/glm_sampling.cuh` include no CUDA and no model type **on purpose**.
Parsing and sampling bugs produce plausible-looking wrong text rather than errors, and a GPU window
is far too expensive a place to find them. **Keep it that way** — anything added there that pulls in
`cuda_runtime.h` gives it up.

## Serve mode

```bash
./infer_glm --model-dir ./glm52-mxfp4 --packed ./packed_experts \
            --serve --port 8081 --max-seq 4096 --tokens 2048 --no-think --o-direct
```

~100 s to load. `curl -s http://127.0.0.1:8081/health` -> `{"status":"ok","busy":false}`.
Full check: `PORT=8081 bash .superpowers/sdd/2026-08-01-dsa-indexer-and-serve/task-9-verify.sh`
(25 checks, ~4 min). The send timeout, the `peer_gone` poll and the immediate-503 threading are the
three design points easiest to break by "simplifying"; the task-9 report gives the reasoning.

`--rep-penalty` and `--degen-window` are **off by default and must stay that way** — greedy argmax
with no penalty is the configuration every correctness result was measured under, and the
byte-identical gate depends on it staying reachable. `test_glm_sampling` asserts that inertness.

## Next steps

1. **One GPU window, three measurements.** Ask first; the vLLM service has to stop for it.
   - **Re-run the three-PR benchmark** with `--o-direct --host-cache 104`. The recorded 21m30s per
     review predates O_DIRECT. Output is byte-identical, so only the timings need refreshing.
     `~/boostrap-llm/bench_code_review.py`, `BENCH_ONLY=windlass`.
   - **Prefill with `--timing`** on the same prompt: fetch against compute after O_DIRECT.
   - **Dump the main model's final hidden states** over one review, so MTP acceptance can be
     measured offline on the box's CPU, without the 18 GiB repack. Before building the dump, check
     that a reference for the GLM MTP head exists: transformers often skips those weights.
2. **Decide from the numbers.** Prefill compute-bound: batched expert matvecs onto tensor cores.
   Acceptance >= 0.9: MTP at k=1. Otherwise the throughput work is done.
3. **Product gaps for the PR-review use**, ranked above MTP: the benchmark review already runs past
   `index_topk` (1464 prompt + 600 generated = 2064 tokens), a range validated one layer at a time:
   no full-chain comparison above 2048 exists, and Task 7 says one needs a forced-selection pass
   because the chain is chaotic at that length; the think budget Plan 3 calls strictly better than `--no-think` is not
   built; and a nightly batch needs a regular GPU window shared with the vLLM service.
4. **A fairness correction is outstanding.** Task 10 compares against a stored Qwen3.5-397B result
   that is degenerate word salad, and that run predates the sampling fixes the same engine later
   gained. windlass cleared a lower bar than the write-up claims. Re-running that model with
   sampling enabled would settle it, and Plan 3 already says so.

## How to resume

Plan 2 used `superpowers:subagent-driven-development` with per-task reports in
`.superpowers/sdd/<plan>/` (gitignored — that directory is the real evidence base). Task briefs are
written by hand; there is no `scripts/` directory in this repo.

**Commit identity is enforced** — `.githooks/pre-commit` rejects anything but
`Sergey Subbotin <ssubbotin@gmail.com>`. A fresh clone must run `git config core.hooksPath .githooks`.
`~/flash-moe` has **no upstream licence**: read it for measurements, never copy code from it.
