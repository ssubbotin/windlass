# Plan 3 — Expert tiering and the fetch path

Backlog produced by a TRIZ invention council (40 agents, 28 measured scout facts) over the
plateau at 0.889 tok/s, then checked one item at a time on the real machine.

## Measured constants (this machine, several instruments each)

```
NVMe 20 MB random reads   9.86 GB/s   saturates at low queue depth; concurrency worth +12%
                                      Gen5 x4 link; the 14.8 GB/s vendor figure is sequential
buffered read (page cache warm)  6.7-6.9 GB/s   slower than O_DIRECT: every read pays copy_to_user
pinned H2D alone         50.76 GB/s   0.395 ms per expert
pinned D2H alone         24.87 GB/s   0.806 ms per expert
CONCURRENT H2D / D2H     19.61 / 19.57 GB/s   aggregate 39.18
pageable H2D             29.17 GB/s (1 thread), 19.94 (8 threads)
host slab allocation     104 GB touched, zero swap-out, page cache 111 GB -> 9 GB
decode-only hit rate     52.7%   (the long-quoted 44% is a whole-run figure diluted by prefill)
```

## Backlog, with outcomes

| # | item | outcome |
|---|---|---|
| 1 | **O_DIRECT on the expert store** | **DONE. 1.058 -> 1.572 tok/s, +48.6%**, output byte-identical |
| 2 | Exclusive host victim slab | **BUILT, MEASURES 2.24 tok/s, KNOWN INCORRECT — not shipped** |
| 3 | Popularity pinning instead of LRU | **DEAD.** Killed three times in simulation |
| 4 | MTP speculative decode | **RE-PRICED** from ~2.4x to +15-27% at perfect acceptance |

### 1. O_DIRECT — done

One flag. The store is 4096-aligned by construction (20,054,016 = 4896 x 4096, expert e at
e x EXPERT_BYTES), so no restructuring was needed. Verified byte-identical greedy output and
`test_glm_chain` still at 1.6928e-06 with top-5 exact.

The repository's headline finding said the pipeline was *structurally shallow-queued* and that
faster storage could not help. The null it rested on (io_threads 4 -> 8, +0.5%) is real and the
mechanism was wrong: the reads were buffered, and buffered reads cap at 6.7-6.9 GB/s even with a
warm cache. That wrong mechanism steered two plans away from the fetch path.

### 2. The host slab — the honest entry

The council's three surviving candidates were the same object: an exclusive host tier holding only
what VRAM evicts. Simulation on real routing traces put it at 0.4999 hit-on-miss against a 0.2124
control for a mirroring tier, i.e. exclusivity carries 58% of the effect.

Measured here at 80 GB / 4283 slots: **the slab does serve 44-47% of GPU misses, exactly as
predicted**, and pinned it reaches 2.24 tok/s. Three things the simulations did not model:

- **Write amplification 1.97-2.11x.** 42,000 evictions written to get 20,000 serves back. Only
  48% of what is pushed down is ever read again, so the D2H is mostly wasted PCIe.
- **Unpinned it is a net regression** (0.940 against 1.058 baseline) and it drags O_DIRECT down
  from 1.572 to 1.481. A pageable `cudaMemcpyAsync` is synchronous and blocks the worker.
- **It is incorrect.** Greedy decode diverges from baseline, which under an identical prompt can
  only mean wrong bytes. Three ordering defects found and fixed, a fourth still open:
  1. a taken slot returned to the free list while its H2D was still pending;
  2. no cross-stream ordering between a slot's D2H writer and its H2D reader (recording the
     event is not the same as waiting on it);
  3. the eviction D2H not ordered behind the outgoing block's own fill, so a block evicted
     before its fill retired was copied out as stale bytes under the right key.

Pageable memory hid every one of them by making each copy synchronous; pinning for speed is what
exposed them. The flag prints a KNOWN INCORRECT banner and must not be used.

**The correct next step is a redesign, not a fourth patch.** The candidate with the best
cost structure is fetch-fill by host memcpy out of the pinned staging buffer, which the bytes
already pass through — zero D2H, no write amplification, and no cross-stream slab ownership at
all, since the slab would then be written only by the CPU. It gives up the pure-victim retention
window, which is what simulation says carries the effect, so it needs its own trace replay first.

**The correctness gate that caught this is worth naming:** greedy decode with a fixed prompt must
produce byte-identical output. It cost one run, it is exact, and `test_glm_chain` cannot see this
class of defect at all, because the arithmetic is untouched and only the bytes fed into it are wrong.

### 3 and 4 — closed and re-priced by measurement

Static popularity pinning is dead: pin fractions 0.25/0.50/0.75/1.00 give 53.11/51.92/46.96/35.52%
against LRU's 53.61 (monotonically harmful), 0.273 against the victim rule's 0.437 at equal
capacity, and out-of-fold 1.989 against 2.147 tok/s.

Speculative decode was priced from the literature's 1.25-1.48x union cost at k=4. Measured per-layer
union on a real GLM-5.2 trace is 1.70x at k=2, 2.33x at k=3, 2.90x at k=4, so bytes per token fall
only to 0.72-0.85x even at perfect acceptance. It also has an unbudgeted prerequisite: layer 78 is a
full BF16 MoE layer, 18.07 GiB, absent from the packed store.

## Also dead, with kill numbers

Lossless byte reduction (entropy 3.769/4, zstd-19 at 0.8946), cross-expert dedup (pairwise nibble
agreement 0.0764 against a chance baseline of 0.0757, zero duplicate blocks), per-layer cache
partitioning (53.59 against 54.11 global), completion-order consumption (within 1 ms in all 14
configurations), store re-layout for fragmentation (128 KiB and 20 MB reads within 5%).

## Ceiling, for planning

Belady oracle on the real trace is 3.64-4.02 tok/s at maximum RAM, and the compulsory-miss floor is
a 0.861 hit rate. **A sub-4-minute review is not reachable by tiering.** Anything faster has to cut
token count or bytes per expert, and both of those are measured closed for now.

## Harness note

The three-PR review harness has 18% drift on an identical binary and cannot resolve anything below
about 20%. All A/B work uses the paired 150-token same-prompt harness with `--ignore-eos`
(mean-level noise floor 0.4%) behind an idle-device gate; the three reviews are a confirmation
population, run once at the end, never used to choose between variants.

## Prior measurements from the earlier engine, and what they change here

The flash-moe fork carries a `cuda` lineage that hit several of these questions first, on the same
card. No code is taken from it (it has no upstream licence); these are its recorded measurements
and the conclusions they force here.

**Hiding fetches under idle compute is dead, and was already paid for.** Overlapping the shared
expert's forward pass with expert I/O on a separate stream was built, measured **neutral cold and
warm, and reverted** — the shared expert is **0.03 ms per layer** against roughly 13 ms of fetch.
That closes resource (e) from this plan's inventory at a factor of ~400. Compute is 3.5% of the
step; there is no meaningful shadow to hide 20 MB reads in.

**The Qwen3.5-397B comparison in Task 10 was against a known-broken configuration, and the write-up
overstates it.** That result is degenerate word association, and the engine that produced it later
gained temperature 0.6, top-p 0.95, a repetition penalty of 1.3 over a 1024-token ring buffer, and a
degeneration detector that forces `</think>` when more than 75% of the last 32 tokens carry no ASCII
letters. The stored benchmark predates all of it. So "no large-MoE-from-storage engine had produced
a real review" is true of that *run*, not of that engine, and Task 10 cleared a lower bar than it
claims. The fix is a re-run of that model with sampling enabled, not a stronger claim here.

**windlass is greedy with no repetition penalty — the exact configuration that degenerated.** Three
clean reviews are not evidence of robustness against this. A repetition penalty and a degeneration
detector are cheap and belong in serve mode before the next benchmark, independent of throughput.

**The thinking finding from Task 8 has a known mechanism and a better fix than `--no-think`.**
Greedy decoding barely favours `</think>` over content on long prompts, which is why a review-sized
budget with thinking on produced no review at all. The earlier engine's answer is a **think budget**:
cap reasoning at `max_tokens/2`, then force-inject `</think>`. That keeps reasoning on a review-sized
budget instead of disabling it, and it is strictly better than the flag Task 10 shipped with.

**A prefill optimisation to approach with care.** KV/delta snapshot save-and-restore was found to
produce different output than a fresh prefill on CUDA, root cause never established, and was worked
around by re-prefilling every request at a cost of ~2 s. Our prefill is **156 s**, so the same
shortcut is roughly eighty times more tempting here and carries a known-unresolved correctness
hazard. Anything of that shape needs the byte-identical-output gate from item 2 above, run first.

## Item 2 redesign: the trace replay, and what it decided

`tools/replay_expert_trace.py` replays a real 108,127-lookup routing trace (1042-token prefill plus
149 decode steps) and charges both directions of PCIe at this machine's measured rates.

```
                              host serves    NVMe    D2H     vs baseline
GPU only (shipped)                    0     56836      0        1.00x
victim-fill,  86 GB               18251     38585  53764        0.75x
fetch-fill,   86 GB                7966     48870      0        1.11x
victim-fill, 112 GB               21828     35008  53764        0.77x
fetch-fill,  112 GB               12218     44618      0        1.19x
```

**Victim-fill loses at every capacity.** It pays 53,764 writes to collect 21,828 serves, and the
council's simulators scored the read side only. This reproduces the live measurement in both
direction and rough magnitude: unpinned victim-fill measured 0.940 against a 1.058 baseline, and it
pulled O_DIRECT down from 1.572 to 1.481. The 2.24 tok/s figure was the corrupted run taking a
cheaper path and was never real.

**Expert weights are read-only, so the writeback was never needed at all.** A host copy taken at
fetch time cannot go stale. The victim rule was importing a writeback discipline from caches that
hold mutable data; here the bytes are immutable, the staging buffer already contains them, and a
host-to-host memcpy puts them in the slab off the PCIe bus entirely. That removes the D2H, the write
amplification, and the entire class of cross-stream slot-ownership races that cost three patches:
the slab becomes CPU-write-only, and the GPU only ever reads from it.

The cost is exclusivity. Fetch-fill duplicates whatever the GPU holds, so it serves 14-21% of misses
against victim-fill's 32-38%. That is the trade the council's 0.4999-vs-0.2124 result measured, and
with the write side charged the cheaper-but-inclusive rule still wins by 1.49-1.53x.

**Decision: build fetch-fill at the largest slab that allocates cleanly (112 GB / 5568 slots).**
Expected 1.19x on top of O_DIRECT, i.e. roughly 1.06 -> 1.87 tok/s against the shipped 1.058.
Caveat carried forward: the replay's absolute tok/s is not calibrated (it divides prefill lookups
across decode tokens); only the ratios are load-bearing, and the build is gated on the
byte-identical-output check, not on the model.

**Ceiling unchanged.** Belady on this trace is 3.64-4.02 tok/s. Tiering does not reach a
sub-4-minute review, and after this item the data-movement family is close to spent.
