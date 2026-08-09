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
