#!/usr/bin/env python3
"""Decide the host-tier redesign on a real trace before writing any CUDA.

Plan 3 item 2 built a victim-fill slab: the GPU evicts, the evicted 20 MB is
written down to host over PCIe, and a later miss reads it back. It measured
2.24 tok/s and is incorrect, and the fix is a redesign rather than a fourth
ordering patch. The candidate replacement is fetch-fill: the bytes already pass
through a pinned staging buffer on their way from NVMe, so a CPU memcpy puts
them in the slab for free and no D2H ever happens.

The question this answers is not "which has the better hit rate" — victim-fill
wins that, because it is exclusive and fetch-fill duplicates whatever the GPU
holds. It is "which is faster once the write side is charged", which is exactly
what the council's simulators left out.

Costs are the measured constants from this machine, not estimates.
"""
import sys, struct
from collections import OrderedDict

EXPERT_MB   = 20.054016
NVME_GBS    = 9.86           # O_DIRECT, 20 MB random, three instruments
H2D_SOLO    = 50.76          # pinned, nothing else on the bus
H2D_CONC    = 19.61          # pinned, with D2H saturating the other direction
D2H_CONC    = 19.57
COMPUTE_MS  = 140.0          # per token, non-fetch

def ms(n, gbs):
    return n * EXPERT_MB / 1000.0 / gbs * 1000.0

def load(path):
    b = open(path, 'rb').read()
    n = len(b) // 4
    out = []
    for i in range(n):
        l, e = struct.unpack_from('<HH', b, i * 4)
        out.append(l * 256 + e)
    return out

def sim(keys, gpu_cap, host_cap, mode):
    """mode: 'gpu' | 'victim' | 'fetch'"""
    gpu  = OrderedDict()          # key -> True, ordered LRU (front = oldest)
    host = OrderedDict()
    gpu_hits = host_hits = nvme = d2h = 0

    for k in keys:
        if k in gpu:
            gpu.move_to_end(k)
            gpu_hits += 1
            continue

        # --- GPU miss: where do the bytes come from? ---
        if k in host:
            host_hits += 1
            if mode == 'victim':
                del host[k]        # exclusive: promoting removes it
            else:
                host.move_to_end(k)
        else:
            nvme += 1
            if mode == 'fetch':
                # Free: the bytes are in the pinned staging buffer already, so
                # this is a host-to-host memcpy off the PCIe bus entirely.
                host[k] = True
                host.move_to_end(k)
                if len(host) > host_cap:
                    host.popitem(last=False)

        # --- admit to GPU, evicting if full ---
        gpu[k] = True
        gpu.move_to_end(k)
        if len(gpu) > gpu_cap:
            victim, _ = gpu.popitem(last=False)
            if mode == 'victim':
                d2h += 1           # 20 MB down the bus, every single eviction
                host[victim] = True
                host.move_to_end(victim)
                if len(host) > host_cap:
                    host.popitem(last=False)
            elif mode == 'fetch':
                # Already resident if it was ever read from NVMe. Refresh its
                # position so it gets a window measured from eviction, which is
                # what the victim rule buys — without paying for a copy.
                if victim in host:
                    host.move_to_end(victim)

    return gpu_hits, host_hits, nvme, d2h

def report(name, keys, gpu_cap, host_cap, mode, ntok):
    gh, hh, nv, dh = sim(keys, gpu_cap, host_cap, mode)
    tot = gh + hh + nv
    h2d_rate = H2D_CONC if dh else H2D_SOLO
    # Every GPU miss costs an H2D regardless of source.
    t = ms(nv, NVME_GBS) + ms(hh + nv, h2d_rate) + ms(dh, D2H_CONC)
    t_tok = t / ntok + COMPUTE_MS
    print(f"{name:<34} gpu_hit={100*gh/tot:5.1f}%  host_serves={hh:6d} "
          f"({100*hh/(hh+nv) if hh+nv else 0:4.1f}% of misses)  nvme={nv:6d}  "
          f"d2h={dh:6d}  ->  {1000.0/t_tok:5.3f} tok/s")
    return 1000.0 / t_tok

if __name__ == '__main__':
    keys = load(sys.argv[1])
    ntok = int(sys.argv[2])
    GPU = 3072
    print(f"{len(keys)} lookups, {ntok} decode tokens, GPU pool {GPU} slots\n")
    base = report("GPU only (shipped)", keys, GPU, 0, 'gpu', ntok)
    print()
    for host_cap in (2000, 4283, 5568):
        gb = host_cap * EXPERT_MB / 1000
        print(f"-- host slab {host_cap} slots ({gb:.0f} GB) --")
        v = report("  victim-fill (built, incorrect)", keys, GPU, host_cap, 'victim', ntok)
        f = report("  fetch-fill  (proposed)", keys, GPU, host_cap, 'fetch', ntok)
        print(f"  victim {v/base:.2f}x baseline | fetch {f/base:.2f}x baseline | "
              f"fetch/victim {f/v:.2f}x\n")
