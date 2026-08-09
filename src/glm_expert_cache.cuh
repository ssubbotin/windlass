/*
 * glm_expert_cache.cuh — LRU cache for routed expert weights resident in
 * VRAM, backed by a packed_experts_glm/layer_N.bin disk pool on miss.
 *
 * Each expert is a fixed 20,054,016-byte block (the layout written by
 * repack_experts_glm.py): gate_w | gate_s | up_w | up_s | down_w | down_s.
 * All six sub-tensors are uint8_t here (MXFP4 payload + E8M0 scale bytes) —
 * unlike the Qwen3.6 cache, none of them is __nv_bfloat16.
 *
 *   key   : (layer_idx - first_layer) * 256 + expert_idx
 *   value : device pointer to the expert's 20 MB block, sub-pointers at
 *           fixed offsets into it.
 *
 * GLM-5.2 layers 0-2 are dense and have no expert files at all, so init()
 * only opens fds for [first_layer, first_layer + num_layers) instead of
 * assuming layer_0.bin exists.
 *
 * Pool: one giant cudaMalloc of capacity × EXPERT_BYTES at startup; cache
 * slots are integer indices into the pool (no per-expert cudaMalloc/Free
 * churn — significant when active_set_size > capacity).
 *
 * Eviction: simple list-based LRU.  O(1) hit, O(1) miss-with-eviction.
 *
 * Disk reads use pread, not mmap — results.tsv in this repo records mmap as
 * a measured 5x regression against pread for this exact access pattern
 * (per-page faults on uncached multi-MB reads). Do not "improve" this.
 *
 * ---------------------------------------------------------------------------
 * TRANSFER PATH (Task 12). Task 11 measured expert fetch at 96.5% of layer
 * time, 5.2 GB/token at 3.6 GB/s. Three things were wrong with the original
 * path and all three are addressed here; the cache POLICY is untouched.
 *
 *   1. Staging was a plain std::vector<uint8_t>, i.e. pageable host memory, so
 *      every cudaMemcpy went through a driver bounce buffer. All staging is
 *      now cudaHostAlloc'd pinned memory.
 *   2. pread -> cudaMemcpy -> compute was fully serialised per expert, one
 *      expert at a time, on the caller's stream. prefetch() now resolves all K
 *      experts of a layer the instant routing is known and hands the misses to
 *      an I/O worker pool; each worker owns two pinned buffers and its own
 *      non-blocking stream, so the pread for expert k+1 overlaps the copy and
 *      the compute for expert k.
 *   3. The preads themselves were issued at queue depth 1. With io_threads > 1
 *      the K misses of a layer are in flight against the NVMe simultaneously.
 *
 * ORDERING, which is the part that fails intermittently if it is wrong:
 *
 *   - Worker streams are created cudaStreamNonBlocking. A stream created with
 *     plain cudaStreamCreate implicitly synchronises with the legacy default
 *     stream, which is the stream run_chain uses — that would be correct but
 *     would serialise away the entire benefit.
 *   - A miss may reuse a slot whose previous occupant is still being read by
 *     compute already enqueued on the caller's stream. prefetch() therefore
 *     records a gate event on the caller's stream BEFORE queueing any task,
 *     and every worker waits on that gate before its cudaMemcpyAsync. Copies
 *     for layer L thus start only after everything enqueued up to layer L's
 *     routing has retired — conservative, and it costs nothing measurable
 *     because compute is 3.5% of the time.
 *   - get_async() blocks until the worker has actually recorded the task's
 *     completion event, then makes the caller's stream wait on it. No host-side
 *     device sync, and the expert's kernels cannot start before its bytes land.
 *   - A worker only overwrites a pinned buffer after cudaEventSynchronize on
 *     the event recorded for that buffer's previous copy.
 *
 * io_threads == 0 keeps the fully synchronous path (pinned staging only), which
 * is what the Task 12 A/B measures optimisation 1 in isolation.
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <string>
#include <unordered_map>
#include <vector>
#include <list>
#include <deque>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "glm_layer_runner.cuh"   // glm::ExpertSource

#ifndef CUDA_OK
#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)
#endif

namespace glm {

class ExpertCache : public ExpertSource {
public:
    static constexpr size_t GW_OFF = 0;
    static constexpr size_t GW_LEN = 6291456;
    static constexpr size_t GS_OFF = GW_OFF + GW_LEN;
    static constexpr size_t GS_LEN = 393216;
    static constexpr size_t UW_OFF = GS_OFF + GS_LEN;
    static constexpr size_t UW_LEN = 6291456;
    static constexpr size_t US_OFF = UW_OFF + UW_LEN;
    static constexpr size_t US_LEN = 393216;
    static constexpr size_t DW_OFF = US_OFF + US_LEN;
    static constexpr size_t DW_LEN = 6291456;
    static constexpr size_t DS_OFF = DW_OFF + DW_LEN;
    static constexpr size_t DS_LEN = 393216;
    static constexpr size_t EXPERT_BYTES = DS_OFF + DS_LEN;     // 20,054,016

    static constexpr uint32_t BUFS_PER_WORKER = 2;   // double buffering
    static constexpr uint32_t NGATE = 4;             // gate-event ring
    static constexpr uint32_t NTASK = 64;            // task ring

    ExpertCache() = default;

    // capacity   = max number of (layer, expert) entries kept in VRAM.
    // packed_dir = directory holding layer_N.bin files.
    // first_layer, num_layers = fds are opened for layers
    //   [first_layer, first_layer + num_layers). GLM has no expert files for
    //   layers 0-2 (dense), so unlike the Qwen cache this never assumes
    //   layer_0.bin exists.
    // io_threads = size of the prefetch worker pool. 0 disables prefetch
    //   entirely and every miss is served synchronously (still out of pinned
    //   staging). Each worker costs BUFS_PER_WORKER * 20 MB of pinned host
    //   memory and one non-blocking CUDA stream.
    // host_gb  = size of the exclusive host victim slab, 0 disables it.
    // host_pin = cudaHostRegister the slab. Measured on this machine: pinning
    //   buys nothing, because the concurrent eviction traffic caps H2D at
    //   19.61 GB/s and pageable already reaches 19.94-29.17 GB/s. Kept as a
    //   flag so the claim stays falsifiable on other hardware.
    // o_direct = bypass the page cache. Buffered reads measure 6.7-6.9 GB/s
    //   against O_DIRECT's 9.86 even when the cache is warm, because every read
    //   pays copy_to_user. Only correct together with the slab, which is what
    //   replaces the page-cache hits this forfeits.
    bool init(uint32_t capacity, const std::string& packed_dir,
              uint32_t first_layer, uint32_t num_layers,
              uint32_t io_threads = 0,
              uint32_t host_gb = 0, bool host_pin = false,
              bool o_direct = false) {
        o_direct_ = o_direct;
        capacity_ = capacity;
        packed_dir_ = packed_dir;
        first_layer_ = first_layer;
        num_layers_ = num_layers;
        // open one file per layer up front (reused on every miss)
        layer_fds_.assign(num_layers, -1);
        for (uint32_t li = 0; li < num_layers; li++) {
            char p[1024]; snprintf(p, sizeof(p), "%s/layer_%u.bin",
                                    packed_dir.c_str(), first_layer + li);
            // The whole store is 4096-aligned by construction: EXPERT_BYTES is
            // 20,054,016 = 4896 x 4096 and expert e lives at e * EXPERT_BYTES,
            // so both offsets and lengths satisfy O_DIRECT. Destinations are
            // cudaHostAlloc'd (page-aligned) or the posix_memalign'd slab.
            int fd = ::open(p, O_RDONLY | (o_direct ? O_DIRECT : 0));
            if (fd < 0 && o_direct && errno == EINVAL) {
                fprintf(stderr, "ExpertCache: O_DIRECT unsupported on %s, falling back\n", p);
                o_direct_ = false;
                fd = ::open(p, O_RDONLY);
            }
            if (fd < 0) {
                fprintf(stderr, "ExpertCache: open %s failed: %s\n", p, std::strerror(errno));
                return false;
            }
            layer_fds_[li] = fd;
        }
        // allocate the VRAM pool: capacity * 20 MB
        size_t pool_bytes = (size_t)capacity_ * EXPERT_BYTES;
        if (cudaMalloc(&pool_, pool_bytes) != cudaSuccess) {
            fprintf(stderr, "ExpertCache: cudaMalloc(%.2f GB) failed; reduce --cache-experts\n",
                    (double)pool_bytes / 1e9);
            return false;
        }
        free_slots_.reserve(capacity_);
        for (uint32_t i = 0; i < capacity_; i++) free_slots_.push_back(i);
        slot_fill_ev_.assign(capacity_, nullptr);

        // pinned staging for the synchronous path
        if (cudaHostAlloc((void**)&sync_buf_, EXPERT_BYTES, cudaHostAllocDefault)
                != cudaSuccess) {
            fprintf(stderr, "ExpertCache: cudaHostAlloc(%.1f MB) failed\n",
                    (double)EXPERT_BYTES / 1e6);
            return false;
        }

        if (io_threads > 0 && !start_workers(io_threads)) return false;

        if (host_gb > 0 && !init_host_slab(host_gb, host_pin)) return false;

        fprintf(stderr, "ExpertCache: %u slots x %.1f MB = %.2f GB pool, %u layer fds "
                "(first_layer=%u), io_threads=%u, pinned staging %.1f MB\n",
                capacity_, (double)EXPERT_BYTES / 1e6,
                (double)pool_bytes / 1e9, num_layers, first_layer, io_threads,
                (double)(EXPERT_BYTES * (1 + io_threads * BUFS_PER_WORKER)) / 1e6);
        return true;
    }

    bool init_host_slab(uint32_t gb, bool pin) {
        const size_t bytes = (size_t)gb << 30;
        host_capacity_ = (uint32_t)(bytes / EXPERT_BYTES);
        if (host_capacity_ == 0) return true;
        const size_t want = (size_t)host_capacity_ * EXPERT_BYTES;
        // posix_memalign, not malloc: O_DIRECT reads never target the slab
        // today, but the slab is a copy destination for D2H and an alignment
        // guarantee here costs nothing and removes a whole class of later bug.
        void* p = nullptr;
        if (posix_memalign(&p, 4096, want) != 0 || !p) {
            fprintf(stderr, "ExpertCache: host slab %.1f GB allocation failed\n",
                    (double)want / 1e9);
            return false;
        }
        host_slab_ = (uint8_t*)p;
        // Touch every page now. An untouched reservation is not memory, and
        // faulting 100 GB lazily during decode would charge the fault to the
        // first token that happens to evict into it.
        std::memset(host_slab_, 0, want);
        if (pin) {
            if (cudaHostRegister(host_slab_, want, cudaHostRegisterDefault) != cudaSuccess) {
                fprintf(stderr, "ExpertCache: cudaHostRegister(%.1f GB) failed, "
                        "continuing unpinned\n", (double)want / 1e9);
                cudaGetLastError();
            } else host_pinned_ = true;
        }
        host_iter_.resize(host_capacity_);
        host_last_ev_.assign(host_capacity_, nullptr);
        host_free_.reserve(host_capacity_);
        for (uint32_t i = host_capacity_; i-- > 0; ) host_free_.push_back(i);
        fprintf(stderr,
            "\n*** ExpertCache: --host-cache IS KNOWN INCORRECT AND MUST NOT BE USED ***\n"
            "    It measures 2.24 tok/s against 1.06 baseline, so the headroom is real,\n"
            "    but greedy decode diverges from the baseline's token stream, which under\n"
            "    identical prompt and sampling can only mean the slab serves wrong bytes.\n"
            "    Three ordering defects were found and fixed and a fourth remains:\n"
            "      1. a taken slot returned to the free list while its H2D was pending;\n"
            "      2. no cross-stream ordering between a slot's D2H writer and H2D reader;\n"
            "      3. the eviction D2H not ordered behind the outgoing block's own fill.\n"
            "    Pageable memory hid all of them by making every copy synchronous.\n"
            "    Use --o-direct alone (+48%%, byte-identical output) until this is redesigned.\n\n");
        fprintf(stderr, "ExpertCache: host victim slab %u slots = %.2f GB (%s), "
                "exclusive of the %u-slot VRAM pool -> %u distinct experts resident "
                "(%.1f%% of the routed set)\n",
                host_capacity_, (double)want / 1e9,
                host_pinned_ ? "pinned" : "pageable",
                capacity_, capacity_ + host_capacity_,
                100.0 * (capacity_ + host_capacity_) / (256.0 * num_layers_));
        return true;
    }

    void close_() {
        stop_workers();
        if (host_slab_) {
            if (host_pinned_) cudaHostUnregister(host_slab_);
            free(host_slab_);
            host_slab_ = nullptr;
        }
        host_slot_for_.clear(); host_lru_.clear(); host_iter_.clear();
        host_free_.clear(); host_pending_.clear(); host_last_ev_.clear();
        for (int fd : layer_fds_) if (fd >= 0) ::close(fd);
        layer_fds_.clear();
        if (pool_) { cudaFree(pool_); pool_ = nullptr; }
        if (sync_buf_) { cudaFreeHost(sync_buf_); sync_buf_ = nullptr; }
        slot_for_.clear(); lru_.clear(); slot_iter_.clear(); free_slots_.clear();
        resolved_.clear();
    }

    ~ExpertCache() override { close_(); }

    // ---- prefetch ---------------------------------------------------------
    // Resolve all `n` experts of one layer at once, the moment routing is
    // known. Hits are bumped in the LRU here; misses get a VRAM slot, are
    // published into the map immediately (so this batch's own bookkeeping is
    // consistent) and their disk read + H2D copy is handed to the worker pool.
    // Accounting (hits_/misses_/evictions_) happens here exactly as the
    // synchronous path would have done it, so hit-rate figures stay comparable
    // to Task 11's.
    void prefetch(uint32_t layer, const int32_t* experts, uint32_t n,
                  cudaStream_t stream) override {
        resolved_.clear();
        if (workers_.empty() || capacity_ < n) return;   // synchronous fallback
        host_drain_pending();
        const uint32_t rel = layer - first_layer_;

        // Gate: nothing may be written into a reused slot before the compute
        // already enqueued on `stream` has retired.
        cudaEvent_t gate = gate_ev_[gate_i_];
        gate_i_ = (gate_i_ + 1) % NGATE;
        CUDA_OK(cudaEventRecord(gate, stream));

        for (uint32_t k = 0; k < n; k++) {
            const uint32_t e = (uint32_t)experts[k];
            const uint32_t key = rel * 256u + e;
            bool dup = false;
            for (const Resolved& r : resolved_) if (r.key == key) { dup = true; break; }
            if (dup) continue;                       // router never repeats, but be safe

            auto it = slot_for_.find(key);
            if (it != slot_for_.end()) {
                const uint32_t slot = it->second;
                lru_.splice(lru_.begin(), lru_, slot_iter_[slot]);
                slot_iter_[slot] = lru_.begin();
                hits_++;
                resolved_.push_back({key, slot, nullptr});
                continue;
            }
            // Take the block out of the host slab BEFORE allocating a VRAM slot:
            // allocating may evict, and the eviction could otherwise pick the
            // very host slot we are about to read from.
            uint32_t hs_taken = UINT32_MAX;
            uint8_t* hsrc = host_take(key, &hs_taken);

            uint32_t evicted_key = UINT32_MAX;
            const uint32_t slot = alloc_slot(&evicted_key);
            slot_for_[key] = slot;
            lru_.push_front(key);
            slot_iter_[slot] = lru_.begin();
            misses_++;

            IoTask* t = next_task();
            t->fd   = layer_fds_[rel];
            t->off  = (off_t)e * (off_t)EXPERT_BYTES;
            t->dst  = slot_ptr(slot);
            t->gate = gate;
            t->layer = layer; t->expert = e;
            t->host_src  = hsrc;
            // host_admit AFTER host_take, and the taken slot is on the pending
            // queue rather than the free list, so this can never alias hsrc.
            t->evict_dst = (evicted_key == UINT32_MAX) ? nullptr : host_admit(evicted_key);
            // Both directions register t->done as the slot's last toucher: the
            // H2D reads host_src, the D2H writes evict_dst, and both retire when
            // t->done does.
            // Read BEFORE overwriting: this is the fill event of the occupant
            // being evicted, not of the block we are about to load.
            t->evict_fill_wait = t->evict_dst ? slot_fill_ev_[slot] : nullptr;
            slot_fill_ev_[slot] = t->done;
            t->src_wait = host_last_ev_of(t->host_src);
            t->dst_wait = host_last_ev_of(t->evict_dst);
            host_mark(t->host_src,  t->done);
            host_mark(t->evict_dst, t->done);
            if (hs_taken != UINT32_MAX) host_pending_.push_back({hs_taken, t->done});
            t->issued.store(false, std::memory_order_relaxed);
            t->in_use.store(true,  std::memory_order_relaxed);
            resolved_.push_back({key, slot, t});
            {
                std::lock_guard<std::mutex> lk(qm_);
                q_.push_back(t);
            }
            qcv_.notify_one();
        }
    }

    // Consume a prefetched expert: block until its H2D copy has been issued,
    // then order `stream` behind that copy. Falls back to the fully
    // synchronous path for anything prefetch() did not resolve.
    uint8_t* get_async(uint32_t layer, uint32_t expert, cudaStream_t stream) override {
        const uint32_t key = (layer - first_layer_) * 256u + expert;
        for (Resolved& r : resolved_) {
            if (r.key != key) continue;
            if (r.task) {
                IoTask* t = r.task;
                {
                    std::unique_lock<std::mutex> lk(im_);
                    icv_.wait(lk, [t]{ return t->issued.load(std::memory_order_acquire); });
                }
                CUDA_OK(cudaStreamWaitEvent(stream, t->done, 0));
                t->in_use.store(false, std::memory_order_release);
                r.task = nullptr;
            }
            return slot_ptr(r.slot);
        }
        return get(layer, expert);
    }

    // Return device pointer to the expert's 20 MB block. Loads from disk on
    // miss, evicts LRU if cache is full. Fully synchronous; staging is pinned.
    uint8_t* get(uint32_t layer_idx, uint32_t expert_idx) override {
        uint32_t rel = layer_idx - first_layer_;
        uint32_t key = rel * 256u + expert_idx;
        auto it = slot_for_.find(key);
        if (it != slot_for_.end()) {
            // hit: bump to front of LRU
            uint32_t slot = it->second;
            lru_.splice(lru_.begin(), lru_, slot_iter_[slot]);
            slot_iter_[slot] = lru_.begin();
            hits_++;
            return slot_ptr(slot);
        }
        // miss: pick a slot
        uint32_t slot = alloc_slot();
        // load from disk + memcpy
        off_t off = (off_t)expert_idx * (off_t)EXPERT_BYTES;
        int fd = layer_fds_[rel];
        size_t total = 0;
        while (total < EXPERT_BYTES) {
            ssize_t got = ::pread(fd, sync_buf_ + total,
                                  EXPERT_BYTES - total, off + (off_t)total);
            if (got <= 0) {
                fprintf(stderr, "ExpertCache: pread l%u e%u: %s\n",
                        layer_idx, expert_idx, std::strerror(errno));
                std::exit(1);
            }
            total += (size_t)got;
        }
        CUDA_OK(cudaMemcpy(slot_ptr(slot), sync_buf_, EXPERT_BYTES,
                           cudaMemcpyHostToDevice));
        slot_for_[key] = slot;
        lru_.push_front(key);
        slot_iter_[slot] = lru_.begin();
        misses_++;
        return slot_ptr(slot);
    }

    void unpack(uint8_t* d_block,
                const uint8_t** gw, const uint8_t** gs,
                const uint8_t** uw, const uint8_t** us,
                const uint8_t** dw, const uint8_t** ds) const
    {
        *gw = d_block + GW_OFF; *gs = d_block + GS_OFF;
        *uw = d_block + UW_OFF; *us = d_block + US_OFF;
        *dw = d_block + DW_OFF; *ds = d_block + DS_OFF;
    }

    void print_stats(const char* tag) const {
        size_t total = hits_ + misses_;
        if (total == 0) { fprintf(stderr, "[cache %s] no requests\n", tag); return; }
        fprintf(stderr,
                "[cache %s] hits=%zu misses=%zu evictions=%zu hit_rate=%.1f%% used=%zu/%u\n",
                tag, hits_, misses_, evictions_,
                100.0 * (double)hits_ / (double)total,
                slot_for_.size(), capacity_);
        if (host_capacity_) {
            // host_hit_rate is deliberately reported against MISSES, not against
            // all lookups: it is the fraction of GPU misses the slab caught, and
            // that is the number the design was measured on (bar 0.40, oracle
            // 0.727). Reporting it against all lookups would flatter it.
            fprintf(stderr,
                    "[host %s] serves=%zu of %zu misses = %.1f%% | writes=%zu evictions=%zu "
                    "used=%zu/%u | NVMe-bound %zu (%.2f GB)\n",
                    tag, host_hits_, misses_,
                    misses_ ? 100.0 * (double)host_hits_ / (double)misses_ : 0.0,
                    host_writes_, host_evictions_, host_slot_for_.size(), host_capacity_,
                    misses_ - host_hits_,
                    (double)(misses_ - host_hits_) * EXPERT_BYTES / 1e9);
            fprintf(stderr,
                    "[host %s] write amplification %.2fx (%zu writes per %zu serves), "
                    "slot stalls %zu\n", tag,
                    host_hits_ ? (double)host_writes_ / (double)host_hits_ : 0.0,
                    host_writes_, host_hits_, host_stalls_);
        }
    }

    void reset_stats() {
        hits_ = misses_ = evictions_ = 0;
        host_hits_ = host_writes_ = host_evictions_ = 0;
        host_serves_.store(0);
    }

    // Accessors for tests that assert on counter deltas rather than parsing
    // print_stats' text.
    size_t hits() const { return hits_; }
    size_t misses() const { return misses_; }
    size_t evictions() const { return evictions_; }

private:
    struct IoTask {
        int         fd = -1;
        off_t       off = 0;
        uint8_t*    dst = nullptr;
        cudaEvent_t gate = nullptr;    // wait before the copy
        cudaEvent_t done = nullptr;    // recorded after the copy
        uint32_t    layer = 0, expert = 0;
        // Host victim slab, both directions. Either may be null.
        uint8_t*    host_src = nullptr;   // serve this miss from the slab, skip the disk
        uint8_t*    evict_dst = nullptr;  // write dst's current occupant here before overwriting
        // The slab slots are shared across worker streams, so a copy that
        // touches one must first wait on whatever copy touched it last. Nothing
        // else orders them: recording the event is not the same as waiting on
        // it, and getting that wrong produces divergent output rather than an
        // error.
        cudaEvent_t src_wait = nullptr;
        cudaEvent_t dst_wait = nullptr;
        // The eviction D2H reads the GPU slot's OUTGOING occupant. That
        // occupant's own fill may still be in flight — a block fetched a few
        // layers ago can already be the LRU tail — in which case the slot holds
        // stale bytes and the slab would be handed 20 MB of the wrong expert
        // under the right key. Silent, and it only shows up as divergent output.
        cudaEvent_t evict_fill_wait = nullptr;
        std::atomic<bool> issued{false};
        std::atomic<bool> in_use{false};
    };
    struct Resolved { uint32_t key; uint32_t slot; IoTask* task; };

    uint8_t* slot_ptr(uint32_t slot) const { return pool_ + (size_t)slot * EXPERT_BYTES; }
    uint8_t* host_ptr(uint32_t hs) const { return host_slab_ + (size_t)hs * EXPERT_BYTES; }

    // ---- host victim slab --------------------------------------------------
    // Exclusive: it holds only what the VRAM pool has evicted, so its capacity
    // ADDS to the pool's rather than mirroring it. Measured on a real routing
    // trace, exclusivity is worth 0.4999 hit-on-miss against 0.2124 for a
    // mirroring tier of the same size — it carries 58% of the whole effect,
    // because an inclusive tier spends its first `capacity_` slots on blocks
    // the GPU already has.
    //
    // Filling it costs one 20 MB D2H per eviction. That is only affordable
    // because H2D and D2H measure 19.61 and 19.57 GB/s CONCURRENTLY on this
    // card (39.18 GB/s aggregate), both still ~2x the 9.86 GB/s the NVMe
    // delivers for this access pattern.

    // Reserve a host slot for `key`, evicting the host LRU tail if needed.
    uint8_t* host_admit(uint32_t key) {
        if (!host_slab_ || host_capacity_ == 0) return nullptr;
        auto it = host_slot_for_.find(key);
        if (it != host_slot_for_.end()) {          // already there, refresh it
            host_lru_.splice(host_lru_.begin(), host_lru_, host_iter_[it->second]);
            host_iter_[it->second] = host_lru_.begin();
            return host_ptr(it->second);
        }
        // A slot may not be handed out while a copy on some other worker stream
        // is still reading from or writing to it. Slots therefore never go
        // straight from the LRU to a caller: they pass through host_pending_,
        // carrying the event of the last copy that touched them, and only reach
        // the free list once that event has retired.
        //
        // Pageable memory hid this bug completely, because a pageable
        // cudaMemcpyAsync is synchronous and every copy was finished before the
        // next call returned. Pinning made the copies genuinely async and the
        // race appeared immediately as divergent greedy output.
        host_drain_pending();
        if (host_free_.empty()) {
            const uint32_t victim = host_lru_.back();
            const uint32_t vs = host_slot_for_[victim];
            host_slot_for_.erase(victim);
            host_lru_.pop_back();
            host_evictions_++;
            host_pending_.push_back({vs, host_last_ev_[vs]});
            host_drain_pending();
            if (host_free_.empty()) {
                // Everything in flight. Wait for the oldest copy rather than
                // hand out a slot someone is still using.
                CUDA_OK(cudaEventSynchronize(host_pending_.front().ev));
                host_stalls_++;
                host_drain_pending();
            }
        }
        const uint32_t hs = host_free_.back(); host_free_.pop_back();
        host_slot_for_[key] = hs;
        host_lru_.push_front(key);
        host_iter_[hs] = host_lru_.begin();
        host_writes_++;
        return host_ptr(hs);
    }

    // Take `key` OUT of the slab. Exclusive means a block promoted back to VRAM
    // must not stay here — leaving it would silently convert the tier to the
    // mirroring design that measured less than half as well.
    //
    // The slot is NOT returned to the free list here. Its bytes are the source
    // of an H2D copy that has only been ENQUEUED, so releasing it immediately
    // lets a later host_admit in this same 8-expert batch hand the same slot out
    // as an eviction destination — and the D2H then overwrites the bytes the
    // pending H2D still has to read. That is not a rare race: within one batch
    // the freed slot is at the top of the free list and gets picked first.
    //
    // Caught by output divergence under greedy decode, which is the cheapest
    // correctness gate this engine has and the only one that would have noticed.
    // Slots go on a pending queue and are released once their copy's event has
    // actually completed.
    uint8_t* host_take(uint32_t key, uint32_t* slot_out) {
        *slot_out = UINT32_MAX;
        if (!host_slab_ || host_capacity_ == 0) return nullptr;
        auto it = host_slot_for_.find(key);
        if (it == host_slot_for_.end()) return nullptr;
        const uint32_t hs = it->second;
        host_lru_.erase(host_iter_[hs]);
        host_slot_for_.erase(it);
        host_hits_++;
        *slot_out = hs;
        return host_ptr(hs);
    }

    // Release host slots whose promoting copy has retired. Conservative by
    // construction: `ev` is recycled with its IoTask, and a recycled event can
    // only have been re-recorded LATER, so a completed query is never early.
    void host_drain_pending() {
        while (!host_pending_.empty()) {
            const PendingFree& p = host_pending_.front();
            if (p.ev && cudaEventQuery(p.ev) != cudaSuccess) break;
            host_free_.push_back(p.hs);
            host_pending_.pop_front();
        }
    }

    // Record which copy last touched slot `hs`, in either direction. Anything
    // that later wants the slot must wait on this before reusing it.
    cudaEvent_t host_last_ev_of(uint8_t* p) const {
        if (!p) return nullptr;
        return host_last_ev_[(size_t)(p - host_slab_) / EXPERT_BYTES];
    }

    void host_mark(uint8_t* p, cudaEvent_t ev) {
        if (!p) return;
        host_last_ev_[(size_t)(p - host_slab_) / EXPERT_BYTES] = ev;
    }

    // Take a free slot, or evict the LRU tail. Never evicts an entry inserted
    // earlier in the current prefetch batch: those were push_front'ed and the
    // victim comes off the back.
    // `evicted_key_out`, when non-null, receives the key whose slot was reused,
    // or UINT32_MAX if a free slot was available. The caller needs it to decide
    // whether the outgoing 20 MB is worth saving into the host slab.
    uint32_t alloc_slot(uint32_t* evicted_key_out = nullptr) {
        if (evicted_key_out) *evicted_key_out = UINT32_MAX;
        if (!free_slots_.empty()) {
            uint32_t s = free_slots_.back(); free_slots_.pop_back();
            slot_iter_.resize(std::max((size_t)s + 1, slot_iter_.size()));
            return s;
        }
        uint32_t evict_key = lru_.back();
        uint32_t evict_slot = slot_for_[evict_key];
        slot_for_.erase(evict_key);
        lru_.pop_back();
        evictions_++;
        if (evicted_key_out) *evicted_key_out = evict_key;
        return evict_slot;
    }

    IoTask* next_task() {
        for (uint32_t tries = 0; tries < NTASK * 4; tries++) {
            IoTask* t = &tasks_[task_i_];
            task_i_ = (task_i_ + 1) % NTASK;
            if (!t->in_use.load(std::memory_order_acquire)) return t;
        }
        fprintf(stderr, "ExpertCache: task ring exhausted (should be impossible)\n");
        std::exit(1);
    }

    bool start_workers(uint32_t n) {
        tasks_ = new IoTask[NTASK];
        for (uint32_t i = 0; i < NTASK; i++)
            CUDA_OK(cudaEventCreateWithFlags(&tasks_[i].done, cudaEventDisableTiming));
        for (uint32_t i = 0; i < NGATE; i++)
            CUDA_OK(cudaEventCreateWithFlags(&gate_ev_[i], cudaEventDisableTiming));

        streams_.resize(n);
        bufs_.resize((size_t)n * BUFS_PER_WORKER, nullptr);
        buf_ev_.resize((size_t)n * BUFS_PER_WORKER, nullptr);
        for (uint32_t w = 0; w < n; w++) {
            CUDA_OK(cudaStreamCreateWithFlags(&streams_[w], cudaStreamNonBlocking));
            for (uint32_t b = 0; b < BUFS_PER_WORKER; b++) {
                const size_t i = (size_t)w * BUFS_PER_WORKER + b;
                if (cudaHostAlloc((void**)&bufs_[i], EXPERT_BYTES, cudaHostAllocDefault)
                        != cudaSuccess) {
                    fprintf(stderr, "ExpertCache: cudaHostAlloc for io worker %u failed; "
                                    "lower --io-threads\n", w);
                    return false;
                }
                CUDA_OK(cudaEventCreateWithFlags(&buf_ev_[i], cudaEventDisableTiming));
            }
        }
        stop_ = false;
        for (uint32_t w = 0; w < n; w++)
            workers_.emplace_back([this, w]{ this->worker_loop(w); });
        return true;
    }

    void stop_workers() {
        if (workers_.empty() && !tasks_) return;
        {
            std::lock_guard<std::mutex> lk(qm_);
            stop_ = true;
        }
        qcv_.notify_all();
        for (auto& th : workers_) if (th.joinable()) th.join();
        workers_.clear();
        for (size_t i = 0; i < bufs_.size(); i++) {
            if (buf_ev_[i]) cudaEventDestroy(buf_ev_[i]);
            if (bufs_[i])   cudaFreeHost(bufs_[i]);
        }
        bufs_.clear(); buf_ev_.clear();
        for (auto s : streams_) if (s) cudaStreamDestroy(s);
        streams_.clear();
        if (tasks_) {
            for (uint32_t i = 0; i < NTASK; i++)
                if (tasks_[i].done) cudaEventDestroy(tasks_[i].done);
            delete[] tasks_; tasks_ = nullptr;
        }
        for (uint32_t i = 0; i < NGATE; i++)
            if (gate_ev_[i]) { cudaEventDestroy(gate_ev_[i]); gate_ev_[i] = nullptr; }
        q_.clear();
    }

    void worker_loop(uint32_t w) {
        const size_t base = (size_t)w * BUFS_PER_WORKER;
        bool used[BUFS_PER_WORKER] = {false};
        uint32_t turn = 0;
        for (;;) {
            IoTask* t = nullptr;
            {
                std::unique_lock<std::mutex> lk(qm_);
                qcv_.wait(lk, [this]{ return stop_ || !q_.empty(); });
                if (stop_ && q_.empty()) return;
                t = q_.front(); q_.pop_front();
            }
            // The gate guards the slot both ways: nothing may be read out of it
            // or written into it until the compute already enqueued on the
            // caller's stream has retired.
            CUDA_OK(cudaStreamWaitEvent(streams_[w], t->gate, 0));

            // Save the outgoing occupant into the host slab before it is
            // overwritten. Same stream as the H2D that follows, so the ordering
            // is structural rather than something a future edit can transpose.
            if (t->evict_dst) {
                if (t->evict_fill_wait)
                    CUDA_OK(cudaStreamWaitEvent(streams_[w], t->evict_fill_wait, 0));
                if (t->dst_wait) CUDA_OK(cudaStreamWaitEvent(streams_[w], t->dst_wait, 0));
                CUDA_OK(cudaMemcpyAsync(t->evict_dst, t->dst, EXPERT_BYTES,
                                        cudaMemcpyDeviceToHost, streams_[w]));
            }

            if (t->host_src) {
                // Slab hit. The bytes are already in host memory, so the pinned
                // staging buffer and the disk are both skipped entirely.
                host_serves_++;
                if (t->src_wait) CUDA_OK(cudaStreamWaitEvent(streams_[w], t->src_wait, 0));
                CUDA_OK(cudaMemcpyAsync(t->dst, t->host_src, EXPERT_BYTES,
                                        cudaMemcpyHostToDevice, streams_[w]));
            } else {
                uint8_t* buf = bufs_[base + turn];
                // Do not overwrite a buffer whose previous H2D copy is still live.
                if (used[turn]) CUDA_OK(cudaEventSynchronize(buf_ev_[base + turn]));

                size_t total = 0;
                while (total < EXPERT_BYTES) {
                    ssize_t got = ::pread(t->fd, buf + total, EXPERT_BYTES - total,
                                          t->off + (off_t)total);
                    if (got <= 0) {
                        fprintf(stderr, "ExpertCache: pread l%u e%u: %s\n",
                                t->layer, t->expert, std::strerror(errno));
                        std::exit(1);
                    }
                    total += (size_t)got;
                }
                CUDA_OK(cudaMemcpyAsync(t->dst, buf, EXPERT_BYTES,
                                        cudaMemcpyHostToDevice, streams_[w]));
                CUDA_OK(cudaEventRecord(buf_ev_[base + turn], streams_[w]));
                used[turn] = true;
                turn = (turn + 1) % BUFS_PER_WORKER;
            }
            CUDA_OK(cudaEventRecord(t->done, streams_[w]));
            {
                std::lock_guard<std::mutex> lk(im_);
                t->issued.store(true, std::memory_order_release);
            }
            icv_.notify_all();
        }
    }

    uint32_t capacity_ = 0;
    std::string packed_dir_;
    uint32_t first_layer_ = 0;
    uint32_t num_layers_ = 0;
    std::vector<int> layer_fds_;
    uint8_t* pool_ = nullptr;
    uint8_t* sync_buf_ = nullptr;               // pinned, synchronous path
    // (layer - first_layer, expert) → slot index
    std::unordered_map<uint32_t, uint32_t> slot_for_;
    // LRU: front = most recent, back = oldest. Holds the (rel_layer*256+expert) keys.
    std::list<uint32_t> lru_;
    // For O(1) splice on hit: per-slot iterator into lru_
    std::vector<std::list<uint32_t>::iterator> slot_iter_;
    std::vector<uint32_t> free_slots_;
    size_t hits_ = 0, misses_ = 0, evictions_ = 0;

    // --- host victim slab ---------------------------------------------------
    bool      o_direct_ = false;
    uint8_t*  host_slab_ = nullptr;
    uint32_t  host_capacity_ = 0;
    bool      host_pinned_ = false;
    std::unordered_map<uint32_t, uint32_t> host_slot_for_;
    std::list<uint32_t>                    host_lru_;
    std::vector<std::list<uint32_t>::iterator> host_iter_;
    std::vector<cudaEvent_t> host_last_ev_;   // last copy to touch each slab slot
    std::vector<cudaEvent_t> slot_fill_ev_;   // last copy to fill each VRAM slot
    std::vector<uint32_t> host_free_;
    struct PendingFree { uint32_t hs; cudaEvent_t ev; };
    std::deque<PendingFree> host_pending_;
    size_t host_hits_ = 0, host_writes_ = 0, host_evictions_ = 0, host_stalls_ = 0;
    std::atomic<size_t> host_serves_{0};

    // --- prefetch machinery (empty/idle when io_threads == 0) ---------------
    std::vector<Resolved>      resolved_;       // this layer's K entries
    IoTask*                    tasks_ = nullptr;
    uint32_t                   task_i_ = 0;
    cudaEvent_t                gate_ev_[NGATE] = {nullptr, nullptr, nullptr, nullptr};
    uint32_t                   gate_i_ = 0;
    std::vector<cudaStream_t>  streams_;
    std::vector<uint8_t*>      bufs_;
    std::vector<cudaEvent_t>   buf_ev_;
    std::vector<std::thread>   workers_;
    std::deque<IoTask*>        q_;
    std::mutex                 qm_;
    std::condition_variable    qcv_;
    std::mutex                 im_;
    std::condition_variable    icv_;
    bool                       stop_ = false;
};

} // namespace glm
