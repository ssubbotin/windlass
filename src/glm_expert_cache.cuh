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
    bool init(uint32_t capacity, const std::string& packed_dir,
              uint32_t first_layer, uint32_t num_layers,
              uint32_t io_threads = 0) {
        capacity_ = capacity;
        packed_dir_ = packed_dir;
        first_layer_ = first_layer;
        num_layers_ = num_layers;
        // open one file per layer up front (reused on every miss)
        layer_fds_.assign(num_layers, -1);
        for (uint32_t li = 0; li < num_layers; li++) {
            char p[1024]; snprintf(p, sizeof(p), "%s/layer_%u.bin",
                                    packed_dir.c_str(), first_layer + li);
            int fd = ::open(p, O_RDONLY);
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

        // pinned staging for the synchronous path
        if (cudaHostAlloc((void**)&sync_buf_, EXPERT_BYTES, cudaHostAllocDefault)
                != cudaSuccess) {
            fprintf(stderr, "ExpertCache: cudaHostAlloc(%.1f MB) failed\n",
                    (double)EXPERT_BYTES / 1e6);
            return false;
        }

        if (io_threads > 0 && !start_workers(io_threads)) return false;

        fprintf(stderr, "ExpertCache: %u slots x %.1f MB = %.2f GB pool, %u layer fds "
                "(first_layer=%u), io_threads=%u, pinned staging %.1f MB\n",
                capacity_, (double)EXPERT_BYTES / 1e6,
                (double)pool_bytes / 1e9, num_layers, first_layer, io_threads,
                (double)(EXPERT_BYTES * (1 + io_threads * BUFS_PER_WORKER)) / 1e6);
        return true;
    }

    void close_() {
        stop_workers();
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
            const uint32_t slot = alloc_slot();
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
    }

    void reset_stats() { hits_ = misses_ = evictions_ = 0; }

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
        std::atomic<bool> issued{false};
        std::atomic<bool> in_use{false};
    };
    struct Resolved { uint32_t key; uint32_t slot; IoTask* task; };

    uint8_t* slot_ptr(uint32_t slot) const { return pool_ + (size_t)slot * EXPERT_BYTES; }

    // Take a free slot, or evict the LRU tail. Never evicts an entry inserted
    // earlier in the current prefetch batch: those were push_front'ed and the
    // victim comes off the back.
    uint32_t alloc_slot() {
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
            CUDA_OK(cudaStreamWaitEvent(streams_[w], t->gate, 0));
            CUDA_OK(cudaMemcpyAsync(t->dst, buf, EXPERT_BYTES,
                                    cudaMemcpyHostToDevice, streams_[w]));
            CUDA_OK(cudaEventRecord(t->done, streams_[w]));
            CUDA_OK(cudaEventRecord(buf_ev_[base + turn], streams_[w]));
            used[turn] = true;
            turn = (turn + 1) % BUFS_PER_WORKER;
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
