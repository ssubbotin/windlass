/*
 * test_glm_expert_cache.cu — LRU behaviour and disk-read correctness for the
 * GLM expert cache.
 *
 * Build: make test_glm_expert_cache
 * Run:   ./test_glm_expert_cache --packed-dir ./packed_experts --layer 3
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cerrno>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "glm_layer_runner.cuh"
#include "glm_expert_cache.cuh"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static bool g_failed = false;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); g_failed = true; } \
    else { fprintf(stderr, "OK: %s\n", msg); } \
} while (0)

int main(int argc, char** argv) {
    std::string packed_dir;
    uint32_t layer = 3;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--packed-dir") && i + 1 < argc) packed_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--layer") && i + 1 < argc) layer = (uint32_t)atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (packed_dir.empty()) {
        fprintf(stderr, "usage: %s --packed-dir DIR [--layer N]\n", argv[0]);
        return 1;
    }

    // ---- Case 1: cold miss then hit ----------------------------------
    {
        glm::ExpertCache cache;
        bool ok = cache.init(/*capacity=*/8, packed_dir, /*first_layer=*/layer, /*num_layers=*/1);
        if (!ok) { fprintf(stderr, "init failed\n"); return 1; }

        cache.get(layer, 0);   // miss: nothing resident yet
        cache.get(layer, 0);   // hit: same (layer, expert) as above
        cache.print_stats("case1");

        CHECK(cache.hits() == 1 && cache.misses() == 1,
              "case1: cold miss then hit -> hits=1 misses=1");
        cache.close_();
    }

    // ---- Case 2: contents match the file ------------------------------
    {
        glm::ExpertCache cache;
        bool ok = cache.init(/*capacity=*/8, packed_dir, /*first_layer=*/layer, /*num_layers=*/1);
        if (!ok) { fprintf(stderr, "init failed\n"); return 1; }

        uint8_t* d_block = cache.get(layer, 7);
        uint8_t h_from_cache[64];
        CUDA_OK(cudaMemcpy(h_from_cache, d_block, sizeof(h_from_cache), cudaMemcpyDeviceToHost));

        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%u.bin", packed_dir.c_str(), layer);
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, std::strerror(errno)); return 1; }
        uint8_t h_from_file[64];
        off_t off = (off_t)7 * (off_t)glm::ExpertCache::EXPERT_BYTES;
        ssize_t got = ::pread(fd, h_from_file, sizeof(h_from_file), off);
        ::close(fd);
        if (got != (ssize_t)sizeof(h_from_file)) {
            fprintf(stderr, "pread short: %zd\n", got); return 1;
        }

        CHECK(std::memcmp(h_from_cache, h_from_file, sizeof(h_from_cache)) == 0,
              "case2: cached block matches pread of layer file at expert offset");
        cache.close_();
    }

    // ---- Case 3: eviction is LRU ---------------------------------------
    {
        glm::ExpertCache cache;
        bool ok = cache.init(/*capacity=*/4, packed_dir, /*first_layer=*/layer, /*num_layers=*/1);
        if (!ok) { fprintf(stderr, "init failed\n"); return 1; }

        cache.get(layer, 0);
        cache.get(layer, 1);
        cache.get(layer, 2);
        cache.get(layer, 3);   // fills capacity 4: [0,1,2,3]
        cache.get(layer, 0);   // hit, bumps 0 to front: order front->back [0,3,2,1]

        cache.reset_stats();
        cache.get(layer, 4);   // miss, evicts LRU tail (1): [4,0,3,2]
        size_t misses_after_4 = cache.misses();
        size_t evictions_after_4 = cache.evictions();

        size_t hits_before = cache.hits();
        cache.get(layer, 0);   // must HIT
        size_t hits_after_0 = cache.hits();

        size_t misses_before_1 = cache.misses();
        cache.get(layer, 1);   // must MISS
        size_t misses_after_1 = cache.misses();

        cache.print_stats("case3");

        CHECK(misses_after_4 == 1 && evictions_after_4 == 1,
              "case3: get(4) causes exactly one miss and one eviction");
        CHECK(hits_after_0 == hits_before + 1,
              "case3: get(0) after get(4) is a HIT (was bumped to front, not evicted)");
        CHECK(misses_after_1 == misses_before_1 + 1,
              "case3: get(1) after get(4) is a MISS (was the true LRU tail, evicted)");

        cache.close_();
    }

    // ---- Case 4: prefetch + get_async (Task 12 async path) ---------------
    // Same LRU semantics as the synchronous path, plus: every block that
    // get_async hands back must contain the right bytes, with the worker pool
    // running at queue depth 4 and eviction churning slots underneath it. A
    // stream-ordering bug here shows up as an occasional wrong block, so the
    // batches are repeated with a rotating expert set rather than run once.
    {
        glm::ExpertCache cache;
        const uint32_t CAP = 16, K = 8, T = 4;
        bool ok = cache.init(CAP, packed_dir, /*first_layer=*/layer,
                             /*num_layers=*/1, /*io_threads=*/T);
        if (!ok) { fprintf(stderr, "init failed\n"); return 1; }

        char path[1024];
        snprintf(path, sizeof(path), "%s/layer_%u.bin", packed_dir.c_str(), layer);
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, std::strerror(errno)); return 1; }

        cudaStream_t s; CUDA_OK(cudaStreamCreate(&s));
        bool content_ok = true;
        // 24 batches of 8 drawn from a 20-expert pool against a 16-slot cache,
        // so consecutive batches overlap (hits) while the pool still overflows
        // the cache (misses + eviction).
        for (uint32_t b = 0; b < 24 && content_ok; b++) {
            int32_t idx[K];
            for (uint32_t k = 0; k < K; k++) idx[k] = (int32_t)((b * 3 + k) % 20);
            cache.prefetch(layer, idx, K, s);
            for (uint32_t k = 0; k < K; k++) {
                uint8_t* d = cache.get_async(layer, (uint32_t)idx[k], s);
                uint8_t h_dev[64], h_file[64];
                // Same stream as the copy the cache ordered against: this is
                // exactly how run_moe consumes the block.
                CUDA_OK(cudaMemcpyAsync(h_dev, d, sizeof(h_dev),
                                        cudaMemcpyDeviceToHost, s));
                CUDA_OK(cudaStreamSynchronize(s));
                off_t off = (off_t)idx[k] * (off_t)glm::ExpertCache::EXPERT_BYTES;
                if (::pread(fd, h_file, sizeof(h_file), off) != (ssize_t)sizeof(h_file)) {
                    fprintf(stderr, "pread short\n"); content_ok = false; break;
                }
                if (std::memcmp(h_dev, h_file, sizeof(h_dev)) != 0) {
                    fprintf(stderr, "batch %u expert %d: block content mismatch\n", b, idx[k]);
                    content_ok = false; break;
                }
            }
        }
        ::close(fd);
        CUDA_OK(cudaStreamSynchronize(s));
        cache.print_stats("case4");
        CHECK(content_ok, "case4: every prefetch/get_async block matches the file "
                          "(24 batches x 8 experts, 4 io threads, eviction churning)");
        CHECK(cache.hits() + cache.misses() == 24 * K,
              "case4: prefetch accounts every request exactly once");
        CHECK(cache.hits() > 0 && cache.misses() > 0,
              "case4: the batch pattern produced both hits and misses");
        CUDA_OK(cudaStreamDestroy(s));
        cache.close_();
    }

    if (g_failed) {
        fprintf(stderr, "FAIL\n");
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
