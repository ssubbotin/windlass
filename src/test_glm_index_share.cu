/*
 * test_glm_index_share.cu — the IndexShare plumbing and the index_mask
 * composition, on synthetic data. No checkpoint, no oracle, no GPU memory to
 * speak of; runs alongside anything.
 *
 * Build: make test_glm_index_share
 * Run:   ./test_glm_index_share
 *
 * What is under test, and why each part is here:
 *
 *  A. Propagation across a whole group. transformers threads one variable
 *     through the layer loop (M:715/717/724); a "shared" layer passes it
 *     through UNCHANGED (M:417-419), so layers 3, 4 and 5 all consume layer 2's
 *     indices. The natural bug is consume-and-clear, which leaves the second
 *     and later members of every group with nothing. Checked on VALUES, not on
 *     printed text, and with a negative control that runs the same check
 *     against a deliberately clearing variant and requires it to fail.
 *
 *  B. A "shared" layer with nothing published must die (M:418 raises). Checked
 *     by forking, because the production behaviour is abort().
 *
 *  C. index_mask composition. Three properties, each with its own trap:
 *       - a full index set produces an all-zero mask and attention output that is
 *         BIT-IDENTICAL to the unmasked call. This is the mechanism behind
 *         Task 4's free oracle, so it gets its own assertion here where it can
 *         be run without a 95 GB checkpoint.
 *       - dropping ONE key must CHANGE the output. Without this the previous
 *         check passes trivially on a mask that was never wired into the
 *         kernel at all — which is exactly the failure mode Task 3's author hit
 *         with a fixture that could not fail.
 *       - a non-causal index (>= T, which the indexer genuinely returns —
 *         Task 1 measured [0,1,2,10] for query row 2) must unmask nothing and
 *         must not write outside the mask.
 *
 *  D. The global-memory fallback for the per-key score array must be
 *     bit-identical to the shared-memory path. Same arithmetic, different
 *     address space; if it is not identical, it is not a fallback.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <csignal>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "glm_primitives.cuh"
#include "glm_kernels.cuh"
#include "glm_loader.cuh"
#include "glm_layer_runner.cuh"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static int g_fail = 0;

static void check(bool ok, const char* what) {
    printf("[%s] %s\n", ok ? "OK  " : "FAIL", what);
    if (!ok) g_fail = 1;
}

// ---------------------------------------------------------------------------
// A. IndexShare propagation
// ---------------------------------------------------------------------------

// GLM-5.2's real shape: "full" at 0, 1, 2, 6, 10, ... 74; "shared" elsewhere.
// Built the way load_config builds it (leader = nearest preceding owner) so the
// test exercises the same ownership relation run_layer will see.
struct Ownership {
    std::vector<bool>     owner;
    std::vector<uint32_t> leader;
};

static Ownership glm_ownership(uint32_t n_layers) {
    Ownership o;
    o.owner.resize(n_layers);
    o.leader.resize(n_layers);
    uint32_t leader = UINT32_MAX;
    for (uint32_t i = 0; i < n_layers; i++) {
        // indexer_types[i] = "full" if (max(i-3+1,0) % 4) == 0  (Task 1)
        const uint32_t m = (i >= 2) ? (i - 2) : 0;
        o.owner[i] = (m % 4) == 0;
        if (o.owner[i]) leader = i;
        o.leader[i] = leader;
    }
    return o;
}

// One device buffer of indices per owning layer, each with CONTENT UNIQUE TO
// THAT LAYER. Uniqueness is the point: if a member were served a stale pointer
// from the previous group, or the buffers all held the same values, the
// value-level comparison below would pass anyway.
struct LeaderIndices {
    std::vector<int32_t> host;
    int32_t* dev = nullptr;
};

static LeaderIndices make_indices(uint32_t layer, uint32_t n) {
    LeaderIndices li;
    li.host.resize(n);
    for (uint32_t i = 0; i < n; i++)
        li.host[i] = (int32_t)(1000 * (layer + 1) + i * 7 + (layer % 5));
    CUDA_OK(cudaMalloc(&li.dev, n * sizeof(int32_t)));
    CUDA_OK(cudaMemcpy(li.dev, li.host.data(), n * sizeof(int32_t),
                       cudaMemcpyHostToDevice));
    return li;
}

// Walks the layer loop exactly as run_chain does. `clear_after_use` is the
// negative control: the consume-and-clear bug this test exists to catch.
// Returns false on the first violation.
static bool walk_group(const Ownership& o, uint32_t n_layers,
                       std::vector<LeaderIndices>& idx, uint32_t n_idx,
                       bool clear_after_use, std::string* why) {
    glm::IndexShare share;
    share.reset();
    for (uint32_t l = 0; l < n_layers; l++) {
        if (o.owner[l]) {
            share.publish(l, idx[l].dev, n_idx);
            continue;
        }
        // A "shared" layer: consult, do not modify.
        if (!share.idx) { *why = "layer " + std::to_string(l) + " got no indices"; return false; }
        const uint32_t want = o.leader[l];
        if (share.producer != want) {
            *why = "layer " + std::to_string(l) + " consumed layer " +
                   std::to_string(share.producer) + ", expected leader " +
                   std::to_string(want);
            return false;
        }
        if (share.idx != idx[want].dev) {
            *why = "layer " + std::to_string(l) + " has the wrong index pointer";
            return false;
        }
        if (share.n != n_idx) {
            *why = "layer " + std::to_string(l) + " has n=" + std::to_string(share.n);
            return false;
        }
        // Values, downloaded from the device — a pointer that merely looks right
        // is not evidence.
        std::vector<int32_t> got(n_idx);
        CUDA_OK(cudaMemcpy(got.data(), share.idx, n_idx * sizeof(int32_t),
                           cudaMemcpyDeviceToHost));
        if (std::memcmp(got.data(), idx[want].host.data(), n_idx * sizeof(int32_t))) {
            *why = "layer " + std::to_string(l) + " received altered index values";
            return false;
        }
        if (clear_after_use) share.reset();   // the bug
    }
    return true;
}

static void test_propagation() {
    const uint32_t N = 78, NI = 11;
    const Ownership o = glm_ownership(N);

    // Sanity on the fixture itself: the groups must be four layers wide, or the
    // "crosses the whole group" claim is untested. GLM-5.2's are {2,3,4,5},
    // {6,7,8,9}, ... with 0 and 1 owning theirs outright.
    uint32_t owners = 0, longest = 0, run = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (o.owner[i]) { owners++; run = 0; }
        else { run++; longest = (run > longest) ? run : longest; }
    }
    check(owners == 21, "fixture: 21 owning layers");
    check(o.owner[0] && o.owner[1] && o.owner[2] && o.owner[6] && o.owner[74],
          "fixture: owners at 0,1,2,6,74");
    check(longest == 3, "fixture: groups have 3 consuming members (propagation "
                        "must cross all of them, not just the first)");
    check(o.leader[3] == 2 && o.leader[4] == 2 && o.leader[5] == 2,
          "fixture: layers 3,4,5 all lead by layer 2");

    std::vector<LeaderIndices> idx(N);
    for (uint32_t i = 0; i < N; i++)
        if (o.owner[i]) idx[i] = make_indices(i, NI);

    std::string why;
    check(walk_group(o, N, idx, NI, /*clear_after_use=*/false, &why),
          why.empty() ? "propagation: every \"shared\" layer receives its leader's "
                        "indices unchanged, across the whole group"
                      : why.c_str());

    // NEGATIVE CONTROL. If the check above cannot fail, it is not a check. The
    // consume-and-clear variant must be caught, and must be caught at the SECOND
    // member of a group (layer 4), not somewhere incidental.
    why.clear();
    const bool caught = !walk_group(o, N, idx, NI, /*clear_after_use=*/true, &why);
    check(caught, "negative control: consume-and-clear is detected");
    check(caught && why.find("layer 4 ") == 0,
          caught ? ("negative control fails at layer 4 (\"" + why + "\")").c_str()
                 : "negative control fails at layer 4");

    for (uint32_t i = 0; i < N; i++) if (idx[i].dev) cudaFree(idx[i].dev);
}

// ---------------------------------------------------------------------------
// B. A "shared" layer with nothing published must die loudly
// ---------------------------------------------------------------------------
static void test_missing_indices_aborts() {
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        // Child: stderr is expected to carry the message; the exit status is
        // what is asserted on.
        glm::IndexShare share;
        share.reset();
        share.require(7);
        _exit(0);          // reached only if require() failed to abort
    }
    int status = 0;
    waitpid(pid, &status, 0);
    const bool died = (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) ||
                      (WIFEXITED(status) && WEXITSTATUS(status) != 0);
    check(died, "a \"shared\" layer with no published indices aborts "
                "(transformers raises, M:418)");

    // And the dense escape hatch must NOT abort — otherwise standalone
    // single-layer runs are impossible and the bit-identity gate has no dense
    // arm to compare against.
    fflush(stdout);
    pid = fork();
    if (pid == 0) {
        glm::IndexShare share;
        share.reset();
        share.dense = true;
        share.require(7);
        _exit(0);
    }
    waitpid(pid, &status, 0);
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "standalone-dense mode does not abort");
}

// ---------------------------------------------------------------------------
// C/D. index_mask composition, and the score-buffer fallback
// ---------------------------------------------------------------------------

static void fill_random(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < v.size(); i++) {
        s = s * 1664525u + 1013904223u;
        v[i] = ((float)(s >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
    }
}

struct AttnFixture {
    static constexpr uint32_t HN = 3, QK_NOPE = 4, QK_ROPE = 4, QK = 8, KVL = 6, T = 16;
    float *q = nullptr, *qabs = nullptr, *kvc = nullptr, *krot = nullptr, *ctx = nullptr;
    uint8_t* mask = nullptr;
    float* scores_g = nullptr;
    int32_t* idx = nullptr;

    void init() {
        std::vector<float> hq(HN * QK), hqa(HN * KVL), hkv(T * KVL), hkr(T * QK_ROPE);
        fill_random(hq, 1); fill_random(hqa, 2); fill_random(hkv, 3); fill_random(hkr, 4);
        auto up = [](float** d, const std::vector<float>& h) {
            CUDA_OK(cudaMalloc(d, h.size() * sizeof(float)));
            CUDA_OK(cudaMemcpy(*d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        };
        up(&q, hq); up(&qabs, hqa); up(&kvc, hkv); up(&krot, hkr);
        CUDA_OK(cudaMalloc(&ctx, HN * KVL * sizeof(float)));
        CUDA_OK(cudaMalloc(&scores_g, HN * T * sizeof(float)));
        CUDA_OK(cudaMalloc(&mask, T + 8));              // +8 = out-of-range canary
        CUDA_OK(cudaMalloc(&idx, (T + 8) * sizeof(int32_t)));
    }

    // Runs attention and returns ctx. `use_mask` selects the mask buffer or null;
    // `use_global` selects the global score buffer or dynamic shared memory.
    std::vector<float> run(bool use_mask, bool use_global) {
        const size_t smem = (size_t)T * sizeof(float);
        glm::mla_decode_absorbed<<<HN, 256, use_global ? 0 : smem>>>(
            q, qabs, kvc, krot, ctx, T, QK, QK_NOPE, QK_ROPE, KVL,
            1.0f / std::sqrt((float)QK),
            use_mask ? mask : nullptr, use_global ? scores_g : nullptr);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());
        std::vector<float> h(HN * KVL);
        CUDA_OK(cudaMemcpy(h.data(), ctx, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    }

    // Builds the mask from `sel`, with a canary past the end of the [T] region.
    std::vector<uint8_t> build(const std::vector<int32_t>& sel) {
        CUDA_OK(cudaMemset(mask, 0xAB, T + 8));         // canary
        CUDA_OK(cudaMemcpy(idx, sel.data(), sel.size() * sizeof(int32_t),
                           cudaMemcpyHostToDevice));
        glm::launch_build_index_mask(idx, (uint32_t)sel.size(), mask, T);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());
        std::vector<uint8_t> h(T + 8);
        CUDA_OK(cudaMemcpy(h.data(), mask, T + 8, cudaMemcpyDeviceToHost));
        return h;
    }

    void done() {
        for (float* p : {q, qabs, kvc, krot, ctx, scores_g}) cudaFree(p);
        cudaFree(mask); cudaFree(idx);
    }
};

static bool bitsame(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static void test_mask() {
    AttnFixture f;
    f.init();
    const uint32_t T = AttnFixture::T;

    // --- the T <= index_topk case: topk returns a permutation of ALL T keys ---
    // Shuffled on purpose: torch's topk is descending by score, so the index list
    // arrives in no particular positional order and the mask build must not care.
    std::vector<int32_t> all(T);
    for (uint32_t i = 0; i < T; i++) all[i] = (int32_t)((i * 7 + 3) % T);
    std::vector<uint8_t> m = f.build(all);
    bool all_clear = true;
    for (uint32_t i = 0; i < T; i++) if (m[i] != 0) all_clear = false;
    check(all_clear, "full index set -> mask is uniformly 0 (nothing dropped)");

    const std::vector<float> unmasked = f.run(/*use_mask=*/false, /*use_global=*/false);
    const std::vector<float> full_set = f.run(/*use_mask=*/true,  /*use_global=*/false);
    check(bitsame(unmasked, full_set),
          "full index set -> attention output BIT-IDENTICAL to the unmasked path");

    // --- the control that makes the line above mean something ---------------
    // If the mask were never wired into the kernel, "bit-identical" would hold
    // vacuously. Drop exactly one key and require the output to move.
    std::vector<int32_t> minus_one;
    for (uint32_t i = 0; i < T; i++) if (i != 5) minus_one.push_back((int32_t)i);
    m = f.build(minus_one);
    check(m[5] == 1, "dropping key 5 -> mask[5] == 1");
    bool others_clear = true;
    for (uint32_t i = 0; i < T; i++) if (i != 5 && m[i] != 0) others_clear = false;
    check(others_clear, "dropping key 5 -> every other key stays selected");
    const std::vector<float> dropped = f.run(true, false);
    check(!bitsame(unmasked, dropped),
          "negative control: dropping one key CHANGES the attention output "
          "(so the mask really reaches the arithmetic)");

    // --- non-causal indices ------------------------------------------------
    // The indexer genuinely returns indices past the causal frontier (Task 1
    // measured [0,1,2,10] for query row 2). They must unmask nothing, and they
    // must not write past the [T] mask region.
    std::vector<int32_t> noncausal;
    for (uint32_t i = 0; i + 1 < T; i++) noncausal.push_back((int32_t)i);
    noncausal.push_back((int32_t)(T + 3));      // beyond the causal frontier
    noncausal.push_back(-1);                    // and a defensive negative
    m = f.build(noncausal);
    check(m[T - 1] == 1,
          "a non-causal index does not stand in for the key it is not "
          "(key T-1 stays dropped)");
    bool canary_ok = true;
    for (uint32_t i = T; i < T + 8; i++) if (m[i] != 0xAB) canary_ok = false;
    check(canary_ok, "a non-causal index writes nothing past the [T] mask");

    // --- D. the global-memory score buffer ---------------------------------
    const std::vector<float> shared_mem = f.run(/*use_mask=*/false, /*use_global=*/false);
    const std::vector<float> global_mem = f.run(/*use_mask=*/false, /*use_global=*/true);
    check(bitsame(shared_mem, global_mem),
          "global-memory score buffer is BIT-IDENTICAL to the shared-memory path");

    f.done();
}

// ---------------------------------------------------------------------------
// E. Above index_topk — the regime the removed abort used to forbid
// ---------------------------------------------------------------------------
// Everything above runs at T = 16, where the mask is a no-op by construction.
// This is the other side: T larger than index_topk, so the top-k really selects,
// and larger than the shared-memory limit, so the score buffer really has to
// fall back to global memory. Synthetic scores — this says nothing about whether
// the indexer's numbers are right (Task 5), only that the machinery survives the
// shapes the 2048 abort used to prevent.
static void test_above_topk() {
    const uint32_t T = 13000;          // > 48 KB / 4 B, so shared memory cannot hold it
    const uint32_t K = 2048;           // index_topk
    const uint32_t HN = 2, QK_NOPE = 4, QK_ROPE = 4, QK = 8, KVL = 6;

    check((size_t)T * sizeof(float) > glm::mla_smem_capacity(),
          "fixture: T is past the shared-memory capacity, so the fallback is exercised");

    std::vector<float> hs(T);
    fill_random(hs, 99);
    float* d_scores = nullptr; int32_t* d_idx = nullptr; uint32_t* d_n = nullptr;
    uint8_t* d_mask = nullptr;
    CUDA_OK(cudaMalloc(&d_scores, T * sizeof(float)));
    CUDA_OK(cudaMemcpy(d_scores, hs.data(), T * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&d_idx, K * sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&d_n, sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_mask, T));

    glm::indexer_topk<<<1, 256>>>(d_scores, T, K, d_idx, d_n);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());
    uint32_t n = 0;
    CUDA_OK(cudaMemcpy(&n, d_n, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    check(n == K, "top-k over T > index_topk returns exactly index_topk indices");

    glm::launch_build_index_mask(d_idx, n, d_mask, T);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> hm(T);
    CUDA_OK(cudaMemcpy(hm.data(), d_mask, T, cudaMemcpyDeviceToHost));
    uint32_t kept = 0;
    for (uint32_t i = 0; i < T; i++) if (hm[i] == 0) kept++;
    check(kept == K, "the mask keeps exactly index_topk of T keys, drops the rest");

    // Attention at this T. The shared-memory launch MUST fail — that is why the
    // global buffer exists — and the global one must produce finite output that
    // differs from the unmasked result (the mask is doing real work here).
    std::vector<float> hq(HN * QK), hqa(HN * KVL), hkv((size_t)T * KVL),
                       hkr((size_t)T * QK_ROPE);
    fill_random(hq, 11); fill_random(hqa, 12); fill_random(hkv, 13); fill_random(hkr, 14);
    float *q = nullptr, *qabs = nullptr, *kvc = nullptr, *krot = nullptr,
          *ctx = nullptr, *sg = nullptr;
    auto up = [](float** d, const std::vector<float>& h) {
        CUDA_OK(cudaMalloc(d, h.size() * sizeof(float)));
        CUDA_OK(cudaMemcpy(*d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    };
    up(&q, hq); up(&qabs, hqa); up(&kvc, hkv); up(&krot, hkr);
    CUDA_OK(cudaMalloc(&ctx, HN * KVL * sizeof(float)));
    CUDA_OK(cudaMalloc(&sg, (size_t)HN * T * sizeof(float)));

    glm::mla_decode_absorbed<<<HN, 256, (size_t)T * sizeof(float)>>>(
        q, qabs, kvc, krot, ctx, T, QK, QK_NOPE, QK_ROPE, KVL,
        1.0f / std::sqrt((float)QK), nullptr, nullptr);
    const cudaError_t smem_err = cudaGetLastError();
    check(smem_err != cudaSuccess,
          "the shared-memory launch really does fail at this T (the fallback is "
          "not decoration)");

    auto run_global = [&](const uint8_t* mask) {
        glm::mla_decode_absorbed<<<HN, 256, 0>>>(
            q, qabs, kvc, krot, ctx, T, QK, QK_NOPE, QK_ROPE, KVL,
            1.0f / std::sqrt((float)QK), mask, sg);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());
        std::vector<float> h(HN * KVL);
        CUDA_OK(cudaMemcpy(h.data(), ctx, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return h;
    };
    const std::vector<float> dense_out  = run_global(nullptr);
    const std::vector<float> sparse_out = run_global(d_mask);
    bool finite = true;
    for (float v : sparse_out) if (!std::isfinite(v)) finite = false;
    check(finite, "sparse attention at T > index_topk produces finite output");
    check(!bitsame(dense_out, sparse_out),
          "sparse attention at T > index_topk differs from dense (2048 of 13000 "
          "keys attended)");

    for (float* p : {q, qabs, kvc, krot, ctx, sg, d_scores}) cudaFree(p);
    cudaFree(d_idx); cudaFree(d_n); cudaFree(d_mask);
}

// ---------------------------------------------------------------------------
// F. The shared-memory boundary — regression for a bug this test did not catch
// ---------------------------------------------------------------------------
// `MLA_SMEM_LIMIT` was originally 48 KB, on the reasoning that 48 KB is the
// per-block dynamic shared memory every architecture guarantees. It is not what
// THIS kernel can ask for: its two block reductions hold 32 floats each, and
// that 256 bytes of static __shared__ comes out of the same budget. The driver
// reports maxDynamicSharedSizeBytes = 48896 on SM 12.0, so every T in
// [12225, 12288] chose the shared-memory path and failed to launch with
// `invalid argument`. The T = 13000 test above did not catch it — 13000 is on
// the far side of the gap — and neither did anything else, because the gap is
// 64 values wide out of 32768.
//
// So: walk the actual boundary. run_layer's decision rule is reproduced exactly.
static void test_smem_boundary() {
    const size_t cap = glm::mla_smem_capacity();
    const uint32_t cap_T = (uint32_t)(cap / sizeof(float));
    printf("       (dynamic shared capacity = %zu B -> max shared-path T = %u; a "
           "48 KB assumption would have claimed %u)\n", cap, cap_T, 49152u / 4u);
    check(cap_T < 49152u / 4u,
          "the real capacity IS below the 48 KB assumption (that gap was the bug)");

    const uint32_t HN = 2, QK_NOPE = 4, QK_ROPE = 4, QK = 8, KVL = 6;
    bool all_ok = true;
    // One below, exactly at, and three above — the failing window was the few
    // values just past cap_T.
    const uint32_t probes[5] = {cap_T - 1, cap_T, cap_T + 1, cap_T + 2, cap_T + 64};
    for (uint32_t pi = 0; pi < 5; pi++) {
        const uint32_t T = probes[pi];
        std::vector<float> hq(HN * QK), hqa(HN * KVL), hkv((size_t)T * KVL),
                           hkr((size_t)T * QK_ROPE);
        fill_random(hq, 21); fill_random(hqa, 22); fill_random(hkv, 23); fill_random(hkr, 24);
        float *q = nullptr, *qabs = nullptr, *kvc = nullptr, *krot = nullptr,
              *ctx = nullptr, *sg = nullptr;
        auto up = [](float** d, const std::vector<float>& h) {
            CUDA_OK(cudaMalloc(d, h.size() * sizeof(float)));
            CUDA_OK(cudaMemcpy(*d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
        };
        up(&q, hq); up(&qabs, hqa); up(&kvc, hkv); up(&krot, hkr);
        CUDA_OK(cudaMalloc(&ctx, HN * KVL * sizeof(float)));
        CUDA_OK(cudaMalloc(&sg, (size_t)HN * T * sizeof(float)));

        // EXACTLY run_layer's rule.
        const size_t smem = (size_t)T * sizeof(float);
        float* scores_g = (smem > cap) ? sg : nullptr;
        glm::mla_decode_absorbed<<<HN, 256, scores_g ? 0 : smem>>>(
            q, qabs, kvc, krot, ctx, T, QK, QK_NOPE, QK_ROPE, KVL,
            1.0f / std::sqrt((float)QK), nullptr, scores_g);
        const cudaError_t e = cudaDeviceSynchronize();
        const cudaError_t e2 = cudaGetLastError();
        bool ok = (e == cudaSuccess && e2 == cudaSuccess);
        if (ok) {
            std::vector<float> h(HN * KVL);
            CUDA_OK(cudaMemcpy(h.data(), ctx, h.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (float v : h) if (!std::isfinite(v)) ok = false;
        }
        if (!ok) {
            printf("       T=%u (%s path) -> %s\n", T,
                   scores_g ? "global" : "shared", cudaGetErrorString(ok ? e : e2));
            all_ok = false;
        }
        for (float* p : {q, qabs, kvc, krot, ctx, sg}) cudaFree(p);
    }
    check(all_ok, "every T across the shared/global changeover launches and "
                  "produces finite output");
}

int main() {
    test_propagation();
    test_missing_indices_aborts();
    test_mask();
    test_above_topk();
    test_smem_boundary();
    printf("\n%s\n", g_fail ? "FAIL" : "ALL CHECKS OK");
    return g_fail;
}
