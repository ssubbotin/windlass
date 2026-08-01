/*
 * test_glm_indexer_loader.cu — DSA indexer weight-loading test (Task 2).
 *
 * No forward pass, no oracle fixtures -- just the loader. Asserts on values
 * read back into C++ structures, not on printed text:
 *
 *   1. Exactly 21 in-scope owners, at the checkpoint's known layer indices.
 *   2. Every layer's Config::indexer_leader resolves to the correct group
 *      leader (the nearest owning layer at or before it).
 *   3. Layer 78 (the MTP head; has indexer tensors in the checkpoint but no
 *      indexer_types entry) is refused by load_layer -- never loaded.
 *   4. An owning layer (2) loads all five indexer tensors non-null; the
 *      indexer KV cache is allocated. A non-owning layer (3) loads them all
 *      null, and its group leader is layer 2.
 *   5. Shapes and dtypes match the spec, read directly from the checkpoint's
 *      own tensor headers (independent of load_layer's internal asserts).
 *
 * Build: make test_glm_indexer_loader
 * Run:   ./test_glm_indexer_loader --model-dir /home/user1/glm52-mxfp4
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "safetensors_io.cuh"
#include "glm_loader.cuh"

static int g_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL(%d): ", __LINE__); \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n"); \
        g_fail = 1; \
    } \
} while (0)

int main(int argc, char** argv) {
    std::string model_dir;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--model-dir") && i + 1 < argc) model_dir = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (model_dir.empty()) {
        fprintf(stderr, "usage: %s --model-dir DIR\n", argv[0]);
        return 1;
    }

    glm::Config c;
    if (!glm::load_config(model_dir, &c)) { fprintf(stderr, "load_config failed\n"); return 1; }
    c.max_seq = 64;   // small; this test only needs alloc_scratch to succeed

    printf("config: n_layers=%u index_topk=%u index_n_heads=%u index_head_dim=%u\n",
           c.n_layers, c.index_topk, c.index_n_heads, c.index_head_dim);

    // --- 1. ownership map: exactly 21 owners at the known indices --------------
    // Sourced from config.json's own indexer_types, verified once by hand
    // (Task 1 / task-1-report.md) -- this is test data confirming the parser
    // read the checkpoint correctly, not a formula the loader relies on.
    static const uint32_t EXPECTED_OWNERS[] = {
        0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74
    };
    const uint32_t N_EXPECTED = (uint32_t)(sizeof(EXPECTED_OWNERS) / sizeof(EXPECTED_OWNERS[0]));
    CHECK(N_EXPECTED == 21, "test fixture itself is wrong: expected list has %u entries", N_EXPECTED);

    std::vector<uint32_t> owners;
    for (uint32_t i = 0; i < c.n_layers; i++) if (c.indexer_owner[i]) owners.push_back(i);
    CHECK(owners.size() == 21, "expected 21 owners, got %zu", owners.size());
    for (uint32_t i = 0; i < N_EXPECTED && i < owners.size(); i++) {
        CHECK(owners[i] == EXPECTED_OWNERS[i], "owner[%u]: expected layer %u, got %u",
              i, EXPECTED_OWNERS[i], owners[i]);
    }

    // --- 2. group leader for every layer ----------------------------------------
    for (uint32_t i = 0; i < c.n_layers; i++) {
        uint32_t expect_leader = UINT32_MAX;
        for (uint32_t k = 0; k < N_EXPECTED; k++)
            if (EXPECTED_OWNERS[k] <= i) expect_leader = EXPECTED_OWNERS[k];
        CHECK(c.indexer_leader[i] == expect_leader,
              "layer %u: expected group leader %u, got %u", i, expect_leader, c.indexer_leader[i]);
        if (c.indexer_owner[i])
            CHECK(c.indexer_leader[i] == i, "owning layer %u must lead itself, got leader %u",
                  i, c.indexer_leader[i]);
    }
    printf("ownership map: %zu owners, all group leaders correct\n", owners.size());

    st::ModelDir M;
    if (!st::open(&M, model_dir)) { fprintf(stderr, "st::open failed\n"); return 1; }

    // --- 3. layer 78 (the MTP head) must never load -----------------------------
    // Checked two ways: the ownership map has no opinion about it (it is out of
    // the n_layers range entirely), and load_layer refuses it outright.
    CHECK(c.n_layers == 78, "expected n_layers=78 on this checkpoint, got %u", c.n_layers);
    {
        glm::LayerWeights w78;
        std::memset(&w78, 0, sizeof(w78));
        bool ok = glm::load_layer(&M, c, 78, &w78);
        CHECK(!ok, "load_layer(78) unexpectedly succeeded -- layer 78 must never load");
        if (ok) glm::free_layer(&w78);   // don't leak if the assertion above ever regresses
    }

    // --- 4. an owning layer (2) and a non-owning one (3) ------------------------
    glm::LayerWeights w2;
    CHECK(glm::load_layer(&M, c, 2, &w2), "load_layer(2) failed");
    CHECK(w2.idx_wq_b != nullptr,         "layer 2 (owner): idx_wq_b is null");
    CHECK(w2.idx_wk != nullptr,           "layer 2 (owner): idx_wk is null");
    CHECK(w2.idx_k_norm_w != nullptr,     "layer 2 (owner): idx_k_norm_w is null");
    CHECK(w2.idx_k_norm_b != nullptr,     "layer 2 (owner): idx_k_norm_b is null");
    CHECK(w2.idx_weights_proj != nullptr, "layer 2 (owner): idx_weights_proj is null");
    CHECK(w2.idx_k_cache != nullptr,      "layer 2 (owner): idx_k_cache (indexer KV cache) is null");

    glm::LayerWeights w3;
    CHECK(glm::load_layer(&M, c, 3, &w3), "load_layer(3) failed");
    CHECK(w3.idx_wq_b == nullptr,         "layer 3 (non-owner): idx_wq_b should be null");
    CHECK(w3.idx_wk == nullptr,           "layer 3 (non-owner): idx_wk should be null");
    CHECK(w3.idx_k_norm_w == nullptr,     "layer 3 (non-owner): idx_k_norm_w should be null");
    CHECK(w3.idx_k_norm_b == nullptr,     "layer 3 (non-owner): idx_k_norm_b should be null");
    CHECK(w3.idx_weights_proj == nullptr, "layer 3 (non-owner): idx_weights_proj should be null");
    CHECK(w3.idx_k_cache == nullptr,      "layer 3 (non-owner): idx_k_cache should be null");
    CHECK(c.indexer_leader[3] == 2, "layer 3's group leader should be 2, got %u", c.indexer_leader[3]);
    printf("layer 2 (owner): all five tensors + KV cache present\n");
    printf("layer 3 (non-owner): all null, leader=%u\n", c.indexer_leader[3]);

    // --- 5. shapes/dtypes straight from the checkpoint's own headers -----------
    // load_layer already asserts these at upload time (it would have returned
    // false above otherwise); this re-checks the raw metadata independently so
    // the test does not rely solely on load_layer's internal checks.
    auto expect_tensor = [&](const char* name, const std::vector<uint64_t>& shape,
                             const char* dtype) {
        const std::string full = "model.layers.2." + std::string(name);
        const st::Tensor* t = st::info(&M, full);
        CHECK(t != nullptr, "missing tensor %s", full.c_str());
        if (!t) return;
        CHECK(t->dtype == dtype, "%s: dtype %s, expected %s", full.c_str(), t->dtype.c_str(), dtype);
        CHECK(t->shape == shape, "%s: shape mismatch (got %zu dims)", full.c_str(), t->shape.size());
    };
    expect_tensor("self_attn.indexer.wq_b.weight",         {4096, 2048}, "BF16");
    expect_tensor("self_attn.indexer.wk.weight",           {128, 6144},  "BF16");
    expect_tensor("self_attn.indexer.k_norm.weight",       {128},        "BF16");
    expect_tensor("self_attn.indexer.k_norm.bias",         {128},        "BF16");
    expect_tensor("self_attn.indexer.weights_proj.weight", {32, 6144},   "BF16");
    printf("layer 2 tensor shapes/dtypes match spec\n");

    glm::free_layer(&w2);
    glm::free_layer(&w3);
    st::close(&M);

    if (g_fail) { printf("\nFAIL\n"); return 1; }
    printf("\nALL CHECKS OK\n");
    return 0;
}
