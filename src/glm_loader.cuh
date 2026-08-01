/*
 * glm_loader.cuh — GLM-5.2 config + per-layer weight loading.
 *
 * Field-by-field provenance is in the plan (Task 5, Step 4.0). Tensor names are
 * the checkpoint's, verified against model.safetensors.index.json.
 */
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>   // std::max, used by alloc_scratch
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "safetensors_io.cuh"

namespace glm {

// The header carries its own error guard — CUDA_OK in test_glm_layer.cu is a
// translation-unit-local macro and must not be relied on from here.
#define GLM_CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    std::abort(); } } while (0)

// Fixed-size backing for Config::indexer_owner / indexer_leader. 128 is headroom
// over GLM-5.2's 78 layers; load_config fails loudly if n_layers exceeds it
// rather than overrunning these arrays.
static constexpr uint32_t MAX_INDEXER_LAYERS = 128;

struct Config {
    // --- published in the Task 5 Interfaces block; unchanged ---
    uint32_t n_layers, hidden, n_routed_experts, n_experts_per_tok, moe_inter,
             n_shared, dense_first, kv_lora_rank, q_lora_rank, vocab;
    float rms_eps, rope_theta;
    // --- additive extension (Step 4.1) ---
    uint32_t n_heads;        // num_attention_heads            = 64
    uint32_t qk_nope;        // qk_nope_head_dim               = 192
    uint32_t qk_rope;        // qk_rope_head_dim               = 64
    uint32_t qk_head;        // qk_nope + qk_rope              = 256
    uint32_t v_head;         // v_head_dim                     = 256
    uint32_t dense_inter;    // intermediate_size              = 12288
    float    routed_scaling; // routed_scaling_factor          = 2.5f
    bool     norm_topk_prob; // true
    uint32_t n_group, topk_group; // 1, 1
    uint32_t index_topk;     // 2048
    uint32_t index_n_heads;  // 32   -- DSA indexer heads
    uint32_t index_head_dim; // 128  -- DSA indexer head width
    uint32_t max_seq;        // KV-cache capacity, set by the caller (not from config.json)

    // --- DSA indexer ownership (Task 2) --------------------------------
    // Read verbatim from config.json's `indexer_types` array — NOT derived from
    // the index_topk_freq/index_skip_topk_offset formula it happens to match on
    // this checkpoint. Task 1 established the formula is not authoritative; a
    // checkpoint whose layout disagrees with it must still load correctly.
    // indexer_owner[i]:  true iff layer i's `indexer_types` entry is "full",
    //                    i.e. layer i carries its own indexer.{wq_b,wk,k_norm,
    //                    weights_proj} tensors.
    // indexer_leader[i]: the owning layer whose indexer output layer i consumes
    //                    -- itself, if indexer_owner[i] is true. Valid for
    //                    i < n_layers only; UINT32_MAX beyond that (includes the
    //                    out-of-range MTP layer 78, which has indexer tensors in
    //                    the checkpoint but no indexer_types entry and is never
    //                    loaded — see load_layer's bounds check).
    bool     indexer_owner[MAX_INDEXER_LAYERS];
    uint32_t indexer_leader[MAX_INDEXER_LAYERS];
};

// NOT config.rms_norm_eps. GlmMoeDsaAttention constructs its two LoRA norms as
// GlmMoeDsaRMSNorm(config.q_lora_rank)  (modeling_glm_moe_dsa.py:339) and
// GlmMoeDsaRMSNorm(self.kv_lora_rank)   (modeling_glm_moe_dsa.py:347)
// with NO eps argument, so both take the class default eps = 1e-6
// (modeling_glm_moe_dsa.py:49). Only input_layernorm / post_attention_layernorm
// (:588-589) and model.norm (:667) are given config.rms_norm_eps, which is 1e-5.
// Using rms_eps for q_a_layernorm / kv_a_layernorm is insidious: the CUDA path and
// the numpy reference would then agree with each other and disagree only with the
// transformers oracle.
static constexpr float LORA_NORM_EPS = 1e-6f;

// run_moe stages the selected experts' indices and weights in host arrays of this
// size before the per-expert launches. 8 today (num_experts_per_tok); the cap is
// enforced in load_config so a larger config fails loudly at load instead of
// smashing run_moe's stack.
static constexpr uint32_t MAX_TOPK = 32;

// Hand-rolled scalar lookup over config.json: finds "\"key\"", skips ':' and
// whitespace, parses the literal. Mirrors the Qwen loader — no JSON dependency.
// The key match requires the closing quote to be followed (modulo whitespace) by
// ':' so that "index_topk" cannot match "index_topk_freq".
static inline bool json_find_value(const std::string& s, const char* key, size_t* vpos) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = 0;
    while ((p = s.find(pat, p)) != std::string::npos) {
        size_t q = p + pat.size();
        while (q < s.size() && (s[q] == ' ' || s[q] == '\t' || s[q] == '\n' || s[q] == '\r')) q++;
        if (q < s.size() && s[q] == ':') {
            q++;
            while (q < s.size() && (s[q] == ' ' || s[q] == '\t' || s[q] == '\n' || s[q] == '\r')) q++;
            *vpos = q;
            return true;
        }
        p += pat.size();
    }
    return false;
}

static inline bool json_num(const std::string& s, const char* key, double* out) {
    size_t q;
    if (!json_find_value(s, key, &q)) return false;
    const char* b = s.c_str() + q;
    char* e = nullptr;
    double v = std::strtod(b, &e);
    if (e == b) return false;
    *out = v;
    return true;
}

static inline bool json_bool(const std::string& s, const char* key, bool* out) {
    size_t q;
    if (!json_find_value(s, key, &q)) return false;
    if (s.compare(q, 4, "true") == 0)  { *out = true;  return true; }
    if (s.compare(q, 5, "false") == 0) { *out = false; return true; }
    return false;
}

// Reads a JSON array of strings, e.g. "indexer_types": ["full", "shared", ...].
// Only what config.json's indexer_types needs: no escapes, no nesting.
static inline bool json_string_array(const std::string& s, const char* key,
                                     std::vector<std::string>* out) {
    size_t q;
    if (!json_find_value(s, key, &q)) return false;
    if (q >= s.size() || s[q] != '[') return false;
    size_t p = q + 1;
    out->clear();
    while (p < s.size() && s[p] != ']') {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' ||
                                s[p] == '\r' || s[p] == ',')) p++;
        if (p >= s.size() || s[p] == ']') break;
        if (s[p] != '"') return false;
        size_t start = ++p;
        while (p < s.size() && s[p] != '"') p++;
        if (p >= s.size()) return false;
        out->push_back(s.substr(start, p - start));
        p++;  // skip closing quote
    }
    return p < s.size();  // found the closing ']'
}

inline bool load_config(const std::string& dir, Config* c) {
    std::string js;
    {
        FILE* f = fopen((dir + "/config.json").c_str(), "rb");
        if (!f) { fprintf(stderr, "load_config: open %s/config.json\n", dir.c_str()); return false; }
        fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
        js.resize((size_t)n);
        if (fread(&js[0], 1, (size_t)n, f) != (size_t)n) { fclose(f); return false; }
        fclose(f);
    }
    double v; bool b;
    #define REQ(key, dst) do { if (!json_num(js, key, &v)) { \
        fprintf(stderr, "load_config: missing %s\n", key); return false; } dst = v; } while (0)
    REQ("num_hidden_layers",   c->n_layers);
    REQ("hidden_size",         c->hidden);
    REQ("n_routed_experts",    c->n_routed_experts);
    REQ("num_experts_per_tok", c->n_experts_per_tok);
    REQ("moe_intermediate_size", c->moe_inter);
    REQ("n_shared_experts",    c->n_shared);
    REQ("first_k_dense_replace", c->dense_first);
    REQ("kv_lora_rank",        c->kv_lora_rank);
    REQ("q_lora_rank",         c->q_lora_rank);
    REQ("vocab_size",          c->vocab);
    REQ("rms_norm_eps",        c->rms_eps);
    REQ("num_attention_heads", c->n_heads);
    REQ("qk_nope_head_dim",    c->qk_nope);
    REQ("qk_rope_head_dim",    c->qk_rope);
    REQ("v_head_dim",          c->v_head);
    REQ("intermediate_size",   c->dense_inter);
    REQ("routed_scaling_factor", c->routed_scaling);
    REQ("n_group",             c->n_group);
    REQ("topk_group",          c->topk_group);
    REQ("index_topk",          c->index_topk);
    REQ("index_n_heads",       c->index_n_heads);
    REQ("index_head_dim",      c->index_head_dim);
    REQ("rope_theta",          c->rope_theta);   // nested under rope_parameters; the
                                                 // scalar scan finds it regardless of nesting
    #undef REQ

    if (c->n_layers > MAX_INDEXER_LAYERS) {
        fprintf(stderr, "load_config: n_layers=%u exceeds MAX_INDEXER_LAYERS=%u\n",
                c->n_layers, MAX_INDEXER_LAYERS);
        return false;
    }
    // indexer_types is explicit in the checkpoint, one entry per layer (Task 1:
    // do not derive this from the freq/offset formula). "full" = owns indexer
    // weights and leads its own group; "shared" = consumes the nearest
    // preceding owner's indices (M:417-419).
    {
        std::vector<std::string> itypes;
        if (!json_string_array(js, "indexer_types", &itypes)) {
            fprintf(stderr, "load_config: missing or malformed indexer_types\n");
            return false;
        }
        if (itypes.size() != c->n_layers) {
            fprintf(stderr, "load_config: indexer_types has %zu entries, expected "
                            "n_layers=%u\n", itypes.size(), c->n_layers);
            return false;
        }
        uint32_t leader = UINT32_MAX;
        for (uint32_t i = 0; i < c->n_layers; i++) {
            if (itypes[i] == "full") {
                c->indexer_owner[i] = true;
                leader = i;
            } else if (itypes[i] == "shared") {
                c->indexer_owner[i] = false;
                if (leader == UINT32_MAX) {
                    fprintf(stderr, "load_config: layer %u is \"shared\" with no "
                                    "preceding owning layer\n", i);
                    return false;
                }
            } else {
                fprintf(stderr, "load_config: layer %u has unknown indexer_types "
                                "value \"%s\"\n", i, itypes[i].c_str());
                return false;
            }
            c->indexer_leader[i] = leader;
        }
        for (uint32_t i = c->n_layers; i < MAX_INDEXER_LAYERS; i++) {
            c->indexer_owner[i] = false;
            c->indexer_leader[i] = UINT32_MAX;
        }
    }
    c->qk_head = c->qk_nope + c->qk_rope;
    c->norm_topk_prob = json_bool(js, "norm_topk_prob", &b) ? b : true;
    // NOTE (config.json ignores head_dim on purpose): GlmMoeDsaConfig.__post_init__
    // overwrites head_dim with qk_rope_head_dim (configuration_glm_moe_dsa.py:153),
    // so the on-disk head_dim: 192 is dead. The RoPE dim is c->qk_rope.
    if (c->n_experts_per_tok > MAX_TOPK) {
        fprintf(stderr, "load_config: num_experts_per_tok=%u exceeds MAX_TOPK=%u — "
                        "run_moe's staging arrays are fixed-size\n",
                c->n_experts_per_tok, MAX_TOPK);
        return false;
    }
    if (c->n_group != 1 || c->topk_group != 1) {
        // Grouped routing is a no-op only at n_group == topk_group == 1
        // (modeling_glm_moe_dsa.py:490-503). Fail loudly rather than silently
        // routing over the wrong expert subset.
        fprintf(stderr, "load_config: n_group=%u topk_group=%u — grouped routing "
                        "is not implemented (see Step 4.4)\n", c->n_group, c->topk_group);
        return false;
    }
    c->max_seq = 0;  // caller sets before load_layer
    return true;
}

// All pointers are device-side. bf16 tensors are kept packed as uint16_t and
// decoded in the matvec (kernels.cuh's matvec_bf16 convention).
struct LayerWeights {
    // norms — [hidden], [q_lora_rank], [kv_lora_rank]
    uint16_t *input_ln, *post_attn_ln, *q_a_ln, *kv_a_ln;
    // attention, all bf16, all bias-free (attention_bias == false, J / C:124)
    uint16_t *q_a_proj;            // [q_lora_rank, hidden]                 [2048, 6144]
    uint16_t *q_b_proj;            // [n_heads*qk_head, q_lora_rank]       [16384, 2048]
    uint16_t *kv_a_proj_with_mqa;  // [kv_lora_rank + qk_rope, hidden]      [576, 6144]
    uint16_t *kv_b_proj;           // [n_heads*(qk_nope+v_head), kv_lora]  [28672, 512]
    uint16_t *o_proj;              // [hidden, n_heads*v_head]             [6144, 16384]
    // MoE (layer >= dense_first)
    uint16_t *router_w;            // mlp.gate.weight   [n_routed_experts, hidden]
    float    *router_bias;         // mlp.gate.e_score_correction_bias [n_routed_experts], F32 on disk
    uint8_t  *sh_gate_w, *sh_gate_s;  // [moe_inter, hidden/2], [moe_inter, hidden/32]
    uint8_t  *sh_up_w,   *sh_up_s;    //  ditto
    uint8_t  *sh_down_w, *sh_down_s;  // [hidden, moe_inter/2], [hidden, moe_inter/32]
    // dense MLP (layer < dense_first), bf16, no scales on disk
    uint16_t *mlp_gate, *mlp_up, *mlp_down;
    // --- per-layer KV cache (see Step 4.3 for why it lives here) ---
    // There is deliberately NO cached_len field. run_layer derives the valid row
    // count as T = pos + 1 from the position it is given, which is the single
    // source of truth; a second, separately-maintained copy of the same number is
    // a desynchronisation waiting to happen. The plan's LayerWeights listed one,
    // but nothing ever read or updated it. The caller owns `pos`.
    float   *kv_c_cache;   // [max_seq, kv_lora_rank]  post-kv_a_layernorm compressed KV
    float   *k_rot_cache;  // [max_seq, qk_rope]       rope'd, NOT normalised
    // --- DSA sparse-attention indexer (Task 2) ---------------------------
    // Non-null ONLY when Config::indexer_owner[layer] is true; null on every
    // "shared" layer, which consumes Config::indexer_leader[layer]'s output
    // instead (Task 3/4) rather than having any of its own.
    uint16_t *idx_wq_b;          // [index_n_heads*index_head_dim, q_lora_rank]  [4096, 2048] bf16
    uint16_t *idx_wk;            // [index_head_dim, hidden]                    [ 128, 6144] bf16
    uint16_t *idx_k_norm_w;      // [index_head_dim]                            [128]        bf16
    uint16_t *idx_k_norm_b;      // [index_head_dim]                            [128]        bf16
    // weights_proj is bf16 on disk like the other four indexer tensors, but
    // transformers forces it to fp32 at inference (_keep_in_fp32_modules,
    // modeling_glm_moe_dsa.py:643) and every downstream use of it is already in
    // fp32 (the qk scores, the head combination, the top-k are all fp32 per
    // Task 1). Converting once here at load time is strictly cheaper than
    // decoding bf16->f32 on every query token in Task 3's kernel, so this is
    // stored as float, already converted, not as the on-disk uint16_t.
    float    *idx_weights_proj;  // [index_n_heads, hidden]                     [32, 6144]   fp32 (converted from bf16 at load)
    // Indexer's OWN KV cache: post-k_norm, post-RoPE 128-dim keys, one per
    // position, bf16. NOT derivable from kv_c_cache/k_rot_cache above -- wk
    // reads the raw hidden state (input_layernorm output), which nothing else
    // retains. Only allocated for owning layers (alloc_scratch); null
    // elsewhere. Per-layer size is index_head_dim * 2 bytes/position; the
    // 5376 B/position figure in the plan is the SUM across all 21 owning
    // layers, not this one buffer.
    uint16_t *idx_k_cache;       // [max_seq, index_head_dim] bf16, owning layers only
    // --- indexer scratch + output (Task 4), owning layers only, else null ---
    // Named s_i* to keep them clear of s_w / s_idx, which belong to the ROUTER
    // and are a different length (n_experts_per_tok, not index_n_heads).
    float   *s_iq;        // [index_n_heads * index_head_dim] post-RoPE indexer query
    float   *s_ik;        // [index_head_dim]                 one key, pre-cache
    float   *s_iw;        // [index_n_heads]                  weights_proj gates, fp32
    float   *s_iscores;   // [index_n_heads * max_seq]        relu'd qk scores
    float   *s_iindex;    // [max_seq]                        head-combined index score
    int32_t *idx_topk;    // [index_topk]  raw top-k indices — the value IndexShare
                          //               propagates, captured BEFORE index_mask
                          //               is built (M:408-415 -> M:453), i.e. still
                          //               containing non-causal entries.
    uint32_t *idx_topk_n; // [1]           how many of idx_topk are valid
    // --- attention masking (Task 4), every layer ---------------------------
    uint8_t *s_index_mask;   // [max_seq] 1 == drop. Rebuilt per token per layer.
    // Overflow buffer for mla_decode_absorbed's per-key score array. Normally
    // that array is dynamic shared memory (T floats); above the 48 KB per-block
    // limit there is no shared memory large enough, so the kernel reads the same
    // values out of global memory instead. Same arithmetic, same order, only a
    // different address space — allocated ONLY when max_seq is large enough to
    // need it, null otherwise (see alloc_scratch).
    float   *s_attn_scores;  // [n_heads * max_seq] or null
    // --- scratch: allocated ONLY by alloc_scratch below, never ad hoc ---
    // ALIASING RULE: s_acc is write-only inside run_moe (run_layer passes it as
    // d_out); nothing later accumulated into it may share it. That is why the
    // routed down-projection lands in s_expert and the shared one in s_shared.
    float *s_norm, *s_qa, *s_q, *s_qabs, *s_ctx, *s_attn, *s_kv, *s_logits,
          *s_w, *s_gate, *s_up, *s_mid, *s_acc, *s_expert, *s_shared;
    int32_t *s_idx;
    // Per-position RoPE table, uploaded by run_layer each step. qk_rope/2 = 32 floats each.
    float *d_cos, *d_sin;
};

// Every device buffer whose size comes from Config: scratch, the RoPE table, and
// the per-layer KV cache. Called by load_layer before any upload. `owns_indexer`
// must be Config::indexer_owner[layer] for the layer being loaded -- it gates
// the indexer KV cache allocation below.
// ALLOCATION trigger for the global per-key score buffer — deliberately NOT the
// real shared-memory capacity, and deliberately well below it.
//
// The real capacity is a runtime property of the kernel, not a constant: it is
// 48 KB minus whatever static __shared__ that kernel already uses.
// mla_decode_absorbed's two block reductions hold 32 floats each, so on SM 12.0
// cudaFuncGetAttributes reports maxDynamicSharedSizeBytes = 48896, not 49152 —
// a 256-byte shortfall that made every T in [12225, 12288] fail to launch with
// `invalid argument`. That is what an assumed constant buys you.
//
// run_layer now asks the driver for the exact figure (mla_smem_capacity()) and
// switches to the global buffer against THAT. This constant only decides whether
// the buffer is allocated at all, and being conservative here is nearly free:
// it costs an unused n_heads*max_seq buffer for max_seq in (8192, ~12224], and
// it guarantees the buffer exists whenever the exact check asks for it, since
// the true capacity is never below 32 KB on any supported architecture.
static constexpr size_t MLA_SMEM_LIMIT = 32u * 1024u;

inline void alloc_scratch(const Config& c, LayerWeights* w, bool owns_indexer) {
    const uint32_t H  = c.hidden;                    // 6144
    const uint32_t SI = c.moe_inter * c.n_shared;    // shared-expert intermediate, M:563-565

    // SIZING TRAP — read before changing any of the next three lines.
    // s_gate / s_up / s_mid are reused by BOTH branches of run_moe: the dense
    // branch (layers 0..dense_first-1) runs them at dense_inter = 12288, the MoE
    // branch at moe_inter = 2048, and the shared expert at SI = 2048. Sizing them
    // from moe_inter alone is a 6x overrun on layers 0-2 — and NO test in this plan
    // would catch it: Task 5's oracle test is layer 3 (MoE), Step 7's layer-0 run is
    // the only dense exercise and it comes after. Size for the maximum, always.
    const uint32_t FF = std::max(c.dense_inter, std::max(c.moe_inter, SI));  // 12288

    auto F = [](float** p, size_t n) { GLM_CUDA_OK(cudaMalloc(p, n * sizeof(float))); };
    F(&w->s_norm,   H);                        // post-norm hidden, feeds every projection
    F(&w->s_qa,     c.q_lora_rank);            // 2048
    F(&w->s_q,      c.n_heads * c.qk_head);    // 64 * 256  = 16384
    F(&w->s_qabs,   c.n_heads * c.kv_lora_rank);   // 64 * 512 = 32768
    F(&w->s_ctx,    c.n_heads * c.kv_lora_rank);   // 64 * 512 = 32768
    F(&w->s_attn,   c.n_heads * c.v_head);     // 64 * 256  = 16384
    F(&w->s_kv,     c.kv_lora_rank + c.qk_rope);   // 576
    F(&w->s_logits, c.n_routed_experts);       // 256
    F(&w->s_w,      c.n_experts_per_tok);      // 8
    F(&w->s_gate,   FF);                       // 12288 — see the SIZING TRAP above
    F(&w->s_up,     FF);                       // 12288
    F(&w->s_mid,    FF);                       // 12288
    F(&w->s_acc,    H);                        // run_moe's d_out; write-only there
    F(&w->s_expert, H);                        // routed down-proj result
    F(&w->s_shared, H);                        // shared-expert down-proj result
    F(&w->d_cos,    c.qk_rope / 2);            // 32
    F(&w->d_sin,    c.qk_rope / 2);            // 32
    GLM_CUDA_OK(cudaMalloc(&w->s_idx, c.n_experts_per_tok * sizeof(int32_t)));   // 8
    // Per-layer KV cache. max_seq must be set by the caller before load_layer —
    // load_config leaves it 0 on purpose. Without this guard a caller that forgets
    // gets two zero-sized allocations and run_layer writes out of bounds at pos 0,
    // silently and on every layer.
    if (c.max_seq == 0) {
        fprintf(stderr, "alloc_scratch: Config::max_seq is 0 — the caller must set "
                        "the KV-cache capacity before load_layer\n");
        std::abort();
    }
    F(&w->kv_c_cache,  (size_t)c.max_seq * c.kv_lora_rank);   // 512 f32 / position
    F(&w->k_rot_cache, (size_t)c.max_seq * c.qk_rope);        //  64 f32 / position

    // Indexer KV cache — SIZING TRAP: this is a PER-LAYER allocation
    // (index_head_dim * 2 bytes/position = 256 B/position at index_head_dim=128),
    // not the plan's 5376 B/position figure, which is that number summed across
    // all 21 owning layers. Sizing this one buffer to 5376 B/position would be a
    // 21x overrun on every owning layer. Allocated ONLY for owning layers; every
    // "shared" layer leaves idx_k_cache null and must look up its group leader's
    // buffer instead (Task 3/4) rather than have one of its own.
    if (owns_indexer) {
        w->idx_k_cache = nullptr;  // set below; keeps the branch symmetric with cudaMalloc's out-param
        GLM_CUDA_OK(cudaMalloc(&w->idx_k_cache,
                               (size_t)c.max_seq * c.index_head_dim * sizeof(uint16_t)));
        // Indexer working set + its output. Owning layers only: a "shared" layer
        // computes nothing and consumes its group leader's idx_topk (Task 4).
        F(&w->s_iq,      (size_t)c.index_n_heads * c.index_head_dim);   // 4096
        F(&w->s_ik,      c.index_head_dim);                             // 128
        F(&w->s_iw,      c.index_n_heads);                              // 32
        F(&w->s_iscores, (size_t)c.index_n_heads * c.max_seq);          // 32 * max_seq
        F(&w->s_iindex,  c.max_seq);
        GLM_CUDA_OK(cudaMalloc(&w->idx_topk, (size_t)c.index_topk * sizeof(int32_t)));
        GLM_CUDA_OK(cudaMalloc(&w->idx_topk_n, sizeof(uint32_t)));
    } else {
        w->idx_k_cache = nullptr;
        w->s_iq = w->s_ik = w->s_iw = w->s_iscores = w->s_iindex = nullptr;
        w->idx_topk = nullptr;
        w->idx_topk_n = nullptr;
    }

    // Drop-mask, EVERY layer — a "shared" layer masks with its leader's indices,
    // so it needs its own mask buffer even though it owns no indexer.
    GLM_CUDA_OK(cudaMalloc(&w->s_index_mask, c.max_seq * sizeof(uint8_t)));

    // mla_decode_absorbed keeps T per-key scores. Dynamic shared memory covers
    // that up to the 48 KB per-block limit, i.e. T <= 12288 at 4 B/key; past
    // that the kernel needs a global buffer. n_heads * max_seq floats is 8.4 MB
    // per layer at max_seq = 32768 — real memory, so it is allocated only when
    // the shared-memory route provably cannot cover max_seq. Below the
    // threshold this stays null and the kernel path is byte-identical to the
    // pre-Task-4 one.
    if ((size_t)c.max_seq * sizeof(float) > MLA_SMEM_LIMIT) {
        F(&w->s_attn_scores, (size_t)c.n_heads * c.max_seq);
    } else {
        w->s_attn_scores = nullptr;
    }
}

inline void free_scratch(LayerWeights* w) {
    for (float** p : {&w->s_norm, &w->s_qa, &w->s_q, &w->s_qabs, &w->s_ctx,
                      &w->s_attn, &w->s_kv, &w->s_logits, &w->s_w, &w->s_gate,
                      &w->s_up, &w->s_mid, &w->s_acc, &w->s_expert, &w->s_shared,
                      &w->d_cos, &w->d_sin, &w->kv_c_cache, &w->k_rot_cache,
                      &w->s_iq, &w->s_ik, &w->s_iw, &w->s_iscores, &w->s_iindex,
                      &w->s_attn_scores}) {
        cudaFree(*p); *p = nullptr;
    }
    cudaFree(w->s_idx); w->s_idx = nullptr;
    cudaFree(w->idx_k_cache); w->idx_k_cache = nullptr;  // no-op on non-owning layers (null)
    cudaFree(w->idx_topk);   w->idx_topk = nullptr;      // ditto
    cudaFree(w->idx_topk_n); w->idx_topk_n = nullptr;
    cudaFree(w->s_index_mask); w->s_index_mask = nullptr;
}

// Counterpart to load_layer: releases the weight tensors it uploaded AND the
// scratch/KV/RoPE buffers alloc_scratch made. Every device pointer in
// LayerWeights is covered, so a caller only ever needs this one call. Safe on a
// partially-loaded struct (load_layer memsets it first, and cudaFree(nullptr) is
// a no-op), which matters because load_layer can fail part-way through.
inline void free_layer(LayerWeights* w) {
    for (uint16_t** p : {&w->input_ln, &w->post_attn_ln, &w->q_a_ln, &w->kv_a_ln,
                         &w->q_a_proj, &w->q_b_proj, &w->kv_a_proj_with_mqa,
                         &w->kv_b_proj, &w->o_proj, &w->router_w,
                         &w->mlp_gate, &w->mlp_up, &w->mlp_down,
                         &w->idx_wq_b, &w->idx_wk, &w->idx_k_norm_w, &w->idx_k_norm_b}) {
        cudaFree(*p); *p = nullptr;
    }
    for (uint8_t** p : {&w->sh_gate_w, &w->sh_gate_s, &w->sh_up_w,
                        &w->sh_up_s, &w->sh_down_w, &w->sh_down_s}) {
        cudaFree(*p); *p = nullptr;
    }
    cudaFree(w->router_bias); w->router_bias = nullptr;
    cudaFree(w->idx_weights_proj); w->idx_weights_proj = nullptr;
    free_scratch(w);   // also releases idx_k_cache
}

// --- tensor upload helpers -------------------------------------------------

// A tensor is MXFP4 iff "<name>_scale" is present in the index (Step 4.0). This
// is a local test that needs no layer-78 special case and cannot drift from the
// shards.
inline bool is_mxfp4(st::ModelDir* M, const std::string& name) {
    return st::info(M, name + "_scale") != nullptr;
}

template <typename T>
inline bool upload_raw(st::ModelDir* M, const std::string& name, T** dst,
                       size_t expect_elems, const char* expect_dtype) {
    const st::Tensor* t = st::info(M, name);
    if (!t) { fprintf(stderr, "load_layer: missing tensor %s\n", name.c_str()); return false; }
    if (expect_dtype && t->dtype != expect_dtype) {
        fprintf(stderr, "load_layer: %s dtype %s, expected %s\n",
                name.c_str(), t->dtype.c_str(), expect_dtype);
        return false;
    }
    if (t->nbytes != expect_elems * sizeof(T)) {
        fprintf(stderr, "load_layer: %s has %llu bytes, expected %zu\n",
                name.c_str(), (unsigned long long)t->nbytes, expect_elems * sizeof(T));
        return false;
    }
    std::vector<uint8_t> host;
    if (!st::read_bytes(M, name, host)) return false;
    GLM_CUDA_OK(cudaMalloc(dst, host.size()));
    GLM_CUDA_OK(cudaMemcpy(*dst, host.data(), host.size(), cudaMemcpyHostToDevice));
    return true;
}

// Reads one tensor by name into device memory. A tensor is MXFP4 iff
// "<name>_scale" is present in the index (see the quantization-scope note in
// Step 4.0). Calls alloc_scratch() before uploading anything.
inline bool load_layer(st::ModelDir* M, const Config& c, uint32_t layer, LayerWeights* out) {
    std::memset(out, 0, sizeof(*out));
    if (layer >= c.n_layers) {
        // Also how layer 78 — the MTP head, which carries indexer.* tensors in
        // the checkpoint but has no indexer_types entry and is out of scope
        // (Task 1) — is kept out: it is simply never a valid `layer` argument.
        fprintf(stderr, "load_layer: layer %u out of range (n_layers=%u)\n",
                layer, c.n_layers);
        return false;
    }
    const bool owns_indexer = c.indexer_owner[layer];
    alloc_scratch(c, out, owns_indexer);

    const std::string P = "model.layers." + std::to_string(layer) + ".";
    const uint32_t H = c.hidden, I = c.moe_inter, SI = c.moe_inter * c.n_shared;

    #define BF16(name, dst, n) do { \
        if (is_mxfp4(M, P + name)) { \
            fprintf(stderr, "load_layer: %s%s unexpectedly has a _scale (MXFP4)\n", \
                    P.c_str(), name); return false; } \
        if (!upload_raw<uint16_t>(M, P + name, &out->dst, (size_t)(n), "BF16")) return false; \
    } while (0)

    BF16("input_layernorm.weight",              input_ln,           H);
    BF16("post_attention_layernorm.weight",     post_attn_ln,       H);
    BF16("self_attn.q_a_layernorm.weight",      q_a_ln,             c.q_lora_rank);
    BF16("self_attn.kv_a_layernorm.weight",     kv_a_ln,            c.kv_lora_rank);
    BF16("self_attn.q_a_proj.weight",           q_a_proj,           (size_t)c.q_lora_rank * H);
    BF16("self_attn.q_b_proj.weight",           q_b_proj,           (size_t)c.n_heads * c.qk_head * c.q_lora_rank);
    BF16("self_attn.kv_a_proj_with_mqa.weight", kv_a_proj_with_mqa, (size_t)(c.kv_lora_rank + c.qk_rope) * H);
    BF16("self_attn.kv_b_proj.weight",          kv_b_proj,          (size_t)c.n_heads * (c.qk_nope + c.v_head) * c.kv_lora_rank);
    BF16("self_attn.o_proj.weight",             o_proj,             (size_t)H * c.n_heads * c.v_head);

    // --- DSA indexer (Task 2), owning layers only -------------------------
    if (owns_indexer) {
        const uint32_t IH = c.index_n_heads, ID = c.index_head_dim;   // 32, 128
        BF16("self_attn.indexer.wq_b.weight",   idx_wq_b,     (size_t)IH * ID * c.q_lora_rank);
        BF16("self_attn.indexer.wk.weight",     idx_wk,       (size_t)ID * H);
        BF16("self_attn.indexer.k_norm.weight", idx_k_norm_w, ID);
        BF16("self_attn.indexer.k_norm.bias",   idx_k_norm_b, ID);

        // weights_proj: bf16 on disk (checkpoint-verified, Task 1), converted to
        // fp32 here — see the LayerWeights::idx_weights_proj comment for why.
        {
            const std::string name = P + "self_attn.indexer.weights_proj.weight";
            const st::Tensor* t = st::info(M, name);
            if (!t) { fprintf(stderr, "load_layer: missing tensor %s\n", name.c_str()); return false; }
            if (is_mxfp4(M, name)) {
                fprintf(stderr, "load_layer: %s unexpectedly has a _scale (MXFP4)\n", name.c_str());
                return false;
            }
            if (t->dtype != "BF16") {
                fprintf(stderr, "load_layer: %s dtype %s, expected BF16\n",
                        name.c_str(), t->dtype.c_str());
                return false;
            }
            const size_t n_elems = (size_t)IH * H;
            if (t->nbytes != n_elems * sizeof(uint16_t)) {
                fprintf(stderr, "load_layer: %s has %llu bytes, expected %zu (bf16)\n",
                        name.c_str(), (unsigned long long)t->nbytes, n_elems * sizeof(uint16_t));
                return false;
            }
            std::vector<uint8_t> raw;
            if (!st::read_bytes(M, name, raw)) return false;
            std::vector<float> f32(n_elems);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (size_t i = 0; i < n_elems; i++) f32[i] = st::bf16_to_f32(src[i]);
            GLM_CUDA_OK(cudaMalloc(&out->idx_weights_proj, n_elems * sizeof(float)));
            GLM_CUDA_OK(cudaMemcpy(out->idx_weights_proj, f32.data(),
                                   n_elems * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    if (layer >= c.dense_first) {
        BF16("mlp.gate.weight", router_w, (size_t)c.n_routed_experts * H);
        if (!upload_raw<float>(M, P + "mlp.gate.e_score_correction_bias",
                               &out->router_bias, c.n_routed_experts, "F32")) return false;
        #define MXFP4(name, wdst, sdst, N, K) do { \
            if (!is_mxfp4(M, P + name)) { \
                fprintf(stderr, "load_layer: %s%s has no _scale — expected MXFP4\n", \
                        P.c_str(), name); return false; } \
            if (!upload_raw<uint8_t>(M, P + name, &out->wdst, (size_t)(N) * (K) / 2, "U8")) return false; \
            if (!upload_raw<uint8_t>(M, P + name + "_scale", &out->sdst, (size_t)(N) * (K) / 32, "U8")) return false; \
        } while (0)
        MXFP4("mlp.shared_experts.gate_proj.weight", sh_gate_w, sh_gate_s, SI, H);
        MXFP4("mlp.shared_experts.up_proj.weight",   sh_up_w,   sh_up_s,   SI, H);
        MXFP4("mlp.shared_experts.down_proj.weight", sh_down_w, sh_down_s, H,  SI);
        #undef MXFP4
    } else {
        BF16("mlp.gate_proj.weight", mlp_gate, (size_t)c.dense_inter * H);
        BF16("mlp.up_proj.weight",   mlp_up,   (size_t)c.dense_inter * H);
        BF16("mlp.down_proj.weight", mlp_down, (size_t)H * c.dense_inter);
    }
    #undef BF16
    (void)I;
    return true;
}

} // namespace glm
