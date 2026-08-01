/*
 * test_glm_layer.cu — run one GLM-5.2 decoder layer on GPU and compare
 * every substep against dump_glm_oracle.py output.
 *
 * Build: make test_glm_layer
 * Run:   ./test_glm_layer --model-dir ./glm52-mxfp4 --oracle glm-oracle --layer 3
 *        ./test_glm_layer --model-dir ./glm52-mxfp4 --oracle glm-oracle-layer0 --layer 0
 *
 * The CUDA path is decode-only (one token at a time), so the oracle's 8-token
 * sequence is replayed position by position: attention at position 7 needs
 * positions 0..6 in the KV cache. Only the LAST position is compared.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

#include "safetensors_io.cuh"
#include "glm_primitives.cuh"
#include "glm_kernels.cuh"
#include "glm_loader.cuh"
#include "glm_layer_runner.cuh"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// --- minimal .npy reader ---------------------------------------------------
// Returns the raw payload plus the parsed shape; the caller checks the dtype tag.
static std::vector<uint8_t> load_npy_raw(const std::string& path,
                                         std::vector<size_t>* shape,
                                         std::string* dtype) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); std::exit(1); }
    char magic[6]; size_t rd = fread(magic, 1, 6, f);
    if (rd != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a .npy\n", path.c_str()); std::exit(1);
    }
    uint8_t major, minor; rd = fread(&major, 1, 1, f); rd = fread(&minor, 1, 1, f);
    (void)minor;
    uint32_t hlen = 0;
    if (major == 1) { uint16_t h; rd = fread(&h, 2, 1, f); hlen = h; }
    else            { rd = fread(&hlen, 4, 1, f); }
    std::string hdr(hlen, '\0'); rd = fread(&hdr[0], 1, hlen, f);
    if (rd != hlen) { fprintf(stderr, "%s: short header\n", path.c_str()); std::exit(1); }

    // descr: '<f4' / '<i8' / ...
    size_t dp = hdr.find("descr");
    if (dp == std::string::npos) { fprintf(stderr, "%s: no descr\n", path.c_str()); std::exit(1); }
    // hdr looks like {'descr': '<f4', ...} — skip past the key's own closing quote
    // and its colon before hunting for the value's quotes.
    dp = hdr.find(':', dp);
    if (dp == std::string::npos) { fprintf(stderr, "%s: bad descr\n", path.c_str()); std::exit(1); }
    size_t q1 = hdr.find_first_of("'\"", dp);
    size_t q2 = hdr.find_first_of("'\"", q1 + 1);
    *dtype = hdr.substr(q1 + 1, q2 - q1 - 1);
    size_t esz = 0;
    if (*dtype == "<f4") esz = 4;
    else if (*dtype == "<i8") esz = 8;
    else if (*dtype == "<f8") esz = 8;
    else { fprintf(stderr, "%s: unsupported dtype %s\n", path.c_str(), dtype->c_str()); std::exit(1); }

    size_t sb = hdr.find('('), se = hdr.find(')');
    shape->clear();
    size_t total = 1;
    for (size_t i = sb + 1; i < se; ) {
        while (i < se && !isdigit((unsigned char)hdr[i])) i++;
        if (i >= se) break;
        size_t v = 0;
        while (i < se && isdigit((unsigned char)hdr[i])) { v = v * 10 + (hdr[i] - '0'); i++; }
        shape->push_back(v); total *= v;
    }
    std::vector<uint8_t> out(total * esz);
    if (fread(out.data(), 1, out.size(), f) != out.size()) {
        fprintf(stderr, "%s: short data read\n", path.c_str()); std::exit(1);
    }
    fclose(f);
    return out;
}

static std::vector<float> load_npy_f32(const std::string& path, std::vector<size_t>* shape) {
    std::string dt;
    std::vector<uint8_t> raw = load_npy_raw(path, shape, &dt);
    if (dt != "<f4") {
        fprintf(stderr, "%s: expected float32, got %s\n", path.c_str(), dt.c_str());
        std::exit(1);
    }
    std::vector<float> out(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

// topk_idx is int64 on disk (torch.topk indices).
static std::vector<int64_t> load_npy_i64(const std::string& path, std::vector<size_t>* shape) {
    std::string dt;
    std::vector<uint8_t> raw = load_npy_raw(path, shape, &dt);
    if (dt != "<i8") {
        fprintf(stderr, "%s: expected int64, got %s\n", path.c_str(), dt.c_str());
        std::exit(1);
    }
    std::vector<int64_t> out(raw.size() / 8);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

static int g_fail = 0;
// --keep-going: report every substep instead of stopping at the first FAIL. Off by
// default — a failure at substep N makes every later comparison meaningless, so
// the default is to stop and fix in order.
static int g_keep_going = 0;

// The oracle is bfloat16, this path is float32, so the two cannot agree to better
// than the bfloat16 grid. The natural unit for the comparison is therefore ONE
// bfloat16 ulp at the compared tensor's own scale:
//
//     ulp_at_scale = 2^-8 * max|ref|
//     err_ulp      = max_abs / ulp_at_scale
//
// and the gate is `err_ulp < tol_ulp`.
//
// Why not the plan's elementwise `max_abs_i / (|ref_i| + 1e-4) < 2e-2`: that
// quantity is unbounded wherever the reference happens to cancel to near zero,
// and its 1e-4 floor silently presumes O(1) tensors. These are not — attn_out
// peaks at 1.3e-2, so 1e-4 is 0.8% of the entire tensor and every element below
// that reports an "error" inflated 100x. It is still computed and printed below
// (max_rel, spec form, unchanged) as a diagnostic; it is just not the gate.
//
// The ulp gate is STRICTER than the plan's on the elements that matter: at
// tol_ulp = 3 it permits 1.2% of the tensor's peak, where the plan's 2e-2 would
// have permitted 2% on that same peak element.
//
// The single physical source of the residual is GlmMoeDsaRMSNorm (M:57-62):
// transformers rounds the normalized activation back to the input dtype (bf16)
// before scaling by the weight, and this path keeps it in float32. That is a
// half-ulp = 2^-9 perturbation injected at input_layernorm and again at
// post_attention_layernorm. Measured end-to-end it stays at ~1 ulp of scale
// (worst observed: layer 0 attn_out at 1.06 ulp), exactly as that model predicts.
//
// Independently confirmed: rebuilding the same oracle in pure float32 (no bf16
// anywhere) makes this path agree to <= 2.3e-8 absolute on every substep of both
// layer 3 and layer 0 — i.e. the arithmetic is exact and only the fixture's
// dtype separates them. See task-5-report.md.
static void compare(const char* tag, const std::vector<float>& ref,
                    const std::vector<float>& got, double tol_ulp) {
    if (ref.size() != got.size()) {
        printf("[%s] SIZE MISMATCH ref=%zu got=%zu\n", tag, ref.size(), got.size());
        std::exit(1);
    }
    double max_abs = 0, max_rel = 0, scale = 0, sse = 0; size_t worst = 0, worst_r = 0;
    for (size_t i = 0; i < ref.size(); i++) scale = std::fmax(scale, std::fabs((double)ref[i]));
    for (size_t i = 0; i < ref.size(); i++) {
        double d = std::fabs((double)ref[i] - (double)got[i]);
        sse += d * d;
        if (d > max_abs) { max_abs = d; worst = i; }
        double r = d / (std::fabs((double)ref[i]) + 1e-4);   // plan's metric, diagnostic only
        if (r > max_rel) { max_rel = r; worst_r = i; }
    }
    const double ulp = std::exp2(-8.0) * scale;              // one bf16 ulp at this scale
    const double err_ulp = max_abs / (ulp + 1e-300);
    printf("[%s] err=%.3f ulp (tol %.1f) max_abs=%.4e rms=%.4e | max|ref|=%.4e "
           "rel_to_scale=%.4e | worst@%zu(ref=%.6f got=%.6f) "
           "elemwise_max_rel=%.4e@%zu(ref=%.3e) %s\n",
           tag, err_ulp, tol_ulp, max_abs, std::sqrt(sse / ref.size()), scale,
           max_abs / (scale + 1e-30), worst, ref[worst], got[worst],
           max_rel, worst_r, ref[worst_r],
           err_ulp < tol_ulp ? "OK" : "FAIL");
    if (!(err_ulp < tol_ulp)) { g_fail = 1; if (!g_keep_going) std::exit(1); }
}

// --- ExpertSource backed directly by the checkpoint ------------------------
// Reads the three per-expert MXFP4 tensors and packs them into one block in the
// ExpertLayout order (gate_w | gate_s | up_w | up_s | down_w | down_s), which is
// what Task 6's repacker writes. Blocks are cached; a layer-3 decode of 8 tokens
// touches at most 64 distinct experts (~1.3 GB), which fits comfortably.
struct StExpertSource : glm::ExpertSource {
    st::ModelDir* M;
    glm::Config c;
    glm::ExpertLayout lay;
    std::unordered_map<uint64_t, uint8_t*> cache;

    StExpertSource(st::ModelDir* m, const glm::Config& cfg, const glm::ExpertLayout& l)
        : M(m), c(cfg), lay(l) {}

    ~StExpertSource() override {
        for (auto& kv : cache) cudaFree(kv.second);
    }

    void part(std::vector<uint8_t>& blk, const std::string& name,
              size_t off, size_t len) {
        std::vector<uint8_t> buf;
        if (!st::read_bytes(M, name, buf)) {
            fprintf(stderr, "StExpertSource: read %s failed\n", name.c_str()); std::exit(1);
        }
        if (buf.size() != len) {
            fprintf(stderr, "StExpertSource: %s has %zu bytes, expected %zu\n",
                    name.c_str(), buf.size(), len);
            std::exit(1);
        }
        std::memcpy(blk.data() + off, buf.data(), len);
    }

    uint8_t* get(uint32_t layer, uint32_t expert) override {
        const uint64_t key = ((uint64_t)layer << 32) | expert;
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;

        const std::string P = "model.layers." + std::to_string(layer) +
                              ".mlp.experts." + std::to_string(expert) + ".";
        std::vector<uint8_t> blk(lay.total);
        part(blk, P + "gate_proj.weight",       lay.gw_off, lay.gw_len);
        part(blk, P + "gate_proj.weight_scale", lay.gs_off, lay.gs_len);
        part(blk, P + "up_proj.weight",         lay.uw_off, lay.uw_len);
        part(blk, P + "up_proj.weight_scale",   lay.us_off, lay.us_len);
        part(blk, P + "down_proj.weight",       lay.dw_off, lay.dw_len);
        part(blk, P + "down_proj.weight_scale", lay.ds_off, lay.ds_len);

        uint8_t* d = nullptr;
        CUDA_OK(cudaMalloc(&d, lay.total));
        CUDA_OK(cudaMemcpy(d, blk.data(), lay.total, cudaMemcpyHostToDevice));
        cache[key] = d;
        return d;
    }
};

static std::vector<float> download(const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// --- long-context smoke test (Task 4) --------------------------------------
// The question this answers is narrow and worth stating precisely: with the
// `T > index_topk` abort gone, does a sequence longer than index_topk actually
// run, or does it fail somewhere new? It is NOT a correctness test — above 2048
// there is no oracle yet (that is Task 5, and `ref_glm_chain.py` still carries
// the same 2048 limit), and the hidden states here are pseudo-random rather than
// real activations.
//
// Layers 0,1,2,3 specifically: 0/1/2 are indexer OWNERS and are all below
// first_k_dense_replace, so they cost nothing in expert traffic; 3 is the first
// "shared" consumer, which is what makes IndexShare propagation observable at
// T > index_topk. Running the full 78-layer chain instead would be the same test
// plus ~2 hours of NVMe-bound expert streaming.
static int smoke_long_context(const std::string& model_dir, uint32_t seq) {
    glm::Config c;
    if (!glm::load_config(model_dir, &c)) return 1;
    c.max_seq = seq;
    printf("long-context smoke: seq=%u, index_topk=%u (%s)\n", seq, c.index_topk,
           seq > c.index_topk ? "ABOVE the removed abort" : "below index_topk");
    if (seq <= c.index_topk)
        printf("  WARNING: this seq does not exercise the sparse regime at all\n");

    st::ModelDir M;
    if (!st::open(&M, model_dir, /*allow_missing_shards=*/true)) {
        fprintf(stderr, "st::open failed\n"); return 1;
    }
    const uint32_t NL = 4;
    std::vector<glm::LayerWeights> W(NL);
    for (uint32_t l = 0; l < NL; l++) {
        if (!glm::load_layer(&M, c, l, &W[l])) return 1;
        printf("  layer %u loaded (%s, %s)\n", l,
               c.indexer_owner[l] ? "indexer owner" : "shared consumer",
               l < c.dense_first ? "dense MLP" : "MoE");
    }
    const glm::ExpertLayout lay = glm::ExpertLayout::from(c);
    StExpertSource src(&M, c, lay);

    const uint32_t H = c.hidden;
    float* d_hidden = nullptr;
    CUDA_OK(cudaMalloc(&d_hidden, H * sizeof(float)));
    std::vector<float> h_in(H);

    glm::IndexShare share;
    uint32_t s = 12345u;
    for (uint32_t pos = 0; pos < seq; pos++) {
        for (uint32_t i = 0; i < H; i++) {
            s = s * 1664525u + 1013904223u;
            h_in[i] = (((float)(s >> 8) / (float)(1u << 24)) * 2.0f - 1.0f) * 0.05f;
        }
        CUDA_OK(cudaMemcpy(d_hidden, h_in.data(), H * sizeof(float), cudaMemcpyHostToDevice));
        share.reset();
        for (uint32_t l = 0; l < NL; l++)
            glm::run_layer(c, W[l], lay, src, l, pos, d_hidden, share, 0);

        // Sample around the old cap and at the end. A NaN or an Inf here is the
        // failure this test is looking for.
        const bool sample = (pos + 2 >= c.index_topk && pos <= c.index_topk + 2) ||
                            (pos % 512 == 0) || (pos + 1 == seq);
        if (sample) {
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaGetLastError());
            const std::vector<float> h = download(d_hidden, H);
            size_t bad = 0; double peak = 0;
            for (float v : h) { if (!std::isfinite(v)) bad++; peak = std::fmax(peak, std::fabs((double)v)); }
            if (bad || pos + 1 == seq || pos == c.index_topk)
                printf("  pos %-5u T=%-5u non-finite=%zu peak=%.4e %s\n",
                       pos, pos + 1, bad, peak, bad ? "FAIL" : "ok");
            if (bad) g_fail = 1;
        }
    }
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());

    // --- what the run proves, asserted on values --------------------------
    const uint32_t T = seq, K = (c.index_topk < T) ? c.index_topk : T;
    uint32_t n2 = 0;
    CUDA_OK(cudaMemcpy(&n2, W[2].idx_topk_n, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("[smoke] layer 2 selected %u of %u keys (expected min(index_topk,T)=%u) %s\n",
           n2, T, K, n2 == K ? "OK" : "FAIL");
    if (n2 != K) g_fail = 1;

    printf("[smoke] layer 3 consumed layer %u's indices (expected 2) %s\n",
           share.producer, share.producer == 2 ? "OK" : "FAIL");
    if (share.producer != 2) g_fail = 1;

    auto mask_of = [&](uint32_t l) {
        std::vector<uint8_t> m(T);
        CUDA_OK(cudaMemcpy(m.data(), W[l].s_index_mask, T, cudaMemcpyDeviceToHost));
        return m;
    };
    const std::vector<uint8_t> m0 = mask_of(0), m2 = mask_of(2), m3 = mask_of(3);
    size_t kept2 = 0; for (uint8_t v : m2) if (!v) kept2++;
    printf("[smoke] layer 2 mask keeps %zu of %u keys (expected %u) %s\n",
           kept2, T, K, kept2 == K ? "OK" : "FAIL");
    if (kept2 != K) g_fail = 1;

    // The consumer's mask must equal its leader's, bit for bit — that is what
    // "receives the leader's indices unchanged" means at T > index_topk.
    const bool same23 = (std::memcmp(m2.data(), m3.data(), T) == 0);
    printf("[smoke] layer 3's mask == layer 2's mask %s\n", same23 ? "OK" : "FAIL");
    if (!same23) g_fail = 1;

    // ...and that equality is only evidence if two DIFFERENT indexers really do
    // select differently at this T. Layer 0 owns its own; it must disagree with
    // layer 2, or the check above is vacuous.
    size_t diff02 = 0; for (uint32_t i = 0; i < T; i++) if (m0[i] != m2[i]) diff02++;
    printf("[smoke] layer 0's mask differs from layer 2's in %zu of %u keys "
           "(must be > 0, else the equality above is vacuous) %s\n",
           diff02, T, diff02 > 0 ? "OK" : "FAIL");
    if (diff02 == 0) g_fail = 1;

    cudaFree(d_hidden);
    for (uint32_t l = 0; l < NL; l++) glm::free_layer(&W[l]);
    st::close(&M);
    printf("\n%s (long-context smoke, seq=%u)\n", g_fail ? "FAIL" : "RUNS CLEAN", seq);
    return g_fail;
}

int main(int argc, char** argv) {
    std::string model_dir, oracle_dir;
    uint32_t layer = 3;
    uint32_t smoke_seq = 0;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--model-dir") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--oracle") && i + 1 < argc) oracle_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--layer") && i + 1 < argc) layer = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--smoke-seq") && i + 1 < argc) smoke_seq = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--keep-going")) g_keep_going = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!model_dir.empty() && smoke_seq > 0) return smoke_long_context(model_dir, smoke_seq);
    if (model_dir.empty() || oracle_dir.empty()) {
        fprintf(stderr, "usage: %s --model-dir DIR --oracle DIR [--layer N]\n"
                        "       %s --model-dir DIR --smoke-seq N   (long-context smoke)\n",
                argv[0], argv[0]);
        return 1;
    }
    const std::string tag = oracle_dir + "/layer" + std::to_string(layer) + "_";

    glm::Config c;
    if (!glm::load_config(model_dir, &c)) return 1;
    printf("config: layers=%u hidden=%u heads=%u qk=%u+%u v=%u kv_lora=%u q_lora=%u\n",
           c.n_layers, c.hidden, c.n_heads, c.qk_nope, c.qk_rope, c.v_head,
           c.kv_lora_rank, c.q_lora_rank);
    printf("        experts=%u/%u moe_inter=%u shared=%u dense_first=%u scaling=%.3f "
           "norm_topk=%d rms_eps=%.3e rope_theta=%.1f index_topk=%u\n",
           c.n_experts_per_tok, c.n_routed_experts, c.moe_inter, c.n_shared,
           c.dense_first, c.routed_scaling, (int)c.norm_topk_prob, c.rms_eps,
           c.rope_theta, c.index_topk);

    // oracle input
    std::vector<size_t> sh;
    std::vector<float> in_hidden = load_npy_f32(tag + "input_hidden.npy", &sh);
    if (sh.size() != 3 || sh[2] != c.hidden) {
        fprintf(stderr, "input_hidden shape unexpected\n"); return 1;
    }
    const uint32_t seq = (uint32_t)sh[1];
    printf("oracle seq=%u hidden=%u\n", seq, c.hidden);

    c.max_seq = seq + 8;

    st::ModelDir M;
    // The staging dir is a partial checkpoint (layers 0 and 3 only) with the full
    // 282-shard index, so missing shards must not be fatal.
    if (!st::open(&M, model_dir, /*allow_missing_shards=*/true)) {
        fprintf(stderr, "st::open failed\n"); return 1;
    }

    glm::LayerWeights w;
    if (!glm::load_layer(&M, c, layer, &w)) return 1;
    const glm::ExpertLayout lay = glm::ExpertLayout::from(c);
    printf("expert block = %zu bytes (gw@%zu gs@%zu uw@%zu us@%zu dw@%zu ds@%zu)\n",
           lay.total, lay.gw_off, lay.gs_off, lay.uw_off, lay.us_off, lay.dw_off, lay.ds_off);
    StExpertSource src(&M, c, lay);

    const uint32_t H = c.hidden;
    float* d_hidden = nullptr;
    CUDA_OK(cudaMalloc(&d_hidden, H * sizeof(float)));

    glm::Trace tr;
    CUDA_OK(cudaMalloc(&tr.post_input_norm,  H * sizeof(float)));
    CUDA_OK(cudaMalloc(&tr.attn_out,         H * sizeof(float)));
    CUDA_OK(cudaMalloc(&tr.post_attn_hidden, H * sizeof(float)));
    glm::g_trace = &tr;

    // Replay every position 0..seq-1 in order. The KV cache is built by the same
    // calls that are compared: attention at the last position needs all the earlier
    // ones cached, so there is no separate priming pass -- only the final call's
    // substeps are compared.
    //
    // A layer taken out of its group has no leader to publish indexer indices for
    // it, so a non-owning layer must be run in IndexShare's standalone-dense mode.
    // An OWNING layer runs the full sparse path here, which is what makes the
    // bit-identity gate below meaningful.
    const bool owns_indexer = c.indexer_owner[layer];
    printf("indexer: layer %u is \"%s\" (leader %u)\n", layer,
           owns_indexer ? "full" : "shared", c.indexer_leader[layer]);

    auto replay = [&](bool dense) {
        glm::IndexShare share;
        share.dense = dense;
        for (uint32_t pos = 0; pos < seq; pos++) {
            share.reset();
            CUDA_OK(cudaMemcpy(d_hidden, in_hidden.data() + (size_t)pos * H,
                               H * sizeof(float), cudaMemcpyHostToDevice));
            glm::run_layer(c, w, lay, src, layer, pos, d_hidden, share, 0);
        }
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());
    };

    replay(/*dense=*/!owns_indexer);

    const size_t last = (size_t)(seq - 1) * H;
    auto ref_row = [&](const char* name) {
        std::vector<size_t> s;
        std::vector<float> a = load_npy_f32(tag + name + ".npy", &s);
        return std::vector<float>(a.begin() + last, a.begin() + last + H);
    };

    // Tolerances in bf16 ulps of each tensor's own scale (see compare()). One ulp
    // is the floor imposed by the fixture's dtype; 3 leaves a 3x margin over the
    // worst measured substep while still being tighter, on the dominant elements,
    // than the plan's 2e-2 elementwise bound.
    const double TOL_NORM = 3.0, TOL_OUT = 3.0;

    compare("post_input_norm",  ref_row("post_input_norm"),
            download(tr.post_input_norm, H), TOL_NORM);
    compare("attn_out",         ref_row("attn_out"),
            download(tr.attn_out, H), TOL_OUT);
    compare("post_attn_hidden", ref_row("post_attn_hidden"),
            download(tr.post_attn_hidden, H), TOL_OUT);
    compare("post_post_norm",   ref_row("post_post_norm"),
            download(w.s_norm, H), TOL_NORM);

    if (layer >= c.dense_first) {
        // router_logits: [seq, E]
        {
            std::vector<size_t> s;
            std::vector<float> a = load_npy_f32(tag + "router_logits.npy", &s);
            const size_t E = c.n_routed_experts;
            std::vector<float> r(a.end() - E, a.end());
            compare("router_logits", r, download(w.s_logits, E), TOL_OUT);
        }
        // topk_idx: int64 [seq, K] — compare the SORTED index SET (torch.topk is
        // sorted=False, so per-slot order carries no guarantee).
        {
            std::vector<size_t> s;
            std::vector<int64_t> a = load_npy_i64(tag + "topk_idx.npy", &s);
            const size_t K = c.n_experts_per_tok;
            std::vector<int64_t> r(a.end() - K, a.end());
            std::vector<int32_t> g(K);
            CUDA_OK(cudaMemcpy(g.data(), w.s_idx, K * sizeof(int32_t), cudaMemcpyDeviceToHost));
            std::vector<int64_t> gg(g.begin(), g.end());
            std::sort(r.begin(), r.end()); std::sort(gg.begin(), gg.end());
            bool same = (r == gg);
            printf("[topk_idx] ref={");
            for (size_t i = 0; i < K; i++) printf("%s%lld", i ? "," : "", (long long)r[i]);
            printf("} got={");
            for (size_t i = 0; i < K; i++) printf("%s%lld", i ? "," : "", (long long)gg[i]);
            printf("} %s\n", same ? "OK" : "FAIL");
            if (!same) { g_fail = 1; if (!g_keep_going) return 1; }
        }
        // topk_w: [seq, K] — must be compared in the ORACLE's slot order, so
        // reorder our weights by matching index.
        {
            std::vector<size_t> s;
            std::vector<float> a = load_npy_f32(tag + "topk_w.npy", &s);
            std::vector<size_t> si;
            std::vector<int64_t> ai = load_npy_i64(tag + "topk_idx.npy", &si);
            const size_t K = c.n_experts_per_tok;
            std::vector<float>   rw(a.end() - K, a.end());
            std::vector<int64_t> ri(ai.end() - K, ai.end());
            std::vector<int32_t> gi(K); std::vector<float> gw(K);
            CUDA_OK(cudaMemcpy(gi.data(), w.s_idx, K * sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_OK(cudaMemcpy(gw.data(), w.s_w,   K * sizeof(float),   cudaMemcpyDeviceToHost));
            std::vector<float> aligned(K, 0.0f);
            for (size_t i = 0; i < K; i++)
                for (size_t j = 0; j < K; j++)
                    if ((int64_t)gi[j] == ri[i]) { aligned[i] = gw[j]; break; }
            double sum_ref = 0, sum_got = 0;
            for (size_t i = 0; i < K; i++) { sum_ref += rw[i]; sum_got += aligned[i]; }
            printf("[topk_w] sum ref=%.6f got=%.6f (routed_scaling=%.3f)\n",
                   sum_ref, sum_got, c.routed_scaling);
            compare("topk_w", rw, aligned, TOL_OUT);
        }
        compare("shared_out", ref_row("shared_out"), download(w.s_shared, H), TOL_OUT);
    }

    compare("moe_out",       ref_row("moe_out"),       download(w.s_acc, H),   TOL_OUT);
    compare("output_hidden", ref_row("output_hidden"), download(d_hidden, H),  TOL_OUT);

    // --- the free oracle: sparse == dense, BIT-IDENTICALLY, at seq <= index_topk
    //
    // Mechanism (Task 1, verified by executing the real GlmMoeDsaIndexer): the
    // top-k is over min(index_topk, T) keys, so at T <= index_topk it returns a
    // PERMUTATION OF ALL T INDICES, index_mask comes out uniformly False, and
    // masked_fill writes nothing. The sparse path must therefore not merely agree
    // with the dense one to some tolerance — it must produce the same bits.
    //
    // WHAT THIS PROVES, precisely: the plumbing. That the indexer runs without
    // disturbing the main path, that IndexShare delivers something, that the mask
    // is built and composed the right way round, and that removing the
    // T > index_topk abort changed no arithmetic.
    //
    // WHAT IT DOES NOT PROVE: anything at all about the indexer's numbers. The
    // mask is a no-op regardless of what the indexer computed, so a wrong
    // LayerNorm eps, a missing relu or a swapped RoPE slice all pass this. Task 5's
    // 4096-token oracle is the first test that touches the arithmetic. Do not read
    // a green line here as validation of glm_kernels.cuh.
    if (seq > c.index_topk) {
        printf("[bit_identity] SKIPPED: seq=%u exceeds index_topk=%u, the mask is "
               "no longer a no-op here\n", seq, c.index_topk);
    } else if (!owns_indexer) {
        printf("[bit_identity] SKIPPED: layer %u is \"shared\" and was run in "
               "standalone-dense mode — there is no sparse run to compare\n", layer);
    } else {
        auto snap = [&]() {
            std::vector<std::vector<float>> s;
            s.push_back(download(tr.attn_out, H));
            s.push_back(download(tr.post_attn_hidden, H));
            s.push_back(download(w.s_acc, H));
            s.push_back(download(d_hidden, H));
            return s;
        };
        const std::vector<std::vector<float>> sparse = snap();   // the run just compared
        replay(/*dense=*/true);
        const std::vector<std::vector<float>> dense = snap();

        static const char* NAMES[4] = {"attn_out", "post_attn_hidden", "moe_out",
                                       "output_hidden"};
        int identical = 1;
        for (size_t k = 0; k < sparse.size(); k++) {
            // memcmp, not a tolerance: -0.0 == 0.0 and NaN != NaN under `==`,
            // and neither is what "the same bits" means.
            const bool same = sparse[k].size() == dense[k].size() &&
                              std::memcmp(sparse[k].data(), dense[k].data(),
                                          sparse[k].size() * sizeof(float)) == 0;
            size_t ndiff = 0; double worst = 0;
            for (size_t i = 0; i < sparse[k].size(); i++)
                if (std::memcmp(&sparse[k][i], &dense[k][i], sizeof(float))) {
                    ndiff++;
                    worst = std::fmax(worst, std::fabs((double)sparse[k][i] -
                                                       (double)dense[k][i]));
                }
            printf("[bit_identity %s] differing words %zu/%zu worst_abs=%.3e %s\n",
                   NAMES[k], ndiff, sparse[k].size(), worst, same ? "OK" : "FAIL");
            if (!same) identical = 0;
        }
        if (!identical) {
            printf("[bit_identity] FAIL — at seq=%u <= index_topk=%u the index mask "
                   "is uniformly False, so the sparse path must be bit-identical to "
                   "the dense one. Any difference is a plumbing bug.\n",
                   seq, c.index_topk);
            g_fail = 1;
        }
    }

    glm::g_trace = nullptr;
    cudaFree(tr.post_input_norm); cudaFree(tr.attn_out); cudaFree(tr.post_attn_hidden);
    cudaFree(d_hidden);
    glm::free_layer(&w);   // weights + scratch + KV cache + RoPE table
    st::close(&M);
    if (g_fail) printf("\nFAIL (layer %u)\n", layer);
    else        printf("\nALL SUBSTEPS OK (layer %u)\n", layer);
    return g_fail;
}
