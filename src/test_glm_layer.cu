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
 *
 * Task 5 adds the long-context arithmetic test, above index_topk, where the
 * index mask stops being a no-op:
 *
 *   ./test_glm_layer --model-dir M --oracle O --layer 2                 (owner)
 *   ./test_glm_layer --model-dir M --oracle O --layer 3 --leader-layer 2 (consumer)
 *   ./test_glm_layer ... --nc topk_minus1        (negative control)
 *
 * --leader-layer L runs layer L on the SAME input hidden state at every position
 * before the compared layer, purely so that its indexer publishes into the
 * shared IndexShare. That is what a "shared" layer consumes in the real chain,
 * and above index_topk it is no longer substitutable by anything else: the
 * selected set is a real 2048-of-T subset and the answer depends on which one.
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
    else if (*dtype == "<i4") esz = 4;   // indexer topk_indices (torch int32)
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
// Worst err_ulp over every gated substep of the run, and which one. This is the
// number the tolerance is derived from (baseline run) and the number a negative
// control's separation is quoted against.
static double g_worst_ulp = 0.0;
static std::string g_worst_tag = "-";
// "native" (the CUDA indexer picks) or "arith" (the oracle's pick is forced in).
static std::string g_phase = "native";
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
// dtype separates them. (That was measured during the original 8-token layer
// bring-up; the DSA plan's task-5-report.md re-derives the gate at seq 4096,
// where attn_out's own peak is 8x smaller and one ulp is correspondingly finer.)
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
    printf("[%s %s] err=%.3f ulp (tol %.1f) max_abs=%.4e rms=%.4e | max|ref|=%.4e "
           "rel_to_scale=%.4e | worst@%zu(ref=%.6f got=%.6f) "
           "elemwise_max_rel=%.4e@%zu(ref=%.3e) %s\n",
           g_phase.c_str(), tag, err_ulp, tol_ulp, max_abs,
           std::sqrt(sse / ref.size()), scale,
           max_abs / (scale + 1e-30), worst, ref[worst], got[worst],
           max_rel, worst_r, ref[worst_r],
           err_ulp < tol_ulp ? "OK" : "FAIL");
    if (err_ulp > g_worst_ulp) { g_worst_ulp = err_ulp; g_worst_tag = tag; }
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

static bool file_exists(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (f) { fclose(f); return true; }
    return false;
}

// A pure report: the same numbers compare() prints, with no gate. Used for the
// indexer's internal tensors, which are DIAGNOSTICS ONLY. They are floored by
// the bf16 indexer key cache (Task 3 measured 4.1e-04 end to end on index
// scores, ~700x the fp32 arithmetic floor) and by tie-breaking, so gating on
// them would be gating on a known-noisy quantity. The gate is on outputs.
static void report(const char* tag, const std::vector<float>& ref,
                   const std::vector<float>& got) {
    if (ref.size() != got.size()) {
        printf("[%s] SIZE MISMATCH ref=%zu got=%zu\n", tag, ref.size(), got.size());
        std::exit(1);
    }
    double max_abs = 0, scale = 0, sse = 0;
    for (size_t i = 0; i < ref.size(); i++) scale = std::fmax(scale, std::fabs((double)ref[i]));
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs((double)ref[i] - (double)got[i]);
        sse += d * d;
        max_abs = std::fmax(max_abs, d);
    }
    printf("[diag %s] max_abs=%.4e rms=%.4e max|ref|=%.4e rel_to_scale=%.4e (n=%zu)\n",
           tag, max_abs, std::sqrt(sse / ref.size()), scale,
           max_abs / (scale + 1e-30), ref.size());
}

// --- negative controls -----------------------------------------------------
// Every one of these defects the CUDA side ONLY, leaving the oracle untouched,
// and is applied to the weights/config of whichever layer actually owns the
// indexer in play (the compared layer if it is "full", else its --leader-layer).
// The question each asks is the same: does the substep comparison against the
// oracle move far enough out of the baseline to be caught?
//
// They are runtime flags rather than source edits so that one build answers all
// of them; the k_norm eps is the exception (a compile-time constant) and is
// injected with -DGLM_INDEXER_LN_EPS at build time instead.
enum NC {
    NC_NONE = 0,
    NC_TOPK_MINUS1,   // select 2047 keys instead of 2048 -- ONE key different
    NC_TOPK_PLUS1,    // select 2049 -- one extra key
    NC_TOPK_SWAP,     // replace --nc-swap N selected indices with unselected ones
    NC_HEADPERM,      // reverse the 32 rows of weights_proj: right gates, wrong heads
    NC_HEADSCALE,     // weights_proj * 2 -- a uniform positive scale
};

static const char* nc_name(int nc) {
    switch (nc) {
        case NC_TOPK_MINUS1: return "topk_minus1";
        case NC_TOPK_PLUS1:  return "topk_plus1";
        case NC_TOPK_SWAP:   return "topk_swap";
        case NC_HEADPERM:    return "headperm";
        case NC_HEADSCALE:   return "headscale";
        default:             return "none";
    }
}

static int nc_parse(const char* s) {
    for (int i = NC_TOPK_MINUS1; i <= NC_HEADSCALE; i++)
        if (!std::strcmp(s, nc_name(i))) return i;
    fprintf(stderr, "unknown --nc %s; one of: topk_minus1 topk_plus1 topk_swap "
                    "headperm headscale\n", s);
    std::exit(1);
}

// There is no `wproj_bf16` control, deliberately, and the reason is worth
// keeping: glm_loader stores weights_proj as fp32 CONVERTED FROM the bf16 on
// disk, so every value in it is already exactly representable in bf16 and
// rounding it back is a provable no-op. A "control" that cannot fail is the
// thing this project keeps finding, so instead the invariant is asserted below
// (nc_wproj_is_bf16_exact) and the fp32-vs-bf16 question — which really lives in
// the matmul's input dtype and accumulation, not in the stored weights — is
// answered on the ORACLE side, by dumping with --weights-proj-dtype bf16 and
// differencing the two oracles. See task-5-report.md.
static size_t nc_wproj_is_bf16_exact(const glm::Config& c, const glm::LayerWeights& w) {
    const size_t N = (size_t)c.index_n_heads * c.hidden;
    std::vector<float> h(N);
    CUDA_OK(cudaMemcpy(h.data(), w.idx_weights_proj, N * sizeof(float),
                       cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (float v : h) {
        uint32_t u; std::memcpy(&u, &v, 4);
        if (u & 0x0000ffffu) bad++;
    }
    return bad;
}

// weights_proj is [index_n_heads, hidden] fp32 on device (glm_loader upcasts it
// there because M:643 keeps it in fp32). These edit it in place, after load.
static void nc_apply_weights(int nc, const glm::Config& c, glm::LayerWeights& w) {
    if (nc != NC_HEADPERM && nc != NC_HEADSCALE) return;
    const size_t H = c.index_n_heads, N = (size_t)H * c.hidden;
    std::vector<float> h(N);
    CUDA_OK(cudaMemcpy(h.data(), w.idx_weights_proj, N * sizeof(float),
                       cudaMemcpyDeviceToHost));
    if (nc == NC_HEADPERM) {
        std::vector<float> p(N);
        for (size_t r = 0; r < H; r++)
            std::memcpy(p.data() + (H - 1 - r) * c.hidden, h.data() + r * c.hidden,
                        c.hidden * sizeof(float));
        h.swap(p);
    } else {  // NC_HEADSCALE
        for (float& v : h) v *= 2.0f;
    }
    CUDA_OK(cudaMemcpy(w.idx_weights_proj, h.data(), N * sizeof(float),
                       cudaMemcpyHostToDevice));
}

// Replace the first `count` entries of one index row with keys that were NOT
// selected. Used two ways:
//   * on the device buffer a leader just published (the real selection), and
//   * on the ORACLE's rows before they are uploaded under --force-oracle-topk,
//     which gives a clean sensitivity curve: the arithmetic residual is then
//     ~0.8 ulp and everything above it is the N misselected keys and nothing
//     else. That is how "how many wrong keys does this test detect" gets a
//     number instead of an opinion.
// Returns how many were actually swapped (fewer than `count` if T is small).
static uint32_t nc_swap_row(int32_t* idx, uint32_t n, uint32_t T, uint32_t count) {
    if (n == 0 || n >= T) return 0;         // below index_topk there is nothing unselected
    std::vector<uint8_t> sel(T, 0);
    for (uint32_t i = 0; i < n; i++)
        if (idx[i] >= 0 && (uint32_t)idx[i] < T) sel[idx[i]] = 1;
    uint32_t swapped = 0;
    for (uint32_t t = 0; t < T && swapped < count; t++)
        if (!sel[t]) { idx[swapped] = (int32_t)t; swapped++; }
    return swapped;
}

static void nc_swap_published(glm::LayerWeights& w, uint32_t T, uint32_t count) {
    uint32_t n = 0;
    CUDA_OK(cudaMemcpy(&n, w.idx_topk_n, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (n == 0 || n >= T) return;
    std::vector<int32_t> idx(n);
    CUDA_OK(cudaMemcpy(idx.data(), w.idx_topk, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    nc_swap_row(idx.data(), n, T, count);
    CUDA_OK(cudaMemcpy(w.idx_topk, idx.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
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
    uint32_t leader_layer = UINT32_MAX;
    int nc = NC_NONE;
    int force_oracle_topk = 0;
    uint32_t nc_swap = 64;
    // Raw [T] fp32 index-score row for the last query. Exists so that two BUILDS
    // can be differenced against each other (the k_norm eps is a compile-time
    // constant, so its control is a rebuild, and "how much did it move" cannot
    // be read off two independent comparisons against the same oracle).
    std::string dump_score;
    double tol_override = 0.0;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--model-dir") && i + 1 < argc) model_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--oracle") && i + 1 < argc) oracle_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--layer") && i + 1 < argc) layer = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--leader-layer") && i + 1 < argc) leader_layer = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--nc") && i + 1 < argc) nc = nc_parse(argv[++i]);
        else if (!std::strcmp(argv[i], "--nc-swap") && i + 1 < argc) nc_swap = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol-ulp") && i + 1 < argc) tol_override = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--force-oracle-topk")) force_oracle_topk = 1;
        else if (!std::strcmp(argv[i], "--dump-index-score") && i + 1 < argc) dump_score = argv[++i];
        else if (!std::strcmp(argv[i], "--smoke-seq") && i + 1 < argc) smoke_seq = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--keep-going")) g_keep_going = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    // A negative control wants EVERY substep's separation, not just the first
    // one that trips: the deliverable is how far each defect moves the
    // comparison, not merely that something moved.
    if (nc != NC_NONE) g_keep_going = 1;

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
    printf("oracle seq=%u hidden=%u  (index_topk=%u -> %s)\n", seq, c.hidden, c.index_topk,
           seq > c.index_topk ? "SPARSE: the index mask really masks"
                              : "below index_topk: the mask is a no-op");

    c.max_seq = seq + 8;

    // The top-k negative controls are a config knob, applied BEFORE load_layer
    // because index_topk sizes idx_topk. Changing it by one key is the smallest
    // perturbation of the selection that exists, which is what makes it the
    // sharpest measure of this test's sensitivity.
    if (nc == NC_TOPK_MINUS1) c.index_topk -= 1;
    if (nc == NC_TOPK_PLUS1)  c.index_topk += 1;
    if (nc != NC_NONE)
        printf("NEGATIVE CONTROL: %s  (index_topk now %u)\n", nc_name(nc), c.index_topk);

    st::ModelDir M;
    // The staging dir is a partial checkpoint (layers 0 and 3 only) with the full
    // 282-shard index, so missing shards must not be fatal.
    if (!st::open(&M, model_dir, /*allow_missing_shards=*/true)) {
        fprintf(stderr, "st::open failed\n"); return 1;
    }

    glm::LayerWeights w;
    if (!glm::load_layer(&M, c, layer, &w)) return 1;

    // --leader-layer: the group leader whose indexer this "shared" layer
    // consumes. It is run on the SAME input hidden state at every position,
    // into its own buffer, purely for its indexer output. Below index_topk this
    // was substitutable (any full index list gave the same mask); above it, it
    // is the only way to reproduce what transformers did.
    const bool has_leader = (leader_layer != UINT32_MAX);
    glm::LayerWeights wl;
    if (has_leader) {
        if (!c.indexer_owner[leader_layer]) {
            fprintf(stderr, "--leader-layer %u does not own an indexer\n", leader_layer);
            return 1;
        }
        if (c.indexer_leader[layer] != leader_layer)
            printf("WARNING: config says layer %u's leader is %u, not %u\n",
                   layer, c.indexer_leader[layer], leader_layer);
        if (!glm::load_layer(&M, c, leader_layer, &wl)) return 1;
        printf("leader: layer %u loaded (runs first at every position, same input)\n",
               leader_layer);
    }
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

    // Which layer's indexer weights the negative controls defect: the compared
    // layer's if it owns one, otherwise the leader's.
    glm::LayerWeights* owner_w = owns_indexer ? &w : (has_leader ? &wl : nullptr);
    if (nc != NC_NONE) {
        if (!owner_w) {
            fprintf(stderr, "--nc needs an indexer in play: either an owning "
                            "--layer or a --leader-layer\n");
            return 1;
        }
        if (nc == NC_TOPK_SWAP && !has_leader && !force_oracle_topk) {
            fprintf(stderr, "--nc topk_swap rewrites a published index row. On a "
                            "\"shared\" layer that is the leader's buffer "
                            "(--leader-layer); on an owning layer the publish and "
                            "the consume happen inside one run_layer call, so use "
                            "--force-oracle-topk, which publishes the oracle's rows "
                            "and lets them be perturbed before upload.\n");
            return 1;
        }
        nc_apply_weights(nc, c, *owner_w);
    }

    // --- --force-oracle-topk: run the CUDA path on TORCH'S selection ---------
    // Above index_topk the two implementations legitimately choose slightly
    // different key sets — Task 3 measured the bf16 key cache floors index-score
    // agreement at ~4e-04 and the selection boundary here is decided by
    // differences of that order. That difference then shows up in attn_out, and
    // without this switch there is no way to tell it apart from an arithmetic
    // bug in the layer.
    //
    // No production code is involved: `IndexShare` already has exactly the
    // interface for it. Marking the compared layer "shared" in this test's own
    // Config copy (AFTER load_layer, so the buffers and weights still exist)
    // makes run_attention consume a published set instead of computing one, and
    // what gets published is the oracle's own row for that position.
    //
    // The oracle's row for position `pos` is the right thing to publish at every
    // pos: below 2048 it contains every causal key plus -inf filler that
    // index_mask_scatter drops as out of range; above 2048 every entry is finite
    // and causal, because there are more than 2048 causal keys to choose from.
    //
    // A normal run does BOTH passes: the native one, gated loosely because a
    // legitimate selection difference lands in it, and this one, gated tightly
    // because nothing but the layer arithmetic is left in it.
    int32_t* d_oracle_idx = nullptr;
    size_t oracle_K = 0;
    int forced_now = force_oracle_topk;
    const uint32_t idx_owner = owns_indexer ? layer : leader_layer;
    const std::string topk_path = oracle_dir + "/layer" +
        std::to_string(owns_indexer ? layer : (has_leader ? leader_layer : layer)) +
        "_topk_indices.npy";
    const bool can_force = (owns_indexer || has_leader) && file_exists(topk_path);
    if (force_oracle_topk && !can_force) {
        fprintf(stderr, "--force-oracle-topk: no %s, or no indexer in play "
                        "(pass --leader-layer for a \"shared\" layer)\n",
                topk_path.c_str());
        return 1;
    }
    if (can_force) {
        const uint32_t owner = idx_owner;
        std::vector<size_t> s; std::string dt;
        std::vector<uint8_t> raw = load_npy_raw(
            oracle_dir + "/layer" + std::to_string(owner) + "_topk_indices.npy", &s, &dt);
        if (dt != "<i4" || s.size() != 3 || s[1] != seq) {
            fprintf(stderr, "topk_indices.npy: expected int32 (1,%u,K)\n", seq); return 1;
        }
        oracle_K = s[2];
        if (nc == NC_TOPK_SWAP) {
            int32_t* rows = (int32_t*)raw.data();
            uint32_t total = 0, rows_hit = 0;
            for (uint32_t p = 0; p < seq; p++) {
                const uint32_t got = nc_swap_row(rows + (size_t)p * oracle_K,
                                                 (uint32_t)oracle_K, p + 1, nc_swap);
                total += got; rows_hit += (got > 0);
            }
            printf("NEGATIVE CONTROL topk_swap: %u indices replaced across %u rows "
                   "(%u requested per row); the last row got %u of %zu keys wrong\n",
                   total, rows_hit, nc_swap,
                   nc_swap < oracle_K ? nc_swap : (uint32_t)oracle_K, oracle_K);
        }
        CUDA_OK(cudaMalloc(&d_oracle_idx, raw.size()));
        CUDA_OK(cudaMemcpy(d_oracle_idx, raw.data(), raw.size(), cudaMemcpyHostToDevice));
        printf("oracle selection available (layer %u, K=%zu)\n", owner, oracle_K);
    }
    if (forced_now) c.indexer_owner[layer] = false;   // consume, do not compute

    float* d_lead = nullptr;
    if (has_leader) CUDA_OK(cudaMalloc(&d_lead, H * sizeof(float)));

    auto replay = [&](bool dense) {
        glm::IndexShare share;
        share.dense = dense;
        for (uint32_t pos = 0; pos < seq; pos++) {
            share.reset();
            if (forced_now && !dense) {
                share.publish(owns_indexer ? layer : leader_layer,
                              d_oracle_idx + (size_t)pos * oracle_K,
                              (uint32_t)oracle_K);
            } else if (has_leader && !dense) {
                // Same input hidden state as the compared layer -- both layers
                // are being run standalone off the oracle's input, exactly as
                // the two oracle dumps were.
                CUDA_OK(cudaMemcpy(d_lead, in_hidden.data() + (size_t)pos * H,
                                   H * sizeof(float), cudaMemcpyHostToDevice));
                glm::run_layer(c, wl, lay, src, leader_layer, pos, d_lead, share, 0);
                if (nc == NC_TOPK_SWAP) {
                    CUDA_OK(cudaDeviceSynchronize());
                    nc_swap_published(wl, pos + 1, nc_swap);
                }
            }
            CUDA_OK(cudaMemcpy(d_hidden, in_hidden.data() + (size_t)pos * H,
                               H * sizeof(float), cudaMemcpyHostToDevice));
            glm::run_layer(c, w, lay, src, layer, pos, d_hidden, share, 0);
        }
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());
    };

    const bool run_dense = !owns_indexer && !has_leader && !forced_now;
    replay(run_dense);

    const size_t last = (size_t)(seq - 1) * H;
    auto ref_row = [&](const char* name) {
        std::vector<size_t> s;
        std::vector<float> a = load_npy_f32(tag + name + ".npy", &s);
        return std::vector<float>(a.begin() + last, a.begin() + last + H);
    };

    // Tolerances in bf16 ulps of each tensor's own scale (see compare()). One ulp
    // is the floor imposed by the fixture's dtype.
    //
    // TASK 5 DERIVATION. The 8-token gate was 3.0 ulp against a worst measured
    // 1.06. That number is NOT imported here: at 4096 the attention sum runs over
    // 512x more keys and the residual is expected to grow, so the gate is
    // re-derived from a measured baseline at this length. See task-5-report.md
    // for the measured worst substep at seq 4096 on layers 2 and 3 and the margin
    // this constant leaves over it. --tol-ulp overrides it for the derivation run
    // itself (use with --keep-going to see every substep).
    //
    // TWO gates, because there are two independent things to gate and one number
    // cannot do both:
    //
    //   TOL_NATIVE — the CUDA indexer picks its own key set. A legitimate
    //     selection difference lands in this pass (measured: 5 of 2048 keys, all
    //     within 2e-4 of the selection boundary, which is inside the 5e-3
    //     index-score noise floor), so the residual here is dominated by that,
    //     not by arithmetic. Baseline measured 6.85 ulp (layer 2) / 10.05 ulp
    //     (layer 3), both on attn_out; the gate is 2x the worse of the two.
    //     It catches a GROSS selection error and nothing finer — measured
    //     detection threshold is ~128 wrong keys of 2048 (see the topk_swap
    //     sensitivity curve in task-5-report.md).
    //
    //   TOL_ARITH — the same run with the ORACLE's key set forced in, so the
    //     selection is identical by construction and only the layer arithmetic
    //     is left. Baseline measured 0.83 ulp (layer 2) / 0.59 ulp (layer 3),
    //     both on post_input_norm, i.e. the bf16 fixture floor and nothing more.
    //     Gate 3.0, a 3.6x margin. This is the tight one, and it is what would
    //     catch an attention/MoE regression that the loose pass would swallow.
    //
    // The 8-token gate of 3.0 ulp is NOT imported into either; both were
    // re-measured at seq 4096, where attn_out's own peak is 8x smaller (the
    // softmax averages over 2048 keys) and one ulp is correspondingly finer.
    const double TOL_ARITH = 3.0;
    // At seq <= index_topk there IS no selection difference to allow for — the
    // top-k is a permutation of all T keys and the mask is a no-op — so the
    // loose gate must not apply there. It would silently weaken every
    // short-context run, including the layer-0 and layer-3 8-token fixtures.
    double TOL_NATIVE = (seq > c.index_topk) ? 20.0 : TOL_ARITH;
    if (tol_override > 0.0) TOL_NATIVE = tol_override;

    auto compare_all = [&](double tol) {
    double TOL_NORM = tol, TOL_OUT = tol;
    printf("gate: %.2f bf16-ulp of each tensor's own scale\n", TOL_OUT);

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
            if (!same) g_fail = 1;
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
    };  // compare_all

    g_phase = forced_now ? "arith" : "native";
    compare_all(forced_now ? TOL_ARITH : TOL_NATIVE);

    // --- indexer diagnostics, NOT a gate ------------------------------------
    // Task 3 measured that the bf16 indexer key cache floors index-score
    // agreement at ~4e-04, ~700x the fp32 arithmetic floor, because 0.048% of
    // cached key words round the other way and each k[t] feeds all 32 heads.
    // Separately, after the relu many scores are exactly 0.0, so above 2048 keys
    // the tie tail is under-determined and CUDA CANNOT match torch index for
    // index. Both are reasons the index SET legitimately differs, which is why
    // the gate above is on outputs. These numbers exist so that a gate failure
    // can be localised, and so the size of the legitimate difference is on the
    // record rather than assumed.
    if (owner_w && owner_w->idx_weights_proj) {
        const size_t bad = nc_wproj_is_bf16_exact(c, *owner_w);
        printf("[invariant] weights_proj fp32 values not exactly bf16: %zu of %zu "
               "%s\n", bad, (size_t)c.index_n_heads * c.hidden,
               bad == 0 ? "OK (so a CUDA-side \"round it to bf16\" control would "
                          "be a provable no-op — see nc_wproj_is_bf16_exact)"
                        : "FAIL — the loader's bf16->fp32 conversion is lossy?!");
        if (bad) g_fail = 1;
    }

    if (owner_w && owner_w->s_iindex && !force_oracle_topk) {
        const uint32_t T = seq;
        const std::string otag = oracle_dir + "/layer" +
                                 std::to_string(owns_indexer ? layer : leader_layer) + "_";
        printf("--- indexer diagnostics (owner layer %u, T=%u) ---\n",
               owns_indexer ? layer : leader_layer, T);

        if (file_exists(otag + "indexer_w_last.npy")) {
            std::vector<size_t> s;
            std::vector<float> r = load_npy_f32(otag + "indexer_w_last.npy", &s);
            report("indexer_w", r, download(owner_w->s_iw, c.index_n_heads));
        }
        if (file_exists(otag + "indexer_q_last.npy")) {
            std::vector<size_t> s;
            std::vector<float> r = load_npy_f32(otag + "indexer_q_last.npy", &s);
            report("indexer_q", r,
                   download(owner_w->s_iq, (size_t)c.index_n_heads * c.index_head_dim));
        }
        if (file_exists(otag + "indexer_k.npy")) {
            // The oracle's k is fp32-of-bf16; ours is the bf16 cache decoded.
            std::vector<size_t> s;
            std::vector<float> r = load_npy_f32(otag + "indexer_k.npy", &s);
            const size_t n = (size_t)T * c.index_head_dim;
            std::vector<uint16_t> raw(n);
            CUDA_OK(cudaMemcpy(raw.data(), owner_w->idx_k_cache, n * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost));
            std::vector<float> g(n);
            for (size_t i = 0; i < n; i++) {
                const uint32_t u = (uint32_t)raw[i] << 16;
                std::memcpy(&g[i], &u, 4);
            }
            r.resize(n);
            report("indexer_k_cache(bf16)", r, g);
        }
        if (!dump_score.empty()) {
            const std::vector<float> g = download(owner_w->s_iindex, T);
            FILE* f = fopen(dump_score.c_str(), "wb");
            if (!f) { fprintf(stderr, "open %s failed\n", dump_score.c_str()); return 1; }
            fwrite(g.data(), sizeof(float), T, f);
            fclose(f);
            printf("index score row written to %s (%u float32)\n", dump_score.c_str(), T);
        }

        // The one INDEXER-specific gate. It is a score vector, not an index set:
        // the ban in the brief is on index sets, whose tail is under-determined,
        // and the reason index sets cannot be gated (a key within the noise of
        // the boundary may fall either way) does not apply to the continuous
        // quantity that decides them.
        //
        // Measured floor: 5.34e-03 relative to the row's own peak, at layer 2,
        // seq 4096. That is NOT the fp32 arithmetic floor (Task 3 measured
        // 5.8e-07) and it is not the bf16 key cache alone either (Task 3's
        // 4.1e-04): transformers keeps the indexer's q in BF16 (wq_b is a bf16
        // Linear, M:232) while this path keeps it in fp32, and the diagnostic
        // above shows q differing by 1.5e-03 of its own scale, ~0.8 bf16 ulp.
        // That term dominates. Gate 2.0e-02, a 3.7x margin over the measured
        // floor.
        std::vector<float> ref_score;
        if (file_exists(otag + "indexer_index_score_last.npy")) {
            std::vector<size_t> s;
            ref_score = load_npy_f32(otag + "indexer_index_score_last.npy", &s);
            ref_score.resize(T);
            const std::vector<float> got = download(owner_w->s_iindex, T);
            double ma = 0, sc = 0;
            for (uint32_t i = 0; i < T; i++) sc = std::fmax(sc, std::fabs((double)ref_score[i]));
            for (uint32_t i = 0; i < T; i++)
                ma = std::fmax(ma, std::fabs((double)ref_score[i] - (double)got[i]));
            const double rel = ma / (sc + 1e-30);
            const double TOL_SCORE = 2.0e-2;
            printf("[index_score] rel_to_scale=%.4e (tol %.1e) max_abs=%.4e "
                   "max|ref|=%.4e %s\n", rel, TOL_SCORE, ma, sc,
                   rel < TOL_SCORE ? "OK" : "FAIL");
            if (!(rel < TOL_SCORE)) { g_fail = 1; if (!g_keep_going) std::exit(1); }
        }

        // Index-set overlap: reported, never gated. Two independent reasons it
        // is < 100% legitimately (above).
        if (file_exists(otag + "topk_indices.npy")) {
            std::vector<size_t> s; std::string dt;
            std::vector<uint8_t> raw = load_npy_raw(otag + "topk_indices.npy", &s, &dt);
            const size_t K = s.back();
            const int32_t* rows = (const int32_t*)raw.data();
            const int32_t* last_row = rows + (size_t)(seq - 1) * K;
            uint32_t n = 0;
            CUDA_OK(cudaMemcpy(&n, owner_w->idx_topk_n, sizeof(uint32_t),
                               cudaMemcpyDeviceToHost));
            std::vector<int32_t> got(n);
            CUDA_OK(cudaMemcpy(got.data(), owner_w->idx_topk, n * sizeof(int32_t),
                               cudaMemcpyDeviceToHost));
            std::vector<uint8_t> in_ref(T, 0);
            for (size_t i = 0; i < K; i++)
                if (last_row[i] >= 0 && (uint32_t)last_row[i] < T) in_ref[last_row[i]] = 1;
            size_t both = 0, only_got = 0;
            double worst_dropped = -1e30, worst_added = 1e30;
            for (uint32_t i = 0; i < n; i++) {
                const int32_t v = got[i];
                if (v >= 0 && (uint32_t)v < T && in_ref[v]) both++;
                else { only_got++;
                       if (!ref_score.empty() && v >= 0 && (uint32_t)v < T)
                           worst_added = std::fmin(worst_added, ref_score[v]); }
            }
            if (!ref_score.empty())
                for (size_t i = 0; i < K; i++) {
                    const int32_t v = last_row[i];
                    if (v < 0 || (uint32_t)v >= T) continue;
                    bool found = false;
                    for (uint32_t j = 0; j < n && !found; j++) found = (got[j] == v);
                    if (!found) worst_dropped = std::fmax(worst_dropped, ref_score[v]);
                }
            printf("[diag topk_set] ref K=%zu cuda n=%u  shared=%zu (%.4f%%)  "
                   "cuda-only=%zu\n", K, n, both, 100.0 * (double)both / (double)n,
                   only_got);
            if (!ref_score.empty() && only_got)
                printf("[diag topk_set] highest oracle score CUDA dropped %.6e, "
                       "lowest oracle score CUDA added %.6e  (all ties at the "
                       "selection boundary)\n", worst_dropped, worst_added);
            size_t zeros = 0;
            for (uint32_t t = 0; t < T && !ref_score.empty(); t++)
                if (ref_score[t] == 0.0f) zeros++;
            if (!ref_score.empty())
                printf("[diag topk_set] oracle index scores exactly 0.0: %zu of %u "
                       "(%.2f%%) — the tie tail Task 1 predicted\n",
                       zeros, T, 100.0 * (double)zeros / (double)T);
        }
    }

    // --- phase 2: the same layer on the ORACLE's selection --------------------
    // Everything above ran with the CUDA indexer choosing. Run it again with
    // torch's own choice forced in, so that the selection is identical by
    // construction and what is left is only the layer's arithmetic — gated 6x
    // tighter. Skipped when --force-oracle-topk already made phase 1 the forced
    // one, and when there is nothing to force.
    double native_worst = g_worst_ulp;
    std::string native_worst_tag = g_worst_tag;
    // Not under a negative control: there the verdict must come from the one
    // pass the defect actually reaches, and phase 2 forces the oracle's
    // selection, which erases every indexer-side defect by construction.
    if (can_force && !forced_now && nc == NC_NONE) {
        printf("\n--- phase 2: ORACLE selection forced (layer %u, K=%zu); the CUDA "
               "indexer does not run, so only the layer arithmetic is left ---\n",
               idx_owner, oracle_K);
        forced_now = 1;
        c.indexer_owner[layer] = false;
        g_worst_ulp = 0.0; g_worst_tag = "-";
        g_phase = "arith";
        replay(/*dense=*/false);
        compare_all(TOL_ARITH);
        printf("phase 2 worst: %.4f ulp (%s), gate %.2f\n",
               g_worst_ulp, g_worst_tag.c_str(), TOL_ARITH);
        c.indexer_owner[layer] = owns_indexer;
        forced_now = force_oracle_topk;
        g_phase = "native";
        // Report phase 1's worst as the headline again -- it is the larger and
        // it is the one that describes the system as it actually runs.
        if (native_worst > g_worst_ulp) { g_worst_ulp = native_worst; g_worst_tag = native_worst_tag; }
    }

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
    if (d_lead) cudaFree(d_lead);
    if (d_oracle_idx) cudaFree(d_oracle_idx);
    glm::free_layer(&w);   // weights + scratch + KV cache + RoPE table
    if (has_leader) glm::free_layer(&wl);
    st::close(&M);
    printf("\nworst gated substep: %.4f ulp (%s), gate %.2f\n",
           g_worst_ulp, g_worst_tag.c_str(),
           force_oracle_topk ? TOL_ARITH : TOL_NATIVE);
    if (nc != NC_NONE) {
        // Under a negative control a FAILURE is the pass condition. Say so
        // explicitly rather than letting a bare exit code be read backwards.
        printf("\nNEGATIVE CONTROL %s: %s\n", nc_name(nc),
               g_fail ? "CAUGHT (the comparison failed, which is the pass "
                        "condition here)"
                      : "NOT CAUGHT — this defect is INVISIBLE to this test");
        return g_fail ? 0 : 1;
    }
    if (g_fail) printf("\nFAIL (layer %u)\n", layer);
    else        printf("\nALL SUBSTEPS OK (layer %u)\n", layer);
    return g_fail;
}
