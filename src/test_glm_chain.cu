/*
 * test_glm_chain.cu — the whole GLM-5.2 stack (78 decoder layers + model.norm +
 * lm_head) in CUDA, compared against ref_glm_chain.py's numpy fixtures.
 *
 * Build: make test_glm_chain
 * Run:   ./test_glm_chain --model-dir ./glm52-mxfp4 \
 *                         --packed ./packed_experts --ref glm-ref
 *
 * WHAT IS COMPARED, in order of authority
 *   1. top-5 token ids, exactly, for every generated token (the end-to-end gate;
 *      --top5-set relaxes it to the SET and the argmax, see g_top5_set_only);
 *   2. per-substep tensors at the --layers set at t=0 (default 0/3/40/77; Task 7
 *      runs 0/2/3/40/77, adding an indexer OWNER and its CONSUMER);
 *   3. the indexer's index score at every sampled owning layer — the only row
 *      that can see a rescaled weights_proj, since top-k is scale-invariant;
 *   4. the final logits vector at every generated token.
 *
 * TWO ROWS ARE REPORTED AND NEVER GATED: the indexer's selected index SET (the
 * two sides keep the indexer's q and its key cache in different dtypes and
 * legitimately disagree at the score noise floor — Task 5 concern 1, Task 6
 * concern 1), and every substep past --gate-depth (Task 7 measured the chain to
 * be chaotic at 1400 tokens; see g_gate_depth).
 *
 * THE TOLERANCE IS NOT INHERITED. Task 5's bf16-ulp gate and Task 9's 1e-2
 * elementwise bar were both calibrated against a *bfloat16* oracle, where the
 * bf16 cast — not the arithmetic — set the floor. Here both sides are float32
 * running the same maths, differing only in summation order (and in this path's
 * absorbed MLA versus the reference's direct MLA, which is the same algebra
 * evaluated in a different order). Importing 1e-2 would be four orders of
 * magnitude too loose and would pass the exact defects this test exists to
 * catch — the layer-3 measurement in task-9-report.md has a flipped MXFP4
 * nibble order sitting at 1.94e-2 on an output hidden state.
 *
 * The gate below is therefore derived from the run itself: --report prints
 * every substep's agreement with no gate applied, and the TOL_REL constant is
 * set from the worst figure that sweep produced, with a stated margin. See
 * task-10-report.md for the numbers and for the negative controls that show the
 * gate can fail.
 *
 * THE METRIC IS rel_to_scale = max_i |ref_i - got_i| / max_i |ref_i|.
 * A whole-tensor relative figure, not an elementwise one: the elementwise
 * d/(|ref|+1e-4) form is unbounded wherever the reference cancels to near zero
 * and its 1e-4 floor presumes O(1) tensors, which these are not. Both the
 * elementwise figure and the bf16-ulp figure are printed alongside for
 * continuity with Tasks 5 and 9; neither is the gate.
 *
 * LAYER 0 IS DENSE (layer < first_k_dense_replace == 3), so router_logits,
 * topk_idx, topk_w and shared_out do not exist there and no fixture was
 * written. They are skipped, not failed.
 */
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "safetensors_io.cuh"
#include "glm_primitives.cuh"
#include "glm_kernels.cuh"
#include "glm_loader.cuh"
#include "glm_layer_runner.cuh"
#include "glm_expert_cache.cuh"

#ifndef CUDA_OK
#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)
#endif

// --- minimal .npy reader ---------------------------------------------------
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

    size_t dp = hdr.find("descr");
    if (dp == std::string::npos) { fprintf(stderr, "%s: no descr\n", path.c_str()); std::exit(1); }
    dp = hdr.find(':', dp);
    size_t q1 = hdr.find_first_of("'\"", dp);
    size_t q2 = hdr.find_first_of("'\"", q1 + 1);
    *dtype = hdr.substr(q1 + 1, q2 - q1 - 1);
    size_t esz = 0;
    if      (*dtype == "<f4") esz = 4;
    else if (*dtype == "<i4") esz = 4;
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

static std::vector<float> load_npy_f32(const std::string& path) {
    std::vector<size_t> sh; std::string dt;
    std::vector<uint8_t> raw = load_npy_raw(path, &sh, &dt);
    if (dt != "<f4") { fprintf(stderr, "%s: expected f32, got %s\n", path.c_str(), dt.c_str()); std::exit(1); }
    std::vector<float> out(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

// ref_chain_topk_{t}.npy and ref_chain_topk_idx_L*.npy are both int32
// (np.argsort(...).astype(np.int32) / idx.astype(np.int32) in ref_glm_chain.py).
static std::vector<int32_t> load_npy_i32(const std::string& path) {
    std::vector<size_t> sh; std::string dt;
    std::vector<uint8_t> raw = load_npy_raw(path, &sh, &dt);
    if (dt != "<i4") { fprintf(stderr, "%s: expected int32, got %s\n", path.c_str(), dt.c_str()); std::exit(1); }
    std::vector<int32_t> out(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

// --- fixture pre-flight ----------------------------------------------------
// The layers the substep fixtures were written for. File scope so the
// pre-flight below can name the exact set of files this run will read.
//
// The default is the four-layer set the committed short-context fixtures under
// glm-ref/ were written with. --layers overrides it; Task 7's long-context run
// uses 0,2,3,40,77, where 2 is an indexer OWNER and 3 a CONSUMER of layer 2's
// indices, so the ownership boundary is inside the compared set rather than
// straddled by it.
static std::vector<uint32_t> g_sampled = {0, 3, 40, 77};

// Indexer taps exist only on "full" (owning) layers, and only in fixture dirs
// produced after Task 6 taught ref_glm_chain.py to write them. --require-indexer
// turns a missing one into a failure instead of a skip.
static int g_require_indexer = 0;
static int g_have_indexer    = 0;   // set by the pre-flight: fixtures are present

// A MISSING ORACLE IS NOT A PASS. Without this, an absent or half-written
// glm-ref/ surfaced either as an exit deep inside load_npy_raw — after the ~78
// layers of weights had already been loaded, minutes in — or, for prompt.txt,
// as the bare line "open glm-ref/prompt.txt" with no indication that the thing
// missing was the oracle this test exists to compare against. Every file the
// run will read is checked up front, and every missing one is named.
static bool fixture_exists(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static const char* INDEXER_TAPS[4] = {"indexer_q", "indexer_w",
                                      "indexer_index_score", "indexer_topk"};

static bool require_fixtures(const std::string& ref_dir, const glm::Config& c,
                             uint32_t n_gen, const std::string& prompt_path) {
    std::vector<std::string> want;
    want.push_back(prompt_path);
    // Are the indexer taps present at all? Decided from the first sampled owning
    // layer, then required everywhere: a dir with taps for layer 0 but not for
    // layer 2 is a half-written fixture set, which is a failure, not a skip.
    for (uint32_t l : g_sampled) {
        if (l < c.n_layers && c.indexer_owner[l] &&
            fixture_exists(ref_dir + "/ref_chain_indexer_index_score_L" +
                           std::to_string(l) + "_t0.npy")) { g_have_indexer = 1; break; }
    }
    if (g_require_indexer) g_have_indexer = 1;
    for (uint32_t l : g_sampled) {
        if (l >= c.n_layers) continue;
        const std::string S = "_L" + std::to_string(l) + "_t0.npy";
        for (const char* n : {"post_input_norm", "attn_out", "post_attn_hidden",
                              "post_post_norm", "moe_out", "output_hidden"})
            want.push_back(ref_dir + "/ref_chain_" + n + S);
        if (l >= c.dense_first)
            for (const char* n : {"router_logits", "topk_idx", "topk_w", "shared_out"})
                want.push_back(ref_dir + "/ref_chain_" + n + S);
        if (g_have_indexer && c.indexer_owner[l])
            for (const char* n : INDEXER_TAPS)
                want.push_back(ref_dir + "/ref_chain_" + n + S);
    }
    for (uint32_t t = 0; t < n_gen; t++) {
        want.push_back(ref_dir + "/ref_chain_logits_" + std::to_string(t) + ".npy");
        want.push_back(ref_dir + "/ref_chain_topk_" + std::to_string(t) + ".npy");
    }
    std::vector<std::string> missing;
    for (const std::string& p : want) if (!fixture_exists(p)) missing.push_back(p);
    if (missing.empty()) return true;
    fprintf(stderr, "test_glm_chain: %zu of %zu oracle fixtures are missing under "
                    "--ref %s — there is nothing to compare against, so this run "
                    "is a FAILURE, not a pass:\n",
            missing.size(), want.size(), ref_dir.c_str());
    for (size_t i = 0; i < missing.size() && i < 12; i++)
        fprintf(stderr, "  missing: %s\n", missing[i].c_str());
    if (missing.size() > 12)
        fprintf(stderr, "  ... and %zu more\n", missing.size() - 12);
    fprintf(stderr, "regenerate them with tools/ref_glm_chain.py\n");
    return false;
}

// --- comparison ------------------------------------------------------------
static int    g_fail = 0;
static int    g_report_only = 0;     // --report: print, never fail; used to derive TOL_REL
static double g_worst_rel = 0.0;
static std::string g_worst_tag = "(none)";

// Derived, not inherited. Measured with --report (no gate) over all 34 sampled
// substeps + 5 logits vectors: the worst agreement was 1.6928e-06, at
// attn_out @ L77, and the whole distribution spans 4.5e-08 .. 1.7e-06 with a
// clear depth trend (L0 ~1e-07, L77 ~1e-06). This gate sits 5.9x above the
// worst baseline.
//
// It can fail: three defects injected into copies of this path (task-10-report.md)
// all trip it, at worst-substep rel 1.86e+00 (MXFP4 nibbles swapped),
// 9.45e-01 (routed_scaling_factor 1.0 instead of 2.5) and 3.38e-01 (the two
// LoRA norms given rms_norm_eps 1e-5 instead of LORA_NORM_EPS 1e-6) —
// 3.4e+04x to 1.9e+05x over this gate. Note the third one still produced the
// CORRECT top-5 at t=0: the end-to-end gate alone would have missed it, and
// only the per-substep comparison caught it (36 failing substeps, the tightest
// at 8.7x over the gate).
//
// --tol overrides it. Task 7 runs this same chain on a 1400-token prompt, where
// every attention output is a sum over 1400 keys instead of 29 and the two
// summation orders have correspondingly more room to diverge; that run derives
// its own gate from its own --report sweep and states the margin. The default
// stays at the short-context figure so the regression keeps gating where it
// always did.
static double TOL_REL = 1e-5;

// The index-score gate. It exists because top-k is SCALE-INVARIANT: multiply
// every head gate by a constant and the selected set, the mask and every output
// tensor are bit-identical, so an output-only comparison cannot see it. Task 5
// measured that defect at 188x on the score and 1.00x on the outputs, Task 6 at
// 251x / 1.00x. Without this row that whole class — a wrong 128**-0.5, a wrong
// 32**-0.5, a scaled weights_proj — is undetectable here too.
static double TOL_INDEX = 2.0e-2;

// --nc jitter's perturbation size. 1e-7 relative is a rounding difference; the
// sweep over larger values is what turns "the chain is sensitive" into a
// measured amplification factor.
static float g_jitter_rel = 1e-7f;

// --gate-depth L: layers <= L are gated at TOL_REL, deeper ones are compared and
// printed but never gated.
//
// This exists because Task 7 MEASURED the 78-layer chain to be chaotic at 1400
// tokens: a 1e-7 relative perturbation of ONE element of each position's
// residual stream after layer 0 moves the final logits by 2.4e-02 relative, and
// a 1e-6 one by 1.7e-02 — i.e. the response saturates, so ANY difference above
// the float32 rounding floor produces the same O(1e-2) spread. The
// CUDA-vs-reference disagreement at layer 40 (7.4e-02) is that spread, not a
// defect: the two implementations differ by summation order alone at layer 3
// (6.3e-07 on the residual stream) and the chain amplifies it ~1.32x per layer.
//
// A gate placed above that spread would be too loose to catch anything; one
// placed below it would fail a correct build. So the deep layers are reported
// and not gated, on the same principle Tasks 5 and 6 applied to the index SET,
// and the end-to-end claim at long context rests on the argmax and the top-5 SET
// (both exact) rather than on the top-5 ORDER. At short context nothing is
// exempt and the default gates every layer.
static uint32_t g_gate_depth = UINT32_MAX;

// --top5-set: gate the top-5 SET and the argmax, report the ORDER. Two logits
// separated by less than the chaotic spread can swap rank without either side
// being wrong; at 1400 tokens ranks 2 and 3 are 0.27 apart against a measured
// spread of 1.55.
static int g_top5_set_only = 0;

// Worst figures, kept apart: report-only rows (indexer_q, indexer_w) must not
// move the headline number the gate is derived from.
static double g_worst_report = 0.0;
static std::string g_worst_report_tag = "(none)";

static void compare(const std::string& tag, const std::vector<float>& ref,
                    const std::vector<float>& got, double tol_rel,
                    bool gated = true) {
    if (ref.size() != got.size()) {
        printf("[%-34s] SIZE MISMATCH ref=%zu got=%zu\n", tag.c_str(), ref.size(), got.size());
        g_fail = 1; return;
    }
    double max_abs = 0, max_rel_e = 0, scale = 0, sse = 0; size_t worst = 0;
    for (size_t i = 0; i < ref.size(); i++) scale = std::fmax(scale, std::fabs((double)ref[i]));
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs((double)ref[i] - (double)got[i]);
        sse += d * d;
        if (d > max_abs) { max_abs = d; worst = i; }
        const double r = d / (std::fabs((double)ref[i]) + 1e-4);
        if (r > max_rel_e) max_rel_e = r;
    }
    const double rel   = max_abs / (scale + 1e-300);
    const double ulp   = std::exp2(-8.0) * scale;
    const double e_ulp = max_abs / (ulp + 1e-300);
    const bool   ok    = (rel < tol_rel);
    if (gated) { if (rel > g_worst_rel) { g_worst_rel = rel; g_worst_tag = tag; } }
    else       { if (rel > g_worst_report) { g_worst_report = rel; g_worst_report_tag = tag; } }
    printf("[%-34s] rel=%.3e (tol %.1e) | max_abs=%.4e rms=%.4e max|ref|=%.4e "
           "| bf16_ulp=%.4f elemwise=%.3e | worst@%zu ref=%.6f got=%.6f  %s\n",
           tag.c_str(), rel, tol_rel, max_abs, std::sqrt(sse / ref.size()), scale,
           e_ulp, max_rel_e, worst, ref[worst], got[worst],
           !gated ? "(reported, not gated)" : (g_report_only ? "-" : (ok ? "OK" : "FAIL")));
    if (!ok && !g_report_only && gated) g_fail = 1;
}

static std::vector<float> download(const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

// --- substep capture -------------------------------------------------------
// Every sampled layer gets its own set of device buffers; the hook snapshots
// them at the one (layer, pos) the fixtures were written at. Nothing here
// touches run_layer's arithmetic.
struct LayerSnap {
    float*   post_input_norm  = nullptr;   // [H]  (from glm::Trace)
    float*   attn_out         = nullptr;   // [H]  (from glm::Trace)
    float*   post_attn_hidden = nullptr;   // [H]  (from glm::Trace)
    float*   post_post_norm   = nullptr;   // [H]  = w.s_norm
    float*   router_logits    = nullptr;   // [E]  = w.s_logits   (MoE layers only)
    float*   topk_w           = nullptr;   // [K]  = w.s_w        (MoE layers only)
    int32_t* topk_idx         = nullptr;   // [K]  = w.s_idx      (MoE layers only)
    float*   shared_out       = nullptr;   // [H]  = w.s_shared   (MoE layers only)
    float*   moe_out          = nullptr;   // [H]  = w.s_acc
    float*   output_hidden    = nullptr;   // [H]  = d_hidden
    // Indexer taps, on "full" (owning) layers only. These are the buffers
    // launch_indexer_query wrote for the last position of the layer's attention
    // pass — which is the position this hook is allowed to observe, so nothing
    // has overwritten them by the time after_layer runs.
    float*   indexer_q        = nullptr;   // [heads*dim] = w.s_iq   (post-RoPE)
    float*   indexer_w        = nullptr;   // [heads]     = w.s_iw
    float*   indexer_score    = nullptr;   // [T]         = w.s_iindex
    int32_t* indexer_topk     = nullptr;   // [k_sel]     = w.idx_topk
    bool     owner            = false;
    bool     seen             = false;
};

// NEGATIVE CONTROL 3 / sensitivity probe. Adds `rel` times the element's own
// magnitude to ONE element of the residual stream. That is the size of a
// floating-point rounding difference, not of a defect: if a perturbation this
// small comes out the far end of the chain at the same magnitude as the
// CUDA-vs-reference disagreement, then the disagreement is amplification of
// rounding and no per-substep gate can separate it from one.
// `h` is the LAST position's row of the layer-major prefill's [n, H] residual
// buffer, so the rows below it are the earlier positions — perturbing all of
// them is what makes this probe say something about the whole prefill rather
// than about one row. (Under --prefill position there is only the one row; the
// probe then perturbs only it, and says so.)
__global__ void jitter_rows(float* last_row, uint32_t n, uint32_t H, float rel) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float* row = last_row - (size_t)(n - 1 - i) * H;
    row[i % H] += row[i % H] * rel;
}

struct SubstepHook : glm::ChainHook {
    glm::Config c;
    uint32_t t0_pos = 0;
    glm::Trace* trace = nullptr;
    std::map<uint32_t, LayerSnap> snap;   // layer -> buffers
    uint32_t jitter_layer = UINT32_MAX;   // --nc jitter
    float    jitter_rel   = 0.0f;

    uint32_t T     = 0;    // keys visible at t0_pos = t0_pos + 1
    uint32_t k_sel = 0;    // min(index_topk, T), M:262
    bool     want_indexer = false;

    void add_layer(uint32_t l) {
        LayerSnap s;
        const size_t H = c.hidden, E = c.n_routed_experts, K = c.n_experts_per_tok;
        auto F = [](float** p, size_t n) { CUDA_OK(cudaMalloc(p, n * sizeof(float))); };
        F(&s.post_input_norm, H); F(&s.attn_out, H); F(&s.post_attn_hidden, H);
        F(&s.post_post_norm, H);  F(&s.moe_out, H);  F(&s.output_hidden, H);
        if (l >= c.dense_first) {
            F(&s.router_logits, E); F(&s.topk_w, K); F(&s.shared_out, H);
            CUDA_OK(cudaMalloc(&s.topk_idx, K * sizeof(int32_t)));
        }
        s.owner = c.indexer_owner[l];
        if (s.owner && want_indexer) {
            F(&s.indexer_q, (size_t)c.index_n_heads * c.index_head_dim);
            F(&s.indexer_w, c.index_n_heads);
            F(&s.indexer_score, T);
            CUDA_OK(cudaMalloc(&s.indexer_topk, (size_t)k_sel * sizeof(int32_t)));
        }
        snap[l] = s;
    }

    // The fixtures are all at t=0, i.e. the last prompt position — which is the
    // one position the layer-major prefill can serve (see ChainHook). Declaring
    // it makes run_chain fail loudly if that ever stops being true, instead of
    // leaving `seen` false and reporting "hook never fired".
    uint32_t observed_pos() const override { return t0_pos; }

    void after_layer(uint32_t layer, uint32_t pos, const glm::LayerWeights& w,
                     const float* d_hidden, cudaStream_t stream) override {
        if (pos != t0_pos) return;
        if (layer == jitter_layer) {
            const uint32_t nrow = glm::g_prefill_layer_major ? (t0_pos + 1) : 1;
            jitter_rows<<<(nrow + 255) / 256, 256, 0, stream>>>(
                const_cast<float*>(d_hidden), nrow, c.hidden, jitter_rel);
        }
        auto it = snap.find(layer);
        if (it == snap.end()) return;
        LayerSnap& s = it->second;
        const size_t H = c.hidden, E = c.n_routed_experts, K = c.n_experts_per_tok;
        auto D2D = [&](void* dst, const void* src, size_t bytes) {
            CUDA_OK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
        };
        D2D(s.post_input_norm,  trace->post_input_norm,  H * sizeof(float));
        D2D(s.attn_out,         trace->attn_out,         H * sizeof(float));
        D2D(s.post_attn_hidden, trace->post_attn_hidden, H * sizeof(float));
        D2D(s.post_post_norm,   w.s_norm,                H * sizeof(float));
        D2D(s.moe_out,          w.s_acc,                 H * sizeof(float));
        D2D(s.output_hidden,    d_hidden,                H * sizeof(float));
        if (layer >= c.dense_first) {
            D2D(s.router_logits, w.s_logits, E * sizeof(float));
            D2D(s.topk_w,        w.s_w,      K * sizeof(float));
            D2D(s.topk_idx,      w.s_idx,    K * sizeof(int32_t));
            D2D(s.shared_out,    w.s_shared, H * sizeof(float));
        }
        if (s.indexer_score) {
            D2D(s.indexer_q,     w.s_iq,     (size_t)c.index_n_heads * c.index_head_dim * sizeof(float));
            D2D(s.indexer_w,     w.s_iw,     (size_t)c.index_n_heads * sizeof(float));
            D2D(s.indexer_score, w.s_iindex, (size_t)T * sizeof(float));
            D2D(s.indexer_topk,  w.idx_topk, (size_t)k_sel * sizeof(int32_t));
        }
        s.seen = true;
    }
};

int main(int argc, char** argv) {
    std::string model_dir, packed_dir, ref_dir, prompt_file, nc, dump_logits;
    uint32_t n_gen = 5;
    size_t reserve_gb = 6;
    uint32_t io_threads = 4;
    for (int i = 1; i < argc; i++) {
        if      (!std::strcmp(argv[i], "--model-dir") && i + 1 < argc) model_dir  = argv[++i];
        else if (!std::strcmp(argv[i], "--packed")    && i + 1 < argc) packed_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--ref")       && i + 1 < argc) ref_dir    = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens")    && i + 1 < argc) n_gen = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reserve-gb")&& i + 1 < argc) reserve_gb = (size_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--io-threads")&& i + 1 < argc) io_threads = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--prompt")    && i + 1 < argc) prompt_file = argv[++i];
        else if (!std::strcmp(argv[i], "--tol")       && i + 1 < argc) TOL_REL   = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--tol-index") && i + 1 < argc) TOL_INDEX = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--nc")        && i + 1 < argc) nc = argv[++i];
        else if (!std::strcmp(argv[i], "--require-indexer")) g_require_indexer = 1;
        else if (!std::strcmp(argv[i], "--layers")    && i + 1 < argc) {
            g_sampled.clear();
            const std::string v = argv[++i];
            for (size_t p = 0; p < v.size(); ) {
                if (!isdigit((unsigned char)v[p])) { p++; continue; }
                uint32_t x = 0;
                while (p < v.size() && isdigit((unsigned char)v[p])) { x = x * 10 + (v[p] - '0'); p++; }
                g_sampled.push_back(x);
            }
            std::sort(g_sampled.begin(), g_sampled.end());
        }
        else if (!std::strcmp(argv[i], "--gate-depth") && i + 1 < argc) g_gate_depth = (uint32_t)atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--top5-set")) g_top5_set_only = 1;
        else if (!std::strcmp(argv[i], "--jitter-rel") && i + 1 < argc) g_jitter_rel = (float)atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump_logits = argv[++i];
        else if (!std::strcmp(argv[i], "--prefill")   && i + 1 < argc) {
            const std::string m = argv[++i];
            if      (m == "layer")    glm::g_prefill_layer_major = true;
            else if (m == "position") glm::g_prefill_layer_major = false;
            else { fprintf(stderr, "--prefill takes layer|position\n"); return 1; }
        }
        else if (!std::strcmp(argv[i], "--report")) g_report_only = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (model_dir.empty() || packed_dir.empty() || ref_dir.empty() || g_sampled.empty()) {
        fprintf(stderr, "usage: %s --model-dir DIR --packed DIR --ref DIR "
                        "[--tokens N] [--layers 0,2,3,40,77] [--prompt FILE] "
                        "[--tol R] [--tol-index R] [--require-indexer] "
                        "[--nc weightsproj2|topk_minus1] "
                        "[--reserve-gb N] [--io-threads N] [--report]\n", argv[0]);
        return 1;
    }
    // A negative control that does not separate must be visible as such, so an
    // --nc run reports and never gates; the separation is read off the printed
    // figures against the baseline run's.
    if (!nc.empty()) {
        if (nc != "weightsproj2" && nc != "topk_minus1" && nc != "jitter") {
            fprintf(stderr, "unknown --nc %s\n", nc.c_str()); return 1;
        }
        printf("NEGATIVE CONTROL: %s — this run is EXPECTED to disagree; "
               "figures are reported, the gate is not applied\n", nc.c_str());
        g_report_only = 1;
    }

    glm::Config c;
    if (!glm::load_config(model_dir, &c)) return 1;

    if (prompt_file.empty()) prompt_file = ref_dir + "/prompt.txt";
    if (!require_fixtures(ref_dir, c, n_gen, prompt_file)) return 1;

    // --- prompt: the same 29 real token ids ref_glm_chain.py was driven with ---
    std::vector<uint32_t> ids;
    {
        FILE* f = fopen(prompt_file.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "test_glm_chain: cannot open the oracle prompt %s\n",
                    prompt_file.c_str());
            return 1;
        }
        std::string txt; char buf[4096]; size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) txt.append(buf, n);
        fclose(f);
        for (size_t i = 0; i < txt.size(); ) {
            if (!isdigit((unsigned char)txt[i])) { i++; continue; }
            uint32_t v = 0;
            while (i < txt.size() && isdigit((unsigned char)txt[i])) { v = v * 10 + (txt[i] - '0'); i++; }
            ids.push_back(v);
        }
    }
    if (ids.empty()) { fprintf(stderr, "prompt.txt held no token ids\n"); return 1; }
    printf("prompt: %zu tokens, first=%u last=%u\n", ids.size(), ids.front(), ids.back());
    const uint32_t t0_pos = (uint32_t)ids.size() - 1;   // the step that emits token 0
    c.max_seq = (uint32_t)ids.size() + n_gen + 8;

    // NEGATIVE CONTROL 2, applied before any buffer is sized so that every
    // consumer of index_topk agrees with it: one key fewer than the reference
    // selects. At T <= index_topk the correct k is min(index_topk, T) == T and
    // the drop-mask is uniformly false; with index_topk = T-1 the lowest-scoring
    // key of the observed position is masked out of attention on EVERY owning
    // layer and propagates to its consumers. It is the smallest possible change
    // to the selection, and unlike the same control at 4096 (Tasks 5/6, where it
    // only swaps two keys either side of the rank-2048 boundary and is
    // undetectable) here it removes a key that the reference kept.
    if (nc == "topk_minus1") {
        if (c.index_topk >= (uint32_t)ids.size()) {
            c.index_topk = (uint32_t)ids.size() - 1;
            printf("  nc topk_minus1: index_topk %u -> %u (one key fewer at the "
                   "observed position)\n", (uint32_t)ids.size(), c.index_topk);
        } else {
            fprintf(stderr, "nc topk_minus1 needs a prompt shorter than index_topk "
                            "(%u); at this length the control is a boundary swap, "
                            "not a drop\n", c.index_topk);
            return 1;
        }
    }

    st::ModelDir M;
    if (!st::open(&M, model_dir, /*allow_missing_shards=*/false)) {
        fprintf(stderr, "st::open failed\n"); return 1;
    }

    // --- all 78 layers' non-expert weights, once ---
    printf("loading %u layers of non-expert weights...\n", c.n_layers);
    std::vector<glm::LayerWeights> W(c.n_layers);
    for (uint32_t l = 0; l < c.n_layers; l++) {
        if (!glm::load_layer(&M, c, l, &W[l])) { fprintf(stderr, "load_layer %u failed\n", l); return 1; }
        if (l % 10 == 0 || l + 1 == c.n_layers) {
            size_t fb = 0, tb = 0; cudaMemGetInfo(&fb, &tb);
            printf("  layer %2u loaded, VRAM free %.2f / %.2f GB\n",
                   l, (double)fb / 1e9, (double)tb / 1e9);
            fflush(stdout);
        }
    }

    // NEGATIVE CONTROL 1: weights_proj x 2 on every owning layer. Top-k is
    // scale-invariant, so the selected set, the mask and every output tensor
    // come out bit-identical to the correct run — the ONLY row that can see this
    // is the index score. Task 5 measured 188x separation on it and 1.00x on the
    // outputs; Task 6 measured 251x / 1.00x. Applied on the device copy after
    // load, so no production file knows this control exists.
    if (nc == "weightsproj2") {
        uint32_t touched = 0;
        const size_t n = (size_t)c.index_n_heads * c.hidden;
        std::vector<float> h(n);
        for (uint32_t l = 0; l < c.n_layers; l++) {
            if (!c.indexer_owner[l]) continue;
            CUDA_OK(cudaMemcpy(h.data(), W[l].idx_weights_proj, n * sizeof(float),
                               cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; i++) h[i] *= 2.0f;
            CUDA_OK(cudaMemcpy(W[l].idx_weights_proj, h.data(), n * sizeof(float),
                               cudaMemcpyHostToDevice));
            touched++;
        }
        printf("  nc weightsproj2: scaled weights_proj by 2 on %u owning layers\n", touched);
    }

    // --- embed / final norm / lm_head ---
    glm::ChainWeights cw;
    if (!glm::upload_raw<uint16_t>(&M, "model.embed_tokens.weight", &cw.embed,
                                   (size_t)c.vocab * c.hidden, "BF16")) return 1;
    if (!glm::upload_raw<uint16_t>(&M, "model.norm.weight", &cw.final_norm, c.hidden, "BF16")) return 1;
    if (!glm::upload_raw<uint16_t>(&M, "lm_head.weight", &cw.lm_head,
                                   (size_t)c.vocab * c.hidden, "BF16")) return 1;
    CUDA_OK(cudaMalloc(&cw.d_hidden, c.hidden * sizeof(float)));
    CUDA_OK(cudaMalloc(&cw.d_hnorm,  c.hidden * sizeof(float)));
    float* d_logits = nullptr;
    CUDA_OK(cudaMalloc(&d_logits, (size_t)c.vocab * sizeof(float)));

    // --- expert cache, sized from what is actually left ---
    const glm::ExpertLayout lay = glm::ExpertLayout::from(c);
    size_t free_b = 0, total_b = 0;
    CUDA_OK(cudaMemGetInfo(&free_b, &total_b));
    const size_t reserve = reserve_gb << 30;
    if (free_b <= reserve) {
        fprintf(stderr, "only %.2f GB free, below the %zu GB reserve\n",
                (double)free_b / 1e9, reserve_gb);
        return 1;
    }
    const uint32_t cap = (uint32_t)((free_b - reserve) / lay.total);
    fprintf(stderr, "expert cache capacity: %u experts (%.1f GB of %.1f GB free, "
                    "%zu GB reserved)\n",
            cap, (double)cap * lay.total / 1e9, (double)free_b / 1e9, reserve_gb);
    glm::ExpertCache cache;
    if (!cache.init(cap, packed_dir, c.dense_first, c.n_layers - c.dense_first, io_threads)) {
        fprintf(stderr, "ExpertCache::init failed — this is a real finding, not "
                        "something to work around by shrinking the pool\n");
        return 1;
    }
    {
        size_t fb = 0, tb = 0; CUDA_OK(cudaMemGetInfo(&fb, &tb));
        printf("VRAM after all allocations: %.2f GB free of %.2f GB "
               "(peak resident %.2f GB)\n",
               (double)fb / 1e9, (double)tb / 1e9, (double)(tb - fb) / 1e9);
    }

    // --- tracing ---
    glm::Trace tr;
    CUDA_OK(cudaMalloc(&tr.post_input_norm,  c.hidden * sizeof(float)));
    CUDA_OK(cudaMalloc(&tr.attn_out,         c.hidden * sizeof(float)));
    CUDA_OK(cudaMalloc(&tr.post_attn_hidden, c.hidden * sizeof(float)));
    glm::g_trace = &tr;

    SubstepHook hook;
    hook.c = c; hook.t0_pos = t0_pos; hook.trace = &tr;
    hook.T = t0_pos + 1;
    hook.k_sel = (c.index_topk < hook.T) ? c.index_topk : hook.T;   // M:262
    hook.want_indexer = (g_have_indexer != 0);
    if (nc == "jitter") {
        hook.jitter_layer = 0;
        hook.jitter_rel   = g_jitter_rel;
        printf("  nc jitter: +%.1e relative on one element of EVERY position's "
               "residual stream after layer 0\n", (double)g_jitter_rel);
    }
    for (uint32_t l : g_sampled) { if (l < c.n_layers) hook.add_layer(l); }
    printf("sampled layers:");
    for (uint32_t l : g_sampled)
        printf(" %u%s", l, (l < c.n_layers && c.indexer_owner[l]) ? "(owner)" : "");
    printf("   indexer taps: %s   T=%u k_sel=%u index_topk=%u\n",
           g_have_indexer ? "compared" : "ABSENT from this fixture dir — SKIPPED",
           hook.T, hook.k_sel, c.index_topk);

    // --- run: prompt, then greedy generation ---
    std::vector<std::vector<float>> got_logits;
    std::vector<int32_t>            got_top1;
    for (uint32_t t = 0; t < n_gen; t++) {
        const uint32_t* toks; uint32_t nt; uint32_t pos0;
        uint32_t next = 0;
        if (t == 0) { toks = ids.data(); nt = (uint32_t)ids.size(); pos0 = 0; }
        else        { next = (uint32_t)got_top1.back(); toks = &next; nt = 1;
                      pos0 = (uint32_t)ids.size() + t - 1; }
        glm::run_chain(c, W, cw, lay, cache, toks, nt, pos0, d_logits, &hook, 0);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());
        got_logits.push_back(download(d_logits, c.vocab));
        const std::vector<float>& L = got_logits.back();
        int32_t am = 0;
        for (size_t i = 1; i < L.size(); i++) if (L[i] > L[am]) am = (int32_t)i;
        got_top1.push_back(am);
        cache.print_stats("chain");
        printf("token %u: pos=%u argmax=%d\n", t, pos0 + nt - 1, am);
        fflush(stdout);
    }

    // Raw logits, for CUDA-vs-CUDA A/B runs that have no reference to compare
    // against (e.g. the two prefill loop orders, which must agree bit for bit).
    if (!dump_logits.empty()) {
        FILE* f = fopen(dump_logits.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", dump_logits.c_str()); return 1; }
        for (uint32_t t = 0; t < n_gen; t++)
            fwrite(got_logits[t].data(), sizeof(float), got_logits[t].size(), f);
        fclose(f);
        printf("wrote %u x %u logits to %s\n", n_gen, c.vocab, dump_logits.c_str());
    }

    // --- per-substep comparison at the sampled layers, t = 0 ---
    printf("\n=== per-substep, t=0 (pos %u) ===\n", t0_pos);
    for (uint32_t l : g_sampled) {
        if (l >= c.n_layers) continue;
        LayerSnap& s = hook.snap[l];
        if (!s.seen) { printf("layer %u: hook never fired — BUG in the test\n", l); g_fail = 1; continue; }
        const std::string P = ref_dir + "/ref_chain_";
        const std::string S = "_L" + std::to_string(l) + "_t0.npy";
        const bool moe = (l >= c.dense_first);
        printf("-- layer %u (%s, %u substeps) --\n", l, moe ? "MoE" : "dense", moe ? 10u : 6u);

        const bool gate_layer = (l <= g_gate_depth);
        if (!gate_layer)
            printf("   (layer %u is past --gate-depth %u: compared and printed, "
                   "NOT gated — see g_gate_depth)\n", l, g_gate_depth);
        auto cmp = [&](const char* name, const float* dev, size_t n) {
            compare(std::string(name) + " L" + std::to_string(l),
                    load_npy_f32(P + name + S), download(dev, n), TOL_REL, gate_layer);
        };
        cmp("post_input_norm",  s.post_input_norm,  c.hidden);
        cmp("attn_out",         s.attn_out,         c.hidden);
        cmp("post_attn_hidden", s.post_attn_hidden, c.hidden);
        cmp("post_post_norm",   s.post_post_norm,   c.hidden);
        if (moe) {
            cmp("router_logits", s.router_logits, c.n_routed_experts);
            // topk_idx: exact SET match. torch/np.argpartition give no order
            // guarantee, and this path emits descending-by-biased-score, so slot
            // order carries no information; the SET is the whole claim.
            const size_t K = c.n_experts_per_tok;
            std::vector<int32_t> ri = load_npy_i32(P + "topk_idx" + S);
            std::vector<int32_t> gi(K);
            CUDA_OK(cudaMemcpy(gi.data(), s.topk_idx, K * sizeof(int32_t), cudaMemcpyDeviceToHost));
            std::vector<int32_t> rs = ri, gs = gi;
            std::sort(rs.begin(), rs.end()); std::sort(gs.begin(), gs.end());
            const bool same = (rs == gs);
            printf("[%-34s] ref={", ("topk_idx L" + std::to_string(l)).c_str());
            for (size_t i = 0; i < K; i++) printf("%s%d", i ? "," : "", rs[i]);
            printf("} got={");
            for (size_t i = 0; i < K; i++) printf("%s%d", i ? "," : "", gs[i]);
            printf("}  %s\n", !gate_layer ? "(reported, not gated)"
                                            : (g_report_only ? "-" : (same ? "OK" : "FAIL")));
            if (!same && !g_report_only && gate_layer) g_fail = 1;
            // topk_w must be compared in the REFERENCE's slot order.
            std::vector<float> rw = load_npy_f32(P + "topk_w" + S);
            std::vector<float> gw(K);
            CUDA_OK(cudaMemcpy(gw.data(), s.topk_w, K * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<float> aligned(K, 0.0f);
            for (size_t i = 0; i < K; i++)
                for (size_t j = 0; j < K; j++)
                    if (gi[j] == ri[i]) { aligned[i] = gw[j]; break; }
            compare("topk_w L" + std::to_string(l), rw, aligned, TOL_REL, gate_layer);
            cmp("shared_out", s.shared_out, c.hidden);
        }
        cmp("moe_out",       s.moe_out,       c.hidden);
        cmp("output_hidden", s.output_hidden, c.hidden);

        // ---- indexer, owning layers only ----
        if (s.indexer_score) {
            // q and w are diagnostics: they localise a disagreement in the index
            // score to one of its two factors. Gating them would add nothing the
            // score row does not already carry, and Task 6 reports them the same
            // way.
            compare("indexer_q L" + std::to_string(l),
                    load_npy_f32(P + "indexer_q" + S),
                    download(s.indexer_q, (size_t)c.index_n_heads * c.index_head_dim),
                    TOL_INDEX, /*gated=*/false);
            compare("indexer_w L" + std::to_string(l),
                    load_npy_f32(P + "indexer_w" + S),
                    download(s.indexer_w, c.index_n_heads), TOL_INDEX, /*gated=*/false);
            // THE SCALE GATE. See TOL_INDEX above: this row, and only this row,
            // can see a monotone rescaling of the index score.
            // Gated only within --gate-depth: the index score is computed from
            // this layer's own inputs, so past the depth where the residual
            // stream itself has diverged it carries the same amplified spread
            // the substeps do (measured 1.03e-01 at layer 50 on a 256-token
            // prompt, against a 6.96e-03 floor at the layers where the stream
            // still agrees).
            compare("indexer_index_score L" + std::to_string(l),
                    load_npy_f32(P + "indexer_index_score" + S),
                    download(s.indexer_score, hook.T), TOL_INDEX, gate_layer);
            // The SET is reported and NEVER gated (Task 5 concern 1, Task 6
            // concern 1): windlass keeps the indexer's q in fp32 and its key
            // cache in bf16, this reference keeps both in fp32, so the two
            // legitimately disagree about keys sitting inside the score noise
            // floor. Overlap is a diagnostic, not a claim.
            std::vector<int32_t> rsel = load_npy_i32(P + "indexer_topk" + S);
            std::vector<int32_t> gsel(hook.k_sel);
            CUDA_OK(cudaMemcpy(gsel.data(), s.indexer_topk,
                               (size_t)hook.k_sel * sizeof(int32_t), cudaMemcpyDeviceToHost));
            std::set<int32_t> R(rsel.begin(), rsel.end()), G(gsel.begin(), gsel.end());
            size_t shared_n = 0;
            for (int32_t v : G) if (R.count(v)) shared_n++;
            printf("[%-34s] %zu of %zu ref keys shared (%.4f%%), ref k=%zu got k=%zu"
                   "  (reported, NEVER gated)\n",
                   ("indexer_topk L" + std::to_string(l)).c_str(),
                   shared_n, R.size(), 100.0 * (double)shared_n / (double)R.size(),
                   R.size(), G.size());
        } else if (c.indexer_owner[l] && g_have_indexer) {
            printf("layer %u is an indexer owner but no indexer tap was captured "
                   "— BUG in the test\n", l);
            g_fail = 1;
        }
        // ref_chain_hidden_L*_t0.npy is byte-identical to
        // ref_chain_output_hidden_L*_t0.npy (ref_glm_chain.py saves the same
        // array under both names). Compared once, on purpose — a second
        // assertion on the same bytes is not a second piece of evidence.
    }

    // --- logits and top-5, every generated token ---
    printf("\n=== logits / top-5 ===\n");
    int top5_all_ok = 1;
    for (uint32_t t = 0; t < n_gen; t++) {
        // The logits are the output of the deepest layer there is, so they
        // carry the full amplification; --top5-set says so and reports them.
        compare("logits t" + std::to_string(t),
                load_npy_f32(ref_dir + "/ref_chain_logits_" + std::to_string(t) + ".npy"),
                got_logits[t], TOL_REL, /*gated=*/!g_top5_set_only);
        std::vector<int32_t> rtop = load_npy_i32(ref_dir + "/ref_chain_topk_" +
                                                 std::to_string(t) + ".npy");
        const size_t k5 = rtop.size();
        // our top-5: partial selection, descending by logit, ties by lower index
        const std::vector<float>& L = got_logits[t];
        std::vector<int32_t> ord(L.size());
        for (size_t i = 0; i < L.size(); i++) ord[i] = (int32_t)i;
        std::partial_sort(ord.begin(), ord.begin() + k5, ord.end(),
                          [&](int32_t a, int32_t b) {
                              if (L[a] != L[b]) return L[a] > L[b];
                              return a < b;
                          });
        std::vector<int32_t> gtop(ord.begin(), ord.begin() + k5);
        const bool same = (gtop == rtop);
        std::vector<int32_t> rs5 = rtop, gs5 = gtop;
        std::sort(rs5.begin(), rs5.end()); std::sort(gs5.begin(), gs5.end());
        const bool same_set = (rs5 == gs5);
        const bool same_argmax = (!rtop.empty() && !gtop.empty() && rtop[0] == gtop[0]);
        printf("[%-34s] ref={", ("top5 t" + std::to_string(t)).c_str());
        for (size_t i = 0; i < k5; i++) printf("%s%d", i ? "," : "", rtop[i]);
        printf("} got={");
        for (size_t i = 0; i < k5; i++) printf("%s%d", i ? "," : "", gtop[i]);
        printf("}  order:%s set:%s argmax:%s\n",
               same ? "MATCH" : "MISMATCH",
               same_set ? "MATCH" : "MISMATCH",
               same_argmax ? "MATCH" : "MISMATCH");
        // Under --top5-set the ORDER is reported and the SET and the argmax are
        // gated. Two logits closer together than the measured chaotic spread can
        // swap rank without either implementation being wrong; which token is
        // generated cannot.
        const bool ok5 = g_top5_set_only ? (same_set && same_argmax) : same;
        if (!ok5) { top5_all_ok = 0; if (!g_report_only) g_fail = 1; }
        if (!same) top5_all_ok = 0;
    }

    printf("\nworst rel over every compared tensor: %.4e  (%s)\n",
           g_worst_rel, g_worst_tag.c_str());
    if (g_worst_report > 0.0)
        printf("worst rel over the reported-not-gated rows: %.4e  (%s)\n",
               g_worst_report, g_worst_report_tag.c_str());
    printf("gate TOL_REL = %.1e", TOL_REL);
    if (g_have_indexer) printf(", TOL_INDEX = %.1e", TOL_INDEX);
    printf("\n");
    cache.print_stats("final");
    {
        size_t fb = 0, tb = 0; CUDA_OK(cudaMemGetInfo(&fb, &tb));
        printf("peak VRAM resident: %.2f GB of %.2f GB\n",
               (double)(tb - fb) / 1e9, (double)tb / 1e9);
    }

    glm::g_trace = nullptr;
    if (g_report_only) { printf("\nREPORT ONLY — no gate applied\n"); return 0; }
    if (g_fail)        { printf("\nFAIL\n"); return 1; }
    if (g_top5_set_only)
        printf("\nALL GATED SUBSTEPS OK (gate-depth %u), top-5 SET and argmax "
               "exact at all %u tokens%s\n",
               g_gate_depth, n_gen, top5_all_ok ? ", order too" : " (order differs)");
    else
        printf("\nALL SUBSTEPS OK, top-5 exact at all %u tokens%s\n",
               n_gen, top5_all_ok ? "" : " (?)");
    return 0;
}
