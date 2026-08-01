/*
 * test_glm_indexer.cu — the DSA indexer forward and top-k, in isolation (Task 3).
 *
 * Build: make test_glm_indexer
 * Run:   python3 tools/ref_glm_indexer.py --out /tmp/idxfix --scale 1.0
 *        ./test_glm_indexer --fixtures /tmp/idxfix
 *
 * NO CHECKPOINT AND NO ORACLE FIXTURES. Random inputs at the real shapes
 * (q_lora 2048, hidden 6144, 32 heads x 128 dims, T keys), against
 * tools/ref_glm_indexer.py's independent numpy reimplementation of the spec
 * Task 1 extracted from transformers. The point of running this before any real
 * weight is that every one of the four traps below produces plausible numbers:
 * nothing about the output looks wrong when it is wrong.
 *
 * THE TOLERANCE IS NOT INHERITED. The 1e-2 bars elsewhere in this project were
 * calibrated against a *bfloat16* oracle, where the cast, not the arithmetic,
 * set the floor. Both sides here are float32 running the same maths and
 * differing only in summation order, so the floor is ~1e-7. TOL_REL below is
 * derived from this test's own --report sweep; see the header comment on it.
 *
 * THE METRIC IS rel_to_scale = max_i |ref_i - got_i| / max_i |ref_i|, the same
 * whole-tensor figure test_glm_chain.cu gates on, for the same reason: the
 * elementwise form is unbounded wherever the reference cancels to near zero,
 * and the index scores do exactly that (the relu makes a large fraction of them
 * exactly 0.0).
 *
 * NEGATIVE CONTROLS ARE THE POINT. A gate that has never been shown to fail is
 * not evidence. Two independent forms are used:
 *   1. In-test, every run: four defective REFERENCES (ref_glm_indexer.py) are
 *      compared against the same CUDA output, and the separation is printed.
 *      Cheap, permanent, and it ships no defect branch in glm_kernels.cuh.
 *   2. Once, by hand: the same four defects injected into glm_kernels.cuh,
 *      rebuilt and run (numbers on TOL_REL below). This is the form that
 *      actually proves the gate would stop a wrong kernel, and it is NOT
 *      redundant with (1): a reference-side defect also perturbs the reference's
 *      bf16 key cache, which inflates the apparent separation. The eps trap
 *      looks 54x separated by form (1) at scale=1.0 and is in fact NOT CAUGHT by
 *      form (2) at that scale. Rerun form (2) if these kernels change.
 *
 * TOP-K IS COMPARED AS A SET, NEVER INDEX-FOR-INDEX. torch's `sorted=True`
 * default is irrelevant -- the indices only ever feed a bool mask. On these
 * random inputs the combined index scores are continuous and ties are
 * vanishingly unlikely, which is what makes an exact set comparison meaningful
 * here. On REAL weights it will not be: after the relu a large fraction of
 * scores are exactly 0.0, so above index_topk keys the selected set is
 * genuinely under-determined and no implementation can match torch's choice.
 * Task 5 must therefore compare attention OUTPUTS, or restrict the set
 * comparison to strictly-positive scores. This test asserts on the set only
 * because it controls the inputs.
 */
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <set>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "glm_primitives.cuh"
#include "glm_kernels.cuh"

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA: %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(1); } } while (0)

// --- minimal .npy reader (f4 / i4 / u2) ------------------------------------
static std::vector<uint8_t> npy_raw(const std::string& path, std::string* dtype,
                                    std::vector<size_t>* shape) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "test_glm_indexer: cannot open fixture %s\n"
                        "  generate it with: python3 tools/ref_glm_indexer.py --out DIR\n",
                path.c_str());
        std::exit(1);
    }
    char magic[6]; size_t rd = fread(magic, 1, 6, f);
    if (rd != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "%s: not a .npy\n", path.c_str()); std::exit(1);
    }
    uint8_t major, minor; rd = fread(&major, 1, 1, f); rd = fread(&minor, 1, 1, f);
    (void)minor;
    uint32_t hlen = 0;
    if (major == 1) { uint16_t h; rd = fread(&h, 2, 1, f); hlen = h; }
    else            { rd = fread(&hlen, 4, 1, f); }
    std::string hdr(hlen, '\0');
    if (fread(&hdr[0], 1, hlen, f) != hlen) {
        fprintf(stderr, "%s: short header\n", path.c_str()); std::exit(1);
    }
    size_t dp = hdr.find("descr"); dp = hdr.find(':', dp);
    size_t q1 = hdr.find_first_of("'\"", dp), q2 = hdr.find_first_of("'\"", q1 + 1);
    *dtype = hdr.substr(q1 + 1, q2 - q1 - 1);
    size_t esz = 0;
    if      (*dtype == "<f4") esz = 4;
    else if (*dtype == "<i4") esz = 4;
    else if (*dtype == "<u2") esz = 2;
    else { fprintf(stderr, "%s: unsupported dtype %s\n", path.c_str(), dtype->c_str()); std::exit(1); }
    size_t sb = hdr.find('('), se = hdr.find(')');
    shape->clear(); size_t total = 1;
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

template <typename T>
static std::vector<T> npy_load(const std::string& path, const char* want) {
    std::string dt; std::vector<size_t> sh;
    std::vector<uint8_t> raw = npy_raw(path, &dt, &sh);
    if (dt != want) {
        fprintf(stderr, "%s: expected %s, got %s\n", path.c_str(), want, dt.c_str());
        std::exit(1);
    }
    std::vector<T> out(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
}

// --- comparison ------------------------------------------------------------
static int    g_fail = 0;
static int    g_report_only = 0;
static double g_worst = 0.0;
static std::string g_worst_tag = "(none)";

// Derived, not inherited. Measured with --report on the scale=1.0 fixture set
// (T=1024, index_topk=512, seed 20260801), stages 1 and 2 only -- both sides
// fp32, reading bit-identical keys, so the only difference is summation order:
//
//     k (post-LN+RoPE), all 1024 positions   5.28e-07
//     q (post-RoPE)                          5.01e-07
//     head weights w                         3.43e-07
//     index_score                            5.83e-07   <- worst
//
// This gate is 17x above that worst figure.
//
// IT CAN FAIL, and the authoritative evidence for that is not the in-test
// controls below but four defects injected into GLM_KERNELS.CUH ITSELF, built
// and run on aeronav-r6000 (task-3-report.md). Each produced exit 1:
//
//   RMSNorm for LayerNorm          k        1.05e-01   1.1e+04x this gate
//   RoPE on the trailing 64        k        1.64e+00   1.6e+05x
//   relu after the combination     index    1.00e+00   1.0e+05x
//   LayerNorm eps 1e-5 for 1e-6    k        2.08e-02   2.1e+03x  -- scale=0.01 ONLY
//
// The eps row carries a caveat that is a finding in its own right: on the
// scale=1.0 fixture the SAME injection produced 2.51e-06, INSIDE this gate, and
// the test passed. The effect of eps scales as eps/var(wk(x)), which is ~2e-6
// when the hidden state is O(1), so no test on these shapes catches it there.
// It is caught only on an eps-sensitive fixture (--scale 0.01, var(wk(x)) ~
// 2.4e-04). The in-test eps control reports a healthier 54x at scale=1.0, but
// that number is inflated by bf16 cache ties (the defective reference rounds
// ~60 key words the other way) and should not be trusted as the catch.
// Do NOT tighten this gate to chase the eps trap: the baseline is 5.83e-07 and
// a gate that only just clears 2.51e-06 is a gate on noise.
static const double TOL_REL = 1e-5;

// Stage 3's separate, looser gate, and stage 1's bf16 cache comparison: both are
// floored by round-to-nearest ties in the bf16 key cache, not by the arithmetic.
// Measured 1.64e-03 (decoded cache) and 4.10e-04 (end-to-end index_score); this
// sits 6.1x and 24x above them. A tighter bar here would be a bar on a coin flip
// -- 0.048% of the cached key words round the other way because the fp32 value
// being rounded differs by ~1e-6, and the same k[t] then feeds all 32 heads.
static const double TOL_E2E = 1e-2;

static double compare(const std::string& tag, const std::vector<float>& ref,
                      const std::vector<float>& got, bool gate, double tol = TOL_REL) {
    if (ref.size() != got.size()) {
        printf("[%-26s] SIZE MISMATCH ref=%zu got=%zu\n", tag.c_str(), ref.size(), got.size());
        g_fail = 1; return 1e30;
    }
    double max_abs = 0, scale = 0; size_t worst = 0;
    for (size_t i = 0; i < ref.size(); i++) scale = std::fmax(scale, std::fabs((double)ref[i]));
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = std::fabs((double)ref[i] - (double)got[i]);
        if (d > max_abs) { max_abs = d; worst = i; }
    }
    const double rel = max_abs / (scale + 1e-300);
    const bool ok = (rel < tol);
    if (gate && tol == TOL_REL && rel > g_worst) { g_worst = rel; g_worst_tag = tag; }
    printf("[%-26s] rel=%.4e (tol %.1e) | max_abs=%.4e max|ref|=%.4e | worst@%zu "
           "ref=%.6g got=%.6g  %s\n",
           tag.c_str(), rel, tol, max_abs, scale, worst, ref[worst], got[worst],
           g_report_only ? "-" : (gate ? (ok ? "OK" : "FAIL") : "(control)"));
    if (gate && !ok && !g_report_only) g_fail = 1;
    return rel;
}

static std::vector<float> dl_f(const float* d, size_t n) {
    std::vector<float> h(n);
    CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

static size_t overlap(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    std::set<int32_t> sa(a.begin(), a.end());
    size_t n = 0;
    for (int32_t v : b) if (sa.count(v)) n++;
    return n;
}

int main(int argc, char** argv) {
    std::string fx;
    for (int i = 1; i < argc; i++) {
        if      (!std::strcmp(argv[i], "--fixtures") && i + 1 < argc) fx = argv[++i];
        else if (!std::strcmp(argv[i], "--report")) g_report_only = 1;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (fx.empty()) {
        fprintf(stderr, "usage: %s --fixtures DIR [--report]\n"
                        "  DIR is produced by tools/ref_glm_indexer.py\n", argv[0]);
        return 1;
    }
    const std::string P = fx + "/";

    // --- meta ---------------------------------------------------------------
    uint32_t H = 0, D = 0, ROT = 0, HID = 0, QL = 0, T = 0, POS = 0, TOPK = 0;
    double scale = 1.0;
    double sep_pred[4] = {0, 0, 0, 0};       // rmsnorm, eps, rope_trailing, relu_after
    {
        FILE* f = fopen((P + "meta.txt").c_str(), "rb");
        if (!f) { fprintf(stderr, "test_glm_indexer: cannot open %smeta.txt — run "
                                  "tools/ref_glm_indexer.py --out %s first\n",
                          P.c_str(), fx.c_str()); return 1; }
        char key[64]; double val;
        while (fscanf(f, "%63s %lf", key, &val) == 2) {
            if      (!std::strcmp(key, "H"))      H    = (uint32_t)val;
            else if (!std::strcmp(key, "D"))      D    = (uint32_t)val;
            else if (!std::strcmp(key, "rot"))    ROT  = (uint32_t)val;
            else if (!std::strcmp(key, "hidden")) HID  = (uint32_t)val;
            else if (!std::strcmp(key, "q_lora")) QL   = (uint32_t)val;
            else if (!std::strcmp(key, "T"))      T    = (uint32_t)val;
            else if (!std::strcmp(key, "pos"))    POS  = (uint32_t)val;
            else if (!std::strcmp(key, "topk"))   TOPK = (uint32_t)val;
            else if (!std::strcmp(key, "scale"))  scale = val;
            else if (!std::strcmp(key, "sep_rmsnorm"))       sep_pred[0] = val;
            else if (!std::strcmp(key, "sep_eps"))           sep_pred[1] = val;
            else if (!std::strcmp(key, "sep_rope_trailing")) sep_pred[2] = val;
            else if (!std::strcmp(key, "sep_relu_after"))    sep_pred[3] = val;
        }
        fclose(f);
    }
    if (!H || !D || !ROT || !HID || !QL || !T) {
        fprintf(stderr, "meta.txt incomplete\n"); return 1;
    }
    printf("fixtures %s: H=%u D=%u rot=%u hidden=%u q_lora=%u T=%u pos=%u topk=%u scale=%g\n",
           fx.c_str(), H, D, ROT, HID, QL, T, POS, TOPK, scale);

    // --- inputs -------------------------------------------------------------
    auto u16 = [&](const char* n) { return npy_load<uint16_t>(P + n + ".npy", "<u2"); };
    auto f32 = [&](const char* n) { return npy_load<float>(P + n + ".npy", "<f4"); };
    auto i32 = [&](const char* n) { return npy_load<int32_t>(P + n + ".npy", "<i4"); };

    std::vector<uint16_t> h_wq_b = u16("wq_b"), h_wk = u16("wk"),
                          h_knw = u16("k_norm_w"), h_knb = u16("k_norm_b");
    std::vector<float> h_wproj = f32("weights_proj"), h_qres = f32("q_resid"),
                       h_x = f32("x_all"), h_cos = f32("cos"), h_sin = f32("sin");

    if (h_wq_b.size() != (size_t)H * D * QL) { fprintf(stderr, "wq_b size\n"); return 1; }
    if (h_x.size() != (size_t)T * HID)       { fprintf(stderr, "x_all size\n"); return 1; }
    if (h_cos.size() != (size_t)T * (ROT / 2)) { fprintf(stderr, "cos size\n"); return 1; }

    // --- device -------------------------------------------------------------
    uint16_t *d_wq_b, *d_wk, *d_knw, *d_knb, *d_cache;
    float *d_wproj, *d_qres, *d_x, *d_cos, *d_sin;
    float *s_q, *s_k, *s_w, *s_scores, *s_index;
    int32_t* d_idx; uint32_t* d_n;
    auto up16 = [&](uint16_t** p, const std::vector<uint16_t>& v) {
        CUDA_OK(cudaMalloc(p, v.size() * 2));
        CUDA_OK(cudaMemcpy(*p, v.data(), v.size() * 2, cudaMemcpyHostToDevice));
    };
    auto upf = [&](float** p, const std::vector<float>& v) {
        CUDA_OK(cudaMalloc(p, v.size() * 4));
        CUDA_OK(cudaMemcpy(*p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    };
    up16(&d_wq_b, h_wq_b); up16(&d_wk, h_wk); up16(&d_knw, h_knw); up16(&d_knb, h_knb);
    upf(&d_wproj, h_wproj); upf(&d_qres, h_qres); upf(&d_x, h_x);
    upf(&d_cos, h_cos); upf(&d_sin, h_sin);

    CUDA_OK(cudaMalloc(&d_cache, (size_t)T * D * sizeof(uint16_t)));
    CUDA_OK(cudaMemset(d_cache, 0, (size_t)T * D * sizeof(uint16_t)));

    CUDA_OK(cudaMalloc(&s_q, (size_t)H * D * sizeof(float)));
    CUDA_OK(cudaMalloc(&s_k, (size_t)D * sizeof(float)));
    CUDA_OK(cudaMalloc(&s_w, (size_t)H * sizeof(float)));
    CUDA_OK(cudaMalloc(&s_scores, (size_t)H * T * sizeof(float)));
    CUDA_OK(cudaMalloc(&s_index, (size_t)T * sizeof(float)));
    CUDA_OK(cudaMalloc(&d_idx, (size_t)T * sizeof(int32_t)));
    CUDA_OK(cudaMalloc(&d_n, sizeof(uint32_t)));
    CUDA_OK(cudaMemset(d_n, 0, sizeof(uint32_t)));

    // ------------------------------------------------------------------------
    // STAGE 1 — the key path, every position, exactly as a prefill would:
    // wk -> k_norm -> RoPE(t) -> bf16 cache.
    //
    // Doing only the LAST position and filling the rest of the cache with random
    // bf16 was this test's first shape. It made the k_norm controls nearly
    // blind: a wrong k_norm then moved 1 of T index scores and left the top-k
    // set completely unchanged (RMSNorm control: 2048/2048 overlap). Measured,
    // not reasoned about.
    // ------------------------------------------------------------------------
    const uint32_t HALF = ROT / 2;
    float* d_kall = nullptr;    // [T, D] fp32, the pre-cast keys, for the tight compare
    CUDA_OK(cudaMalloc(&d_kall, (size_t)T * D * sizeof(float)));
    for (uint32_t t = 0; t < T; t++) {
        glm::launch_indexer_key(d_wk, d_knw, d_knb, d_x + (size_t)t * HID,
                                d_cos + (size_t)t * HALF, d_sin + (size_t)t * HALF,
                                d_cache, s_k, D, ROT, HID, t, glm::INDEXER_LN_EPS, 0);
        CUDA_OK(cudaMemcpyAsync(d_kall + (size_t)t * D, s_k, D * sizeof(float),
                                cudaMemcpyDeviceToDevice, 0));
    }
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());

    printf("\n=== stage 1: key path, all %u positions (fp32 vs fp32) ===\n", T);
    compare("k (post-LN+RoPE)", f32("ref_k"), dl_f(d_kall, (size_t)T * D), true);
    {
        // The cache is bf16 (Task 1's precision table: transformers keeps the
        // indexer keys in bf16), so the two sides round the SAME cast the same
        // way but round DIFFERENT floats — they differ by fp32 summation order in
        // wk(x). A key sitting within that distance of a bf16 tie lands either
        // way. Reported as a decoded-float agreement, which is the figure that
        // actually propagates, plus the raw word count as a diagnostic.
        std::vector<uint16_t> ref_kb = u16("ref_k_bf16"), got_kb((size_t)T * D);
        CUDA_OK(cudaMemcpy(got_kb.data(), d_cache, (size_t)T * D * 2, cudaMemcpyDeviceToHost));
        size_t bad = 0;
        for (size_t i = 0; i < got_kb.size(); i++) if (ref_kb[i] != got_kb[i]) bad++;
        auto dec = [](const std::vector<uint16_t>& v) {
            std::vector<float> o(v.size());
            for (size_t i = 0; i < v.size(); i++) {
                const uint32_t u = (uint32_t)v[i] << 16;
                std::memcpy(&o[i], &u, 4);
            }
            return o;
        };
        printf("  (%zu of %zu bf16 words differ = %.3f%%, all at a rounding tie)\n",
               bad, got_kb.size(), 100.0 * (double)bad / (double)got_kb.size());
        compare("cache (decoded bf16)", dec(ref_kb), dec(got_kb), true, TOL_E2E);
    }

    // ------------------------------------------------------------------------
    // STAGE 2 — the query path on the REFERENCE's cache.
    //
    // Both sides then read bit-identical keys, so this comparison sees only the
    // query arithmetic and its fp32 summation order. It is the tight gate, and
    // it is what the negative controls are measured against: mixing the bf16
    // tie noise of stage 1 into them would raise the floor by three orders of
    // magnitude and blunt every control (measured: index_score agreement 4.1e-04
    // with CUDA's own cache versus 3.5e-07 with the reference's).
    // ------------------------------------------------------------------------
    uint16_t* d_refcache = nullptr;
    {
        std::vector<uint16_t> ref_kb = u16("ref_k_bf16");
        CUDA_OK(cudaMalloc(&d_refcache, ref_kb.size() * 2));
        CUDA_OK(cudaMemcpy(d_refcache, ref_kb.data(), ref_kb.size() * 2, cudaMemcpyHostToDevice));
    }
    glm::launch_indexer_query(d_wq_b, d_wproj, d_qres, d_x + (size_t)POS * HID,
                              d_cos + (size_t)POS * HALF, d_sin + (size_t)POS * HALF,
                              d_refcache, s_q, s_w, s_scores, s_index, d_idx, d_n,
                              H, D, ROT, HID, QL, T, TOPK, 0);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());

    printf("\n=== stage 2: query path on the reference's keys (fp32 vs fp32) ===\n");
    compare("q (post-RoPE)",  f32("ref_q"), dl_f(s_q, (size_t)H * D), true);
    compare("head weights w", f32("ref_w"), dl_f(s_w, H),             true);
    const std::vector<float> ref_index = f32("ref_index");
    const std::vector<float> got_index = dl_f(s_index, T);
    compare("index_score",    ref_index,    got_index,                true);

    // --- top-k, as a SET ----------------------------------------------------
    const std::vector<int32_t> ref_top = i32("ref_topk");
    uint32_t got_n = 0;
    CUDA_OK(cudaMemcpy(&got_n, d_n, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::vector<int32_t> got_top(got_n);
    CUDA_OK(cudaMemcpy(got_top.data(), d_idx, got_n * sizeof(int32_t), cudaMemcpyDeviceToHost));
    {
        const bool n_ok = (got_n == ref_top.size());
        const size_t ov = overlap(ref_top, got_top);
        std::set<int32_t> uniq(got_top.begin(), got_top.end());
        const bool ok = n_ok && ov == ref_top.size() && uniq.size() == got_n;
        printf("[%-26s] k=%u (ref %zu) overlap %zu/%zu, %zu distinct  %s\n",
               "topk set", got_n, ref_top.size(), ov, ref_top.size(), uniq.size(),
               g_report_only ? "-" : (ok ? "OK" : "FAIL"));
        if (!ok && !g_report_only) g_fail = 1;
    }

    // --- the k >= T branch (topk = min(index_topk, T), M:262) ---------------
    // At T <= index_topk topk() returns a permutation of ALL T indices -- this
    // is the mechanism behind Task 4's bit-identity gate, so it is worth its own
    // assertion rather than being inferred.
    {
        CUDA_OK(cudaMemset(d_n, 0, sizeof(uint32_t)));
        glm::indexer_topk<<<1, 256>>>(s_index, T, T + 7, d_idx, d_n);
        CUDA_OK(cudaDeviceSynchronize());
        uint32_t n = 0;
        CUDA_OK(cudaMemcpy(&n, d_n, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::vector<int32_t> v(n);
        CUDA_OK(cudaMemcpy(v.data(), d_idx, n * sizeof(int32_t), cudaMemcpyDeviceToHost));
        std::sort(v.begin(), v.end());
        bool perm = (n == T);
        for (uint32_t i = 0; perm && i < n; i++) perm = (v[i] == (int32_t)i);
        printf("[%-26s] n=%u (T=%u), permutation of [0,T)  %s\n",
               "topk with k>=T", n, T, g_report_only ? "-" : (perm ? "OK" : "FAIL"));
        if (!perm && !g_report_only) g_fail = 1;
    }

    // ------------------------------------------------------------------------
    // STAGE 3 — end to end, on CUDA's OWN cache. This is what Task 4 will run,
    // and its floor is set by stage 1's bf16 tie rounding, not by the query
    // arithmetic: one key word that rounds the other way moves that key's index
    // score by ~1e-3, because the same k[t] feeds all 32 heads and the head
    // combination sums those perturbations coherently. Gated separately and
    // loosely on purpose; a tight bar here would be a bar on a coin flip.
    // ------------------------------------------------------------------------
    glm::launch_indexer_query(d_wq_b, d_wproj, d_qres, d_x + (size_t)POS * HID,
                              d_cos + (size_t)POS * HALF, d_sin + (size_t)POS * HALF,
                              d_cache, s_q, s_w, s_scores, s_index, d_idx, d_n,
                              H, D, ROT, HID, QL, T, TOPK, 0);
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaGetLastError());
    printf("\n=== stage 3: end to end, CUDA's own bf16 cache ===\n");
    {
        const std::vector<float> e2e = dl_f(s_index, T);
        double max_abs = 0, scale = 0;
        for (uint32_t i = 0; i < T; i++) scale = std::fmax(scale, std::fabs((double)ref_index[i]));
        for (uint32_t i = 0; i < T; i++)
            max_abs = std::fmax(max_abs, std::fabs((double)ref_index[i] - (double)e2e[i]));
        const double rel = max_abs / (scale + 1e-300);
        const bool ok = (rel < TOL_E2E);
        printf("[%-26s] rel=%.4e (tol %.1e)  %s\n", "index_score end-to-end", rel, TOL_E2E,
               g_report_only ? "-" : (ok ? "OK" : "FAIL"));
        if (!ok && !g_report_only) g_fail = 1;

        uint32_t n2 = 0;
        CUDA_OK(cudaMemcpy(&n2, d_n, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::vector<int32_t> t2(n2);
        CUDA_OK(cudaMemcpy(t2.data(), d_idx, n2 * sizeof(int32_t), cudaMemcpyDeviceToHost));
        const size_t ov = overlap(ref_top, t2);
        // NOT gated on an exact set: at this noise level a key within ~1e-3 of
        // the selection boundary can legitimately fall either side. Reported
        // because the number is the whole reason Task 5 must compare attention
        // OUTPUTS rather than index lists.
        printf("[%-26s] overlap %zu/%zu with the reference's set (not gated)\n",
               "topk end-to-end", ov, ref_top.size());
    }

    // --- negative controls --------------------------------------------------
    // Same CUDA output, four defective references. Each must be separated from
    // the baseline agreement by orders of magnitude, or the gate above is
    // decoration.
    printf("\n=== negative controls (same CUDA output, defective reference) ===\n");
    static const char* DEF[4] = {"rmsnorm", "eps", "rope_trailing", "relu_after"};
    static const char* WHAT[4] = {
        "k_norm as RMSNorm, not LayerNorm",
        "LayerNorm eps 1e-5 (rms_norm_eps) not 1e-6",
        "RoPE on the trailing 64, not the leading 64",
        "relu after the head combination, not before"};
    for (int i = 0; i < 4; i++) {
        const std::string tag = std::string("index ") + DEF[i];
        const double rel = compare(tag, f32((std::string("ref_index_") + DEF[i]).c_str()),
                                   got_index, false);
        const std::vector<int32_t> dt = i32((std::string("ref_topk_") + DEF[i]).c_str());
        const size_t ov = overlap(dt, got_top);
        printf("    %-44s separation %8.1fx over TOL_REL, topk overlap %zu/%zu\n",
               WHAT[i], rel / TOL_REL, ov, dt.size());
        // A fixture can only exercise a control that actually moves the numbers
        // on it. The reference reports what it measured for its own defective
        // variant (sep_*); when that is itself inside the gate, the fixture is
        // incapable of testing this trap and saying so is the honest outcome —
        // failing here would just mean "the eps trap is unobservable when
        // var(wk(x)) is O(1)", which is true and is a finding, not a defect.
        if (sep_pred[i] > 0.0 && sep_pred[i] <= 10.0 * TOL_REL) {
            printf("    ^ NOT EXERCISED by this fixture: the reference's own defect "
                   "moves index_score by only %.2e (<= 10x TOL_REL). Run a fixture "
                   "set that makes this trap observable.\n", sep_pred[i]);
            continue;
        }
        if (rel <= TOL_REL && !g_report_only) {
            printf("    ^ NOT CAUGHT: the reference predicted a %.2e separation but the "
                   "comparison did not see it — the gate is blind to this trap\n",
                   sep_pred[i]);
            g_fail = 1;
        }
    }

    printf("\nworst baseline rel: %.4e  (%s)\ngate TOL_REL = %.1e\n",
           g_worst, g_worst_tag.c_str(), TOL_REL);

    cudaFree(d_wq_b); cudaFree(d_wk); cudaFree(d_knw); cudaFree(d_knb);
    cudaFree(d_wproj); cudaFree(d_qres); cudaFree(d_x); cudaFree(d_cos); cudaFree(d_sin);
    cudaFree(d_cache); cudaFree(d_kall); cudaFree(d_refcache);
    cudaFree(s_q); cudaFree(s_k); cudaFree(s_w);
    cudaFree(s_scores); cudaFree(s_index); cudaFree(d_idx); cudaFree(d_n);

    if (g_report_only) { printf("\nREPORT ONLY — no gate applied\n"); return 0; }
    if (g_fail)        { printf("\nFAIL\n"); return 1; }
    printf("\nALL CHECKS OK\n");
    return 0;
}
