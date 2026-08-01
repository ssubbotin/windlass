/*
 * infer_glm.cu — end-to-end inference driver for GLM-5.2 (MXFP4 experts) on a
 * single RTX PRO 6000 Blackwell.
 *
 * Everything except the routed experts is resident: 78 layers of attention +
 * norms + shared experts + router, plus embed / model.norm / lm_head. The 19,200
 * routed experts (75 MoE layers x 256) live in packed_experts_glm/layer_N.bin at
 * 20,054,016 bytes each — 385 GB, far past the card — and are served by
 * glm::ExpertCache, an LRU over one contiguous VRAM pool whose capacity is
 * derived at run time from cudaMemGetInfo minus a reserve. It is NOT hardcoded:
 * the number of experts that fit is a function of how much the resident weights
 * happened to take, and baking in a constant would silently under- or over-commit
 * the card the first time anything upstream changes size.
 *
 * Usage:
 *   ./infer_glm --model-dir DIR --packed DIR --prompt "..." --tokens N
 *               [--repeat N] [--cache-experts N] [--reserve-gb N] [--io-threads N]
 *               [--timing] [--cache-telemetry] [--no-think] [--raw]
 *               [--token-ids "id,id,..."] [--ignore-eos] [--max-seq L]
 *
 * --repeat N runs the same prompt N times in ONE process, reporting tok/s for
 * each. That is the measurement that matters: the expert cache starts empty and
 * fills across requests, so a single cold run understates steady-state
 * throughput and a single warm run overstates a cold start. Report the curve.
 *
 * Sampling is greedy only. A temperature knob would make the throughput figure
 * depend on a seed for no benefit — the gate is about tok/s, not about text
 * quality, and greedy is what Task 10's correctness chain was verified with.
 */
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>

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

static double now_s() { return glm::glm_now(); }

// ---------------------------------------------------------------------------
// Tokenizer subprocess client — same protocol as infer_qwen36.cu / infer_kimi.cu
// (`python3 kimi_tokenize.py --model-dir DIR serve`). The script is model-agnostic;
// it just hands the directory to AutoTokenizer, which picks up GLM's tokenizer.json
// and chat_template.jinja.
// ---------------------------------------------------------------------------
struct TokClient {
    pid_t pid = -1;
    FILE* to_child   = nullptr;
    FILE* from_child = nullptr;
    int   vocab_size = 0;
    int   bos_id     = -1;
    std::vector<int> eos_ids;
};

static const char* b64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string b64encode(const std::string& in) {
    std::string out; out.reserve(((in.size() + 2) / 3) * 4);
    int val = 0, valb = -6;
    for (unsigned char c : in) {
        val = (val << 8) | c; valb += 8;
        while (valb >= 0) { out.push_back(b64_chars[(val >> valb) & 0x3F]); valb -= 6; }
    }
    if (valb > -6) out.push_back(b64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (out.size() % 4) out.push_back('=');
    return out;
}
static std::string b64decode(const std::string& in) {
    std::vector<int> T(256, -1);
    for (int i = 0; i < 64; i++) T[(unsigned char)b64_chars[i]] = i;
    std::string out; int val = 0, valb = -8;
    for (unsigned char c : in) {
        if (c == '=' || T[c] == -1) continue;
        val = (val << 6) | T[c]; valb += 6;
        if (valb >= 0) { out.push_back(char((val >> valb) & 0xFF)); valb -= 8; }
    }
    return out;
}

static std::string tok_call(TokClient* tc, const std::string& cmd, bool* ok) {
    fputs(cmd.c_str(), tc->to_child); fputc('\n', tc->to_child); fflush(tc->to_child);
    char* line = nullptr; size_t cap = 0;
    ssize_t n = getline(&line, &cap, tc->from_child);
    if (n <= 0) { if (line) free(line); *ok = false; return ""; }
    while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = 0;
    std::string s(line, (size_t)n); free(line);
    if (s.size() >= 3 && s.compare(0, 3, "OK\t") == 0) { *ok = true;  return s.substr(3); }
    if (s == "OK")                                     { *ok = true;  return ""; }
    if (s.size() >= 4 && s.compare(0, 4, "ERR\t") == 0) {
        *ok = false; fprintf(stderr, "tok ERR: %s\n", s.c_str() + 4); return "";
    }
    fprintf(stderr, "tok unexpected: %s\n", s.c_str()); *ok = false; return "";
}

static bool tok_open(TokClient* tc, const std::string& model_dir) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) { perror("pipe"); return false; }
    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        dup2(in_pipe[0], 0); close(in_pipe[1]);
        dup2(out_pipe[1], 1); close(out_pipe[0]);
        close(in_pipe[0]); close(out_pipe[1]);
        const char* py = std::getenv("KIMI_PYTHON");
        if (!py || !*py) py = "python3";
        char self[4096]; ssize_t sn = readlink("/proc/self/exe", self, sizeof(self) - 1);
        std::string script;
        if (sn > 0) {
            self[sn] = 0; std::string s = self; auto sl = s.find_last_of('/');
            script = (sl == std::string::npos ? "" : s.substr(0, sl + 1)) + "kimi_tokenize.py";
        } else script = "kimi_tokenize.py";
        execlp(py, py, script.c_str(), "--model-dir", model_dir.c_str(), "serve", (char*)nullptr);
        _exit(127);
    }
    close(in_pipe[0]); close(out_pipe[1]);
    tc->pid = pid;
    tc->to_child   = fdopen(in_pipe[1],  "w");
    tc->from_child = fdopen(out_pipe[0], "r");
    char* line = nullptr; size_t cap = 0;
    if (getline(&line, &cap, tc->from_child) <= 0) { if (line) free(line); return false; }
    free(line);
    bool ok; std::string s = tok_call(tc, "info", &ok);
    if (!ok) return false;
    size_t p1 = s.find('\t'), p2 = s.find('\t', p1 + 1);
    tc->vocab_size = std::atoi(s.substr(0, p1).c_str());
    tc->bos_id     = std::atoi(s.substr(p1 + 1, p2 - p1 - 1).c_str());
    std::string eos = s.substr(p2 + 1);
    size_t i = 0;
    while (i < eos.size()) {
        size_t j = eos.find(',', i); if (j == std::string::npos) j = eos.size();
        int id = std::atoi(eos.substr(i, j - i).c_str());
        if (id >= 0) tc->eos_ids.push_back(id);
        i = j + 1;
    }
    fprintf(stderr, "tokenizer ready: vocab=%d bos=%d eos=", tc->vocab_size, tc->bos_id);
    for (size_t k = 0; k < tc->eos_ids.size(); k++)
        fprintf(stderr, "%s%d", k ? "," : "", tc->eos_ids[k]);
    fprintf(stderr, "\n");
    return true;
}

static void tok_close(TokClient* tc) {
    if (tc->to_child) { bool ok; tok_call(tc, "quit", &ok); fclose(tc->to_child); tc->to_child = nullptr; }
    if (tc->from_child) { fclose(tc->from_child); tc->from_child = nullptr; }
    if (tc->pid > 0) { int st; waitpid(tc->pid, &st, 0); tc->pid = -1; }
}

static std::vector<int> parse_csv_ids(const std::string& s) {
    std::vector<int> out; size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && !(std::isdigit((unsigned char)s[i]) || s[i] == '-')) i++;
        if (i >= s.size()) break;
        char* end; long v = std::strtol(s.c_str() + i, &end, 10);
        out.push_back((int)v); i = (size_t)(end - s.c_str());
    }
    return out;
}
static std::vector<int> tok_encode(TokClient* tc, const std::string& text) {
    bool ok; std::string s = tok_call(tc, "encode\t" + b64encode(text), &ok);
    if (!ok) return {}; return parse_csv_ids(s);
}
// One user turn through GLM's chat_template.jinja. `think` selects between the
// template's two generation prefixes: `<think>` (reasoning on, the default in
// the checkpoint) and `<think></think>` (reasoning off). For a 40-token budget
// the difference is everything — with reasoning on, all 40 tokens are internal
// monologue and nothing that looks like an answer is ever reached.
static std::vector<int> tok_chat_user(TokClient* tc, const std::string& text, bool think) {
    std::string j = "[{\"role\":\"user\",\"content\":\"";
    for (char ch : text) {
        if (ch == '"' || ch == '\\') { j += '\\'; j += ch; }
        else if (ch == '\n') j += "\\n";
        else if (ch == '\r') j += "\\r";
        else if (ch == '\t') j += "\\t";
        else if ((unsigned char)ch < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", ch); j += b; }
        else j += ch;
    }
    j += "\"}]";
    bool ok;
    std::string cmd = "chat\t" + b64encode(j) + (think ? "" : "\t0");
    std::string s = tok_call(tc, cmd, &ok);
    if (!ok) return {}; return parse_csv_ids(s);
}
static void tok_decode_reset(TokClient* tc) { bool ok; tok_call(tc, "decode_stream_reset", &ok); }
static std::string tok_decode_push(TokClient* tc, int id) {
    char buf[64]; snprintf(buf, sizeof(buf), "decode_stream_push\t%d", id);
    bool ok; std::string s = tok_call(tc, buf, &ok);
    if (!ok) return ""; return b64decode(s);
}

// ---------------------------------------------------------------------------
static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s --model-dir DIR --packed DIR (--prompt TEXT | --token-ids \"id,...\")\n"
        "          [--tokens N] [--repeat N] [--cache-experts N] [--reserve-gb N]\n"
        "          [--io-threads N]\n"
        "          [--timing] [--cache-telemetry] [--no-think] [--raw] [--ignore-eos]\n"
        "          [--max-seq L]\n", prog);
}

int main(int argc, char** argv) {
    std::string model_dir, packed_dir, prompt_text, token_ids_csv;
    uint32_t n_gen = 40, repeat = 1;
    uint32_t forced_cap = 0;          // --cache-experts; 0 = derive from free VRAM
    size_t   reserve_gb = 6;
    uint32_t io_threads = 4;          // --io-threads; 0 = fully synchronous fetch
    int      max_seq_arg = 0;
    bool     want_timing = false, want_telemetry = false;
    bool     think = true, raw = false, ignore_eos = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--model-dir"     && i + 1 < argc) model_dir     = argv[++i];
        else if (a == "--packed"        && i + 1 < argc) packed_dir    = argv[++i];
        else if (a == "--prompt"        && i + 1 < argc) prompt_text   = argv[++i];
        else if (a == "--token-ids"     && i + 1 < argc) token_ids_csv = argv[++i];
        else if (a == "--tokens"        && i + 1 < argc) n_gen         = (uint32_t)atoi(argv[++i]);
        else if (a == "--repeat"        && i + 1 < argc) repeat        = (uint32_t)atoi(argv[++i]);
        else if (a == "--cache-experts" && i + 1 < argc) forced_cap    = (uint32_t)atoi(argv[++i]);
        else if (a == "--reserve-gb"    && i + 1 < argc) reserve_gb    = (size_t)atoi(argv[++i]);
        else if (a == "--io-threads"    && i + 1 < argc) io_threads    = (uint32_t)atoi(argv[++i]);
        else if (a == "--max-seq"       && i + 1 < argc) max_seq_arg   = atoi(argv[++i]);
        else if (a == "--timing")          want_timing    = true;
        else if (a == "--cache-telemetry") want_telemetry = true;
        else if (a == "--no-think")        think          = false;
        else if (a == "--raw")             raw            = true;
        else if (a == "--ignore-eos")      ignore_eos     = true;
        else { usage(argv[0]); return 1; }
    }
    if (model_dir.empty() || packed_dir.empty() ||
        (prompt_text.empty() == token_ids_csv.empty())) { usage(argv[0]); return 1; }
    if (repeat == 0 || n_gen == 0) { usage(argv[0]); return 1; }

    cudaDeviceProp dp{}; cudaGetDeviceProperties(&dp, 0);
    fprintf(stderr, "device: %s sm_%d%d  total mem=%.2f GB\n",
            dp.name, dp.major, dp.minor, (double)dp.totalGlobalMem / 1e9);

    glm::Config c;
    if (!glm::load_config(model_dir, &c)) return 1;
    fprintf(stderr, "config: %u layers (%u dense), hidden=%u, %u experts x top-%u, "
                    "moe_inter=%u, vocab=%u\n",
            c.n_layers, c.dense_first, c.hidden, c.n_routed_experts,
            c.n_experts_per_tok, c.moe_inter, c.vocab);

    // --- tokenize -----------------------------------------------------------
    TokClient tok{};
    const bool need_tok = token_ids_csv.empty();
    if (need_tok && !tok_open(&tok, model_dir)) {
        fprintf(stderr, "failed to spawn tokenizer\n"); return 1;
    }
    std::vector<int> prompt_ids;
    if (!token_ids_csv.empty())  prompt_ids = parse_csv_ids(token_ids_csv);
    else if (raw)                prompt_ids = tok_encode(&tok, prompt_text);
    else                         prompt_ids = tok_chat_user(&tok, prompt_text, think);
    if (prompt_ids.empty()) { fprintf(stderr, "no prompt tokens\n"); return 1; }
    fprintf(stderr, "prompt: %zu tokens (%s)\n", prompt_ids.size(),
            !token_ids_csv.empty() ? "raw ids" : (raw ? "plain encode" :
            (think ? "chat template, thinking ON" : "chat template, thinking OFF")));
    for (int id : prompt_ids)
        if (id < 0 || (uint32_t)id >= c.vocab) {
            fprintf(stderr, "prompt token %d out of range for vocab %u\n", id, c.vocab);
            return 1;
        }

    c.max_seq = max_seq_arg > 0 ? (uint32_t)max_seq_arg
                                : (uint32_t)prompt_ids.size() + n_gen + 8;
    if (c.max_seq > c.index_topk) {
        // run_layer aborts past index_topk (the indexer-free dense path stops
        // being exact there). Fail here, with a message, instead of inside a
        // kernel launch loop 78 layers deep.
        fprintf(stderr, "max_seq %u exceeds index_topk %u — the dense attention "
                        "path is only exact up to %u positions\n",
                c.max_seq, c.index_topk, c.index_topk);
        return 1;
    }

    // --- resident weights ---------------------------------------------------
    st::ModelDir M;
    if (!st::open(&M, model_dir, /*allow_missing_shards=*/false)) {
        fprintf(stderr, "st::open failed\n"); return 1;
    }
    const double t_load0 = now_s();
    fprintf(stderr, "loading %u layers of non-expert weights (max_seq=%u)...\n",
            c.n_layers, c.max_seq);
    std::vector<glm::LayerWeights> W(c.n_layers);
    for (uint32_t l = 0; l < c.n_layers; l++) {
        if (!glm::load_layer(&M, c, l, &W[l])) {
            fprintf(stderr, "load_layer %u failed\n", l); return 1;
        }
        if (l % 10 == 0 || l + 1 == c.n_layers) {
            size_t fb = 0, tb = 0; cudaMemGetInfo(&fb, &tb);
            fprintf(stderr, "  layer %2u loaded, VRAM free %.2f / %.2f GB\n",
                    l, (double)fb / 1e9, (double)tb / 1e9);
        }
    }
    glm::ChainWeights cw;
    if (!glm::upload_raw<uint16_t>(&M, "model.embed_tokens.weight", &cw.embed,
                                   (size_t)c.vocab * c.hidden, "BF16")) return 1;
    if (!glm::upload_raw<uint16_t>(&M, "model.norm.weight", &cw.final_norm,
                                   c.hidden, "BF16")) return 1;
    if (!glm::upload_raw<uint16_t>(&M, "lm_head.weight", &cw.lm_head,
                                   (size_t)c.vocab * c.hidden, "BF16")) return 1;
    CUDA_OK(cudaMalloc(&cw.d_hidden, c.hidden * sizeof(float)));
    CUDA_OK(cudaMalloc(&cw.d_hnorm,  c.hidden * sizeof(float)));
    float* d_logits = nullptr;
    CUDA_OK(cudaMalloc(&d_logits, (size_t)c.vocab * sizeof(float)));

    {
        size_t fb = 0, tb = 0; CUDA_OK(cudaMemGetInfo(&fb, &tb));
        fprintf(stderr, "resident (non-expert) weights: %.2f GB, %.2f GB free, "
                        "%.1fs\n", (double)(tb - fb) / 1e9, (double)fb / 1e9,
                now_s() - t_load0);
    }

    // --- expert cache, sized from what is actually left ----------------------
    const glm::ExpertLayout lay = glm::ExpertLayout::from(c);
    uint32_t cap = forced_cap;
    if (cap == 0) {
        size_t free_b = 0, total_b = 0;
        CUDA_OK(cudaMemGetInfo(&free_b, &total_b));
        const size_t reserve = reserve_gb << 30;
        if (free_b <= reserve) {
            fprintf(stderr, "only %.2f GB free, below the %zu GB reserve\n",
                    (double)free_b / 1e9, reserve_gb);
            return 1;
        }
        cap = (uint32_t)((free_b - reserve) / lay.total);
    }
    const uint32_t total_experts = (c.n_layers - c.dense_first) * c.n_routed_experts;
    fprintf(stderr, "expert cache: %u experts = %.2f GB (%u GB reserved); residency "
                    "%u/%u = %.1f%% of the routed set\n",
            cap, (double)cap * lay.total / 1e9, (unsigned)reserve_gb,
            cap, total_experts, 100.0 * (double)cap / (double)total_experts);
    glm::ExpertCache cache;
    if (!cache.init(cap, packed_dir, c.dense_first, c.n_layers - c.dense_first, io_threads)) {
        fprintf(stderr, "ExpertCache::init failed\n"); return 1;
    }
    size_t peak_resident = 0;
    {
        size_t fb = 0, tb = 0; CUDA_OK(cudaMemGetInfo(&fb, &tb));
        peak_resident = tb - fb;
        fprintf(stderr, "VRAM after all allocations: %.2f GB resident of %.2f GB "
                        "(%.2f GB free)\n",
                (double)(tb - fb) / 1e9, (double)tb / 1e9, (double)fb / 1e9);
    }

    glm::Timing timing;
    if (want_timing) glm::g_timing = &timing;

    auto is_eos = [&](int id) {
        for (int e : tok.eos_ids) if (id == e) return true;
        return false;
    };

    std::vector<float> logits(c.vocab);
    std::vector<double> tok_s_decode(repeat, 0.0), tok_s_e2e(repeat, 0.0);
    std::vector<double> hit_rate_req(repeat, 0.0), hit_rate_cum(repeat, 0.0);
    std::vector<uint32_t> produced(repeat, 0);

    // --- the request loop ---------------------------------------------------
    for (uint32_t r = 0; r < repeat; r++) {
        const size_t h0 = cache.hits(), m0 = cache.misses();
        if (want_timing) timing.reset();
        // The KV cache is keyed by absolute position and every request replays
        // the same prompt from position 0, so nothing needs clearing: run_layer
        // overwrites rows [0, pos] and reads only T = pos + 1 of them.
        std::vector<uint32_t> ids(prompt_ids.begin(), prompt_ids.end());

        if (need_tok) { tok_decode_reset(&tok); }
        printf("\n=== request %u/%u ===\n", r + 1, repeat);
        fflush(stdout);

        const double t_req0 = now_s();
        // prefill: the whole prompt in one run_chain call, logits for the last
        // position only (run_chain's contract).
        glm::run_chain(c, W, cw, lay, cache, ids.data(), (uint32_t)ids.size(), 0,
                       d_logits, nullptr, 0);
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemcpy(logits.data(), d_logits, (size_t)c.vocab * sizeof(float),
                           cudaMemcpyDeviceToHost));
        const double t_prefill = now_s() - t_req0;

        auto argmax = [&]() {
            int best = 0;
            for (size_t i = 1; i < logits.size(); i++) if (logits[i] > logits[best]) best = (int)i;
            return best;
        };

        int next = argmax();
        uint32_t emitted = 0;
        uint32_t pos = (uint32_t)ids.size();
        const double t_dec0 = now_s();
        uint32_t decode_steps = 0;
        std::string text;
        while (emitted < n_gen) {
            if (!ignore_eos && need_tok && is_eos(next)) {
                fprintf(stderr, "\n[eos %d after %u tokens]\n", next, emitted);
                break;
            }
            if (need_tok) {
                std::string frag = tok_decode_push(&tok, next);
                text += frag;
                fwrite(frag.data(), 1, frag.size(), stdout); fflush(stdout);
            } else {
                printf(" %d", next); fflush(stdout);
            }
            emitted++;
            if (emitted >= n_gen) break;
            if (pos + 1 >= c.max_seq) {
                fprintf(stderr, "\n[max_seq %u reached]\n", c.max_seq); break;
            }
            const uint32_t t = (uint32_t)next;
            glm::run_chain(c, W, cw, lay, cache, &t, 1, pos, d_logits, nullptr, 0);
            CUDA_OK(cudaDeviceSynchronize());
            CUDA_OK(cudaGetLastError());
            CUDA_OK(cudaMemcpy(logits.data(), d_logits, (size_t)c.vocab * sizeof(float),
                               cudaMemcpyDeviceToHost));
            pos++; decode_steps++;
            next = argmax();
            if (want_telemetry && emitted % 10 == 0) cache.print_stats("telemetry");
        }
        const double t_decode = now_s() - t_dec0;
        const double t_total  = now_s() - t_req0;
        printf("\n");

        // Two figures, both reported, neither hidden:
        //   decode  — steady-state generation rate, the number this project's
        //             prior results.tsv rows and the 2.0 gate are stated in.
        //   e2e     — every token produced divided by the whole request wall
        //             clock including prefill, which is what a user feels.
        tok_s_decode[r] = decode_steps > 0 ? (double)decode_steps / t_decode : 0.0;
        tok_s_e2e[r]    = t_total > 0 ? (double)emitted / t_total : 0.0;
        produced[r]     = emitted;

        const size_t hd = cache.hits() - h0, md = cache.misses() - m0;
        hit_rate_req[r] = (hd + md) ? 100.0 * (double)hd / (double)(hd + md) : 0.0;
        hit_rate_cum[r] = (cache.hits() + cache.misses())
            ? 100.0 * (double)cache.hits() / (double)(cache.hits() + cache.misses()) : 0.0;

        fprintf(stderr,
                "[req %u] prompt=%zu gen=%u | prefill %.2fs (%.2f tok/s over %zu) | "
                "decode %.2fs over %u steps = %.3f tok/s | e2e %.3f tok/s | "
                "hit_rate this=%.1f%% cumulative=%.1f%%\n",
                r + 1, ids.size(), emitted, t_prefill,
                t_prefill > 0 ? ids.size() / t_prefill : 0.0, ids.size(),
                t_decode, decode_steps, tok_s_decode[r], tok_s_e2e[r],
                hit_rate_req[r], hit_rate_cum[r]);
        cache.print_stats("end-of-request");

        if (want_timing) {
            const double tt = timing.total();
            const uint64_t L = timing.layers ? timing.layers : 1;
            fprintf(stderr,
                "[req %u --timing] %llu layer-passes, %llu expert fetches, %.2fs accounted\n"
                "    attention      %8.2fs  %5.1f%%   %.3f ms/layer\n"
                "    router+dense   %8.2fs  %5.1f%%   %.3f ms/layer\n"
                "    expert-fetch   %8.2fs  %5.1f%%   %.3f ms/layer\n"
                "    expert-compute %8.2fs  %5.1f%%   %.3f ms/layer\n",
                r + 1, (unsigned long long)timing.layers,
                (unsigned long long)timing.fetches, tt,
                timing.attn,   tt > 0 ? 100 * timing.attn   / tt : 0.0, 1e3 * timing.attn   / L,
                timing.router, tt > 0 ? 100 * timing.router / tt : 0.0, 1e3 * timing.router / L,
                timing.fetch,  tt > 0 ? 100 * timing.fetch  / tt : 0.0, 1e3 * timing.fetch  / L,
                timing.expert, tt > 0 ? 100 * timing.expert / tt : 0.0, 1e3 * timing.expert / L);
        }
        {
            size_t fb = 0, tb = 0; CUDA_OK(cudaMemGetInfo(&fb, &tb));
            if (tb - fb > peak_resident) peak_resident = tb - fb;
        }
        fflush(stdout); fflush(stderr);
    }

    // --- summary ------------------------------------------------------------
    printf("\n=== summary: %u requests, %u tokens each, io_threads=%u ===\n",
           repeat, n_gen, io_threads);
    printf("req\tgen\tdecode_tok_s\te2e_tok_s\thit_req%%\thit_cum%%\n");
    for (uint32_t r = 0; r < repeat; r++)
        printf("%u\t%u\t%.3f\t%.3f\t%.1f\t%.1f\n", r + 1, produced[r],
               tok_s_decode[r], tok_s_e2e[r], hit_rate_req[r], hit_rate_cum[r]);
    printf("cold (request 1) decode: %.3f tok/s\n", tok_s_decode[0]);
    printf("warm (request %u)  decode: %.3f tok/s\n", repeat, tok_s_decode[repeat - 1]);
    {
        size_t fb = 0, tb = 0; CUDA_OK(cudaMemGetInfo(&fb, &tb));
        printf("expert cache capacity: %u experts (%.2f GB); peak VRAM resident "
               "%.2f GB of %.2f GB\n", cap, (double)cap * lay.total / 1e9,
               (double)peak_resident / 1e9, (double)tb / 1e9);
    }
    cache.print_stats("final");

    glm::g_timing = nullptr;
    if (need_tok) tok_close(&tok);
    return 0;
}
