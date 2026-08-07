/*
 * glm_http.cuh — the HTTP/SSE layer for windlass serve mode.
 *
 * Deliberately free of CUDA and of any model type: this header knows how to read
 * a request, pull the handful of fields an OpenAI-compatible chat completion
 * carries, and write chunks back. It does not know what produces the tokens.
 * That separation is what lets test_glm_http build with a plain host compiler on
 * a machine with no CUDA toolchain, which is where these parsers get exercised.
 *
 * Scope is the three endpoints Task 9 asks for — GET /health, GET /v1/models,
 * POST /v1/chat/completions — with SSE streaming on the last. No tool calling,
 * no Anthropic-format routes, no sampling parameters: the engine is greedy, so
 * accepting a temperature would be accepting a field it cannot honour.
 *
 * The JSON reader is a scanner, not a parser. It handles exactly the shapes a
 * chat-completions body takes and reports failure on anything else rather than
 * guessing. A request that this file cannot understand must produce a 400 with a
 * reason, never a silent default — a serve mode that quietly reinterprets its
 * input produces output nobody can attribute.
 */
#pragma once

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

namespace glm {
namespace http {

// ---------------------------------------------------------------------------
// JSON output
// ---------------------------------------------------------------------------

// Escapes for a JSON string body (no surrounding quotes). Bytes >= 0x20 pass
// through untouched, which keeps UTF-8 sequences intact — the decoder upstream
// emits real UTF-8 and re-encoding it as \u escapes would only risk splitting a
// code point across two SSE chunks.
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
                else out += (char)c;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// JSON input — a scanner over the shapes a chat-completions body actually takes
// ---------------------------------------------------------------------------

// Skips whitespace starting at i. Returns the index of the first non-space byte,
// or s.size().
inline size_t json_skip_ws(const std::string& s, size_t i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++;
    return i;
}

// Reads a JSON string starting at s[i] == '"'. On success sets *out to the
// unescaped value and returns the index one past the closing quote. On failure
// returns std::string::npos, leaving *out untouched.
inline size_t json_read_string(const std::string& s, size_t i, std::string* out) {
    if (i >= s.size() || s[i] != '"') return std::string::npos;
    i++;
    std::string v;
    while (i < s.size()) {
        char c = s[i];
        if (c == '"') { *out = v; return i + 1; }
        if (c == '\\') {
            if (i + 1 >= s.size()) return std::string::npos;
            char e = s[i + 1];
            switch (e) {
                case '"':  v += '"';  i += 2; break;
                case '\\': v += '\\'; i += 2; break;
                case '/':  v += '/';  i += 2; break;
                case 'n':  v += '\n'; i += 2; break;
                case 'r':  v += '\r'; i += 2; break;
                case 't':  v += '\t'; i += 2; break;
                case 'b':  v += '\b'; i += 2; break;
                case 'f':  v += '\f'; i += 2; break;
                case 'u': {
                    if (i + 5 >= s.size()) return std::string::npos;
                    unsigned cp = 0;
                    for (int k = 0; k < 4; k++) {
                        char h = s[i + 2 + k];
                        unsigned d;
                        if      (h >= '0' && h <= '9') d = (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') d = (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') d = (unsigned)(h - 'A' + 10);
                        else return std::string::npos;
                        cp = (cp << 4) | d;
                    }
                    i += 6;
                    // A surrogate pair must be joined before encoding, otherwise
                    // every non-BMP character in a diff (emoji in a commit
                    // message, say) turns into two replacement-shaped blobs.
                    if (cp >= 0xD800 && cp <= 0xDBFF && i + 1 < s.size() &&
                        s[i] == '\\' && s[i + 1] == 'u' && i + 5 < s.size()) {
                        unsigned lo = 0;
                        bool ok = true;
                        for (int k = 0; k < 4; k++) {
                            char h = s[i + 2 + k];
                            unsigned d;
                            if      (h >= '0' && h <= '9') d = (unsigned)(h - '0');
                            else if (h >= 'a' && h <= 'f') d = (unsigned)(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') d = (unsigned)(h - 'A' + 10);
                            else { ok = false; break; }
                            lo = (lo << 4) | d;
                        }
                        if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                    }
                    if (cp < 0x80) v += (char)cp;
                    else if (cp < 0x800) {
                        v += (char)(0xC0 | (cp >> 6));
                        v += (char)(0x80 | (cp & 0x3F));
                    } else if (cp < 0x10000) {
                        v += (char)(0xE0 | (cp >> 12));
                        v += (char)(0x80 | ((cp >> 6) & 0x3F));
                        v += (char)(0x80 | (cp & 0x3F));
                    } else {
                        v += (char)(0xF0 | (cp >> 18));
                        v += (char)(0x80 | ((cp >> 12) & 0x3F));
                        v += (char)(0x80 | ((cp >> 6) & 0x3F));
                        v += (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default: return std::string::npos;
            }
        } else {
            v += c;
            i++;
        }
    }
    return std::string::npos;
}

// Walks past one complete JSON value at s[i] — scalar, string, object or array,
// nesting included. Returns the index one past it, or npos.
//
// This exists so find_key can skip a value it is not interested in without
// mistaking a '}' inside a string for the end of the enclosing object. A diff in
// a "content" field is full of braces and quotes, so a naive brace count is
// exactly the bug this avoids.
inline size_t json_skip_value(const std::string& s, size_t i) {
    i = json_skip_ws(s, i);
    if (i >= s.size()) return std::string::npos;
    if (s[i] == '"') { std::string tmp; return json_read_string(s, i, &tmp); }
    if (s[i] == '{' || s[i] == '[') {
        const char open = s[i], close = (open == '{') ? '}' : ']';
        int depth = 0;
        while (i < s.size()) {
            if (s[i] == '"') {
                std::string tmp;
                size_t n = json_read_string(s, i, &tmp);
                if (n == std::string::npos) return std::string::npos;
                i = n;
                continue;
            }
            if (s[i] == open) depth++;
            else if (s[i] == close) { depth--; if (depth == 0) return i + 1; }
            i++;
        }
        return std::string::npos;
    }
    // Scalar: number, true, false, null.
    const size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
           s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r') i++;
    return i > start ? i : std::string::npos;
}

// Finds `key` at the top level of the object beginning at s[obj]. Returns the
// index of its value, or npos if the object has no such key.
//
// Top level is the point. `"content"` inside a nested message object must not
// answer a search for a top-level `"content"`, and a `"max_tokens"` string
// appearing inside a diff must not answer anything at all.
inline size_t json_find_key(const std::string& s, size_t obj, const std::string& key) {
    obj = json_skip_ws(s, obj);
    if (obj >= s.size() || s[obj] != '{') return std::string::npos;
    size_t i = json_skip_ws(s, obj + 1);
    if (i < s.size() && s[i] == '}') return std::string::npos;
    while (i < s.size()) {
        std::string k;
        size_t n = json_read_string(s, i, &k);
        if (n == std::string::npos) return std::string::npos;
        i = json_skip_ws(s, n);
        if (i >= s.size() || s[i] != ':') return std::string::npos;
        i = json_skip_ws(s, i + 1);
        if (k == key) return i;
        n = json_skip_value(s, i);
        if (n == std::string::npos) return std::string::npos;
        i = json_skip_ws(s, n);
        if (i < s.size() && s[i] == ',') { i = json_skip_ws(s, i + 1); continue; }
        return std::string::npos;   // '}' or malformed — key is absent either way
    }
    return std::string::npos;
}

// ---------------------------------------------------------------------------
// The parsed request
// ---------------------------------------------------------------------------

struct ChatRequest {
    std::string model;          // echoed back; not used to select anything
    std::string prompt;         // the flattened conversation
    int         max_tokens = 0; // 0 == absent, caller supplies the default
    bool        stream = false;
    bool        has_system = false;
};

// Flattens `messages` into a single user turn.
//
// windlass applies GLM's chat template to one user string, so a multi-turn
// conversation has to be folded into that string. Roles are labelled rather than
// dropped: an unlabelled concatenation would let a prior assistant turn read as
// part of the user's request. A leading system message is prepended plainly,
// which is the shape bench_code_review.py sends anyway (one user message, no
// system), so the common path stays exactly the text the caller wrote.
inline bool parse_chat_request(const std::string& body, ChatRequest* out, std::string* err) {
    const size_t root = json_skip_ws(body, 0);
    if (root >= body.size() || body[root] != '{') { *err = "body is not a JSON object"; return false; }

    size_t v = json_find_key(body, root, "model");
    if (v != std::string::npos) json_read_string(body, v, &out->model);

    v = json_find_key(body, root, "stream");
    if (v != std::string::npos) out->stream = (body.compare(v, 4, "true") == 0);

    v = json_find_key(body, root, "max_tokens");
    if (v == std::string::npos) v = json_find_key(body, root, "max_completion_tokens");
    if (v != std::string::npos) {
        out->max_tokens = std::atoi(body.c_str() + v);
        if (out->max_tokens < 0) { *err = "max_tokens must not be negative"; return false; }
    }

    v = json_find_key(body, root, "messages");
    if (v == std::string::npos) { *err = "missing \"messages\""; return false; }
    v = json_skip_ws(body, v);
    if (v >= body.size() || body[v] != '[') { *err = "\"messages\" is not an array"; return false; }

    std::string flat;
    int n_msg = 0;
    size_t i = json_skip_ws(body, v + 1);
    while (i < body.size() && body[i] != ']') {
        if (body[i] != '{') { *err = "\"messages\" element is not an object"; return false; }
        const size_t msg = i;
        size_t after = json_skip_value(body, msg);
        if (after == std::string::npos) { *err = "unterminated message object"; return false; }

        std::string role, content;
        size_t rv = json_find_key(body, msg, "role");
        if (rv != std::string::npos) json_read_string(body, rv, &role);
        size_t cv = json_find_key(body, msg, "content");
        if (cv != std::string::npos) {
            cv = json_skip_ws(body, cv);
            if (cv < body.size() && body[cv] == '"') {
                json_read_string(body, cv, &content);
            } else if (cv < body.size() && body[cv] == '[') {
                // Content parts: concatenate every {"type":"text","text":...}.
                size_t p = json_skip_ws(body, cv + 1);
                while (p < body.size() && body[p] != ']') {
                    if (body[p] != '{') { *err = "content part is not an object"; return false; }
                    size_t tv = json_find_key(body, p, "text");
                    if (tv != std::string::npos) {
                        std::string t;
                        if (json_read_string(body, tv, &t) != std::string::npos) content += t;
                    }
                    size_t np = json_skip_value(body, p);
                    if (np == std::string::npos) { *err = "unterminated content part"; return false; }
                    p = json_skip_ws(body, np);
                    if (p < body.size() && body[p] == ',') p = json_skip_ws(body, p + 1);
                }
            } else {
                *err = "\"content\" is neither a string nor an array";
                return false;
            }
        }

        if (!content.empty()) {
            if (!flat.empty()) flat += "\n\n";
            if (role == "system") { out->has_system = true; flat += content; }
            else if (role == "user")      flat += n_msg > 0 ? "User: "      + content : content;
            else if (role == "assistant") flat += "Assistant: " + content;
            else if (!role.empty())       flat += role + ": " + content;
            else                          flat += content;
        }
        n_msg++;

        i = json_skip_ws(body, after);
        if (i < body.size() && body[i] == ',') { i = json_skip_ws(body, i + 1); continue; }
    }
    if (n_msg == 0)   { *err = "\"messages\" is empty"; return false; }
    if (flat.empty()) { *err = "no message carried any content"; return false; }
    out->prompt = flat;
    return true;
}

// ---------------------------------------------------------------------------
// Socket writing
// ---------------------------------------------------------------------------

// Every write to a client socket goes through one of these. The keepalive thread
// and the generating thread share a connection during prefill, so the mutex is
// load-bearing rather than defensive.
struct Conn {
    int        fd = -1;
    std::mutex mu;
    bool       dead = false;   // set once a write fails; stops further attempts

    bool write_all(const char* p, size_t n) {
        std::lock_guard<std::mutex> lk(mu);
        return write_all_locked(p, n);
    }
    bool write_all(const std::string& s) { return write_all(s.data(), s.size()); }

    bool write_all_locked(const char* p, size_t n) {
        if (dead) return false;
        while (n > 0) {
            ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
            if (w < 0) {
                if (errno == EINTR) continue;
                dead = true;
                return false;
            }
            if (w == 0) { dead = true; return false; }
            p += w; n -= (size_t)w;
        }
        return true;
    }
    bool alive() { std::lock_guard<std::mutex> lk(mu); return !dead; }
};

inline void send_response(Conn* c, int code, const char* reason,
                          const char* content_type, const std::string& body) {
    char head[512];
    int n = snprintf(head, sizeof(head),
                     "HTTP/1.1 %d %s\r\n"
                     "Content-Type: %s\r\n"
                     "Content-Length: %zu\r\n"
                     "Access-Control-Allow-Origin: *\r\n"
                     "Connection: close\r\n"
                     "\r\n",
                     code, reason, content_type, body.size());
    std::lock_guard<std::mutex> lk(c->mu);
    if (!c->write_all_locked(head, (size_t)n)) return;
    c->write_all_locked(body.data(), body.size());
}

inline void send_json(Conn* c, int code, const char* reason, const std::string& json) {
    send_response(c, code, reason, "application/json", json);
}

// OpenAI's error envelope, which is what a client library will look for.
inline void send_error(Conn* c, int code, const char* reason,
                       const char* type, const std::string& message) {
    std::string j = "{\"error\":{\"message\":\"" + json_escape(message) +
                    "\",\"type\":\"" + type + "\",\"param\":null,\"code\":null}}";
    send_json(c, code, reason, j);
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

// Headers only. Sent before prefill starts so the client sees a response well
// before the first token — at 156 s to first token on a real prompt, a client
// that waits for any byte at all will time out otherwise.
inline bool sse_begin(Conn* c) {
    static const char head[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "X-Accel-Buffering: no\r\n"
        "\r\n";
    return c->write_all(head, sizeof(head) - 1);
}

// An SSE comment. Carries no event, so a client's line reader discards it, but
// it is bytes on the wire and that is the whole job: it resets the read timeout
// during the minutes of prefill when there is nothing else to send.
inline bool sse_comment(Conn* c, const char* text) {
    std::string s = ": ";
    s += text;
    s += "\n\n";
    return c->write_all(s);
}

inline std::string chunk_prelude(const std::string& id, long created,
                                 const std::string& model) {
    char b[256];
    snprintf(b, sizeof(b),
             "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"",
             id.c_str(), created);
    return std::string(b) + json_escape(model) + "\"";
}

inline bool sse_role(Conn* c, const std::string& id, long created, const std::string& model) {
    std::string s = "data: " + chunk_prelude(id, created, model) +
                    ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
                    "\"finish_reason\":null}]}\n\n";
    return c->write_all(s);
}

inline bool sse_delta(Conn* c, const std::string& id, long created,
                      const std::string& model, const std::string& text) {
    std::string s = "data: " + chunk_prelude(id, created, model) +
                    ",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
                    json_escape(text) + "\"},\"finish_reason\":null}]}\n\n";
    return c->write_all(s);
}

inline bool sse_finish(Conn* c, const std::string& id, long created,
                       const std::string& model, const char* finish_reason,
                       int prompt_tokens, int completion_tokens) {
    char usage[256];
    snprintf(usage, sizeof(usage),
             ",\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
             prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    std::string s = "data: " + chunk_prelude(id, created, model) +
                    ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" +
                    finish_reason + "\"}]" + usage;
    if (!c->write_all(s)) return false;
    return c->write_all("data: [DONE]\n\n");
}

// A non-streaming completion body.
inline std::string completion_json(const std::string& id, long created,
                                   const std::string& model, const std::string& content,
                                   const char* finish_reason,
                                   int prompt_tokens, int completion_tokens) {
    char head[256];
    snprintf(head, sizeof(head),
             "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,\"model\":\"",
             id.c_str(), created);
    char tail[256];
    snprintf(tail, sizeof(tail),
             "\",\"finish_reason\":\"%s\"}],"
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
             finish_reason, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    return std::string(head) + json_escape(model) +
           "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
           json_escape(content) + "\"}" + tail;
}

// ---------------------------------------------------------------------------
// Request reading
// ---------------------------------------------------------------------------

struct Request {
    std::string method;
    std::string path;
    std::string body;
};

// Reads request line, headers and a Content-Length body. Chunked transfer
// encoding is refused rather than mis-read; no client this serves uses it.
//
// `max_body` caps what a single request may deliver. A review prompt is a few
// kilobytes, so the cap is generous, but unbounded would let one connection
// exhaust the process while it holds the only inference slot.
inline bool read_request(int fd, Request* out, size_t max_body, std::string* err) {
    std::string buf;
    char tmp[8192];
    size_t header_end = std::string::npos;

    while (header_end == std::string::npos) {
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            *err = (errno == EAGAIN || errno == EWOULDBLOCK)
                 ? "timed out waiting for request headers" : "recv failed";
            return false;
        }
        if (n == 0) { *err = "client closed before headers completed"; return false; }
        buf.append(tmp, (size_t)n);
        header_end = buf.find("\r\n\r\n");
        if (header_end == std::string::npos && buf.size() > 64 * 1024) {
            *err = "headers exceed 64 KiB"; return false;
        }
    }

    const std::string head = buf.substr(0, header_end);
    size_t sp1 = head.find(' ');
    if (sp1 == std::string::npos) { *err = "malformed request line"; return false; }
    size_t sp2 = head.find(' ', sp1 + 1);
    if (sp2 == std::string::npos) { *err = "malformed request line"; return false; }
    out->method = head.substr(0, sp1);
    out->path   = head.substr(sp1 + 1, sp2 - sp1 - 1);
    if (size_t q = out->path.find('?'); q != std::string::npos) out->path.resize(q);

    // Header names are case-insensitive, and clients do vary the spelling.
    std::string lower = head;
    for (char& ch : lower) ch = (char)std::tolower((unsigned char)ch);
    if (lower.find("\r\ntransfer-encoding: chunked") != std::string::npos) {
        *err = "chunked transfer encoding is not supported"; return false;
    }
    size_t cl = lower.find("\r\ncontent-length:");
    size_t want = 0;
    if (cl != std::string::npos) {
        want = (size_t)std::strtoul(lower.c_str() + cl + 17, nullptr, 10);
        if (want > max_body) { *err = "request body too large"; return false; }
    }

    out->body = buf.substr(header_end + 4);
    while (out->body.size() < want) {
        ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            *err = (errno == EAGAIN || errno == EWOULDBLOCK)
                 ? "timed out waiting for the request body" : "recv failed mid-body";
            return false;
        }
        if (n == 0) { *err = "client closed mid-body"; return false; }
        out->body.append(tmp, (size_t)n);
        if (out->body.size() > max_body) { *err = "request body too large"; return false; }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Listener
// ---------------------------------------------------------------------------

// Returns a listening socket, or -1 with a reason on stderr.
inline int listen_on(const char* host, int port, int backlog) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); return -1; }
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((uint16_t)port);
    if (!host || !*host || std::strcmp(host, "0.0.0.0") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_ANY);
    } else if (::inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
        fprintf(stderr, "serve: --host %s is not an IPv4 address\n", host);
        ::close(fd);
        return -1;
    }
    if (::bind(fd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); ::close(fd); return -1; }
    if (::listen(fd, backlog) < 0) { perror("listen"); ::close(fd); return -1; }
    return fd;
}

// Nagle would sit on a one-token SSE chunk waiting for company that arrives a
// second and a half later. Streaming is the point, so turn it off.
inline void set_nodelay(int fd) {
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// Both directions need a deadline, for different reasons.
//
// Receive: a connection that opens and then says nothing would otherwise pin a
// handler thread forever, and enough of them reach the connection cap and wedge
// the server.
//
// Send: this one is the serious case. If a client stops reading mid-stream —
// killed, suspended, a proxy that went away — the socket buffer fills and send()
// blocks. The thread blocking there is the one holding the single inference
// slot, so the server would answer 503 to everything else for as long as that
// dead client stayed half-open, which is indefinitely. A send timeout turns that
// into a failed write, which marks the Conn dead and ends the generation.
inline void set_timeouts(int fd, int recv_seconds, int send_seconds) {
    timeval rt{}; rt.tv_sec = recv_seconds;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rt, sizeof(rt));
    timeval st{}; st.tv_sec = send_seconds;
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &st, sizeof(st));
}

inline std::string make_id(const char* prefix, uint64_t n) {
    char b[96];
    snprintf(b, sizeof(b), "%s-%020llu", prefix, (unsigned long long)n);
    return b;
}

}  // namespace http
}  // namespace glm
