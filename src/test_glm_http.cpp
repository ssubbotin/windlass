/*
 * test_glm_http.cpp — the serve mode's protocol layer, tested without a GPU.
 *
 * glm_http.cuh deliberately includes no CUDA, so this builds with a host
 * compiler:
 *
 *     make test_glm_http && ./test_glm_http
 *
 * That matters more than convenience. The parsing here is where a serve mode
 * quietly goes wrong — a max_tokens read out of the middle of a diff, a brace
 * inside a string ending an object early — and those are bugs that a run on the
 * inference machine would not surface, because the output would still look like
 * a plausible review. They surface here, in a second, on any machine.
 *
 * Several cases carry a NEGATIVE CONTROL note. This project has found eleven
 * checks that could not fail, so each parser test states the specific wrong
 * behaviour it would catch rather than merely asserting the right one.
 */
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include <sys/socket.h>
#include <unistd.h>

#include "glm_http.cuh"

using namespace glm::http;

static int g_fail = 0, g_run = 0;

static void check(bool ok, const char* what) {
    g_run++;
    if (!ok) { g_fail++; printf("  FAIL  %s\n", what); }
}
static void check_str(const std::string& got, const std::string& want, const char* what) {
    g_run++;
    if (got != want) {
        g_fail++;
        printf("  FAIL  %s\n        got  [%s]\n        want [%s]\n",
               what, got.c_str(), want.c_str());
    }
}
static void check_int(long got, long want, const char* what) {
    g_run++;
    if (got != want) {
        g_fail++;
        printf("  FAIL  %s: got %ld, want %ld\n", what, got, want);
    }
}

// ---------------------------------------------------------------------------
static void test_escape() {
    printf("json_escape\n");
    check_str(json_escape("plain"), "plain", "passthrough");
    check_str(json_escape("a\"b"), "a\\\"b", "quote");
    check_str(json_escape("a\\b"), "a\\\\b", "backslash");
    check_str(json_escape("a\nb"), "a\\nb", "newline");
    check_str(json_escape("a\tb"), "a\\tb", "tab");
    check_str(json_escape(std::string("a\x01" "b")), "a\\u0001b", "control byte");
    // UTF-8 must survive byte-for-byte. The decoder emits real UTF-8 and the
    // Task 8 prompt is half Russian, so mangling here would corrupt every review.
    check_str(json_escape("Левинсон"), "Левинсон", "utf-8 passthrough");
}

static void test_read_string() {
    printf("json_read_string\n");
    std::string v;
    check(json_read_string("\"abc\"", 0, &v) == 5 && v == "abc", "plain");
    check(json_read_string("\"a\\\"b\"", 0, &v) != std::string::npos && v == "a\"b", "escaped quote");
    check(json_read_string("\"a\\nb\"", 0, &v) != std::string::npos && v == "a\nb", "escaped newline");
    check(json_read_string("\"\\u0041\"", 0, &v) != std::string::npos && v == "A", "\\u ascii");
    check(json_read_string("\"\\u0416\"", 0, &v) != std::string::npos && v == "Ж", "\\u cyrillic");
    // Surrogate pair. NEGATIVE CONTROL: a reader that encodes each half
    // separately produces six bytes of two invalid code points instead of the
    // four-byte character, so the length assertion below fails on that bug.
    check(json_read_string("\"\\ud83d\\ude00\"", 0, &v) != std::string::npos &&
          v == "\xF0\x9F\x98\x80" && v.size() == 4, "surrogate pair joins");
    check(json_read_string("\"unterminated", 0, &v) == std::string::npos, "unterminated rejected");
    check(json_read_string("notastring", 0, &v) == std::string::npos, "non-string rejected");
}

static void test_skip_value() {
    printf("json_skip_value\n");
    check_int((long)json_skip_value("123,", 0), 3, "number");
    check_int((long)json_skip_value("true}", 0), 4, "true");
    check_int((long)json_skip_value("null]", 0), 4, "null");
    check_int((long)json_skip_value("\"ab\"", 0), 4, "string");
    check_int((long)json_skip_value("{\"a\":1}", 0), 7, "object");
    check_int((long)json_skip_value("[1,2,[3]]", 0), 9, "nested array");
    // The one that matters. A brace-counting skipper reads the '}' inside the
    // string as the end of the object and returns 12 instead of 20.
    // NEGATIVE CONTROL: this exact input fails on a naive counter.
    check_int((long)json_skip_value("{\"a\":\"}}}}\",\"b\":1}", 0), 18,
              "brace inside string does not close the object");
    check_int((long)json_skip_value("{\"a\":\"\\\"}\"}", 0), 11,
              "escaped quote inside string");
}

static void test_find_key() {
    printf("json_find_key\n");
    const std::string o = "{\"a\":1,\"b\":\"x\",\"c\":{\"d\":2}}";
    size_t v = json_find_key(o, 0, "a");
    check(v != std::string::npos && o.compare(v, 1, "1") == 0, "first key");
    v = json_find_key(o, 0, "b");
    check(v != std::string::npos && o[v] == '"', "middle key");
    v = json_find_key(o, 0, "c");
    check(v != std::string::npos && o[v] == '{', "object value");
    check(json_find_key(o, 0, "zz") == std::string::npos, "absent key");
    // Nested keys must not answer a top-level search.
    // NEGATIVE CONTROL: a substring search for "\"d\"" finds it and returns a
    // value; this asserts npos, so that implementation fails here.
    check(json_find_key(o, 0, "d") == std::string::npos, "nested key is not top level");
    check(json_find_key("{}", 0, "a") == std::string::npos, "empty object");
    check(json_find_key("[1]", 0, "a") == std::string::npos, "array is not an object");
}

static void test_parse_basic() {
    printf("parse_chat_request — the shape bench_code_review.py sends\n");
    const std::string body =
        "{\"model\":\"windlass\",\"messages\":[{\"role\":\"user\",\"content\":\"Review this PR.\"}],"
        "\"max_tokens\":800,\"temperature\":0.3}";
    ChatRequest r; std::string err;
    check(parse_chat_request(body, &r, &err), "parses");
    check_str(r.model, "windlass", "model");
    check_str(r.prompt, "Review this PR.", "prompt is the bare user text");
    check_int(r.max_tokens, 800, "max_tokens");
    check(!r.stream, "stream defaults false");
    check(!r.has_system, "no system message");
}

static void test_parse_stream_and_alias() {
    printf("parse_chat_request — stream, max_completion_tokens\n");
    ChatRequest r; std::string err;
    check(parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true}", &r, &err),
        "stream:true parses");
    check(r.stream, "stream true");
    check_int(r.max_tokens, 0, "absent max_tokens is 0, not a guess");

    ChatRequest r2; std::string e2;
    check(parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_completion_tokens\":42}",
        &r2, &e2), "max_completion_tokens parses");
    check_int(r2.max_tokens, 42, "max_completion_tokens alias honoured");

    ChatRequest r3; std::string e3;
    check(parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false}", &r3, &e3),
        "stream:false parses");
    check(!r3.stream, "stream false");
}

static void test_parse_diff_payload() {
    printf("parse_chat_request — a diff in the content cannot leak fields\n");
    // The review prompt embeds a raw diff: braces, quotes and newlines, all of
    // which must pass through without ending an object early or truncating the
    // prompt. Note what this case does NOT prove — a substring scan for
    // "\"max_tokens\"" cannot be fooled by message content, because a JSON
    // encoder always escapes those quotes. The nested-key case below is where
    // that bug actually lives.
    const std::string body =
        "{\"messages\":[{\"role\":\"user\",\"content\":\""
        "```diff\\n+   \\\"max_tokens\\\": 999999,\\n"
        "+ struct Cfg { int max_tokens = 999999; };\\n"
        "+ if (x) { y(\\\"}\\\"); }\\n```\"}],"
        "\"max_tokens\":600}";
    ChatRequest r; std::string err;
    check(parse_chat_request(body, &r, &err), "parses a diff payload");
    check_int(r.max_tokens, 600, "top-level max_tokens survives a diff payload");
    check(r.prompt.find("999999") != std::string::npos, "the diff text survives intact");
    check(r.prompt.find("```diff") == 0, "content starts where it should");
    check(r.prompt.find("if (x) { y(\"}\"); }") != std::string::npos,
          "braces and quotes in the diff reach the prompt whole");
    // And a "stream" mentioned inside the diff must not switch on streaming.
    const std::string body2 =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"the \\\"stream\\\":true flag\"}]}";
    ChatRequest r2; std::string e2;
    check(parse_chat_request(body2, &r2, &e2), "parses");
    check(!r2.stream, "\"stream\":true inside content does not enable streaming");

    // The leak that is actually reachable through valid JSON: a key of the same
    // name nested inside a message object. Clients do attach per-message fields,
    // and this is well-formed, so nothing escapes it at the encoder.
    // NEGATIVE CONTROL: any scanner that searches the whole body rather than the
    // top level returns 999999 for max_tokens and true for stream here.
    const std::string body3 =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\","
        "\"max_tokens\":999999,\"stream\":true}],\"max_tokens\":600}";
    ChatRequest r3; std::string e3;
    check(parse_chat_request(body3, &r3, &e3), "parses nested-key body");
    check_int(r3.max_tokens, 600, "nested max_tokens does not override the top level");
    check(!r3.stream, "nested stream does not enable streaming");

    // And with no top-level max_tokens at all, a nested one must not supply it —
    // absent has to stay absent so the caller's default applies.
    const std::string body4 =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\",\"max_tokens\":999999}]}";
    ChatRequest r4; std::string e4;
    check(parse_chat_request(body4, &r4, &e4), "parses");
    check_int(r4.max_tokens, 0, "nested max_tokens is not adopted when the top level omits it");
}

static void test_parse_multiturn() {
    printf("parse_chat_request — system and multi-turn flattening\n");
    const std::string body =
        "{\"messages\":["
        "{\"role\":\"system\",\"content\":\"Be terse.\"},"
        "{\"role\":\"user\",\"content\":\"First.\"},"
        "{\"role\":\"assistant\",\"content\":\"Ack.\"},"
        "{\"role\":\"user\",\"content\":\"Second.\"}]}";
    ChatRequest r; std::string err;
    check(parse_chat_request(body, &r, &err), "parses");
    check(r.has_system, "system flagged");
    // Roles are labelled from the second message on. An unlabelled join would
    // read "Ack." as part of the user's request.
    check_str(r.prompt, "Be terse.\n\nUser: First.\n\nAssistant: Ack.\n\nUser: Second.",
              "flattened with roles");
}

static void test_parse_content_parts() {
    printf("parse_chat_request — content as an array of parts\n");
    const std::string body =
        "{\"messages\":[{\"role\":\"user\",\"content\":["
        "{\"type\":\"text\",\"text\":\"one \"},"
        "{\"type\":\"text\",\"text\":\"two\"}]}]}";
    ChatRequest r; std::string err;
    check(parse_chat_request(body, &r, &err), "parses parts");
    check_str(r.prompt, "one two", "parts concatenated");
}

static void test_parse_rejects() {
    printf("parse_chat_request — refuses rather than guesses\n");
    ChatRequest r; std::string err;
    check(!parse_chat_request("not json", &r, &err), "non-object rejected");
    check(!err.empty(), "reason given");
    err.clear();
    check(!parse_chat_request("{\"model\":\"x\"}", &r, &err), "missing messages rejected");
    check(err.find("messages") != std::string::npos, "reason names the field");
    check(!parse_chat_request("{\"messages\":[]}", &r, &err), "empty messages rejected");
    check(!parse_chat_request("{\"messages\":\"nope\"}", &r, &err), "messages not an array rejected");
    check(!parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"\"}]}", &r, &err),
        "all-empty content rejected");
    check(!parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":-5}", &r, &err),
        "negative max_tokens rejected");
    check(!parse_chat_request(
        "{\"messages\":[{\"role\":\"user\",\"content\":7}]}", &r, &err),
        "numeric content rejected");
}

static void test_response_shapes() {
    printf("response bodies match what a client parses\n");
    // bench_code_review.py reads choices[0].message.content and
    // usage.completion_tokens. Both must be present and spelled exactly.
    const std::string j = completion_json("cmpl-1", 1700000000, "windlass",
                                          "Line one\nLine \"two\"", "stop", 1464, 600);
    check(j.find("\"object\":\"chat.completion\"") != std::string::npos, "object field");
    check(j.find("\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"") != std::string::npos,
          "choices[0].message");
    check(j.find("\"content\":\"Line one\\nLine \\\"two\\\"\"") != std::string::npos,
          "content escaped");
    check(j.find("\"finish_reason\":\"stop\"") != std::string::npos, "finish_reason");
    check(j.find("\"completion_tokens\":600") != std::string::npos, "usage.completion_tokens");
    check(j.find("\"total_tokens\":2064") != std::string::npos, "usage.total_tokens sums");
}

// A Conn backed by a socketpair, so the SSE writers can be read back verbatim.
struct Pair {
    int a = -1, b = -1;
    Conn c;
    Pair() {
        int fds[2];
        socketpair(AF_UNIX, SOCK_STREAM, 0, fds);
        a = fds[0]; b = fds[1];
        c.fd = a;
    }
    ~Pair() { if (a >= 0) close(a); if (b >= 0) close(b); }
    std::string drain() {
        shutdown(a, SHUT_WR);
        std::string out; char buf[4096]; ssize_t n;
        while ((n = ::recv(b, buf, sizeof(buf), 0)) > 0) out.append(buf, (size_t)n);
        return out;
    }
};

static void test_sse() {
    printf("SSE framing\n");
    {
        Pair p;
        sse_begin(&p.c);
        sse_role(&p.c, "id1", 1700000000, "windlass");
        sse_delta(&p.c, "id1", 1700000000, "windlass", "Hello");
        sse_delta(&p.c, "id1", 1700000000, "windlass", "\n> quoted \"bit\"");
        sse_finish(&p.c, "id1", 1700000000, "windlass", "length", 10, 2);
        const std::string s = p.drain();
        check(s.find("Content-Type: text/event-stream") != std::string::npos, "content type");
        check(s.find("X-Accel-Buffering: no") != std::string::npos, "proxy buffering disabled");
        // The client matches lines beginning exactly "data: {".
        check(s.find("\ndata: {") != std::string::npos || s.find("data: {") != std::string::npos,
              "data lines start with 'data: {'");
        check(s.find("\"delta\":{\"content\":\"Hello\"}") != std::string::npos, "delta content");
        check(s.find("\\n> quoted \\\"bit\\\"") != std::string::npos, "delta escaped");
        check(s.find("\"finish_reason\":\"length\"") != std::string::npos, "finish_reason");
        check(s.find("data: [DONE]\n\n") != std::string::npos, "terminator");
        // Every event ends with a blank line, or a client's line reader stalls.
        check(s.rfind("\n\n") != std::string::npos, "blank-line separated");
    }
    {
        // A keepalive comment must not look like an event to the client.
        Pair p;
        sse_comment(&p.c, "prefill");
        const std::string s = p.drain();
        check_str(s, ": prefill\n\n", "comment framing");
        check(s.compare(0, 6, "data: ") != 0, "comment is not a data line");
    }
}

static void test_read_request() {
    printf("read_request\n");
    {
        Pair p;
        const std::string req =
            "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: h\r\n"
            "Content-Length: 9\r\n\r\n{\"a\":123}";
        ::send(p.b, req.data(), req.size(), 0);
        shutdown(p.b, SHUT_WR);
        Request r; std::string err;
        check(read_request(p.a, &r, 1 << 20, &err), "reads");
        check_str(r.method, "POST", "method");
        check_str(r.path, "/v1/chat/completions", "query string stripped from path");
        check_str(r.body, "{\"a\":123}", "body");
    }
    {
        // Header case varies between clients; requests sends "Content-Length",
        // curl sends the same, but Go's stdlib normalises differently.
        Pair p;
        const std::string req = "POST /x HTTP/1.1\r\ncontent-length: 2\r\n\r\nhi";
        ::send(p.b, req.data(), req.size(), 0);
        shutdown(p.b, SHUT_WR);
        Request r; std::string err;
        check(read_request(p.a, &r, 1 << 20, &err), "lowercase content-length");
        check_str(r.body, "hi", "body read");
    }
    {
        Pair p;
        const std::string req = "GET /health HTTP/1.1\r\nHost: h\r\n\r\n";
        ::send(p.b, req.data(), req.size(), 0);
        shutdown(p.b, SHUT_WR);
        Request r; std::string err;
        check(read_request(p.a, &r, 1 << 20, &err), "GET with no body");
        check_str(r.method, "GET", "method");
        check_str(r.path, "/health", "path");
        check(r.body.empty(), "empty body");
    }
    {
        // A body arriving in pieces must be reassembled, not truncated.
        Pair p;
        std::thread w([&] {
            const std::string h = "POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\n01234";
            ::send(p.b, h.data(), h.size(), 0);
            ::send(p.b, "56789", 5, 0);
            shutdown(p.b, SHUT_WR);
        });
        Request r; std::string err;
        bool ok = read_request(p.a, &r, 1 << 20, &err);
        w.join();
        check(ok, "split body reads");
        check_str(r.body, "0123456789", "reassembled");
    }
    {
        Pair p;
        const std::string req = "POST /x HTTP/1.1\r\nContent-Length: 5000\r\n\r\nab";
        ::send(p.b, req.data(), req.size(), 0);
        shutdown(p.b, SHUT_WR);
        Request r; std::string err;
        check(!read_request(p.a, &r, 100, &err), "oversized body refused");
        check(err.find("too large") != std::string::npos, "reason given");
    }
    {
        Pair p;
        const std::string req =
            "POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n";
        ::send(p.b, req.data(), req.size(), 0);
        shutdown(p.b, SHUT_WR);
        Request r; std::string err;
        check(!read_request(p.a, &r, 1 << 20, &err), "chunked refused rather than mis-read");
    }
}

static void test_conn_death() {
    printf("Conn survives a client that hangs up\n");
    Pair p;
    close(p.b);
    p.b = -1;
    // The peer is gone. MSG_NOSIGNAL means EPIPE rather than SIGPIPE killing the
    // process — a client that closes its browser tab must not take the server
    // down while it holds the only inference slot.
    bool first = p.c.write_all("x", 1);
    for (int i = 0; i < 64 && p.c.alive(); i++) p.c.write_all(std::string(4096, 'x'));
    check(!p.c.alive() || first, "write reports failure without raising a signal");
    check(!p.c.write_all("y", 1) || p.c.alive(), "dead connection stops accepting writes");
}

int main() {
    printf("=== test_glm_http ===\n");
    test_escape();
    test_read_string();
    test_skip_value();
    test_find_key();
    test_parse_basic();
    test_parse_stream_and_alias();
    test_parse_diff_payload();
    test_parse_multiturn();
    test_parse_content_parts();
    test_parse_rejects();
    test_response_shapes();
    test_sse();
    test_read_request();
    test_conn_death();
    printf("=== %d checks, %d failed ===\n", g_run, g_fail);
    return g_fail ? 1 : 0;
}
