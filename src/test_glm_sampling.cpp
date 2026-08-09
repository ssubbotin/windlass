/*
 * test_glm_sampling.cpp — the degeneration defences, tested without a GPU.
 *
 *     make test_glm_sampling && ./test_glm_sampling
 *
 * Every case states the specific wrong behaviour it would catch. This project
 * has found thirteen checks that could not fail, and the two most recent both
 * came from asserting that the right thing is present rather than that the
 * wrong thing is absent.
 */
#include <cstdio>
#include <string>
#include <vector>
#include "glm_sampling.cuh"

using namespace glm::sampling;

static int g_fail = 0, g_run = 0;
static void check(bool ok, const char* what) {
    g_run++;
    if (!ok) { g_fail++; printf("  FAIL  %s\n", what); }
}

static void test_repetition() {
    printf("repetition penalty\n");
    {   // Off by default. Every correctness number in this repository was
        // measured with greedy argmax and no penalty, and the
        // byte-identical-output gate needs that configuration reachable.
        RepetitionPenalty rp;
        check(!rp.active(), "inert at penalty 1.0");
        std::vector<float> lg = {1.0f, -2.0f, 3.0f};
        rp.observe(0); rp.observe(1);
        rp.apply(lg.data(), 3);
        check(lg[0] == 1.0f && lg[1] == -2.0f, "inert really means untouched");
    }
    {
        RepetitionPenalty rp(2.0f, 8);
        std::vector<float> lg = {4.0f, -4.0f, 9.0f};
        rp.observe(0); rp.observe(1);
        rp.apply(lg.data(), 3);
        check(lg[0] == 2.0f, "positive logit is divided");
        // NEGATIVE CONTROL. A plain `v /= penalty` for every token passes a test
        // that only checks positive logits, and makes an already-disfavoured
        // token MORE likely — the exact opposite of the intent, and invisible
        // unless a negative logit is asserted on. -4/2 = -2 would be wrong.
        check(lg[1] == -8.0f, "negative logit is multiplied, not divided");
        check(lg[2] == 9.0f, "unseen token untouched");
    }
    {   // The window must actually evict, or the penalty grows unbounded over a
        // long generation and eventually suppresses the whole vocabulary.
        RepetitionPenalty rp(1.5f, 4);
        for (int i = 0; i < 100; i++) rp.observe(i);
        check(rp.tracked() == 4, "ring buffer is bounded by the window");
        std::vector<float> lg(100, 2.0f);
        rp.apply(lg.data(), 100);
        check(lg[0] == 2.0f, "token evicted from the window is no longer penalised");
        check(lg[99] != 2.0f, "token still in the window is penalised");
    }
    {   // Out-of-range ids must not corrupt memory. The tokenizer reports
        // eos ids at and above vocab_size on this checkpoint.
        RepetitionPenalty rp(2.0f, 8);
        rp.observe(-1); rp.observe(999999);
        std::vector<float> lg = {1.0f};
        rp.apply(lg.data(), 1);
        check(lg[0] == 1.0f, "out-of-range ids ignored");
    }
}

static void test_degeneration() {
    printf("degeneration detector\n");
    {
        DegenerationDetector d(64, 0.25f);
        d.observe(std::string(64, 'a'));
        check(!d.degenerate(), "ordinary letters are not degenerate");
    }
    {   // The recorded failure: distinct valid tokens, collapsed character mix.
        DegenerationDetector d(64, 0.25f);
        d.observe(std::string(64, '\x01'));
        check(d.degenerate(), "a window with no letters fires");
    }
    {   // NEGATIVE CONTROL, and the one that matters most operationally. private
        // is a Russian codebase and reviews of it are mostly Cyrillic, which is
        // multi-byte UTF-8 with the high bit set. An ASCII-only letter test
        // scores such a review at 0.0 letters and kills it on the first
        // sentence — a detector that destroys the primary use case.
        DegenerationDetector d(64, 0.25f);
        d.observe("значение турбулентности для расчёта EDR по методу Левинсона");
        check(!d.degenerate(), "Russian review text is NOT flagged");
        check(d.letter_fraction() > 0.9f, "Cyrillic counts as letter-ish");
    }
    {   // A real review is dense in punctuation and identifiers; it must survive.
        DegenerationDetector d(64, 0.25f);
        d.observe("*   **Potential Division by Zero** (`math_core.cpp:141`): if "
                  "`tpP == 0.0`, the update divides by zero and NaN propagates.");
        check(!d.degenerate(), "code-review text with heavy punctuation survives");
    }
    {   // NEGATIVE CONTROL: a partial window is not evidence. Firing before the
        // window fills would kill a generation on its first few characters,
        // which is the cheapest possible way to make this feature look correct
        // in a unit test and destroy every real request.
        DegenerationDetector d(256, 0.25f);
        d.observe("...");
        check(!d.degenerate(), "does not fire before the window is full");
    }
    {   // Eviction must decrement the letter count, or the fraction only ever
        // rises and the detector silently stops working after a letter-rich prefix.
        DegenerationDetector d(8, 0.25f);
        d.observe("aaaaaaaa");
        d.observe(std::string(8, '\x01'));
        check(d.degenerate(), "letters leaving the window are un-counted");
    }
}

// The serve driver constructs both defences unconditionally and relies on the
// DEFAULT arguments being fully inert, because that is the configuration every
// correctness result in this repository was measured under and the one the
// byte-identical-output gate needs reachable. If a default ever changes, this
// fails rather than silently altering generated text on every request.
static void test_defaults_are_inert() {
    printf("driver defaults are inert\n");
    RepetitionPenalty rp(1.0f, 1024);          // ServeCtx defaults
    DegenerationDetector d(0, 0.25f);
    check(!rp.active(), "default penalty 1.0 is inert");
    check(!d.active(), "default degen window 0 is inert");
    std::vector<float> lg = {5.0f, -5.0f};
    for (int i = 0; i < 50; i++) { rp.observe(0); rp.observe(1); }
    rp.apply(lg.data(), 2);
    check(lg[0] == 5.0f && lg[1] == -5.0f, "logits are bit-unchanged at defaults");
    for (int i = 0; i < 50; i++) d.observe(std::string(64, '\x01'));
    check(!d.degenerate(), "a disabled detector never fires, whatever it sees");
}

static void test_think_budget() {
    printf("think budget\n");
    {
        ThinkBudget tb(false, 600);
        check(!tb.thinking(), "inert when thinking is off");
        check(!tb.must_close(), "never forces a close when thinking is off");
    }
    {   // Task 8: 600 tokens with thinking on produced no review at all.
        ThinkBudget tb(true, 600, 0.5f, 16);
        check(tb.cap() == 300, "caps reasoning at half the budget");
        for (int i = 0; i < 299; i++) tb.observe();
        check(!tb.must_close(), "does not close early");
        tb.observe();
        check(tb.must_close(), "forces the close at the cap");
    }
    {   // NEGATIVE CONTROL. Greedy decode barely favours the closing tag over
        // content, in both directions: it can also fire immediately. Accepting
        // a close on token 1 throws away the whole trace and looks like a model
        // that refuses to think.
        ThinkBudget tb(true, 600, 0.5f, 16);
        tb.observe();
        check(!tb.accept_close(), "a close before min_tokens is refused");
        for (int i = 0; i < 20; i++) tb.observe();
        check(tb.accept_close(), "a close after min_tokens is honoured");
        check(!tb.thinking(), "closing ends the trace");
        check(!tb.must_close(), "a closed trace is never force-closed again");
    }
}

int main() {
    test_repetition();
    test_degeneration();
    test_think_budget();
    test_defaults_are_inert();
    printf("=== %d checks, %d failed ===\n", g_run, g_fail);
    return g_fail ? 1 : 0;
}
