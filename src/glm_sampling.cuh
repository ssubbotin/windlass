/*
 * glm_sampling.cuh — decode-time defences against degeneration.
 *
 * No CUDA and no model types, so test_glm_sampling builds with a host compiler
 * and runs on a workstation that cannot build CUDA at all. Same reasoning as
 * glm_http.cuh: a defect here produces plausible-looking wrong text rather than
 * an error, and a GPU window is far too expensive a place to find it.
 *
 * Why this exists. windlass decodes greedily with no repetition penalty, which
 * is exactly the configuration that produced the degenerate word-association
 * output recorded for another engine on the same benchmark: a review that opens
 * correctly and then runs a thousand tokens of "...service level agreements key
 * performance indicators objectives key results balanced scoreboard..." without
 * ever stopping. Three clean reviews are not evidence of immunity.
 *
 * Everything here is OFF unless asked for. Greedy argmax with no penalty is the
 * configuration every correctness result in this repository was measured under,
 * and the byte-identical-output gate depends on it staying reachable.
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <deque>

namespace glm { namespace sampling {

// ---------------------------------------------------------------------------
// Repetition penalty
// ---------------------------------------------------------------------------

// Divide the logit of every recently-emitted token by `penalty`, or multiply
// when the logit is negative. The sign split is not a detail: a plain division
// would make an already-disfavoured token MORE likely, which is the opposite of
// the intent and the classic way this gets written wrong.
class RepetitionPenalty {
public:
    RepetitionPenalty(float penalty = 1.0f, size_t window = 1024)
        : penalty_(penalty), window_(window) {}

    bool active() const { return penalty_ > 1.0f && window_ > 0; }

    void observe(int token) {
        if (!active()) return;
        recent_.push_back(token);
        if (recent_.size() > window_) recent_.pop_front();
    }

    void apply(float* logits, int vocab) const {
        if (!active()) return;
        for (int t : recent_) {
            if (t < 0 || t >= vocab) continue;
            float& v = logits[t];
            v = (v > 0.0f) ? (v / penalty_) : (v * penalty_);
        }
    }

    void reset() { recent_.clear(); }
    size_t tracked() const { return recent_.size(); }

private:
    float  penalty_;
    size_t window_;
    std::deque<int> recent_;
};

// ---------------------------------------------------------------------------
// Degeneration detector
// ---------------------------------------------------------------------------

// Watches the decoded text, not token ids. A model that has fallen into a loop
// keeps producing *valid distinct tokens*, so an id-level repeat check does not
// see it; what collapses is the character mix. The recorded failure ran through
// Latin words, then CJK, then Greek letters, all distinct ids.
//
// Fires when the recent window is overwhelmingly free of ASCII letters. That
// deliberately does not fire on ordinary code review output, which is dense in
// identifiers, and does not fire on Russian text either — see `letterish`.
class DegenerationDetector {
public:
    DegenerationDetector(size_t window = 256, float min_letter_frac = 0.25f)
        : window_(window), min_frac_(min_letter_frac) {}

    bool active() const { return window_ > 0 && min_frac_ > 0.0f; }

    // Counted as "letter-ish": ASCII letters, and any byte with the high bit
    // set. The second half matters — every Cyrillic character is multi-byte
    // UTF-8, and counting only ASCII would flag a perfectly good Russian review
    // as degenerate on its first sentence.
    static bool letterish(unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c & 0x80);
    }

    void observe(const std::string& frag) {
        if (!active()) return;
        for (unsigned char c : frag) {
            chars_.push_back(c);
            if (letterish(c)) letters_++;
            if (chars_.size() > window_) {
                if (letterish(chars_.front())) letters_--;
                chars_.pop_front();
            }
        }
    }

    // Only meaningful once the window is full; a short prefix is not evidence.
    bool degenerate() const {
        if (!active() || chars_.size() < window_) return false;
        return (float)letters_ / (float)chars_.size() < min_frac_;
    }

    float letter_fraction() const {
        return chars_.empty() ? 1.0f : (float)letters_ / (float)chars_.size();
    }

    void reset() { chars_.clear(); letters_ = 0; }

private:
    size_t window_;
    float  min_frac_;
    std::deque<unsigned char> chars_;
    size_t letters_ = 0;
};

// ---------------------------------------------------------------------------
// Think budget
// ---------------------------------------------------------------------------

// Task 8 measured that a review-sized budget with thinking on yields NO review:
// all 600 tokens went to the reasoning trace. The mechanism is known — greedy
// decoding barely favours the end-of-thinking token over more content on a long
// prompt — so the answer is not to disable reasoning but to bound it.
//
// Caps reasoning at `frac` of the total budget, then reports that the closing
// tag should be injected. Also refuses to close in the first `min_tokens`,
// because the same weak preference can end the trace immediately.
class ThinkBudget {
public:
    ThinkBudget(bool thinking, uint32_t max_tokens,
                float frac = 0.5f, uint32_t min_tokens = 16)
        : thinking_(thinking), min_(min_tokens),
          cap_(thinking ? (uint32_t)(max_tokens * frac) : 0) {}

    bool thinking() const { return thinking_ && !closed_; }

    // Call once per emitted token while still inside the trace.
    void observe() { if (thinking_ && !closed_) emitted_++; }

    // Should the closing tag be forced now?
    bool must_close() const { return thinking_ && !closed_ && emitted_ >= cap_; }

    // The model produced the closing tag itself. Honoured only after `min_`
    // tokens, so a premature close does not throw away the whole trace.
    bool accept_close() {
        if (!thinking_ || closed_) return false;
        if (emitted_ < min_) return false;
        closed_ = true;
        return true;
    }

    void force_close() { closed_ = true; }
    uint32_t emitted() const { return emitted_; }
    uint32_t cap() const { return cap_; }

private:
    bool     thinking_;
    bool     closed_ = false;
    uint32_t min_;
    uint32_t cap_;
    uint32_t emitted_ = 0;
};

}}  // namespace glm::sampling
