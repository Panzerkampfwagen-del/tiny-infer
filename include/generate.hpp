#pragma once
#include "model.hpp"

#include <chrono>
#include <optional>

namespace ti {

/* Single autoregressive stream bound to one model instance. Owns its KV
 * caches and the last-step logits. Fields are exposed for the CLI and tests;
 * the speculative-verification protocol lives in spec_greedy_step(). */
struct DecodeState {
    const Transformer* model = nullptr;
    std::vector<KVCache> caches;
    std::vector<int> tokens;          // full history (prompt + generated)
    Tensor last_logits;               // (vocab,) -- predicts after tokens.back()

    void init(const Transformer& m, const std::vector<int>& prompt);
    /* Force-feed one known token (append + one decode forward). Used to
     * replay an existing sequence and internally by step()/spec path. */
    void feed(int tok);
    /* Sample (or greedy) the next token from last_logits, append it, run one
     * decode forward, refresh last_logits. Returns the emitted token. */
    int step(const Sampler& smp);
    int greedy() const;               // argmax of last_logits
    void set_pos(int p) { pos_ = p; } // used by speculative verification

private:
    int pos_ = 0;                     // next absolute decode position
};

/* Lossless greedy prompt-lookup speculative decoding step: drafts up to
 * `ndraft` continuations of the trailing `ngram`-gram found earlier in the
 * history, verifies the whole draft window in ONE forward pass, rolls back
 * rejected KV entries, and adopts the deepest verified prediction row.
 * Returns the number of NEW tokens committed this call (>= 1). Greedy only;
 * sampled decoding must use step(). */
int spec_greedy_step(DecodeState& st, size_t ngram, size_t ndraft);
int spec_greedy_step(DecodeState& st, size_t ngram, size_t ndraft);

/* Continuous-batching scheduler: N sequences share one model in a round-robin
 * decode loop, prompts prefilled on admission (interleaved prefill/decode).
 * This is the CPU analogue of vLLM-style batching without paging. */
class BatchRunner {
public:
    explicit BatchRunner(const Transformer& m) : model_(m) {}

    void admit(int id, const std::vector<int>& prompt, const Sampler& smp,
               int max_new);
    /* One decode step for EVERY unfinished sequence. Sequences stop when
     * they hit max_seq_len; callers observe tokens and call mark_finished()
     * on EOS or their own stopping rules. */
    void step();
    bool finished() const { return active_ == 0; }
    size_t active() const { return active_; }
    bool mark_finished(int id);

    const std::vector<int>& tokens_of(int id) const;
    int last_token_of(int id) const;

private:
    struct Slot {
        int id;
        DecodeState st;
        Sampler smp;
        bool done = false;
        size_t prompt_len = 0;
        int budget = 0;                 // max_new tokens
    };
    std::vector<Slot> slots_;
    const Transformer& model_;
    size_t active_ = 0;
};

/* Wall-clock helper shared by CLI benchmarks. */
double monotonic_ms();

}  // namespace ti