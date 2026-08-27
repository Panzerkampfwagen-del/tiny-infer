#include "generate.hpp"

#include <chrono>
#include <cstring>

namespace ti {

double monotonic_ms()
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

/* ---------------------------- DecodeState ------------------------------- */

void DecodeState::init(const Transformer& m, const std::vector<int>& prompt)
{
    model = &m;
    caches.resize(m.config().n_layers);
    for (auto& c : caches)
        c.init(m.config().n_kv_heads, m.config().dim / m.config().n_heads,
               m.config().max_seq_len);
    tokens = prompt;
    Tensor logits = model->forward(tokens, 0, caches);
    pos_ = (int)prompt.size();
    last_logits = Tensor({logits.cols()});
    std::memcpy(last_logits.ptr(),
                &logits[(size_t)(logits.rows() - 1) * logits.cols()],
                sizeof(float) * logits.cols());
}

int DecodeState::greedy() const
{
    int best = 0;
    for (uint32_t i = 1; i < last_logits.cols(); i++)
        if (last_logits[i] > last_logits[best]) best = (int)i;
    return best;
}

void DecodeState::feed(int tok)
{
    Tensor lg = model->forward({tok}, pos_, caches);
    pos_++;
    tokens.push_back(tok);
    if (last_logits.cols() != lg.cols())
        last_logits = Tensor({lg.cols()});
    std::memcpy(last_logits.ptr(), lg.ptr(), sizeof(float) * lg.cols());
}

int DecodeState::step(const Sampler& smp)
{
    Tensor row = last_logits;                    // sampler wants one flat row
    const int tok = (int)smp.sample(row);
    feed(tok);
    return tok;
}

/* ------------------ greedy speculative decoding step -------------------- */

int spec_greedy_step(DecodeState& st, size_t ngram, size_t ndraft)
{
    // 1) one certain greedy token, forwarded through the ordinary path
    Sampler greedy_sampler;
    greedy_sampler.temperature = 0.f;            // greedy, never sampled
    st.step(greedy_sampler);

    // The next greedy token is argmax(last_logits); the first draft token
    // must match it or there is nothing speculative to do.
    const std::vector<int> D =
        prompt_lookup_draft(st.tokens, ngram, ndraft);
    if (D.empty() || st.greedy() != D[0])
        return 1;

    // 2) verify the whole draft window in ONE batched forward over absolute
    //    positions [base .. base+F-1]. A duplicate of the final draft token
    //    pads the block so a full-match commit still has its own prediction
    //    row available.
    const size_t base = st.tokens.size();        // draft starts here
    std::vector<int> block = D;
    block.push_back(D.back());
    Tensor rows = st.model->forward(block, (int)base, st.caches);

    // 3) acceptance chain: row j consumes draft token j, so argmax(row j)
    //    predicts the token after it; accepted while it equals draft[j+1].
    auto argmax_row = [&](size_t r) {
        const float* p = &rows[(size_t)r * rows.cols()];
        int best = 0;
        for (uint32_t i = 1; i < rows.cols(); i++)
            if (p[i] > p[best]) best = (int)i;
        return best;
    };
    size_t m = 0;                                // committed count - 1
    while (m + 1 < D.size() && argmax_row(m) == D[m + 1]) m++;

    // 4) commit D[0..m]: roll every cache back to base+m+1 entries (dropping
    //    the rejected tail including the pad), adopt row m's argmax as the
    //    running next-token distribution, extend the history.
    const size_t keep_len = base + m + 1;
    for (auto& c : st.caches)
        if (c.len > keep_len) c.len = keep_len;
    for (size_t j = 0; j <= m; ++j) st.tokens.push_back(D[j]);
    const int new_pos = (int)(base + m + 1);
    std::memcpy(st.last_logits.ptr(), &rows[(size_t)m * rows.cols()],
                sizeof(float) * rows.cols());
    st.set_pos(new_pos);
    return (int)(m + 1);
}

/* ------------------------- continuous batching --------------------------- */

void BatchRunner::admit(int id, const std::vector<int>& prompt,
                        const Sampler& smp, int max_new)
{
    Slot s;
    s.id = id;
    s.smp = smp;
    s.prompt_len = prompt.size();
    s.budget = max_new > 0 ? max_new : 0;
    if ((size_t)s.budget + prompt.size() > model_.config().max_seq_len)
        s.budget = (int)model_.config().max_seq_len - (int)prompt.size();
    s.st.init(model_, prompt);
    slots_.push_back(std::move(s));
    active_++;
}

void BatchRunner::step()
{
    // One decode step for every running slot; prompts were already prefilled
    // on admission, so every call here is a batch-wide decode round.
    for (auto& s : slots_) {
        if (s.done) continue;
        s.st.step(s.smp);
        if ((int)(s.st.tokens.size() - s.prompt_len) >= s.budget) {
            s.done = true;
            active_--;
        }
    }
}

bool BatchRunner::mark_finished(int id)
{
    for (auto& s : slots_)
        if (s.id == id && !s.done) { s.done = true; active_--; return true; }
    return false;
}

const std::vector<int>& BatchRunner::tokens_of(int id) const
{
    static const std::vector<int> empty;
    for (const auto& s : slots_)
        if (s.id == id) return s.st.tokens;
    return empty;
}

int BatchRunner::last_token_of(int id) const
{
    for (const auto& s : slots_)
        if (s.id == id && !s.st.tokens.empty())
            return s.st.tokens.back();
    return -1;
}

}  // namespace ti