/*
 * bench_batch -- continuous-batching goodput vs serial decode.
 *
 * Runs a fixed number of synthetic sequences twice over the same model:
 *   serial  : one DecodeState at a time
 *   batched : BatchRunner round-robin (interleaved prompts prefilled on
 *             admission, batch-wide decode rounds)
 * Reports goodput tok/s for both and the speedup. Sequences are greedy so
 * both modes do identical work; the model is tiny/synthetic and only exists
 * to make the scheduler measurable. Reproduce with `make bench-batch`.
 */
#include "../include/generate.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace ti;

int main(int argc, char** argv)
{
    int n_seq = argc > 1 ? atoi(argv[1]) : 8;
    const int prompt_len = 16;
    const int new_tokens = 32;

    LMConfig c;
    c.dim = 256; c.hidden_dim = 512; c.n_layers = 4;
    c.n_heads = 8; c.n_kv_heads = 2;
    c.vocab_size = 1000; c.max_seq_len = 128;
    c.tie_embeddings = true;
    Transformer m = Transformer::random_init(c, 5);

    std::mt19937 rng(3);
    std::vector<std::vector<int>> prompts((size_t)n_seq);
    for (auto& p : prompts) {
        for (int i = 0; i < prompt_len; i++) p.push_back((int)(rng() % c.vocab_size));
    }

    Sampler g;                                     // temperature 0 => greedy
    g.temperature = 0.f;
    double t0, t1;

    // ---- serial ----
    long serial_tok = 0;
    t0 = monotonic_ms();
    for (auto& p : prompts) {
        DecodeState st;
        st.init(m, p);
        serial_tok += (long)st.tokens.size() - p.size(); // prefill is separate
        double prefill_ms = monotonic_ms() - t0;
        (void)prefill_ms;
        for (int i = 0; i < new_tokens; i++) { st.step(g); serial_tok++; }
    }
    t1 = monotonic_ms();
    const double serial_ms = t1 - t0;

    // ---- batched ----
    t0 = monotonic_ms();
    BatchRunner runner(m);
    for (size_t i = 0; i < prompts.size(); i++)
        runner.admit((int)i, prompts[i], g, new_tokens);
    long batched_tok = (long)runner.active() * 0; // generation counted below
    while (!runner.finished()) runner.step();
    for (int id = 0; id < n_seq; id++) batched_tok += new_tokens;
    t1 = monotonic_ms();
    const double batched_ms = t1 - t0;

    printf("sequences=%d prompt_len=%d new_tokens=%d\n", n_seq,
           prompt_len, new_tokens);
    printf("serial : %6ld tok in %7.1f ms -> %7.1f tok/s\n",
           serial_tok, serial_ms, serial_tok * 1000.0 / serial_ms);
    printf("batched: %6ld tok in %7.1f ms -> %7.1f tok/s  (%.2fx)\n",
           batched_tok, batched_ms, batched_tok * 1000.0 / batched_ms,
           (batched_tok / batched_ms) / (serial_tok / serial_ms));
    return 0;
}