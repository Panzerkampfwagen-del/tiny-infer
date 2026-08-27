#pragma once
#include "tensor.hpp"

#include <string>
#include <vector>

namespace ti {

/* ---- Llama-2-style transformer LM (the "system" layer) -------------------
 *
 * Loads a weights file produced by tools/export.py:
 *
 *   u32 json_len | json | raw tensor bytes (BF16, no padding)
 *
 * json: {"config": {...}, "tensors": [{"name","shape","dtype","offset"}...]}
 * The JSON is parsed by a dependency-free mini-parser in model.cpp.
 */

struct LMConfig {
    uint32_t dim = 0;            // d_model
    uint32_t hidden_dim = 0;     // FFN intermediate
    uint32_t n_layers = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;     // GQA (== n_heads for MHA)
    uint32_t vocab_size = 0;
    uint32_t max_seq_len = 0;
    float    rope_theta = 10000.0f;
    float    norm_eps = 1e-5f;
    bool     tie_embeddings = false;   // lm_head == tok_embeddings
};

class Tokenizer;

class Transformer {
public:
    enum class Dtype { Fp32, Int8 };

    static Transformer load(const char* path, Dtype dtype = Dtype::Fp32);
    // Random init for tests / batching benchmarks (not a language model).
    static Transformer random_init(const LMConfig& cfg, uint64_t seed);

    const LMConfig& config() const { return cfg_; }

    /* Forward tokens[0..T) at absolute positions pos0..pos0+T-1, appending
     * K/V to caches[layer] as it goes. Returns logits (T, vocab).
     * caches must be pre-initialized per layer. Causal attention is
     * offset-aware: query i sees pos0+i+1 keys. */
    Tensor forward(const std::vector<int>& tokens, int pos0,
                   std::vector<KVCache>& caches) const;

private:
    struct Layer {
        // fp32 path (bias vectors are zero -- HF Llama has no biases)
        Linear wq, wk, wv, wo, w1, w2, w3;
        Tensor attn_norm, ffn_norm;
        // int8 path for the big projections
        QLinearInt8 qwq, qwk, qwv, qwo, qw1, qw2, qw3;
    };

    LMConfig cfg_;
    Tensor tok_emb_;      // (vocab, dim)
    Tensor final_norm_;
    Tensor lm_head_;      // (vocab, dim); empty if tied
    bool   use_int8_ = false;
    std::vector<Layer> layers_;

    const Linear&   L(const Layer& ly, int which) const;
    const QLinearInt8& Q(const Layer& ly, int which) const;
    Tensor project(const Layer& ly, int which, const Tensor& x) const;
};

/* Prompt-lookup speculative decoding helpers (lossless greedy). */
// Best continuation of the last `n` tokens found earlier in the history,
// up to `max_draft` tokens; empty vector when there is no match.
std::vector<int> prompt_lookup_draft(const std::vector<int>& tokens,
                                     size_t ngram, size_t max_draft);

}  // namespace ti