#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

namespace ti {

// Row-major dense float32 tensor. Contiguous storage only -- inference
// engines live and die by layout simplicity.
struct Tensor {
    std::vector<float>    data;
    std::vector<uint32_t> shape;

    Tensor() = default;
    explicit Tensor(std::vector<uint32_t> s) : shape(std::move(s)) {
        data.resize(numel());
    }

    size_t numel() const {
        size_t n = 1;
        for (uint32_t d : shape) n *= d;
        return n;
    }
    uint32_t rows() const { return shape.size() >= 2 ? shape[shape.size() - 2] : 1; }
    uint32_t cols() const { return shape.size() >= 1 ? shape.back() : 1; }
    float*       ptr()       { return data.data(); }
    const float* ptr() const { return data.data(); }
    float& operator[](size_t i)             { return data[i]; }
    float  operator[](size_t i) const       { return data[i]; }

    void zeros() { std::fill(data.begin(), data.end(), 0.0f); }
};

/* ---- matmul: (M,K) @ (K,N) -> (M,N), both row-major -------------------- */
Tensor matmul_naive(const Tensor& A, const Tensor& B);
Tensor matmul_tiled(const Tensor& A, const Tensor& B, int block = 64);

/* ---- elementwise / normalization ops ----------------------------------- */
Tensor relu(const Tensor& X);
Tensor gelu(const Tensor& X);            // tanh approximation
Tensor add(const Tensor& A, const Tensor& B);
Tensor softmax_lastdim(const Tensor& X); // numerically stable
Tensor rmsnorm_lastdim(const Tensor& X, const Tensor* weight);

/* ---- fused linear layer -------------------------------------------------*/
struct Linear {
    Tensor W;   // (out_features, in_features)
    Tensor b;   // (out_features,)
    Tensor forward(const Tensor& X) const;                    // X: (..., in)
};

Tensor linear_bias_relu(const Linear& L, const Tensor& X);    // fused variant

/* ---- transformer primitives --------------------------------------------*/

// Rotary position embedding (HF/NeoX rotate_half convention), applied in
// place-style over X of shape (T, H*D): each row t uses angles
// theta_j = (pos_offset + t) * base^(-2j/D).
Tensor rope(const Tensor& X, uint32_t n_heads, uint32_t pos_offset = 0,
            float base = 10000.0f);

// Scaled dot-product attention over (T, H, Dh) tensors.
//   scores = softmax_over_keys(QK^T / sqrt(Dh) [+ causal mask]) @ V
// With causal=true, query i attends keys 0..i only. Returns (T, H, Dh).
Tensor sdpa(const Tensor& Q, const Tensor& K, const Tensor& V,
            bool causal = false);

// Per-layer key/value cache for autoregressive decoding: fixed-capacity
// ring-less buffer of (capacity, H, Dh), appended one token at a time.
struct KVCache {
    uint32_t n_heads = 0, head_dim = 0, capacity = 0, len = 0;
    std::vector<float> k, v;   // interleaved (t, h, d)

    void init(uint32_t heads, uint32_t dim, uint32_t cap);
    void append(const Tensor& Kt, const Tensor& Vt);   // (1, H, Dh)
    Tensor K() const;                                   // live view (len,H,Dh)
    Tensor V() const;
};

// SwiGLU feed-forward activation: silu(gate) * up, elementwise.
// silu(x) = x * sigmoid(x).
Tensor swiglu(const Tensor& gate, const Tensor& up);

/* ---- quantization --------------------------------------------------------*/

// Per-output-row symmetric INT8 quantization of a Linear layer, dequantized
// on the fly during the matmul (fp32 accumulation throughout).
struct QLinearInt8 {
    std::vector<int8_t> w;     // (out_features, in_features), row-major
    std::vector<float> scales; // one FP32 scale per output row
    Tensor b;                  // FP32 bias, applied exactly (not quantized)
    uint32_t out_features = 0, in_features = 0;

    static QLinearInt8 from_linear(const Linear& L);
    Tensor forward(const Tensor& X) const;             // X: (..., in)
    size_t weight_bytes() const { return w.size(); }
};

/* ---- sampling ------------------------------------------------------------*/

// Greedy / temperature / top-k / top-p sampling over one logits row. The
// generator seeds lazily from `seed` on first draw; two identically
// configured Samplers produce identical sequences.
struct Sampler {
    float temperature = 1.0f;
    uint32_t top_k = 0;        // 0 = disabled
    float top_p = 1.0f;        // nucleus threshold in (0, 1]
    uint64_t seed = 42;

    uint32_t sample(const Tensor& logits) const;       // logits: (vocab,)

private:
    mutable std::mt19937_64 rng_{};
    mutable bool rng_ready_ = false;
};

/* ---- model -------------------------------------------------------------*/
// 3-layer MLP classifier: d -> 4d (GELU) -> 4d (GELU) -> vocab
struct MLP {
    Linear fc0, fc1, fc2;
    uint32_t d_model = 0, d_hidden = 0, vocab = 0;

    static MLP random_init(uint32_t d_model, uint32_t d_hidden, uint32_t vocab,
                           uint64_t seed);
    Tensor forward(const Tensor& X) const;   // X: (batch, d_model) -> logits
    int   argmax_class(const Tensor& x_row) const;

    void save(const char* path) const;
    static MLP load(const char* path);
};

}  // namespace ti
