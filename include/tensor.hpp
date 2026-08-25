#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
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
