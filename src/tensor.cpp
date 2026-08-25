#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace ti {

/* ------------------------------------------------------------ matmul ---- */

Tensor matmul_naive(const Tensor& A, const Tensor& B)
{
    assert(A.cols() == B.shape[0]);
    const uint32_t M = A.rows(), K = A.cols(), N = B.cols();
    Tensor C({M, N});
    std::fill(C.data.begin(), C.data.end(), 0.0f);

    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t k = 0; k < K; k++) {
            const float a = A[(size_t)i * K + k];
            for (uint32_t j = 0; j < N; j++)
                C[(size_t)i * N + j] += a * B[(size_t)k * N + j];
        }
    }
    return C;
}

/*
 * Register-blocked matmul: processes two rows of C per pass so every loaded
 * element of B does double duty (halves B memory traffic), keeps accumulators
 * in registers, and blocks K so A/B slices stay cache-resident. Same math,
 * verifiable against matmul_naive.
 */
Tensor matmul_tiled(const Tensor& A, const Tensor& B, int block)
{
    assert(A.cols() == B.shape[0]);
    const uint32_t M = A.rows(), K = A.cols(), N = B.cols();
    Tensor C({M, N});
    std::fill(C.data.begin(), C.data.end(), 0.0f);
    const float* __restrict ap = A.data.data();
    const float* __restrict bp = B.data.data();
    float*       __restrict cp = C.data.data();

    if (block <= 0 || (uint32_t)block > K) block = (int)K;

    // pairs of rows in parallel; each thread owns disjoint C rows
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t ii = 0; ii < (int64_t)M - 1; ii += 2) {
        const uint32_t i2 = (uint32_t)ii;
        float* __restrict crow0 = cp + (size_t)i2 * N;
        float* __restrict crow1 = cp + (size_t)(i2 + 1) * N;
        for (uint32_t kk = 0; kk < K; kk += (uint32_t)block) {
            const uint32_t kend = std::min(K, kk + (uint32_t)block);
            for (uint32_t k = kk; k < kend; k++) {
                const float a0 = ap[(size_t)i2 * K + k];
                const float a1 = ap[(size_t)(i2 + 1) * K + k];
                const float* __restrict brow = bp + (size_t)k * N;
                for (uint32_t j = 0; j < N; j++) {
                    crow0[j] += a0 * brow[j];
                    crow1[j] += a1 * brow[j];
                }
            }
        }
    }
    if (M & 1u) {                           // odd-M tail row
        float* __restrict crow = cp + (size_t)(M - 1) * N;
        for (uint32_t k = 0; k < K; k++) {
            const float a = ap[(size_t)(M - 1) * K + k];
            const float* __restrict brow = bp + (size_t)k * N;
            for (uint32_t j = 0; j < N; j++) crow[j] += a * brow[j];
        }
    }
    return C;
}

/* ------------------------------------------------- elementwise / norms -- */

Tensor relu(const Tensor& X)
{
    Tensor Y = X;
    for (float& v : Y.data) v = v > 0.0f ? v : 0.0f;
    return Y;
}

Tensor gelu(const Tensor& X)
{
    // tanh approximation used by GPT-family models
    Tensor Y(X.shape);
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    for (size_t i = 0; i < X.data.size(); i++) {
        float x = X.data[i];
        float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
        Y.data[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
    return Y;
}

Tensor add(const Tensor& A, const Tensor& B)
{
    assert(A.shape == B.shape && "add requires identical shapes");
    Tensor C(A.shape);
    for (size_t i = 0; i < A.data.size(); i++)
        C.data[i] = A.data[i] + B.data[i];
    return C;
}

Tensor softmax_lastdim(const Tensor& X)
{
    Tensor Y(X.shape);
    const uint32_t cols = X.cols();
    const size_t   rows = X.data.empty() ? 0 : X.numel() / cols;
    if (cols == 0 || rows == 0) return Y;   // avoid div-by-zero on empty dims

    for (size_t r = 0; r < rows; r++) {
        const float* xin = &X.data[r * cols];
        float* yout = &Y.data[r * cols];
        float m = xin[0];
        for (uint32_t j = 1; j < cols; j++) m = std::max(m, xin[j]);
        float sum = 0.0f;
        for (uint32_t j = 0; j < cols; j++) { yout[j] = std::exp(xin[j] - m); sum += yout[j]; }
        for (uint32_t j = 0; j < cols; j++) yout[j] /= sum;
    }
    return Y;
}

Tensor rmsnorm_lastdim(const Tensor& X, const Tensor* weight)
{
    Tensor Y(X.shape);
    const uint32_t cols = X.cols();
    const size_t   rows = X.data.empty() ? 0 : X.numel() / cols;
    if (cols == 0 || rows == 0) return Y;
    assert(!weight || (weight->data.size() == cols && "weight must be (cols,)"));

    const float eps = 1e-5f;

    for (size_t r = 0; r < rows; r++) {
        const float* x = &X.data[r * cols];
        float ss = 0.0f;
        for (uint32_t j = 0; j < cols; j++) ss += x[j] * x[j];
        const float inv = 1.0f / std::sqrt(ss / cols + eps);
        for (uint32_t j = 0; j < cols; j++) {
            float w = weight ? (*weight)[j] : 1.0f;
            Y.data[r * cols + j] = x[j] * inv * w;
        }
    }
    return Y;
}

/* ---------------------------------------------------------- linear ------ */

Tensor Linear::forward(const Tensor& X) const
{
    // X: (batch, in) ; W: (out, in) row-major => logits = X @ W^T + b
    assert(X.cols() == W.cols());
    const uint32_t batch = X.rows(), in = W.cols(), out = W.rows();
    const float*   bias  = b.data.data();
    Tensor Y({batch, out});
    std::fill(Y.data.begin(), Y.data.end(), 0.0f);

    for (uint32_t r = 0; r < batch; r++) {
        const float* xrow = &X.data[(size_t)r * in];
        float*       yrow = &Y.data[(size_t)r * out];
        for (uint32_t o = 0; o < out; o++) {
            const float* wrow = &W.data[(size_t)o * in];
            float acc = bias[o];
            for (uint32_t i = 0; i < in; i++) acc += xrow[i] * wrow[i];
            yrow[o] = acc;
        }
    }
    return Y;
}

Tensor linear_bias_relu(const Linear& L, const Tensor& X)
{
    Tensor Y = L.forward(X);          // one pass over outputs
    for (float& v : Y.data)
        if (v < 0.0f) v = 0.0f;       // fused activation: no extra read pass
    return Y;
}

/* ------------------------------------------------------------- model ---- */

static void fill_random(Tensor& t, uint64_t seed, float scale)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, scale);
    for (float& v : t.data) v = dist(rng);
}

MLP MLP::random_init(uint32_t d_model, uint32_t d_hidden, uint32_t vocab,
                     uint64_t seed)
{
    MLP m;
    m.d_model = d_model; m.d_hidden = d_hidden; m.vocab = vocab;
    m.fc0.W = Tensor({d_hidden, d_model});  m.fc0.b = Tensor({d_hidden});
    m.fc1.W = Tensor({d_hidden, d_hidden}); m.fc1.b = Tensor({d_hidden});
    m.fc2.W = Tensor({vocab, d_hidden});    m.fc2.b = Tensor({vocab});
    fill_random(m.fc0.W, seed + 0, 0.02f);  fill_random(m.fc0.b, seed + 1, 0.01f);
    fill_random(m.fc1.W, seed + 2, 0.02f);  fill_random(m.fc1.b, seed + 3, 0.01f);
    fill_random(m.fc2.W, seed + 4, 0.02f);  fill_random(m.fc2.b, seed + 5, 0.01f);
    return m;
}

Tensor MLP::forward(const Tensor& X) const
{
    // documented architecture: d -> h (GELU) -> h (GELU) -> vocab
    Tensor h = gelu(fc0.forward(X));
    h = gelu(fc1.forward(h));
    return fc2.forward(h);
}

int MLP::argmax_class(const Tensor& x_row) const
{
    Tensor xin = x_row;
    if (xin.shape.size() == 1) xin.shape.insert(xin.shape.begin(), 1);
    assert(xin.rows() == 1 && "argmax_class expects a single row");
    Tensor logits = forward(xin);
    int best = 0;
    for (uint32_t j = 1; j < logits.cols(); j++)
        if (logits[j] > logits[best]) best = (int)j;
    return best;
}

/* ------------------------------------------------------- serialization --
 *
 *   file:  "TIIN" | u32 d_model | u32 d_hidden | u32 vocab | tensor*
 *   tensor:"TNSR" | u32 ndims | u32 dim[ndims] | f32 payload[numel]
 *
 * All reads are validated; corrupt/truncated files throw std::runtime_error
 * instead of tripping asserts (which vanish under -DNDEBUG) or segfaulting.
 */

static void write_u32(std::ofstream& f, uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
}

static uint32_t read_u32(std::ifstream& f) {
    uint32_t v = 0;
    if (!f.read(reinterpret_cast<char*>(&v), 4))
        throw std::runtime_error("tiny_infer: truncated weight file");
    return v;
}

static void write_tensor(std::ofstream& f, const Tensor& t) {
    f.write("TNSR", 4);
    write_u32(f, (uint32_t)t.shape.size());
    for (uint32_t d : t.shape) write_u32(f, d);
    f.write(reinterpret_cast<const char*>(t.data.data()),
            sizeof(float) * t.numel());
}

static Tensor read_tensor(std::ifstream& f) {
    char magic[4] = {0, 0, 0, 0};
    if (!f.read(magic, 4) || memcmp(magic, "TNSR", 4) != 0)
        throw std::runtime_error("tiny_infer: corrupt tensor block");
    uint32_t ndims = read_u32(f);
    if (ndims == 0 || ndims > 8)
        throw std::runtime_error("tiny_infer: implausible ndims");
    Tensor t;
    t.shape.resize(ndims);
    size_t n = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        t.shape[i] = read_u32(f);
        n *= t.shape[i];
    }
    t.data.resize(n);
    f.read(reinterpret_cast<char*>(t.data.data()), sizeof(float) * n);
    if (f.gcount() != (std::streamsize)(sizeof(float) * n))
        throw std::runtime_error("tiny_infer: truncated tensor payload");
    return t;
}

void MLP::save(const char* path) const
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("tiny_infer: cannot open ") +
                                 path + " for writing");
    f.write("TIIN", 4);
    write_u32(f, d_model); write_u32(f, d_hidden); write_u32(f, vocab);
    write_tensor(f, fc0.W); write_tensor(f, fc0.b);
    write_tensor(f, fc1.W); write_tensor(f, fc1.b);
    write_tensor(f, fc2.W); write_tensor(f, fc2.b);
    f.flush();
    if (!f.good())
        throw std::runtime_error(std::string("tiny_infer: failed writing ") +
                                 path);
}

MLP MLP::load(const char* path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("tiny_infer: cannot open ") +
                                 path);
    char magic[4] = {0, 0, 0, 0};
    if (!f.read(magic, 4) || memcmp(magic, "TIIN", 4) != 0)
        throw std::runtime_error("not a tiny_infer model (bad magic)");
    MLP m;
    m.d_model = read_u32(f); m.d_hidden = read_u32(f); m.vocab = read_u32(f);

    m.fc0.W = read_tensor(f); m.fc0.b = read_tensor(f);
    m.fc1.W = read_tensor(f); m.fc1.b = read_tensor(f);
    m.fc2.W = read_tensor(f); m.fc2.b = read_tensor(f);

    // cross-check loaded tensors against the header dims
    const Linear* L[3] = {&m.fc0, &m.fc1, &m.fc2};
    const uint32_t outs[3] = {m.d_hidden, m.d_hidden, m.vocab};
    for (int i = 0; i < 3; i++) {
        if (L[i]->W.rows() != outs[i] || L[i]->b.data.size() != outs[i])
            throw std::runtime_error(
                "tiny_infer: tensor/header dim mismatch in model file");
    }
    if (!(L[0]->W.cols() == m.d_model && L[1]->W.cols() == m.d_hidden &&
          L[2]->W.cols() == m.d_hidden))
        throw std::runtime_error(
            "tiny_infer: tensor/header dim mismatch in model file");
    return m;
}

}  // namespace ti
