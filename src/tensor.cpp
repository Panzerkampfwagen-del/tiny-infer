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
    // X: (batch, in) ; W: (out, in) row-major => Y = X @ W^T + b
    if (X.cols() != W.cols())
        throw std::runtime_error("Linear::forward: input width " +
                                 std::to_string(X.cols()) +
                                 " != weight input width " +
                                 std::to_string(W.cols()));
    if (b.data.size() != W.rows())
        throw std::runtime_error("Linear::forward: bias size " +
                                 std::to_string(b.data.size()) +
                                 " != output width " +
                                 std::to_string(W.rows()));
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


/* ---- transformer primitives ---------------------------------------------*/

// Rotary position embedding, HF/NeoX rotate_half convention:
//   out[:D/2]  = x[:D/2]*cos(t) - x[D/2:]*sin(t)
//   out[D/2:]  = x[D/2:]*cos(t) + x[:D/2]*sin(t)
// with theta_j = pos * base^(-2j/D). X is (T, H*D), row-major per token.
Tensor rope(const Tensor& X, uint32_t n_heads, uint32_t pos_offset, float base)
{
    const uint32_t T = X.rows();
    const uint32_t D = X.cols() / n_heads;      // head dim
    const uint32_t half = D / 2;
    Tensor Y = X;

    // precompute inverse frequencies once
    std::vector<float> inv_freq(half);
    for (uint32_t j = 0; j < half; j++)
        inv_freq[j] = std::pow(base, -2.0f * (float)j / (float)D);

    for (uint32_t t = 0; t < T; t++) {
        const float pos = (float)(pos_offset + t);
        const float* xr = X.ptr() + (size_t)t * X.cols();
        float* yr = &Y[(size_t)t * Y.cols()];
        for (uint32_t h = 0; h < n_heads; h++, xr += D, yr += D) {
            for (uint32_t j = 0; j < half; j++) {
                const float ang = pos * inv_freq[j];
                const float c = std::cos(ang), s = std::sin(ang);
                const float a = xr[j], b = xr[j + half];
                yr[j]         = a * c - b * s;
                yr[j + half] = b * c + a * s;
            }
        }
    }
    return Y;
}

// Scaled dot-product attention over explicit (T, H, Dh) tensors.
Tensor sdpa(const Tensor& Q, const Tensor& K, const Tensor& V, bool causal)
{
    // Tq may differ from Tk (cache decode attends L keys with one query);
    // heads and head dims must match.
    assert(Q.shape.size() == 3 && K.shape.size() == 3 && V.shape.size() == 3);
    const uint32_t Tq = Q.shape[0], Tk = K.shape[0];
    const uint32_t H = Q.shape[1], Dh = Q.shape[2];
    assert(K.shape[1] == H && K.shape[2] == Dh);
    assert(V.shape[1] == H && V.shape[2] == Dh);
    assert(!causal || Tq <= Tk);   // causal row i maps to key i

    Tensor out({Tq, H, Dh});
    const float scale = 1.0f / std::sqrt((float)Dh);

    std::vector<float> scores(Tk);              // one query row at a time
    for (uint32_t h = 0; h < H; h++) {
        for (uint32_t qi = 0; qi < Tq; qi++) {
            const float* q = Q.ptr() + ((size_t)qi * H + h) * Dh;
            // logits for every key this query may see
            uint32_t visible = causal ? (qi < Tk ? qi + 1 : Tk) : Tk;
            float m = -INFINITY;
            for (uint32_t ki = 0; ki < visible; ki++) {
                const float* k = K.ptr() + ((size_t)ki * H + h) * Dh;
                float s = 0;
                for (uint32_t d = 0; d < Dh; d++) s += q[d] * k[d];
                s *= scale;
                scores[ki] = s;
                if (s > m) m = s;
            }
            // stable softmax over the visible keys only
            float z = 0;
            for (uint32_t ki = 0; ki < visible; ki++) {
                scores[ki] = std::exp(scores[ki] - m);
                z += scores[ki];
            }
            float* o = &out[((size_t)qi * H + h) * Dh];
            for (uint32_t d = 0; d < Dh; d++) o[d] = 0;
            for (uint32_t ki = 0; ki < visible; ki++) {
                const float p = scores[ki] / z;
                const float* v = V.ptr() + ((size_t)ki * H + h) * Dh;
                for (uint32_t d = 0; d < Dh; d++) o[d] += p * v[d];
            }
        }
    }
    return out;
}

void KVCache::init(uint32_t heads, uint32_t dim, uint32_t cap)
{
    n_heads = heads; head_dim = dim; capacity = cap; len = 0;
    k.assign((size_t)cap * heads * dim, 0.f);
    v = k;
}

void KVCache::append(const Tensor& Kt, const Tensor& Vt)
{
    // one token: (1, H, Dh)
    assert(Kt.shape.size() == 3 && Kt.shape[0] == 1);
    assert(Kt.shape[1] == n_heads && Kt.shape[2] == head_dim);
    assert(Vt.shape == Kt.shape);
    assert(len < capacity);
    std::memcpy(&k[(size_t)len * n_heads * head_dim], Kt.ptr(),
                (size_t)n_heads * head_dim * sizeof(float));
    std::memcpy(&v[(size_t)len * n_heads * head_dim], Vt.ptr(),
                (size_t)n_heads * head_dim * sizeof(float));
    len++;
}

Tensor KVCache::K() const
{
    Tensor out({len, n_heads, head_dim});
    std::memcpy(out.ptr(), k.data(), (size_t)len * n_heads * head_dim * sizeof(float));
    return out;
}

Tensor KVCache::V() const
{
    Tensor out({len, n_heads, head_dim});
    std::memcpy(out.ptr(), v.data(), (size_t)len * n_heads * head_dim * sizeof(float));
    return out;
}

// silu(x) = x * sigmoid(x)
static inline float silu_scalar(float x)
{
    return x / (1.0f + std::exp(-x));
}

Tensor swiglu(const Tensor& gate, const Tensor& up)
{
    assert(gate.shape == up.shape);
    Tensor Y(gate.shape);
    for (size_t i = 0; i < gate.numel(); i++)
        Y[i] = silu_scalar(gate[i]) * up[i];
    return Y;
}

/* ---- quantization --------------------------------------------------------*/

QLinearInt8 QLinearInt8::from_linear(const Linear& L)
{
    QLinearInt8 q;
    q.out_features = L.W.rows();
    q.in_features  = L.W.cols();
    q.b            = L.b;      // bias stays fp32, added exactly
    q.w.resize((size_t)q.out_features * q.in_features);
    q.scales.resize(q.out_features);

    for (uint32_t r = 0; r < q.out_features; r++) {
        const float* row = L.W.ptr() + (size_t)r * q.in_features;
        float amax = 0;
        for (uint32_t c = 0; c < q.in_features; c++)
            amax = std::fmax(amax, std::fabs(row[c]));
        const float scale = amax / 127.0f;
        q.scales[r] = scale;
        int8_t* qr = &q.w[(size_t)r * q.in_features];
        for (uint32_t c = 0; c < q.in_features; c++) {
            float v = scale != 0 ? std::nearbyint(row[c] / scale) : 0.f;
            if (v > 127) v = 127;
            if (v < -127) v = -127;   // keep symmetric range
            qr[c] = (int8_t)v;
        }
    }
    return q;
}

Tensor QLinearInt8::forward(const Tensor& X) const
{
    const uint32_t M = X.rows();
    Tensor Y({M, out_features});
    for (uint32_t i = 0; i < M; i++) {
        const float* xr = X.ptr() + (size_t)i * in_features;
        for (uint32_t r = 0; r < out_features; r++) {
            const int8_t* wr = &w[(size_t)r * in_features];
            // weight-only quantization: activations stay fp32, each int8
            // weight is dequantized on the fly, accumulation in fp32.
            float acc = 0;
            for (uint32_t c = 0; c < in_features; c++)
                acc += xr[c] * (float)wr[c];
            Y[(size_t)i * out_features + r] =
                acc * scales[r] + b[r];          // fp32 bias applied exactly
        }
    }
    return Y;
}

/* ---- sampling -------------------------------------------------------------*/

uint32_t Sampler::sample(const Tensor& logits) const
{
    if (!rng_ready_) {
        rng_.seed(seed);
        rng_ready_ = true;      // lazily seeded once per Sampler instance
    }

    const uint32_t V = (uint32_t)logits.data.size();
    std::vector<float> p(logits.data.begin(), logits.data.end());

    if (temperature > 0.f)
        for (auto& x : p) x /= temperature;

    // top-k: keep only the k largest logits
    if (top_k > 0 && top_k < V) {
        std::vector<uint32_t> idx(V);
        for (uint32_t i = 0; i < V; i++) idx[i] = i;
        std::nth_element(idx.begin(), idx.begin() + top_k - 1, idx.end(),
                         [&](uint32_t a, uint32_t b) { return p[a] > p[b]; });
        const float cutoff = p[idx[top_k - 1]];
        for (uint32_t i = 0; i < V; i++)
            if (p[i] < cutoff) p[i] = -INFINITY;
    }

    // top-p: smallest prefix of sorted probs with mass >= top_p
    if (top_p < 1.0f) {
        std::vector<float> sorted(p);
        std::sort(sorted.begin(), sorted.end(), std::greater<float>());
        float sum = 0;
        for (float x : sorted) sum += x;
        float csum = 0;
        uint32_t kept = 0;
        for (; kept < V; kept++) {
            csum += sorted[kept];
            if (csum >= top_p * sum) break;
        }
        const float cutoff = sorted[kept];
        for (uint32_t i = 0; i < V; i++)
            if (p[i] < cutoff) p[i] = -INFINITY;
    }

    if (temperature <= 0.f) {                 // greedy over survivors
        uint32_t best = 0;
        for (uint32_t i = 1; i < V; i++)
            if (p[i] > p[best]) best = i;
        return best;
    }

    // softmax over survivors, inverse-CDF draw
    float m = -INFINITY;
    for (float x : p) m = std::fmax(m, x);
    float z = 0;
    for (auto& x : p) {
        x = (x == -INFINITY) ? 0.f : std::exp(x - m);
        z += x;
    }
    std::uniform_real_distribution<float> u(0.f, 1.f);
    const float r = u(rng_) * z;
    float c = 0;
    for (uint32_t i = 0; i < V; i++) {
        c += p[i];
        if (r <= c || i == V - 1) return i;
    }
    return V - 1;
}

}  // namespace ti
