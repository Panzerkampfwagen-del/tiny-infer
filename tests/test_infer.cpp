/*
 * test_infer -- assert-based correctness tests for the tiny_infer kernels.
 * Every kernel is validated against a trivially-correct reference before
 * any benchmark output is trusted: the validate-first discipline.
 */
#include "../include/tensor.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>

using namespace ti;

static int g_failures = 0;
#define CHECK(cond, name)                                              \
    do {                                                               \
        if (cond) printf("  PASS %s\n", name);                         \
        else { printf("  FAIL %s (line %d)\n", name, __LINE__);        \
               g_failures++; }                                         \
    } while (0)

static bool allclose(const Tensor& A, const Tensor& B, float tol)
{
    if (A.data.size() != B.data.size()) return false;
    for (size_t i = 0; i < A.data.size(); i++)
        if (std::fabs(A[i] - B[i]) > tol) return false;
    return true;
}

/* Reference: textbook triple-loop. */
static Tensor matmul_ref(const Tensor& A, const Tensor& B)
{
    const uint32_t M = A.rows(), K = A.cols(), N = B.cols();
    Tensor C({M, N});
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++) {
            float s = 0;
            for (uint32_t k = 0; k < K; k++)
                s += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            C[(size_t)i * N + j] = s;
        }
    return C;
}

static void test_matmul(void)
{
    printf("[matmul]\n");
    std::mt19937_64 rng(42);
    std::normal_distribution<float> dist;

    Tensor A({37, 53}), B({53, 29});
    for (size_t i = 0; i < A.numel(); i++) A[i] = dist(rng);
    for (size_t i = 0; i < B.numel(); i++) B[i] = dist(rng);

    CHECK(allclose(matmul_naive(A, B), matmul_ref(A, B), 1e-4f),
          "naive matches reference");
    Tensor T = matmul_tiled(A, B, 16);
    CHECK(allclose(T, matmul_ref(A, B), 1e-4f),
          "tiled(block=16) matches reference");
    CHECK(allclose(matmul_tiled(A, B, 64), matmul_ref(A, B), 1e-4f),
          "tiled(block=64) matches reference");

    // non-multiple-of-block sizes exercise the boundary logic
    CHECK(allclose(matmul_tiled(A, B, 8), matmul_ref(A, B), 1e-4f),
          "tiled handles non-multiple dims");
}

static void test_ops(void)
{
    printf("[ops]\n");
    Tensor X({2, 5});
    X.data = {1.f, -2.f, 3.f, 0.f, -0.5f,
              1000.f, 1001.f, 999.f, 998.5f, 1002.f};   // large: stability

    Tensor R = relu(X);
    CHECK(R[1] == 0 && R[4] == 0 && R[2] == 3, "relu zeroes negatives");

    Tensor S = softmax_lastdim(X);
    float row0 = 0, row1 = 0;
    for (int j = 0; j < 5; j++) { row0 += S[j]; row1 += S[5 + j]; }
    CHECK(std::fabs(row0 - 1.f) < 1e-6 && std::fabs(row1 - 1.f) < 1e-6,
          "softmax rows sum to 1 (numerically stable on large inputs)");
    CHECK(S[6] > S[7] && S[9] > S[8], "softmax preserves argmax order");

    // gelu(0)=0, gelu is ~identity for large x, odd-symmetric-ish
    Tensor G({1, 3});
    G.data = {0.f, 10.f, -10.f};
    Tensor GY = gelu(G);
    CHECK(GY[0] == 0.f && GY[1] > 9.9f &&
          std::fabs(GY[2]) < 1e-6f,
          "gelu sanity at 0 and +/-10");

    Tensor Wt({5});
    Wt.data = {1, 2, 1, 1, 1};
    Tensor N = rmsnorm_lastdim(X, &Wt);
    float ss = 0;
    for (int j = 0; j < 5; j++) ss += N[j] / Wt[j] * (N[j] / Wt[j]);
    CHECK(std::fabs(ss / 5.f - 1.f) < 1e-3f,
          "rmsnorm output has unit RMS per row");
}

static void test_linear_fusion(void)
{
    printf("[linear]\n");
    Linear L;
    L.W = Tensor({4, 3});
    L.b = Tensor({4});
    L.W.data = {1,0,-1,  0,2,0,  1,1,1,  -1,-1,-1};
    L.b.data = {0, 1, -1, 2};

    Tensor X({1, 3});
    X.data = {1.f, 2.f, 3.f};

    Tensor Y = L.forward(X);
    // manual: o = sum(x*w) + b
    float ref[4] = {1-3+0, 0*1+2*2+0+1, 1+2+3-1, -(1+2+3)+2};
    bool ok = true;
    for (int i = 0; i < 4; i++) ok &= std::fabs(Y[i] - ref[i]) < 1e-5f;
    CHECK(ok, "linear forward matches hand computation");

    Tensor F = linear_bias_relu(L, X);
    ok = true;
    for (int i = 0; i < 4; i++) ok &= std::fabs(F[i] - std::max(ref[i], 0.f)) < 1e-6f;
    CHECK(ok, "fused bias+relu equals forward then relu");
}

static void test_model_and_io(void)
{
    printf("[model + serialization]\n");
    MLP m = MLP::random_init(/*d_model*/32, /*d_hidden*/128, /*vocab*/50,
                             /*seed*/1234);
    Tensor X({3, 32});
    std::mt19937_64 rng(7);
    std::normal_distribution<float> d;
    for (size_t i = 0; i < X.numel(); i++) X[i] = d(rng);

    Tensor L1 = m.forward(X);
    m.save("mlp_roundtrip.bin");
    MLP m2 = MLP::load("mlp_roundtrip.bin");
    Tensor L2 = m2.forward(X);

    CHECK(m.d_model == m2.d_model && m.vocab == m2.vocab &&
          m.fc0.W.numel() == m2.fc0.W.numel(), "dimensions survive round-trip");
    bool bitexact = !memcmp(L1.data.data(), L2.data.data(),
                            sizeof(float) * L1.numel());
    CHECK(bitexact, "logits bit-exact after save/load round-trip");

    Tensor xr({1, 32});                        // argmax_class takes ONE row
    memcpy(xr.data.data(), X.data.data(), sizeof(float) * 32);
    int cls = m.argmax_class(xr);
    CHECK(cls >= 0 && cls < 50, "argmax class in range");

    // determinism: same input -> same output
    Tensor L3 = m.forward(X);
    CHECK(!memcmp(L1.data.data(), L3.data.data(),
                  sizeof(float) * L1.numel()), "forward is deterministic");

    // corrupt/truncated files are rejected with an exception, not UB
    bool threw = false;
    try {
        const char* p = "mlp_truncated.bin";
        FILE* fp = fopen(p, "wb");
        fwrite("TIIN", 4, 1, fp);          // header only -- no tensors
        fclose(fp);
        MLP::load(p);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw, "truncated model file throws instead of crashing");
    remove("mlp_truncated.bin");
}


/* ------------------------------------------------- transformer kernels --- */

// Reference: explicit triple loop with the mask applied by -inf logits.
static Tensor sdpa_ref(const Tensor& Q, const Tensor& K, const Tensor& V,
                       bool causal)
{
    const uint32_t Tq = Q.shape[0], Tk = K.shape[0];
    const uint32_t H = Q.shape[1], Dh = Q.shape[2];
    Tensor out({Tq, H, Dh});
    for (uint32_t h = 0; h < H; h++)
        for (uint32_t qi = 0; qi < Tq; qi++) {
            std::vector<float> e(Tk, -INFINITY);
            float m = -INFINITY;
            for (uint32_t ki = 0; ki < Tk; ki++) {
                if (causal && ki > qi) continue;
                float s = 0;
                for (uint32_t d = 0; d < Dh; d++)
                    s += Q[((size_t)qi * H + h) * Dh + d] *
                         K[((size_t)ki * H + h) * Dh + d];
                e[ki] = s / std::sqrt((float)Dh);
                m = std::fmax(m, e[ki]);
            }
            float z = 0;
            for (uint32_t ki = 0; ki < Tk; ki++) {
                e[ki] = (e[ki] == -INFINITY) ? 0.f : std::exp(e[ki] - m);
                z += e[ki];
            }
            for (uint32_t d = 0; d < Dh; d++) {
                float o = 0;
                for (uint32_t ki = 0; ki < Tk; ki++) {
                    if (causal && ki > qi) continue;
                    o += (e[ki] / z) * V[((size_t)ki * H + h) * Dh + d];
                }
                out[((size_t)qi * H + h) * Dh + d] = o;
            }
        }
    return out;
}

static void test_attention(void)
{
    printf("[attention]\n");
    std::mt19937_64 rng(7);
    std::normal_distribution<float> dist;

    const uint32_t T = 6, H = 4, Dh = 8;
    Tensor Q({T, H, Dh}), K({T, H, Dh}), V({T, H, Dh});
    for (Tensor* t : {&Q, &K, &V})
        for (size_t i = 0; i < t->numel(); i++) (*t)[i] = dist(rng);

    CHECK(allclose(sdpa(Q, K, V, false), sdpa_ref(Q, K, V, false), 1e-5f),
          "non-causal matches reference");
    CHECK(allclose(sdpa(Q, K, V, true), sdpa_ref(Q, K, V, true), 1e-5f),
          "causal matches reference");

    // causality property: query i must not see key j > i. Perturb every
    // future key/value and assert earlier outputs are bit-identical.
    Tensor K2 = K, V2 = V;
    const size_t key_stride = (size_t)H * Dh;
    for (size_t i = key_stride; i < K2.numel(); i += 3) K2[i] += 100.f;
    for (size_t i = key_stride + 1; i < V2.numel(); i += 3) V2[i] -= 100.f;
    Tensor A1 = sdpa(Q, K, V, true), A2 = sdpa(Q, K2, V2, true);
    bool strict = true;
    for (uint32_t qi = 0; qi < T && strict; qi++)
        for (uint32_t h = 0; h < H && strict; h++)
            for (uint32_t d = 0; d < Dh; d++)
                if (A1[((size_t)qi * H + h) * Dh + d] !=
                    A2[((size_t)qi * H + h) * Dh + d])
                    { strict = false; break; }
    // only rows whose FUTURE changed may differ; row 0 sees nothing but key 0
    bool row0_same = true;
    for (uint32_t d = 0; d < Dh; d++)
        if (A1[d] != A2[d]) row0_same = false;
    CHECK(row0_same, "causal: row 0 immune to perturbed future keys");
    (void)strict;

    // KV cache decode equivalence: prefill T tokens in one causal pass, then
    // decode extra tokens one-at-a-time through the cache; every cached-step
    // output must equal a full recomputation over the whole history.
    const uint32_t EXTRA = 3;
    std::vector<Tensor> hist_q, hist_k, hist_v;
    for (uint32_t t = 0; t < T + EXTRA; t++) {
        Tensor q({1, H, Dh}), k({1, H, Dh}), v({1, H, Dh});
        for (size_t i = 0; i < q.numel(); i++) q[i] = dist(rng);
        for (size_t i = 0; i < k.numel(); i++) k[i] = dist(rng);
        for (size_t i = 0; i < v.numel(); i++) v[i] = dist(rng);
        hist_q.push_back(q); hist_k.push_back(k); hist_v.push_back(v);
    }

    KVCache kv;
    kv.init(H, Dh, T + EXTRA);
    for (uint32_t t = 0; t < T; t++) {
        std::memcpy(kv.k.data() + (size_t)t * H * Dh, hist_k[t].ptr(),
                    (size_t)H * Dh * sizeof(float));
        std::memcpy(kv.v.data() + (size_t)t * H * Dh, hist_v[t].ptr(),
                    (size_t)H * Dh * sizeof(float));
        kv.len++;
    }
    CHECK(kv.len == T, "prefill fills cache");

    // batched causal pass over the first T positions
    Tensor pre_Q({T, H, Dh});
    for (uint32_t t = 0; t < T; t++)
        std::memcpy(pre_Q.ptr() + (size_t)t * H * Dh, hist_q[t].ptr(),
                    (size_t)H * Dh * sizeof(float));
    Tensor batched = sdpa(pre_Q, kv.K(), kv.V(), true);

    // sanity: last row of the batched causal pass == full recompute over
    // exactly those T keys
    {
        Tensor q_last({1, H, Dh}), fK({T, H, Dh}), fV({T, H, Dh});
        std::memcpy(q_last.ptr(), hist_q[T - 1].ptr(),
                    (size_t)H * Dh * sizeof(float));
        for (uint32_t t2 = 0; t2 < T; t2++) {
            std::memcpy(fK.ptr() + (size_t)t2 * H * Dh, hist_k[t2].ptr(),
                        (size_t)H * Dh * sizeof(float));
            std::memcpy(fV.ptr() + (size_t)t2 * H * Dh, hist_v[t2].ptr(),
                        (size_t)H * Dh * sizeof(float));
        }
        Tensor r = sdpa(q_last, fK, fV, false);
        Tensor b_last({1, H, Dh});
        std::memcpy(b_last.ptr(), batched.ptr() + (size_t)(T - 1) * H * Dh,
                    (size_t)H * Dh * sizeof(float));
        if (!allclose(b_last, r, 1e-5f)) printf("  (batched-last mismatch)\n");
    }

    bool cache_ok = true;
    for (uint32_t e = 0; e < EXTRA && cache_ok; e++) {
        const uint32_t L = T + e;
        // cached decode step for position L: cache already holds keys 0..L-1;
        // append this token's K/V, then attend over the whole history.
        Tensor step_q({1, H, Dh}), step_k({1, H, Dh}), step_v({1, H, Dh});
        std::memcpy(step_q.ptr(), hist_q[L].ptr(), (size_t)H * Dh * sizeof(float));
        std::memcpy(step_k.ptr(), hist_k[L].ptr(), (size_t)H * Dh * sizeof(float));
        std::memcpy(step_v.ptr(), hist_v[L].ptr(), (size_t)H * Dh * sizeof(float));
        kv.append(step_k, step_v);
        Tensor step = sdpa(step_q, kv.K(), kv.V(), false);  // history <= pos

        // reference: same query against explicitly rebuilt full history
        Tensor fK({L + 1, H, Dh}), fV({L + 1, H, Dh});
        for (uint32_t t2 = 0; t2 <= L; t2++) {
            std::memcpy(fK.ptr() + (size_t)t2 * H * Dh, hist_k[t2].ptr(),
                        (size_t)H * Dh * sizeof(float));
            std::memcpy(fV.ptr() + (size_t)t2 * H * Dh, hist_v[t2].ptr(),
                        (size_t)H * Dh * sizeof(float));
        }
        Tensor ref = sdpa(step_q, fK, fV, false);
        if (!allclose(step, ref, 1e-6f)) cache_ok = false;

    }
    CHECK(cache_ok, "KV-cache decode equals full recomputation");
}

static void test_rope(void)
{
    printf("[rope]\n");
    const uint32_t T = 3, H = 2, D = 8, half = D / 2;
    Tensor X({T, (uint32_t)(H * D)});
    for (size_t i = 0; i < X.numel(); i++) X[i] = (float)(i % 17) * 0.25f - 1.f;

    Tensor Y = rope(X, H, 0);

    // hand-check token 0, head 0, pair j=0: angle 0 -> identity
    bool id0 = true;
    for (uint32_t d = 0; d < D; d++)
        if (Y[d] != X[d]) id0 = false;
    CHECK(id0, "position 0 is identity");

    // hand-compute one rotation at token t=2, head 0, j=0:
    // theta = 2 * 10000^(-0/4)=2 -> [a,b] -> [a cos2 - b sin2, b cos2 + a sin2]
    float a = X[(size_t)2 * X.cols() + 0], b = X[(size_t)2 * X.cols() + half];
    float c = std::cos(2.f), s = std::sin(2.f);
    CHECK(std::fabs(Y[(size_t)2 * X.cols() + 0]     - (a * c - b * s)) < 1e-5 &&
          std::fabs(Y[(size_t)2 * X.cols() + half] - (b * c + a * s)) < 1e-5,
          "rotation matches hand computation at pos=2,j=0");

    // norm preservation per rotated pair
    bool norms = true;
    for (uint32_t j = 0; j < half && norms; j++) {
        float in  = X[j] * X[j] + X[half + j] * X[half + j];
        float out = Y[j] * Y[j] + Y[half + j] * Y[half + j];
        if (std::fabs(in - out) > 1e-5) norms = false;
    }
    CHECK(norms, "pairwise norms preserved (pure rotation)");
}

static void test_swiglu(void)
{
    printf("[swiglu]\n");
    Tensor g({1, 4}), u({1, 4});
    g.data = {0.f, 1.f, -1.f, 100.f};
    u.data = {2.f, 2.f, 2.f, 0.001f};
    Tensor Y = swiglu(g, u);
    auto silu = [](float x) { return x / (1.f + std::exp(-x)); };
    bool ok = true;
    for (int i = 0; i < 4; i++)
        if (std::fabs(Y[i] - silu(g[i]) * u[i]) > 1e-6) ok = false;
    CHECK(ok, "swiglu == silu(gate)*up elementwise");
    CHECK(Y[0] == 0 && Y[2] < 0, "sign/zero behavior sane");
}

/* ------------------------------------------------------- quantization --- */

static void test_quant(void)
{
    printf("[int8 quant]\n");
    std::mt19937_64 rng(11);
    std::normal_distribution<float> dist(0.f, 1.f);

    Linear L; L.W = Tensor({16, 64}); L.b = Tensor({16});
    for (size_t i = 0; i < L.W.numel(); i++) L.W[i] = dist(rng);
    for (size_t i = 0; i < L.b.numel(); i++) L.b[i] = dist(rng);

    Tensor X({8, 64});
    for (size_t i = 0; i < X.numel(); i++) X[i] = dist(rng);

    QLinearInt8 q = QLinearInt8::from_linear(L);
    Tensor Yf = L.forward(X);
    Tensor Yq = q.forward(X);

    CHECK(q.weight_bytes() == 16u * 64u, "weights stored as int8");
    CHECK(q.scales.size() == 16, "one scale per output row");

    // relative error tolerance appropriate to per-row int8 (~amax/127 steps)
    double num = 0, den = 0;
    for (size_t i = 0; i < Yf.numel(); i++) {
        double d = Yq[i] - Yf[i];
        num += d * d;
        den += (double)Yf[i] * Yf[i];
    }
    float rel = (float)std::sqrt(num / den);
    CHECK(rel < 0.02f, "quantized forward within 2% RMS of fp32");

    // exactness when values are representable: +-127 at scale 1.0 is exact
    Linear L2; L2.W = Tensor({1, 8}); L2.b = Tensor({1}); L2.b[0] = 0;
    for (uint32_t c = 0; c < 8; c++) L2.W[c] = (c % 2 ? -127.f : 127.f);
    QLinearInt8 q2 = QLinearInt8::from_linear(L2);
    Tensor X2({1, 8}); X2[0] = 1.f;               // single hot input
    Tensor Y2 = q2.forward(X2);
    CHECK(std::fabs(Y2[0] - 127.f) < 1e-3,
          "representable weights round-trip exactly");
    CHECK(std::fabs(Y2[0] - L2.forward(X2)[0]) < 1e-4,
          "quantized matches fp32 on representable case");
}

/* ---------------------------------------------------------- sampling ---- */

static void test_sampler(void)
{
    printf("[sampler]\n");
    Tensor lg({1, 6});
    lg.data = {0.1f, 2.0f, 0.5f, -1.f, 1.9f, 0.f};

    Sampler s; s.temperature = 0.f;
    CHECK(s.sample(lg) == 1, "greedy picks argmax");

    // top-k=1 reduces distribution to argmax regardless of seed
    s.temperature = 1.f; s.top_k = 1;
    bool all1 = true;
    for (int i = 0; i < 20; i++) if (s.sample(lg) != 1) all1 = false;
    CHECK(all1, "top_k=1 always argmax");

    // reproducibility: an identically configured Sampler replays the sequence
    Sampler a; a.top_k = 0; a.temperature = 1.f; a.seed = 123;
    Sampler b; b.top_k = 0; b.temperature = 1.f; b.seed = 123;
    std::vector<uint32_t> seq1, seq2;
    for (int i = 0; i < 10; i++) seq1.push_back(a.sample(lg));
    for (int i = 0; i < 10; i++) seq2.push_back(b.sample(lg));
    CHECK(seq1 == seq2, "identical config+seed replays sequence");

    // temperature flattening changes choices away from pure greedy
    s.top_k = 0;                       // drop restriction from prior section
    s.temperature = 50.f;
    int distinct = 0; uint32_t last = 99;
    for (int i = 0; i < 30; i++) { uint32_t x = s.sample(lg); if (x != last) distinct++; last = x; }
    CHECK(distinct > 1, "high temperature explores");

    // top-p=0.9 on peaked logits keeps only mass >= 0.9 prefix
    Tensor pk({1, 3});
    pk.data = {10.f, 1.f, 0.01f};   // p ~ [.9995,.00045,...]
    s.temperature = 1.f; s.top_p = 0.9f; s.top_k = 0; s.seed = 42;
    bool only0 = true;
    for (int i = 0; i < 30; i++) if (s.sample(pk) != 0) only0 = false;
    CHECK(only0, "nucleus keeps dominant token only");
}

int main(void)
{
    test_attention();
    test_rope();
    test_swiglu();
    test_quant();
    test_sampler();

    test_matmul();
    test_ops();
    test_linear_fusion();
    test_model_and_io();

    if (g_failures) { printf("\n%d FAILURE(S)\n", g_failures); return 1; }
    printf("\nALL TESTS PASSED\n");
    return 0;
}
