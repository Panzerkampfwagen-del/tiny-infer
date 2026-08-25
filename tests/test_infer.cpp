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

int main(void)
{
    test_matmul();
    test_ops();
    test_linear_fusion();
    test_model_and_io();

    if (g_failures) { printf("\n%d FAILURE(S)\n", g_failures); return 1; }
    printf("\nALL TESTS PASSED\n");
    return 0;
}
