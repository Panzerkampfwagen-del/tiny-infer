/*
 * bench_matmul -- naive vs tiled GFLOP/s, plus end-to-end model throughput.
 * naive is single-threaded; tiled uses OpenMP (row-parallel). Numbers are
 * whatever this machine is at run time -- reproduce with `make bench`.
 */
#include "../include/tensor.hpp"

#include <chrono>
#include <cstdio>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ti;

static double now_ms(void)
{
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

int main(void)
{
    std::mt19937_64 rng(1);
    std::normal_distribution<float> dist;

#ifdef _OPENMP
    printf("OpenMP threads: %d\n", omp_get_max_threads());
#endif
    printf("%8s %10s %12s %12s\n", "N", "reps", "naive GF/s", "tiled GF/s");
    for (uint32_t N : {256u, 512u, 1024u}) {
        Tensor A({N, N}), B({N, N});
        for (size_t i = 0; i < A.numel(); i++) A[i] = dist(rng);
        for (size_t i = 0; i < B.numel(); i++) B[i] = dist(rng);

        const double flops = 2.0 * N * N * N;
        int reps = (int)(2e9 / flops); if (reps < 1) reps = 1;

        double t0 = now_ms();
        volatile float sink1 = 0;
        for (int r = 0; r < reps; r++) { Tensor C = matmul_naive(A, B); sink1 += C[0]; }
        double t_naive = now_ms() - t0;

        t0 = now_ms();
        volatile float sink2 = 0;
        for (int r = 0; r < reps; r++) { Tensor C = matmul_tiled(A, B); sink2 += C[0]; }
        double t_tiled = now_ms() - t0;

        printf("%8u %10d %12.2f %12.2f\n", N, reps,
               reps * flops / (t_naive * 1e6), reps * flops / (t_tiled * 1e6));
    }

    // model throughput
    MLP m = MLP::random_init(512, 2048, 1000, 9);
    Tensor X({64, 512});
    for (size_t i = 0; i < X.numel(); i++) X[i] = dist(rng);
    double t0 = now_ms();
    const int reps = 20;
    for (int r = 0; r < reps; r++) { Tensor L = m.forward(X); }
    double dt = now_ms() - t0;
    printf("MLP d=512 h=2048 vocab=1000 batch=64: %.3f ms/forward (%.0f fwd/s)\n",
           dt / reps, reps * 1000.0 / dt);

    // int8 weight-only quantized linear vs fp32 Linear, same shapes
    {
        const uint32_t OUT = 2048, IN = 512;
        Linear L; L.W = Tensor({OUT, IN}); L.b = Tensor({OUT});
        for (size_t i = 0; i < L.W.numel(); i++) L.W[i] = dist(rng);
        for (size_t i = 0; i < L.b.numel(); i++) L.b[i] = dist(rng);
        Tensor Xi({64, IN});
        for (size_t i = 0; i < Xi.numel(); i++) Xi[i] = dist(rng);

        QLinearInt8 q = QLinearInt8::from_linear(L);
        const double flops = (double)64 * OUT * IN * 2.0;

        t0 = now_ms();
        volatile float s1 = 0;
        for (int r = 0; r < reps; r++) { Tensor Yf = L.forward(Xi); s1 += Yf[0]; }
        double t_fp = now_ms() - t0;

        t0 = now_ms();
        volatile float s2 = 0;
        for (int r = 0; r < reps; r++) { Tensor Yq = q.forward(Xi); s2 += Yq[0]; }
        double t_q = now_ms() - t0;

        double num = 0, den = 0;
        { Tensor Yf = L.forward(Xi), Yq = q.forward(Xi);
          for (size_t i = 0; i < Yf.numel(); i++) {
              double d = Yq[i] - Yf[i]; num += d * d; den += (double)Yf[i] * Yf[i]; } }
        printf("int8-linear out=%u in=%u batch=64: fp32 %.1f GFLOP/s | int8-w-only %.1f GFLOP/s | weights %zu KB vs %zu KB | rel RMS err %.4f\n",
               OUT, IN, reps * flops / (t_fp * 1e6), reps * flops / (t_q * 1e6),
               q.weight_bytes() / 1024, L.W.numel() * 4 / 1024,
               (float)std::sqrt(num / den));
    }
    return 0;
}
