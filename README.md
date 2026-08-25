# tinyinfer-cpu — a minimal C++17 inference engine core

Row-major float32 tensors, validated matmul kernels, fused linear layers,
and an MLP forward pass with bit-exact weight serialization — every kernel
checked against a trivially-correct reference before any benchmarking is
believed. CPU-only, OpenMP-parallel where it wins, honest about where it
doesn't.

## Layout

```
include/tensor.hpp     Tensor struct + kernel/model API
src/tensor.cpp         naive & register-blocked OpenMP matmul, relu/gelu,
                       stable softmax, RMSNorm, Linear, fused bias+relu,
                       MLP, flat-binary serialization
tests/test_infer.cpp   correctness suite (16 checks)
bench/bench_matmul.cpp GFLOP/s table + model throughput
```

## Build & run

```sh
make test     # correctness suite (16 checks)
make bench    # GFLOP/s: naive vs tiled+parallel; MLP fwd/s
make clean
```

Requires g++ ≥ 12 with OpenMP. Nothing else.

## Kernels

- **`matmul_naive`** — i-k-j loop order: the inner loop sweeps B's row `k`
  contiguously while the accumulator stays in a register. Already beats the
  textbook k-i-j ordering on modern compilers.
- **`matmul_tiled`** — the version that actually wins:
  1. two output rows per pass (register blocking): each loaded element of B
     does double duty, halving B traffic;
  2. K-blocking so A/B slices stay cache-resident;
  3. OpenMP over row pairs with the canonical bound form (`ii < M - 1`,
     odd-M tail handled serially) — each thread owns disjoint C rows, no
     atomics or reductions.
- **`softmax_lastdim`** — max-subtraction before exp; stays finite at inputs
  ≈1000 where the naive form overflows to inf.
- **`rmsnorm_lastdim`** — `y = x / sqrt(mean(x²)+ε) · w`, ε = 1e-5.
- **`gelu`** — tanh approximation; fp32 saturation means gelu(−10)
  evaluates to exactly 0.0 (asserted by tolerance, not sign).
- **`linear_bias_relu`** — bias folded into the dot-product accumulator,
  ReLU applied in the same pass over outputs: one pass total.

## Weight serialization

Flat binary, little-endian, tightly packed:

```text
file:   "TIIN" | u32 d_model | u32 d_hidden | u32 vocab | tensor*
tensor: "TNSR" | u32 ndims | u32 dim[ndims] | f32 payload[numel]
```

All reads are validated; truncated or corrupt files throw
`std::runtime_error` instead of tripping asserts (which vanish under
`-DNDEBUG`) or segfaulting. Loaded tensors are cross-checked against the
header dims. Save→load round-trips are bit-exact (`memcmp`, not tolerance).

## Correctness suite (16 checks)

Every kernel is compared against an obviously-correct triple-loop
reference within stated tolerances:

matmul naive/tiled(block=16/64) vs reference · tiled at non-multiple-of-K
block sizes · relu zeroes negatives only · softmax rows sum to 1 at ~1000-
magnitude inputs · softmax preserves argmax · gelu at 0/+10/−10 with
saturation tolerance · RMSNorm unit RMS per row · linear matches hand
computation (W is `(out,in)`!) · fused bias+relu ≡ forward-then-relu ·
dims survive save/load · logits bit-exact after round-trip · argmax class
in range · deterministic forward · truncated model file throws instead of
crashing.

## Benchmarks

Measured with `make bench` (fixed seed, warm-up, volatile sink so nothing
is elided). Absolute GF/s varies with machine load — reproduce locally.

| N    | naive GF/s | tiled GF/s | speedup                                  |
|------|-----------|------------|------------------------------------------|
| 256  | ~1.1      | ~1.1       | ~1.0× — thread launch overhead dominates |
| 512  | ~1.3      | ~2.2       | ~1.7×                                    |
| 1024 | ~1.1      | ~5.0       | ~4.6×                                    |

On an unloaded box the same code measures roughly 5–6 naive vs 25–30 tiled
at N=1024. The small-N regression is a finding, not a bug: parallelizing
tiny GEMMs costs more than tiling saves (grain-size problem).

Model throughput: MLP d=512, h=2048, vocab=1000, batch=64 also printed by
`make bench`.

## Design notes

- Tensors are contiguous row-major float32 vectors; no strides, no dtype
  zoo. Inference engines live and die by layout simplicity.
- Validation-first discipline: no kernel is benchmarked until it matches
  its reference. The fused path is tested for exact equivalence against
  the unfused one.
- `argmax_class` takes exactly one row (asserted); batched callers slice
  explicitly rather than silently argmaxing row 0.
