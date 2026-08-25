# scoped_down/tiny_infer

A minimal C++17 inference engine: row-major float32 tensors, validated matmul
kernels, fused linear layers, and an MLP forward pass with bit-exact weight
serialization. Mirrors `~/tiny_infer/tiny-infer` at CPU-only undergraduate
scope; every kernel is checked against a trivially-correct reference before
being benchmarked — the original repo's core discipline.

## Layout

```
include/tensor.hpp     Tensor struct + kernel/model API
src/tensor.cpp         kernels: naive & register-blocked OpenMP matmul,
                       relu/gelu, stable softmax, RMSNorm, Linear,
                       fused bias+relu, MLP, flat-binary serialization
tests/test_infer.cpp   correctness tests (all kernels vs reference)
bench/bench_matmul.cpp honest benchmark: GFLOP/s table + model throughput
```

## Build & run

```sh
make test     # correctness suite (16 checks)
make bench    # GFLOP/s: naive vs tiled+parallel; MLP fwd/s
```

## Results (measured with `make bench`; absolute GF/s varies with machine load)

| N   | naive GF/s | tiled GF/s | speedup |
|-----|-----------|------------|---------|
| 256 | 1.1       | 1.1        | ~1.0x — thread launch overhead dominates |
| 512 | 1.3       | 2.2        | ~1.7x   |
| 1024| 1.1       | 5.0        | ~4.6x   |

(On an unloaded box the same code measures ~5–6 naive and ~25–30 tiled at
N=1024; reproduce locally with `make bench`.)

The small-N regression is itself a finding: parallelizing tiny GEMMs costs more
than it saves (grain-size problem). The kernels are register-blocked (two C
rows per pass halves B traffic) plus row-parallel OpenMP.

## Key concepts demonstrated

- Validation-first kernel engineering: every kernel must match a dumb-but-
  obviously-correct reference within tolerance before any benchmarking.
- Cache/register blocking and loop ordering; why ikj beats kji.
- Numerical stability (max-subtraction softmax), RMSNorm math.
- Fusion: bias-add + ReLU in one pass over outputs.
- Bit-exact save/load round-trip as a serialization test.
