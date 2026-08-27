# tinyinfer-cpu — a minimal C++17 LLM inference core

Row-major float32 tensors, validated matmul kernels, fused linear layers,
a Llama-2-style transformer forward pass, real tokenizer-driven text
generation, lossless prompt-lookup speculative decoding, and continuous
batching — every kernel checked against a trivially-correct reference before
any benchmarking is believed. CPU-only, OpenMP-parallel where it wins,
honest about where it doesn't.

## Layout

```
include/tensor.hpp     Tensor struct + kernel/model API
src/tensor.cpp         naive & register-blocked OpenMP matmul, relu/gelu,
                       stable softmax, RMSNorm, SwiGLU, Linear, fused
                       bias+relu, RoPE, causal SDPA attention + KV cache,
                       per-row INT8 quantized linear, seeded sampler,
                       MLP, flat-binary serialization
include/model.hpp      Transformer LM (GQA attention, RoPE, SwiGLU FFN),
                       bf16 weights loader w/ dependency-free JSON header,
                       optional INT8 weight-only runtime
include/generate.hpp   DecodeState (KV-cache stepping), BatchRunner
                       (continuous batching), speculative decoding
include/tokenizer.hpp  SentencePiece BPE encode/decode (llama2.c format)
src/cli_main.cpp       streaming text generation CLI (--spec/--chat/--stats)
tools/export.py        HuggingFace -> our flat bf16 weights + tokenizer.bin
tests/test_infer.cpp   kernel correctness suite (37 checks)
tests/test_model.cpp   system tests: cache==full-recompute, spec-decode
                       losslessness vs greedy, batching == serial decode,
                       BPE round-trips on a synthetic vocab
bench/bench_matmul.cpp GFLOP/s table with arithmetic-intensity + "% of
                       calibrated peak" roofline columns, MLP throughput,
                       int8 comparison
bench/bench_batch.cpp  continuous-batching goodput vs serial decode
```

## Build & run

```sh
make test          # kernel suite (37 checks)
make test-model    # transformer / tokenizer / batching suite
make bench         # GFLOP/s + roofline columns + int8 comparison
make bench-batch   # batching goodput scaling
make clean

# text generation (after exporting weights):
./build/tinyinfer_cli --model tinyllama.bin --tokenizer tokenizer.bin \
    --prompt "Once upon a time" --max-tokens 128 --temperature 0.8 \
    [--dtype int8] [--spec 3 6] [--chat] [--stats]

# export weights (needs python + torch/transformers/sentencepiece/hf_hub):
python3 tools/export.py \
    --model TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T --outdir .
```

Requires g++ ≥ 12 with OpenMP; the engine itself has no dependencies.

## Kernels

- **`matmul_naive`** — i-k-j loop order: the inner loop sweeps B's row `k`
  contiguously while the accumulator stays in a register.
- **`matmul_tiled`** — the version that actually wins:
  1. two output rows per pass (register blocking): each loaded element of B
     does double duty, halving B traffic;
  2. K-blocking so A/B slices stay cache-resident;
  3. OpenMP over row pairs (`ii < M - 1`, odd-M tail handled serially) —
     each thread owns disjoint C rows, no atomics or reductions.
- **`linear_notrans`** — row-dot GEMV over the embedding table for the tied
  LM head.
- **`softmax_lastdim`** — max-subtraction before exp; stays finite at inputs
  ≈1000 where the naive form overflows to inf.
- **`rmsnorm_lastdim`** — `y = x / sqrt(mean(x²)+ε) · w`, ε = 1e-5.
- **`gelu`** — tanh approximation; fp32 saturation means gelu(−10)
  evaluates to exactly 0.0 (asserted by tolerance, not sign).
- **`linear_bias_relu`** — bias folded into the dot-product accumulator,
  ReLU applied in the same pass over outputs: one pass total.
- **`rope`** — rotary position embedding, HF/NeoX `rotate_half` convention:
  pairwise rotation is norm-preserving (asserted), position 0 is identity.
- **`sdpa`** — multi-head scaled dot-product attention over `(T,H,Dh)`
  tensors with optional causal masking; validated against a reference AND
  by a perturbation property test (row i's output is bit-identical when
  only *future* keys change).
- **`KVCache`** — fixed-capacity per-layer K/V buffer; decode steps through
  the cache are checked to equal full recomputation over the whole history.
- **`swiglu`** — `silu(gate) * up`, the LLaMA feed-forward activation.
- **`QLinearInt8`** — per-output-row symmetric INT8 weight quantization with
  FP32 scales; activations stay FP32 and weights dequantize on the fly
  (~2x smaller weights, small measured accuracy cost — see `make bench`).
- **`Sampler`** — greedy / temperature / top-k / top-p (nucleus) over one
  logits row; identically configured samplers replay identical sequences.

## Weight serialization

MLP files keep the original format:

```text
file:   "TIIN" | u32 d_model | u32 d_hidden | u32 vocab | tensor*
tensor: "TNSR" | u32 ndims | u32 dim[ndims] | f32 payload[numel]
```

All reads are validated; truncated or corrupt files throw
`std::runtime_error` instead of tripping asserts (which vanish under
`-DNDEBUG`) or segfaulting. Save→load round-trips are bit-exact (`memcmp`,
not tolerance).

Transformer weights use the exporter's format:

```text
u32 json_len | json{"config":{...}, "tensors":[{name,shape,dtype,offset}]}
| bf16 payload, tightly packed
```

The JSON header is parsed by an in-repo mini recursive-descent parser (no
dependencies); every tensor offset is bounds-checked and all loaded shapes
are cross-checked against the config after load.

## The system layer

- **Transformer LM** — prefill a whole prompt block or decode one token per
  step against per-layer KV caches; offset-aware causal masking;
  **grouped-query attention** (n_kv_heads broadcast across query heads);
  `--dtype int8` swaps every big projection to the existing `QLinearInt8`
  path (~½ memory, fp32 bias-free HF layout).
- **Tokenizer** — byte-fallback SentencePiece BPE in the llama2.c binary
  format (score-float | len-int | bytes per piece), merge-by-score encode,
  `<0xXX>`-aware decode. BOS is context only: its piece is never printed
  and the post-BOS space strips, matching llama2.c usage.
- **Speculative decoding (`--spec N K`, greedy)** — drafts up to K
  continuations of the trailing N-gram found earlier in the history,
  verifies the whole window in ONE forward pass, rolls rejected KV entries
  back (cache length is simply rewound) and adopts the deepest verified
  prediction row. Provably lossless: the suite asserts the emitted sequence
  equals plain greedy decoding token-for-token while committing several
  tokens per verification forward.
- **Continuous batching** — `BatchRunner` prefills each admitted prompt
  immediately (interleaved prefill/decode) then runs batch-wide decode
  rounds under per-sequence max-new-token budgets; batched sequences are
  asserted token-identical to serial execution.
- **CLI** — streams pieces as they are produced; EOS stops generation;
  `--stats` prints decode tok/s; `--chat` gives an interactive REPL.

## Benchmarks

The matmul table carries arithmetic intensity (AI = 2MNK / bytes moved) and
each row's share of peak calibrated from the best tiled measurement, so
small-N rows can be *seen* to be grain-bound rather than bandwidth-bound:

| N    | naive GF/s | tiled GF/s | AI FLOP/B | % cal. peak |
|------|-----------:|-----------:|----------:|------------:|
| 256  | ~5.2       | ~4.2       | 42.7      | ~17%        |
| 512  | ~6.3       | ~15.7      | 85.3      | ~63%        |
| 1024 | ~6.2       | ~24.9      | 170.7     | 100%        |

On an unloaded box tiled reaches ~25–30 GF/s at N=1024. The small-N
regression is a finding, not a bug: parallelizing tiny GEMMs costs more
than tiling saves (grain-size problem). Absolute numbers vary with machine
load — reproduce with `make bench`. `make bench-batch` reports batching
goodput vs serial decode on a synthetic model.

## Correctness suites

Kernel suite (37 checks): matmul naive/tiled vs reference · tiled at
non-multiple-of-K block sizes · relu zeroes negatives only · softmax rows
sum to 1 at ~1000-magnitude inputs · softmax preserves argmax · gelu at
0/+10/−10 with saturation tolerance · RMSNorm unit RMS per row · linear
matches hand computation (W is `(out,in)`!) · fused bias+relu ≡
forward-then-relu · dims survive save/load · logits bit-exact after
round-trip · argmax class in range · deterministic forward · truncated
model file throws instead of crashing.

System suite: tokenizer round-trips incl. UTF-8 byte fallback; cached
decode == full recomputation over the whole history (with GQA); incremental
replay == whole-block forward row-for-row; speculative sequence == plain
greedy with multiple tokens committed per verification forward; continuous
batching == serial decoding.

## Design notes

- Tensors are contiguous row-major float32 vectors; no strides, no dtype
  zoo. Inference engines live and die by layout simplicity.
- Validation-first discipline: no kernel is benchmarked until it matches
  its reference. The fused path is tested for exact equivalence against the
  unfused one; KV-cache rollback makes speculative rejection free.
- `Linear::forward` raises descriptive exceptions (widths, bias size) so a
  mis-wired model names the offender instead of tripping `-DNDEBUG` asserts.
- CI (`.github/workflows/ci.yml`) builds every binary and runs both suites
  on ubuntu-latest.