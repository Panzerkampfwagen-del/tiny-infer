/*
 * test_model -- correctness tests for the "system" layer:
 *   Tokenizer (BPE encode/decode round-trips on a synthetic vocab)
 *   Transformer (decode-with-cache == full recompute; GQA)
 *   Speculative decoding (lossless vs plain greedy; multi-token verification)
 *   BatchRunner (continuous batching == serial decode)
 */
#include "../include/model.hpp"
#include "../include/generate.hpp"
#include "../include/tokenizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

using namespace ti;

static int g_failures = 0;
#define CHECK(cond, name)                                                  \
    do {                                                                   \
        if (cond) printf("  PASS %s\n", name);                             \
        else { printf("  FAIL %s (line %d)\n", name, __LINE__);            \
               g_failures++; }                                             \
    } while (0)

static bool allclose(const Tensor& A, const Tensor& B, float tol)
{
    if (A.data.size() != B.data.size()) return false;
    for (size_t i = 0; i < A.data.size(); i++)
        if (std::fabs(A[i] - B[i]) > tol) return false;
    return true;
}

/* -------- synthetic tokenizer.bin in llama2.c format ------------------- */
static void write_test_tokenizer(const char* path)
{
    // Convention required by encoder/decoder byte fallback (+3):
    //   id 0 "<unk>", id 1 "<s>" (BOS), id 2 "</s>" (EOS),
    //   then raw-byte pieces at ids 3..258, then a small merge vocabulary.
    std::vector<std::string> pieces;
    char buf[16];
    pieces.push_back("<unk>");
    pieces.push_back("<s>");
    pieces.push_back("</s>");
    for (int i = 0; i < 256; i++) {
        snprintf(buf, sizeof(buf), "<0x%02X>", i);
        pieces.push_back(buf);
    }
    const std::vector<std::string> extra = {" ",
        "e", "h", "l", "o", "eh", "he", "ll", "lo", "hell", "hello",
        "llo", "o ", " ello"};
    pieces.insert(pieces.end(), extra.begin(), extra.end());
    FILE* f = fopen(path, "wb");
    int n = (int)pieces.size(), max_len = 6, bos = 1, eos = 2;
    fwrite(&n, 4, 1, f); fwrite(&max_len, 4, 1, f);
    fwrite(&bos, 4, 1, f); fwrite(&eos, 4, 1, f);
    for (int i = 0; i < n; i++) {
        float score = -(float)i;           // earlier pieces merge first
        int len = (int)pieces[i].size();
        fwrite(&score, 4, 1, f);
        fwrite(&len, 4, 1, f);
        fwrite(pieces[i].data(), 1, (size_t)len, f);
    }
    fclose(f);
}

static LMConfig test_cfg()
{
    LMConfig c;
    c.dim = 32; c.hidden_dim = 64; c.n_layers = 2;
    c.n_heads = 4; c.n_kv_heads = 2;             // GQA 2:1
    c.vocab_size = 31; c.max_seq_len = 48;
    c.rope_theta = 10000.f; c.norm_eps = 1e-5f;
    c.tie_embeddings = true;
    return c;
}

static void test_tokenizer(void)
{
    printf("[tokenizer]\n");
    write_test_tokenizer("/tmp/tinyinfer_tok.bin");
    Tokenizer tok;
    tok.load("/tmp/tinyinfer_tok.bin");
    CHECK(tok.bos() == 1 && tok.eos() == 2 && tok.vocab_size() == 273,
          "header fields load");
    CHECK(tok.piece(258) == "<0xFF>", "byte piece readback");

    std::vector<int> ids = tok.encode("hello");
    CHECK(ids.size() >= 2 && ids[0] == tok.bos(),
          "encode prepends BOS (with space)");
    // "hello": specials(3) + bytes(256) + offset in `extra` = 269
    CHECK(ids.back() == 269,
          "merge pass collapses to the single longest piece");

    // decode round-trip on merged output (usage-correct: BOS is context, the
    // first decoded piece is the standalone " " whose leading space strips)
    std::string out;
    int prev = tok.bos();
    for (size_t k = 1; k < ids.size(); ++k) {
        out += tok.decode(prev, ids[k]);
        prev = ids[k];
    }
    CHECK(out == "hello", "decode(encode(text)) == text");

    // byte fallback: every UTF-8 byte survives the round-trip
    const char raw[] = "h\xc3\xa9!";   // h é !
    ids = tok.encode(raw);
    out.clear(); prev = tok.bos();
    for (size_t k = 1; k < ids.size(); ++k) {
        out += tok.decode(prev, ids[k]);
        prev = ids[k];
    }
    CHECK(out == raw, "byte-fallback round-trip is exact");
}

/* decode-with-cache must equal full recomputation over the whole history */
static void test_cache_equivalence(void)
{
    printf("[transformer]\n");
    Transformer m = Transformer::random_init(test_cfg(), 7);

    std::vector<int> prompt = {3, 5, 8, 13, 21, 1};
    DecodeState st;
    st.init(m, prompt);
    Sampler smp;                                   // temperature 0 => greedy
    smp.temperature = 0.f;
    for (int step = 0; step < 5; step++) st.step(smp);

    // full recompute of the entire history from scratch
    std::vector<KVCache> fresh(m.config().n_layers);
    for (auto& c : fresh)
        c.init(m.config().n_kv_heads,
               m.config().dim / m.config().n_heads,
               m.config().max_seq_len);
    Tensor full = m.forward(st.tokens, 0, fresh);

    Tensor inc_last({full.cols()});
    std::memcpy(inc_last.ptr(), st.last_logits.ptr(),
                sizeof(float) * full.cols());
    Tensor full_last({full.cols()});
    std::memcpy(full_last.ptr(),
                &full[(size_t)(full.rows() - 1) * full.cols()],
                sizeof(float) * full.cols());
    CHECK(allclose(inc_last, full_last, 2e-4f),
          "cached decode logits == full recomputation (GQA)");

    // sequential stepping must match whole-block forward at every row:
    // replay the exact block tokens through a fresh decode state
    bool rows_ok = true;
    DecodeState seq;
    seq.init(m, {st.tokens[0]});
    for (size_t r = 1; r < st.tokens.size(); r++) {
        seq.feed(st.tokens[r]);   // forced token, logits after it
        const float* blk = &full[(size_t)r * full.cols()];
        for (uint32_t i = 0; i < full.cols(); i++)
            if (std::fabs(seq.last_logits[i] - blk[i]) > 2e-4f) rows_ok = false;
    }
    CHECK(rows_ok,
          "incremental replay matches whole-block forward at every row");
}

/* speculative decoding must produce EXACTLY the plain greedy sequence */
static void test_spec_lossless(void)
{
    printf("[speculative]\n");
    Transformer m = Transformer::random_init(test_cfg(), 11);
    // repetitive prefix maximizes prompt-lookup hits
    std::vector<int> prompt = {2, 2, 7, 7, 7, 4, 9, 9};

    DecodeState a;
    a.init(m, prompt);
    Sampler g;                                     // temperature 0 => greedy
    g.temperature = 0.f;
    std::vector<int> want(a.tokens);
    for (int i = 0; i < 12; i++) { a.step(g); want.push_back(a.tokens.back()); }

    DecodeState b;
    b.init(m, prompt);
    int committed = 0, calls = 0;
    bool in_order = true;
    for (int i = 0; i < 12;) {
        int n = spec_greedy_step(b, 2, 4);         // n new tokens, 1 forward
        ++calls;
        committed += n;
        i += n;
        if ((size_t)(prompt.size() + committed) >=
            m.config().max_seq_len - 4)
            break;
    }
    // every generated token must equal what plain greedy decoding produced
    // (compare only the reference-covered prefix: one verification call can
    // legitimately commit past the 12-token budget)
    const size_t cmp = std::min(b.tokens.size(), want.size());
    for (size_t j = 0; j < cmp; j++)
        if (b.tokens[j] != want[j]) in_order = false;
    CHECK(in_order && cmp >= (size_t)(prompt.size() + 12),
          "speculative sequence is identical to plain greedy decoding");
    CHECK(committed > calls + 2,
          "one verification forward commits multiple tokens");

    // after N tokens the speculative state equals N sequential greedy steps
    DecodeState c;
    c.init(m, prompt);
    for (int i = 0; i < committed; i++) c.step(g);
    bool bit_ok = true;
    for (uint32_t i = 0; i < c.last_logits.cols(); i++)
        if (std::fabs(c.last_logits[i] - b.last_logits[i]) > 2e-4f)
            bit_ok = false;
    CHECK(bit_ok, "after N tokens: speculative logits == plain greedy "
                  "logits");
}

/* continuous batching must match running each sequence alone */
static void test_batching_equivalence(void)
{
    printf("[batch runner]\n");
    Transformer m = Transformer::random_init(test_cfg(), 13);
    std::vector<std::vector<int>> prompts =
        {{3, 4, 5}, {9, 9, 1}, {6}};

    // serial reference
    std::vector<std::vector<int>> want;
    for (auto& p : prompts) {
        DecodeState st;
        st.init(m, p);
        Sampler gs; gs.temperature = 0.f;
        for (int i = 0; i < 6; i++) st.step(gs);
        want.push_back(st.tokens);
    }

    BatchRunner runner(m);
    for (size_t i = 0; i < prompts.size(); i++) {
        Sampler gb; gb.temperature = 0.f;
        runner.admit((int)i, prompts[i], gb, 6);
    }
    int rounds = 0;
    while (!runner.finished() && rounds++ < 10) runner.step();

    bool same = true;
    for (size_t i = 0; i < prompts.size(); i++)
        if (runner.tokens_of((int)i) != want[i]) same = false;
    CHECK(same, "continuous batching == serial decode (all sequences)");
}

int main(void)
{
    test_tokenizer();
    test_cache_equivalence();
    test_spec_lossless();
    test_batching_equivalence();

    if (g_failures) { printf("\n%d FAILURE(S)\n", g_failures); return 1; }
    printf("\nALL MODEL TESTS PASSED\n");
    return 0;
}