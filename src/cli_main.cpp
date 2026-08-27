/*
 * tinyinfer_cli -- text generation over the tiny_infer kernels.
 *
 *   ./build/tinyinfer_cli --model weights.bin --tokenizer tokenizer.bin \
 *       --prompt "Once upon a time" --max-tokens 128 \
 *       [--temperature 0.8] [--top-k 40] [--top-p 0.9] [--seed 42] \
 *       [--dtype int8] [--spec 3 6] [--chat] [--stats]
 *
 * Greedy + --spec enables prompt-lookup speculative decoding (lossless).
 */
#include "generate.hpp"
#include "tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace ti;

int main(int argc, char** argv)
{
    const char* model_path = "tinyllama.bin";
    const char* tok_path = "tokenizer.bin";
    std::string prompt = "Once upon a time";
    int max_tokens = 64;
    float temperature = 0.0f;
    uint32_t top_k = 0;
    float top_p = 1.0f;
    uint64_t seed = 42;
    Transformer::Dtype dtype = Transformer::Dtype::Fp32;
    size_t spec_ngram = 0, spec_draft = 0;
    bool chat = false, stats = false;

    auto need = [&](int i) {
        if (i >= argc) { fprintf(stderr, "missing value for %s\n", argv[i - 1]); exit(1); }
        return argv[i];
    };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model") model_path = need(++i);
        else if (a == "--tokenizer") tok_path = need(++i);
        else if (a == "--prompt") prompt = need(++i);
        else if (a == "--max-tokens") max_tokens = atoi(need(++i));
        else if (a == "--temperature") temperature = atof(need(++i));
        else if (a == "--top-k") top_k = (uint32_t)atoi(need(++i));
        else if (a == "--top-p") top_p = atof(need(++i));
        else if (a == "--seed") seed = strtoull(need(++i), nullptr, 10);
        else if (a == "--dtype") dtype = strcmp(need(++i), "int8") == 0
                                       ? Transformer::Dtype::Int8
                                       : Transformer::Dtype::Fp32;
        else if (a == "--spec") { spec_ngram = atoi(need(++i)); spec_draft = atoi(need(++i)); }
        else if (a == "--chat") chat = true;
        else if (a == "--stats") stats = true;
        else { fprintf(stderr, "unknown flag %s\n", argv[i]); return 1; }
    }

    try {
        Tokenizer tok;
        tok.load(tok_path);
        fprintf(stderr, "[loading %s ...]\n", model_path);
        Transformer model = Transformer::load(model_path, dtype);
        const LMConfig& c = model.config();
        fprintf(stderr,
                "[dim=%u layers=%u heads=%u kv_heads=%u vocab=%u "
                "max_seq=%u%s]\n",
                c.dim, c.n_layers, c.n_heads, c.n_kv_heads, c.vocab_size,
                c.max_seq_len, dtype == Transformer::Dtype::Int8 ? " int8" : "");

        Sampler smp;
        smp.temperature = temperature;
        smp.top_k = top_k;
        smp.top_p = top_p;
        smp.seed = seed;

        auto run_turn = [&](const std::string& text) {
            std::vector<int> ids = tok.encode(text);
            DecodeState st;
            st.init(model, ids);

            printf("%s", text.c_str());
            fflush(stdout);
            double t0 = monotonic_ms();
            int emitted = 0, forwards = 0;
            for (int i = 0; i < max_tokens; i++) {
                if (temperature > 0.0f || !spec_ngram || !spec_draft) {
                    int t = st.step(smp);
                    ++forwards;
                    if (t == tok.eos()) break;
                    printf("%s", tok.decode(st.tokens.size() >= 2
                                                ? st.tokens[st.tokens.size() - 2]
                                                : tok.bos(),
                                            t).c_str());
                    fflush(stdout);
                } else {
                    // lossless greedy speculative decoding
                    const size_t before = st.tokens.size();
                    int n_new = spec_greedy_step(st, spec_ngram, spec_draft);
                    forwards += (int)(n_new > 1 ? n_new : 1); // verify batch or 1 step
                    if (n_new <= 0) break;
                    for (size_t j = before; j < st.tokens.size(); j++) {
                        if (st.tokens[j] == tok.eos()) goto done;
                        printf("%s",
                               tok.decode((int)j - 1 >= 0 ? st.tokens[j - 1]
                                                          : tok.bos(),
                                          st.tokens[j]).c_str());
                    }
                    fflush(stdout);
                    continue;
                done:
                    break;
                }
                emitted++;
            }
            double dt = monotonic_ms() - t0;
            printf("\n");
            if (stats)
                fprintf(stderr,
                        "[%d new tokens in %.1f ms -> %.1f tok/s decode]\n",
                        emitted, dt, dt > 0 ? emitted * 1000.0 / dt : 0);
            return std::vector<int>{};      // turn history restarts per turn
        };

        if (chat) {
            char line[2048];
            while (true) {
                printf("\n> ");
                fflush(stdout);
                if (!fgets(line, sizeof(line), stdin)) break;
                size_t n = strlen(line);
                while (n && (line[n - 1] == '\n' || line[n - 1] == '\r'))
                    line[--n] = '\0';
                if (!strcmp(line, "/quit")) break;
                run_turn(line);
            }
        } else {
            run_turn(prompt);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}