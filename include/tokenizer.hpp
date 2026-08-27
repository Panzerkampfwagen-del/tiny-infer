#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ti {

/* Minimal SentencePiece BPE tokenizer (port of llama2.c / tiny-infer's
 * tokenizer.c to C++). Reads the llama2.c-format tokenizer.bin produced by
 * tools/export.py:
 *
 *   file:  i32 vocab_size | i32 max_token_length | i32 bos | i32 eos
 *          then per token: f32 score | i32 len | bytes[len]
 *
 * Pieces keep the literal "<0xXX>" byte-fallback form; decode expands them.
 * The leading U+2581 has already been converted to a space by the exporter.
 */
class Tokenizer {
public:
    void load(const char* path);

    // Encode text into BPE tokens; BOS is prepended. Returns token count.
    std::vector<int> encode(const std::string& text) const;

    // Decode one step: piece for `token`, relative to previous token
    // (strips the leading space of the first real token after BOS).
    std::string decode(int prev_token, int token) const;
    std::string piece(int token) const { return vocab_[token]; }

    int bos() const { return bos_; }
    int eos() const { return eos_; }
    int vocab_size() const { return vocab_size_; }

private:
    int str_lookup(const std::string& s) const;

    std::vector<std::string> vocab_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int> lookup_;
    int vocab_size_ = 0, max_token_length_ = 0, bos_ = 0, eos_ = 0;
};

}  // namespace ti