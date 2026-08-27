#include "tokenizer.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ti {

void Tokenizer::load(const char* path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    auto rd = [&](void* p, size_t n) {
        if (!f.read(reinterpret_cast<char*>(p), (std::streamsize)n))
            throw std::runtime_error("truncated tokenizer file");
    };
    int hdr[4];
    rd(hdr, sizeof(hdr));
    vocab_size_       = hdr[0];
    max_token_length_ = hdr[1];
    bos_              = hdr[2];
    eos_              = hdr[3];
    if (vocab_size_ <= 0 || vocab_size_ > (1 << 22) || max_token_length_ <= 0 ||
        max_token_length_ > 1024)
        throw std::runtime_error("implausible tokenizer header");

    vocab_.resize(vocab_size_);
    scores_.resize(vocab_size_);
    for (int i = 0; i < vocab_size_; i++) {
        int len;
        rd(&scores_[i], 4);
        rd(&len, 4);
        if (len < 0 || len > 4096)
            throw std::runtime_error("bad piece length in tokenizer file");
        std::string s(len, '\0');
        rd(s.data(), (size_t)len);
        vocab_[i] = s;
    }
    for (int i = 0; i < vocab_size_; i++)
        lookup_.emplace(vocab_[i], i);   // first id wins on duplicates
}

int Tokenizer::str_lookup(const std::string& s) const
{
    auto it = lookup_.find(s);
    return it == lookup_.end() ? -1 : it->second;
}

std::vector<int> Tokenizer::encode(const std::string& text) const
{
    std::vector<int> tok;
    tok.push_back(bos_);
    // SentencePiece prepends a space to non-empty input.
    if (!text.empty()) {
        int sp = str_lookup(" ");
        if (sp != -1) tok.push_back(sp);
    }
    // First pass: UTF-8 codepoint -> piece, else byte fallback (id = byte+3).
    std::string buf(max_token_length_ + 4, '\0');
    size_t len = 0;
    for (size_t c = 0; c < text.size(); ++c) {
        unsigned char ch = (unsigned char)text[c];
        if ((ch & 0xC0) != 0x80) len = 0;         // start of a codepoint
        buf[len++] = (char)ch;
        bool more = (c + 1 < text.size()) &&
                    ((unsigned char)text[c + 1] & 0xC0) == 0x80 && len < 4;
        if (more) continue;
        buf[len] = '\0';
        int id = str_lookup(buf.substr(0, len));
        if (id != -1) {
            tok.push_back(id);
        } else {
            for (size_t b = 0; b < len; ++b)
                tok.push_back((unsigned char)buf[b] + 3);
        }
        len = 0;
    }

    // Merge pass: repeatedly merge the best-scoring adjacent pair.
    while (true) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (size_t i = 0; i + 1 < tok.size(); ++i) {
            int id = str_lookup(vocab_[tok[i]] + vocab_[tok[i + 1]]);
            if (id != -1 && scores_[id] > best_score) {
                best_score = scores_[id];
                best_id = id;
                best_idx = (int)i;
            }
        }
        if (best_idx == -1) break;
        tok[best_idx] = best_id;
        tok.erase(tok.begin() + best_idx + 1);
    }
    return tok;
}

std::string Tokenizer::decode(int prev_token, int token) const
{
    const std::string& piece = vocab_[token];
    // SentencePiece strips the leading space of the first real token after BOS.
    std::string out;
    if (prev_token == bos_ && !piece.empty() && piece[0] == ' ')
        out = piece.substr(1);
    else
        out = piece;
    // Expand raw-byte tokens "<0xXX>".
    unsigned int bv;
    if (out.size() == 6 && out[0] == '<' && out[1] == '0' && out[2] == 'x' &&
        sscanf(out.c_str(), "<0x%02X>", &bv) == 1)
        out.assign(1, (char)bv);
    return out;
}

}  // namespace ti