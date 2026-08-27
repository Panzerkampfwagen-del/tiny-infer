#include "model.hpp"
#include "tokenizer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cctype>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

namespace ti {

/* ============================ mini JSON parser =========================== */
/* Dependency-free recursive-descent JSON -> variant tree. Enough for the
 * weights header produced by tools/export.py; every failure throws. */
namespace {

struct JVal {
    enum class T { Null, Bool, Num, Str, Arr, Obj } t = T::Null;
    bool b = false;
    double num = 0;
    std::string s;
    std::vector<JVal> arr;
    std::map<std::string, JVal> obj;

    const JVal& at(const std::string& k) const {
        auto it = obj.find(k);
        if (it == obj.end())
            throw std::runtime_error("weights header missing key '" + k + "'");
        return it->second;
    }
    uint32_t as_u32() const {
        if (t != T::Num || num < 0 || num > 4e9)
            throw std::runtime_error("expected number in weights header");
        return (uint32_t)(int64_t)num;
    }
    double as_f64() const {
        if (t != T::Num) throw std::runtime_error("expected number in header");
        return num;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& src) : src_(src) {}
    JVal parse() {
        JVal v = parse_value();
        skip_ws();
        if (pos_ != src_.size()) throw std::runtime_error("trailing json bytes");
        return v;
    }

private:
    const std::string& src_;
    size_t pos_ = 0;

    void skip_ws() { while (pos_ < src_.size() && isspace((unsigned char)src_[pos_])) pos_++; }
    char peek() { skip_ws(); if (pos_ >= src_.size()) throw std::runtime_error("unexpected end of json"); return src_[pos_]; }
    void expect(char c) { if (peek() != c) throw std::runtime_error(std::string("json: expected '") + c + "'"); ++pos_; }
    void literal(const char* w) {
        skip_ws();
        size_t n = strlen(w);
        if (src_.compare(pos_, n, w) != 0)
            throw std::runtime_error("bad json literal");
        pos_ += n;
    }
    JVal parse_value() {
        char c = peek();
        if (c == '{') return parse_obj();
        if (c == '[') return parse_arr();
        if (c == '"') { JVal v; v.t = JVal::T::Str; v.s = parse_string(); return v; }
        if (c == 't') { literal("true"); JVal v; v.t = JVal::T::Bool; v.b = true; return v; }
        if (c == 'f') { literal("false"); JVal v; v.t = JVal::T::Bool; v.b = false; return v; }
        if (c == 'n') { literal("null"); return JVal{}; }
        return parse_number();
    }
    JVal parse_obj() {
        expect('{');
        JVal v; v.t = JVal::T::Obj;
        if (peek() == '}') { ++pos_; return v; }
        while (true) {
            std::string k = parse_string();
            expect(':');
            v.obj[k] = parse_value();
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == '}') { ++pos_; break; }
            throw std::runtime_error("bad json object");
        }
        return v;
    }
    JVal parse_arr() {
        expect('[');
        JVal v; v.t = JVal::T::Arr;
        if (peek() == ']') { ++pos_; return v; }
        while (true) {
            v.arr.push_back(parse_value());
            char c = peek();
            if (c == ',') { ++pos_; continue; }
            if (c == ']') { ++pos_; break; }
            throw std::runtime_error("bad json array");
        }
        return v;
    }
    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= src_.size()) throw std::runtime_error("unterminated string");
            char c = src_[pos_++];
            if (c == '"') break;
            if (c == '\\') {
                if (pos_ >= src_.size()) throw std::runtime_error("bad escape");
                char e = src_[pos_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'u': pos_ += 4; break;   // ascii headers only
                    default: throw std::runtime_error("unknown escape");
                }
            } else out += c;
        }
        return out;
    }
    JVal parse_number() {
        skip_ws();
        size_t start = pos_;
        while (pos_ < src_.size() &&
               (isdigit((unsigned char)src_[pos_]) || strchr("+-.eE", src_[pos_])))
            pos_++;
        if (start == pos_) throw std::runtime_error("bad json number");
        JVal v; v.t = JVal::T::Num;
        v.num = std::stod(src_.substr(start, pos_ - start));
        return v;
    }
};

/* BF16 pair -> fp32 (exponent/mantissa in the top 16 bits, as HF stores it). */
inline float bf16_to_f32(uint16_t bits16)
{
    uint32_t bits = (uint32_t)bits16 << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

}  // anonymous namespace

/* ============================== loading ================================== */

namespace {

struct LoadedTensor {
    std::vector<uint32_t> shape;
    std::vector<float> data;   // converted to fp32 immediately
};

LoadedTensor take(const JVal& hdr, const std::string& name,
                  const std::vector<uint8_t>& blob)
{
    LoadedTensor lt;
    bool found = false;
    for (const JVal& t : hdr.at("tensors").arr) {
        if (t.at("name").s != name) continue;
        found = true;
        if (t.at("dtype").s != "bf16")
            throw std::runtime_error(name + ": only bf16 export supported");
        size_t off = (size_t)t.at("offset").as_u32();
        for (const JVal& d : t.at("shape").arr)
            lt.shape.push_back(d.as_u32());
        size_t numel = 1;
        for (uint32_t d : lt.shape) numel *= d;
        if (off + 2 * numel > blob.size())
            throw std::runtime_error(name + ": tensor out of file bounds");
        lt.data.resize(numel);
        for (size_t i = 0; i < numel; i++)
            lt.data[i] = bf16_to_f32((uint16_t)(blob[off + 2 * i] |
                                     ((uint32_t)blob[off + 2 * i + 1] << 8)));
        break;
    }
    if (!found)
        throw std::runtime_error("weights missing tensor '" + name + "'");
    return lt;
}

Linear to_linear(LoadedTensor&& W)
{
    Linear L;
    L.W.shape = std::move(W.shape);
    L.W.data = std::move(W.data);
    L.b = Tensor({L.W.rows()});
    L.b.zeros();      // HF Llama projections have no biases
    return L;
}

}  // namespace anon2

Transformer Transformer::load(const char* path, Dtype dtype)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    auto rd = [&](void* p, size_t n) {
        if (!f.read(reinterpret_cast<char*>(p), (std::streamsize)n))
            throw std::runtime_error("truncated weight file");
    };
    uint32_t hlen;
    rd(&hlen, 4);
    if (hlen == 0 || hlen > (64u << 20))
        throw std::runtime_error("implausible weight header size");
    std::string json(hlen, '\0');
    rd(json.data(), hlen);
    JVal hdr = JsonParser(json).parse();

    Transformer m;
    m.use_int8_ = dtype == Dtype::Int8;
    const JVal& c = hdr.at("config");
    m.cfg_.dim            = c.at("dim").as_u32();
    m.cfg_.hidden_dim     = c.at("hidden_dim").as_u32();
    m.cfg_.n_layers       = c.at("n_layers").as_u32();
    m.cfg_.n_heads        = c.at("n_heads").as_u32();
    m.cfg_.n_kv_heads     = c.at("n_kv_heads").as_u32();
    m.cfg_.vocab_size     = c.at("vocab_size").as_u32();
    m.cfg_.max_seq_len    = c.at("max_seq_len").as_u32();
    m.cfg_.rope_theta     = (float)c.at("rope_theta").as_f64();
    m.cfg_.norm_eps       = (float)c.at("norm_eps").as_f64();
    m.cfg_.tie_embeddings = c.at("tie_embeddings").b;

    if (!m.cfg_.dim || !m.cfg_.hidden_dim || !m.cfg_.n_layers ||
        !m.cfg_.n_heads || !m.cfg_.n_kv_heads || !m.cfg_.vocab_size ||
        !m.cfg_.max_seq_len)
        throw std::runtime_error("zero dimension in model config");
    const uint32_t hd0 = m.cfg_.dim / m.cfg_.n_heads;
    if (!hd0 || hd0 % 2 || m.cfg_.dim % m.cfg_.n_heads)
        throw std::runtime_error("head_dim must be positive and even");
    if (m.cfg_.n_heads % m.cfg_.n_kv_heads)
        throw std::runtime_error("GQA requires n_heads % n_kv_heads == 0");
    const uint32_t D = m.cfg_.dim, HD = hd0;

    // data blob: everything after the header
    f.seekg(0, std::ios::end);
    std::streamoff fsz = f.tellg();
    if (fsz < (std::streamoff)(4 + hlen)) throw std::runtime_error("tiny weight file");
    std::vector<uint8_t> blob((size_t)fsz - 4 - hlen);
    f.seekg(4 + (std::streamoff)hlen);
    rd(blob.data(), blob.size());

    LoadedTensor e = take(hdr, "tok_embeddings", blob);
    m.tok_emb_.shape = std::move(e.shape);
    m.tok_emb_.data = std::move(e.data);
    if (m.tok_emb_.rows() != m.cfg_.vocab_size || m.tok_emb_.cols() != D)
        throw std::runtime_error("tok_embeddings shape mismatch");

    LoadedTensor fn = take(hdr, "final_norm", blob);
    m.final_norm_.shape = {D};
    m.final_norm_.data = std::move(fn.data);

    m.layers_.resize(m.cfg_.n_layers);
    const std::string nm[7] = {"wq", "wk", "wv", "wo", "w1", "w2", "w3"};
    const uint32_t outs[7] = {m.cfg_.n_heads * HD, m.cfg_.n_kv_heads * HD,
                              m.cfg_.n_kv_heads * HD, D, m.cfg_.hidden_dim,
                              D, m.cfg_.hidden_dim};
    const uint32_t ins[7]  = {D, D, D, m.cfg_.n_heads * HD, D,
                              m.cfg_.hidden_dim, D};
    for (uint32_t i = 0; i < m.cfg_.n_layers; i++) {
        Layer& ly = m.layers_[i];
        auto nmly = [&](const char* w) {
            return "layers." + std::to_string(i) + "." + w;
        };
        LoadedTensor an = take(hdr, nmly("attn_norm"), blob);
        LoadedTensor ff = take(hdr, nmly("ffn_norm"), blob);
        ly.attn_norm.shape = {D};
        ly.attn_norm.data = std::move(an.data);
        ly.ffn_norm.shape = {D};
        ly.ffn_norm.data = std::move(ff.data);
        for (int w = 0; w < 7; w++) {
            LoadedTensor W = take(hdr, nmly(nm[w].c_str()), blob);
            if (W.shape.size() != 2 || W.shape[0] != outs[w] || W.shape[1] != ins[w])
                throw std::runtime_error(nm[w] + " shape mismatch in layer " +
                                         std::to_string(i));
            Linear Lx = to_linear(std::move(W));
            if (m.use_int8_) {
                QLinearInt8 q = QLinearInt8::from_linear(Lx);
                (&ly.qwq)[w] = std::move(q);
            } else {
                (&ly.wq)[w] = std::move(Lx);
            }
        }
    }

    if (m.cfg_.tie_embeddings)
        m.lm_head_ = m.tok_emb_;                       // tied weights
    else {
        LoadedTensor lh = take(hdr, "lm_head", blob);
        m.lm_head_.shape = std::move(lh.shape);
        m.lm_head_.data = std::move(lh.data);
    }

    // cross-check norms and lm head after all loads
    for (const Layer& ly : m.layers_)
        if (ly.attn_norm.data.size() != D ||
            ly.ffn_norm.data.size() != D ||
            m.final_norm_.data.size() != D)
            throw std::runtime_error("layernorm shape mismatch");
    if (m.lm_head_.rows() != m.cfg_.vocab_size || m.lm_head_.cols() != D)
        throw std::runtime_error("lm_head shape mismatch");
    return m;
}

Transformer Transformer::random_init(const LMConfig& cfg, uint64_t seed)
{
    Transformer m;
    m.cfg_ = cfg;
    m.use_int8_ = false;
    uint64_t s = seed | 1;
    auto nxt = [&]() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return (float)((double)((int64_t)s >> 11) / 4503599627370496.0 - 1.0);
    };
    m.tok_emb_ = Tensor({cfg.vocab_size, cfg.dim});
    for (auto& v : m.tok_emb_.data) v = nxt() * 0.05f;
    m.lm_head_ = m.tok_emb_;      // always tied for synthetic models
    m.final_norm_ = Tensor({cfg.dim});
    for (auto& v : m.final_norm_.data) v = 1.f;
    const uint32_t hd0 = cfg.dim / cfg.n_heads;
    m.layers_.resize(cfg.n_layers);
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        Layer& ly = m.layers_[l];
        ly.attn_norm = Tensor({cfg.dim});
        ly.ffn_norm = Tensor({cfg.dim});
        for (auto& v : ly.attn_norm.data) v = 1.f;
        for (auto& v : ly.ffn_norm.data) v = 1.f;
        Linear tmp[7];
        const uint32_t outs[7] = {cfg.n_heads * hd0, cfg.n_kv_heads * hd0,
                                  cfg.n_kv_heads * hd0, cfg.dim,
                                  cfg.hidden_dim, cfg.dim, cfg.hidden_dim};
        const uint32_t ins[7]  = {cfg.dim, cfg.dim, cfg.dim, cfg.n_heads * hd0,
                                  cfg.dim, cfg.hidden_dim, cfg.dim};
        for (int w = 0; w < 7; w++) {
            tmp[w].W = Tensor({outs[w], ins[w]});
            tmp[w].b = Tensor({outs[w]});          // zero biases (no HF biases)
            const float sc = 0.02f / std::sqrt((float)ins[w]);
            for (auto& v : tmp[w].W.data) v = nxt() * sc;
            (&ly.wq)[w] = std::move(tmp[w]);
        }
    }
    return m;
}

/* ============================ forward pass =============================== */

namespace {

// Offset-aware causal SDPA with GQA broadcast.
//   Q: (T, Hq, Dh) contiguous; K/V: the KV-cache (L, Hkv, Dh) views.
//   Query absolute position is off+i; it attends keys [0 .. off+i].
void gqa_sdpa(const float* Q, uint32_t T, uint32_t Hq, uint32_t Dh,
              const KVCache& kc, const KVCache& vc, float* O, int off)
{
    assert(off >= 0 && Hq % kc.n_heads == 0);
    const uint32_t Hkv = kc.n_heads, ratio = Hq / Hkv;
    const uint32_t L = kc.len;
    const float scale = 1.0f / std::sqrt((float)Dh);
    std::vector<float> scores(L);
    for (uint32_t qi = 0; qi < T; qi++) {
        const uint32_t vis = (uint32_t)off + qi + 1;
        assert(vis <= L);
        for (uint32_t h = 0; h < Hq; h++) {
            const float* q = Q + ((size_t)qi * Hq + h) * Dh;
            const uint32_t hv = h / ratio;
            float m = -INFINITY;
            for (uint32_t ki = 0; ki < vis; ki++) {
                const float* k = &kc.k[(size_t)ki * Hkv * Dh + (size_t)hv * Dh];
                float sc = 0;
                for (uint32_t d = 0; d < Dh; d++) sc += q[d] * k[d];
                sc *= scale;
                scores[ki] = sc;
                if (sc > m) m = sc;
            }
            float z = 0;
            for (uint32_t ki = 0; ki < vis; ki++) {
                scores[ki] = std::exp(scores[ki] - m);
                z += scores[ki];
            }
            float* o = O + ((size_t)qi * Hq + h) * Dh;
            for (uint32_t d = 0; d < Dh; d++) o[d] = 0;
            for (uint32_t ki = 0; ki < vis; ki++) {
                const float p = scores[ki] / z;
                const float* v = &vc.v[(size_t)ki * Hkv * Dh + (size_t)hv * Dh];
                for (uint32_t d = 0; d < Dh; d++) o[d] += p * v[d];
            }
        }
    }
}

// Y = X @ W^T for W stored (out,in), used for the LM head / tied embeddings.
Tensor linear_notrans(const Tensor& X, const Tensor& W)
{
    const uint32_t M = X.rows(), K = X.cols(), N = W.rows();
    Tensor Y({M, N});
    const float* xp = X.data.data();
    const float* wp = W.data.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < (int64_t)M; i++) {
        const float* xr = xp + (size_t)i * K;
        float* yr = Y.data.data() + (size_t)i * N;
        for (uint32_t o = 0; o < N; o++) {
            const float* wr = wp + (size_t)o * K;
            float acc = 0;
            for (uint32_t c = 0; c < K; c++) acc += xr[c] * wr[c];
            yr[o] = acc;
        }
    }
    return Y;
}

}  // anonymous namespace

const Linear& Transformer::L(const Layer& ly, int which) const
{
    const Linear* base = &ly.wq;
    return base[which];
}
const QLinearInt8& Transformer::Q(const Layer& ly, int which) const
{
    const QLinearInt8* base = &ly.qwq;
    return base[which];
}
Tensor Transformer::project(const Layer& ly, int which, const Tensor& x) const
{
    return use_int8_ ? Q(ly, which).forward(x) : L(ly, which).forward(x);
}

Tensor Transformer::forward(const std::vector<int>& tokens, int pos0,
                            std::vector<KVCache>& caches) const
{
    const uint32_t D = cfg_.dim, Hd = D / cfg_.n_heads;
    const uint32_t T = (uint32_t)tokens.size();
    assert(caches.size() == cfg_.n_layers);
    assert(pos0 >= 0 && pos0 + (int)T <= (int)cfg_.max_seq_len);

    Tensor x({T, D});
    for (uint32_t t = 0; t < T; t++)
        std::memcpy(&x[(size_t)t * D],
                    tok_emb_.data.data() + (size_t)tokens[t] * D,
                    D * sizeof(float));

    for (uint32_t li = 0; li < cfg_.n_layers; li++) {
        const Layer& ly = layers_[li];

        Tensor h = rmsnorm_lastdim(x, &ly.attn_norm);
        Tensor q = project(ly, 0, h);          // (T, Hq*Hd)
        Tensor k = project(ly, 1, h);          // (T, Hkv*Hd)
        Tensor v = project(ly, 2, h);

        // RoPE in place (rotate_half convention matches the exporter)
        Tensor qr = rope(q, cfg_.n_heads, (uint32_t)pos0, cfg_.rope_theta);
        Tensor kr = rope(k, cfg_.n_kv_heads, (uint32_t)pos0, cfg_.rope_theta);

        KVCache& kc = caches[li];
        for (uint32_t t = 0; t < T; t++) {   // append this block's K/V
            Tensor kt({1, cfg_.n_kv_heads, Hd}), vt({1, cfg_.n_kv_heads, Hd});
            std::memcpy(kt.ptr(), &kr[(size_t)t * cfg_.n_kv_heads * Hd],
                        sizeof(float) * cfg_.n_kv_heads * Hd);
            std::memcpy(vt.ptr(), &v[(size_t)t * cfg_.n_kv_heads * Hd],
                        sizeof(float) * cfg_.n_kv_heads * Hd);
            kc.append(kt, vt);
        }

        Tensor att({T, cfg_.n_heads, Hd});
        gqa_sdpa(qr.ptr(), T, cfg_.n_heads, Hd, kc, kc, att.ptr(), pos0);

        Tensor att2d({T, cfg_.n_heads * Hd});
        att2d.data = att.data;
        Tensor o = project(ly, 3, att2d);
        for (size_t i = 0; i < x.numel(); i++) x[i] += o[i];

        h = rmsnorm_lastdim(x, &ly.ffn_norm);
        Tensor gate = project(ly, 4, h);       // w1
        Tensor up   = project(ly, 6, h);       // w3
        Tensor act  = swiglu(gate, up);
        Tensor down = project(ly, 5, act);     // w2
        for (size_t i = 0; i < x.numel(); i++) x[i] += down[i];
    }

    Tensor hn = rmsnorm_lastdim(x, &final_norm_);
    return linear_notrans(hn, lm_head_);     // (T, vocab) = X @ E^T
}

/* ===================== prompt-lookup speculative decoding ================ */

std::vector<int> prompt_lookup_draft(const std::vector<int>& tokens,
                                     size_t ngram, size_t max_draft)
{
    const size_t n = tokens.size();
    if (ngram == 0 || n <= ngram || max_draft == 0) return {};
    const size_t suf = n - ngram;
    for (size_t i = 0; i + ngram < n; ++i) {
        if (std::equal(tokens.begin() + (long)i,
                       tokens.begin() + (long)(i + ngram),
                       tokens.begin() + (long)suf)) {
            std::vector<int> draft;
            for (size_t k = 0; k < max_draft && i + ngram + k < n; ++k)
                draft.push_back(tokens[i + ngram + k]);
            return draft;
        }
    }
    return {};
}

}  // namespace ti