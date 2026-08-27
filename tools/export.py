#!/usr/bin/env python3
"""Export a HuggingFace Llama-2-style model to the tiny_infer flat binary.

Outputs:
  <weights>.bin - bf16 weights with a JSON header:
      {"config": {...}, "tensors": [{"name","shape","dtype","offset"}...]}
  tokenizer.bin - SentencePiece vocab in llama2.c format

RoPE convention: HF Llama uses rotate_half (NeoX style); exported q/k weights
are stored as-is, matching tiny_infer's rope() exactly.
"""
import argparse, json, os, struct, sys
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# HF name -> exported name. {i} filled per layer.
PER_LAYER = [
    ("model.layers.{i}.input_layernorm.weight",          "layers.{i}.attn_norm"),
    ("model.layers.{i}.self_attn.q_proj.weight",         "layers.{i}.wq"),
    ("model.layers.{i}.self_attn.k_proj.weight",         "layers.{i}.wk"),
    ("model.layers.{i}.self_attn.v_proj.weight",         "layers.{i}.wv"),
    ("model.layers.{i}.self_attn.o_proj.weight",         "layers.{i}.wo"),
    ("model.layers.{i}.post_attention_layernorm.weight", "layers.{i}.ffn_norm"),
    ("model.layers.{i}.mlp.gate_proj.weight",            "layers.{i}.w1"),
    ("model.layers.{i}.mlp.up_proj.weight",              "layers.{i}.w3"),
    ("model.layers.{i}.mlp.down_proj.weight",            "layers.{i}.w2"),
]


def bf16_bytes(t):
    b = t.detach().to(torch.bfloat16).contiguous().cpu()
    return b.view(torch.int16).numpy().astype("<i2").tobytes()


def collect_tensors(model, cfg):
    sd = model.state_dict()
    out = [("tok_embeddings", sd["model.embed_tokens.weight"])]
    for i in range(cfg.num_hidden_layers):
        for hf, name in PER_LAYER:
            out.append((name.format(i=i), sd[hf.format(i=i)]))
    out.append(("final_norm", sd["model.norm.weight"]))
    if not cfg.tie_word_embeddings and "lm_head.weight" in sd:
        out.append(("lm_head", sd["lm_head.weight"]))
    return out


def write_weights(path, tensors, cfg, rope_theta):
    header, data_parts, offset = [], [], 0
    for name, t in tensors:
        raw = bf16_bytes(t)
        header.append({"name": name, "shape": list(t.shape),
                       "dtype": "bf16", "offset": offset})
        data_parts.append(raw)
        offset += len(raw)
    doc = {"config": {
        "dim": cfg.hidden_size,
        "hidden_dim": cfg.intermediate_size,
        "n_layers": cfg.num_hidden_layers,
        "n_heads": cfg.num_attention_heads,
        "n_kv_heads": getattr(cfg, "num_key_value_heads",
                              cfg.num_attention_heads),
        "vocab_size": cfg.vocab_size,
        "max_seq_len": getattr(cfg, "max_position_embeddings", 2048),
        "rope_theta": float(rope_theta),
        "norm_eps": float(cfg.rms_norm_eps),
        "tie_embeddings": bool(getattr(cfg, "tie_word_embeddings", False)),
    }, "tensors": header}
    hjson = json.dumps(doc).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(hjson)))
        f.write(hjson)
        for part in data_parts:
            f.write(part)
    print(f"  wrote {path}: {len(header)} tensors, {offset/1e6:.1f} MB data")


def write_tokenizer(path, model_id):
    from huggingface_hub import hf_hub_download
    import sentencepiece as spm
    spm_path = hf_hub_download(repo_id=model_id, filename="tokenizer.model")
    sp = spm.SentencePieceProcessor(model_file=spm_path)
    n = sp.get_piece_size()
    def piece_bytes(p):
        return p.replace("▁", " ").encode("utf-8")
    blobs = [piece_bytes(sp.id_to_piece(i)) for i in range(n)]
    scores = [sp.get_score(i) for i in range(n)]
    max_len = max(len(b) for b in blobs)
    with open(path, "wb") as f:
        f.write(struct.pack("<iiii", n, max_len, sp.bos_id(), sp.eos_id()))
        for score, b in zip(scores, blobs):
            f.write(struct.pack("<fi", score, len(b)))
            f.write(b)
    print(f"  wrote {path}: {n} tokens, bos={sp.bos_id()} eos={sp.eos_id()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--weights", default="tinyllama.bin")
    ap.add_argument("--tokenizer", default="tokenizer.bin")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoConfig
    print(f"loading {args.model} ...")
    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    rope_theta = getattr(cfg, "rope_theta", 10000.0)

    os.makedirs(args.outdir, exist_ok=True)
    write_weights(os.path.join(args.outdir, args.weights),
                  collect_tensors(model, cfg), cfg, rope_theta)
    write_tokenizer(os.path.join(args.outdir, args.tokenizer), args.model)
    print("done.")


if __name__ == "__main__":
    sys.exit(main())