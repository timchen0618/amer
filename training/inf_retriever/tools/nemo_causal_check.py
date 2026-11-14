"""GPU check of --retriever nemotron --force_causal (session N, 2026-11-14), on the real Nemotron weights.

  * query encoder causal: a prefix's hidden states do not change when tokens are appended (bidirectional: they do);
  * doc encoder still bidirectional, and encode_documents still mean-pools (equal to the base bidirectional path);
  * query pooling = last token (causal) / mean (bidirectional default, unchanged);
  * multi generate() runs and its first embedding equals the last-token pooled causal question embedding.
Run in the nemotron.ext3 overlay with HF_HUB_OFFLINE=1:  python training/inf_retriever/tools/nemo_causal_check.py
"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import inbatch, nemotron  # noqa: E402


def make_opt(**kw):
    o = dict(norm_doc=True, norm_query=True, label_smoothing=0.0, loss_fn="hungarian", training_mode="multi",
             temperature=0.05, force_causal=False, retriever_model_id="x", use_lora=False)
    o.update(kw)
    return types.SimpleNamespace(**o)


def main():
    tok = nemotron.load_tokenizer()
    cls = nemotron.nemotron_class(inbatch.EmbeddingModelFrozenDocEnc)
    m_c = cls(make_opt(force_causal=True)).cuda().eval()
    m_b = cls(make_opt(force_causal=False)).cuda().eval()
    assert m_c.encoder.config.is_causal is True and m_c.doc_encoder.config.is_causal is False
    assert m_b.encoder.config.is_causal is False and m_c.force_causal is False
    a = nemotron.format_query("who wrote the book")
    b = nemotron.format_query("who wrote the book about the history of rome and greece")
    xa = tok([a], return_tensors="pt").to("cuda")
    xb = tok([b], return_tensors="pt").to("cuda")
    n = xa["input_ids"].shape[1]
    with torch.no_grad():
        for name, enc, want_causal in [("query causal", m_c.encoder, True), ("doc (causal model)", m_c.doc_encoder, False),
                                       ("query bidir", m_b.encoder, False)]:
            ha = enc(**xa).last_hidden_state.float()
            hb = enc(**xb).last_hidden_state.float()
            d = (ha[0] - hb[0, :n]).abs().max().item()
            print(f"{name}: prefix diff {d:.4f}")
            assert (d < 0.15) == want_causal, (name, d)
        # padded batch: causal + last-token pool equals the unpadded last token
        xab = tok([a, b], padding=True, return_tensors="pt").to("cuda")
        hab = m_c.encoder(**xab).last_hidden_state
        pooled = m_c.last_token_pool(hab, xab["attention_mask"]).float()
        ref = m_c.encoder(**xa).last_hidden_state[0, -1].float()
        print("causal last-token pool (padded) vs unpadded last token:", (pooled[0] - ref).abs().max().item())
        assert torch.allclose(pooled[0], ref, atol=0.15)
        # documents: mean pooling through the bidirectional frozen copy in both models
        p = tok([nemotron.format_passage("Rome. Rome is the capital of Italy."),
                 nemotron.format_passage("Greece. Greece is a country in southeastern Europe with many islands.")],
                padding=True, return_tensors="pt").to("cuda")
        ids = torch.cat([p["input_ids"], p["input_ids"]]); msk = torch.cat([p["attention_mask"], p["attention_mask"]])
        dc, _ = m_c.encode_documents(ids, msk)
        db, _ = m_b.encode_documents(ids, msk)
        ref = nemotron.mean_pool(m_b.doc_encoder(**p).last_hidden_state, p["attention_mask"])
        print("doc emb causal-model vs bidir-model:", (dc - db).abs().max().item(), " vs mean-pool ref:",
              (dc - ref).abs().max().item())
        cos = torch.nn.functional.cosine_similarity(dc.float(), ref.float(), dim=-1)
        print("doc emb cosine to mean-pool ref:", cos.tolist())
        assert torch.equal(dc, db) and (cos > 0.999).all()
        # default (bidirectional) query pooling is still the mean
        hb_ = m_b.encoder(**xab).last_hidden_state
        assert torch.equal(m_b.last_token_pool(hb_, xab["attention_mask"]), nemotron.mean_pool(hb_, xab["attention_mask"]))
        # multi generate on the causal model: first embedding = last-token pooled question
        # (the multi collators left-pad queries; generate() asserts it)
        tok.padding_side = "left"
        xl = tok([a, b], padding=True, return_tensors="pt").to("cuda")
        tok.padding_side = "right"
        pos = (xl["attention_mask"].cumsum(1) - 1).clamp(min=0)
        out = m_c.generate(xl["input_ids"], xl["attention_mask"], pos, max_new_tokens=3)
        e = out[0] if isinstance(out, (tuple, list)) else out
        print("generate output shape:", tuple(e.shape))
        assert torch.isfinite(e.float()).all()
        hl = m_c.encoder(input_ids=xl["input_ids"], attention_mask=xl["attention_mask"], position_ids=pos).last_hidden_state
        first = hl[:, -1].float()
        e0 = e[:, 0].float() if e.dim() == 3 else e.float()
        cos = torch.nn.functional.cosine_similarity(e0, first, dim=-1)
        print("generate step-1 embedding vs last-token pooled question, cosine:", cos.tolist())
        assert (cos > 0.99).all()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
