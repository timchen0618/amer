"""GPU check of --retriever qwen3 (session N, 2026-11-15), on the real Qwen3-Embedding-0.6B weights. div.ext3:
  python training/inf_retriever/tools/qwen3_check.py

  * our query / passage path (src/qwen3emb.py) equals the model card's sentence-transformers path (cosine ~1);
  * the backbone is causal (a prefix's hidden states do not change when tokens are appended);
  * the frozen-doc-encoder model classes build with the Qwen3 id, documents pool the last (appended) token, and the
    multi generate() step-1 embedding equals the last-token pooled question;
  * training queries are formatted 'Instruct: <task>\\nQuery:<q>'.
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import inbatch, qwen3emb  # noqa: E402


def make_opt(**kw):
    o = dict(norm_doc=True, norm_query=True, label_smoothing=0.0, loss_fn="hungarian", training_mode="multi",
             temperature=0.05, force_causal=False, retriever_model_id="infly/inf-retriever-v1-1.5b", use_lora=False)
    o.update(kw)
    return types.SimpleNamespace(**o)


def main():
    tok = qwen3emb.load_tokenizer()
    lm = qwen3emb.load_backbone().cuda().eval()
    qs = ["who wrote the book", "which movies did christopher nolan direct?"]
    ps = ["Rome Rome is the capital of Italy.", "Greece Greece is a country in southeastern Europe with many islands."]
    ours_q = qwen3emb.embed_texts([qwen3emb.format_query(q) for q in qs], lm, tok, 1024)
    ours_p = qwen3emb.embed_texts(ps, lm, tok, 512)
    print("query text:", repr(qwen3emb.format_query(qs[0])))
    print("token ids end:", tok([ps[0]])["input_ids"][0][-3:], "pad id", tok.pad_token_id)
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(qwen3emb.QWEN3_ID, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
        ref_q = st.encode(qs, prompt=f"Instruct: {qwen3emb.TASK}\nQuery:", normalize_embeddings=True)
        ref_p = st.encode(ps, normalize_embeddings=True)
        cq = (torch.tensor(ours_q) * torch.tensor(ref_q)).sum(-1)
        cp = (torch.tensor(ours_p) * torch.tensor(ref_p)).sum(-1)
        print("cosine to sentence-transformers: queries", cq.tolist(), "passages", cp.tolist())
        assert (cq > 0.99).all() and (cp > 0.99).all()
    except ImportError as e:
        print("sentence-transformers unavailable:", e)
    print("scores q x p:", (torch.tensor(ours_q) @ torch.tensor(ours_p).T).tolist())

    a, b = qwen3emb.format_query(qs[0]), qwen3emb.format_query(qs[0] + " about the history of rome and greece")
    xa = tok([a], return_tensors="pt", add_special_tokens=False).to("cuda")
    xb = tok([b], return_tensors="pt", add_special_tokens=False).to("cuda")
    n = xa["input_ids"].shape[1]
    assert torch.equal(xa["input_ids"][0], xb["input_ids"][0, :n])
    with torch.no_grad():  # bf16 hidden states reach ~1e3-1e4 (attention-sink tokens): compare per-token cosines
        ha, hb = lm(**xa).last_hidden_state[0].float(), lm(**xb).last_hidden_state[0, :n].float()
        bidir = lm(**xb).last_hidden_state[0].float()
    c = F.cosine_similarity(ha, hb, dim=-1)
    print(f"prefix tokens: min cosine {c.min().item():.5f}, max |diff| {(ha - hb).abs().max().item():.3f}, "
          f"max |h| {ha.abs().max().item():.0f} (cosine 1 => causal)")
    assert c.min() > 0.999
    # contrast: the last prefix token vs the same position when later tokens could be seen would differ; with a
    # causal mask the appended tokens only change positions >= n
    print(f"appended positions differ from prefix end: cosine {F.cosine_similarity(ha[-1], bidir[n], dim=0).item():.3f}")
    del lm

    for cls_name, mode in [("EmbeddingModelFrozenDocEncSingleQuery", "standard_org_q"), ("EmbeddingModelFrozenDocEnc", "multi")]:
        cls = qwen3emb.qwen3_class(getattr(inbatch, cls_name))
        m = cls(make_opt(training_mode=mode)).cuda().eval()
        assert m.encoder.config.model_type == "qwen3" and m.doc_encoder.config.model_type == "qwen3"
        p = tok(ps, padding=True, return_tensors="pt").to("cuda")
        ids = torch.cat([p["input_ids"], p["input_ids"]]); msk = torch.cat([p["attention_mask"], p["attention_mask"]])
        with torch.no_grad():
            dpos, _ = m.encode_documents(ids, msk)
        c = F.cosine_similarity(dpos.float(), torch.tensor(ours_p, device="cuda"), dim=-1)
        print(cls_name, "doc emb cosine to base path:", c.tolist())
        assert (c > 0.99).all()
        if mode == "multi":
            tok.padding_side = "left"
            xl = tok([qwen3emb.format_query(q) for q in qs], padding=True, return_tensors="pt").to("cuda")
            tok.padding_side = "right"
            pos = (xl["attention_mask"].cumsum(1) - 1).clamp(min=0)
            with torch.no_grad():
                out = m.generate(xl["input_ids"], xl["attention_mask"], pos, max_new_tokens=3)
            e = out[0] if isinstance(out, (tuple, list)) else out
            print("generate output shape:", tuple(e.shape))
            c = F.cosine_similarity(e[:, 0].float(), torch.tensor(ours_q, device="cuda"), dim=-1)
            print("generate step-1 vs base query embedding, cosine:", c.tolist())
            assert (c > 0.99).all()
        del m
        torch.cuda.empty_cache()

    from src import finetuning_data
    qwen3emb.use_qwen3_text_format()
    print("training query format:", repr(finetuning_data.get_detailed_instruct("T", "q?")))
    assert finetuning_data.get_detailed_instruct("T", "q?") == "Instruct: T\nQuery:q?"
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
