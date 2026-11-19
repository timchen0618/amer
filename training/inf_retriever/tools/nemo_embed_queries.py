"""Embed the questions of a jsonl with base Nemotron-3.5-Embed-1B (session N, 2026-11-13), for hard-negative mining.

Same query path as retrieval with a Nemotron checkpoint at step 0 (verified cos 1.0000 to the model-card path):
'query: <question>' without BOS / EOS, bf16, mean pooling, L2 normalization. Saves (n, 2048) float32 in the file's
order; retrieval_inf.py --load_query_embeddings then runs the usual faiss search in the nli environment.

  python training/inf_retriever/tools/nemo_embed_queries.py --data <jsonl> --out <npy>
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import nemotron  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()
    qs = []
    for line in open(args.data):
        x = json.loads(line)
        qs.append(x.get("question") or x.get("question_text") or "")
    tok = nemotron.load_tokenizer()
    lm = nemotron.load_backbone(grad_ckpt=False).cuda().eval()
    out = np.zeros((len(qs), lm.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(qs), args.batch_size):
            b = tok([nemotron.format_query(q) for q in qs[i:i + args.batch_size]], padding=True,
                    return_tensors="pt").to("cuda")
            h = lm(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
            out[i:i + len(h)] = F.normalize(nemotron.mean_pool(h, b["attention_mask"]).float(), dim=-1).cpu().numpy()
    assert np.isfinite(out).all()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, out)
    print(f"saved {out.shape} to {args.out}")


if __name__ == "__main__":
    main()
