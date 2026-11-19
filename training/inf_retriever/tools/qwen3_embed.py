"""Base Qwen3-Embedding-0.6B embeddings (session N, 2026-11-15), the counterpart of nemo_embed_corpus.py /
nemo_embed_queries.py. Runs in div.ext3 (transformers 4.56).

  corpus:  python tools/qwen3_embed.py corpus --output_dir <dir> --shard_id 3      (gen_embed_new.py's shard split,
           ids and '<title> <text>' passages; pickle (ids, float16 [n, 1024]) at <dir>/passages_<shard:02d>)
  queries: python tools/qwen3_embed.py queries --data <jsonl> --out <npy>          ((n, 1024) float32, file order)

Same path as training (src/qwen3emb.py + the training collators): tokenizer with its appended <|endoftext|>,
truncation at --maxlen (passages 512 like the collator; queries 1024 like retrieval_inf.py), bf16 forward, last-token
pooling, L2 normalization. Queries: 'Instruct: <training task>\\nQuery:<q>'; passages: no prompt.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import qwen3emb  # noqa: E402


@torch.no_grad()
def embed_sorted(texts, model, tok, maxlen, max_tokens):
    enc = tok(texts, max_length=maxlen, truncation=True)["input_ids"]
    assert all(x[-1] == tok.pad_token_id for x in enc[:100]), "expected the appended <|endoftext|>"
    order = np.argsort([-len(x) for x in enc], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    i, nb, t0 = 0, 0, time.time()
    while i < len(order):
        bs = max(1, max_tokens // len(enc[order[i]]))
        idx = order[i:i + bs]
        L = max(len(enc[j]) for j in idx)
        ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, :len(enc[j])] = torch.tensor(enc[j])
            mask[r, :len(enc[j])] = 1
        ids, mask = ids.cuda(), mask.cuda()
        h = model(input_ids=ids, attention_mask=mask).last_hidden_state
        e = F.normalize(qwen3emb.last_token_pool(h, mask).float(), dim=-1)
        out[idx] = e.cpu().numpy().astype(np.float16)
        i += len(idx)
        nb += 1
        if nb % 200 == 0:
            print(f"{i}/{len(order)} ({i / (time.time() - t0):.0f}/s)", flush=True)
    print(f"embedded {len(order)} in {time.time() - t0:.0f} s ({len(order) / (time.time() - t0):.0f}/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["corpus", "queries"])
    ap.add_argument("--passages", default="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv")
    ap.add_argument("--output_dir")
    ap.add_argument("--prefix", default="passages")
    ap.add_argument("--shard_id", type=int)
    ap.add_argument("--num_shards", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=131072, help="tokens per forward batch (corpus)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: embed only the first n passages, no save")
    ap.add_argument("--data")
    ap.add_argument("--out")
    args = ap.parse_args()
    tok = qwen3emb.load_tokenizer()

    if args.what == "queries":
        qs = [json.loads(l) for l in open(args.data)]
        qs = [qwen3emb.format_query(x.get("question") or x.get("question_text") or "") for x in qs]
        model = qwen3emb.load_backbone().cuda().eval()
        out = qwen3emb.embed_texts(qs, model, tok, args.maxlen or 1024)
        assert np.isfinite(out).all()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        np.save(args.out, out)
        print(f"saved {out.shape} to {args.out}; first query: {qs[0]!r}")
        return

    from nemo_embed_corpus import read_shard  # gen_embed_new.py's split, verbatim
    out_file = os.path.join(args.output_dir, f"{args.prefix}_{args.shard_id:02d}")
    if os.path.exists(out_file) and not args.limit:
        print(f"{out_file} exists, skipping")
        return
    ids, texts = read_shard(args.passages, args.shard_id, args.num_shards)
    if args.limit:
        ids, texts = ids[:args.limit], texts[:args.limit]
    model = qwen3emb.load_backbone().cuda().eval()
    emb = embed_sorted(texts, model, tok, args.maxlen or 512, args.max_tokens)
    assert np.isfinite(emb).all()
    print(f"embeddings {emb.shape}, norms {np.linalg.norm(emb[:1000].astype(np.float32), axis=1).mean():.4f}")
    if args.limit:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    tmp = out_file + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump((ids, emb), f)
    os.replace(tmp, out_file)
    print(f"saved {len(ids)} to {out_file}")


if __name__ == "__main__":
    main()
