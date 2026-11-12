"""Embed one shard of the passage corpus with base Nemotron-3.5-Embed-1B (session N, 2026-11-12): the frozen
document encoder's corpus embedding, the Nemotron counterpart of wikipedia_embeddings/qampari/shared_frozen_base_512.

Same shard split, passage ids, passage text ('<title> <text>') and output format as gen_embed_new.py (pickle of
(ids, float16 [n, d] array), file <output_dir>/passages_<shard:02d>), so retrieval_inf.py / retrieve_ckpt.sbatch read
it unchanged. Nemotron-specific (src/nemotron.py): 'passage: ' input without BOS / EOS, truncated at
--passage_maxlength tokens like the training collator, bf16 forward, mean pooling, L2 normalization.
Batches are length-sorted (less padding); the output keeps the corpus order.

  python training/inf_retriever/tools/nemo_embed_corpus.py --passages <tsv> --output_dir <dir> --shard_id 3
"""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import nemotron  # noqa: E402


def read_shard(path, shard_id, num_shards):
    """gen_embed_new.py's split, verbatim (header row, then shard_size rows from start_idx)."""
    total = sum(1 for _ in open(path)) - 1
    shard_size = math.ceil(total / num_shards)
    start_idx = shard_id * shard_size
    chunk = next(iter(pd.read_csv(path, chunksize=shard_size, skiprows=start_idx, delimiter="\t",
                                  dtype={"id": str, "title": str, "text": str})))
    ids = [row.iloc[0] for _, row in chunk.iterrows()]
    texts = [str(row.iloc[2]) + " " + str(row.iloc[1]) for _, row in chunk.iterrows()]
    print(f"total {total}, shard {shard_id}/{num_shards}: {len(ids)} passages from {start_idx}", flush=True)
    return ids, texts


@torch.no_grad()
def embed(texts, model, tok, maxlen, max_tokens):
    enc = tok([nemotron.format_passage(t) for t in texts], max_length=maxlen, truncation=True)["input_ids"]
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
        e = F.normalize(nemotron.mean_pool(h, mask).float(), dim=-1)
        out[idx] = e.cpu().numpy().astype(np.float16)
        i += len(idx)
        nb += 1
        if nb % 200 == 0:
            print(f"{i}/{len(order)} ({i / (time.time() - t0):.0f}/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix", default="passages")
    ap.add_argument("--shard_id", type=int, required=True)
    ap.add_argument("--num_shards", type=int, default=32)
    ap.add_argument("--passage_maxlength", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=131072, help="tokens per forward batch")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: embed only the first n passages, no save")
    args = ap.parse_args()

    out_file = os.path.join(args.output_dir, f"{args.prefix}_{args.shard_id:02d}")
    if os.path.exists(out_file) and not args.limit:
        print(f"{out_file} exists, skipping")
        return
    ids, texts = read_shard(args.passages, args.shard_id, args.num_shards)
    if args.limit:
        ids, texts = ids[:args.limit], texts[:args.limit]
    tok = nemotron.load_tokenizer()
    model = nemotron.load_backbone(grad_ckpt=False).cuda().eval()
    emb = embed(texts, model, tok, args.passage_maxlength, args.max_tokens)
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
