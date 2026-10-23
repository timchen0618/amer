"""The paper's large / small gold-distance subsets, rebuilt for the QAMPARI dev set (objective audit, 2026-10-23).

analysis/group_by_distance.py scores each question by the mean pairwise L2 distance between its gold passages,
embedded with base INF-Retriever (SentenceTransformer, 'title text', unit-normalized; one passage per gold
cluster), and splits at quantiles. The paper's index files (data/qampari/{large,small}_distance_indices_inf.txt,
over its 1000-question file) are thirds. This script
  1. recomputes the distances for the paper's file and checks the thirds reproduce those index files;
  2. scores our full dev set (data/phaseA/qampari_cleanv2dev500.jsonl) the same way and applies the paper file's
     thresholds, writing data/qampari/dev500_{large,small}_distance_indices_inf.txt (indices into the dev file).
Dev subsets can then select checkpoints for large-distance behaviour without touching test.

    python training/inf_retriever/tools/dev_distance_subsets.py   (1 GPU)
"""
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PAPER = "/scratch/hc3337/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl"
PAPER_IDX = {s: f"data/qampari/{s}_distance_indices_inf.txt" for s in ("large", "small")}
DEV = "data/phaseA/qampari_cleanv2dev500.jsonl"
OUT = {s: f"data/qampari/dev500_{s}_distance_indices_inf.txt" for s in ("large", "small")}


def golds(inst):
    key = "ground_truths" if inst.get("ground_truths") else "positive_ctxs"
    g = inst[key]
    return [c[0] if isinstance(c, list) else c for c in g]


def mean_pairwise(model, passages):
    texts = [str(p.get("title", "")) + " " + str(p["text"]) for p in passages]
    e = torch.from_numpy(model.encode(texts, batch_size=128, normalize_embeddings=True))
    d = torch.triu(torch.cdist(e, e), diagonal=1)
    nz = d[d.nonzero(as_tuple=True)]
    return nz.mean().item() if nz.numel() else 0.0


def main():
    model = SentenceTransformer("infly/inf-retriever-v1-1.5b", trust_remote_code=True, device="cuda")
    model.max_seq_length = 8192
    paper = [json.loads(l) for l in open(PAPER)]
    pd_ = np.array([mean_pairwise(model, golds(x)) for x in paper])
    lo, hi = np.quantile(pd_, 1 / 3, method="nearest"), np.quantile(pd_, 2 / 3, method="nearest")
    mine = {"large": set(np.where(pd_ > hi)[0].tolist()), "small": set(np.where(pd_ < lo)[0].tolist())}
    for s, p in PAPER_IDX.items():
        ref = {int(x) for x in open(p).read().split()}
        print(f"paper file {s}: reproduced {len(mine[s])}, released {len(ref)}, overlap {len(mine[s] & ref)}", flush=True)
    dev = [json.loads(l) for l in open(DEV)]
    dd = np.array([mean_pairwise(model, golds(x)) for x in dev])
    sub = {"large": np.where(dd > hi)[0].tolist(), "small": np.where(dd < lo)[0].tolist()}
    for s, idx in sub.items():
        open(OUT[s], "w").write("\n".join(map(str, idx)) + "\n")
        print(f"dev {s}: {len(idx)} of {len(dev)} -> {OUT[s]}", flush=True)
    print(f"thresholds (paper-file thirds): small < {lo:.4f}, large > {hi:.4f}; dev distance median {np.median(dd):.4f}")


if __name__ == "__main__":
    main()
