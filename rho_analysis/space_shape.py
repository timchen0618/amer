"""Step 1 -- How squashed is the document embedding space?

Answers: does cosine distance 0.10 between same-query golds mean "these
documents are nearly identical", or "this space is anisotropic and 0.10 is
actually a large gap"? Real contextual embedding spaces occupy a narrow cone
(Ethayarajh 2019; Gao et al. 2019), which compresses all cosine distances
toward zero and makes cross-space distance comparisons meaningless.

Usage:
    python space_shape.py --corpus corpus_emb.npy [--sample 200000]
"""
import argparse
import json

import numpy as np

from io_utils import l2norm, sample_rows


def space_shape(X, sample=200_000, seed=0):
    Z = l2norm(sample_rows(X, sample, seed)).astype(np.float64)
    mu = Z.mean(0)
    lam = np.linalg.svd(Z - mu, compute_uv=False) ** 2 / (len(Z) - 1)
    lam = np.maximum(lam, 0.0)
    p = lam / lam.sum()
    pnz = p[p > 0]
    d = Z.shape[1]
    return {
        "n_sampled": int(len(Z)),
        "dim": int(d),
        "participation_ratio": float(lam.sum() ** 2 / (lam**2).sum()),
        "effective_rank": float(np.exp(-(pnz * np.log(pnz)).sum())),
        "pr_over_dim": float((lam.sum() ** 2 / (lam**2).sum()) / d),
        "mean_norm": float(np.linalg.norm(mu)),
        "top1_var_frac": float(p[0]),
        "top10_var_frac": float(p[:10].sum()),
        "top50_var_frac": float(p[:50].sum()),
    }


def interpret(s):
    lines = []
    if s["mean_norm"] > 0.7:
        lines.append(
            f"CONE COLLAPSE: mean direction has norm {s['mean_norm']:.2f}. Every pair of "
            "documents shares a large common component, so all cosine similarities are "
            "inflated toward 1. Raw cosine gaps understate real semantic separation."
        )
    elif s["mean_norm"] > 0.4:
        lines.append(f"Moderate common component (mean norm {s['mean_norm']:.2f}).")
    else:
        lines.append(f"Space is roughly centered (mean norm {s['mean_norm']:.2f}).")

    if s["pr_over_dim"] < 0.10:
        lines.append(
            f"RANK COLLAPSE: participation ratio {s['participation_ratio']:.1f} of "
            f"{s['dim']} dims ({s['pr_over_dim']:.1%}). Distances are dominated by a few "
            "directions; whatever separates same-query golds likely lives in low-variance "
            "directions that barely register. -> whitening.py is worth running, and the "
            "document space may be the binding constraint, not the query count."
        )
    else:
        lines.append(
            f"Space uses {s['participation_ratio']:.1f}/{s['dim']} dims "
            f"({s['pr_over_dim']:.1%}); not severely rank-collapsed. If rho is still small "
            "after this, the benchmarks really are homogeneous."
        )
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="(N, d) .npy of document embeddings")
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write stats as JSON")
    a = ap.parse_args()

    X = np.load(a.corpus, mmap_mode="r")
    s = space_shape(X, a.sample, a.seed)
    print(json.dumps(s, indent=2))
    print()
    for line in interpret(s):
        print("  " + line)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(s, f, indent=2)


if __name__ == "__main__":
    main()
