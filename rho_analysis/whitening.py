"""Step 3 -- Whiten the embedding space and recompute rho.

If the corpus is rank-collapsed, distances are dominated by a few high-variance
directions and the directions that separate "released 4 Nov" from "released 16
Nov" contribute almost nothing. Whitening rescales each principal direction to
unit variance so low-variance semantic axes count as much as the dominant ones.

IMPORTANT: a linear map applied to documents ALONE is vacuous -- it folds into
the query side and cannot change any ranking. Whitening must be applied to both
sides, which changes the metric to a Mahalanobis form and genuinely moves
nearest neighbors. This script therefore transforms corpus and queries together.

Read the result as a diagnostic, not a proposed system:
  rho unchanged after whitening   -> golds really are semantically close;
                                     the benchmark cannot show what AMER is for
  rho rises sharply               -> golds ARE separable but the encoder's
                                     geometry hides it. The binding constraint
                                     is document-side, not query count -- which
                                     no query-side method (expansion,
                                     decomposition, MMR, AMER) can fix.

Usage:
    python whitening.py --corpus corpus_emb.npy --out-dir whitened/ \
        --queries q_single.npy --shrink 0.05
"""
import argparse
import os

import numpy as np

from io_utils import l2norm, sample_rows


def fit_whitener(X, sample=200_000, shrink=0.05, n_components=None, seed=0):
    """Return (mu, W) with W of shape (d, r). Apply as (x - mu) @ W."""
    Z = l2norm(sample_rows(X, sample, seed)).astype(np.float64)
    mu = Z.mean(0)
    Zc = Z - mu
    cov = (Zc.T @ Zc) / (len(Zc) - 1)
    lam, V = np.linalg.eigh(cov)
    lam, V = lam[::-1], V[:, ::-1]
    if n_components:
        lam, V = lam[:n_components], V[:, :n_components]
    # shrinkage toward the mean eigenvalue: keeps the tail from exploding
    lam_s = (1 - shrink) * lam + shrink * lam.mean()
    W = V / np.sqrt(np.maximum(lam_s, 1e-12))
    return mu.astype(np.float32), W.astype(np.float32)


def apply_whitener(X, mu, W, batch=100_000, renorm=True):
    out = np.empty((len(X), W.shape[1]), dtype=np.float32)
    for i in range(0, len(X), batch):
        B = l2norm(np.asarray(X[i : i + batch], dtype=np.float32))
        out[i : i + batch] = (B - mu) @ W
    return l2norm(out) if renorm else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", nargs="*", default=[], help="query .npy files to transform too")
    ap.add_argument("--out-dir", default="whitened")
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--n-components", type=int, default=None)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    X = np.load(a.corpus, mmap_mode="r")
    mu, W = fit_whitener(X, a.sample, a.shrink, a.n_components)
    np.savez(os.path.join(a.out_dir, "whitener.npz"), mu=mu, W=W)

    cp = os.path.join(a.out_dir, os.path.basename(a.corpus))
    np.save(cp, apply_whitener(X, mu, W))
    print(f"corpus -> {cp}")

    for q in a.queries:
        Q = np.load(q)
        shape = Q.shape
        Qf = Q.reshape(-1, shape[-1])  # handles (Q,d) and (Q,m,d)
        out = apply_whitener(Qf, mu, W).reshape(*shape[:-1], W.shape[1])
        p = os.path.join(a.out_dir, os.path.basename(q))
        np.save(p, out)
        print(f"queries -> {p}")

    print("\n  Now re-run space_shape.py and rho.py on the whitened files and compare.")


if __name__ == "__main__":
    main()
