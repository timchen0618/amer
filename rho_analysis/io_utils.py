"""Shared loading / geometry helpers.

Metric convention: everything is L2-normalized, and "distance" means Euclidean
distance on the unit sphere. This is monotone in cosine distance
(d_euc^2 = 2 - 2*cos), so rankings are identical, but Euclidean distance
satisfies the triangle inequality, which the ball-covering argument needs.
"""
import json

import numpy as np


# ---------------------------------------------------------------- loading


def load_emb(path, mmap=True):
    """Load an (N, d) float array of embeddings."""
    X = np.load(path, mmap_mode="r" if mmap else None)
    if X.ndim not in (2, 3):
        raise ValueError(f"{path}: expected 2D (N,d) or 3D (Q,m,d), got {X.shape}")
    return X


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_golds(path):
    """golds.jsonl: one object per query, {"qid": str, "gold_ids": [int, ...]}

    gold_ids are row indices into the corpus embedding matrix.
    """
    rows = load_jsonl(path)
    return [(r["qid"], np.asarray(r["gold_ids"], dtype=np.int64)) for r in rows]


def load_results(path, field="mrecall"):
    """results.jsonl: {"qid": str, "mrecall": 0 or 1}  ->  {qid: value}"""
    return {r["qid"]: float(r[field]) for r in load_jsonl(path)}


# ---------------------------------------------------------------- geometry


def l2norm(X, eps=1e-12):
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


def sample_rows(X, n, seed=0):
    rng = np.random.default_rng(seed)
    if len(X) <= n:
        return np.asarray(X, dtype=np.float32)
    idx = np.sort(rng.choice(len(X), n, replace=False))
    return np.asarray(X[idx], dtype=np.float32)


def pairwise_diameter(V):
    """Max pairwise Euclidean distance within a small set of unit vectors."""
    V = np.asarray(V, dtype=np.float32)
    if len(V) < 2:
        return 0.0
    G = V @ V.T
    d2 = np.maximum(2.0 - 2.0 * G, 0.0)
    return float(np.sqrt(d2.max()))


def knn_radii(queries, corpus, ks, chunk=4096, use_faiss=True):
    """Distance from each query to its k-th nearest corpus vector, for each k in ks.

    Returns (n_queries, len(ks)) array. Both inputs must be L2-normalized.
    """
    ks = sorted(int(k) for k in ks)
    kmax = max(ks)
    take = [k - 1 for k in ks]
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    if use_faiss:
        try:
            import faiss

            index = faiss.IndexFlatIP(corpus.shape[1])
            for i in range(0, len(corpus), 200_000):
                index.add(np.ascontiguousarray(corpus[i : i + 200_000], dtype=np.float32))
            sims, _ = index.search(queries, kmax)
            d = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))
            return d[:, take]
        except ImportError:
            pass

    # numpy fallback: chunked over the corpus, keep a running top-kmax
    out = np.full((len(queries), kmax), np.inf, dtype=np.float32)
    for i in range(0, len(corpus), chunk):
        block = np.asarray(corpus[i : i + chunk], dtype=np.float32)
        sims = queries @ block.T
        d = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))
        merged = np.concatenate([out, d], axis=1)
        merged.partition(kmax - 1, axis=1)
        out = np.sort(merged[:, :kmax], axis=1)
    return out[:, take]


# ---------------------------------------------------------------- stats


def auc(scores, labels):
    """AUROC via Mann-Whitney U. labels in {0,1}."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    pos, neg = labels.sum(), (1 - labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.arange(i + 1, j + 2).mean()
        i = j + 1
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def bootstrap_ci(fn, *arrays, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(fn(*[np.asarray(a)[idx] for a in arrays]))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))
