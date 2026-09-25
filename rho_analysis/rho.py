"""Step 2 -- rho = (gold-set diameter) / (retrieval-ball radius).

Raw pairwise distance between golds is the wrong x-axis: golds 0.3 apart are
trivially coverable in a sparse corpus region and hopeless in a dense one. What
determines single-vector feasibility is the gold-set diameter RELATIVE to how
far you have to reach to collect k documents.

  D(q)   = max pairwise distance among the m golds
  r_k(q) = distance from the query vector to the k-th nearest corpus document
  rho    = D / r_k

A set of diameter D needs an enclosing ball of radius >= D/2, so rho <= 2 is
NECESSARY for a single vector to cover all golds at rank k. In high dimensions
Jung's theorem gives the sufficient bound D/sqrt(2), i.e. rho <= ~1.41 always
works. Between 1.41 and 2 it depends on the configuration.

  rho < 1.41   single vector suffices in principle -> multi-vector should NOT help
  rho > 2      no single vector can do it -> multi-vector is the only route

We compute r_k at two centers:
  - the actual baseline query vector      (what your system does)
  - the gold centroid                     (the best any single vector could do)
The centroid version is the clean necessary-condition test; a large gap between
the two means your single-vector model is misplaced rather than fundamentally
limited, which is a different paper.

We also report r_{k/m}: the shrunken ball each of m query vectors gets when they
split a top-k budget round-robin. r_k / r_{k/m} is the price of going
multi-vector, and explains why diversity hurts when targets are similar.

Usage:
    python rho.py --corpus corpus_emb.npy --golds golds.jsonl \
        --queries q_single.npy --qids qids.json --k 100 --m 5 --out rho.jsonl
"""
import argparse
import json

import numpy as np

from io_utils import knn_radii, l2norm, load_golds, pairwise_diameter


def compute_rho(corpus, golds, query_vecs=None, k=100, m=5, use_faiss=True):
    """corpus: (N,d) normalized. golds: [(qid, gold_ids)]. query_vecs: (Q,d) aligned."""
    qids = [g[0] for g in golds]
    diam, centroids, n_gold = [], [], []
    for _, gid in golds:
        V = l2norm(np.asarray(corpus[gid], dtype=np.float32))
        diam.append(pairwise_diameter(V))
        c = V.mean(0)
        centroids.append(c / max(np.linalg.norm(c), 1e-12))
        n_gold.append(len(gid))
    diam = np.asarray(diam, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)

    ks = sorted({k, max(1, k // max(m, 1))})
    r_c = knn_radii(centroids, corpus, ks, use_faiss=use_faiss)
    r_ck = r_c[:, ks.index(k)]
    r_ckm = r_c[:, ks.index(max(1, k // max(m, 1)))]

    rows = []
    if query_vecs is not None:
        r_q = knn_radii(l2norm(query_vecs), corpus, [k], use_faiss=use_faiss)[:, 0]
    else:
        r_q = np.full(len(qids), np.nan, dtype=np.float32)

    for i, qid in enumerate(qids):
        rows.append(
            {
                "qid": qid,
                "n_gold": int(n_gold[i]),
                "diameter": float(diam[i]),
                "r_k_centroid": float(r_ck[i]),
                "r_k_over_m_centroid": float(r_ckm[i]),
                "r_k_query": float(r_q[i]),
                "rho_oracle": float(diam[i] / max(r_ck[i], 1e-9)),
                "rho_query": float(diam[i] / max(r_q[i], 1e-9)),
                "budget_shrink": float(r_ck[i] / max(r_ckm[i], 1e-9)),
            }
        )
    return rows


def summarize(rows, key="rho_oracle"):
    v = np.asarray([r[key] for r in rows], dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {}
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.90)),
        "frac_below_1.41_single_vector_sufficient": float((v < np.sqrt(2)).mean()),
        "frac_above_2.0_single_vector_impossible": float((v > 2.0).mean()),
        "mean_budget_shrink": float(
            np.mean([r["budget_shrink"] for r in rows if np.isfinite(r["budget_shrink"])])
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--golds", required=True)
    ap.add_argument("--queries", default=None, help="(Q,d) single-query baseline vectors")
    ap.add_argument("--qids", default=None, help="JSON list aligning --queries rows to qids")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--no-faiss", action="store_true")
    ap.add_argument("--out", default="rho.jsonl")
    a = ap.parse_args()

    corpus = l2norm(np.load(a.corpus, mmap_mode="r"))
    golds = load_golds(a.golds)

    qv = None
    if a.queries:
        Q = np.load(a.queries)
        if a.qids:
            with open(a.qids) as f:
                order = {q: i for i, q in enumerate(json.load(f))}
            Q = Q[[order[g[0]] for g in golds]]
        qv = Q

    rows = compute_rho(corpus, golds, qv, a.k, a.m, use_faiss=not a.no_faiss)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"wrote {a.out}  (n={len(rows)})")
    for key in ("rho_oracle", "rho_query"):
        s = summarize(rows, key)
        if s:
            print(f"\n[{key}]")
            print(json.dumps(s, indent=2))
    print(
        "\n  Plot recall against rho_oracle instead of raw pairwise distance. rho is "
        "comparable across datasets and embedding spaces; raw distance is not."
    )


if __name__ == "__main__":
    main()
