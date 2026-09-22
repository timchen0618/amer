"""Step 5 (the rigorous version) -- rho_oracle, via a genuine full-corpus k-NN
search centered on the gold centroid, not the baseline's actual query vector.

rho_query (pipeline_query.py) answers "is the baseline's own query vector
close enough to its own k-th nearest neighbor to cover the gold spread." That
conflates two things: geometric necessity (can ANY single vector do this) and
training quality (did THIS model find a good point). rho_oracle isolates the
first question by searching from the best-case point instead -- the centroid
of the actual gold embeddings -- against the real corpus.

This reuses retrieval_base.py's retrieve(), unmodified: it is already a
generic "search these query vectors against these sharded passage
embeddings" function (shard-by-shard, ~2.5GB resident at a time, not the full
~80GB corpus), so gold centroids work as "queries" exactly the same way real
question embeddings do -- no new FAISS code needed.

Requires rho_analysis/outputs/<dataset>/rho_query.jsonl to already exist
(run pipeline_query.py first) -- diameter, m_used, and n_gold_embedded are
reused from there rather than recomputed.

Usage:
    python pipeline_oracle.py --dataset qampari --k 100
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_query import (  # noqa: E402
    DATASETS,
    l2norm_row,
    qtext,
    resolve_gold_ambigqa,
    resolve_gold_qampari,
    scan_shards_for_ids,
    sim_to_dist,
)
from src.eval_utils import SimpleTokenizer, read_jsonl  # noqa: E402
from retrieval_base import retrieve  # noqa: E402

EMBEDDING_SIZE = 1536
NUM_SHARDS = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--gold-scan-depth", type=int, default=500)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--ks",
        default="5,10,20,50,100",
        help="comma-separated list of k values to report rho_oracle/r_k_oracle at, computed from a "
        "single search with top_k_per_query=max(ks) -- no extra corpus searches needed. Always "
        "includes --k.",
    )
    a = ap.parse_args()
    ks = sorted(set(int(x) for x in a.ks.split(",")) | {a.k})
    k_search = max(ks)

    cfg = DATASETS[a.dataset]
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", a.dataset)
    os.makedirs(out_dir, exist_ok=True)

    rho_query_path = os.path.join(out_dir, "rho_query.jsonl")
    rho_query_rows = {r["qid"]: r for r in read_jsonl(rho_query_path)}

    eval_path = os.path.join(ROOT, cfg["eval_path"])
    standard_path = os.path.join(ROOT, cfg["standard_path"])
    eval_rows = read_jsonl(eval_path)
    baseline_rows = read_jsonl(standard_path)
    assert len(eval_rows) == len(baseline_rows), (len(eval_rows), len(baseline_rows))

    tok = SimpleTokenizer() if not cfg["has_gold_id"] else None

    # ---- 1. resolve gold ids again (same logic as pipeline_query.py)
    per_example_gold_ids = []
    for ex, base in zip(eval_rows, baseline_rows):
        if cfg["has_gold_id"]:
            gids = resolve_gold_qampari(ex)
        else:
            gids, _n_asp = resolve_gold_ambigqa(ex, base["ctxs"], tok, a.gold_scan_depth)
        per_example_gold_ids.append(gids)

    needed_ids = set()
    for gids in per_example_gold_ids:
        needed_ids.update(gids)
    print(f"[{a.dataset}] {len(needed_ids)} distinct gold ids to look up for centroids", flush=True)

    # ---- 2. pull gold embeddings, build one centroid per example
    id2emb = scan_shards_for_ids(cfg["shard_glob"], needed_ids)

    qids = []
    centroids = []
    ms = []
    for ex, gids in zip(eval_rows, per_example_gold_ids):
        qid = qtext(ex)
        vecs = [id2emb[g] for g in gids if g in id2emb]
        if not vecs or qid not in rho_query_rows:
            continue  # no resolvable gold -- excluded, same convention as rho_query's near-zero-coverage tail
        centroid = l2norm_row(np.mean(np.stack([l2norm_row(v) for v in vecs]), axis=0))
        qids.append(qid)
        centroids.append(centroid)
        ms.append(max(len(vecs), 1))

    print(f"[{a.dataset}] {len(qids)}/{len(eval_rows)} examples have a resolvable gold centroid", flush=True)
    query_matrix = np.stack(centroids).astype(np.float32)

    # ---- 3. one real full-corpus k-NN search, centroids as queries -- request the largest k we
    # need; every smaller k is then a free slice of the same ranked list, no extra searches.
    shard_glob = os.path.join(ROOT, cfg["shard_glob"])
    print(f"[{a.dataset}] searching {shard_glob} with {len(qids)} centroid queries, k_search={k_search}", flush=True)
    top_ids_and_scores = retrieve(
        query_matrix,
        NUM_SHARDS,
        shard_glob,
        passage_id_map=None,
        embedding_size=EMBEDDING_SIZE,
        top_k_per_query=k_search,
        top_k=k_search,
        save_or_load_index=False,
        use_gpu=False,
    )
    assert len(top_ids_and_scores) == len(qids), (len(top_ids_and_scores), len(qids))

    # ---- 4. r_k_oracle, r_{k/m}_oracle, rho_oracle, for every k in ks
    rows = []
    for qid, m, (ids_, scores_) in zip(qids, ms, top_ids_and_scores):
        base_row = rho_query_rows[qid]
        diam = base_row["diameter"]

        by_k = {}
        for kk in ks:
            k_eff = min(kk, len(scores_))
            km_eff = max(1, min(kk // m, len(scores_)))
            r_k = sim_to_dist(scores_[k_eff - 1])
            r_km = sim_to_dist(scores_[km_eff - 1])
            by_k[str(kk)] = {
                "r_k_oracle": r_k,
                "r_k_over_m_oracle": r_km,
                "rho_oracle": float(diam / max(r_k, 1e-9)),
                "budget_shrink_oracle": float(r_k / max(r_km, 1e-9)),
            }

        primary = by_k[str(a.k)]
        rows.append(
            {
                "qid": qid,
                "n_aspects_total": base_row["n_aspects_total"],
                "n_aspects_resolved": base_row["n_aspects_resolved"],
                "gold_coverage": base_row["gold_coverage"],
                "n_gold_embedded": base_row["n_gold_embedded"],
                "diameter": diam,
                "m_used": m,
                "r_k_oracle": primary["r_k_oracle"],
                "r_k_over_m_oracle": primary["r_k_over_m_oracle"],
                "rho_oracle": primary["rho_oracle"],
                "budget_shrink_oracle": primary["budget_shrink_oracle"],
                "by_k": by_k,
                # for direct side-by-side comparison against the query-centered variant
                "rho_query": base_row["rho_query"],
                "r_k_query": base_row["r_k_query"],
            }
        )

    out_path = os.path.join(out_dir, "rho_oracle.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{a.dataset}] wrote {out_path} (n={len(rows)})", flush=True)

    rho = np.array([r["rho_oracle"] for r in rows])
    print(
        f"[{a.dataset}] rho_oracle: median={np.median(rho):.3f} p95={np.percentile(rho, 95):.3f} "
        f"max={rho.max():.3f} frac>=sqrt2={float(np.mean(rho >= np.sqrt(2))):.4f}",
        flush=True,
    )
    print(f"[{a.dataset}] done.", flush=True)


if __name__ == "__main__":
    main()
