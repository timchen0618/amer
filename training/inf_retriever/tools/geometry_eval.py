"""Query/gold geometry of one checkpoint on the full corpus (CPU; run before the corpus embeddings are deleted).

Inputs: the query embeddings retrieval used (retrieval_inf.py --save_query_embeddings: (n, k, d) multi-query
or (n, d) single-query), the checkpoint's corpus embedding shards (gen_embed_new.py pickles of (ids, emb)),
and the eval jsonl. Gold clusters come from `ground_truths` (list of clusters) or `positive_ctxs` (list of
clusters, or a flat list = one cluster per passage); they need corpus ids, so AmbigQA test (no ids) is skipped.

Each gold cluster is represented by its first listed passage that is in the corpus ("rep"). All cosines are
between L2-normalized vectors. Writes a json with:
  align_cos, step_{j}_align_cos  query embedding j vs the gold cluster it is matched to by a Hungarian
                                 assignment (rectangular when k != #clusters) maximizing total cosine; the
                                 cosine to a cluster is the max over its passages (any passage counts in eval)
  align_margin                   matched cosine minus the best cosine to any other cluster of the query
  query_pair_cos, _min           mean / min cosine between a query's k embeddings (k > 1)
  query_rand_cos                 mean cosine between query embeddings and random corpus passages (scale reference)
  gold_pair_cos, _min            mean / min cosine between a query's gold reps (queries with >= 2 reps)
  gold_rand_cos                  mean cosine between gold reps and random corpus passages
  gold_cohesion                  per-query gold_pair_cos - gold_rand_cos, averaged (comparable across doc spaces)
  sibling_rank_*                 per gold rep: number of corpus passages closer to it than its nearest sibling
                                 rep (another gold cluster of the same query); median / quartiles / fractions
Per-query values are in <out>.per_query.jsonl.

  python training/inf_retriever/tools/geometry_eval.py --qemb <npy> --emb_glob '<dir>/passages_*' \
      --data <eval jsonl> --out <json>
"""
import argparse
import glob
import json
import pickle
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def gold_clusters(ex):
    """List of clusters (lists of passage ids) for one example, or None if the golds have no ids."""
    golds = ex.get("ground_truths") or ex.get("positive_ctxs") or []
    clusters = [[g.get("id") for g in c] if isinstance(c, list) else [c.get("id")] for c in golds]
    clusters = [[str(i) for i in c if i is not None] for c in clusters]
    return clusters if any(clusters) else None


def load_shard(path):
    ids, emb = pickle.load(open(path, "rb"))
    emb = torch.from_numpy(np.asarray(emb, dtype=np.float32))
    return [str(i) for i in ids], emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-12)


def main(a):
    torch.set_num_threads(a.threads)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    data = [json.loads(l) for l in open(a.data)]
    clusters = [gold_clusters(ex) for ex in data]
    if all(c is None for c in clusters):
        json.dump({"skipped": "gold passages have no corpus ids", "data": a.data}, open(a.out, "w"), indent=1)
        log(f"skipped: {a.data} has no gold ids"); return
    q = np.load(a.qemb).astype(np.float32)
    if q.ndim == 2: q = q[:, None]
    assert len(q) == len(data), (q.shape, len(data))
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    n, k, d = q.shape
    files = sorted(glob.glob(a.emb_glob)); assert files, a.emb_glob
    need = {g for cl in clusters if cl for c in cl for g in c}

    # pass 1: gold vectors and a random sample of passages
    rng = np.random.default_rng(0)
    gv, rand = {}, []
    for f in files:
        ids, emb = load_shard(f)
        for i, x in enumerate(ids):
            if x in need: gv[x] = emb[i]
        rand.append(emb[rng.choice(len(ids), min(len(ids), a.n_random // len(files) + 1), replace=False)])
        log(f"pass 1 {f.split('/')[-1]}: {len(gv)} / {len(need)} gold ids found")
    rand = torch.cat(rand)
    qt = torch.from_numpy(q)
    query_rand = float((qt.reshape(-1, d) @ rand.T).mean())

    clusters_found = []                            # per query: its clusters restricted to passages in the corpus
    per_query, reps, rep_owner = [], [], []        # reps: gold rep vectors; rep_owner[r] = (query, cluster)
    for i in range(n):
        cl = [[g for g in c if g in gv] for c in (clusters[i] or [])]
        cl = [c for c in cl if c]
        clusters_found.append(cl)
        rec = {"idx": i, "n_clusters": len(clusters[i] or []), "n_clusters_found": len(cl)}
        if cl:
            cos = np.stack([(torch.stack([gv[g] for g in c]) @ qt[i].T).max(0).values.numpy() for c in cl], 1)  # (k, m)
            rows, cols = linear_sum_assignment(cos, maximize=True)
            rec["step_align_cos"] = {int(r): float(cos[r, c]) for r, c in zip(rows, cols)}
            if len(cl) > 1:
                other = cos.copy(); other[rows, cols] = -np.inf
                rec["align_margin"] = float(np.mean(cos[rows, cols] - other[rows].max(1)))
            rv = torch.stack([gv[c[0]] for c in cl])
            if len(cl) > 1:
                pp = (rv @ rv.T)[~torch.eye(len(cl), dtype=torch.bool)]
                rec["gold_pair_cos"], rec["gold_pair_cos_min"] = float(pp.mean()), float(pp.min())
            rec["gold_rand_cos"] = float((rv @ rand.T).mean())
            if "gold_pair_cos" in rec: rec["gold_cohesion"] = rec["gold_pair_cos"] - rec["gold_rand_cos"]
            for c in range(len(cl)):
                reps.append(rv[c]); rep_owner.append((i, c))
        if k > 1:
            qq = (qt[i] @ qt[i].T)[~torch.eye(k, dtype=torch.bool)]
            rec["query_pair_cos"], rec["query_pair_cos_min"] = float(qq.mean()), float(qq.min())
        per_query.append(rec)

    # pass 2: nearest-sibling rank of every gold rep that has a sibling
    R = torch.stack(reps)
    thr = torch.full((len(reps),), float("nan"))
    by_query = {}
    for r, (i, c) in enumerate(rep_owner): by_query.setdefault(i, []).append(r)
    for rs in by_query.values():
        if len(rs) < 2: continue
        s = R[rs] @ R[rs].T; s.fill_diagonal_(-2.0)
        thr[rs] = s.max(1).values
    has_sib = ~torch.isnan(thr)
    sib_idx = has_sib.nonzero().squeeze(1)
    # The rep itself and its siblings are excluded by id, not by "-1": the threshold and the scan are
    # separate matmuls, so rounding can put the nearest sibling just above its own threshold.
    own = {}                                          # passage id -> positions in sib_idx whose own/sibling set has it
    rep_ids = [clusters_found[i][c][0] for i, c in rep_owner]
    for pos, r in enumerate(sib_idx.tolist()):
        for r2 in by_query[rep_owner[r][0]]:
            own.setdefault(rep_ids[r2], []).append(pos)
    count = torch.zeros(len(sib_idx), dtype=torch.long)
    for f in files:
        ids, emb = load_shard(f)
        excl = [(pos, col) for col, x in enumerate(ids) if x in own for pos in own[x]]
        for b in range(0, len(sib_idx), a.rep_chunk):
            ix = sib_idx[b:b + a.rep_chunk]
            above = (R[ix] @ emb.T) > thr[ix, None]
            for pos, col in excl:
                if b <= pos < b + a.rep_chunk and above[pos - b, col]: above[pos - b, col] = False
            count[b:b + a.rep_chunk] += above.sum(1)
        log(f"pass 2 {f.split('/')[-1]}")
    sib_rank = count.numpy()

    def mean(key):
        v = [p[key] for p in per_query if key in p]
        return float(np.mean(v)) if v else None
    step_vals = {}
    for p in per_query:
        for j, v in p.get("step_align_cos", {}).items(): step_vals.setdefault(j, []).append(v)
    out = {
        "data": a.data, "qemb": a.qemb, "emb_glob": a.emb_glob, "n_queries": n, "k": k,
        "gold_ids_found": len(gv), "gold_ids_needed": len(need),
        "queries_with_all_clusters": int(sum(p["n_clusters_found"] == p["n_clusters"] > 0 for p in per_query)),
        "align_cos": float(np.mean([np.mean(list(p["step_align_cos"].values())) for p in per_query if p.get("step_align_cos")])),
        **{f"step_{j}_align_cos": float(np.mean(v)) for j, v in sorted(step_vals.items())},
        "align_margin": mean("align_margin"),
        "query_pair_cos": mean("query_pair_cos"), "query_pair_cos_min": mean("query_pair_cos_min"),
        "query_rand_cos": query_rand,
        "gold_pair_cos": mean("gold_pair_cos"), "gold_pair_cos_min": mean("gold_pair_cos_min"),
        "gold_rand_cos": mean("gold_rand_cos"),
        "gold_cohesion": mean("gold_cohesion"),
        "n_gold_reps_with_sibling": int(len(sib_rank)),
        "sibling_rank_median": float(np.median(sib_rank)) if len(sib_rank) else None,
        "sibling_rank_p25": float(np.percentile(sib_rank, 25)) if len(sib_rank) else None,
        "sibling_rank_p75": float(np.percentile(sib_rank, 75)) if len(sib_rank) else None,
        "sibling_rank_frac_lt10": float(np.mean(sib_rank < 10)) if len(sib_rank) else None,
        "sibling_rank_frac_lt100": float(np.mean(sib_rank < 100)) if len(sib_rank) else None,
    }
    # attach sibling ranks per query (in cluster order)
    for pos, r in enumerate(sib_idx.tolist()):
        i, _ = rep_owner[r]
        per_query[i].setdefault("sibling_rank", []).append(int(sib_rank[pos]))
    json.dump(out, open(a.out, "w"), indent=1)
    with open(a.out.replace(".json", "") + ".per_query.jsonl", "w") as f:
        for p in per_query: f.write(json.dumps(p) + "\n")
    log("done: " + json.dumps({x: out[x] for x in ("align_cos", "align_margin", "query_pair_cos", "gold_cohesion", "sibling_rank_median") if x in out}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qemb", required=True)
    ap.add_argument("--emb_glob", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_random", type=int, default=20000)
    ap.add_argument("--rep_chunk", type=int, default=512)
    ap.add_argument("--threads", type=int, default=16)
    main(ap.parse_args())
