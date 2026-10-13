"""F0 (Session F, objective audit 2026-10-13): headroom bounds and gold geometry in the frozen base document space.

Every QAMPARI gold cluster here is a single passage, so a gold's embedding is its passage's row in the shared frozen
corpus embedding (wikipedia_embeddings/qampari/shared_frozen_base_512). No model is run; the untrained base model's
own question embedding is scored separately (retrieve_ckpt.sbatch with the HF model).

Bounds (each list is the top 500 over the full corpus; multi-query lists are fused with RRF, 1 / (60 + rank), as in
retrieval_inf.aggregate_rrf; scored at top 100 by gold id, as tools/perstep_eval.py):
  gold_query     5 golds as queries (all golds when n <= 5, so trivially 100 there; for n > 5 the gold nearest each of 5
                 spherical k-means centres), RRF. Optimistic: a gold used as a query finds itself.
  centroid       one query = normalized mean of all golds. Optimistic single-query counterpart of gold_query.
  loo_single     gold i is searched with one query = normalized mean of the other n - 1 golds.
  loo_multi      gold i is searched with min(5, n - 1) queries = k-means centres of the other n - 1 golds, RRF.
For the leave-one-out bounds a question counts for MRecall when every gold is found by its own held-out query.

Geometry: quantiles of per-question mean gold-gold cosine and of gold-pair cosine, shares of gold pairs above
0.90 / 0.95, against random passage pairs.

    python training/inf_retriever/tools/f0_bounds.py --out results/phaseC/frozen_f/f0   (GPU job: f0_bounds.sbatch)
"""
import argparse, glob, json, os, pickle, time
import numpy as np
import torch

DATA = {"dev": "data/phaseA/qampari_cleanv2dev500.jsonl", "test": "data/amer_data/eval_data/qampari.jsonl"}
EMB = "wikipedia_embeddings/qampari/shared_frozen_base_512"
TOPK, DEPTH, K = 100, 500, 5


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def shards():
    return sorted(glob.glob(f"{EMB}/passages_*"))


def kmeans(x, k, seed=0, iters=50, restarts=5):
    """Spherical k-means on unit rows of x (n x d); returns unit centres (k x d)."""
    best, best_obj = None, -1e9
    for r in range(restarts):
        rng = np.random.default_rng(seed + r)
        c = x[rng.choice(len(x), k, replace=False)].copy()
        for _ in range(iters):
            a = (x @ c.T).argmax(1)
            new = np.stack([x[a == j].sum(0) if (a == j).any() else c[j] for j in range(k)])
            new /= np.linalg.norm(new, axis=1, keepdims=True)
            if np.allclose(new, c):
                break
            c = new
        obj = (x @ c.T).max(1).sum()
        if obj > best_obj:
            best, best_obj = c, obj
    return best


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rrf(lists):
    s = {}
    for ids in lists:
        for r, d in enumerate(ids, 1):
            s[d] = s.get(d, 0.0) + 1.0 / (r + 60)
    return [d for d, _ in sorted(s.items(), key=lambda t: -t[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/phaseC/frozen_f/f0")
    ap.add_argument("--chunk", type=int, default=1024)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")

    qs = {s: [json.loads(l) for l in open(p)] for s, p in DATA.items()}
    golds = {s: [[c[0]["id"] if isinstance(c, list) else c["id"] for c in (q.get("positive_ctxs") or q.get("ground_truths"))]
                 for q in qs[s]] for s in qs}
    need = {str(g) for s in golds for q in golds[s] for g in q}

    # pass 1: gold embeddings and a random passage sample
    files, emb_of, rand = shards(), {}, []
    rng = np.random.default_rng(0)
    for f in files:
        ids, e = pickle.load(open(f, "rb"))
        ids = [str(i) for i in ids]
        for j, i in enumerate(ids):
            if i in need:
                emb_of[i] = np.asarray(e[j], dtype=np.float32)
        rand.append(np.asarray(e[rng.choice(len(ids), 200, replace=False)], dtype=np.float32))
        log("pass 1", os.path.basename(f), len(emb_of), "/", len(need))
    missing = need - set(emb_of)
    log("gold ids missing from corpus:", len(missing))
    rand = unit(np.concatenate(rand))

    # queries: (split, question idx, kind, target gold idx or -1, list of query vectors)
    groups, qvecs = [], []
    for s in golds:
        for qi, gl in enumerate(golds[s]):
            gl = [g for g in map(str, gl) if g in emb_of]
            G = unit(np.stack([emb_of[g] for g in gl]))
            n = len(G)
            if n <= K:
                gq = G
            else:
                c, used = kmeans(G, K, seed=qi), []
                for row in c @ G.T:  # nearest not-yet-used gold to each centre: 5 distinct golds
                    used.append(next(j for j in np.argsort(-row) if j not in used))
                gq = G[used]
            groups.append((s, qi, "gold_query", -1, len(qvecs), len(gq))); qvecs.append(gq)
            groups.append((s, qi, "centroid", -1, len(qvecs), 1)); qvecs.append(unit(G.mean(0))[None])
            for i in range(n):
                rest = np.delete(G, i, 0)
                groups.append((s, qi, "loo_single", i, len(qvecs), 1)); qvecs.append(unit(rest.mean(0))[None])
                m = rest if len(rest) <= K else kmeans(rest, K, seed=1000 + qi * 31 + i)
                groups.append((s, qi, "loo_multi", i, len(qvecs), len(m))); qvecs.append(m)
    starts = np.cumsum([0] + [len(v) for v in qvecs])
    Q = torch.from_numpy(np.concatenate(qvecs)).to(dev, torch.float16)
    log("queries", Q.shape[0], "groups", len(groups))

    # pass 2: exact top-DEPTH over the corpus
    best_s = torch.full((Q.shape[0], DEPTH), -1e4, device=dev, dtype=torch.float32)
    best_i = torch.full((Q.shape[0], DEPTH), -1, device=dev, dtype=torch.long)
    all_ids, off = [], 0
    for f in files:
        ids, e = pickle.load(open(f, "rb"))
        E = torch.from_numpy(np.asarray(e)).to(dev, torch.float16)
        for a in range(0, Q.shape[0], args.chunk):
            sc = (Q[a:a + args.chunk] @ E.T).float()
            v, ix = sc.topk(DEPTH, dim=1)
            s2 = torch.cat([best_s[a:a + args.chunk], v], 1)
            i2 = torch.cat([best_i[a:a + args.chunk], ix + off], 1)
            v, o = s2.topk(DEPTH, dim=1)
            best_s[a:a + args.chunk], best_i[a:a + args.chunk] = v, i2.gather(1, o)
        all_ids.extend(str(i) for i in ids); off += len(ids)
        del E
        log("pass 2", os.path.basename(f))
    best_i = best_i.cpu().numpy()
    all_ids = np.array(all_ids, dtype=object)

    def lists(start, cnt):
        return [list(all_ids[best_i[r]]) for r in range(starts[start], starts[start] + cnt)]

    res, per_q = {}, {s: [dict(n=len(golds[s][qi])) for qi in range(len(golds[s]))] for s in golds}
    for s, qi, kind, tgt, st, cnt in groups:
        L = lists(st, cnt)
        top = (rrf(L) if cnt > 1 else L[0])[:TOPK]
        gl = [g for g in map(str, golds[s][qi]) if g in emb_of]
        if tgt < 0:
            hit = [g in set(top) for g in gl]
            per_q[s][qi][kind] = (int(all(hit)), sum(hit) / len(hit))
        else:
            per_q[s][qi].setdefault(kind, []).append(gl[tgt] in set(top))
    for s in golds:
        out = {}
        for kind in ("gold_query", "centroid", "loo_single", "loo_multi"):
            rows = []
            for p in per_q[s]:
                v = p[kind]
                rows.append(v if isinstance(v, tuple) else (int(all(v)), sum(v) / len(v)))
            out[kind] = {"mrecall@100": round(100 * np.mean([r[0] for r in rows]), 2),
                         "recall@100": round(100 * np.mean([r[1] for r in rows]), 2)}
            by_n = {}
            for p, r in zip(per_q[s], rows):
                by_n.setdefault(min(p["n"], 8), []).append(r[0])
            out[kind]["mrecall_by_gold_count"] = {f"{n}{'+' if n == 8 else ''}": [round(100 * np.mean(v), 2), len(v)]
                                                  for n, v in sorted(by_n.items())}
        res[s] = out
        log(s, json.dumps({k: (v["mrecall@100"], v["recall@100"]) for k, v in out.items()}))

    # geometry
    geo = {}
    rp = rand @ rand.T
    rand_cos = float(rp[np.triu_indices(len(rand), 1)].mean())
    for s in golds:
        means, pairs = [], []
        for gl in golds[s]:
            G = unit(np.stack([emb_of[g] for g in map(str, gl) if g in emb_of]))
            p = (G @ G.T)[np.triu_indices(len(G), 1)]
            means.append(p.mean()); pairs.append(p)
        pairs = np.concatenate(pairs); means = np.array(means)
        geo[s] = {"random_pair_cos": round(rand_cos, 4),
                  "question_mean_gg_cos": {q: round(float(np.quantile(means, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                  "gold_pair_cos": {q: round(float(np.quantile(pairs, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                  "share_pairs_above_0.90": round(float((pairs > 0.90).mean()), 4),
                  "share_pairs_above_0.95": round(float((pairs > 0.95).mean()), 4),
                  "share_questions_mean_above_0.90": round(float((means > 0.90).mean()), 4)}
        log(s, "geometry", json.dumps(geo[s]))
    json.dump({"bounds": res, "geometry": geo, "missing_gold_ids": len(missing),
               "notes": __doc__}, open(f"{args.out}/f0_bounds.json", "w"), indent=1)
    json.dump(per_q, open(f"{args.out}/f0_per_question.json", "w"))
    log("wrote", f"{args.out}/f0_bounds.json")


if __name__ == "__main__":
    main()
