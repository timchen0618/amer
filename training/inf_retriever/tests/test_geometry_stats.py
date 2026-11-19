"""Checks for the query/gold geometry diagnostics (CPU, single process).

1. inbatch.geometry_stats against a brute-force reference (all k! assignments) on random tensors.
2. Its assignment equals the one HungarianContrastiveLoss makes (log-softmax over the pool, then
   linear_sum_assignment on the example's k x k block).
3. A permuted copy of the golds gives align_cos = 1; 3-D and flattened positives agree; k = 1 logs
   only alignment and gold_neg_cos.
4. tools/geometry_eval.py on a synthetic corpus (2 shards) against brute force, including the
   nearest-sibling rank.

Run from the repo root:  python training/inf_retriever/tests/test_geometry_stats.py
"""
import itertools
import json
import os
import pickle
import sys
import tempfile
import types

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from src import inbatch  # noqa: E402
import geometry_eval  # noqa: E402


def close(a, b, tol=1e-5):
    assert abs(a - b) < tol, (a, b)


def reference(q, p, n):
    """Brute-force geometry_stats: q, p (bsz, k, d), n (m, d), all numpy and unnormalized."""
    nz = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    q, p, n = nz(q), nz(p), nz(n)
    bsz, k, _ = q.shape
    out = {"align_cos": [], "align_margin": [], "gold_pair_cos": [], "gold_pair_cos_min": [], "gold_neg_cos": [],
           "query_pair_cos": [], "query_pair_cos_min": [], "gold_cohesion": []}
    steps = [[] for _ in range(k)]
    off = ~np.eye(k, dtype=bool)
    for i in range(bsz):
        c = q[i] @ p[i].T
        perm = max(itertools.permutations(range(k)), key=lambda pm: sum(c[j, pm[j]] for j in range(k)))
        m = np.array([c[j, perm[j]] for j in range(k)])
        for j in range(k): steps[j].append(m[j])
        out["align_cos"].append(m.mean())
        out["align_margin"].append(np.mean([m[j] - max(c[j, x] for x in range(k) if x != perm[j]) for j in range(k)]))
        pp = (p[i] @ p[i].T)[off]; qq = (q[i] @ q[i].T)[off]
        gn = (p[i] @ n.T).mean()
        out["gold_pair_cos"].append(pp.mean()); out["gold_pair_cos_min"].append(pp.min())
        out["query_pair_cos"].append(qq.mean()); out["query_pair_cos_min"].append(qq.min())
        out["gold_neg_cos"].append(gn); out["gold_cohesion"].append(pp.mean() - gn)
    res = {x: float(np.mean(v)) for x, v in out.items()}
    res.update({f"step_{j}_align_cos": float(np.mean(steps[j])) for j in range(k)})
    return res


def test_against_reference():
    g = torch.Generator().manual_seed(0)
    bsz, k, d = 7, 5, 16
    q, p, n = (torch.randn(*s, generator=g) for s in ((bsz, k, d), (bsz, k, d), (bsz * k, d)))
    got = {x: s / c for x, (s, c) in inbatch.geometry_stats(q, p, n).items()}
    ref = reference(q.numpy(), p.numpy(), n.numpy())
    assert set(got) == set(ref), set(got) ^ set(ref)
    for x in ref: close(got[x], ref[x])
    flat = {x: s / c for x, (s, c) in inbatch.geometry_stats(q, p.reshape(bsz * k, d), n).items()}
    for x in ref: close(flat[x], got[x])
    print("ok: geometry_stats matches brute force (3-D and flattened positives)")


def test_matches_loss_assignment():
    g = torch.Generator().manual_seed(1)
    bsz, k, d, T = 6, 4, 8, 0.05
    q, p, n = (torch.randn(*s, generator=g) for s in ((bsz, k, d), (bsz, k, d), (bsz, k, d)))
    qn, pn, nn_ = F.normalize(q, dim=-1), F.normalize(p, dim=-1), F.normalize(n, dim=-1)
    pool = torch.cat([pn.reshape(-1, d), nn_.reshape(-1, d)])
    logsm = torch.log_softmax(qn.reshape(-1, d) @ pool.T / T, dim=1)       # as HungarianContrastiveLoss
    for i in range(bsz):
        _, loss_cols = linear_sum_assignment(logsm[k * i:k * (i + 1), k * i:k * (i + 1)].numpy(), maximize=True)
        _, cos_cols = linear_sum_assignment((qn[i] @ pn[i].T).numpy(), maximize=True)
        assert (loss_cols == cos_cols).all(), (i, loss_cols, cos_cols)
    print("ok: cosine assignment equals the loss's log-softmax assignment")


def test_edge_cases():
    g = torch.Generator().manual_seed(2)
    p = torch.randn(3, 4, 8, generator=g)
    perm = torch.tensor([2, 0, 3, 1])
    st = inbatch.geometry_stats(p[:, perm] * 3.0, p, torch.randn(5, 8, generator=g))
    close(st["align_cos"][0] / st["align_cos"][1], 1.0)
    one = inbatch.geometry_stats(torch.randn(4, 1, 8, generator=g), torch.randn(4, 8, generator=g), torch.randn(4, 8, generator=g))
    assert set(one) == {"align_cos", "step_0_align_cos", "gold_neg_cos"}, set(one)
    n5 = torch.randn(5, 8, generator=g)
    plain = inbatch.geometry_stats(p[:, perm], p, n5)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):                  # as in training
        auto = inbatch.geometry_stats(p[:, perm].bfloat16(), p.bfloat16(), n5.bfloat16())
    for x in plain: close(auto[x][0] / auto[x][1], plain[x][0] / plain[x][1], 2e-2)
    it = inbatch.add_geometry_stats({}, "train", p, p, n5)
    assert all(x.startswith("train/") and isinstance(v, tuple) for x, v in it.items())
    print("ok: permuted golds give align_cos 1; k=1 logs alignment only; bf16 autocast works; add_geometry_stats keys")


def test_geometry_eval_end_to_end():
    rng = np.random.default_rng(3)
    d, N = 12, 400
    emb = rng.normal(size=(N, d)).astype(np.float32)
    ids = [f"{i}__0" for i in range(N)]
    # 4 queries; clusters given as lists of corpus ids; query 3 has no gold in the corpus
    data = [{"ground_truths": [[{"id": "1__0"}, {"id": "2__0"}], [{"id": "10__0"}], [{"id": "20__0"}]]},
            {"positive_ctxs": [{"id": "30__0"}, {"id": "31__0"}]},
            {"ground_truths": [[{"id": "40__0"}]]},
            {"ground_truths": [[{"id": "missing"}]]}]
    q = rng.normal(size=(4, 2, d)).astype(np.float32)
    with tempfile.TemporaryDirectory() as t:
        for s, (a, b) in enumerate([(0, 250), (250, N)]):
            pickle.dump((ids[a:b], emb[a:b].astype(np.float16)), open(f"{t}/passages_{s:02d}", "wb"))
        emb = np.stack([emb[i].astype(np.float16).astype(np.float32) for i in range(N)])   # stored precision
        with open(f"{t}/data.jsonl", "w") as f:
            for ex in data: f.write(json.dumps(ex) + "\n")
        np.save(f"{t}/q.npy", q)
        args = types.SimpleNamespace(qemb=f"{t}/q.npy", emb_glob=f"{t}/passages_*", data=f"{t}/data.jsonl",
                                     out=f"{t}/geom.json", n_random=N, rep_chunk=2, threads=1)
        geometry_eval.main(args)
        out = json.load(open(args.out))
        per = [json.loads(l) for l in open(f"{t}/geom.per_query.jsonl")]
    E = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    Q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    # query 0: clusters {1,2}, {10}, {20}; k=2 -> rectangular assignment
    cos = np.stack([np.max(E[[1, 2]] @ Q[0].T, 0), E[10] @ Q[0].T, E[20] @ Q[0].T], 1)        # (k=2, m=3)
    best = max(itertools.permutations(range(3), 2), key=lambda pm: cos[0, pm[0]] + cos[1, pm[1]])
    for j in range(2): close(per[0]["step_align_cos"][str(j)], cos[j, best[j]], 1e-4)
    reps = E[[1, 10, 20]]
    pp = (reps @ reps.T)[~np.eye(3, dtype=bool)]
    close(per[0]["gold_pair_cos"], pp.mean(), 1e-4); close(per[0]["gold_pair_cos_min"], pp.min(), 1e-4)
    # sibling rank of rep 1 (query 0): passages closer to it than its nearest sibling (10 or 20), minus itself
    for rep, sibs in ((1, (10, 20)), (10, (1, 20)), (20, (1, 10))):
        sims = E @ E[rep]
        want = int(sum(sims[x] > max(sims[y] for y in sibs) for x in range(N) if x not in (rep,) + sibs))
        got = per[0]["sibling_rank"][[1, 10, 20].index(rep)]
        assert got == want, (rep, got, want)
    assert "gold_pair_cos" not in per[2] and "sibling_rank" not in per[2]          # one cluster: no pair/sibling
    assert per[3]["n_clusters_found"] == 0 and "step_align_cos" not in per[3]
    assert out["queries_with_all_clusters"] == 3 and out["n_gold_reps_with_sibling"] == 5, out
    close(out["query_pair_cos"], float(np.mean([Q[i, 0] @ Q[i, 1] for i in range(4)])), 1e-4)
    with tempfile.TemporaryDirectory() as t:                                          # no gold ids -> skipped
        with open(f"{t}/d.jsonl", "w") as f: f.write(json.dumps({"positive_ctxs": [{"title": "x", "text": "y"}]}) + "\n")
        geometry_eval.main(types.SimpleNamespace(qemb="unused", emb_glob="unused", data=f"{t}/d.jsonl", out=f"{t}/g.json",
                                                 n_random=10, rep_chunk=2, threads=1))
        assert "skipped" in json.load(open(f"{t}/g.json"))
    print("ok: geometry_eval.py matches brute force (alignment, gold pairs, sibling rank, skips)")


if __name__ == "__main__":
    test_against_reference()
    test_matches_loss_assignment()
    test_edge_cases()
    test_geometry_eval_end_to_end()
    print("all geometry tests passed")
