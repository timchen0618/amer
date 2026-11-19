"""Per-target query similarity for F's runs, base INF-Retriever and the AMER reproduction (session F, 2026-11-16).

geometry_eval.py's align_cos scores the k query embeddings against their matched targets only, so a single-query
model is scored on its best target. This scores every target of a question instead and averages over targets:
  closest    each target's cosine to its closest query embedding (max over the k embeddings)
  hungarian  targets matched one-to-one to the k embeddings (Hungarian, max total cosine) use their matched
             embedding's cosine; the unmatched targets (questions with more targets than k) use the closest one
For single / base (k = 1) both equal the plain average of the query's cosine over its targets.
A target (gold cluster) counts as its closest passage, as in geometry_eval.py. Per question: mean over its
targets; then mean over questions on full test and the large / small distance thirds. lift = value - q-random
(mean cosine of the query embeddings to 20k random passages; base INF-Retriever's space is anisotropic).

Runs and their seeds are the rows of geometry_compare.py (dirs holding test_qemb.npy). One pass over the corpus
collects the target vectors, then every run is scored.

  sbatch --account=torch_pr_152_echoi --time=01:30:00 --mem=96G --cpus-per-task=16 \
      --wrap "singularity exec ... python training/inf_retriever/tools/target_coverage.py"
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry_compare import ROWS, SUB  # noqa: E402
from geometry_eval import gold_clusters, load_shard  # noqa: E402

DATA = "data/amer_data/eval_data/qampari.jsonl"
EMB = "wikipedia_embeddings/qampari/shared_frozen_base_512/passages_*"
OUT = "results/phaseC/geometry_ref/target_coverage"


def score(q, cl, gv):
    """q (k, d) normalized; cl = clusters of passage ids found in the corpus. Returns (closest, hungarian)."""
    qt = torch.from_numpy(q)
    cos = np.stack([(torch.stack([gv[g] for g in c]) @ qt.T).max(0).values.numpy() for c in cl], 1)  # (k, m)
    closest = cos.max(0)                                   # (m,)
    hung = closest.copy()
    rows, cols = linear_sum_assignment(cos, maximize=True)
    hung[cols] = cos[rows, cols]
    return float(closest.mean()), float(hung.mean())


def main():
    torch.set_num_threads(16)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
    data = [json.loads(l) for l in open(DATA)]
    clusters = [gold_clusters(ex) or [] for ex in data]
    need = {g for cl in clusters for c in cl for g in c}
    rng = np.random.default_rng(0)
    gv, rand = {}, []
    files = sorted(glob.glob(EMB))
    for f in files:
        ids, emb = load_shard(f)
        for i, x in enumerate(ids):
            if x in need:
                gv[x] = emb[i].clone()
        rand.append(emb[rng.choice(len(ids), 20000 // len(files) + 1, replace=False)])
        log(f"{f.split('/')[-1]}: {len(gv)} / {len(need)} target passages found")
    rand = torch.cat(rand)
    found = [[[g for g in c if g in gv] for c in cl] for cl in clusters]
    found = [[c for c in cl if c] for cl in found]
    thirds = {"all": list(range(len(data))),
              **{s: [int(x) for x in open(f).read().split()] for s, f in SUB.items()}}
    os.makedirs(OUT, exist_ok=True)
    table = ["| Model | Seeds | q-random | closest full | closest large | closest small | Hungarian full | Hungarian large "
             "| Hungarian small | lift closest full / large | lift Hungarian full / large |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    allres = {}
    for label, _grp, dirs in ROWS:
        res = []
        for d in dirs():
            f = f"{d}/test_qemb.npy"
            if not os.path.exists(f):
                continue
            q = np.load(f).astype(np.float32)
            if q.ndim == 2:
                q = q[:, None]
            q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
            per = [score(q[i], found[i], gv) if found[i] else None for i in range(len(data))]
            qr = float((torch.from_numpy(q.reshape(-1, q.shape[-1])) @ rand.T).mean())
            r = {"dir": d, "k": q.shape[1], "q_random": qr}
            for part, idx in thirds.items():
                v = [per[i] for i in idx if per[i]]
                r[f"closest_{part}"] = float(np.mean([x[0] for x in v]))
                r[f"hungarian_{part}"] = float(np.mean([x[1] for x in v]))
            res.append(r)
            log(f"{label}: {d} k={q.shape[1]} closest {r['closest_all']:.4f} hungarian {r['hungarian_all']:.4f}")
        allres[label] = res
        if not res:
            table.append(f"| {label} | 0 | not available |" + " |" * 8)
            continue

        def cell(key, sub=None):
            v = [r[key] - (r["q_random"] if sub else 0) for r in res]
            return f"{np.mean(v):.3f}" + (f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else "")

        table.append(f"| {label} | {len(res)} | {cell('q_random')} | {cell('closest_all')} | {cell('closest_large')} | "
                     f"{cell('closest_small')} | {cell('hungarian_all')} | {cell('hungarian_large')} | "
                     f"{cell('hungarian_small')} | {cell('closest_all', 1)} / {cell('closest_large', 1)} | "
                     f"{cell('hungarian_all', 1)} / {cell('hungarian_large', 1)} |")
    txt = "\n".join(table)
    print(txt)
    open(f"{OUT}/table.md", "w").write(txt + "\n")
    json.dump(allres, open(f"{OUT}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
