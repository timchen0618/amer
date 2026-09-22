"""Generate small synthetic data matching the expected file contract.

Use this to smoke-test the pipeline before pointing it at a 25M-passage index.
It builds a deliberately anisotropic corpus (most variance in a few directions)
so space_shape.py and whitening.py have something to detect.

    python make_demo_data.py --out-dir demo
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="demo")
    ap.add_argument("--n-corpus", type=int, default=20000)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m", type=int, default=3)
    ap.add_argument("--anisotropy", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    # anisotropic corpus: power-law spectrum + shared mean direction (cone)
    scale = np.linspace(1.0, 1.0 / a.anisotropy, a.dim)
    X = rng.normal(size=(a.n_corpus, a.dim)) * scale
    X += 1.5 * np.eye(a.dim)[0]
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X.astype(np.float32)
    np.save(os.path.join(a.out_dir, "corpus_emb.npy"), X)

    # golds: m documents sampled near a seed doc, with varying spread
    qids, golds, qvecs, base_res, sys_res = [], [], [], [], []
    for i in range(a.n_queries):
        qid = f"q{i}"
        seed_doc = rng.integers(a.n_corpus)
        spread = rng.uniform(0.02, 0.6)  # drives rho
        sims = X @ X[seed_doc]
        pool = np.argsort(-sims)[: max(a.m, int(spread * 2000))]
        gid = rng.choice(pool, a.m, replace=False)
        V = X[gid]
        c = V.mean(0)
        c = c / np.linalg.norm(c)
        qvecs.append(c + 0.05 * rng.normal(size=a.dim))

        # fake outcomes correlated with spread: baseline degrades, AMER less so
        p_base = float(np.clip(0.9 - 1.1 * spread, 0.02, 0.98))
        p_amer = float(np.clip(0.9 - 0.7 * spread, 0.02, 0.98))
        qids.append(qid)
        golds.append({"qid": qid, "gold_ids": [int(g) for g in gid]})
        base_res.append({"qid": qid, "mrecall": int(rng.random() < p_base)})
        sys_res.append({"qid": qid, "mrecall": int(rng.random() < p_amer)})

    Q = np.asarray(qvecs, dtype=np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    np.save(os.path.join(a.out_dir, "q_single.npy"), Q)
    with open(os.path.join(a.out_dir, "qids.json"), "w") as f:
        json.dump(qids, f)
    for name, rows in (
        ("golds.jsonl", golds),
        ("baseline_results.jsonl", base_res),
        ("amer_results.jsonl", sys_res),
    ):
        with open(os.path.join(a.out_dir, name), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"demo data in {a.out_dir}/")


if __name__ == "__main__":
    main()
