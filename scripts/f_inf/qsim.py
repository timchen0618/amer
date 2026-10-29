"""Pairwise cosine of each question's generated query embeddings (F paper-stack runs, 2026-10-27).
For every <run>.npy in qdir with k > 1 vectors per question (rows question-major, k = rows / 531): mean over questions
of the mean cosine over all k(k-1)/2 pairs, the share of questions whose mean pair cosine > 0.95, and the mean cosine
of each adjacent pair (1-2, 2-3, ...). usage: python qsim.py <qdir> <out.json>"""
import glob, json, os, sys
import numpy as np
NQ = 531
qdir, out = sys.argv[1], sys.argv[2]
res = {}
for p in sorted(glob.glob(f"{qdir}/*.npy")):
    run = os.path.basename(p)[:-4]
    if run.endswith("_lengths"):
        continue
    a = np.load(p).astype(np.float32)
    k = len(a) // NQ
    assert k * NQ == len(a), (run, a.shape)
    if k < 2:
        continue
    a = a.reshape(NQ, k, -1)
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    s = np.einsum("qid,qjd->qij", a, a)
    iu = np.triu_indices(k, 1)
    per_q = s[:, iu[0], iu[1]].mean(1)
    res[run] = {"k": k, "mean_pair_cos": float(per_q.mean()), "share_q_above_0.95": float((per_q > 0.95).mean()),
                "adjacent": [float(s[:, i, i + 1].mean()) for i in range(k - 1)]}
    print(run, res[run], flush=True)
json.dump(res, open(out, "w"), indent=1)
