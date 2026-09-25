"""Checks whether rho is scale-free in k, i.e. whether ranking examples by rho
is stable across retrieval budgets.

rho = D / r_k is not literally scale-free: r_k grows with k (a bigger ball
reaches further), so rho shrinks as k grows. Locally, if the corpus has
effective dimension d_eff around a query, count-within-radius scales as
r^d_eff, so r_k ~ k^(1/d_eff) and rho ~ k^(-1/d_eff). In high dimensions
(embedding spaces typically have local d_eff in the tens) that exponent is
small, so absolute rho shifts with k but the RANKING of examples by rho should
be close to invariant, since every example's r_k scales by roughly the same
factor. This is the thing to actually check rather than assume -- see
pipeline_query.py/pipeline_oracle.py's --ks output (the "by_k" field on each
row) for the raw numbers this reads.

Usage:
    python multik_analysis.py --dataset qampari --variant query
    python multik_analysis.py --dataset qampari --variant oracle
"""
import argparse
import json
import math

import numpy as np
from scipy.stats import spearmanr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variant", required=True, choices=["query", "oracle"])
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    out_dir = a.out_dir or f"outputs/{a.dataset}"
    rho_key = f"rho_{a.variant}"
    path = f"{out_dir}/rho_{a.variant}.jsonl"

    rows = [json.loads(l) for l in open(path)]
    ks = sorted(int(k) for k in rows[0]["by_k"].keys())

    rho_by_k = {k: np.array([r["by_k"][str(k)][rho_key] for r in rows]) for k in ks}

    print(f"=== {a.dataset} / {rho_key} (n={len(rows)}) ===")
    print(f"{'k':>6} {'median':>8} {'p95':>8} {'max':>8} {'frac>=sqrt2':>12} {'frac>2':>8}")
    for k in ks:
        rho = rho_by_k[k]
        print(
            f"{k:>6} {np.median(rho):>8.3f} {np.percentile(rho, 95):>8.3f} {rho.max():>8.3f} "
            f"{np.mean(rho >= math.sqrt(2)):>12.4f} {np.mean(rho > 2.0):>8.4f}"
        )

    print()
    print("Spearman rank correlation between rho rankings at different k (pairwise vs. smallest k):")
    k0 = ks[0]
    for k in ks[1:]:
        rho_corr, _ = spearmanr(rho_by_k[k0], rho_by_k[k])
        print(f"  k={k0} vs k={k}: rho_spearman={rho_corr:.4f}")

    print()
    print("Spearman rank correlation, consecutive k pairs:")
    for k_prev, k in zip(ks[:-1], ks[1:]):
        rho_corr, _ = spearmanr(rho_by_k[k_prev], rho_by_k[k])
        print(f"  k={k_prev} vs k={k}: rho_spearman={rho_corr:.4f}")

    all_corrs = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            c, _ = spearmanr(rho_by_k[ks[i]], rho_by_k[ks[j]])
            all_corrs.append(c)
    print()
    print(f"min pairwise Spearman across all {len(ks)} k values: {min(all_corrs):.4f}")


if __name__ == "__main__":
    main()
