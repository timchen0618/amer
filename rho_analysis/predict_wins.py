"""Step 4 -- Does rho predict where AMER beats the single-query baseline?

This is the experiment that upgrades the paper. If rho separates wins from
non-wins, the contribution stops being "we tried a thing and got 4%" and becomes
"here is the criterion that determines when query-side multi-vector retrieval
helps", with AMER as the demonstration. That claim survives unimpressive
absolute numbers.

Critically, this compares rho against raw mean pairwise distance -- the x-axis
currently used in Figure 1. The argument for rho is that it normalizes away
local corpus density, so it should separate strictly better. If it does not,
say so; that is also a result.

Also reports the per-bin ABSOLUTE difference with bootstrap CIs. Relative gains
computed on a base of 1.5 MRecall points (as in the low-similarity subsets) are
noise amplification, not statistics.

Usage:
    python predict_wins.py --rho rho.jsonl \
        --baseline baseline_results.jsonl --system amer_results.jsonl
"""
import argparse
import json

import numpy as np

from io_utils import auc, bootstrap_ci, load_jsonl, load_results


def align(rho_rows, base, sys_):
    keep = [r for r in rho_rows if r["qid"] in base and r["qid"] in sys_]
    b = np.array([base[r["qid"]] for r in keep], dtype=np.float64)
    s = np.array([sys_[r["qid"]] for r in keep], dtype=np.float64)
    return keep, b, s


def bin_table(x, b, s, n_bins=4, labels=None):
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() == 0:
            continue
        diff = s[m] - b[m]
        lo, hi = bootstrap_ci(lambda a: float(np.mean(a)), diff)
        out.append(
            {
                "bin": labels[i] if labels else f"Q{i + 1}",
                "range": [float(edges[i]), float(edges[i + 1])],
                "n": int(m.sum()),
                "baseline": float(b[m].mean()),
                "system": float(s[m].mean()),
                "abs_diff": float(diff.mean()),
                "abs_diff_ci95": [lo, hi],
                "win_rate": float((s[m] > b[m]).mean()),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", required=True, help="rho.jsonl from rho.py")
    ap.add_argument("--baseline", required=True, help="single-query results.jsonl")
    ap.add_argument("--system", required=True, help="AMER results.jsonl")
    ap.add_argument("--field", default="mrecall")
    ap.add_argument("--rho-key", default="rho_oracle")
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rho_rows = load_jsonl(a.rho)
    base = load_results(a.baseline, a.field)
    sys_ = load_results(a.system, a.field)
    keep, b, s = align(rho_rows, base, sys_)
    if not keep:
        raise SystemExit("no overlapping qids between rho file and results files")

    rho = np.array([r[a.rho_key] for r in keep], dtype=np.float64)
    diam = np.array([r["diameter"] for r in keep], dtype=np.float64)
    win = (s > b).astype(int)

    report = {
        "n": len(keep),
        "baseline_mean": float(b.mean()),
        "system_mean": float(s.mean()),
        "abs_diff": float((s - b).mean()),
        "abs_diff_ci95": list(bootstrap_ci(lambda a_: float(np.mean(a_)), s - b)),
        "win_rate": float(win.mean()),
        "predictors_of_win": {},
    }

    for name, x in (("rho", rho), ("raw_diameter", diam)):
        lo, hi = bootstrap_ci(auc, x, win)
        report["predictors_of_win"][name] = {
            "auc": auc(x, win),
            "auc_ci95": [lo, hi],
        }

    report["bins_by_rho"] = bin_table(rho, b, s, a.bins)
    report["bins_by_raw_diameter"] = bin_table(diam, b, s, a.bins)

    # feasibility regimes from the covering argument
    regimes = {
        "rho < 1.41 (single vector sufficient)": rho < np.sqrt(2),
        "1.41 <= rho <= 2.0 (configuration-dependent)": (rho >= np.sqrt(2)) & (rho <= 2.0),
        "rho > 2.0 (single vector impossible)": rho > 2.0,
    }
    report["regimes"] = {}
    for name, m in regimes.items():
        if m.sum() == 0:
            continue
        lo, hi = bootstrap_ci(lambda a_: float(np.mean(a_)), (s - b)[m])
        report["regimes"][name] = {
            "n": int(m.sum()),
            "frac": float(m.mean()),
            "baseline": float(b[m].mean()),
            "system": float(s[m].mean()),
            "abs_diff": float((s - b)[m].mean()),
            "abs_diff_ci95": [lo, hi],
        }

    print(json.dumps(report, indent=2))
    ra = report["predictors_of_win"]["rho"]["auc"]
    da = report["predictors_of_win"]["raw_diameter"]["auc"]
    print()
    if ra > da + 0.02:
        print(f"  rho predicts wins better than raw distance ({ra:.3f} vs {da:.3f}).")
        print("  -> replace the Figure 1 x-axis with rho; it is the right normalization.")
    elif abs(ra - da) <= 0.02:
        print(f"  rho and raw distance are comparable ({ra:.3f} vs {da:.3f}).")
        print("  -> local corpus density is roughly uniform here; report it and move on.")
    else:
        print(f"  raw distance predicts better ({da:.3f} vs {ra:.3f}); investigate r_k noise.")
    if report["regimes"].get("rho > 2.0 (single vector impossible)", {}).get("frac", 0) < 0.05:
        print(
            "  WARNING: almost no examples are in the rho > 2 regime. These benchmarks "
            "cannot demonstrate what a multi-vector query encoder is for -- the gains you "
            "measure come from better centroid placement, not from covering disjoint "
            "clusters. Consider a higher-rho benchmark (e.g. NERetrieve)."
        )

    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
