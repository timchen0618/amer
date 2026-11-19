"""Phase A: compare cheap (reduced-corpus) scores with full-corpus scores.

Reads training/inf_retriever/tools/phaseA_validation_manifest.tsv and each checkpoint's
results/phaseA/validation/<dataset>/<name>/eval_metrics.txt, then reports per dataset:
  - Spearman and Kendall rank correlation of MRecall@100 (cheap vs full)
  - pairwise order agreement over checkpoint pairs whose full MRecall@100 differs by
    at least --min_gap points (pairs closer than that are within noise)
  - mean absolute difference (the cheap eval is expected to read higher, since the
    reduced corpus has fewer distractors; only the ordering matters)
Writes results/phaseA/validation_summary.md.

Usage (repo root): python training/inf_retriever/tools/phaseA_summarize.py
"""
import argparse
import csv
import itertools
import os
import re

from scipy.stats import kendalltau, spearmanr

MAN = "training/inf_retriever/tools/phaseA_validation_manifest.tsv"


def read_metrics(path):
    if not os.path.exists(path):
        return None
    vals = [m for m in re.findall(r"MRecall: ([0-9.]+) \| Recall: ([0-9.]+)", open(path).read())]
    if len(vals) < 2:
        return None
    (mr100, r100), (mr10, r10) = vals[0], vals[1]
    return float(mr100), float(r100), float(mr10), float(r10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_gap", type=float, default=1.0)
    ap.add_argument("--out", default="results/phaseA/validation_summary.md")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(MAN), delimiter="\t"))
    lines = ["# Phase A validation: cheap (reduced-corpus) vs full-corpus test scores", ""]
    for ds in ("qampari", "ambigqa"):
        R = []
        for r in rows:
            if r["dataset"] != ds:
                continue
            m = read_metrics(f"results/phaseA/validation/{ds}/{r['name']}/eval_metrics.txt")
            R.append((r["name"], float(r["full_mrecall100"]), float(r["full_recall100"]), m))
        done = [x for x in R if x[3] is not None]
        lines += [f"## {ds} ({len(done)}/{len(R)} evaluated)", "",
                  "| checkpoint | full MRecall@100 | cheap MRecall@100 | full Recall@100 | cheap Recall@100 |",
                  "| --- | --- | --- | --- | --- |"]
        for name, fmr, fr, m in sorted(R, key=lambda x: -x[1]):
            c = (f"{m[0]:.2f}", f"{m[1]:.2f}") if m else ("pending", "pending")
            lines.append(f"| {name} | {fmr:.2f} | {c[0]} | {fr:.2f} | {c[1]} |")
        if len(done) >= 3:
            full = [x[1] for x in done]
            cheap = [x[3][0] for x in done]
            rho = spearmanr(full, cheap).correlation
            tau = kendalltau(full, cheap).correlation
            rho_r = spearmanr([x[2] for x in done], [x[3][1] for x in done]).correlation
            pairs = [(a, b) for a, b in itertools.combinations(done, 2) if abs(a[1] - b[1]) >= args.min_gap]
            agree = sum((a[1] - b[1]) * (a[3][0] - b[3][0]) > 0 for a, b in pairs)
            mae = sum(abs(f - c) for f, c in zip(full, cheap)) / len(full)
            lines += ["", f"- Spearman (MRecall@100): **{rho:.3f}**; Kendall: {tau:.3f}; Spearman (Recall@100): {rho_r:.3f}",
                      f"- Pairwise order agreement (pairs with full gap >= {args.min_gap}): **{agree}/{len(pairs)}**"
                      + (f" ({agree/len(pairs):.0%})" if pairs else ""),
                      f"- Mean absolute difference cheap vs full: {mae:.2f} points"]
        lines.append("")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
