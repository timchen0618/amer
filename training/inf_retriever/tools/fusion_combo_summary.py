"""Summarize fusion_combo.py output and compute combination 4 (per-family rule) from fusion_eval.py output.

Selection is on full dev only, as in fusion_summary.py: one parameter per method family, highest mean dev over the
setting's checkpoints. "Best by dev" = the checkpoint with the highest dev under that method, and its test scores.

    python training/inf_retriever/tools/fusion_combo_summary.py \
        --combo results/phaseC/fusion/combo_frozen.json results/phaseC/fusion/combo_joint.json \
        --fe frozen=results/phaseC/fusion/frozen.json,results/phaseC/fusion/extra_besttest.json \
             joint=results/phaseC/fusion/joint.json
"""
import argparse, json, re
import numpy as np

COMBO_FAMILIES = [
    ("Single model alone (reference)", r"single$"),
    ("Plain RRF over the 5 queries (reference)", r"rrf_c60$"),
    ("1. Weighted summed scores (query 1 weight w)", r"wsum_w[\d.]+$"),
    ("2. Hybrid: RRF over single list (weight w) + queries 2-5", r"hyb_rrf_ws[\d.]+$"),
    ("2b. Hybrid: RRF over single list (weight w) + all 5 queries", r"hyb5_rrf_ws[\d.]+$"),
    ("2c. Hybrid: summed scores, single (weight w) + queries 2-5", r"hyb_sum_ws[\d.]+$"),
    ("3. Gated hybrid: single alone when queries agree (cos > t)", r"gate[\d.]+_hyb_rrf_ws[\d.]+$"),
]
COLS = ("dev", "test", "test_large", "test_small")


def mean(runs, m, c):
    return float(np.mean([runs[r][m][c][0] for r in runs]))


def combo(path):
    runs = json.load(open(path))
    methods = list(next(iter(runs.values())).keys())
    ref = {c: mean(runs, "single", c) for c in COLS}
    print(f"\n### {path.rsplit('/', 1)[-1]} ({len(runs)} multi-query checkpoints)\n")
    print("| Method | Chosen on dev | Dev | Test | Large | Small | Best checkpoint by dev: dev / test / large / small |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for name, pat in COMBO_FAMILIES:
        cand = [m for m in methods if re.match(pat, m)]
        best = max(cand, key=lambda m: (mean(runs, m, "dev"), -cand.index(m)))
        v = {c: mean(runs, best, c) for c in COLS}
        cells = [f"{v[c]:.2f}" + ("" if best == "single" else f" ({v[c] - ref[c]:+.2f})") for c in COLS]
        top = max(runs, key=lambda r: runs[r][best]["dev"][0])
        b = " / ".join(f"{runs[top][best][c][0]:.2f}" for c in COLS)
        print(f"| {name} | `{best}` | " + " | ".join(cells) + f" | {b} ({top}) |")
    print("\nMeans over checkpoints; brackets = change vs the single model alone.")


def family(run):
    return "hps" if "hps" in run else "other"


def per_family(setting, paths):
    runs = {}
    for p in paths.split(","):
        runs.update({k: v for k, v in json.load(open(p)).items() if "rrf_c60" in v})
    methods = [m for m in next(iter(runs.values())) if not m.startswith("_")]
    glob = max(methods, key=lambda m: mean(runs, m, "dev"))
    chosen = {}
    for fam in sorted({family(r) for r in runs}):
        sub = {r: runs[r] for r in runs if family(r) == fam}
        chosen[fam] = max(methods, key=lambda m: mean(sub, m, "dev"))
    pf = {c: float(np.mean([runs[r][chosen[family(r)]][c][0] for r in runs])) for c in COLS}
    g = {c: mean(runs, glob, c) for c in COLS}
    base = {c: mean(runs, "rrf_c60", c) for c in COLS}
    print(f"\n### 4. Per-family rule, {setting} ({len(runs)} checkpoints; families: "
          + ", ".join(f"{f} ({sum(family(r) == f for r in runs)}) -> `{m}`" for f, m in chosen.items()) + ")\n")
    print("| Rule | Dev | Test | Large | Small |\n| --- | --- | --- | --- | --- |")
    for n, v in (("Plain RRF", base), (f"One rule for all: `{glob}`", g), ("Rule chosen per family", pf)):
        print(f"| {n} | " + " | ".join(f"{v[c]:.2f}" for c in COLS) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", nargs="*", default=[])
    ap.add_argument("--fe", nargs="*", default=[], help="setting=json[,json]")
    a = ap.parse_args()
    for p in a.combo:
        combo(p)
    for s in a.fe:
        k, v = s.split("=", 1)
        per_family(k, v)


if __name__ == "__main__":
    main()
