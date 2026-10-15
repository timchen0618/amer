"""Summarize fusion_eval.py output (objective audit, 2026-10-15): one setting per method family per setting.

For each setting (frozen / joint) and method family, the parameter value with the highest mean full-dev MRecall@100
over that setting's checkpoints is chosen (dev only; the test subsets are never used to select). Prints markdown:
  1. per family: chosen value, mean dev / test / test-large / test-small, and the change vs plain RRF (c = 60);
  2. per checkpoint: test / large / small for plain RRF and every chosen method.

    python training/inf_retriever/tools/fusion_summary.py results/phaseC/fusion/frozen.json results/phaseC/fusion/joint.json
"""
import json, re, sys
import numpy as np

FAMILIES = [
    ("Plain RRF, c = 60 (headline)", r"rrf_c60$"),
    ("Round robin", r"rr$"),
    ("1. First query leads (top N of query 1, then round robin)", r"lead1_N\d+$"),
    ("2. Weighted RRF (query 1 weight w, constant c)", r"wrrf_w[\d.]+_c\d+$"),
    ("3a. Score fusion: best score over queries", r"max_score$"),
    ("3b. Score fusion: summed scores", r"sum_score$"),
    ("4. Drop near-duplicate queries (cos > t), then RRF", r"dedup[\d.]+_rrf$"),
    ("5. Agreement gate (mean q-q cos > t: query 1 only)", r"gate[\d.]+$"),
]
COLS = ("dev", "test", "test_large", "test_small")


def mean(runs, m, col):
    return float(np.mean([runs[r][m][col][0] for r in runs]))


def summarize(title, runs):
    methods = list(next(iter(runs.values())).keys())
    base = {c: mean(runs, "rrf_c60", c) for c in COLS}
    print(f"\n### {title} ({len(runs)} checkpoints)\n")
    print("| Method | Chosen on dev | Dev | Test | Test, large distance | Test, small distance |")
    print("| --- | --- | --- | --- | --- | --- |")
    chosen = {}
    for name, pat in FAMILIES:
        cand = [m for m in methods if re.match(pat, m)]
        best = max(cand, key=lambda m: (mean(runs, m, "dev"), -cand.index(m)))
        chosen[name] = best
        v = {c: mean(runs, best, c) for c in COLS}
        cells = [f"{v[c]:.2f}" + ("" if name.startswith("Plain") else f" ({v[c] - base[c]:+.2f})") for c in COLS]
        print(f"| {name} | `{best}` | " + " | ".join(cells) + " |")
    print(f"\nMeans over the checkpoints; brackets = change vs plain RRF.\n")
    print("| Checkpoint | " + " | ".join(f"{n.split(' (')[0]}: `{m}`" for n, m in chosen.items()) + " |")
    print("| --- |" + " --- |" * len(chosen))
    for r in runs:
        cells = ["{} / {} / {}".format(*(f"{runs[r][m][c][0]:.1f}" for c in COLS[1:])) for m in chosen.values()]
        print(f"| {r} | " + " | ".join(cells) + " |")
    print("\nCells: test / large / small MRecall@100.")
    return chosen


def main():
    for p in sys.argv[1:]:
        d = json.load(open(p))
        singles = {k: v["single"] for k, v in d.items() if "single" in v}
        runs = {k: v for k, v in d.items() if "rrf_c60" in v}
        if singles:
            print("\n### Single-query references\n\n| Run | Test | Test, large distance | Test, small distance |\n| --- | --- | --- | --- |")
            for k, v in singles.items():
                print(f"| {k} | {v['test'][0]:.2f} | {v['test_large'][0]:.2f} | {v['test_small'][0]:.2f} |")
        if runs:
            summarize(p.rsplit("/", 1)[-1].replace(".json", ""), runs)


if __name__ == "__main__":
    main()
