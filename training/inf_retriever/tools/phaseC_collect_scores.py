"""Collect every phase C score into results/phaseC/all_scores.md: cheap dev MRecall@100 at
every saved step for every run, and full-corpus dev/test MRecall@100 / Recall@100."""
import glob, os, re
R = "results/phaseC"
def mrecall(f):
    try:
        m = re.search(r"MRecall: ([\d.]+) \| Recall: ([\d.]+)", open(f).read())  # first block = top-100
        return (float(m.group(1)), float(m.group(2))) if m else None
    except OSError:
        return None
def tags(name):
    t = "pre-fix" if "_fix" not in name else ("fixed + hard neg." if "fixhn" in name else "fixed")
    return t
out = ["# Phase C scores (all steps)", "",
       "Cheap dev = reduced-corpus dev MRecall@100 (biased toward base-like retrieval and hard-negative training; not comparable across modes). "
       "Full = full-corpus MRecall@100 / Recall@100. \"pre-fix\" = trained before the loader fix (QAMPARI golds leak the label through whitespace; AmbigQA data has no leak).", ""]
for ds in ("qampari", "ambigqa"):
    runs = sorted(d for d in glob.glob(f"{R}/phaseC_{ds}_*") if os.path.isdir(d))
    steps = sorted({int(s[4:]) for d in runs for s in os.listdir(d) if s.startswith("step")})
    out += [f"## {ds.upper()}", "", "### Cheap dev MRecall@100 by step", "",
            "| Run | Mode | Loader | LR | " + " | ".join(f"step {s}" for s in steps) + " |",
            "| --- | --- | --- | --- | " + " | ".join("---" for _ in steps) + " |"]
    for d in runs:
        n = os.path.basename(d); mode = "multi" if "_multi_" in n else "single"
        lr = n.rsplit("_lr", 1)[1]
        cells = []
        for s in steps:
            v = mrecall(f"{d}/step{s}/eval_metrics.txt")
            cells.append(f"{v[0]:.2f}" if v else "")
        out.append(f"| `{n.replace('phaseC_'+ds+'_','')}` | {mode} | {tags(n)} | {lr} | " + " | ".join(cells) + " |")
    out += ["", "### Full corpus", "", "| Checkpoint | Mode | Loader | Dev MRecall / Recall | Test MRecall / Recall |", "| --- | --- | --- | --- | --- |"]
    rows = []
    for d in sorted(glob.glob(f"{R}/full/{ds}/*")):
        n = os.path.basename(d); dv = mrecall(f"{d}/dev/eval_metrics.txt"); tv = mrecall(f"{d}/test/eval_metrics.txt")
        if not (dv or tv): continue
        mode = "single" if "single" in n else "multi"
        loader = "reference" if n.startswith("ref_") else tags(n)
        f = lambda v: f"{v[0]:.2f} / {v[1]:.2f}" if v else "pending"
        rows.append((-(tv[0] if tv else -1), f"| `{n}` | {mode} | {loader} | {f(dv)} | {f(tv)} |"))
    out += [r for _, r in sorted(rows)] + [""]
open(f"{R}/all_scores.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
