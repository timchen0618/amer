"""Query-target geometry over seeds, split by the paper's test distance thirds (session F, 2026-11-16).

Reads the <run>_s<step>/test_geom.per_query.jsonl files that f_run.sh writes with tools/geometry_eval.py
(no new embedding or retrieval) and prints, per run, seed and step, on full test / large third / small third:
  align_cos   mean cosine between each query embedding and the gold (target) passage it is Hungarian-matched to
              (the cosine to a gold cluster is the max over its passages); single = its one embedding vs its
              best-matching target
  q_pair_cos  mean cosine between a query's k embeddings (multi only)
Seed 0 is the original run; seeds 1 and 2 are the _rseedN reruns. Missing checkpoints print "-".

  python training/inf_retriever/tools/geometry_seed_summary.py [--steps sel|10000] [--json out.json]
"""
import argparse
import json
import os

import numpy as np

R = "results/phaseC/full/qampari/clean_v3_f_qampari_"
SUB = {s: f"data/qampari/test531_{s}_distance_indices_inf.txt" for s in ("large", "small")}
# run -> (seed 0 dev-selected step, rerun base name); later seeds' dev-selected steps are read from the driver log
RUNS = {
    "cap": ("S0_cap10k_onelabel_causal", 10000),
    "no-cap": ("S0_10k_onelabel_causal", 9000),
    "no-cap io_proj": ("S0_10k_onelabel_causal_ioproj", 7000),
    "single": ("S0_single_10k_2step_causal", 10000),
}


def dev_selected(run):
    f = f"results/phaseC/clean_v3_f_qampari_{run}.driver.out"
    if not os.path.exists(f):
        return None
    for line in open(f, errors="ignore"):
        if "RUN DONE (dev-selected" in line:
            return int(line.split("dev-selected ")[1].split(",")[0])
    for line in open(f, errors="ignore"):  # training + dev done, test still running
        if "TRAINED + DEV DONE: full dev ranking" in line:
            return int(line.split("ranking ")[1].split()[0])
    return None


def load(run, step):
    f = f"{R}{run}_s{step}/test_geom.per_query.jsonl"
    if not os.path.exists(f):
        return None
    rows = [json.loads(l) for l in open(f)]
    return {r["idx"]: r for r in rows}


def stats(rows, idx):
    a, q = [], []
    for i in idx:
        r = rows.get(i)
        if not r or not r.get("step_align_cos"):
            continue
        a.append(np.mean(list(r["step_align_cos"].values())))
        if r.get("query_pair_cos") is not None:
            q.append(r["query_pair_cos"])
    return (float(np.mean(a)) if a else None, float(np.mean(q)) if q else None, len(a))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", default="sel", help="'sel' = each seed's dev-selected step, or a step number")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    thirds = {s: [int(x) for x in open(f).read().split()] for s, f in SUB.items()}
    out = {}
    fmt = lambda v: "-" if v is None else f"{v:.4f}"
    print(f"{'run':16s} {'seed':4s} {'step':>5s} | align_cos full / large / small | q_pair_cos full / large / small")
    for name, (base, s0step) in RUNS.items():
        for seed in (0, 1, 2):
            run = base if seed == 0 else f"{base}_rseed{seed}"
            step = (s0step if seed == 0 else dev_selected(run)) if a.steps == "sel" else int(a.steps)
            rows = load(run, step) if step else None
            if rows is None:
                print(f"{name:16s} {seed:4d} {str(step or '-'):>5s} | not available yet")
                continue
            res = {"all": stats(rows, sorted(rows)), **{s: stats(rows, ix) for s, ix in thirds.items()}}
            out[f"{name}|{seed}"] = {"run": run, "step": step, **res}
            al = " / ".join(fmt(res[k][0]) for k in ("all", "large", "small"))
            qp = "n/a (single)" if name == "single" else " / ".join(fmt(res[k][1]) for k in ("all", "large", "small"))
            print(f"{name:16s} {seed:4d} {step:5d} | {al} | {qp}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
