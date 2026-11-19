"""Seed noise of F's four headline QAMPARI runs (session F, 2026-11-18).

Seeds 0, 1, 2 of the single base and the cap / no-cap / no-cap io_proj multi bases (seed 0 = the original run, seeds 1
and 2 = the *_rseedN reruns with the seed mixed into the batch sampler, e6262c5). Per run and seed, at the dev-selected
step and at step 10000 (a collapsed checkpoint, dev 0, is shown and left out of the mean): full dev (RRF k=5 for multi, MRecall@100 for single), full test and the test large / small
distance thirds, read from the fusion_eval.py outputs (rrf_c60 = the headline; single = MRecall@100). Then mean and
sample sd over seeds, and the multi - single gap of the means.

    python training/inf_retriever/tools/seed_report.py [--md out.md] [--json out.json]
"""
import argparse
import json
import re

import statistics as st

P = "results/phaseC/"
RUNS = {  # label -> (run base, mode)
    "single": ("S0_single_10k_2step_causal", "single"),
    "cap": ("S0_cap10k_onelabel_causal", "multi"),
    "no-cap": ("S0_10k_onelabel_causal", "multi"),
    "no-cap io_proj": ("S0_10k_onelabel_causal_ioproj", "multi"),
}
COLS = ("dev", "test", "test_large", "test_small")


def driver_steps(run):
    """(dev-selected step, {step: full dev}) from the driver log."""
    txt = open(f"{P}clean_v3_f_qampari_{run}.driver.out", errors="ignore").read()
    rank = re.findall(r"full dev ranking ([0-9. ]+)", txt)[-1].split()
    dev = {int(rank[i]): float(rank[i + 1]) for i in range(0, len(rank), 2)}
    return max(dev, key=lambda s: (dev[s], -s)), dev


def driver_test(run, step):
    m = re.findall(rf"RESULT full clean_v3_f_qampari_{run}_s{step}: DEV [0-9.]+ \| TEST ([0-9.]+)",
                   open(f"{P}clean_v3_f_qampari_{run}.driver.out", errors="ignore").read())
    return float(m[-1]) if m else None


def scores(run, mode, step, fus):
    sel, dev = driver_steps(run)
    step = sel if step == "sel" else step
    key = f"clean_v3_f_qampari_{run}_s{step}" if mode == "multi" else f"{run}_s{step}"
    r = {"step": step, "dev": dev.get(step)}
    if key in fus:
        v = fus[key]["rrf_c60" if mode == "multi" else "single"]
        r.update({c: v[c][0] for c in COLS[1:]})
    else:  # e.g. a collapsed step with no thirds file: full test from the driver log only
        r.update({"test": driver_test(run, step), "test_large": None, "test_small": None})
    if r["dev"] == 0:  # collapsed checkpoint (single rseed1 step 10000: the last update broke the weights)
        r = {"step": f"{step} (collapsed)", **{c: None for c in COLS}}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    fus = {}
    for f in ("fusion_S0_test.json", "fusion_S0_rseed_test.json", "fusion_S0_single_test.json",
              "fusion_S0_single_rseed_test.json"):
        fus.update(json.load(open(P + f)))
    out, lines = {}, []
    fmt = lambda v: "–" if v is None else f"{v:.2f}"
    for which in ("sel", 10000):
        lines += [f"\n### {'Dev-selected step' if which == 'sel' else 'Step 10000'}\n",
                  "| Run | Seed | Step | Dev | Test | Large | Small |", "|---|---|---|---|---|---|---|"]
        means = {}
        for label, (base, mode) in RUNS.items():
            rows = [scores(base if s == 0 else f"{base}_rseed{s}", mode, which, fus) for s in (0, 1, 2)]
            out[f"{label}|{which}"] = rows
            for s, r in enumerate(rows):
                lines.append(f"| {label} | {s} | {r['step']} | " + " | ".join(fmt(r[c]) for c in COLS) + " |")
            ms = {}
            for c in COLS:
                v = [r[c] for r in rows if r[c] is not None]
                ms[c] = (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v))
            means[label] = ms
            lines.append(f"| **{label} mean ± sd** | | | " + " | ".join(
                f"**{m:.2f} ± {sd:.2f}**" + ("" if n == 3 else f" (n={n})") for m, sd, n in ms.values()) + " |")
        lines += ["", "| Gap of means (multi - single) | Test | Large | Small |", "|---|---|---|---|"]
        for label in ("cap", "no-cap", "no-cap io_proj"):
            lines.append(f"| {label} | " + " | ".join(
                f"{means[label][c][0] - means['single'][c][0]:+.2f}" for c in COLS[1:]) + " |")
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
