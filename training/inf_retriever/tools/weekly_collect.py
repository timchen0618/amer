"""Collect QAMPARI full-eval results of the clean_v3_abl / clean_v3_abl2 rounds into one JSON.

Reads results/phaseC/full/qampari/<run>_s<step>/{dev,test}/eval_metrics.txt, {dev,test}_kagg.json and
{dev,test}_geom.json, the planned full-eval steps from results/phaseC/phaseC.log, and the training state
from squeue. Selection: per run, the evaluated step with the best full-dev headline (RRF k=5 for
multi-query, MRecall@100 for single-query); test is never used. Light (small JSON files): fine on the
login node.

    python training/inf_retriever/tools/weekly_collect.py > reports/weekly_summary_data.json
"""
import glob, json, os, re, subprocess, sys

ROOT = "/scratch/hc3337/projects/autoregressive"
FULL = f"{ROOT}/results/phaseC/full/qampari"
LOG = f"{ROOT}/results/phaseC/phaseC.log"

# (run id, label, group, mode, what changes against the round-1 baseline of the same mode)
RUNS = [
    ("clean_v3_abl_qampari_single", "single", "baseline", "single", "single-query baseline (LR 1e-5)"),
    ("clean_v3_abl_qampari_multi", "multi", "baseline", "multi", "multi-query baseline: Hungarian loss, scheduled sampling ramping to 1 at the end"),
    ("clean_v3_abl_qampari_masked", "masked", "objective", "multi", "masked Hungarian: the example's other golds are not negatives"),
    ("clean_v3_abl2_qampari_masked_div05", "masked + div 0.5", "objective", "multi", "masked + diversity penalty, λ = 0.5"),
    ("clean_v3_abl2_qampari_masked_div2", "masked + div 2", "objective", "multi", "masked + diversity penalty, λ = 2"),
    ("clean_v3_abl_qampari_hps", "hps 0.5", "objective", "multi", "Hungarian + single-query loss on the first query, weight 0.5"),
    ("clean_v3_abl2_qampari_hps_w025", "hps 0.25", "objective", "multi", "Hungarian + single-query loss, weight 0.25"),
    ("clean_v3_abl2_qampari_hps_w010", "hps 0.1", "objective", "multi", "Hungarian + single-query loss, weight 0.1"),
    ("clean_v3_abl_qampari_fullsamp", "full sampling", "sampling", "multi", "sampling rate 1 from the start (always the model's own outputs)"),
    ("clean_v3_abl2_qampari_sat25", "saturate 25%", "sampling", "multi", "sampling rate reaches 1 at step 625"),
    ("clean_v3_abl2_qampari_sat50", "saturate 50%", "sampling", "multi", "sampling rate reaches 1 at step 1250"),
    ("clean_v3_abl2_qampari_single_lr3e-5", "single, LR 3e-5", "learning rate", "single", "single-query, LR 3e-5"),
    ("clean_v3_abl2_qampari_multi_lr3e-5", "multi, LR 3e-5", "learning rate", "multi", "multi-query, LR 3e-5"),
    ("clean_v3_abl2_qampari_hps_fullsamp", "hps + full sampling", "combination (wave 2)", "multi", "best hps weight + sampling rate 1 from the start"),
    ("clean_v3_abl2_qampari_hps_sat50", "hps + saturate 50%", "combination (wave 2)", "multi", "best hps weight + rate reaching 1 at step 1250"),
    ("clean_v3_abl2_qampari_C_single", "C single", "doc encoder × attention (wave 3)", "single", "frozen doc encoder, bidirectional"),
    ("clean_v3_abl2_qampari_C_multi", "C multi", "doc encoder × attention (wave 3)", "multi", "frozen doc encoder, bidirectional"),
    ("clean_v3_abl2_qampari_A_single", "A single", "doc encoder × attention (wave 3)", "single", "frozen doc encoder, causal query attention"),
    ("clean_v3_abl2_qampari_A_multi", "A multi", "doc encoder × attention (wave 3)", "multi", "frozen doc encoder, causal query attention"),
    ("clean_v3_abl2_qampari_B_single", "B single", "doc encoder × attention (wave 3)", "single", "joint doc encoder, causal query attention"),
    ("clean_v3_abl2_qampari_B_multi", "B multi", "doc encoder × attention (wave 3)", "multi", "joint doc encoder, causal query attention"),
]


def mrecall(path):
    try:
        m = re.search(r"MRecall: ([0-9.]+)", open(path).read())
        return float(m.group(1)) if m else None
    except OSError:
        return None


def kagg(path):
    try:
        return {(r["agg"], r["k"]): r["mrecall@100"] for r in json.load(open(path))["results"]}
    except (OSError, ValueError, KeyError):
        return None


def geom(path):
    try:
        g = json.load(open(path))
    except (OSError, ValueError):
        return None
    keys = ("query_pair_cos", "gold_pair_cos", "gold_rand_cos", "gold_cohesion", "align_cos", "align_margin",
            "sibling_rank_median")
    out = {k: g.get(k) for k in keys}
    if out["query_pair_cos"] is not None and out["gold_pair_cos"] is not None:
        out["gap"] = out["query_pair_cos"] - out["gold_pair_cos"]
    return out


def step_result(run, step, mode):
    d = f"{FULL}/{run}_s{step}"
    r = {"step": step}
    for split in ("dev", "test"):
        merged = mrecall(f"{d}/{split}/eval_metrics.txt")
        if mode == "multi":
            ka = kagg(f"{d}/{split}_kagg.json")
            if ka is None:
                return None  # scoring not finished
            r[split] = ka[("rrf", 5)]
            r[split + "_k1"] = ka[("rrf", 1)]
            r[split + "_curve"] = [ka[("rrf", k)] for k in range(1, 6)]
            r[split + "_rr5"] = ka.get(("round_robin", 5))
        else:
            if merged is None:
                return None
            r[split] = merged
        r[split + "_geom"] = geom(f"{d}/{split}_geom.json")
        try:  # each embedding alone (tools/perstep_eval.py)
            ps = json.load(open(f"{d}/{split}_perstep.json"))["results"]
            r[split + "_perstep"] = [x["mrecall@100"] for x in ps]
            r[split + "_perstep_recall"] = [x["recall@100"] for x in ps]
        except (OSError, ValueError, KeyError):
            pass
    return r


def planned_steps():
    out = {}
    for line in open(LOG):
        m = re.search(r"queued full evals for (\S+): steps ([0-9 ]+)", line)
        if m:
            out[m.group(1)] = [int(s) for s in m.group(2).split()]
    return out


def training_state():
    try:
        q = subprocess.run(["squeue", "-u", os.environ.get("USER", "hc3337"), "-h", "-o", "%j %T"],
                           capture_output=True, text=True, timeout=60).stdout
    except Exception:
        q = ""
    st = {}
    for line in q.splitlines():
        name, state = line.split()
        if name.startswith("phaseC_clean_v3_abl"):
            st[name[len("phaseC_"):]] = state
    done = set(re.findall(r"\[(\S+)\] RUN DONE", open(LOG).read()))
    return st, done


def main():
    plan = planned_steps()
    queue, done = training_state()
    out = []
    for run, label, group, mode, desc in RUNS:
        steps = sorted({int(p.rsplit("_s", 1)[1]) for p in glob.glob(f"{FULL}/{run}_s*")})
        evals = [r for r in (step_result(run, s, mode) for s in steps) if r]
        want = plan.get(run, steps)  # round 1: every evaluated step
        missing = [s for s in want if s not in {e["step"] for e in evals}]
        if run in queue:
            status = "training" if queue[run] == "RUNNING" else "waiting for GPUs"
        elif not evals and run not in done and not os.path.isdir(f"{ROOT}/checkpoints/qampari/{run}"):
            status = "not launched"
        elif run not in done and run not in plan and not evals:
            status = "cheap evals"   # trained; checkpoints being scored on the reduced corpus
        elif missing or not want:
            status = "full evals"
        else:
            status = "done"
        sel = max(evals, key=lambda e: e["dev"]) if evals else None
        tie = None
        if sel:
            close = [e["step"] for e in evals if e is not sel and sel["dev"] - e["dev"] < 2.0]
            tie = close or None
        final = next((e for e in evals if e["step"] == 2500), None)
        out.append(dict(run=run, label=label, group=group, mode=mode, desc=desc, status=status,
                        planned=want, missing=missing, evals=evals,
                        selected=sel["step"] if sel else None, ties=tie,
                        final=final["step"] if final else None))
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
