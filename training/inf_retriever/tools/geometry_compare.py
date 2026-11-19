"""One table of query/target cosines: F's runs, base INF-Retriever and the AMER reproduction (session F, 2026-11-16).

Everything is measured the same way: tools/geometry_eval.py on the 531-question QAMPARI test file
(data/amer_data/eval_data/qampari.jsonl) against the full 25.86M-passage index embedded with base INF-Retriever
(wikipedia_embeddings/qampari/shared_frozen_base_512), which both F (frozen doc encoder) and the AMER
reproduction retrieve from. Base and repro geometry come from results/phaseC/geometry_ref/ (geom_ref.sbatch).

Columns (mean over questions; seeds averaged, ± = sd over seeds):
  q-target   cosine between each query embedding and the gold passage it is Hungarian-matched to; single /
             base = its one embedding vs its best-matching gold. Full test / large / small distance third
             (data/qampari/test531_{large,small}_distance_indices_inf.txt, the thirds fusion_eval.py reports).
  q-q        mean pairwise cosine between a question's k = 5 embeddings (multi only; the repro report's
             "pairwise cosine")
  gold-gold  mean pairwise cosine between a question's gold passages (one passage per gold cluster), full / large /
             small third; the same for every row (all share base INF-Retriever's document embeddings)
  q-random   mean cosine between query embeddings and 20k random corpus passages (scale reference). Base
             INF-Retriever's space is anisotropic (q-random 0.58, gold-random 0.74); trained query embeddings sit
             near 0, so raw q-target is not comparable between base and trained models.
  lift       q-target - q-random (full / large / small; the run-level q-random is subtracted from each third)

  python training/inf_retriever/tools/geometry_compare.py [--md out.md]
"""
import argparse
import json
import os

import numpy as np

F = "results/phaseC/full/qampari/clean_v3_f_qampari_"
REF = "results/phaseC/geometry_ref/"
SUB = {s: f"data/qampari/test531_{s}_distance_indices_inf.txt" for s in ("large", "small")}


def f_dirs(base, s0step):
    """Seed 0 at its dev-selected step plus the _rseedN reruns at theirs (as geometry_seed_summary.py)."""
    from geometry_seed_summary import dev_selected
    out = [f"{F}{base}_s{s0step}"]
    for s in (1, 2):
        step = dev_selected(f"{base}_rseed{s}")
        if step:
            out.append(f"{F}{base}_rseed{s}_s{step}")
    return out


ROWS = [
    ("Base INF-Retriever (no training)", "base", lambda: [REF + "base_inf_retriever"]),
    ("F single (S0_single_10k_2step_causal)", "F", lambda: f_dirs("S0_single_10k_2step_causal", 10000)),
    ("F multi, cap", "F", lambda: f_dirs("S0_cap10k_onelabel_causal", 10000)),
    ("F multi, no-cap", "F", lambda: f_dirs("S0_10k_onelabel_causal", 9000)),
    ("F multi, no-cap io_proj", "F", lambda: f_dirs("S0_10k_onelabel_causal_ioproj", 7000)),
] + [
    (f"Repro {lm} {kind}", "repro", (lambda lm=lm, kind=kind: [f"{REF}repro_{lm}_{kind}_{s}" for s in (0, 1, 2)]))
    for lm in ("llama-1b", "llama-3b", "qwen3-4b", "llama-8b") for kind in ("single", "amer")
]


def per_query(d):
    f = f"{d}/test_geom.per_query.jsonl"
    if not os.path.exists(f):
        return None
    return {r["idx"]: r for r in map(json.loads, open(f))}


def run_stats(d, rows, thirds):
    out = {}
    for part, idx in (("all", sorted(rows)), *thirds.items()):
        a = [np.mean(list(rows[i]["step_align_cos"].values())) for i in idx if rows.get(i, {}).get("step_align_cos")]
        out[part] = float(np.mean(a))
    q = [r["query_pair_cos"] for r in rows.values() if r.get("query_pair_cos") is not None]
    g = [r["gold_pair_cos"] for r in rows.values() if r.get("gold_pair_cos") is not None]
    out["qq"] = float(np.mean(q)) if q else None
    out["gg"] = float(np.mean(g)) if g else None
    for part, idx in thirds.items():
        gp = [rows[i]["gold_pair_cos"] for i in idx if rows.get(i, {}).get("gold_pair_cos") is not None]
        out["gg_" + part] = float(np.mean(gp)) if gp else None
    out["qr"] = json.load(open(d + "/test_geom.json"))["query_rand_cos"]
    for part in ("all",) + tuple(thirds):
        out["lift_" + part] = out[part] - out["qr"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--md", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args()
    thirds = {s: [int(x) for x in open(f).read().split()] for s, f in SUB.items()}
    lines = ["| Model | Seeds | q-target full | q-target large | q-target small | q-random | lift full | lift large | lift small | q-q (k=5) | gold-gold full | gold-gold large | gold-gold small |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    js = {}
    for label, _grp, dirs in ROWS:
        res = [run_stats(d, r, thirds) for d, r in ((d, per_query(d)) for d in dirs()) if r]
        if not res:
            lines.append(f"| {label} | 0 | not available |" + " |" * 10)
            continue

        def cell(k, res=res):
            v = [r[k] for r in res if r[k] is not None]
            if not v:
                return "—"
            return f"{np.mean(v):.3f}" + (f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else "")

        js[label] = res
        lines.append(f"| {label} | {len(res)} | {cell('all')} | {cell('large')} | {cell('small')} | {cell('qr')} | {cell('lift_all')} | {cell('lift_large')} | {cell('lift_small')} | {cell('qq')} | {cell('gg')} | {cell('gg_large')} | {cell('gg_small')} |")
    txt = "\n".join(lines)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
    if a.json:
        json.dump(js, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
