"""Parameter drift of training checkpoints relative to the same run's step-0 checkpoint.

For each checkpoint it reports:
  - frac_changed: fraction of scalar weights whose value differs from step 0
  - rel_l2:       ||theta - theta_0|| / ||theta_0|| over all weights
  - by_type:      frac_changed per parameter type (q_proj.weight, mlp, layernorm, embeddings, ...)

This is the measure that separated the pure-bf16 fallback (~7-10% of weights ever
changed, because bf16 rounds most AdamW updates away) from mixed-precision runs
(~100%). Comparisons are done in fp32; checkpoints are memory-mapped.

Usage (from the repo root, on a compute node):
    python training/inf_retriever/tools/param_drift.py \
        --run_dir checkpoints/qampari/<run_name> --steps 250 500 1000 2500 \
        [--out results/param_drift/<run_name>.jsonl]
"""
import argparse
import json
import os
from collections import defaultdict

import torch


def param_type(name):
    if "layers." in name:
        parts = name.split(".")
        return ".".join(parts[-2:])  # e.g. self_attn.q_proj.weight -> q_proj.weight
    return name


def load_model(path):
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)["model"]


def drift(sd, base):
    total = changed = 0
    num = den = 0.0
    by_type = defaultdict(lambda: [0, 0])
    for name, b in base.items():
        a = sd[name].float()
        b = b.float()
        d = a - b
        c = int((d != 0).sum())
        total += b.numel()
        changed += c
        num += float((d * d).sum())
        den += float((b * b).sum())
        t = by_type[param_type(name)]
        t[0] += c
        t[1] += b.numel()
    return {
        "frac_changed": changed / total,
        "rel_l2": (num / den) ** 0.5 if den else float("nan"),
        "by_type": {k: v[0] / v[1] for k, v in sorted(by_type.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="checkpoints/<dataset>/<run_name>")
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--base_step", type=int, default=0)
    ap.add_argument("--out", default=None, help="append one JSON line per checkpoint")
    args = ap.parse_args()

    ckpt = lambda s: os.path.join(args.run_dir, "checkpoint", f"step-{s}", "checkpoint.pth")
    base = load_model(ckpt(args.base_step))
    for s in args.steps:
        rec = {"run": os.path.basename(os.path.normpath(args.run_dir)), "step": s, "base_step": args.base_step}
        rec.update(drift(load_model(ckpt(s)), base))
        print(f"{rec['run']} step {s}: weights changed {rec['frac_changed']:.3%}, "
              f"rel. distance {rec['rel_l2']:.3e}", flush=True)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
