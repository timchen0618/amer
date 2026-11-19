#!/usr/bin/env python3
"""
Hyperparameter sweep launcher.

Usage:
    python launch.py configs/sweeps/ambignq.yaml
    python launch.py configs/sweeps/ambignq.yaml --dry-run
"""

import argparse
import itertools
import os
import subprocess
import time
import yaml


def build_arg_string(cfg: dict) -> str:
    """Convert a flat dict of training args to a CLI argument string.
    Booleans become --flag (true) or are omitted (false). None values are omitted.
    """
    parts = []
    for key, val in cfg.items():
        if val is None:
            continue
        if isinstance(val, bool):
            if val:
                parts.append(f"--{key}")
        else:
            parts.append(f"--{key} {val}")
    return " \\\n      ".join(parts)


def make_exp_name(prefix: str, combo: dict) -> str:
    """Generate an experiment name from the sweep combination."""
    suffix = "_".join(f"{k}{v}" for k, v in combo.items())
    return f"{prefix}_{suffix}"


def generate_sbatch(exp_name: str, args_str: str, cfg: dict) -> str:
    slurm = cfg["slurm"]
    job = cfg["job"]

    python_cmd = (
        "accelerate launch train_distributed.py"
        if slurm.get("multiple_gpus", True)
        else "python train.py"
    )
    preemption_line = (
        '#SBATCH --comment="preemption=yes;requeue=yes"'
        if slurm.get("preemption", False)
        else ""
    )
    output_file = os.path.join(job["output_dir"], f"run_{exp_name}.out")

    return f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={slurm["cpus_per_task"]}
#SBATCH --time={slurm["time"]}
#SBATCH --mem={slurm["mem"]}
#SBATCH --job-name={exp_name}
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output={output_file}
#SBATCH --gres=gpu:{slurm["gpus"]}
#SBATCH --constraint={slurm["constraint"]}
#SBATCH --account={slurm["account"]}
{preemption_line}

SINGULARITY_IMAGE={job["singularity_image"]}
OVERLAY_FILE={job["overlay_file"]}

ARGS="{args_str}"

singularity exec --nv --overlay ${{OVERLAY_FILE}}:ro $SINGULARITY_IMAGE /bin/bash -c \\
  "source /ext3/env.sh; cd {job["work_dir"]}; \\
   (trap 'kill 0' SIGINT; HF_TOKEN={job["hf_token"]} TORCH_DISTRIBUTED_DEBUG=INFO \\
   {python_cmd} $ARGS & wait)"
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to sweep YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Generate SBATCH files but don't submit")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_defaults = cfg["train"]
    sweep_space = cfg["sweep"]
    exp_prefix = cfg["experiment"]["prefix"]
    job = cfg["job"]
    dry_run = args.dry_run or job.get("dry_run", False)

    os.makedirs(job["sbatch_dir"], exist_ok=True)
    os.makedirs(job["output_dir"], exist_ok=True)

    # Cartesian product of sweep values
    keys = list(sweep_space.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*sweep_space.values())]
    print(f"Config: {args.config}")
    print(f"Total combinations: {len(combos)}")
    if dry_run:
        print("DRY RUN — SBATCH files will be created but not submitted\n")

    submitted = []
    for combo in combos:
        exp_name = make_exp_name(exp_prefix, combo)

        # Merge fixed args + sweep combo + experiment name
        run_cfg = {**train_defaults, **combo, "name": exp_name}
        args_str = build_arg_string(run_cfg)

        sbatch_file = os.path.join(job["sbatch_dir"], f"run_{exp_name}.SBATCH")
        with open(sbatch_file, "w") as f:
            f.write(generate_sbatch(exp_name, args_str, cfg))

        if dry_run:
            print(f"  [dry-run] {sbatch_file}")
        else:
            result = subprocess.run(["sbatch", sbatch_file], capture_output=True, text=True)
            job_id = result.stdout.strip().split()[-1] if result.returncode == 0 else "FAILED"
            submitted.append(job_id)
            print(f"  Submitted {exp_name} → job {job_id}")
            delay = job.get("submission_delay", 0)
            if delay:
                time.sleep(delay)

    print(f"\n{'DRY RUN complete' if dry_run else f'{len(submitted)} jobs submitted'}.")
    if submitted:
        print(f"Job IDs: {' '.join(submitted)}")
        print("Monitor with: squeue -u $USER")


if __name__ == "__main__":
    main()
