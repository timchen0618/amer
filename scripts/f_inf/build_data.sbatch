#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --output=/scratch/hc3337/projects/autoregressive/f_paper_data/logs/%x_%j.out
# F paper-stack runs (reports/objective_audit_2026-10-13/F_paper_stack_runs_2026-10-27.md): data builds with
# data_creation/repro_build_data.py (R's builder + F's 'inf' LM and seeded-split datasets).
#   P1 dataset (CPU): sbatch --job-name=f_hf_p1 build_data.sbatch hf --raw /scratch/hc3337/projects/autoregressive/paper_repro_data/raw_data --lms inf --datasets qampari
#   P2 raw (CPU):     sbatch --job-name=f_prep_p2 build_data.sbatch f_prep --datasets qampari_s0 qampari_s0dev
#   P2 embed (GPU):   sbatch --gres=gpu:1 --constraint=h200 --partition=h200_courant,h200_public --job-name=f_embed_p2 build_data.sbatch embed --datasets qampari_s0 qampari_s0dev
#   P2 dataset (CPU): sbatch --job-name=f_hf_p2 build_data.sbatch hf --lms inf --datasets qampari_s0 qampari_s0dev
W=/scratch/hc3337/projects/autoregressive/.claude/worktrees/f-paper-inf
cd $W
NV=""; [ -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ] && NV="--nv"
singularity exec $NV --overlay /scratch/hc3337/envs/div.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -c "source /ext3/env.sh; python data_creation/repro_build_data.py $* --out /scratch/hc3337/projects/autoregressive/f_paper_data"
