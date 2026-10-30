#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --constraint=h200
#SBATCH --partition=h200_courant,h200_public
#SBATCH --gres=gpu:1
#SBATCH --job-name=repro_embed
#SBATCH --output=/scratch/hc3337/projects/autoregressive/paper_repro_data/logs/embed_%j.out
# Paper-repro data build, stage 2: strip + embed golds, seeded random negatives (see data_creation/repro_build_data.py)
cd /scratch/hc3337/projects/autoregressive/.claude/worktrees/paper-repro
singularity exec --nv --overlay /scratch/hc3337/envs/div.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -c "source /ext3/env.sh; python data_creation/repro_build_data.py embed"
