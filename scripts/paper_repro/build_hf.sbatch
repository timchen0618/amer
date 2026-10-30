#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --partition=cpu_short
#SBATCH --job-name=repro_hf
#SBATCH --output=/scratch/hc3337/projects/autoregressive/paper_repro_data/logs/hf_%x_%j.out
# Paper-repro data build, stage 3: per-LM tokenized HF datasets (paper's create_input_embeddings_for_contrastive). usage: sbatch build_hf.sbatch <lm> [<lm> ...]
cd /scratch/hc3337/projects/autoregressive/.claude/worktrees/paper-repro
singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -c "source /ext3/env.sh; python data_creation/repro_build_data.py hf --lms $*"
