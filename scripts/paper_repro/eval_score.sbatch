#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=48GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --partition=cpu_short
#SBATCH --output=/scratch/hc3337/projects/autoregressive/paper_repro_data/logs/score_%x_%j.out
# Paper-repro eval step 3 (CPU): round-robin + RRF, paper scorer. usage: sbatch --job-name=repro_score_<lm>_<ds> eval_score.sbatch <lm> <ds>
W=/scratch/hc3337/projects/autoregressive/.claude/worktrees/paper-repro; D=/scratch/hc3337/projects/autoregressive/paper_repro_data
singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif bash -c \
  "source /ext3/env.sh; PAPER_CODE=$W python $W/scripts/paper_repro/score.py $2 $D/eval/$1/$2/scores.json $D/eval/$1/$2/top500_full.pkl"
