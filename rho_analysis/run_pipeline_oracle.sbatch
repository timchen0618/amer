#!/bin/bash
#SBATCH --job-name=rho_oracle
#SBATCH --account=torch_pr_152_courant
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/hc3337/projects/autoregressive/rho_analysis/logs/rho_oracle_%A_%a.out
#SBATCH --error=/scratch/hc3337/projects/autoregressive/rho_analysis/logs/rho_oracle_%A_%a.err

set -euo pipefail
cd /scratch/hc3337/projects/autoregressive

DATASETS=(qampari ambigqa ambigqa_2docs)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Running pipeline_oracle.py for dataset=$DATASET"

singularity exec --overlay /scratch/hc3337/envs/nli.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; conda activate nli; python3 rho_analysis/pipeline_oracle.py --dataset $DATASET --k 100"

echo "Done: $DATASET"
