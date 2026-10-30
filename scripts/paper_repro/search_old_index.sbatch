#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=160GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --constraint=h200
#SBATCH --partition=h200_courant,h200_public
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/hc3337/projects/autoregressive/paper_repro_data/logs/searchold_%x_%j.out
# Like-for-like with the paper's tables: search repro query embeddings against the paper's old 21.12M index
# (fe0bd78 gen_ret_and_eval uses embeddings_root/inf/qampari_embeddings for both QAMPARI and AmbigQA).
# usage: sbatch search_old_index.sbatch <lm>/<ds> [<lm>/<ds> ...]   -> $D/eval/<lm>/<ds>/top500_old21m.pkl
W=/scratch/hc3337/projects/autoregressive/.claude/worktrees/paper-repro
D=/scratch/hc3337/projects/autoregressive/paper_repro_data
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
for x in "$@"; do
  singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG bash -c "source /ext3/env.sh; conda activate nli; \
    PAPER_CODE=$W python $W/scripts/paper_repro/search.py '/scratch/hc3337/embeddings/inf/qampari_embeddings/*' $D/eval/$x/qemb $D/eval/$x/top500_old21m.pkl"
  echo "searched $x"
done
