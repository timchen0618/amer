#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=8:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=run_reranking_qampari_stage1
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=run_reranking_qampari_stage1.out
#SBATCH --gres=gpu:a100:1

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/nli.ext3

singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive/baselines/reranking/ ; (trap 'kill 0' SIGINT; HF_TOKEN=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 python gen_doc_similarity.py --base_retriever qampari_stage1 & wait)" 
