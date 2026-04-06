#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=qampari_infly_embedding_model_doc_enc_no_proj
#SBATCH --output=sbatch_outputs/qampari_infly_embedding_model_doc_enc_no_proj.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; bash training/inf_retriever/finetune.sh"
