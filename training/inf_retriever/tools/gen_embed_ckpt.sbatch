#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --array=0-31
#SBATCH --job-name=gen_embed_ckpt
#SBATCH --output=sbatch_outputs/gen_embed_%x_%a.out

# Embed the full passage corpus with one joint-trained checkpoint (32-way array,
# one ~808K-passage shard per task, ~75 GB total). Parameterized version of the
# per-checkpoint gen_embed_*.sbatch scripts.
#   sbatch --job-name=<tag> --export=ALL,CKPT=<checkpoint dir>,EMB_DIR=<output dir> \
#       training/inf_retriever/tools/gen_embed_ckpt.sbatch
: "${CKPT:?set CKPT}" "${EMB_DIR:?set EMB_DIR}"
PASSAGES="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
mkdir -p "$EMB_DIR"

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python gen_embed_new.py --passages $PASSAGES --output_dir $EMB_DIR --shard_id ${SLURM_ARRAY_TASK_ID} --num_shards 32 --model_name_or_path $CKPT --per_gpu_batch_size 512"
