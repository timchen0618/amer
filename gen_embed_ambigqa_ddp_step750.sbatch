#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=gen_embed_ambigqa_ddp_step750
#SBATCH --output=sbatch_outputs/gen_embed_ambigqa_ddp_step750_%a.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --array=0-31

PASSAGES="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
MODEL_NAME_OR_PATH="checkpoints/ambigqa/ambigqa_joint_repro_steps800_t0.05_lr0.00001_ws30_bs50_hungarian_ddp/checkpoint/step-750/"
OUTPUT_DIR="wikipedia_embeddings/ambigqa/ddp_step750"
SHARD_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=32
PER_GPU_BATCH_SIZE=512

mkdir -p "$OUTPUT_DIR"

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python gen_embed_new.py --passages $PASSAGES --output_dir $OUTPUT_DIR --shard_id $SHARD_ID --num_shards $NUM_SHARDS --model_name_or_path $MODEL_NAME_OR_PATH --per_gpu_batch_size $PER_GPU_BATCH_SIZE"
