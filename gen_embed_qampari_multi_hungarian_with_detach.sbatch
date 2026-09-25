#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=gen_embed_qampari_multi_hungarian_with_detach
#SBATCH --output=sbatch_outputs/gen_embed_qampari_multi_hungarian_with_detach_%a.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --array=0-31

# Corpus embeddings for the multi_hungarian_with_detach checkpoint (joint
# doc-encoder training, apparent gradient-detach fix attempt for the
# collapse confirmed in multi_hungarian_failed). Uses the ORIGINAL 32-way
# parallel array-job pattern (gen_embed.sbatch's own default), one shard of
# ~808K passages per task -- learned from today's multi_hungarian_failed
# saga, where a hand-rolled NUM_SHARDS=2 sequential job produced two ~40GB
# files that both timed out and later OOM'd retrieval. 32 small parallel
# shards avoids both problems.

PASSAGES="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
MODEL_NAME_OR_PATH="checkpoints/qampari/enc_trained_qampari_infly_multi_finetuned_steps5000_t0.05_lr0.00001_ws200_bs50_gradchkpt_refiltered_5to8_multi_hungarian_with_detach/checkpoint/best_model/"
OUTPUT_DIR="wikipedia_embeddings/qampari/multi_hungarian_with_detach"
SHARD_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=32
PER_GPU_BATCH_SIZE=512

mkdir -p "$OUTPUT_DIR"

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python gen_embed_new.py --passages $PASSAGES --output_dir $OUTPUT_DIR --shard_id $SHARD_ID --num_shards $NUM_SHARDS --model_name_or_path $MODEL_NAME_OR_PATH --per_gpu_batch_size $PER_GPU_BATCH_SIZE"
