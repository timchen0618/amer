#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=gen_embed_qampari_infly_multi_hungarian
#SBATCH --output=sbatch_outputs/gen_embed_qampari_infly_multi_hungarian.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --array=7-31

PASSAGES="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"

TRAINING_MODE="multi_hungarian"

OUTPUT_DIR="wikipedia_embeddings/${TRAINING_MODE}"

if [ "$TRAINING_MODE" == "standard" ]; then
    MODEL_NAME_OR_PATH="checkpoints/enc_trained_qampari_infly_standard_org_q_finetuned_steps5000_t0.05_lr0.00001_ws200_bs256_gradchkpt_refiltered_5to8/checkpoint/best_model/"
elif [ "$TRAINING_MODE" == "multi_hungarian_masked" ]; then
    MODEL_NAME_OR_PATH="checkpoints/enc_trained_qampari_infly_multi_finetuned_steps5000_t0.05_lr0.00001_ws200_bs50_gradchkpt_refiltered_5to8_multi_hungarian_masked/checkpoint/best_model/"
elif [ "$TRAINING_MODE" == "multi_hungarian" ]; then
    MODEL_NAME_OR_PATH="checkpoints/enc_trained_qampari_infly_multi_finetuned_steps5000_t0.05_lr0.00001_ws200_bs50_gradchkpt_refiltered_5to8_multi_hungarian/checkpoint/best_model/"
else
    echo "Invalid training mode"
    exit 1
fi

SHARD_ID=${SLURM_ARRAY_TASK_ID}
NUM_SHARDS=32
PER_GPU_BATCH_SIZE=512

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python gen_embed_new.py --passages $PASSAGES --output_dir $OUTPUT_DIR --shard_id $SHARD_ID --num_shards $NUM_SHARDS --model_name_or_path $MODEL_NAME_OR_PATH --per_gpu_batch_size $PER_GPU_BATCH_SIZE"


