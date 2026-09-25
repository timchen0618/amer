#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=retrieve_qampari_lora
#SBATCH --output=sbatch_outputs/retrieve_qampari_lora.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1

# Retrieval for the QAMPARI LoRA checkpoints (experiment_plan.md priority #2:
# does multi-query show a genuine positive effect on more-diverse-target
# data?). Same pattern as retrieve_phase1_lora.sbatch: uses the merged
# (LoRA-folded-into-base-weights) checkpoints, reuses the existing shared
# corpus embeddings (frozen doc encoder = identical to base model), and runs
# under nli (which has the working faiss-gpu build, no peft needed since
# these are merged/plain checkpoints).

DATA_NAME="qampari"
DATA_PATH="data/amer_data/eval_data/${DATA_NAME}.jsonl"
PASSAGES="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
PASSAGES_EMBEDDINGS="/scratch/hc3337/embeddings/inf/qampari_embeddings/*"
PROJECTION_SIZE=1536
N_DOCS=500
NUM_SHARDS=16
PER_GPU_BATCH_SIZE=4

declare -A CKPT_PATHS=(
  ["frozenDocEnc_causal_lora"]="checkpoints/qampari/frozenDocEnc_causal_lora_qampari_infly_multi_finetuned_steps800_t0.05_lr0.00001_ws30_bs150_hungarian_fullsr_merged/checkpoint/best_model/"
  ["frozenDocEnc_singlequery_lora"]="checkpoints/qampari/frozenDocEnc_lora_qampari_infly_standard_org_q_finetuned_steps800_t0.05_lr0.00001_ws30_bs150_auto_merged/checkpoint/best_model/"
)

for TRAINING_MODE in "frozenDocEnc_causal_lora" "frozenDocEnc_singlequery_lora"; do
    MODEL_NAME_OR_PATH="${CKPT_PATHS[$TRAINING_MODE]}"
    OUTPUT_DIR="results/finetuned/${DATA_NAME}/${TRAINING_MODE}/"
    OUTPUT_FILE="${DATA_NAME}.jsonl"

    echo "=== Retrieving with checkpoint: $TRAINING_MODE ($MODEL_NAME_OR_PATH) ==="

    # max_new_tokens (k) is an empirical, per-dataset choice for multi-query
    # checkpoints only (QAMPARI=5) -- retrieval_inf.py now errors if it's
    # required but unset, rather than silently defaulting.
    MAX_NEW_TOKENS_ARG=""
    if [ "$TRAINING_MODE" == "frozenDocEnc_causal_lora" ]; then
        MAX_NEW_TOKENS_ARG="--max_new_tokens 5"
    fi

    singularity exec --nv \
                --overlay /scratch/hc3337/envs/nli.ext3:ro \
                /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
                /bin/bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; python retrieval_inf.py \
                    --model_name_or_path $MODEL_NAME_OR_PATH \
                    --passages $PASSAGES \
                    --passages_embeddings '$PASSAGES_EMBEDDINGS' \
                    --data $DATA_PATH \
                    --output_dir $OUTPUT_DIR \
                    --projection_size $PROJECTION_SIZE \
                    --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
                    --n_docs $N_DOCS \
                    --num_shards $NUM_SHARDS \
                    --use_gpu \
                    --output_file $OUTPUT_FILE \
                    $MAX_NEW_TOKENS_ARG"
done
