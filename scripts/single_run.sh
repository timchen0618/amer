#!/bin/bash
# Set default hyperparameters (using first values from arrays)
LEARNING_RATE=5e-5
TEMPERATURE=0.05
BATCH_SIZE=32
NUM_EPOCHS=120
WARMUP_RATIO=0.05
SAMPLE_RATE_MULTIPLIER=10
USE_HARD_NEGATIVES=false
LR_MIN_RATIO=0.0

# =================================================================
# BASE CONFIGURATION
# =================================================================

# Project name (will be used in wandb)
BASE_PROJECT="diverse_retrieval"
full_finetuning=false            # whether to use full finetuning
all_data=false                  # whether to train on all data
multiple_gpus=true              # whether to use multiple GPUs
save_only_improve=true          # whether to save only improve
save_best_model=true            # whether to save best model
normalize=true
pred_length_labels=false

doc_encoder="inf"
base_model_name="llama-8b"  # llama-1b, qwen3-4b, llama-3b, llama-8b

LOG_WITH="wandb"
machine="torch" # greene, torch

mix_one_label_shuffled=true
resume_from_checkpoint=false
use_stateful_dataloader=false
use_l40s=false

MODEL_TYPE="EmbeddingModelSSVariableLeftPad"
MODE="hungarian_contrastive"
FORCE_SAMPLING="--force_sampling"

# MODES -> 
# 1. hungarian_contrastive
# 2. contrastive_first_label
# 3. contrastive_one_label_shuffled
# 4. contrastive_all_labels_ordered
# 5. contrastive_all_labels_shuffled
# 6. mse_first_label

# # Loss function (options: MSE, Hungarian_MSE, Contrastive, Hungarian_Contrastive)
# LOSS_FUNCTION="Hungarian_Contrastive"


if [ "$MODE" == "hungarian_contrastive" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_ordered" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_shuffled" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_one_label_shuffled" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST="--take_first"
    QUESTION_ONLY=""
elif [ "$MODE" == "hungarian_contrastive_woseq" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive_woseq"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_ordered_woseq" ]; then
    LOSS_FUNCTION="Contrastive_woseq"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_all_labels_shuffled_woseq" ]; then
    LOSS_FUNCTION="Contrastive_woseq"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "contrastive_first_label" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST="--take_first"
    QUESTION_ONLY=""
elif [ "$MODE" == "mse_all_labels" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""

elif [ "$MODE" == "mse_all_labels_shuffled" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "hungarian_contrastive_no_shuffle" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY=""
elif [ "$MODE" == "mse_one_label_shuffled" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY="--first_label_only"
elif [ "$MODE" == "mse_first_label" ]; then
    LOSS_FUNCTION="MSE"
    SHUFFLE_SEQUENCE=""
    TAKE_FIRST=""
    QUESTION_ONLY="--first_label_only"
fi


# Experiment prefix
if [ "$MODEL_TYPE" == "EmbeddingModel" ]; then
    MODEL_STR=""
    SCHEDULE_SAMPLING=""
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelFixed" ]; then
    MODEL_STR="_Fixed"
    SCHEDULE_SAMPLING=""
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSS" ]; then
    MODEL_STR="_SS"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariable" ]; then
    MODEL_STR="_SSVariable"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariableLeftPad" ]; then
    MODEL_STR="_SSVariableLeftPad"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING="--left_padding"
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSPredLength" ]; then
    MODEL_STR="_SSPredLength"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING="--left_padding"
    PRED_LENGTH="--pred_length"
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariableLeftPadPredLength" ]; then
    MODEL_STR="_SSVariableLeftPadPredLength"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING="--left_padding"
    PRED_LENGTH="--pred_length"
fi

# =================================================================
if [ "$full_finetuning" = true ]; then
    FULL_FINETUNING="--full_finetuning" # FULL_FINETUNING=""
    FINETUNING_STR="_full_finetuning"
else
    FINETUNING_STR=""
    FULL_FINETUNING=""
fi

if [ "$all_data" = true ]; then
    TRAIN_ON_ALL_DATA="--train_on_all_data"
else
    TRAIN_ON_ALL_DATA=""
fi

if [ "$multiple_gpus" = true ]; then
    GPUS_PREFIX="_4gpu"
else
    GPUS_PREFIX=""
fi

if [ "$save_only_improve" = true ]; then
    SAVE_ONLY_IMPROVE="--save_only_improve"
else
    SAVE_ONLY_IMPROVE=""
fi

if [ "$save_best_model" = true ]; then
    SAVE_BEST_MODEL="--save_best_model"
else
    SAVE_BEST_MODEL=""
fi

if [ "$normalize" = true ]; then
    normalize_prefix="normalized_"
    NORMALIZE_STR="--normalize_embeddings"
else
    normalize_prefix=""
    NORMALIZE_STR=""
fi

if [ "$base_model_name" = "llama-1b" ]; then
    base_model_prefix=""
else
    base_model_prefix="${base_model_name}_"
fi  


if [ "$resume_from_checkpoint" = true ]; then
    RESUME_FROM_CHECKPOINT="--resume_from_checkpoint"
else
    RESUME_FROM_CHECKPOINT=""
fi

if [ "$use_stateful_dataloader" = true ]; then
    USE_STATEFUL_DATALOADER="--use_stateful_dataloader"
else
    USE_STATEFUL_DATALOADER=""
fi

if [ "$pred_length_labels" = true ]; then
    pred_length_labels_str="_pred_length"
else
    pred_length_labels_str=""
fi

if [ "$mix_one_label_shuffled" = true ]; then
    MIX_ONE_LABEL_SHUFFLED="--mix_one_label_shuffled"
    mix_one_label_shuffled_prefix="mix_one_"
else
    MIX_ONE_LABEL_SHUFFLED=""
    mix_one_label_shuffled_prefix=""
fi

# Experiment prefix
EXP_PREFIX="force_sampling_${mix_one_label_shuffled_prefix}${base_model_prefix}${normalize_prefix}qampari${GPUS_PREFIX}${FINETUNING_STR}${MODEL_STR}_${MODE}"

# Base directory for saving results
BASE_SAVE_PATH="results/${base_model_name}/qampari_${doc_encoder}"

# Training dataset path
if [ "$USE_HARD_NEGATIVES" = true ]; then
    BASE_TRAIN_PATH="training_datasets/${base_model_name}/qampari/${doc_encoder}/autoregressive_qampari_${doc_encoder}_train_dataset_1b_contrastive_hard_negative_5_to_8_ctxs${pred_length_labels_str}/"
else
    BASE_TRAIN_PATH="training_datasets/${base_model_name}/qampari/${doc_encoder}/autoregressive_qampari_${doc_encoder}_train_dataset_1b_contrastive_5_to_8_ctxs${pred_length_labels_str}/"
fi



# Model checkpoints
if [ "$base_model_name" = "inf" ]; then
    MODEL_ID="infly/inf-retriever-v1-1.5b"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
elif [ "$base_model_name" = "llama-1b" ]; then
    MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
elif [ "$base_model_name" = "qwen3-4b" ]; then
    MODEL_ID="Qwen/Qwen3-4B-Instruct-2507"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
elif [ "$base_model_name" = "llama-3b" ]; then
    MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
elif [ "$base_model_name" = "llama-8b" ]; then
    MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
    BASE_ADAPTER_PATH=None
    BASE_LINEAR_CHECKPOINT_PATH=None
else
    echo "Invalid base model name"
    exit 1
fi



# =================================================================
# FIXED HYPERPARAMETERS
# =================================================================
# Model embedding dimension
if [ "$doc_encoder" == "inf" ]; then
    EMBEDDING_MODEL_DIM=1536
else
    EMBEDDING_MODEL_DIM=1024
fi

# How often to save checkpoints (in steps)
SAVE_EVERY_N_STEPS=500

# Gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS=1

# Weight decay
WEIGHT_DECAY=0.01

# Scheduler type
SCHEDULER="linear"

# Maximum gradient norm for clipping
MAX_GRAD_NORM=5.0


# Generate experiment name
EXP_NAME="${EXP_PREFIX}_lr${LEARNING_RATE}_temp${TEMPERATURE}_batch${BATCH_SIZE}_ep${NUM_EPOCHS}_warmup${WARMUP_RATIO}"

# Set paths
SAVE_PATH="${BASE_SAVE_PATH}"
OUTPUT_LOG="single_run_output.log"

# Create save directory
mkdir -p "$SAVE_PATH"

echo "Starting single training run..."
echo "Experiment name: $EXP_NAME"
echo "Save path: $SAVE_PATH"
echo "Output log: $OUTPUT_LOG"
echo ""


# Build training arguments
ARGS="--project ${BASE_PROJECT} \
      --save_path ${SAVE_PATH} \
      --name ${EXP_NAME} \
      --train_path ${BASE_TRAIN_PATH} \
      --adapter_path ${BASE_ADAPTER_PATH} \
      --linear_checkpoint_path ${BASE_LINEAR_CHECKPOINT_PATH} \
      --model_id ${MODEL_ID} \
      --lr ${LEARNING_RATE} \
      --temperature ${TEMPERATURE} \
      --batch_size_training ${BATCH_SIZE} \
      --num_epochs ${NUM_EPOCHS} \
      --warmup_ratio ${WARMUP_RATIO} \
      --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
      --weight_decay ${WEIGHT_DECAY} \
      --scheduler ${SCHEDULER} \
      --max_grad_norm ${MAX_GRAD_NORM} \
      --loss_function ${LOSS_FUNCTION} \
      --lr_min_ratio ${LR_MIN_RATIO} \
      ${SHUFFLE_SEQUENCE} \
      ${SAVE_ONLY_IMPROVE} \
      ${TAKE_FIRST} \
      ${QUESTION_ONLY} \
      --embedding_model_dim ${EMBEDDING_MODEL_DIM} \
      --save_every_n_steps ${SAVE_EVERY_N_STEPS} \
      --model_type ${MODEL_TYPE} \
      ${FULL_FINETUNING} \
      ${SAVE_BEST_MODEL} \
      ${SCHEDULE_SAMPLING} \
      ${TRAIN_ON_ALL_DATA} \
      ${LEFT_PADDING} \
      ${NORMALIZE_STR} \
      ${FORCE_SAMPLING} \
      --sample_rate_multiplier ${SAMPLE_RATE_MULTIPLIER} \
      --log_with ${LOG_WITH} \
      ${RESUME_FROM_CHECKPOINT} \
      ${USE_STATEFUL_DATALOADER} \
      ${PRED_LENGTH} \
      ${MIX_ONE_LABEL_SHUFFLED}
      "
