#!/bin/bash
# Set hyperparameters
LEARNING_RATE=5e-5
TEMPERATURE=0.05
BATCH_SIZE=32
NUM_EPOCHS=120
WARMUP_RATIO=0.05
SAMPLE_RATE_MULTIPLIER=10
LR_MIN_RATIO=0.0


# =================================================================
# BASE CONFIGURATION
# =================================================================
EXP_NAME="[exp_name]"
LOG_WITH="wandb"
BASE_PROJECT="[project_name]"    # Project name (will be used in wandb)
# Base directory for saving results
BASE_SAVE_PATH="[save_path]"
# Training dataset path
BASE_TRAIN_PATH="[data_path]"

MODEL_ID="[model_id]"            # Model ID to be finetuned. If "llama-1b", we use "meta-llama/Llama-3.2-1B-Instruct". If "qwen3-4b", we use "Qwen/Qwen3-4B-Instruct-2507". If "llama-3b", we use "meta-llama/Llama-3.2-3B-Instruct". If "llama-8b", we use "meta-llama/Llama-3.1-8B-Instruct".
BASE_ADAPTER_PATH=None
BASE_LINEAR_CHECKPOINT_PATH=None

full_finetuning=false            # whether to use full finetuning. If false, we use LoRA.
all_data=false                   # whether to train on all data. If false, we reserve the last 10% of the data for validation.
multiple_gpus=true               # whether to use multiple GPUs. If true, we use the accelerate package.

EMBEDDING_MODEL_DIM=1536         # For inf-retriever, the embedding dimension is 1536. 

MODEL_TYPE="EmbeddingModelSSVariableLeftPad"   # Use "EmbeddingModelSSVariableLeftPad" for real data, "EmbeddingModel" for synthetic data.
MODE="single"                    # single, multi_scheduled_sampling, multi_always_sampling. For more details, see the README.md file.

save_only_improve=true           # whether to save only improve
save_best_model=true             # whether to save best model
resume_from_checkpoint=false     # whether to resume from checkpoint. If true, we resume from the last checkpoint.
use_stateful_dataloader=false    # whether to use stateful dataloader. If true, we use the stateful dataloader.
normalize=true                   # whether to normalize the embeddings. If true, we normalize the embeddings.

# Other Configs
SAVE_EVERY_N_STEPS=500
GRADIENT_ACCUMULATION_STEPS=1
WEIGHT_DECAY=0.01
SCHEDULER="linear"
MAX_GRAD_NORM=5.0


if [ "$MODE" == "multi_scheduled_sampling" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
    FORCE_SAMPLING=""
    mix_one_label_shuffled=true
elif [ "$MODE" == "multi_always_sampling" ]; then
    LOSS_FUNCTION="Hungarian_Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST=""
    QUESTION_ONLY=""
    FORCE_SAMPLING="--force_sampling"
    mix_one_label_shuffled=true
elif [ "$MODE" == "single" ]; then
    LOSS_FUNCTION="Contrastive"
    SHUFFLE_SEQUENCE="--shuffle_sequence"
    TAKE_FIRST="--take_first"
    QUESTION_ONLY=""
    FORCE_SAMPLING=""
    mix_one_label_shuffled=false
fi


# Experiment prefix
if [ "$MODEL_TYPE" == "EmbeddingModel" ]; then
    MODEL_STR=""
    SCHEDULE_SAMPLING=""
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSS" ]; then
    MODEL_STR="_SS"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING=""
    PRED_LENGTH=""
elif [ "$MODEL_TYPE" == "EmbeddingModelSSVariableLeftPad" ]; then
    MODEL_STR="_SSVariableLeftPad"
    SCHEDULE_SAMPLING="--schedule_sampling"
    LEFT_PADDING="--left_padding"
    PRED_LENGTH=""
fi

# =================================================================
if [ "$full_finetuning" = true ]; then
    FULL_FINETUNING="--full_finetuning" # FULL_FINETUNING=""
else
    FINETUNING_STR=""
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
    NORMALIZE_STR="--normalize_embeddings"
else
    NORMALIZE_STR=""
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

if [ "$mix_one_label_shuffled" = true ]; then
    MIX_ONE_LABEL_SHUFFLED="--mix_one_label_shuffled"
else
    MIX_ONE_LABEL_SHUFFLED=""
fi


# Set paths
OUTPUT_LOG="single_run_output.log"
# Create save directory
mkdir -p "$BASE_SAVE_PATH"

echo "Starting single training run..."
echo "Experiment name: $EXP_NAME"
echo "Save path: $BASE_SAVE_PATH"
echo "Output log: $OUTPUT_LOG"
echo ""


# Build training arguments
ARGS="--project ${BASE_PROJECT} \
      --save_path ${BASE_SAVE_PATH} \
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

echo "Training arguments:"
echo "$ARGS"
echo ""

# Set HuggingFace token if provided
if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN="$HF_TOKEN"
fi

# Run training
echo "Starting training at $(date)"
echo "Logging output to: $OUTPUT_LOG"
echo "Python command: $PYTHON_COMMAND"

# Use accelerate launch for distributed training support
HF_TOKEN="$HF_TOKEN" ${PYTHON_COMMAND} $ARGS 2>&1 | tee "$OUTPUT_LOG"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "Training completed at $(date)"
echo "Exit code: $EXIT_CODE"

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "Training completed successfully!"
    echo "Results saved in: $SAVE_PATH"
    echo "Output log saved as: $OUTPUT_LOG"
else
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check the output log for details: $OUTPUT_LOG"
      ${MIX_ONE_LABEL_SHUFFLED}
      "

if [ "$multiple_gpus" = true ]; then
    accelerate launch train_distributed.py $ARGS
else
    python train.py $ARGS
fi